/*
 * Copyright (C) 2024, Graz University of Technology
 * This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
 */

 #pragma once

 #include "../auxiliary.h"
 
 #include <cooperative_groups.h>
 namespace cg = cooperative_groups;

__device__ __inline__ uint64_t constructSortKey(uint32_t tile_id, float depth)
{
    uint64_t key = tile_id;
    key <<= 32;
    key |= *((uint32_t*)&depth);
    return key;
}

// Given a ray and a Gaussian primitive, compute the intersection depth.
__device__ __inline__ bool getIntersectPoint(
    const int W, const int H,
    const float fx, const float fy,
    const float2 scale, 
    const glm::vec2 pixel_center,
    const float* view2gaussian,
    float& depth
){
 
    // Fisrt compute two homogeneous planes, See Eq. (8)
	float3 Tu = {view2gaussian[0], view2gaussian[1], view2gaussian[2]};
	float3 Tv = {view2gaussian[3], view2gaussian[4], view2gaussian[5]};
	float3 Tw = {view2gaussian[6], view2gaussian[7], view2gaussian[8]};
	float3 k = {-Tu.x + pixel_center.x * Tw.x, -Tu.y + pixel_center.x * Tw.y, -Tu.z + pixel_center.x * Tw.z};
	float3 l = {-Tv.x + pixel_center.y * Tw.x, -Tv.y + pixel_center.y * Tw.y, -Tv.z + pixel_center.y * Tw.z};
	// cross product of two planes is a line (i.e., homogeneous point), See Eq. (10)
	float3 p = crossProduct(k, l);
	
	if (p.z == 0.0) return false; // there is not intersection
	// TODO: no intersection if distance < scale 
	
	// 3d homogeneous point to 2d point on the splat
	float2 s = {p.x / p.z, p.y / p.z};
	// 3d distance. Compute Mahalanobis distance in the canonical splat' space
	float rho3d = (s.x * s.x + s.y * s.y);

	depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;  // splat depth 
    return true;
}

 
template<bool TILE_BASED_CULLING = false, bool LOAD_BALANCING = true>
__global__ void duplicateWithKeys_extended(
    int P, 
    int W, int H,
    const float focal_x, const float focal_y,
    const float2* __restrict__ points_xy,
    const float* __restrict__ depths,
    const float2* __restrict__ scales,
    const float* __restrict__ view2gaussians,
    const uint32_t* __restrict__  offsets,
    const int* __restrict__ radii,
    const float2* __restrict__ rects,
    uint64_t* __restrict__ gaussian_keys_unsorted,
    uint32_t* __restrict__ gaussian_values_unsorted,
    dim3 grid)
{	
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

	// Since the projection of the quadratic surface on the image is non-convex, 
	// there is no explicit solution for computing the pixel with the maximum weight on the image,
	// and tile-based culling is not performed.
    constexpr bool EVAL_MAX_CONTRIB_POS = false;
    constexpr bool PER_TILE_DEPTH = true;

#define RETURN_OR_INACTIVE() if constexpr(LOAD_BALANCING) { active = false; } else { return; }
    uint32_t idx = cg::this_grid().thread_rank();
    bool active = true;
    if (idx >= P) {
	    RETURN_OR_INACTIVE();
	    idx = P - 1;
    }

    const int radius = radii[idx];
    if (radius <= 0) {
	    RETURN_OR_INACTIVE();
    }

	// If the thread exceeds the Gaussian index, the Gaussian projection is zero, 
	// and there are no Gaussians to process in the current warp, return.
    if constexpr(LOAD_BALANCING)
	    if (__ballot_sync(WARP_MASK, active) == 0)
		    return;

    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off_init = (idx == 0) ? 0 : offsets[idx - 1];

    const int offset_to_init = offsets[idx];
    const float global_depth_init = depths[idx];

    const float2 xy_init = points_xy[idx];
    const float2 rect_dims_init = rects[idx];

    __shared__ float2 s_xy[BLOCK_SIZE];
    __shared__ float2 s_rect_dims[BLOCK_SIZE];
    __shared__ float s_radius[BLOCK_SIZE];
    s_xy[block.thread_rank()] = xy_init;
    s_rect_dims[block.thread_rank()] = rect_dims_init;
    s_radius[block.thread_rank()] = radius;

    uint2 rect_min_init, rect_max_init;
#if FAST_INFERENCE
    if (radius > MAX_BILLBOARD_SIZE)
	    getRectOld(xy_init, radius, rect_min_init, rect_max_init, grid);
	else
	    getRect(xy_init, rect_dims_init, rect_min_init, rect_max_init, grid);
# else
    getRectOld(xy_init, radius, rect_min_init, rect_max_init, grid);
#endif

    __shared__ float s_view2gaussians[BLOCK_SIZE * 9];
    __shared__ float2 s_scales[BLOCK_SIZE];

    if (PER_TILE_DEPTH)
    {
	    s_scales[block.thread_rank()] = scales[idx];
	    for (int ii = 0; ii < 9; ii++)
		    s_view2gaussians[9 * block.thread_rank() + ii] = view2gaussians[idx * 9 + ii];
    }

    constexpr uint32_t SEQUENTIAL_TILE_THRESH = 32U; // all tiles above this threshold will be computed cooperatively
    const uint32_t rect_width_init = (rect_max_init.x - rect_min_init.x);
    const uint32_t tile_count_init = (rect_max_init.y - rect_min_init.y) * rect_width_init;

    // Generate no key/value pair for invisible Gaussians
    if (tile_count_init == 0)	{
	    RETURN_OR_INACTIVE();
    }
    auto tile_function = [&](const int W, const int H,
						     const float fx, const float fy,
						     float2 xy,
						     int x, int y,// tile ID
						     const float2 scale, 
						     const float* view2gaussian, 
						     const float global_depth,
						     float& depth)  
	    {
		    const glm::vec2 tile_min(x * BLOCK_X, y * BLOCK_Y);
		    const glm::vec2 tile_max((x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1); // 像素坐标

		    glm::vec2 max_pos;
		    if constexpr (PER_TILE_DEPTH) 
		    {	
			    glm::vec2 target_pos = {max(min(xy.x, tile_max.x), tile_min.x), max(min(xy.y, tile_max.y), tile_min.y)};

				// Or select the tile's center pixel as the target_pos.
			    // const glm::vec2 tile_center = (tile_min + tile_max) * 0.5f;
			    // glm::vec2 target_pos = tile_center;

			    bool intersect = getIntersectPoint(
				    W, H, fx, fy, scale, target_pos, view2gaussian, depth); // Compute the intersection point of the quadratic surface.
			    if (intersect)
				    depth = max(0.0f, depth);
			    else // If there is no intersection, sort by the Gaussian centroid.
				    depth = global_depth;
		    }
		    else
		    {
			    depth = global_depth;
		    }

			// Since the quadratic surface is non-convex, tile-based culling is not performed.
		    // return (!TILE_BASED_CULLING) || max_opac_factor <= opacity_factor_threshold;
		    return true; 
	    };

    if (active)
    {
	    const float2 scale_init = {
		    s_scales[block.thread_rank()].x, 
		    s_scales[block.thread_rank()].y};

	    float view2gaussian_init[9];
	    for (int ii = 0; ii < 9; ii++)
		    view2gaussian_init[ii] = s_view2gaussians[9 * block.thread_rank() + ii];

	    for (uint32_t tile_idx = 0; tile_idx < tile_count_init && (!LOAD_BALANCING || tile_idx < SEQUENTIAL_TILE_THRESH); tile_idx++)
	    {
		    const int y = (tile_idx / rect_width_init) + rect_min_init.y;
		    const int x = (tile_idx % rect_width_init) + rect_min_init.x;

		    float depth;
		    bool write_tile = tile_function(
				    W, H, focal_x, focal_y,
				    xy_init, x, y, scale_init, view2gaussian_init, global_depth_init, depth);
		    if (write_tile)
		    {
			    if (off_init < offset_to_init)
			    {
				    const uint32_t tile_id = y * grid.x + x;
				    gaussian_values_unsorted[off_init] = idx;
				    gaussian_keys_unsorted[off_init] = constructSortKey(tile_id, depth);
			    }
			    else
			    {
#ifdef DUPLICATE_OPT_DEBUG
				    printf("Error (sequential): Too little memory reserved in preprocess: off=%d off_to=%d idx=%d\n", off_init, offset_to_init, idx);
#endif
			    }
			    off_init++;
		    }
	    }
    }

#undef RETURN_OR_INACTIVE

    if (!LOAD_BALANCING) // Coordinate to handle the unprocessed tasks of other threads within the same warp.
	    return;

    const uint32_t idx_init = idx; // Current thread idx.
    const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
    const uint32_t warp_idx = cg::this_thread_block().thread_rank() / WARP_SIZE;
    unsigned int lane_mask_allprev_excl = 0xFFFFFFFFU >> (WARP_SIZE - lane_idx);

    const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH; // Determine whether additional idle threads are needed for computation.
    const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);
    if (remaining_threads == 0)
	    return;
 
    uint32_t n_remaining_threads = __popc(remaining_threads); // The number of threads required for collaborative computation.
    for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++) 
    {
	    int i = __fns(remaining_threads, 0, n+1); // find lane index of next remaining thread

	    uint32_t idx_coop = __shfl_sync(WARP_MASK, idx_init, i); 
	    uint32_t off_coop = __shfl_sync(WARP_MASK, off_init, i); 

	    const uint32_t offset_to = __shfl_sync(WARP_MASK, offset_to_init, i);
	    const float global_depth = __shfl_sync(WARP_MASK, global_depth_init, i);

	    const float2 xy = s_xy[warp.meta_group_rank() * WARP_SIZE + i];
	    const float2 rect_dims = s_rect_dims[warp.meta_group_rank() * WARP_SIZE + i];
	    const float rad = s_radius[warp.meta_group_rank() * WARP_SIZE + i];
	    const float2 scale = {
		    s_scales[warp.meta_group_rank() * WARP_SIZE + i].x, 
		    s_scales[warp.meta_group_rank() * WARP_SIZE + i].y};
	    float view2gaussian[9];
	    for (int ii = 0; ii < 9; ii++)
		    view2gaussian[ii] = s_view2gaussians[9 * (warp.meta_group_rank() * WARP_SIZE + i) + ii];

	    uint2 rect_min, rect_max;
#if FAST_INFERENCE
        if (radius > MAX_BILLBOARD_SIZE)
	        getRectOld(xy, rad, rect_min, rect_max, grid);
	    else
	        getRect(xy, rect_dims, rect_min, rect_max, grid);
#else
        getRectOld(xy, rad, rect_min, rect_max, grid);
#endif

	    const uint32_t rect_width = (rect_max.x - rect_min.x);
	    const uint32_t tile_count = (rect_max.y - rect_min.y) * rect_width;
	    const uint32_t remaining_tile_count = tile_count - SEQUENTIAL_TILE_THRESH;
	    const int32_t n_iterations = (remaining_tile_count + WARP_SIZE - 1) / WARP_SIZE;
	    for (int it = 0; it < n_iterations; it++)
	    {
		    int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH; // it*32 + local_warp_idx + 32
		    int active_curr_it = tile_idx < tile_count;
 
		    int y = (tile_idx / rect_width) + rect_min.y;
		    int x = (tile_idx % rect_width) + rect_min.x;

		    float depth;
		    bool write_tile = tile_function(
			    W, H, focal_x, focal_y,
			    xy, x, y, scale, view2gaussian, global_depth, depth
		    );

		    const uint32_t write = active_curr_it && write_tile;

		    uint32_t n_writes, write_offset;
		    if constexpr (!TILE_BASED_CULLING)
		    {
			    n_writes = WARP_SIZE;
			    write_offset = off_coop + lane_idx;
		    }
		    else
		    {
			    const uint32_t write_ballot = __ballot_sync(WARP_MASK, write);
			    n_writes = __popc(write_ballot);
 
			    const uint32_t write_offset_it = __popc(write_ballot & lane_mask_allprev_excl);
			    write_offset = off_coop + write_offset_it;
		    }

		    if (write)
		    {
			    if (write_offset < offset_to)
			    {
				    const uint32_t tile_id = y * grid.x + x;
				    gaussian_values_unsorted[write_offset] = idx_coop;
				    gaussian_keys_unsorted[write_offset] = constructSortKey(tile_id, depth);
			    }
 #ifdef DUPLICATE_OPT_DEBUG
			    else
			    {
				    printf("Error (parallel): Too little memory reserved in preprocess: off=%d off_to=%d idx=%d tile_count=%d it=%d | x=%d y=%d rect=(%d %d - %d %d)\n", 
							write_offset, offset_to, idx_coop, tile_count, it, x, y, rect_min.x, rect_min.y, rect_max.x, rect_max.y);
			    }
 #endif
		    }
		    off_coop += n_writes;
	    }

	    __syncwarp();
    }
 }
