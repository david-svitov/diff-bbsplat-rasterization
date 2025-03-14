
// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.

#include "../auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;


template <uint32_t CHANNELS>
__device__ __inline__ void updateDistortionMap(
    const float depth,
    const float alpha,
    const float T,
    float& dist1,
    float& dist2,
    float* all_map
) {
    float A = 1 - T;
    float mapped_max_t = (FAR_PLANE * depth - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth);
    float error = mapped_max_t * mapped_max_t * A + dist2 - 2 * mapped_max_t * dist1;
	all_map[CHANNELS + DISTORTION_OFFSET] += error * alpha * T;

	dist1 += mapped_max_t * alpha * T;
	dist2 += mapped_max_t * mapped_max_t * alpha * T;
}

template <uint32_t CHANNELS>
__device__ __inline__ void updateMap(
    const float depth,
    const float alpha,
    const float T,
    const float* normal,
    float* all_map
) {
    for (int ch = 0; ch < CHANNELS; ch++)
		all_map[CHANNELS + NORMAL_OFFSET + ch] += normal[ch] * alpha * T;
    all_map[CHANNELS + DEPTH_OFFSET] += depth * alpha * T;
	all_map[CHANNELS + ALPHA_OFFSET] += alpha * T;
    if (T > 0.5)
        all_map[CHANNELS + MIDDEPTH_OFFSET] = depth;
}


template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderBufferCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ texture_alpha,
	const float* __restrict__ texture_color,
	int texture_size,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float3* __restrict__ normal_array,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	float* impact)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5, (float)pix.y + 0.5};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_normal[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	
	// Used for buffering.
    float sort_depths[BUFFER_LENGTH];
    float sort_alphas[BUFFER_LENGTH];
    float sort_tex_color[BUFFER_LENGTH * 3];
    float sort_normals[BUFFER_LENGTH * 3];
    int sort_ids[BUFFER_LENGTH];
    int sort_num = 0;
    for (int i = 0; i < BUFFER_LENGTH; ++i)
    {
        sort_depths[i] = FLT_MAX;
        // just to suppress warnings:
        sort_alphas[i] = 0;
        for (int ch = 0; ch < CHANNELS; ch++) 
        {
            sort_normals[i * CHANNELS + ch] = 0;
            sort_tex_color[i * CHANNELS + ch] = 0;
        }
        sort_ids[i] = -1;
    }

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS + OUTPUT_CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float D = { 0 };
	float N[3] = {0};
	float dist1 = {0};
	float dist2 = {0};
	float distortion = {0};
	float median_depth = {100};
	float median_weight = {0};
	float median_contributor = {-1};

#endif

    auto blend_one = [&](){
        if (sort_num == 0)
            return;
        --sort_num;
        float test_T = T * (1 - sort_alphas[0]);

        if (test_T < 0.0001f) {
            done = true;
            return;
        }
#if RENDER_AXUTILITY
        updateDistortionMap<CHANNELS>(sort_depths[0], sort_alphas[0], T, dist1, dist2, C);
#endif
        updateMap<CHANNELS>(sort_depths[0], sort_alphas[0], T, sort_normals, C);
		
        for (int ch = 0; ch < CHANNELS; ch++)
            C[ch] += (sort_tex_color[ch] + features[sort_ids[0] * CHANNELS + ch]) * sort_alphas[0] * T;

        if (T > 0.5){
            median_contributor = contributor;
            C[CHANNELS + MIDDEPTH_OFFSET] = sort_depths[0];
        }

        T = test_T;
        atomicAdd(&(impact[sort_ids[0]]),  sort_alphas[0] * T);

        for (int i = 1; i < BUFFER_LENGTH; ++i)
        {
            sort_depths[i - 1] = sort_depths[i];
            sort_alphas[i - 1] = sort_alphas[i];
            sort_ids[i - 1] = sort_ids[i];
            for (int ch = 0; ch < CHANNELS; ch++) 
            {
                sort_normals[(i - 1) * CHANNELS + ch] = sort_normals[i * CHANNELS + ch];
                sort_tex_color[(i - 1) * CHANNELS + ch] = sort_tex_color[i * CHANNELS + ch];
            }
        }
        sort_depths[BUFFER_LENGTH - 1] = FLT_MAX;
    };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_normal[block.thread_rank()] = normal_array[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
            // When the buffer is full, immediately retrieve the Gaussian rendering that is the closest.
            if (sort_num == BUFFER_LENGTH)
                blend_one();
			
            if (done == true)
                break;
                
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			float3 Tu = collected_Tu[j];
			float3 Tv = collected_Tv[j];
			float3 Tw = collected_Tw[j];
			float3 k = {-Tu.x + pixf.x * Tw.x, -Tu.y + pixf.x * Tw.y, -Tu.z + pixf.x * Tw.z};
			float3 l = {-Tv.x + pixf.y * Tw.x, -Tv.y + pixf.y * Tw.y, -Tv.z + pixf.y * Tw.z};
			// cross product of two planes is a line (i.e., homogeneous point), See Eq. (10)
			float3 p = crossProduct(k, l);
#if BACKFACE_CULL
			// May hanle this by replacing a low pass filter,
			// but this case is extremely rare.
			if (p.z == 0.0) continue; // there is not intersection
#endif
			// 3d homogeneous point to 2d point on the splat
			float2 s = {p.x / p.z, p.y / p.z};
			// 3d distance. Compute Mahalanobis distance in the canonical splat' space
			float rho3d = (s.x * s.x + s.y * s.y);

			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;  // splat depth
			if (depth < NEAR_PLANE) continue;
			float3 nor = collected_normal[j];
			float normal[3] = {nor.x, nor.y, nor.z};

            int texture_pixels = texture_size * texture_size;
            int texture_offset = collected_id[j] * texture_pixels;
            float sampled_alpha_array[1];
            biliniar_texture_sampler(sampled_alpha_array, s.x, s.y, texture_alpha + texture_offset, texture_size, 1);
            float sampled_alpha = sampled_alpha_array[0];

			float alpha = min(0.99f, sampled_alpha);
			if (alpha < 1.0f / 255.0f)
				continue;
			
			float sampled_color[3];
			texture_offset = collected_id[j] * texture_pixels * 3;
			biliniar_texture_sampler(sampled_color, s.x, s.y, texture_color + texture_offset, texture_size, 3);

            int id = collected_id[j];
#pragma unroll
            for (int s = 0; s < BUFFER_LENGTH; s++){
                
                if (depth < sort_depths[s]){
                    swap_T(depth, sort_depths[s]);
                    swap_T(alpha, sort_alphas[s]);
                    swap_T(id, sort_ids[s]);
                    for (int ch = 0; ch < CHANNELS; ch++)
                    {
                        swap_T(normal[ch], sort_normals[s*CHANNELS + ch]);
                        swap_T(sampled_color[ch], sort_tex_color[s*CHANNELS + ch]);
                    }
                }
            }
            ++sort_num;
        }
    }
	
    if (!done) {
        while (sort_num > 0)
            blend_one();
    }

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;

        for (int ch = 0; ch < CHANNELS; ch++){
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
        }
        
#if RENDER_AXUTILITY
        n_contrib[pix_id + H * W] = median_contributor;
        final_T[pix_id + H * W] = dist1;
        final_T[pix_id + 2 * H * W] = dist2;
        out_others[DEPTH_OFFSET * H * W + pix_id] = C[CHANNELS + DEPTH_OFFSET];
        out_others[ALPHA_OFFSET * H * W + pix_id] = C[CHANNELS + ALPHA_OFFSET];
        for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = C[CHANNELS + NORMAL_OFFSET+ch];
        out_others[MIDDEPTH_OFFSET * H * W + pix_id] = C[CHANNELS + MIDDEPTH_OFFSET];
        out_others[DISTORTION_OFFSET * H * W + pix_id] = C[CHANNELS + DISTORTION_OFFSET];
        out_others[MEDIAN_WEIGHT_OFFSET * H * W + pix_id] = C[CHANNELS + MEDIAN_WEIGHT_OFFSET];
#endif
    }
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderkBufferBackwardCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float* __restrict__ texture_alpha,
	const float* __restrict__ texture_color,
	int texture_size,
	const float2* __restrict__ points_xy_image,
	const float3* __restrict__ normal_array,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ out_colors,
	const float* __restrict__ out_others,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dtexture_alpha,
	float* __restrict__ dL_dtexture_color)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x + 0.5, (float)pix.y + 0.5};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float3 collected_normal[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

    // Note that we still use forward traversal during backpropagation.
    const float T_final = inside ? final_Ts[pix_id] : 0;
    float T = 1.0f; // Thus, the transmittance is initially set to 1.
    float acc_colors[C] = { 0, 0, 0 };
    float acc_depths = { 0 };
    float acc_normals[C] = { 0, 0, 0 };
    float acc_alphas = { 0 };

    float sort_depths[BUFFER_LENGTH];
    float sort_Gs[BUFFER_LENGTH];
    float2 sort_ps[BUFFER_LENGTH];
    float3 sort_nor[BUFFER_LENGTH];
    float3 sort_k[BUFFER_LENGTH];
	float3 sort_l[BUFFER_LENGTH];
	float3 sort_p[BUFFER_LENGTH];
	float3 sort_Tw[BUFFER_LENGTH];

    int sort_ids[BUFFER_LENGTH];
    int sort_num = 0;
    for (int i = 0; i < BUFFER_LENGTH; ++i)
    {
        sort_depths[i] = FLT_MAX;
        // just to suppress warnings:
        sort_Gs[i] = 0;
        sort_ps[i] = {0.0f, 0.0f};
        sort_nor[i] = {0.0f, 0.0f, 0.0f};
        sort_k[i] = {0.0f, 0.0f, 0.0f};
	    sort_l[i] = {0.0f, 0.0f, 0.0f};
	    sort_p[i] = {0.0f, 0.0f, 0.0f};
	    sort_Tw[i] = {0.0f, 0.0f, 0.0f};
        sort_ids[i] = -1;
    }

	// We start from the front.
	uint32_t contributor = 0;
	float dL_dpixel[C];
	float final_color[C];
    

#if RENDER_AXUTILITY
	float dL_dreg = 0.0f;
	float dL_ddepth = 0.0f;
	float dL_daccum = 0.0f;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth = 0.0f;
	float dL_dmax_dweight = 0.0f;
	
	float final_normal[C];
    float final_depth;
    float final_alpha;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
        {
            dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];
            final_normal[i] = out_others[(NORMAL_OFFSET + i) * H * W + pix_id];
        }

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
		
		final_depth = out_others[DEPTH_OFFSET * H * W + pix_id];
		final_alpha = out_others[ALPHA_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
        {
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
			final_color[i] = out_colors[i * H * W + pix_id] - T_final * bg_color[i];
        }
	}
	
	auto blend_one = [&]() {
	    if (sort_num == 0)
			return;
		--sort_num;
		
		//Store variables to use in gradient calculation
		int global_id = sort_ids[0]; 
		float G = sort_Gs[0];
		float2 s = sort_ps[0];
		float3 nor = sort_nor[0];
		float3 k = sort_k[0];
		float3 l = sort_l[0];
		float3 p = sort_p[0];
		float3 Tw = sort_Tw[0];
		float depth = sort_depths[0];
		
		// We need this because we go from front to back
		const float alpha = min(0.99f, G);
		float test_T = T * (1 - alpha);
		if(test_T  < 0.0001f){
			done = true;
			return;
		}
	    
		const float dchannel_dcolor = alpha * T;

		// Propagate gradients to per-Gaussian colors and keep
		// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
		// pair).
		float dL_dalpha = 0.0f;
		float perchannel_grads[3];

		int texture_pixels = texture_size * texture_size;
        int texture_offset = global_id * texture_pixels * 3;
		float sampled_color[3];
		biliniar_texture_sampler(sampled_color, s.x, s.y, texture_color + texture_offset, texture_size, 3);

		for (int ch = 0; ch < C; ch++)
		{
			const float c = sampled_color[ch] + colors[global_id * C + ch];
			
			acc_colors[ch] += c * alpha * T;
			// Update last color (to be used in the next iteration)
			float accum_rec_ch = (final_color[ch] - acc_colors[ch]) / test_T;

			const float dL_dchannel = dL_dpixel[ch];
			dL_dalpha += (c - accum_rec_ch) * dL_dchannel;
			// Update the gradients w.r.t. color of the Gaussian. 
			// Atomic, since this pixel is just one of potentially
			// many that were affected by this Gaussian.
			float dL_dcolor = dchannel_dcolor * dL_dchannel;
			atomicAdd(&(dL_dcolors[global_id * C + ch]), dL_dcolor);
			perchannel_grads[ch] = dL_dcolor;
		}
		float2 DL_ds_color = backward_biliniar_texture_sampler(perchannel_grads, s.x, s.y,
		                                                       texture_color + texture_offset,
		                                                       dL_dtexture_color + texture_offset,
		                                                       texture_size, 3);

		float dL_dz = 0.0f;
		float dL_dweight = 0;
#if RENDER_AXUTILITY
		float m_d = (FAR_PLANE * depth - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth);
		float dmd_dd = (FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth * depth);
		if (contributor == median_contributor) {
			dL_dz += dL_dmedian_depth;
			dL_dweight += dL_dmax_dweight;
		}
#if DETACH_WEIGHT 
		// if not detached weight, sometimes 
		// it will bia toward creating extragated 2D Gaussians near front
		dL_dweight += 0;
#else
		dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
		dL_dalpha += dL_dweight - last_dL_dT;
		// propagate the current weight W_{i} to next weight W_{i-1}
		last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
		float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
		dL_dz += dL_dmd * dmd_dd;

		// Propagate gradients w.r.t ray-splat depths
		acc_depths += depth * alpha * T;
		float accum_depth_rec = (final_depth - acc_depths) / test_T;
		dL_dalpha += (depth - accum_depth_rec) * dL_ddepth;
		// Propagate gradients w.r.t. color ray-splat alphas
		acc_alphas += alpha * T;
		float accum_alpha_rec = (final_alpha - acc_alphas) / test_T;
		dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

        float normal[3] = {nor.x, nor.y, nor.z};

		// Propagate gradients to per-Gaussian normals
		for (int ch = 0; ch < 3; ch++) {
		    acc_normals[ch] += normal[ch] * alpha * T;
			float accum_normal_rec_ch = (final_normal[ch] - acc_normals[ch]) / test_T;
			dL_dalpha += (normal[ch] - accum_normal_rec_ch) * dL_dnormal2D[ch];
			atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
		}
#endif

		dL_dalpha *= T;
		
		// Because the rewritten version uses a different recursive form, we do not need to update last_alpha.
		// last_alpha = alpha;

		// Account for fact that alpha also influences how much of
		// the background color is added if nothing left to blend
		float bg_dot_dpixel = 0;
		for (int i = 0; i < C; i++)
			bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
		dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

#if RENDER_AXUTILITY
		dL_dz += alpha * T * dL_ddepth; 
#endif
		texture_offset = global_id * texture_pixels;
		float alpha_grads[1] = {dL_dalpha};
		//if (alpha == 0 && dL_dalpha != 0) {
		//    printf("Non zero gradient for zero alpha \n");
		//}
		float2 DL_ds_alpha = backward_biliniar_texture_sampler(alpha_grads, s.x, s.y, texture_alpha + texture_offset, dL_dtexture_alpha + texture_offset, texture_size, 1);

        // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
        float2 dL_ds = {
            DL_ds_alpha.x + dL_dz * Tw.x + DL_ds_color.x,
            DL_ds_alpha.y + dL_dz * Tw.y + DL_ds_color.y
        };
        float3 dz_dTw = {s.x, s.y, 1.0};
        float dsx_pz = dL_ds.x / p.z;
        float dsy_pz = dL_ds.y / p.z;
        float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
        float3 dL_dk = crossProduct(l, dL_dp);
        float3 dL_dl = crossProduct(dL_dp, k);

        float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
        float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
        float3 dL_dTw = {
            pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x,
            pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y,
            pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};

        // Update gradients w.r.t. 3D covariance (3x3 matrix)
        atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
        atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
        atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);

        T = test_T;

		// Updating the buffer's sorting.
		for (int i = 1; i < BUFFER_LENGTH; i++){
			sort_ids[i - 1] = sort_ids[i];
			sort_depths[i - 1] = sort_depths[i];
			sort_Gs[i - 1] = sort_Gs[i];
			sort_ps[i - 1] = sort_ps[i];
			sort_nor[i - 1] = sort_nor[i];
	        sort_k[i - 1] = sort_k[i];
	        sort_l[i - 1] = sort_l[i];
	        sort_p[i - 1] = sort_p[i];
	        sort_Tw[i - 1] = sort_Tw[i];
		}
		sort_depths[BUFFER_LENGTH - 1] = FLT_MAX;
	};

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
	    int all_done = __syncthreads_and(done);
		if (all_done)
			break;
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal[block.thread_rank()] = normal_array[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
		    if (sort_num == BUFFER_LENGTH)
				blend_one();

			if (done)
				break;	
				
			contributor++;
			// compute ray-splat intersection as before
			float3 Tu = collected_Tu[j];
			float3 Tv = collected_Tv[j];
			float3 Tw = collected_Tw[j];
			// compute two planes intersection as the ray intersection
			float3 k = {-Tu.x + pixf.x * Tw.x, -Tu.y + pixf.x * Tw.y, -Tu.z + pixf.x * Tw.z};
			float3 l = {-Tv.x + pixf.y * Tw.x, -Tv.y + pixf.y * Tw.y, -Tv.z + pixf.y * Tw.z};
			// cross product of two planes is a line (i.e., homogeneous point), See Eq. (10)
			float3 p = crossProduct(k, l);
#if BACKFACE_CULL
			// May hanle this by replacing a low pass filter,
			// but this case is extremely rare.
			if (p.z == 0.0) continue; // there is not intersection
#endif
			float2 s = {p.x / p.z, p.y / p.z};
			
			// Compute accurate depth when necessary
			float c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			if (c_d < NEAR_PLANE) continue;
			
			float3 nor = collected_normal[j];

            int texture_pixels = texture_size * texture_size;
            int id = collected_id[j];
            int texture_offset = id * texture_pixels;
            float sampled_alpha_array[1];
            biliniar_texture_sampler(sampled_alpha_array, s.x, s.y, texture_alpha + texture_offset, texture_size, 1);
            float sampled_alpha = sampled_alpha_array[0];

			float G = sampled_alpha;
			const float alpha = min(0.99f, G);
			if (alpha < 1.0f / 255.0f)
				continue;
			
#pragma unroll
			for(int ii = 0; ii < BUFFER_LENGTH; ii++){
				
				if (c_d < sort_depths[ii]){
					swap_T(c_d, sort_depths[ii]);
					swap_T(id, sort_ids[ii]);
					swap_T(G, sort_Gs[ii]);
					swap_T(s, sort_ps[ii]);
					swap_T(nor, sort_nor[ii]);
					swap_T(k, sort_k[ii]);
	                swap_T(l, sort_l[ii]);
	                swap_T(p, sort_p[ii]);
	                swap_T(Tw, sort_Tw[ii]);
				}
			}
			++sort_num;		
		}
	}
	if (!done){
		while (sort_num > 0)
			blend_one();
	}
}

