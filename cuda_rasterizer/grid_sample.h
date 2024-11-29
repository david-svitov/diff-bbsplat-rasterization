#ifndef CUDA_GRID_SAMPLE_H_INCLUDED
#define CUDA_GRID_SAMPLE_H_INCLUDED

#include <cuda.h>

__forceinline__ __device__
bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

__forceinline__ __device__
void biliniar_texture_sampler(float* result, float u, float v, const float* __restrict__ texture, int texture_size, int channels) {
    float x = ((u + 1) / 2) * (texture_size - 1);
    float y = ((v + 1) / 2) * (texture_size - 1);
    int ix_nw = static_cast<int>(::floor(x));
    int iy_nw = static_cast<int>(::floor(y));

    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    // get surfaces to each neighbor:
    float nw = (ix_se - x)    * (iy_se - y);
    float ne = (x    - ix_sw) * (iy_sw - y);
    float sw = (ix_ne - x)    * (y    - iy_ne);
    float se = (x    - ix_nw) * (y    - iy_nw);

    int texture_area = texture_size * texture_size;
    // calculate bilinear weighted pixel value and set output pixel
    for (int c = 0; c < channels; ++c) {
        result[c] = 0;
        if (within_bounds_2d(iy_nw, ix_nw, texture_size, texture_size)) {
            result[c] += texture[c*texture_area + iy_nw*texture_size + ix_nw] * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, texture_size, texture_size)) {
            result[c] += texture[c*texture_area + iy_ne*texture_size + ix_ne] * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, texture_size, texture_size)) {
            result[c] += texture[c*texture_area + iy_sw*texture_size + ix_sw] * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, texture_size, texture_size)) {
            result[c] += texture[c*texture_area + iy_se*texture_size + ix_se] * se;
        }
    }
}

__forceinline__ __device__
void safe_add_2d(float *data, int y, int x, int texture_size, float delta) {
    if (within_bounds_2d(y, x, texture_size, texture_size)) {
        atomicAdd(data + y * texture_size + x, delta);
    }
}

__forceinline__ __device__
float2 backward_biliniar_texture_sampler(
    float * perchannel_grads,
    float u, float v,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int texture_size,
    int channels) {
    float x = ((u + 1) / 2) * (texture_size - 1);
    float y = ((v + 1) / 2) * (texture_size - 1);

    int ix_nw = static_cast<int>(::floor(x));
    int iy_nw = static_cast<int>(::floor(y));

    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    // get surfaces to each neighbor:
    float nw = (ix_se - x)    * (iy_se - y);
    float ne = (x    - ix_sw) * (iy_sw - y);
    float sw = (ix_ne - x)    * (y    - iy_ne);
    float se = (x    - ix_nw) * (y    - iy_nw);

    float gix = 0, giy = 0;
    // calculate and set grad_input
    for (int c = 0; c < channels; ++c) {
        int channel_offset = c * texture_size * texture_size;
        float g_out = perchannel_grads[c];
        safe_add_2d(grad_input + channel_offset, iy_nw, ix_nw, texture_size, nw * g_out);
        safe_add_2d(grad_input + channel_offset, iy_ne, ix_ne, texture_size, ne * g_out);
        safe_add_2d(grad_input + channel_offset, iy_sw, ix_sw, texture_size, sw * g_out);
        safe_add_2d(grad_input + channel_offset, iy_se, ix_se, texture_size, se * g_out);

        // calculate grad_grid
        if (within_bounds_2d(iy_nw, ix_nw, texture_size, texture_size)) {
            float nw_val = input[channel_offset + iy_nw * texture_size + ix_nw];
            gix -= nw_val * (iy_se - y) * g_out;
            giy -= nw_val * (ix_se - x) * g_out;
        }
        if (within_bounds_2d(iy_ne, ix_ne, texture_size, texture_size)) {
            float ne_val = input[channel_offset + iy_ne * texture_size + ix_ne];
            gix += ne_val * (iy_sw - y) * g_out;
            giy -= ne_val * (x - ix_sw) * g_out;
        }
        if (within_bounds_2d(iy_sw, ix_sw, texture_size, texture_size)) {
            float sw_val = input[channel_offset + iy_sw * texture_size + ix_sw];
            gix -= sw_val * (y - iy_ne) * g_out;
            giy += sw_val * (ix_ne - x) * g_out;
        }
        if (within_bounds_2d(iy_se, ix_se, texture_size, texture_size)) {
            float se_val = input[channel_offset + iy_se * texture_size + ix_se];
            gix += se_val * (y - iy_nw) * g_out;
            giy += se_val * (x - ix_nw) * g_out;
        }
    }
    float2 grad_grid;
    grad_grid.x = gix * (texture_size - 1) / 2;
    grad_grid.y = giy * (texture_size - 1) / 2;
    return grad_grid;
}

#endif