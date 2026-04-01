/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <functional>
#include <glm/glm.hpp>

namespace FORWARD {
// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(int P, int D, int M, const float *orig_points,
                const glm::vec3 *scales, const float scale_modifier,
                const glm::vec4 *rotations, const float *opacities,
                const float *dc, const float *shs, bool *clamped,
                const float *cov3D_precomp, const float *colors_precomp,
                const float *viewmatrix, const float *projmatrix,
                const glm::vec3 *cam_pos, const int W, int H,
                const float focal_x, float focal_y, const float tan_fovx,
                float tan_fovy, int *radii, float2 *points_xy_image,
                float *depths, float *cov3Ds, float *colors,
                float4 *conic_opacity, const dim3 grid, uint32_t *tiles_touched,
                bool prefiltered, cudaStream_t stream);

// Main rasterization method.
void render(const dim3 grid, dim3 block, const uint2 *ranges,
            const uint32_t *point_list, const uint32_t *per_tile_bucket_offset,
            uint32_t *bucket_to_tile, float *sampled_T, float *sampled_ar,
            int W, int H, const float2 *means2D, const float *colors,
            const float4 *conic_opacity, float *final_T, uint32_t *n_contrib,
            uint32_t *max_contrib, const float *bg_color, float *out_color,
            cudaStream_t stream, float *PixelT, const bool store_lb,
            const int *div_per_tile);

void render_load_balancing(const dim3 grid, dim3 block, const uint2 *ranges,
                           const uint32_t *point_list, const uint2 *tiles_ID,
                           const int num_tiles, int W, int H,
                           const float2 *means2D, const float *colors,
                           const float4 *conic_opacity, const float *bg_color,
                           float *out_color, float *pixel_tr,
                           cudaStream_t stream, short *out_lb_set,
                           const bool lb_last_set);
} // namespace FORWARD

#endif