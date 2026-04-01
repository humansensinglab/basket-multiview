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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <functional>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

namespace CudaRasterizer {
class Rasterizer {
public:
  static void markVisible(int P, float *means3D, float *viewmatrix,
                          float *projmatrix, bool *present);

  static std::tuple<int, int, int, std::vector<int>, std::vector<int>>
  forward(std::function<char *(size_t)> geometryBuffer,
          std::function<char *(size_t)> binningBuffer,
          std::function<char *(size_t)> imageBuffer,
          std::function<char *(size_t)> sampleBuffer,
          std::function<char *(size_t)> rangesBuffer,
          std::function<char *(size_t)> gs_listBuffer,
          std::function<char *(size_t)> tiles_idBuffer,
          std::function<char *(size_t)> pixel_trBuffer, const int P, int D,
          int M, const float *background, const int width, int height,
          const float *means3D, const float *dc, const float *shs,
          const float *colors_precomp, const float *opacities,
          const float *scales, const float scale_modifier,
          const float *rotations, const float *cov3D_precomp,
          const float *viewmatrix, const float *projmatrix,
          const float *cam_pos, const float tan_fovx, float tan_fovy,
          const bool prefiltered, float *out_color, float *out_color_blending,
          cudaStream_t stream, int *radii = nullptr, bool debug = false,
          const bool store = false, const bool precomp = false,
          const int num_buckets = 0, const int num_rendered = 0,
          char *gs_list = nullptr, char *ranges = nullptr,
          const bool use_cuda_graph = false, char *img_buff = nullptr,
          char *geom_buff = nullptr, char *sample_buff = nullptr,
          const bool use_load_balancing = false, char *tiles_id = nullptr,
          const int num_tiles = 0, char *pixel_tr = nullptr,
          float *out_depth = nullptr, const bool depth_reg = false,
          const bool lb_last_set = false, short *out_lb_set = nullptr);

  static int load_balancer(const uint2 *ranges, std::vector<uint2> *ranges_new,
                           std::vector<uint2> *tiles_order,
                           const int tile_number,
                           std::vector<int> *div_per_tile);

  static void backward(
      const int P, int D, int M, int R, int B, const float *background,
      const int width, int height, const float *means3D, const float *dc,
      const float *shs, const float *colors_precomp, const float *opacities,
      const float *scales, const float scale_modifier, const float *rotations,
      const float *cov3D_precomp, const float *viewmatrix,
      const float *projmatrix, const float *campos, const float tan_fovx,
      float tan_fovy, const int *radii, cudaStream_t stream, char *geom_buffer,
      char *binning_buffer, char *image_buffer, char *sample_buffer,
      const float *dL_dpix, float *dL_dmean2D, float *dL_dconic,
      float *dL_dopacity, float *dL_dcolor, float *dL_dmean3D, float *dL_dcov3D,
      float *dL_ddc, float *dL_dsh, float *dL_dscale, float *dL_drot,
      bool debug, bool precomp, char *gs_list, char *ranges,
      char *tiles_id = nullptr, const int num_tiles = 0,
      char *pixel_tr = nullptr, const bool use_load_balancing = false,
      const bool lb_last_set = false, const short *lb_last_num = nullptr);
};
} // namespace CudaRasterizer
#endif