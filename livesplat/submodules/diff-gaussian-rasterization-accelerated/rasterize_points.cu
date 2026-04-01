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

#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <ATen/cuda/CUDAContext.h> // For CUDA context management
#include <c10/cuda/CUDAStream.h>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long)N});
    return reinterpret_cast<char *>(t.contiguous().data_ptr());
  };

  return lambda;
}

std::function<int *(size_t N)> resizeIntFunctional(torch::Tensor &t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long)N});
    return t.contiguous().data_ptr<int>();
  };

  return lambda;
}

std::function<float *(size_t N)> resizeFloatFunctional(torch::Tensor &t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long)N});
    return t.contiguous().data_ptr<float>();
  };

  return lambda;
}

// Forward
std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, int, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &background, const torch::Tensor &means_3d,
    const torch::Tensor &colors_precomp, const torch::Tensor &opacities,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov_3d_precomp,
    const torch::Tensor &view_matrix, const torch::Tensor &proj_matrix,
    const float tan_fov_x, const float tan_fov_y, const int image_height,
    const int image_width, const torch::Tensor &dc, const torch::Tensor &sh,
    const int degree, const torch::Tensor &campos, const bool prefiltered,
    const bool debug, const bool store, const bool precomp,
    const torch::Tensor &gs_list, const torch::Tensor &ranges,
    const int num_buckets, const int num_rendered, const bool use_cuda_graph,
    const torch::Tensor &img_buffer, const torch::Tensor &geom_buffer,
    const torch::Tensor &sample_buffer, const torch::Tensor &tiles_id,
    const int num_tiles, const torch::Tensor &pixel_tr,
    torch::Tensor &output_img, const bool use_load_balancing) {

  if (means_3d.ndimension() != 2 || means_3d.size(1) != 3) {
    AT_ERROR("means_3d must have dimensions (num_points, 3)");
  }

  cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream().stream();

  const int num_points = means_3d.size(0);
  const int height = image_height;
  const int width = image_width;

  auto int16_opts = means_3d.options().dtype(torch::kShort);
  auto float_opts = means_3d.options().dtype(torch::kFloat32);

  torch::Tensor out_depth;
  torch::Tensor out_lb_set;
  torch::Tensor radii =
      torch::full({num_points}, 0, means_3d.options().dtype(torch::kInt32));

  out_depth = torch::empty({0}, float_opts); // to remove not yet
  out_lb_set = torch::empty({0}, int16_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  torch::Tensor geom_buffer_internal =
      torch::empty({0}, options.device(device));
  torch::Tensor binning_buffer_internal =
      torch::empty({0}, options.device(device));
  torch::Tensor img_buffer_internal = torch::empty({0}, options.device(device));
  torch::Tensor sample_buffer_internal =
      torch::empty({0}, options.device(device));

  std::function<char *(size_t)> geom_resize_fn =
      resizeFunctional(geom_buffer_internal);
  std::function<char *(size_t)> binning_resize_fn =
      resizeFunctional(binning_buffer_internal);
  std::function<char *(size_t)> img_resize_fn =
      resizeFunctional(img_buffer_internal);
  std::function<char *(size_t)> sample_resize_fn =
      resizeFunctional(sample_buffer_internal);

  // Same structure as Buffers to store ranges and Gaussian list
  torch::Tensor ranges_buffer_internal =
      torch::empty({0}, options.device(device));
  torch::Tensor gs_list_buffer_internal =
      torch::empty({0}, options.device(device));
  torch::Tensor tiles_id_buffer_internal =
      torch::empty({0}, options.device(device));
  torch::Tensor pixel_tr_buffer_internal =
      torch::empty({0}, options.device(device));

  std::function<char *(size_t)> ranges_resize_fn =
      resizeFunctional(ranges_buffer_internal);
  std::function<char *(size_t)> gs_list_resize_fn =
      resizeFunctional(gs_list_buffer_internal);
  std::function<char *(size_t)> tiles_id_resize_fn =
      resizeFunctional(tiles_id_buffer_internal);
  std::function<char *(size_t)> pixel_tr_resize_fn =
      resizeFunctional(pixel_tr_buffer_internal);

  int rendered = 0;
  int rendered_buckets = 0;
  int tiles = 0;

  if (num_points != 0) {

    int sh_dim = 0;
    if (sh.size(0) != 0) {
      sh_dim = sh.size(1);
    }

    auto tup = CudaRasterizer::Rasterizer::forward(
        geom_resize_fn, binning_resize_fn, img_resize_fn, sample_resize_fn,
        ranges_resize_fn, gs_list_resize_fn, tiles_id_resize_fn,
        pixel_tr_resize_fn, num_points, degree, sh_dim,
        background.contiguous().data<float>(), width, height,
        means_3d.contiguous().data<float>(), dc.contiguous().data_ptr<float>(),
        sh.contiguous().data_ptr<float>(),
        colors_precomp.contiguous().data<float>(),
        opacities.contiguous().data<float>(),
        scales.contiguous().data_ptr<float>(), scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov_3d_precomp.contiguous().data<float>(),
        view_matrix.contiguous().data<float>(),
        proj_matrix.contiguous().data<float>(),
        campos.contiguous().data<float>(), tan_fov_x, tan_fov_y, prefiltered,
        output_img.contiguous().data<float>(), nullptr, current_stream,
        radii.contiguous().data<int>(), debug, store, precomp, num_buckets,
        num_rendered, reinterpret_cast<char *>(gs_list.contiguous().data_ptr()),
        reinterpret_cast<char *>(ranges.contiguous().data_ptr()),
        use_cuda_graph,
        reinterpret_cast<char *>(img_buffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(geom_buffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(sample_buffer.contiguous().data_ptr()),
        use_load_balancing,
        reinterpret_cast<char *>(tiles_id.contiguous().data_ptr()), num_tiles,
        reinterpret_cast<char *>(pixel_tr.contiguous().data_ptr()),
        out_depth.contiguous().data<float>(), false, false,
        out_lb_set.contiguous().data<short>());

    rendered = std::get<0>(tup);
    rendered_buckets = std::get<1>(tup);
    tiles = std::get<2>(tup);
  }

  return std::make_tuple(
      rendered, rendered_buckets, ranges_buffer_internal,
      gs_list_buffer_internal, output_img, radii, geom_buffer_internal,
      binning_buffer_internal, img_buffer_internal, sample_buffer_internal,
      use_load_balancing ? tiles_id_buffer_internal : torch::Tensor(), tiles,
      use_load_balancing ? pixel_tr_buffer_internal : torch::Tensor(),
      out_lb_set);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &radii, const torch::Tensor &colors,
    const torch::Tensor &opacities, const torch::Tensor &scales,
    const torch::Tensor &rotations, const float scale_modifier,
    const torch::Tensor &cov3D_precomp, const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix, const float tan_fovx, const float tan_fovy,
    const torch::Tensor &dL_dout_color, const torch::Tensor &dc,
    const torch::Tensor &sh, const int degree, const torch::Tensor &campos,
    const torch::Tensor &geomBuffer, const int R,
    const torch::Tensor &binningBuffer, const torch::Tensor &imageBuffer,
    const int B, const torch::Tensor &sampleBuffer, const bool debug,
    const bool precomp, const bool use_cuda_graph, const torch::Tensor &gs_list,
    const torch::Tensor &ranges, const torch::Tensor &dL_ddc_pre,
    const torch::Tensor &dL_dsh_pre, const torch::Tensor &dL_dcolors_pre,
    const torch::Tensor &tiles_id, const int num_tiles,
    const torch::Tensor &pixel_tr, const bool use_load_balancing,
    const bool lb_last_set, const torch::Tensor &lb_last_num) {

  cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream().stream();
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if (sh.size(0) != 0)
    M = sh.size(1);

  // Zero-size tensors for unused gradients — no memory cost
  torch::Tensor dL_dmeans3D = torch::zeros({0, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({0, 3}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({0, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({0, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({0, 6}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({0, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({0, 4}, means3D.options());

  // Color gradients — only allocate for non-graphable path
  torch::Tensor dL_dcolors, dL_ddc, dL_dsh;
  if (!use_cuda_graph) {
    dL_dcolors = torch::zeros({P, NUM_CHAFFELS}, means3D.options());
    dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
    dL_dsh = torch::zeros({P, M, 3}, means3D.options());

    // For non-precomp path, allocate actual gradient tensors
    if (!precomp) {
      dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
      dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
      dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
      dL_dopacity = torch::zeros({P, 1}, means3D.options());
      dL_dcov3D = torch::zeros({P, 6}, means3D.options());
      dL_dscales = torch::zeros({P, 3}, means3D.options());
      dL_drotations = torch::zeros({P, 4}, means3D.options());
    }
  }

  if (P != 0) {
    CudaRasterizer::Rasterizer::backward(
        P, degree, M, R, B, background.contiguous().data<float>(), W, H,
        means3D.contiguous().data<float>(), dc.contiguous().data<float>(),
        sh.contiguous().data<float>(), colors.contiguous().data<float>(),
        opacities.contiguous().data<float>(), scales.data_ptr<float>(),
        scale_modifier, rotations.data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        campos.contiguous().data<float>(), tan_fovx, tan_fovy,
        radii.contiguous().data<int>(), current_stream,
        reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(sampleBuffer.contiguous().data_ptr()),
        dL_dout_color.contiguous().data<float>(),
        dL_dmeans2D.contiguous().data<float>(),
        dL_dconic.contiguous().data<float>(),
        dL_dopacity.contiguous().data<float>(),
        use_cuda_graph ? dL_dcolors_pre.contiguous().data<float>()
                       : dL_dcolors.contiguous().data<float>(),
        dL_dmeans3D.contiguous().data<float>(),
        dL_dcov3D.contiguous().data<float>(),
        use_cuda_graph ? dL_ddc_pre.contiguous().data<float>()
                       : dL_ddc.contiguous().data<float>(),
        use_cuda_graph ? dL_dsh_pre.contiguous().data<float>()
                       : dL_dsh.contiguous().data<float>(),
        dL_dscales.contiguous().data<float>(),
        dL_drotations.contiguous().data<float>(), debug, precomp,
        reinterpret_cast<char *>(gs_list.contiguous().data_ptr()),
        reinterpret_cast<char *>(ranges.contiguous().data_ptr()),
        reinterpret_cast<char *>(tiles_id.contiguous().data_ptr()), num_tiles,
        reinterpret_cast<char *>(pixel_tr.contiguous().data_ptr()),
        use_load_balancing, lb_last_set,
        lb_last_num.contiguous().data<short>());
  }

  if (use_cuda_graph)
    return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor(),
                           torch::Tensor(), torch::Tensor(), torch::Tensor(),
                           torch::Tensor(), torch::Tensor(), torch::Tensor());

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D,
                         dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix) {
  const int P = means3D.size(0);

  torch::Tensor present =
      torch::full({P}, false, means3D.options().dtype(at::kBool));

  if (P != 0) {
    CudaRasterizer::Rasterizer::markVisible(
        P, means3D.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        present.contiguous().data<bool>());
  }

  return present;
}