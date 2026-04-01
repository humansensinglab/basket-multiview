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

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>

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
    torch::Tensor &output_img, const bool use_load_balancing);

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
    const bool lb_last_set, const torch::Tensor &lb_last_num);

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix);

torch::Tensor fusedssim(float C1, float C2, torch::Tensor &img1,
                        torch::Tensor &img2);

torch::Tensor fusedssim_backward(float C1, float C2, torch::Tensor &img1,
                                 torch::Tensor &img2, torch::Tensor &dL_dmap);