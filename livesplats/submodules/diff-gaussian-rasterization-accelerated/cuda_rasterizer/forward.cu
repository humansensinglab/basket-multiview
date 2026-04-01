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

#include "auxiliary.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "forward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs,
                                        const glm::vec3 *means,
                                        glm::vec3 campos, const float *dc,
                                        const float *shs, bool *clamped) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  glm::vec3 pos = means[idx];
  glm::vec3 dir = pos - campos;
  dir = dir / glm::length(dir);

  glm::vec3 *direct_color = ((glm::vec3 *)dc) + idx;
  glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
  glm::vec3 result = SH_C0 * direct_color[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[3] + SH_C2[1] * yz * sh[4] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
               SH_C2[3] * xz * sh[6] + SH_C2[4] * (xx - yy) * sh[7];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
                 SH_C3[1] * xy * z * sh[9] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
                 SH_C3[5] * z * (xx - yy) * sh[13] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
      }
    }
  }
  result += 0.5f;

  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  clamped[3 * idx + 0] = (result.x < 0);
  clamped[3 * idx + 1] = (result.y < 0);
  clamped[3 * idx + 2] = (result.z < 0);
  return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3 &mean, float focal_x, float focal_y,
                               float tan_fovx, float tan_fovy,
                               const float *cov3D, const float *viewmatrix) {
  // The following models the steps outlined by equations 29
  // and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  float3 t = transformPoint4x3(mean, viewmatrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J =
      glm::mat3(focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z), 0.0f,
                focal_y / t.z, -(focal_y * t.y) / (t.z * t.z), 0, 0, 0);

  glm::mat3 W = glm::mat3(viewmatrix[0], viewmatrix[4], viewmatrix[8],
                          viewmatrix[1], viewmatrix[5], viewmatrix[9],
                          viewmatrix[2], viewmatrix[6], viewmatrix[10]);

  glm::mat3 T = W * J;

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3],
                            cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.

  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod,
                             const glm::vec4 rot, float *cov3D) {
  // Create scaling matrix
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;

  // Normalize quaternion to get valid rotation
  glm::vec4 q = rot; // / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z),
                          2.f * (x * z + r * y), 2.f * (x * y + r * z),
                          1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                          2.f * (x * z - r * y), 2.f * (y * z + r * x),
                          1.f - 2.f * (x * x + y * y));

  glm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  glm::mat3 Sigma = glm::transpose(M) * M;

  // Covariance is symmetric, only store upper right
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(
    int P, int D, int M, const float *orig_points, const glm::vec3 *scales,
    const float scale_modifier, const glm::vec4 *rotations,
    const float *opacities, const float *dc, const float *shs, bool *clamped,
    const float *cov3D_precomp, const float *colors_precomp,
    const float *viewmatrix, const float *projmatrix, const glm::vec3 *cam_pos,
    const int W, int H, const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y, int *radii, float2 *points_xy_image,
    float *depths, float *cov3Ds, float *rgb, float4 *conic_opacity,
    const dim3 grid, uint32_t *tiles_touched, bool prefiltered) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx] = 0;
  tiles_touched[idx] = 0;

  // Perform near culling, quit if outside.
  float3 p_view;
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered,
                  p_view))
    return;

  // Transform point by projecting
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1],
                   orig_points[3 * idx + 2]};
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

  // If 3D covariance matrix is precomputed, use it, otherwise compute
  // from scaling and rotation parameters.
  const float *cov3D;
  if (cov3D_precomp != nullptr) {
    cov3D = cov3D_precomp + idx * 6;
  } else {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }

  // Compute 2D screen-space covariance matrix
  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D,
                            viewmatrix);

  constexpr float h_var = 0.3f;
  const float det_cov = cov.x * cov.z - cov.y * cov.y;
  cov.x += h_var;
  cov.z += h_var;
  const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
  float h_convolution_scaling = sqrt(max(
      0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
  // Invert covariance (EWA algorithm)
  const float det = det_cov_plus_h_cov;
  // Invert covariance (EWA algorithm)
  // float det = (cov.x * cov.z - cov.y * cov.y);
  if (det == 0.0f)
    return;
  float det_inv = 1.f / det;
  float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

  // Compute extent in screen space (by finding eigenvalues of
  // 2D covariance matrix). Use extent to compute a bounding rectangle
  // of screen-space tiles that this Gaussian overlaps with. Quit if
  // rectangle covers 0 tiles.
  float mid = 0.5f * (cov.x + cov.z);
  float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
  float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
  uint2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid);
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
    return;

  // If colors have been precomputed, use them, otherwise convert
  // spherical harmonics coefficients to RGB color.
  if (colors_precomp == nullptr) {
    glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3 *)orig_points,
                                          *cam_pos, dc, shs, clamped);
    rgb[idx * C + 0] = result.x;
    rgb[idx * C + 1] = result.y;
    rgb[idx * C + 2] = result.z;
  }

  // Store some useful helper data for the next steps.
  depths[idx] = p_view.z;
  radii[idx] = my_radius;
  // if(idx==0)printf("%d",radii[idx]);
  points_xy_image[idx] = point_image;

  // Inverse 2D covariance and opacity neatly pack into one float4
  conic_opacity[idx] = {conic.x, conic.y, conic.z,
                        opacities[idx] * h_convolution_scaling};
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderCUDA(const uint2 *__restrict__ ranges,
               const uint32_t *__restrict__ point_list,
               const uint32_t *__restrict__ per_tile_bucket_offset,
               uint32_t *__restrict__ bucket_to_tile,
               float *__restrict__ sampled_T, float *__restrict__ sampled_ar,
               int W, int H, const float2 *__restrict__ points_xy_image,
               const float *__restrict__ features,
               const float4 *__restrict__ conic_opacity,
               float *__restrict__ final_T, uint32_t *__restrict__ n_contrib,
               uint32_t *__restrict__ max_contrib,
               const float *__restrict__ bg_color,
               float *__restrict__ out_color, const bool store,
               float *__restrict__ pixel_T,
               const int *__restrict__ div_per_tile) {

  // This kernel assumes TILE_PARTITION divides BLOCK_SIZE.
  constexpr int TILE_PARTITION = 64;

  auto block = cg::this_thread_block();

  const uint32_t tiles_x = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min = {block.group_index().x * BLOCK_X,
                         block.group_index().y * BLOCK_Y};
  const uint2 pix_max = {min(pix_min.x + BLOCK_X, (uint32_t)W),
                         min(pix_min.y + BLOCK_Y, (uint32_t)H)};
  const uint2 pix = {pix_min.x + block.thread_index().x,
                     pix_min.y + block.thread_index().y};
  const uint32_t pix_id = (uint32_t)W * pix.y + pix.x;
  const float2 pixf = {(float)pix.x, (float)pix.y};

  const bool inside = (pix.x < (uint32_t)W) && (pix.y < (uint32_t)H);
  bool done = !inside;

  const uint32_t tile_id =
      block.group_index().y * tiles_x + block.group_index().x;

  uint2 range = ranges[tile_id];

  int to_do = (int)(range.y - range.x);
  const int rounds = (to_do + BLOCK_SIZE - 1) / BLOCK_SIZE;

  uint32_t bbm = (tile_id == 0) ? 0 : per_tile_bucket_offset[tile_id - 1];

  const int num_buckets = (to_do + 31) / 32;
  for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
    const int bucket_idx = i * BLOCK_SIZE + (int)block.thread_rank();
    if (bucket_idx < num_buckets) {
      bucket_to_tile[bbm + (uint32_t)bucket_idx] = tile_id;
    }
  }

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  float T = 1.0f;           // transmittance
  uint32_t contributor = 0; // processed gaussians (including skipped)
  uint32_t last_contributor =
      0;                   // last contributing gaussian index (for n_contrib)
  float C[CHANNELS] = {0}; // accumulated radiance (not including background)
  int last_stored_partition = -1;

  int remaining = to_do;
  for (int batch = 0; batch < rounds; ++batch, remaining -= BLOCK_SIZE) {

    // Stop if all threads in the block are done (all pixels fully opaque or out
    // of bounds)
    const int num_done = __syncthreads_count(done);
    if (num_done == BLOCK_SIZE) {
      break;
    }

    // (a) Cooperative load: pull a batch of gaussian data into shared memory
    const int progress = batch * BLOCK_SIZE + (int)block.thread_rank();
    if ((uint32_t)(range.x + progress) < range.y) {
      const int g = (int)point_list[range.x + (uint32_t)progress];
      collected_id[block.thread_rank()] = g;
      collected_xy[block.thread_rank()] = points_xy_image[g];
      collected_conic_opacity[block.thread_rank()] = conic_opacity[g];
    }
    block.sync();

    // (b) Iterate within this batch
    const int batch_count = min(BLOCK_SIZE, remaining);
    for (int j = 0; !done && j < batch_count; ++j) {

      // Snapshot state every 32 gaussians (bucket boundary)
      if ((j % 32) == 0) {
        sampled_T[bbm * BLOCK_SIZE + block.thread_rank()] = T;
        for (int ch = 0; ch < (int)CHANNELS; ++ch) {
          sampled_ar[bbm * BLOCK_SIZE * CHANNELS + ch * BLOCK_SIZE +
                     block.thread_rank()] = C[ch];
        }
        ++bbm;
      }

      // Store per-partition transmittance for later load-balanced
      // replay. TILE_PARTITION partitions within each tile are indexed by:
      //   global_partition = div_per_tile[tile_id] + local_partition
      // where div_per_tile is a PREFIX OFFSET (base partition index) per tile.

      if (store && j % TILE_PARTITION == 0 && j != 0) {
        const int local_partition =
            (j - 1) / TILE_PARTITION + batch * (BLOCK_SIZE / TILE_PARTITION);
        const int global_partition = div_per_tile[tile_id] + local_partition;
        pixel_T[global_partition * BLOCK_SIZE + block.thread_rank()] = T;
      }

      // --- Evaluate Gaussian contribution at this pixel ---
      ++contributor;

      const float2 xy = collected_xy[j];
      const float2 d = {xy.x - pixf.x, xy.y - pixf.y};

      const float4 con_o = collected_conic_opacity[j];
      const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                          con_o.y * d.x * d.y;

      // Outside ellipse support in exponent space
      if (power > 0.0f)
        continue;

      // Opacity-weighted Gaussian (clamped for stability)
      const float alpha = min(0.99f, con_o.w * exp(power));
      if (alpha < (1.0f / 255.0f))
        continue;

      const float test_T = T * (1.0f - alpha);

      // Early termination: pixel becomes effectively opaque
      if (test_T < 1e-4f) {
        done = true;
        if (store) {
          const int local_partition =
              j / TILE_PARTITION + batch * (BLOCK_SIZE / TILE_PARTITION);
          const int global_partition = div_per_tile[tile_id] + local_partition;
          pixel_T[global_partition * BLOCK_SIZE + block.thread_rank()] = test_T;
        }
        continue;
      }

      // Accumulate color: C += feature * alpha * T
      const int g = collected_id[j];
#pragma unroll
      for (int ch = 0; ch < (int)CHANNELS; ++ch) {
        C[ch] += features[g * CHANNELS + ch] * alpha * T;
      }

      T = test_T;
      last_contributor = contributor;
    }

    if (store && !done) {
      const int last_j = batch_count - 1;
      const int local_partition =
          last_j / TILE_PARTITION + batch * (BLOCK_SIZE / TILE_PARTITION);
      const int global_partition = div_per_tile[tile_id] + local_partition;
      pixel_T[global_partition * BLOCK_SIZE + block.thread_rank()] = T;
    }
  }

  if (inside) {
    final_T[pix_id] = T;
    n_contrib[pix_id] = last_contributor;

// Planar output: channel-major
#pragma unroll
    for (int ch = 0; ch < (int)CHANNELS; ++ch) {
      out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    }
  }

  // Per-tile diagnostic: max contributors in this tile (reduce across block)
  using BlockReduce = cub::BlockReduce<uint32_t, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const uint32_t max_last =
      BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());

  if (block.thread_rank() == 0) {
    max_contrib[tile_id] = max_last;
  }
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y) renderCUDAloadBalancing(
    const uint2 *__restrict__ ranges, const uint32_t *__restrict__ point_list,
    const uint2 *__restrict__ tiles_ID, const int num_tiles, int W, int H,
    const float2 *__restrict__ points_xy_image,
    const float *__restrict__ features,
    const float4 *__restrict__ conic_opacity,
    const float *__restrict__ bg_color, float *__restrict__ out_color,
    float *__restrict__ pixel_tr, short *__restrict__ out_lb_set,
    const bool lb_last_set) {
  //  Load balancer partitions each original tile into subtiles of at
  // most TILE_PARTITION gaussians. This allows us to load one subtile’s worth
  // of gaussian data into shared memory once.

  constexpr int TILE_PARTITION = 64;

  auto block = cg::this_thread_block();

  const int idx = blockIdx.y * gridDim.x + blockIdx.x;
  if (idx >= num_tiles)
    return;

  constexpr float EPS = 1e-4f;

  // tiles_ID encodes:
  //   tile_id:      original tile index in the regular tile grid
  //   order_in_tile: which subtile partition within that original tile
  //   (0,1,2,...)
  const uint32_t tile_id = tiles_ID[idx].x;
  const uint32_t order_in_tile = tiles_ID[idx].y;

  // ranges[idx] gives [start,end) in point_list for this subtile partition.
  const uint2 range = ranges[idx];
  const int toDo =
      int(range.y - range.x); // number of gaussians assigned to this subtile
                              // (toDo <= TILE_PARTITION)

  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 tile_2d = {tile_id % horizontal_blocks,
                         tile_id / horizontal_blocks};

  const uint2 pix_min = {tile_2d.x * BLOCK_X, tile_2d.y * BLOCK_Y};
  const uint2 pix_max = {min(pix_min.x + BLOCK_X, (uint32_t)W),
                         min(pix_min.y + BLOCK_Y, (uint32_t)H)};

  // Each thread corresponds to one pixel in the tile.
  const uint2 pix = {pix_min.x + block.thread_index().x,
                     pix_min.y + block.thread_index().y};

  const uint32_t pix_id = uint32_t(W) * pix.y + pix.x;
  const float2 pixf = {float(pix.x), float(pix.y)};

  const bool inside = (pix.x < (uint32_t)W) && (pix.y < (uint32_t)H);
  bool done = !inside; // out-of-bounds threads participate in loads/sync but do
                       // not rasterize.

  // We start each subtile with an incoming transmittance T:
  // - For the first subtile in a tile (order_in_tile==0), T = 1
  // - For later subtiles, we restore T from the previous subtile’s stored
  // per-pixel transmittance.

  float T = 0.0f;
  if (!done)
    T = (order_in_tile == 0)
            ? 1.0f
            : pixel_tr[(idx - 1) * BLOCK_SIZE + block.thread_rank()];

  float C[CHANNELS] = {0};

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  if (block.thread_rank() < toDo) {
    const int coll_id = int(point_list[range.x + block.thread_rank()]);
    collected_id[block.thread_rank()] = coll_id;
    collected_xy[block.thread_rank()] = points_xy_image[coll_id];
    collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
  }
  block.sync();

  // If we are already fully opaque (or numerically ~0), no need to do work.
  if (fabsf(T) < EPS)
    return;

  // Iterate over the gaussians assigned to this subtile (front-to-back).
  for (int j = 0; j < min(BLOCK_SIZE, toDo); ++j) {

    const float2 xy = collected_xy[j];
    const float2 d = {xy.x - pixf.x, xy.y - pixf.y};

    const float4 con_o = collected_conic_opacity[j];
    const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                        con_o.y * d.x * d.y;

    if (power > 0.0f)
      continue;

    const float alpha = min(0.99f, con_o.w * expf(power));
    if (alpha < 1.0f / 255.0f)
      continue;

    const float test_T = T * (1.0f - alpha);

    if (test_T < 1e-4f) {
      // if (lb_last_set)
      //   out_lb_set[pix_id] = (short)j;
      done = true;
      break;
    }

    const int gid = collected_id[j];

#pragma unroll
    for (int ch = 0; ch < CHANNELS; ++ch) {
      C[ch] += features[gid * CHANNELS + ch] * alpha * T;
    }

    T = test_T;
  }

  // Atomically accumulate this subtile’s contribution into the global output
  // buffer.

  bool is_last_subtile =
      (idx == num_tiles - 1) || (tiles_ID[idx + 1].x != tile_id);

  if (inside) {
#pragma unroll
    for (int ch = 0; ch < CHANNELS; ch++) {
      float contrib = C[ch];
      if (is_last_subtile)
        contrib += T * bg_color[ch];
      atomicAdd(&out_color[ch * H * W + pix_id], contrib);
    }
  }
}

void FORWARD::render(const dim3 grid, dim3 block, const uint2 *ranges,
                     const uint32_t *point_list,
                     const uint32_t *per_tile_bucket_offset,
                     uint32_t *bucket_to_tile, float *sampled_T,
                     float *sampled_ar, int W, int H, const float2 *means2D,
                     const float *colors, const float4 *conic_opacity,
                     float *final_T, uint32_t *n_contrib, uint32_t *max_contrib,
                     const float *bg_color, float *out_color,
                     cudaStream_t stream, float *PixelT, const bool store_lb,
                     const int *div_per_tile) {

  renderCUDA<NUM_CHAFFELS><<<grid, block, 0, stream>>>(
      ranges, point_list, per_tile_bucket_offset, bucket_to_tile, sampled_T,
      sampled_ar, W, H, means2D, colors, conic_opacity, final_T, n_contrib,
      max_contrib, bg_color, out_color, store_lb, PixelT, div_per_tile);
}

void FORWARD::render_load_balancing(
    const dim3 grid, dim3 block, const uint2 *ranges,
    const uint32_t *point_list, const uint2 *tiles_ID, const int num_tiles,
    int W, int H, const float2 *means2D, const float *colors,
    const float4 *conic_opacity, const float *bg_color, float *out_color,
    float *pixel_tr, cudaStream_t stream, short *out_lb_set,
    const bool lb_last_set) {

  int dimx = ceil(sqrt(num_tiles));
  dim3 tile_grid = {dimx, ceil(static_cast<float>(num_tiles) / dimx), 1};

  renderCUDAloadBalancing<NUM_CHAFFELS><<<tile_grid, block, 0, stream>>>(
      ranges, point_list, tiles_ID, num_tiles, W, H, means2D, colors,
      conic_opacity, bg_color, out_color, pixel_tr, out_lb_set, lb_last_set);
}

void FORWARD::preprocess(int P, int D, int M, const float *means3D,
                         const glm::vec3 *scales, const float scale_modifier,
                         const glm::vec4 *rotations, const float *opacities,
                         const float *dc, const float *shs, bool *clamped,
                         const float *cov3D_precomp,
                         const float *colors_precomp, const float *viewmatrix,
                         const float *projmatrix, const glm::vec3 *cam_pos,
                         const int W, int H, const float focal_x, float focal_y,
                         const float tan_fovx, float tan_fovy, int *radii,
                         float2 *means2D, float *depths, float *cov3Ds,
                         float *rgb, float4 *conic_opacity, const dim3 grid,
                         uint32_t *tiles_touched, bool prefiltered,
                         cudaStream_t stream) {

  preprocessCUDA<NUM_CHAFFELS><<<(P + 255) / 256, 256, 0, stream>>>(
      P, D, M, means3D, scales, scale_modifier, rotations, opacities, dc, shs,
      clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, cam_pos,
      W, H, tan_fovx, tan_fovy, focal_x, focal_y, radii, means2D, depths,
      cov3Ds, rgb, conic_opacity, grid, tiles_touched, prefiltered);
}