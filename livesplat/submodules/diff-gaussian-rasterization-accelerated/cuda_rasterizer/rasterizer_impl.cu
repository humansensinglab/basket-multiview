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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rasterizer_impl.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <glm/glm.hpp>
#include <torch/extension.h>
#include <torch/torch.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "backward.h"
#include "forward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n) {
  uint32_t msb = sizeof(n) * 4;
  uint32_t step = msb;
  while (step > 1) {
    step /= 2;
    if (n >> msb)
      msb += step;
    else
      msb -= step;
  }
  if (n >> msb)
    msb++;
  return msb;
}

__device__ inline float evaluate_opacity_factor(const float dx, const float dy,
                                                const float4 co) {
  return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

template <uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT>
__device__ inline float max_contrib_power_rect_gaussian_float(
    const float4 co, const float2 mean, const glm::vec2 rect_min,
    const glm::vec2 rect_max, glm::vec2 &max_pos) {
  const float x_min_diff = rect_min.x - mean.x;
  const float x_left = x_min_diff > 0.0f;
  // const float x_left = mean.x < rect_min.x;
  const float not_in_x_range = x_left + (mean.x > rect_max.x);

  const float y_min_diff = rect_min.y - mean.y;
  const float y_above = y_min_diff > 0.0f;
  // const float y_above = mean.y < rect_min.y;
  const float not_in_y_range = y_above + (mean.y > rect_max.y);

  max_pos = {mean.x, mean.y};
  float max_contrib_power = 0.0f;

  if ((not_in_y_range + not_in_x_range) > 0.0f) {
    const float px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
    const float py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;

    const float dx = copysign(float(PATCH_WIDTH), x_min_diff);
    const float dy = copysign(float(PATCH_HEIGHT), y_min_diff);

    const float diffx = mean.x - px;
    const float diffy = mean.y - py;

    const float rcp_dxdxcox =
        __frcp_rn(PATCH_WIDTH * PATCH_WIDTH * co.x); // = 1.0 / (dx*dx*co.x)
    const float rcp_dydycoz =
        __frcp_rn(PATCH_HEIGHT * PATCH_HEIGHT * co.z); // = 1.0 / (dy*dy*co.z)

    const float tx =
        not_in_y_range *
        __saturatef((dx * co.x * diffx + dx * co.y * diffy) * rcp_dxdxcox);
    const float ty =
        not_in_x_range *
        __saturatef((dy * co.y * diffx + dy * co.z * diffy) * rcp_dydycoz);
    max_pos = {px + tx * dx, py + ty * dy};

    const float2 max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
    max_contrib_power =
        evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
  }

  return max_contrib_power;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P, const float *orig_points,
                             const float *viewmatrix, const float *projmatrix,
                             bool *present) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;

  float3 p_view;
  present[idx] =
      in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(int P, const float2 *points_xy,
                                  const float4 *__restrict__ conic_opacity,
                                  const float *depths, const uint32_t *offsets,
                                  uint64_t *gaussian_keys_unsorted,
                                  uint32_t *gaussian_values_unsorted,
                                  int *radii, dim3 grid, int2 *rects) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;

  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] > 0) {
    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    const uint32_t offset_to = offsets[idx];
    uint2 rect_min, rect_max;

    if (rects == nullptr)
      getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
    else
      getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

    const float2 xy = points_xy[idx];
    const float4 co = conic_opacity[idx];
    const float opacity_threshold = 1.0f / 255.0f;
    const float opacity_factor_threshold = logf(co.w / opacity_threshold);

    // For each tile that the bounding rect overlaps, emit a
    // key/value pair. The key is |  tile ID  |      depth      |,
    // and the value is the ID of the Gaussian. Sorting the values
    // with this key yields Gaussian IDs in a list, such that they
    // are first sorted by tile and then by depth.
    for (int y = rect_min.y; y < rect_max.y; y++) {
      for (int x = rect_min.x; x < rect_max.x; x++) {
        const glm::vec2 tile_min(x * BLOCK_X, y * BLOCK_Y);
        const glm::vec2 tile_max((x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1);

        glm::vec2 max_pos;
        float max_opac_factor = 0.0f;
        max_opac_factor =
            max_contrib_power_rect_gaussian_float<BLOCK_X - 1, BLOCK_Y - 1>(
                co, xy, tile_min, tile_max, max_pos);

        uint64_t key = y * grid.x + x;
        key <<= 32;
        key |= *((uint32_t *)&depths[idx]);
        if (max_opac_factor <= opacity_factor_threshold) {
          gaussian_keys_unsorted[off] = key;
          gaussian_values_unsorted[off] = idx;
          off++;
        }
      }
    }

    for (; off < offset_to; ++off) {
      uint64_t key = (uint32_t)-1;
      key <<= 32;
      const float depth = FLT_MAX;
      key |= *((uint32_t *)&depth);
      gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
      gaussian_keys_unsorted[off] = key;
    }
  }
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t *point_list_keys,
                                   uint2 *ranges) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= L)
    return;

  // Read tile ID from key. Update start/end of tile range if at limit.
  uint64_t key = point_list_keys[idx];
  uint32_t currtile = key >> 32;
  bool valid_tile = currtile != (uint32_t)-1;

  if (idx == 0)
    ranges[currtile].x = 0;
  else {
    uint32_t prevtile = point_list_keys[idx - 1] >> 32;
    if (currtile != prevtile) {
      ranges[prevtile].y = idx;
      if (valid_tile)
        ranges[currtile].x = idx;
    }
  }
  if (idx == L - 1 && valid_tile)
    ranges[currtile].y = L;
}

// for each tile, see how many buckets/warps are needed to store the state
__global__ void perTileBucketCountPre(int T, const int *ranges,
                                      uint32_t *bucketCount) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= T)
    return;

  uint2 range;
  range.x = ranges[idx * 2];
  range.y = ranges[idx * 2 + 1];
  int num_splats = range.y - range.x;
  int num_buckets = (num_splats + 31) / 32;
  bucketCount[idx] = (uint32_t)num_buckets;
}

// for each tile, see how many buckets/warps are needed to store the state
__global__ void perTileBucketCount(int T, uint2 *ranges,
                                   uint32_t *bucketCount) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= T)
    return;

  uint2 range = ranges[idx];
  int num_splats = range.y - range.x;
  int num_buckets = (num_splats + 31) / 32;
  bucketCount[idx] = (uint32_t)num_buckets;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(int P, float *means3D,
                                             float *viewmatrix,
                                             float *projmatrix, bool *present) {
  checkFrustum<<<(P + 255) / 256, 256>>>(P, means3D, viewmatrix, projmatrix,
                                         present);
}

CudaRasterizer::GeometryState
CudaRasterizer::GeometryState::fromChunk(char *&chunk, size_t P,
                                         cudaStream_t stream = 0) {
  GeometryState geom;
  obtain(chunk, geom.depths, P, 128);
  obtain(chunk, geom.clamped, P * 3, 128);
  obtain(chunk, geom.internal_radii, P, 128);
  obtain(chunk, geom.means2D, P, 128);
  obtain(chunk, geom.cov3D, P * 6, 128);
  obtain(chunk, geom.conic_opacity, P, 128);
  obtain(chunk, geom.rgb, P * 3, 128);
  obtain(chunk, geom.tiles_touched, P, 128);
  cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched,
                                geom.tiles_touched, P, stream);
  obtain(chunk, geom.scanning_space, geom.scan_size, 128);
  obtain(chunk, geom.point_offsets, P, 128);
  return geom;
}

CudaRasterizer::ImageState
CudaRasterizer::ImageState::fromChunk(char *&chunk, size_t N,
                                      cudaStream_t stream = 0) {
  ImageState img;
  obtain(chunk, img.accum_alpha, N, 128);
  obtain(chunk, img.n_contrib, N, 128);
  obtain(chunk, img.ranges, N, 128);
  int *dummy;
  int *wummy;

  obtain(chunk, img.max_contrib, N, 128);
  obtain(chunk, img.pixel_colors, N * NUM_CHAFFELS, 128);
  obtain(chunk, img.bucket_count, N, 128);
  obtain(chunk, img.bucket_offsets, N, 128);
  cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size,
                                img.bucket_count, img.bucket_count, N, stream);
  obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size,
         128);

  return img;
}

CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(char *&chunk,
                                                                   size_t C) {
  SampleState sample;
  obtain(chunk, sample.bucket_to_tile, C * BLOCK_SIZE, 128);
  obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
  obtain(chunk, sample.ar, NUM_CHAFFELS * C * BLOCK_SIZE, 128);
  return sample;
}

CudaRasterizer::BinningState
CudaRasterizer::BinningState::fromChunk(char *&chunk, size_t P,
                                        cudaStream_t stream = 0) {
  BinningState binning;
  obtain(chunk, binning.point_list, P, 128);
  obtain(chunk, binning.point_list_unsorted, P, 128);
  obtain(chunk, binning.point_list_keys, P, 128);
  obtain(chunk, binning.point_list_keys_unsorted, P, 128);
  cub::DeviceRadixSort::SortPairs(
      nullptr, binning.sorting_size, binning.point_list_keys_unsorted,
      binning.point_list_keys, binning.point_list_unsorted, binning.point_list,
      P, 0, sizeof(int) * 8, stream);
  obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
  return binning;
}

CudaRasterizer::OrderBin CudaRasterizer::OrderBin::fromChunk(char *&chunk,
                                                             size_t N) {
  OrderBin ordered_bin;
  obtain(chunk, ordered_bin.gaussian_list, N, 128);

  return ordered_bin;
}
CudaRasterizer::TransState CudaRasterizer::TransState::fromChunk(char *&chunk,
                                                                 size_t N) {
  TransState pixel;
  obtain(chunk, pixel.T, N, 128);

  return pixel;
}
CudaRasterizer::OrderImg CudaRasterizer::OrderImg::fromChunk(char *&chunk,
                                                             size_t N) {
  OrderImg ordered_img;
  obtain(chunk, ordered_img.ranges, N, 128);

  return ordered_img;
}

__global__ void zero(int N, int *space) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= N)
    return;
  space[idx] = 0;
}

__global__ void set(int N, uint32_t *where, int *space) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= N)
    return;

  int off = (idx == 0) ? 0 : where[idx - 1];

  space[off] = 1;
}

namespace CudaRasterizer {
static void preprocess_and_sort(
    std::function<char *(size_t)> geometryBuffer,
    std::function<char *(size_t)> binningBuffer,
    std::function<char *(size_t)> imageBuffer,
    std::function<char *(size_t)> sampleBuffer, const int P, int D, int M,
    const float *background, const int width, const int height,
    const float *means3D, const float *dc, const float *shs,
    const float *colors_precomp, const float *opacities, const float *scales,
    const float scale_modifier, const float *rotations,
    const float *cov3D_precomp, const float *viewmatrix,
    const float *projmatrix, const float *cam_pos, const float tan_fovx,
    const float tan_fovy, const float focal_x, const float focal_y,
    const bool prefiltered, const dim3 tile_grid, cudaStream_t stream,
    bool debug, int *radii, GeometryState &geomState,
    BinningState &binningState, ImageState &imgState, SampleState &sampleState,
    int &rendered_count, unsigned int &bucket_sum) {

  // ------------------------------------------------------------------------
  // 1. Geometry preprocessing state
  // ------------------------------------------------------------------------

  size_t geom_chunk_size = required<GeometryState>(P);
  char *geom_chunk_ptr = geometryBuffer(geom_chunk_size);
  geomState = GeometryState::fromChunk(geom_chunk_ptr, P, stream);

  // Use internal radii buffer if none was provided
  if (radii == nullptr) {
    radii = geomState.internal_radii;
  }

  if (NUM_CHAFFELS != 3 && colors_precomp == nullptr) {
    throw std::runtime_error(
        "For non-RGB output, precomputed Gaussian colors must be provided.");
  }

  CHECK_CUDA(FORWARD::preprocess(
                 P, D, M, means3D, (glm::vec3 *)scales, scale_modifier,
                 (glm::vec4 *)rotations, opacities, dc, shs, geomState.clamped,
                 cov3D_precomp, colors_precomp, viewmatrix, projmatrix,
                 (glm::vec3 *)cam_pos, width, height, focal_x, focal_y,
                 tan_fovx, tan_fovy, radii, geomState.means2D, geomState.depths,
                 geomState.cov3D, geomState.rgb, geomState.conic_opacity,
                 tile_grid, geomState.tiles_touched, prefiltered, stream),
             debug);

  // ------------------------------------------------------------------------
  // 2. Image state allocation
  // ------------------------------------------------------------------------

  size_t img_chunk_size = required<ImageState>(width * height);
  char *img_chunk_ptr = imageBuffer(img_chunk_size);
  imgState = ImageState::fromChunk(img_chunk_ptr, width * height, stream);

  // ------------------------------------------------------------------------
  // 3. Compute number of duplicated Gaussian instances
  //    - prefix sum over tiles_touched
  // ------------------------------------------------------------------------

  CHECK_CUDA(cub::DeviceScan::InclusiveSum(
                 geomState.scanning_space, geomState.scan_size,
                 geomState.tiles_touched, geomState.point_offsets, P, stream),
             debug);

  // Read back total number of duplicated instances
  CHECK_CUDA(cudaMemcpyAsync(&rendered_count, geomState.point_offsets + P - 1,
                             sizeof(int), cudaMemcpyDeviceToHost, stream),
             debug);
  cudaStreamSynchronize(stream);

  // ------------------------------------------------------------------------
  // 4. Binning state allocation (one entry per duplicated instance)
  // ------------------------------------------------------------------------

  size_t binning_chunk_size = required<BinningState>(rendered_count);
  char *binning_chunk_ptr = binningBuffer(binning_chunk_size);
  binningState =
      BinningState::fromChunk(binning_chunk_ptr, rendered_count, stream);

  // ------------------------------------------------------------------------
  // 5. Duplicate Gaussians per touched tile and generate sort keys
  //    - key = [tile_id | depth]
  // ------------------------------------------------------------------------

  duplicateWithKeys<<<(P + 255) / 256, 256, 0, stream>>>(
      P, geomState.means2D, geomState.conic_opacity, geomState.depths,
      geomState.point_offsets, binningState.point_list_keys_unsorted,
      binningState.point_list_unsorted, radii, tile_grid, nullptr);
  CHECK_CUDA(, debug);

  // ------------------------------------------------------------------------
  // 6. Sort duplicated Gaussian instances by (tile | depth)
  // ------------------------------------------------------------------------

  int bit = getHigherMsb(tile_grid.x * tile_grid.y);

  CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                 binningState.list_sorting_space, binningState.sorting_size,
                 binningState.point_list_keys_unsorted,
                 binningState.point_list_keys, binningState.point_list_unsorted,
                 binningState.point_list, rendered_count, 0, 32 + bit, stream),
             debug);

  // ------------------------------------------------------------------------
  // 7. Identify per-tile ranges in sorted Gaussian list
  // ------------------------------------------------------------------------

  CHECK_CUDA(cudaMemsetAsync(imgState.ranges, 0,
                             tile_grid.x * tile_grid.y * sizeof(uint2), stream),
             debug);
  cudaStreamSynchronize(stream);

  if (rendered_count > 0) {
    identifyTileRanges<<<(rendered_count + 255) / 256, 256, 0, stream>>>(
        rendered_count, binningState.point_list_keys, imgState.ranges);
  }
  CHECK_CUDA(, debug);

  // ------------------------------------------------------------------------
  // 8. Bucket counting (used to allocate sample state)
  // ------------------------------------------------------------------------

  int grid_tile_count = tile_grid.x * tile_grid.y;

  perTileBucketCount<<<(grid_tile_count + 255) / 256, 256>>>(
      grid_tile_count, imgState.ranges, imgState.bucket_count);

  CHECK_CUDA(cub::DeviceScan::InclusiveSum(
                 imgState.bucket_count_scanning_space,
                 imgState.bucket_count_scan_size, imgState.bucket_count,
                 imgState.bucket_offsets, grid_tile_count, stream),
             debug);

  CHECK_CUDA(cudaMemcpyAsync(
                 &bucket_sum, imgState.bucket_offsets + grid_tile_count - 1,
                 sizeof(unsigned int), cudaMemcpyDeviceToHost, stream),
             debug);
  cudaStreamSynchronize(stream);

  // ------------------------------------------------------------------------
  // 9. Sample state allocation (one entry per bucket)
  // ------------------------------------------------------------------------

  size_t sample_chunk_size = required<SampleState>(bucket_sum);
  char *sample_chunk_ptr = sampleBuffer(sample_chunk_size);
  sampleState = SampleState::fromChunk(sample_chunk_ptr, bucket_sum);
}

std::tuple<int, int, int, std::vector<int>, std::vector<int>>
export_precomputed_data(GeometryState &geomState, BinningState &binningState,
                        ImageState &imgState, SampleState &sampleState,
                        std::function<char *(size_t)> gs_listBuffer,
                        std::function<char *(size_t)> rangesBuffer,
                        std::function<char *(size_t)> tiles_idBuffer,
                        std::function<char *(size_t)> pixel_trBuffer,
                        dim3 tile_grid, dim3 block, int width, int height,
                        const float *background, int rendered_count,
                        unsigned int bucket_sum, bool use_load_balancing,
                        const float *colors_precomp, float *out_color,
                        float *depth_out, cudaStream_t stream, bool debug,
                        bool lb_last_set) {

  // ------------------------------------------------------------------------
  //    Export ordering buffers (store=true)
  //     - baseline: ranges + gaussian_list on original tile grid
  //     - load_balancing: subdivided ranges + tile mapping + pixelState.T
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
  // BASELINE (no load balancing)
  // ------------------------------------------------------------------------

  if (!use_load_balancing) {

    size_t ord_bin_size = required<OrderBin>(rendered_count);
    char *ord_binning_chunkptr = gs_listBuffer(ord_bin_size);
    OrderBin ord_binningState =
        OrderBin::fromChunk(ord_binning_chunkptr, rendered_count);

    size_t ord_img_size = required<OrderImg>(tile_grid.x * tile_grid.y);
    char *ord_image_chunkptr = rangesBuffer(ord_img_size);
    OrderImg ord_imgState =
        OrderImg::fromChunk(ord_image_chunkptr, tile_grid.x * tile_grid.y);

    cudaMemcpyAsync(ord_imgState.ranges, imgState.ranges,
                    tile_grid.x * tile_grid.y * sizeof(uint2),
                    cudaMemcpyDeviceToDevice, stream);

    cudaMemcpyAsync(ord_binningState.gaussian_list, binningState.point_list,
                    rendered_count * sizeof(int), cudaMemcpyDeviceToDevice,
                    stream);

    return std::make_tuple(rendered_count, bucket_sum, 0, std::vector<int>(),
                           std::vector<int>());
  }

  // ------------------------------------------------------------------------
  // LOAD BALANCING PATH
  // ------------------------------------------------------------------------

  size_t ord_bin_size = required<OrderBin>(rendered_count);
  char *ord_binning_chunkptr = gs_listBuffer(ord_bin_size);
  OrderBin ord_binningState =
      OrderBin::fromChunk(ord_binning_chunkptr, rendered_count);

  std::vector<uint2> ranges_new;
  std::vector<uint2> ranges_old(tile_grid.x * tile_grid.y);
  std::vector<uint2> tiles_and_order;
  std::vector<int> num_of_div_per_tile;

  cudaMemcpyAsync(ranges_old.data(), imgState.ranges,
                  tile_grid.x * tile_grid.y * sizeof(uint2),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int total_tiles = Rasterizer::load_balancer(
      ranges_old.data(), &ranges_new, &tiles_and_order,
      tile_grid.x * tile_grid.y, &num_of_div_per_tile);

  size_t ord_pix_size = required<TransState>(total_tiles * 256);
  char *ord_pix_chunkptr = pixel_trBuffer(ord_pix_size);
  TransState pixelState =
      TransState::fromChunk(ord_pix_chunkptr, total_tiles * 256);

  size_t ord_img_size = required<OrderImg>(total_tiles);
  char *ord_image_chunkptr = rangesBuffer(ord_img_size);
  OrderImg ord_imgState = OrderImg::fromChunk(ord_image_chunkptr, total_tiles);

  size_t ord_tiles_size = required<OrderImg>(total_tiles);
  char *ord_tiles_chunkptr = tiles_idBuffer(ord_tiles_size);
  OrderImg ord_tilesState =
      OrderImg::fromChunk(ord_tiles_chunkptr, total_tiles);

  int *div_per_tile = nullptr;
  cudaMalloc((void **)&div_per_tile, sizeof(int) * tile_grid.x * tile_grid.y);

  cudaMemcpyAsync(div_per_tile, num_of_div_per_tile.data(),
                  tile_grid.x * tile_grid.y * sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  cudaMemcpyAsync(ord_imgState.ranges, ranges_new.data(),
                  total_tiles * sizeof(uint2), cudaMemcpyHostToDevice, stream);

  cudaMemcpyAsync(ord_binningState.gaussian_list, binningState.point_list,
                  rendered_count * sizeof(int), cudaMemcpyDeviceToDevice,
                  stream);

  cudaMemcpyAsync(ord_tilesState.ranges, tiles_and_order.data(),
                  total_tiles * sizeof(uint2), cudaMemcpyHostToDevice, stream);

  CHECK_CUDA(cudaMemsetAsync(pixelState.T, 0, total_tiles * 256 * sizeof(float),
                             stream),
             debug);

  cudaStreamSynchronize(stream);

  const float *feature_ptr =
      colors_precomp != nullptr ? colors_precomp : geomState.rgb;

  CHECK_CUDA(FORWARD::render(tile_grid, block, imgState.ranges,
                             binningState.point_list, imgState.bucket_offsets,
                             sampleState.bucket_to_tile, sampleState.T,
                             sampleState.ar, width, height, geomState.means2D,
                             feature_ptr, geomState.conic_opacity,
                             imgState.accum_alpha, imgState.n_contrib,
                             imgState.max_contrib, background, out_color,
                             stream, pixelState.T, true, div_per_tile),
             debug);

  return std::make_tuple(rendered_count, bucket_sum, total_tiles,
                         std::vector<int>(), std::vector<int>());
}

} // namespace CudaRasterizer

std::tuple<int, int, int, std::vector<int>, std::vector<int>>
CudaRasterizer::Rasterizer::forward(
    std::function<char *(size_t)> geometryBuffer,
    std::function<char *(size_t)> binningBuffer,
    std::function<char *(size_t)> imageBuffer,
    std::function<char *(size_t)> sampleBuffer,
    std::function<char *(size_t)> rangesBuffer,
    std::function<char *(size_t)> gs_listBuffer,
    std::function<char *(size_t)> tiles_idBuffer,
    std::function<char *(size_t)> pixel_trBuffer, const int P, int D, int M,
    const float *background, const int width, int height, const float *means3D,
    const float *dc, const float *shs, const float *colors_precomp,
    const float *opacities, const float *scales, const float scale_modifier,
    const float *rotations, const float *cov3D_precomp, const float *viewmatrix,
    const float *projmatrix, const float *cam_pos, const float tan_fovx,
    float tan_fovy, const bool prefiltered, float *out_color,
    float *out_color_blending, cudaStream_t stream, int *radii, bool debug,
    const bool store_data, const bool precomputed, const int num_buckets,
    const int num_rendered, char *gs_list, char *ranges,
    const bool use_cuda_graph, char *img_buff, char *geom_buff,
    char *sample_buff, const bool use_load_balancing, char *tiles_id,
    const int num_tiles, char *pixel_tr, float *depth_out, const bool depth_reg,
    const bool lb_last_set, short *out_lb_set) {

  const float focal_y = height / (2.0f * tan_fovy);
  const float focal_x = width / (2.0f * tan_fovx);

  dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X,
                 (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  dim3 block(BLOCK_X, BLOCK_Y, 1);

  if (!precomputed) {

    // --------------------------------------------------------------------------
    // Full pipeline (no precomputation)
    // --------------------------------------------------------------------------

    GeometryState geomState;
    BinningState binningState;
    ImageState imgState;
    SampleState sampleState;
    int rendered_count = 0;
    unsigned int bucket_sum = 0;

    preprocess_and_sort(
        geometryBuffer, binningBuffer, imageBuffer, sampleBuffer, P, D, M,
        background, width, height, means3D, dc, shs, colors_precomp, opacities,
        scales, scale_modifier, rotations, cov3D_precomp, viewmatrix,
        projmatrix, cam_pos, tan_fovx, tan_fovy, focal_x, focal_y, prefiltered,
        tile_grid, stream, debug, radii, geomState, binningState, imgState,
        sampleState, rendered_count, bucket_sum);

    if (store_data)
      return export_precomputed_data(
          geomState, binningState, imgState, sampleState, gs_listBuffer,
          rangesBuffer, tiles_idBuffer, pixel_trBuffer, tile_grid, block, width,
          height, background, rendered_count, bucket_sum, use_load_balancing,
          colors_precomp, out_color, depth_out, stream, debug, lb_last_set);

    // ------------------------------------------------------------------------
    // Rendering (standard per-tile blending; store=false path,
    // precopm=false)
    // ------------------------------------------------------------------------

    const float *feature_ptr =
        colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::render(tile_grid, block, imgState.ranges,
                               binningState.point_list, imgState.bucket_offsets,
                               sampleState.bucket_to_tile, sampleState.T,
                               sampleState.ar, width, height, geomState.means2D,
                               feature_ptr, geomState.conic_opacity,
                               imgState.accum_alpha, imgState.n_contrib,
                               imgState.max_contrib, background, out_color,
                               stream, nullptr, false, nullptr),
               debug)

    CHECK_CUDA(cudaMemcpyAsync(imgState.pixel_colors, out_color,
                               sizeof(float) * width * height * NUM_CHAFFELS,
                               cudaMemcpyDeviceToDevice, stream),
               debug);
    cudaStreamSynchronize(stream);
    return std::make_tuple(rendered_count, bucket_sum, 0, std::vector<int>(),
                           std::vector<int>());
  }

  // --------------------------------------------------------------------------
  // Precomputed pipeline (precomp == true)
  //
  // Inputs provide precomputed ordering:
  //   - rangesState: per-(sub)tile [start,end) ranges into gs_listState
  //   - gs_listState: (duplicated) gaussian indices sorted by (tile|depth)
  //   - tilesState / pixelState: load-balancing metadata (only used if
  //   enabled)
  //
  // Notes:
  //   - We still run preprocess() to compute per-Gaussian screen-space
  //   quantities.
  //   - graphable controls whether we allocate states via callbacks or use
  //     preallocated buffers for CUDA graph capture.
  // --------------------------------------------------------------------------
  else {

    // ------------------------------------------------------------------------
    // 0. Reconstruct precomputed ordering / load-balancing state from raw
    // buffers
    // ------------------------------------------------------------------------

    OrderImg rangesState = OrderImg::fromChunk(ranges, num_tiles);
    OrderBin gs_listState = OrderBin::fromChunk(gs_list, num_rendered);
    OrderImg tilesState = OrderImg::fromChunk(tiles_id, num_tiles);
    TransState pixelState =
        TransState::fromChunk(pixel_tr, num_tiles * BLOCK_SIZE);

    // 1. Allocate states

    GeometryState geomState = GeometryState::fromChunk(geom_buff, P, stream);
    ImageState imgState = ImageState::fromChunk(img_buff, width * height, stream);
    SampleState sampleState = SampleState::fromChunk(sample_buff, num_buckets);

    if (radii == nullptr)
      radii = geomState.internal_radii;

    if (NUM_CHAFFELS != 3 && colors_precomp == nullptr)
      throw std::runtime_error(
          "For non-RGB, provide precomputed Gaussian colors!");

    // ----------------------------------------------------------------------
    // 2. Preprocess Gaussians (projection, conics, SH→RGB, etc.)
    //    Note: ordering is precomputed, but geometry still needs
    //    preprocessing.
    // ----------------------------------------------------------------------

    CHECK_CUDA(FORWARD::preprocess(
                   P, D, M, means3D, (glm::vec3 *)scales, scale_modifier,
                   (glm::vec4 *)rotations, opacities, dc, shs,
                   geomState.clamped, cov3D_precomp, colors_precomp, viewmatrix,
                   projmatrix, (glm::vec3 *)cam_pos, width, height, focal_x,
                   focal_y, tan_fovx, tan_fovy, radii, geomState.means2D,
                   geomState.depths, geomState.cov3D, geomState.rgb,
                   geomState.conic_opacity, tile_grid, geomState.tiles_touched,
                   prefiltered, stream),
               debug);

    // ----------------------------------------------------------------------
    // 3. Rendering using precomputed ordering
    // ----------------------------------------------------------------------

    const float *feature_ptr =
        colors_precomp != nullptr ? colors_precomp : geomState.rgb;

    if (use_load_balancing) {

      CHECK_CUDA(FORWARD::render_load_balancing(
                     tile_grid, block, rangesState.ranges,
                     gs_listState.gaussian_list, tilesState.ranges, num_tiles,
                     width, height, geomState.means2D, feature_ptr,
                     geomState.conic_opacity, background, out_color,
                     pixelState.T, stream, out_lb_set, lb_last_set),
                 debug);
    } else {

      int grid_tile_count = tile_grid.x * tile_grid.y;

      perTileBucketCount<<<(grid_tile_count + 255) / 256, 256, 0, stream>>>(
          grid_tile_count, rangesState.ranges, imgState.bucket_count);

      CHECK_CUDA(cub::DeviceScan::InclusiveSum(
                     imgState.bucket_count_scanning_space,
                     imgState.bucket_count_scan_size, imgState.bucket_count,
                     imgState.bucket_offsets, grid_tile_count, stream),
                 debug);

      CHECK_CUDA(FORWARD::render(
                     tile_grid, block, rangesState.ranges,
                     gs_listState.gaussian_list, imgState.bucket_offsets,
                     sampleState.bucket_to_tile, sampleState.T, sampleState.ar,
                     width, height, geomState.means2D, feature_ptr,
                     geomState.conic_opacity, imgState.accum_alpha,
                     imgState.n_contrib, imgState.max_contrib, background,
                     out_color, stream, nullptr, false, nullptr),
                 debug);
    }

    CHECK_CUDA(cudaMemcpyAsync(imgState.pixel_colors, out_color,
                               sizeof(float) * width * height * NUM_CHAFFELS,
                               cudaMemcpyDeviceToDevice, stream),
               debug);

    if (!use_cuda_graph)
      cudaStreamSynchronize(stream);

    return std::make_tuple(num_rendered, num_buckets, 0, std::vector<int>(),
                           std::vector<int>());
  }
}

int CudaRasterizer::Rasterizer::load_balancer(const uint2 *ranges,
                                              std::vector<uint2> *ranges_new,
                                              std::vector<uint2> *tiles_order,
                                              const int tile_number,
                                              std::vector<int> *div_per_tile) {
  int BN = 64;
  int total_tiles = 0;
  int div_per_tiles = 0;
  for (unsigned int k = 0; k < tile_number; k++) {
    div_per_tile->push_back(div_per_tiles);
    div_per_tiles++;
    int range_size = ranges[k].y - ranges[k].x;
    unsigned it = 1;
    while (range_size > BN) {
      ranges_new->push_back(
          {ranges[k].x + BN * (it - 1), ranges[k].x + BN * (it)});
      tiles_order->push_back({k, it - 1});
      it++;
      range_size -= BN;
      total_tiles++;
      div_per_tiles++;
    }
    ranges_new->push_back(
        {ranges[k].x + BN * (it - 1), min(ranges[k].y, ranges[k].x + BN * it)});
    tiles_order->push_back({k, it - 1});
    total_tiles++;
  }

  return total_tiles;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
    const int P, int D, int M, int R, int B, const float *background,
    const int width, int height, const float *means3D, const float *dc,
    const float *shs, const float *colors_precomp, const float *opacities,
    const float *scales, const float scale_modifier, const float *rotations,
    const float *cov3D_precomp, const float *viewmatrix,
    const float *projmatrix, const float *campos, const float tan_fovx,
    float tan_fovy, const int *radii, cudaStream_t stream, char *geom_buffer,
    char *binning_buffer, char *img_buffer, char *sample_buffer,
    const float *dL_dpix, float *dL_dmean2D, float *dL_dconic,
    float *dL_dopacity, float *dL_dcolor, float *dL_dmean3D, float *dL_dcov3D,
    float *dL_ddc, float *dL_dsh, float *dL_dscale, float *dL_drot, bool debug,
    bool precomp, char *gs_list, char *ranges, char *tiles_id,
    const int num_tiles, char *pixel_tr, const bool use_load_balancing,
    const bool lb_last_set, const short *lb_last_num) {

  dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X,
                 (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  GeometryState geomState = GeometryState::fromChunk(geom_buffer, P, stream);
  ImageState imgState =
      ImageState::fromChunk(img_buffer, width * height, stream);
  SampleState sampleState = SampleState::fromChunk(sample_buffer, B);

  if (radii == nullptr)
    radii = geomState.internal_radii;

  const float focal_y = height / (2.0f * tan_fovy);
  const float focal_x = width / (2.0f * tan_fovx);

  const float *color_ptr =
      (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;

  if (!precomp) {
    BinningState binningState =
        BinningState::fromChunk(binning_buffer, R, stream);

    CHECK_CUDA(BACKWARD::render(
                   tile_grid, block, imgState.ranges, binningState.point_list,
                   width, height, R, B, imgState.bucket_offsets,
                   sampleState.bucket_to_tile, sampleState.T, sampleState.ar,
                   background, geomState.means2D, geomState.conic_opacity,
                   color_ptr, imgState.accum_alpha, imgState.n_contrib,
                   imgState.max_contrib, imgState.pixel_colors, dL_dpix,
                   (float3 *)dL_dmean2D, (float4 *)dL_dconic, dL_dopacity,
                   dL_dcolor, stream),
               debug)
  } else {

    OrderImg rangesState =
        OrderImg::fromChunk(ranges, tile_grid.x * tile_grid.y);
    OrderBin gs_listState = OrderBin::fromChunk(gs_list, R);
    OrderImg tilesState = OrderImg::fromChunk(tiles_id, num_tiles);
    TransState pixelState =
        TransState::fromChunk(pixel_tr, num_tiles * BLOCK_SIZE);

    CHECK_CUDA(BACKWARD::renderPre(
                   tile_grid, block, rangesState.ranges,
                   gs_listState.gaussian_list, width, height, R, B,
                   imgState.bucket_offsets, sampleState.bucket_to_tile,
                   sampleState.T, sampleState.ar, background, geomState.means2D,
                   geomState.conic_opacity, color_ptr, imgState.accum_alpha,
                   imgState.n_contrib, imgState.max_contrib,
                   imgState.pixel_colors, dL_dpix, (float3 *)dL_dmean2D,
                   (float4 *)dL_dconic, dL_dopacity, dL_dcolor,
                   tilesState.ranges, num_tiles, pixelState.T, stream,
                   use_load_balancing, lb_last_set, lb_last_num),
               debug)
  }

  const float *cov3D_ptr =
      (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
  CHECK_CUDA(BACKWARD::preprocess(
                 P, D, M, (float3 *)means3D, radii, dc, shs, geomState.clamped,
                 opacities, (glm::vec3 *)scales, (glm::vec4 *)rotations,
                 scale_modifier, cov3D_ptr, viewmatrix, projmatrix, focal_x,
                 focal_y, tan_fovx, tan_fovy, (glm::vec3 *)campos,
                 (float3 *)dL_dmean2D, dL_dconic, dL_dopacity,
                 (glm::vec3 *)dL_dmean3D, dL_dcolor, dL_dcov3D, dL_ddc, dL_dsh,
                 (glm::vec3 *)dL_dscale, (glm::vec4 *)dL_drot, precomp, stream),
             debug)
}