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

#include <torch/extension.h>
#include <chrono>

torch::Tensor drive_backward(
    torch::Tensor &xyz_grads,
    torch::Tensor &chains,
    bool cpu)
{
  torch::Tensor rot_xyz_grads;
  if (cpu)
  {
    int64_t *chain_data = chains.data_ptr<int64_t>();
    xyz_grads = xyz_grads.contiguous();
    rot_xyz_grads = xyz_grads.detach().clone();
    float *xyz_grad_data = xyz_grads.data_ptr<float>();
    float *base_grad_data = rot_xyz_grads.data_ptr<float>();

    int N = chains.size(0);
    int L = chains.size(1);
    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < L; j++)
      {
        int64_t id = chain_data[i * L + j];
        if (id == -1)
          break;

        for (int k = 0; k < 3; k++)
          base_grad_data[id * 3 + k] += xyz_grad_data[i * 3 + k];
      }
    }
  }
  else
  {
    throw std::runtime_error("Not supported");
  }

  return rot_xyz_grads;
}

torch::Tensor drive(
    torch::Tensor &rot_xyz,
    torch::Tensor &chains,
    bool cpu)
{
  torch::Tensor xyz;
  if (cpu)
  {
    rot_xyz = rot_xyz.contiguous();
    float *base_data = rot_xyz.data_ptr<float>();
    chains = chains.contiguous();
    int64_t *chain_data = chains.data_ptr<int64_t>();
    xyz = rot_xyz.detach().clone();
    float *xyz_data = xyz.data_ptr<float>();

    int N = chains.size(0);
    int L = chains.size(1);
    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < L; j++)
      {
        int64_t id = chain_data[i * L + j];
        if (id == -1)
          break;

        for (int k = 0; k < 3; k++)
          xyz_data[i * 3 + k] += base_data[id * 3 + k];
      }
    }
  }
  else
  {
    throw std::runtime_error("Not supported");
  }
  return xyz;
}

torch::Tensor bk_gather_backward(int D, torch::Tensor &dst_grads, torch::Tensor &indices)
{
  dst_grads = dst_grads.contiguous();
  indices = indices.contiguous();
  auto b_sizes = indices.sizes();
  int last_dim = dst_grads.sizes().back();

  torch::Tensor grads = torch::zeros({D, last_dim}).contiguous(); 

  int* ind_ptr = indices.data<int>();
  float* dst_ptr = dst_grads.data<float>();
  float* src_ptr = grads.data<float>();

  int N = b_sizes[0];
  int M = b_sizes[1];

  for(int i = 0; i < N; i++)
  {
    bool saw_zero = false;
    for(int j = 0; j < M; j++)
    {
      int64_t base = (i * M + j);
      for(int k = 0; k < last_dim; k++)
      {
        int indir = ind_ptr[base];
        if(indir == 0)
        {
          if(saw_zero)
            break;
          saw_zero = true;
        }

        src_ptr[indir * last_dim + k] += dst_ptr[base * last_dim + k];
      }
    }
  }

  return grads;
}

void bk_gather_cuda(int N, int M, int L, int* inds, float* src, float* dst);

torch::Tensor bk_gather(torch::Tensor &space, torch::Tensor &indices)
{
  space = space.contiguous();
  indices = indices.contiguous();
  auto a_sizes = space.sizes();
  auto b_sizes = indices.sizes();

  int N = b_sizes[0];
  int M = b_sizes[1];
  int last_dim = a_sizes.back();

  torch::Tensor gathered = torch::zeros({N, M, last_dim}, torch::device(torch::kCUDA)).contiguous(); 
  //auto before = std::chrono::system_clock::now();
  int* ind_ptr = indices.data<int>();
  float* src_ptr = space.data<float>();
  float* dst_ptr = gathered.data<float>();

  bk_gather_cuda(N, M, last_dim, ind_ptr, src_ptr, dst_ptr);

  //auto after = std::chrono::system_clock::now();

  //std::cout << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << std::endl;
  // for(int i = 0; i < N; i++)
  // {
  //   bool saw_zero = false;
  //   for(int j = 0; j < M; j++)
  //   {
  //     int64_t base = (i * M + j);
  //     for(int k = 0; k < last_dim; k++)
  //     {
  //       int indir = ind_ptr[base];
  //       if(indir == 0)
  //        {
  //         if(saw_zero)
  //           break;
  //         saw_zero = true;
  //        }

  //       dst_ptr[base * last_dim + k] = src_ptr[indir * last_dim + k];
  //     }
  //   }
  // }

  return gathered;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("drive", &drive);
  m.def("drive_backward", &drive_backward);
  m.def("bk_gather", &bk_gather);
  m.def("bk_gather_backward", &bk_gather_backward);
}