#include <cuda_runtime_api.h>
#include <iostream>

__global__ void gather_kernel(int N, int M, int L, int* inds, float* src, float* dst)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= N)
        return;

    int base = idx * M;
    bool seen_zero = false;
    
    for(int i = 0; i < M && !seen_zero; i++)
    {
        
        int loc = base + i; 
        int ind = inds[loc];
        
        if(ind == 0)
            seen_zero = true;

        for(int j = 0; j < L ; j++)
        {
            dst[loc * L + j] = src[ind * L + j];
        }
    }
}

void bk_gather_cuda(int N, int M, int L, int* inds, float* src, float* dst)
{
    gather_kernel<<<(N + 127) / 128, 128>>>(N, M, L, inds, src, dst);
}