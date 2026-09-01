#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "config.cuh"

// Full mask for warp of 32 threads.
static constexpr unsigned kFullMask = 0xffffffff;

// Warp-level reduction: sum across all 32 lanes using butterfly XOR.
__device__ inline float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(kFullMask, val, 16);
    val += __shfl_xor_sync(kFullMask, val, 8);
    val += __shfl_xor_sync(kFullMask, val, 4);
    val += __shfl_xor_sync(kFullMask, val, 2);
    val += __shfl_xor_sync(kFullMask, val, 1);
    return val;
}

// Block-level reduction using shared memory across warps.
// smem_scratch must have at least (blockDim.x / 32) floats available.
__device__ inline float block_reduce_sum(float val, float* smem_scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) smem_scratch[warp] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? smem_scratch[threadIdx.x] : 0.0f;
    val = warp_reduce_sum(val);
    // Broadcast result from lane 0 to all lanes.
    val = __shfl_sync(kFullMask, val, 0);
    return val;
}

// In-place RMSNorm on a d-dimensional fp16 vector residing in shared memory.
//
// Each thread handles ceil(d / blockDim.x) elements.
// smem_scratch: temporary float buffer of size >= (blockDim.x / 32).
// weight: fp32 learned scale vector from global memory, length d.
// eps: small constant for numerical stability.
//
// After this call vec[0..d-1] = vec[i] / rms * weight[i].
__device__ void smem_rms_norm(__half* vec,
                               int d,
                               float* smem_scratch,
                               const float* weight,
                               float eps = 1e-5f) {
    // Accumulate sum of squares.
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = __half2float(vec[i]);
        local_sq += v * v;
    }

    float mean_sq = block_reduce_sum(local_sq, smem_scratch) / (float)d;
    float rms_inv = rsqrtf(mean_sq + eps);
    __syncthreads();

    // Scale in-place.
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = __half2float(vec[i]) * rms_inv * weight[i];
        vec[i] = __float2half(v);
    }
    __syncthreads();
}
