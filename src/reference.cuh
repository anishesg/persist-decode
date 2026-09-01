#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "config.cuh"
#include "decode_attn.cuh"

// Multi-launch reference implementation: separate kernel per operation.
// Used as a correctness oracle only. Does NOT optimize for performance.

// Applies RMSNorm: out[i] = in[i] / rms(in) * weight[i]
__global__ void k_rms_norm(const __half* in, __half* out, const float* weight,
                            int dim, float eps = 1e-5f);

// Linear projection: out[rows] = W[rows, cols] * in[cols]
// Naive single-block, all threads handle one output element.
__global__ void k_linear(const __half* W, const __half* in, __half* out,
                          int rows, int cols);

// Elementwise residual add: out[i] += residual[i]
__global__ void k_residual_add(const __half* residual, __half* out, int dim);

// SiLU activation: out[i] = silu(gate[i]) * up[i]
__global__ void k_silu_mul(const __half* gate, const __half* up, __half* out, int dim);

// Single-query attention kernel. Reads Q/K/V from global memory, returns attention output.
// Q: [num_heads, head_dim], K/V cache: [kv_seq_len, num_kv_heads, head_dim]
// out: [num_heads, head_dim]
__global__ void k_single_query_attn(const __half* Q,
                                     const __half* K, const __half* V,
                                     __half* out,
                                     int num_heads, int num_kv_heads, int head_dim,
                                     int kv_seq_len);

// Host function: runs a full transformer layer using separate kernel launches.
// Allocates internal temporary global-memory buffers on the fly.
void launch_reference_layer(const __half* hidden_in,
                             __half* hidden_out,
                             const LayerWeights& weights,
                             const ModelConfig& cfg,
                             KVCache& kvc,
                             cudaStream_t stream = nullptr);
