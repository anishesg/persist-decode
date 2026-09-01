#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "config.cuh"
#include "tiled_matvec.cuh"

// SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// SwiGLU FFN operating on smem-resident hidden state.
//
// Computes: down_proj( silu(gate_proj(x)) * up_proj(x) )
//
// hidden_smem[0..hidden_dim-1]: input in smem, overwritten with FFN output.
// inter_buf_a: smem buffer for gate_proj output, size >= intermediate_dim halves.
// inter_buf_b: smem buffer for up_proj output,  size >= intermediate_dim halves.
// tile_a, tile_b: ping-pong tile buffers for tiled_matvec, each TILE_K * blockDim.x halves.
//
// Sequence:
//   1. gate_proj matvec: hidden -> inter_buf_a (gate output, intermediate_dim)
//   2. up_proj matvec:   hidden -> inter_buf_b (up output, intermediate_dim)
//   3. silu(gate) * up elementwise in-place in inter_buf_a
//   4. down_proj matvec: inter_buf_a -> hidden_smem (back to hidden_dim)
template <int TILE_K = 64>
__device__ void swiglu_ffn_smem(__half* hidden_smem,
                                 const LayerWeights& w,
                                 const ModelConfig& cfg,
                                 __half* inter_buf_a,
                                 __half* inter_buf_b,
                                 __half* tile_a,
                                 __half* tile_b) {
    int tid = threadIdx.x;

    // gate_proj: [intermediate_dim, hidden_dim] * hidden -> inter_buf_a
    tiled_matvec<TILE_K>(w.gate_proj, hidden_smem, inter_buf_a,
                          cfg.intermediate_dim, cfg.hidden_dim, tile_a, tile_b);

    // up_proj: [intermediate_dim, hidden_dim] * hidden -> inter_buf_b
    tiled_matvec<TILE_K>(w.up_proj, hidden_smem, inter_buf_b,
                          cfg.intermediate_dim, cfg.hidden_dim, tile_a, tile_b);

    // Fused SiLU(gate) * up, written back to inter_buf_a.
    for (int i = tid; i < cfg.intermediate_dim; i += blockDim.x) {
        float g = silu(__half2float(inter_buf_a[i]));
        float u = __half2float(inter_buf_b[i]);
        inter_buf_a[i] = __float2half(g * u);
    }
    __syncthreads();

    // down_proj: [hidden_dim, intermediate_dim] * inter_buf_a -> hidden_smem
    tiled_matvec<TILE_K>(w.down_proj, inter_buf_a, hidden_smem,
                          cfg.hidden_dim, cfg.intermediate_dim, tile_a, tile_b);
}
