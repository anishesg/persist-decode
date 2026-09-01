#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "config.cuh"
#include "decode_attn.cuh"

// Persistent single-layer kernel: one thread block, one kernel launch.
//
// hidden_in:  [hidden_dim] fp16 input from global memory.
// hidden_out: [hidden_dim] fp16 output to global memory.
// weights:    LayerWeights pointing into pre-loaded device weight buffer.
// cfg:        model dimensions.
// kvc:        mutable KV cache for this layer (kv_seq_len updated on host after return).
//
// Dynamic shared memory layout (in bytes, from smem base):
//   [0, hidden_dim*2)           : working hidden state (__half)
//   [hidden_dim*2, hidden_dim*2 + q_dim*2) : projection scratch (__half)
//   [above, + inter_a)          : FFN gate intermediate (__half)
//   [above, + inter_b)          : FFN up intermediate   (__half)
//   [above, + TILE_K*bsz*2*2)   : tile_a + tile_b for tiled_matvec (__half)
//   [above, + num_warps*4)       : warp reduction scratch (float)
//
// The kernel computes required smem size; host must pass the right dynSmem value.
__global__ void k_persistent_layer(const __half* hidden_in,
                                    __half* hidden_out,
                                    LayerWeights weights,
                                    ModelConfig cfg,
                                    KVCache kvc);

// Returns the dynamic shared memory bytes required for one call to k_persistent_layer.
size_t persistent_layer_smem_bytes(const ModelConfig& cfg, int block_size, int tile_k = 64);

// Host wrapper: launches k_persistent_layer with appropriate smem configuration.
// Updates kvc.kv_seq_len on the host after the kernel completes.
void launch_persistent_layer(const __half* hidden_in,
                              __half* hidden_out,
                              const LayerWeights& weights,
                              const ModelConfig& cfg,
                              KVCache& kvc,
                              cudaStream_t stream = nullptr,
                              int block_size = 256,
                              int tile_k = 64);
