#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "config.cuh"
#include "decode_attn.cuh"

// Array of per-layer weight pointers passed to the multi-layer kernel.
// All layers must share the same ModelConfig dimensions.
struct LayerWeightsArray {
    const LayerWeights* layers;  // host-accessible array of per-layer weight structs
    int num_layers;
};

// Multi-layer persistent kernel: processes num_layers transformer layers in a
// single kernel launch.
//
// hidden_in:  [hidden_dim] fp16 input (global memory).
// hidden_out: [hidden_dim] fp16 output (global memory).
// lwa:        array of per-layer weight structs.
// cfg:        model dimensions (cfg.num_layers must equal lwa.num_layers).
// kvcs:       array of KVCache structs, one per layer (device pointers inside).
//
// The hidden state is loaded from global memory once at kernel entry and kept
// in shared memory for the entire duration. Only the weight tiles are streamed
// from global memory per layer via double-buffered tiled matvec.
__global__ void k_persistent_multi_layer(const __half* hidden_in,
                                          __half* hidden_out,
                                          LayerWeightsArray lwa,
                                          ModelConfig cfg,
                                          KVCache* kvcs);

// Returns the dynamic shared memory bytes required for k_persistent_multi_layer.
size_t persistent_multi_layer_smem_bytes(const ModelConfig& cfg,
                                          int block_size,
                                          int tile_k = 64);

// Host wrapper: validates shared memory budget, sets max dynamic smem,
// launches k_persistent_multi_layer. Updates kvc.kv_seq_len for each layer.
void launch_persistent_multi_layer(const __half* hidden_in,
                                    __half* hidden_out,
                                    const LayerWeightsArray& lwa,
                                    const ModelConfig& cfg,
                                    KVCache* kvcs,
                                    cudaStream_t stream = nullptr,
                                    int block_size = 256,
                                    int tile_k = 64);
