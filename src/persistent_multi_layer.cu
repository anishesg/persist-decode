#include "persistent_multi_layer.cuh"
#include "smem_ops.cuh"
#include "ffn.cuh"
#include <stdexcept>
#include <string>

static constexpr int kTileKML = 64;

__global__ void k_persistent_multi_layer(const __half* hidden_in,
                                          __half* hidden_out,
                                          LayerWeightsArray lwa,
                                          ModelConfig cfg,
                                          KVCache* kvcs) {
    extern __shared__ char smem_raw[];

    int bsz       = blockDim.x;
    int num_warps = (bsz + 31) / 32;
    int q_dim     = cfg.num_heads * cfg.head_dim;
    int inter     = cfg.intermediate_dim;

    // Parse shared memory layout (identical to single-layer kernel).
    size_t off = 0;
    auto alloc_smem = [&](size_t bytes) -> char* {
        char* p = smem_raw + off;
        off += (bytes + 15) & ~(size_t)15;
        return p;
    };

    __half* hidden_smem = reinterpret_cast<__half*>(alloc_smem(cfg.hidden_dim * sizeof(__half)));
    __half* proj_buf    = reinterpret_cast<__half*>(alloc_smem(max(q_dim, cfg.hidden_dim) * sizeof(__half)));
    __half* inter_a     = reinterpret_cast<__half*>(alloc_smem(inter * sizeof(__half)));
    __half* inter_b     = reinterpret_cast<__half*>(alloc_smem(inter * sizeof(__half)));
    __half* tile_a      = reinterpret_cast<__half*>(alloc_smem((size_t)kTileKML * bsz * sizeof(__half)));
    __half* tile_b      = reinterpret_cast<__half*>(alloc_smem((size_t)kTileKML * bsz * sizeof(__half)));
    float*  warp_scr    = reinterpret_cast<float*>(alloc_smem(num_warps * sizeof(float)));

    // Load hidden state from global memory once. It will remain in smem for
    // all layers; only weights are re-streamed per layer.
    for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz)
        hidden_smem[i] = hidden_in[i];
    __syncthreads();

    // Outer loop over layers. Each iteration:
    //   1. Streams in the layer's weights via tiled matvec (double-buffered).
    //   2. Runs attn_norm, attention, residual, ffn_norm, FFN, residual entirely
    //      in shared memory.
    //   3. The hidden state stays in hidden_smem; only kv-cache pointers
    //      (already in global memory) change per layer.
    for (int l = 0; l < cfg.num_layers; ++l) {
        const LayerWeights& w = lwa.layers[l];
        KVCache& kvc          = kvcs[l];

        // --- Save residual before attention norm ---
        __half* residual = proj_buf;
        for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz)
            residual[i] = hidden_smem[i];
        __syncthreads();

        // --- Attention norm (in-place in hidden_smem) ---
        smem_rms_norm(hidden_smem, cfg.hidden_dim, warp_scr, w.attn_norm_weight);

        // --- Decode attention (Q/K/V proj + online softmax + O proj) ---
        decode_attn_smem<kTileKML>(hidden_smem, w, cfg, kvc,
                                    warp_scr, proj_buf, tile_a, tile_b);

        // --- Residual add (attention) ---
        for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz) {
            float r = __half2float(residual[i]) + __half2float(hidden_smem[i]);
            hidden_smem[i] = __float2half(r);
        }
        __syncthreads();

        // --- Save residual before FFN norm ---
        for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz)
            residual[i] = hidden_smem[i];
        __syncthreads();

        // --- FFN norm ---
        smem_rms_norm(hidden_smem, cfg.hidden_dim, warp_scr, w.ffn_norm_weight);

        // --- SwiGLU FFN ---
        swiglu_ffn_smem<kTileKML>(hidden_smem, w, cfg, inter_a, inter_b, tile_a, tile_b);

        // --- Residual add (FFN) ---
        for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz) {
            float r = __half2float(residual[i]) + __half2float(hidden_smem[i]);
            hidden_smem[i] = __float2half(r);
        }
        __syncthreads();

        // kv_seq_len is incremented on the host after all layers complete.
    }

    // Write final hidden state to global memory.
    for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz)
        hidden_out[i] = hidden_smem[i];
}

size_t persistent_multi_layer_smem_bytes(const ModelConfig& cfg,
                                          int block_size, int tile_k) {
    int q_dim     = cfg.num_heads * cfg.head_dim;
    int num_warps = (block_size + 31) / 32;

    auto align16 = [](size_t s) { return (s + 15) & ~(size_t)15; };

    size_t total = 0;
    total += align16(cfg.hidden_dim * sizeof(__half));
    total += align16(max(q_dim, cfg.hidden_dim) * sizeof(__half));
    total += align16(cfg.intermediate_dim * sizeof(__half));
    total += align16(cfg.intermediate_dim * sizeof(__half));
    total += align16((size_t)tile_k * block_size * sizeof(__half));
    total += align16((size_t)tile_k * block_size * sizeof(__half));
    total += align16(num_warps * sizeof(float));
    return total;
}

void launch_persistent_multi_layer(const __half* hidden_in,
                                    __half* hidden_out,
                                    const LayerWeightsArray& lwa,
                                    const ModelConfig& cfg,
                                    KVCache* kvcs,
                                    cudaStream_t stream,
                                    int block_size,
                                    int tile_k) {
    size_t smem = persistent_multi_layer_smem_bytes(cfg, block_size, tile_k);

    // Query hardware shared memory limit.
    int device;
    cudaGetDevice(&device);
    int smem_max;
    cudaDeviceGetAttribute(&smem_max,
                            cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if ((int)smem > smem_max) {
        throw std::runtime_error(
            "persistent_multi_layer requires " + std::to_string(smem) +
            " bytes of shared memory, but device limit is " +
            std::to_string(smem_max) + " bytes");
    }

    cudaError_t err = cudaFuncSetAttribute(
        k_persistent_multi_layer,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaFuncSetAttribute failed: ") + cudaGetErrorString(err));
    }

    k_persistent_multi_layer<<<1, block_size, smem, stream>>>(
        hidden_in, hidden_out, lwa, cfg, kvcs);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("k_persistent_multi_layer launch failed: ") +
            cudaGetErrorString(err));
    }

    // Increment kv_seq_len for every layer.
    cudaStreamSynchronize(stream == nullptr ? 0 : stream);
    for (int l = 0; l < cfg.num_layers; ++l)
        kvcs[l].kv_seq_len += 1;
}
