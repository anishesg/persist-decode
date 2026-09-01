#include "persistent_layer.cuh"
#include "smem_ops.cuh"
#include "ffn.cuh"
#include <stdexcept>
#include <string>

// Tile width used by tiled_matvec within the persistent kernel.
static constexpr int kTileK = 64;

__global__ void k_persistent_layer(const __half* hidden_in,
                                    __half* hidden_out,
                                    LayerWeights weights,
                                    ModelConfig cfg,
                                    KVCache kvc) {
    // Parse dynamic shared memory into sub-regions.
    extern __shared__ char smem_raw[];

    int  bsz       = blockDim.x;
    int  num_warps = (bsz + 31) / 32;
    int  q_dim     = cfg.num_heads * cfg.head_dim;
    int  inter     = cfg.intermediate_dim;

    size_t off = 0;
    auto alloc_smem = [&](size_t bytes) -> char* {
        char* p = smem_raw + off;
        off += (bytes + 15) & ~(size_t)15; // 16-byte align each region
        return p;
    };

    __half* hidden_smem = reinterpret_cast<__half*>(alloc_smem(cfg.hidden_dim * sizeof(__half)));
    __half* proj_buf    = reinterpret_cast<__half*>(alloc_smem(max(q_dim, cfg.hidden_dim) * sizeof(__half)));
    __half* inter_a     = reinterpret_cast<__half*>(alloc_smem(inter * sizeof(__half)));
    __half* inter_b     = reinterpret_cast<__half*>(alloc_smem(inter * sizeof(__half)));
    __half* tile_a      = reinterpret_cast<__half*>(alloc_smem((size_t)kTileK * bsz * sizeof(__half)));
    __half* tile_b      = reinterpret_cast<__half*>(alloc_smem((size_t)kTileK * bsz * sizeof(__half)));
    float*  warp_scr    = reinterpret_cast<float*>(alloc_smem(num_warps * sizeof(float)));

    // --- Load hidden state into smem ---
    for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz)
        hidden_smem[i] = hidden_in[i];
    __syncthreads();

    // --- Save residual copy before attention norm ---
    // Keep a second smem copy for residual: reuse proj_buf momentarily (q_dim >= hidden_dim is not guaranteed,
    // so we store the residual in hidden_smem temporarily by computing in-place after adding back).
    // Simpler: allocate residual inline in proj_buf only when q_dim >= hidden_dim.
    // For Llama-7B, q_dim = hidden_dim = 4096, so this is fine.
    __half* residual = proj_buf; // hidden_dim halves; will be overwritten after residual add

    for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz)
        residual[i] = hidden_smem[i];
    __syncthreads();

    // --- Attention norm ---
    smem_rms_norm(hidden_smem, cfg.hidden_dim, warp_scr, weights.attn_norm_weight);

    // --- Decode attention (Q/K/V proj, online softmax, O proj) ---
    // decode_attn_smem overwrites hidden_smem with o_proj output.
    decode_attn_smem<kTileK>(hidden_smem, weights, cfg, kvc,
                               warp_scr, proj_buf, tile_a, tile_b);

    // --- Residual add (attn) ---
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
    smem_rms_norm(hidden_smem, cfg.hidden_dim, warp_scr, weights.ffn_norm_weight);

    // --- SwiGLU FFN ---
    swiglu_ffn_smem<kTileK>(hidden_smem, weights, cfg, inter_a, inter_b, tile_a, tile_b);

    // --- Residual add (FFN) ---
    for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz) {
        float r = __half2float(residual[i]) + __half2float(hidden_smem[i]);
        hidden_smem[i] = __float2half(r);
    }
    __syncthreads();

    // --- Store result to global memory ---
    for (int i = threadIdx.x; i < cfg.hidden_dim; i += bsz)
        hidden_out[i] = hidden_smem[i];
}

size_t persistent_layer_smem_bytes(const ModelConfig& cfg, int block_size, int tile_k) {
    int q_dim     = cfg.num_heads * cfg.head_dim;
    int num_warps = (block_size + 31) / 32;

    auto align16 = [](size_t s) { return (s + 15) & ~(size_t)15; };

    size_t total = 0;
    total += align16(cfg.hidden_dim * sizeof(__half));           // hidden_smem
    total += align16(max(q_dim, cfg.hidden_dim) * sizeof(__half)); // proj_buf / residual
    total += align16(cfg.intermediate_dim * sizeof(__half));    // inter_a
    total += align16(cfg.intermediate_dim * sizeof(__half));    // inter_b
    total += align16((size_t)tile_k * block_size * sizeof(__half)); // tile_a
    total += align16((size_t)tile_k * block_size * sizeof(__half)); // tile_b
    total += align16(num_warps * sizeof(float));                 // warp scratch
    return total;
}

void launch_persistent_layer(const __half* hidden_in,
                              __half* hidden_out,
                              const LayerWeights& weights,
                              const ModelConfig& cfg,
                              KVCache& kvc,
                              cudaStream_t stream,
                              int block_size,
                              int tile_k) {
    size_t smem = persistent_layer_smem_bytes(cfg, block_size, tile_k);

    // Request maximum shared memory if needed.
    cudaError_t err = cudaFuncSetAttribute(
        k_persistent_layer,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem
    );
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaFuncSetAttribute failed: ") + cudaGetErrorString(err));
    }

    k_persistent_layer<<<1, block_size, smem, stream>>>(
        hidden_in, hidden_out, weights, cfg, kvc
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("k_persistent_layer launch failed: ") + cudaGetErrorString(err));
    }

    // Increment host-side kv_seq_len after kernel appended new K/V.
    kvc.kv_seq_len += 1;
}
