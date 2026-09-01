#pragma once
#include <cstdint>
#include <cuda_fp16.h>

// 128-byte alignment matches cache line and satisfies cp.async requirements
static constexpr int kSmemAlign = 128;

struct ModelConfig {
    int num_layers;
    int hidden_dim;      // d
    int intermediate_dim; // d_ffn (e.g. 11008 for Llama-7B)
    int num_heads;
    int num_kv_heads;    // may be < num_heads for GQA
    int head_dim;        // hidden_dim / num_heads
};

// Pointers into a flat packed weight buffer for one transformer layer.
// Each matrix pointer is 128-byte aligned within the buffer.
struct LayerWeights {
    // Attention projections: shapes (hidden_dim, hidden_dim) except kv which are (hidden_dim, num_kv_heads*head_dim)
    const __half* q_proj;       // [hidden_dim, num_heads * head_dim]
    const __half* k_proj;       // [hidden_dim, num_kv_heads * head_dim]
    const __half* v_proj;       // [hidden_dim, num_kv_heads * head_dim]
    const __half* o_proj;       // [num_heads * head_dim, hidden_dim]

    // FFN projections
    const __half* gate_proj;    // [hidden_dim, intermediate_dim]
    const __half* up_proj;      // [hidden_dim, intermediate_dim]
    const __half* down_proj;    // [intermediate_dim, hidden_dim]

    // Normalization weights (vectors, not matrices)
    const float* attn_norm_weight; // [hidden_dim]
    const float* ffn_norm_weight;  // [hidden_dim]
};

// Align a byte offset up to kSmemAlign.
__host__ __device__ inline size_t align_up(size_t offset) {
    return (offset + kSmemAlign - 1) & ~(size_t)(kSmemAlign - 1);
}

// Compute per-matrix byte offsets within a flat packed weight buffer.
// Returns the total buffer size required.
inline size_t layer_buffer_layout(const ModelConfig& cfg,
                                   size_t offsets[9]) {
    size_t off = 0;
    auto next = [&](size_t bytes) -> size_t {
        size_t o = off;
        off = align_up(off + bytes);
        return o;
    };

    int q_cols = cfg.num_heads * cfg.head_dim;
    int kv_cols = cfg.num_kv_heads * cfg.head_dim;

    offsets[0] = next((size_t)cfg.hidden_dim * q_cols  * sizeof(__half)); // q_proj
    offsets[1] = next((size_t)cfg.hidden_dim * kv_cols * sizeof(__half)); // k_proj
    offsets[2] = next((size_t)cfg.hidden_dim * kv_cols * sizeof(__half)); // v_proj
    offsets[3] = next((size_t)q_cols * cfg.hidden_dim  * sizeof(__half)); // o_proj
    offsets[4] = next((size_t)cfg.hidden_dim * cfg.intermediate_dim * sizeof(__half)); // gate_proj
    offsets[5] = next((size_t)cfg.hidden_dim * cfg.intermediate_dim * sizeof(__half)); // up_proj
    offsets[6] = next((size_t)cfg.intermediate_dim * cfg.hidden_dim * sizeof(__half)); // down_proj
    offsets[7] = next((size_t)cfg.hidden_dim * sizeof(float)); // attn_norm_weight
    offsets[8] = next((size_t)cfg.hidden_dim * sizeof(float)); // ffn_norm_weight

    return off;
}

// Build a LayerWeights view into the flat buffer starting at base.
__host__ inline LayerWeights make_layer_weights(const char* base,
                                                 const ModelConfig& cfg) {
    size_t offsets[9];
    layer_buffer_layout(cfg, offsets);
    LayerWeights w;
    w.q_proj           = reinterpret_cast<const __half*>(base + offsets[0]);
    w.k_proj           = reinterpret_cast<const __half*>(base + offsets[1]);
    w.v_proj           = reinterpret_cast<const __half*>(base + offsets[2]);
    w.o_proj           = reinterpret_cast<const __half*>(base + offsets[3]);
    w.gate_proj        = reinterpret_cast<const __half*>(base + offsets[4]);
    w.up_proj          = reinterpret_cast<const __half*>(base + offsets[5]);
    w.down_proj        = reinterpret_cast<const __half*>(base + offsets[6]);
    w.attn_norm_weight = reinterpret_cast<const float*>(base + offsets[7]);
    w.ffn_norm_weight  = reinterpret_cast<const float*>(base + offsets[8]);
    return w;
}
