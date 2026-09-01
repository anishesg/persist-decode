#include "reference.cuh"
#include <stdexcept>
#include <string>
#include <cstdlib>

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(_e)); \
} while(0)

// ---------------------------------------------------------------------------
// RMSNorm kernel
// ---------------------------------------------------------------------------
__global__ void k_rms_norm(const __half* in, __half* out, const float* weight,
                            int dim, float eps) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    float local_sq = 0.0f;
    for (int i = tid; i < dim; i += bsz) {
        float v = __half2float(in[i]);
        local_sq += v * v;
    }

    // Warp reduce.
    for (int mask = 16; mask > 0; mask >>= 1)
        local_sq += __shfl_xor_sync(0xffffffff, local_sq, mask);

    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) smem[warp] = local_sq;
    __syncthreads();

    float total = 0.0f;
    int num_warps = (bsz + 31) >> 5;
    if (tid < num_warps) total = smem[tid];
    for (int mask = 16; mask > 0; mask >>= 1)
        total += __shfl_xor_sync(0xffffffff, total, mask);

    float rms_inv = rsqrtf(total / (float)dim + eps);
    for (int i = tid; i < dim; i += bsz)
        out[i] = __float2half(__half2float(in[i]) * rms_inv * weight[i]);
}

// ---------------------------------------------------------------------------
// Linear projection kernel (naive, correctness only)
// ---------------------------------------------------------------------------
__global__ void k_linear(const __half* W, const __half* in, __half* out,
                          int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c)
        sum += __half2float(W[(size_t)row * cols + c]) * __half2float(in[c]);
    out[row] = __float2half(sum);
}

// ---------------------------------------------------------------------------
// Residual add
// ---------------------------------------------------------------------------
__global__ void k_residual_add(const __half* residual, __half* out, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    out[i] = __float2half(__half2float(out[i]) + __half2float(residual[i]));
}

// ---------------------------------------------------------------------------
// SiLU * up elementwise
// ---------------------------------------------------------------------------
__global__ void k_silu_mul(const __half* gate, const __half* up, __half* out, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float g = __half2float(gate[i]);
    float silu_g = g / (1.0f + expf(-g));
    out[i] = __float2half(silu_g * __half2float(up[i]));
}

// ---------------------------------------------------------------------------
// Single-query attention: one warp per head
// ---------------------------------------------------------------------------
__global__ void k_single_query_attn(const __half* Q,
                                     const __half* K, const __half* V,
                                     __half* out,
                                     int num_heads, int num_kv_heads, int head_dim,
                                     int kv_seq_len) {
    int h = blockIdx.x; // one block per head
    if (h >= num_heads) return;

    int kv_h = h / (num_heads / num_kv_heads);
    int tid  = threadIdx.x;
    int bsz  = blockDim.x;

    extern __shared__ float attn_smem[];
    float* scores = attn_smem;                    // [kv_seq_len]
    float* v_acc  = attn_smem + kv_seq_len;       // [head_dim]
    float* warp_s = v_acc + head_dim;             // [(bsz/32)] scratch

    float scale = rsqrtf((float)head_dim);

    // Compute attention scores.
    for (int pos = tid; pos < kv_seq_len; pos += bsz) {
        const __half* q = Q + h * head_dim;
        const __half* k = K + ((size_t)pos * num_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(q[d]) * __half2float(k[d]);
        scores[pos] = dot * scale;
    }
    __syncthreads();

    // Softmax.
    float local_max = -1e38f;
    for (int pos = tid; pos < kv_seq_len; pos += bsz)
        local_max = fmaxf(local_max, scores[pos]);
    for (int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));

    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) warp_s[warp] = local_max;
    __syncthreads();
    float gmax = (tid == 0) ? warp_s[0] : 0.0f;
    for (int w = 0; w < (bsz + 31) / 32; ++w)
        gmax = fmaxf(gmax, warp_s[w]);
    gmax = __shfl_sync(0xffffffff, gmax, 0);

    float local_sum = 0.0f;
    for (int pos = tid; pos < kv_seq_len; pos += bsz) {
        scores[pos] = expf(scores[pos] - gmax);
        local_sum += scores[pos];
    }
    __syncthreads();
    for (int mask = 16; mask > 0; mask >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
    if (lane == 0) warp_s[warp] = local_sum;
    __syncthreads();
    float gsum = 0.0f;
    for (int w = 0; w < (bsz + 31) / 32; ++w)
        gsum += warp_s[w];
    gsum = __shfl_sync(0xffffffff, gsum, 0);

    float inv_sum = (gsum > 0.0f) ? 1.0f / gsum : 0.0f;
    for (int pos = tid; pos < kv_seq_len; pos += bsz)
        scores[pos] *= inv_sum;
    __syncthreads();

    // Weighted sum of V.
    for (int d = tid; d < head_dim; d += bsz) v_acc[d] = 0.0f;
    __syncthreads();

    for (int pos = 0; pos < kv_seq_len; ++pos) {
        float a = scores[pos];
        const __half* v = V + ((size_t)pos * num_kv_heads + kv_h) * head_dim;
        for (int d = tid; d < head_dim; d += bsz)
            v_acc[d] += a * __half2float(v[d]);
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += bsz)
        out[h * head_dim + d] = __float2half(v_acc[d]);
}

// ---------------------------------------------------------------------------
// Host orchestration: multi-launch reference layer
// ---------------------------------------------------------------------------
void launch_reference_layer(const __half* hidden_in,
                             __half* hidden_out,
                             const LayerWeights& weights,
                             const ModelConfig& cfg,
                             KVCache& kvc,
                             cudaStream_t stream) {
    int d     = cfg.hidden_dim;
    int inter = cfg.intermediate_dim;
    int q_dim = cfg.num_heads    * cfg.head_dim;
    int k_dim = cfg.num_kv_heads * cfg.head_dim;

    // Temporary global memory buffers.
    __half *norm_out, *q_buf, *k_buf, *v_buf, *attn_out, *gate_buf, *up_buf, *residual_buf;
    CUDA_CHECK(cudaMalloc(&norm_out,    d     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&q_buf,       q_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&k_buf,       k_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&v_buf,       k_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&attn_out,    q_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&gate_buf,    inter * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&up_buf,      inter * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&residual_buf, d    * sizeof(__half)));

    // Copy input to hidden_out (residual stream).
    CUDA_CHECK(cudaMemcpyAsync(hidden_out, hidden_in, d * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(residual_buf, hidden_in, d * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream));

    int block = 256;
    int num_warps = (block + 31) / 32;
    size_t norm_smem = num_warps * sizeof(float);

    // --- Attention norm ---
    k_rms_norm<<<1, block, norm_smem, stream>>>(hidden_in, norm_out, weights.attn_norm_weight, d);

    // --- Q, K, V projections ---
    k_linear<<<(q_dim + block - 1) / block, block, 0, stream>>>(weights.q_proj, norm_out, q_buf, q_dim, d);
    k_linear<<<(k_dim + block - 1) / block, block, 0, stream>>>(weights.k_proj, norm_out, k_buf, k_dim, d);
    k_linear<<<(k_dim + block - 1) / block, block, 0, stream>>>(weights.v_proj, norm_out, v_buf, k_dim, d);

    // Append K, V to cache.
    int kv_seq = kvc.kv_seq_len;
    CUDA_CHECK(cudaMemcpyAsync(kvc.k + (size_t)kv_seq * cfg.num_kv_heads * cfg.head_dim,
                                k_buf, k_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(kvc.v + (size_t)kv_seq * cfg.num_kv_heads * cfg.head_dim,
                                v_buf, k_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    int new_kv_len = kv_seq + 1;

    // --- Single-query attention ---
    size_t attn_smem = (size_t)(new_kv_len + cfg.head_dim + num_warps) * sizeof(float);
    k_single_query_attn<<<cfg.num_heads, block, attn_smem, stream>>>(
        q_buf, kvc.k, kvc.v, attn_out,
        cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, new_kv_len);

    // --- O projection ---
    k_linear<<<(d + block - 1) / block, block, 0, stream>>>(weights.o_proj, attn_out, norm_out, d, q_dim);

    // --- Residual add (attn) ---
    k_residual_add<<<(d + block - 1) / block, block, 0, stream>>>(residual_buf, norm_out, d);

    // --- Save post-attn residual ---
    CUDA_CHECK(cudaMemcpyAsync(residual_buf, norm_out, d * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream));

    // --- FFN norm ---
    k_rms_norm<<<1, block, norm_smem, stream>>>(norm_out, norm_out, weights.ffn_norm_weight, d);

    // --- Gate and up projections ---
    k_linear<<<(inter + block - 1) / block, block, 0, stream>>>(weights.gate_proj, norm_out, gate_buf, inter, d);
    k_linear<<<(inter + block - 1) / block, block, 0, stream>>>(weights.up_proj,   norm_out, up_buf,   inter, d);

    // --- SiLU * up ---
    k_silu_mul<<<(inter + block - 1) / block, block, 0, stream>>>(gate_buf, up_buf, gate_buf, inter);

    // --- Down projection ---
    k_linear<<<(d + block - 1) / block, block, 0, stream>>>(weights.down_proj, gate_buf, hidden_out, d, inter);

    // --- Residual add (FFN) ---
    k_residual_add<<<(d + block - 1) / block, block, 0, stream>>>(residual_buf, hidden_out, d);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    kvc.kv_seq_len = new_kv_len;

    cudaFree(norm_out);
    cudaFree(q_buf);
    cudaFree(k_buf);
    cudaFree(v_buf);
    cudaFree(attn_out);
    cudaFree(gate_buf);
    cudaFree(up_buf);
    cudaFree(residual_buf);
}
