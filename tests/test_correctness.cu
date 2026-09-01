#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <random>

#include "config.cuh"
#include "persistent_layer.cuh"
#include "reference.cuh"

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// Allocate and fill a device half buffer with random fp16 values in [-scale, scale].
static __half* make_device_half(int n, float scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> h(n);
    for (auto& v : h) v = dist(rng);

    // Convert to half on host.
    std::vector<__half> hh(n);
    for (int i = 0; i < n; ++i) hh[i] = __float2half(h[i]);

    __half* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d, hh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    return d;
}

// Allocate and fill a device float buffer (for norm weights) with values near 1.
static float* make_device_float(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.8f, 1.2f);
    std::vector<float> h(n);
    for (auto& v : h) v = dist(rng);
    float* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// Compute cosine similarity and max absolute error between two device half vectors.
static void compare_outputs(const __half* a, const __half* b, int n,
                             double& cos_sim, double& max_err) {
    std::vector<__half> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(__half), cudaMemcpyDeviceToHost));

    double dot = 0, na2 = 0, nb2 = 0, me = 0;
    for (int i = 0; i < n; ++i) {
        double va = __half2float(ha[i]);
        double vb = __half2float(hb[i]);
        dot += va * vb;
        na2 += va * va;
        nb2 += vb * vb;
        me = std::max(me, std::abs(va - vb));
    }
    cos_sim = dot / (std::sqrt(na2) * std::sqrt(nb2) + 1e-12);
    max_err = me;
}

struct TestCase {
    int hidden_dim;
    int intermediate_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int kv_len;
    const char* name;
};

static bool run_test(const TestCase& tc, std::mt19937& rng) {
    printf("  [%s] hidden=%d inter=%d heads=%d kv_heads=%d head_dim=%d kv_len=%d\n",
           tc.name, tc.hidden_dim, tc.intermediate_dim,
           tc.num_heads, tc.num_kv_heads, tc.head_dim, tc.kv_len);

    ModelConfig cfg;
    cfg.num_layers      = 1;
    cfg.hidden_dim      = tc.hidden_dim;
    cfg.intermediate_dim = tc.intermediate_dim;
    cfg.num_heads       = tc.num_heads;
    cfg.num_kv_heads    = tc.num_kv_heads;
    cfg.head_dim        = tc.head_dim;

    // Allocate weight buffer.
    size_t offsets[9];
    size_t total_w = layer_buffer_layout(cfg, offsets);
    char* weight_buf_h = (char*)malloc(total_w);
    if (!weight_buf_h) { fprintf(stderr, "OOM\n"); return false; }

    // Fill each matrix with random fp16.
    std::uniform_real_distribution<float> wdist(-0.02f, 0.02f);
    for (size_t i = 0; i < total_w; ++i) weight_buf_h[i] = 0;

    // Fill half-valued matrices.
    auto fill_half = [&](size_t off, int rows, int cols) {
        __half* p = reinterpret_cast<__half*>(weight_buf_h + off);
        for (int i = 0; i < rows * cols; ++i)
            p[i] = __float2half(wdist(rng));
    };
    auto fill_float = [&](size_t off, int n) {
        float* p = reinterpret_cast<float*>(weight_buf_h + off);
        std::uniform_real_distribution<float> nd(0.8f, 1.2f);
        for (int i = 0; i < n; ++i) p[i] = nd(rng);
    };

    int q_cols  = cfg.num_heads    * cfg.head_dim;
    int kv_cols = cfg.num_kv_heads * cfg.head_dim;
    fill_half(offsets[0], cfg.hidden_dim, q_cols);
    fill_half(offsets[1], cfg.hidden_dim, kv_cols);
    fill_half(offsets[2], cfg.hidden_dim, kv_cols);
    fill_half(offsets[3], q_cols, cfg.hidden_dim);
    fill_half(offsets[4], cfg.hidden_dim, cfg.intermediate_dim);
    fill_half(offsets[5], cfg.hidden_dim, cfg.intermediate_dim);
    fill_half(offsets[6], cfg.intermediate_dim, cfg.hidden_dim);
    fill_float(offsets[7], cfg.hidden_dim);
    fill_float(offsets[8], cfg.hidden_dim);

    char* weight_dev;
    CUDA_CHECK(cudaMalloc(&weight_dev, total_w));
    CUDA_CHECK(cudaMemcpy(weight_dev, weight_buf_h, total_w, cudaMemcpyHostToDevice));
    free(weight_buf_h);

    LayerWeights lw = make_layer_weights(weight_dev, cfg);

    // KV cache (shared between both runs, reset before each).
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;
    int max_seq = tc.kv_len + 2;
    __half *kv_k_ref, *kv_v_ref, *kv_k_per, *kv_v_per;
    CUDA_CHECK(cudaMalloc(&kv_k_ref, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&kv_v_ref, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&kv_k_per, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&kv_v_per, (size_t)max_seq * kv_dim * sizeof(__half)));

    // Pre-fill KV cache with random data (simulating prior decode steps).
    {
        int total_kv = tc.kv_len * kv_dim;
        std::vector<float> tmp(total_kv);
        std::uniform_real_distribution<float> kv_dist(-0.1f, 0.1f);
        std::vector<__half> kv_h(total_kv);
        for (int i = 0; i < total_kv; ++i) kv_h[i] = __float2half(kv_dist(rng));
        CUDA_CHECK(cudaMemcpy(kv_k_ref, kv_h.data(), total_kv * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(kv_v_ref, kv_h.data(), total_kv * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(kv_k_per, kv_h.data(), total_kv * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(kv_v_per, kv_h.data(), total_kv * sizeof(__half), cudaMemcpyHostToDevice));
    }

    // Hidden state input.
    __half* hidden_in = make_device_half(cfg.hidden_dim, 0.5f, rng);
    __half *out_ref, *out_per;
    CUDA_CHECK(cudaMalloc(&out_ref, cfg.hidden_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&out_per, cfg.hidden_dim * sizeof(__half)));

    // Run reference.
    KVCache kvc_ref{ kv_k_ref, kv_v_ref, max_seq, tc.kv_len };
    try {
        launch_reference_layer(hidden_in, out_ref, lw, cfg, kvc_ref);
    } catch (const std::exception& e) {
        fprintf(stderr, "    reference failed: %s\n", e.what());
        return false;
    }

    // Run persistent.
    KVCache kvc_per{ kv_k_per, kv_v_per, max_seq, tc.kv_len };
    try {
        launch_persistent_layer(hidden_in, out_per, lw, cfg, kvc_per);
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        fprintf(stderr, "    persistent failed: %s\n", e.what());
        return false;
    }

    double cos_sim, max_err;
    compare_outputs(out_ref, out_per, cfg.hidden_dim, cos_sim, max_err);

    bool pass = (cos_sim > 0.998);
    printf("    cosine_similarity=%.6f  max_abs_error=%.6f  %s\n",
           cos_sim, max_err, pass ? "PASS" : "FAIL");

    cudaFree(weight_dev);
    cudaFree(kv_k_ref); cudaFree(kv_v_ref);
    cudaFree(kv_k_per); cudaFree(kv_v_per);
    cudaFree(hidden_in);
    cudaFree(out_ref); cudaFree(out_per);

    return pass;
}

int main() {
    printf("=== persist-decode correctness test ===\n\n");

    std::mt19937 rng(42);

    TestCase cases[] = {
        { 2048, 5504,  16, 16, 128, 512,  "llama-small" },
        { 4096, 11008, 32, 32, 128, 2048, "llama-7b"    },
    };

    int pass = 0, total = 0;
    for (const auto& tc : cases) {
        ++total;
        if (run_test(tc, rng)) ++pass;
    }

    printf("\n%d / %d tests passed\n", pass, total);
    return (pass == total) ? 0 : 1;
}
