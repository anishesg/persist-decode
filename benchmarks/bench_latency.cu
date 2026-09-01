#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <functional>

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

// CUDA event timer utility.
struct CudaTimer {
    cudaEvent_t start_, stop_;
    CudaTimer()  { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~CudaTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start(cudaStream_t s = nullptr) { CUDA_CHECK(cudaEventRecord(start_, s)); }
    float stop(cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, s));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

static float time_kernel(int warmup, int iters, std::function<void()> fn) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) fn();
    float ms = t.stop();
    return ms / iters;
}

struct BenchConfig {
    int hidden_dim      = 4096;
    int intermediate_dim = 11008;
    int num_heads       = 32;
    int num_kv_heads    = 32;
    int head_dim        = 128;
};

static void run_bench(const BenchConfig& bc, int kv_len) {
    printf("\n--- kv_len=%d ---\n", kv_len);

    ModelConfig cfg;
    cfg.num_layers       = 1;
    cfg.hidden_dim       = bc.hidden_dim;
    cfg.intermediate_dim = bc.intermediate_dim;
    cfg.num_heads        = bc.num_heads;
    cfg.num_kv_heads     = bc.num_kv_heads;
    cfg.head_dim         = bc.head_dim;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);

    // Allocate weight buffer.
    size_t offsets[9];
    size_t total_w = layer_buffer_layout(cfg, offsets);
    char* weight_buf_h = (char*)malloc(total_w);
    memset(weight_buf_h, 0, total_w);

    auto fill_half = [&](size_t off, int n) {
        __half* p = reinterpret_cast<__half*>(weight_buf_h + off);
        for (int i = 0; i < n; ++i) p[i] = __float2half(dist(rng));
    };
    auto fill_float = [&](size_t off, int n) {
        float* p = reinterpret_cast<float*>(weight_buf_h + off);
        std::uniform_real_distribution<float> nd(0.8f, 1.2f);
        for (int i = 0; i < n; ++i) p[i] = nd(rng);
    };

    int q_cols  = cfg.num_heads    * cfg.head_dim;
    int kv_cols = cfg.num_kv_heads * cfg.head_dim;
    fill_half(offsets[0], cfg.hidden_dim * q_cols);
    fill_half(offsets[1], cfg.hidden_dim * kv_cols);
    fill_half(offsets[2], cfg.hidden_dim * kv_cols);
    fill_half(offsets[3], q_cols * cfg.hidden_dim);
    fill_half(offsets[4], cfg.hidden_dim * cfg.intermediate_dim);
    fill_half(offsets[5], cfg.hidden_dim * cfg.intermediate_dim);
    fill_half(offsets[6], cfg.intermediate_dim * cfg.hidden_dim);
    fill_float(offsets[7], cfg.hidden_dim);
    fill_float(offsets[8], cfg.hidden_dim);

    char* weight_dev;
    CUDA_CHECK(cudaMalloc(&weight_dev, total_w));
    CUDA_CHECK(cudaMemcpy(weight_dev, weight_buf_h, total_w, cudaMemcpyHostToDevice));
    free(weight_buf_h);

    LayerWeights lw = make_layer_weights(weight_dev, cfg);

    // KV cache.
    int kv_dim  = cfg.num_kv_heads * cfg.head_dim;
    int max_seq = kv_len + 2;
    __half *kv_k_ref, *kv_v_ref, *kv_k_per, *kv_v_per;
    CUDA_CHECK(cudaMalloc(&kv_k_ref, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&kv_v_ref, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&kv_k_per, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&kv_v_per, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMemset(kv_k_ref, 0, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMemset(kv_v_ref, 0, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMemset(kv_k_per, 0, (size_t)max_seq * kv_dim * sizeof(__half)));
    CUDA_CHECK(cudaMemset(kv_v_per, 0, (size_t)max_seq * kv_dim * sizeof(__half)));

    __half* hidden_in;
    CUDA_CHECK(cudaMalloc(&hidden_in, cfg.hidden_dim * sizeof(__half)));
    CUDA_CHECK(cudaMemset(hidden_in, 0, cfg.hidden_dim * sizeof(__half)));

    __half *out_ref, *out_per;
    CUDA_CHECK(cudaMalloc(&out_ref, cfg.hidden_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&out_per, cfg.hidden_dim * sizeof(__half)));

    int warmup = 3, iters = 10;

    // Benchmark reference.
    float ref_ms = time_kernel(warmup, iters, [&]() {
        KVCache kvc{ kv_k_ref, kv_v_ref, max_seq, kv_len };
        launch_reference_layer(hidden_in, out_ref, lw, cfg, kvc);
    });

    // Benchmark persistent.
    float per_ms = time_kernel(warmup, iters, [&]() {
        KVCache kvc{ kv_k_per, kv_v_per, max_seq, kv_len };
        launch_persistent_layer(hidden_in, out_per, lw, cfg, kvc);
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    float speedup = ref_ms / per_ms;

    // Global memory traffic eliminated: hidden state store + reload per avoided round-trip.
    // Reference does ~10 round-trips: read+write per intermediate activation.
    // Persistent does 1 load + 1 store.
    int avoided_roundtrips = 8; // conservative: post-norm, q/k/v, attn_out, post-attn, ffn-norm, gate/up, intermediate
    double traffic_saved_mb = (double)avoided_roundtrips * cfg.hidden_dim * sizeof(__half) / (1024.0 * 1024.0);

    printf("  reference  : %.3f ms/iter\n", ref_ms);
    printf("  persistent : %.3f ms/iter\n", per_ms);
    printf("  speedup    : %.2fx\n", speedup);
    printf("  est. smem traffic saved: %.2f MB (%.0f round-trips x %d B)\n",
           traffic_saved_mb, (double)avoided_roundtrips, cfg.hidden_dim * (int)sizeof(__half));

    cudaFree(weight_dev);
    cudaFree(kv_k_ref); cudaFree(kv_v_ref);
    cudaFree(kv_k_per); cudaFree(kv_v_per);
    cudaFree(hidden_in);
    cudaFree(out_ref); cudaFree(out_per);
}

int main() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("=== persist-decode latency benchmark ===\n");
    double bw_gbs = 2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8.0 / 1e9;
    printf("Device: %s  (sm_%d%d, %.1f GB/s peak)\n",
           prop.name, prop.major, prop.minor, bw_gbs);

    BenchConfig bc;  // Llama-7B dimensions.
    printf("\nModel: hidden=%d inter=%d heads=%d kv_heads=%d head_dim=%d\n",
           bc.hidden_dim, bc.intermediate_dim, bc.num_heads, bc.num_kv_heads, bc.head_dim);

    int kv_lens[] = {512, 2048, 8192};
    for (int kv_len : kv_lens) {
        run_bench(bc, kv_len);
    }

    printf("\nDone.\n");
    return 0;
}
