#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

// Double-buffered tiled matvec: y[rows] = W[rows, cols] * x[cols]
//
// W is in global memory (row-major). x is in shared memory. y written to shared memory.
//
// Each thread handles rows: tid, tid+blockDim.x, tid+2*blockDim.x, ...
// For each assigned row, the column dimension is tiled with width TILE_K.
//
// cp.async double buffering: smem_tile_a/b each hold blockDim.x * TILE_K half values.
// Row r's slice occupies smem_tile[r_local * TILE_K .. (r_local+1)*TILE_K].
// While computing with the current tile, the next W column-strip is being fetched.
//
// smem_tile_a, smem_tile_b: caller-allocated, each of size blockDim.x * TILE_K * sizeof(half).

template <int TILE_K = 64, typename ACC_T = float>
__device__ void tiled_matvec(const __half* __restrict__ W,
                              const __half* __restrict__ x_smem,
                              __half* __restrict__ y_smem,
                              int rows,
                              int cols,
                              __half* smem_tile_a,
                              __half* smem_tile_b) {
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    __half* bufs[2] = {smem_tile_a, smem_tile_b};

    // Number of full or partial column tiles.
    int num_tiles = (cols + TILE_K - 1) / TILE_K;

    // Per-row accumulators in registers: one per row owned by this thread.
    // rows <= 2 * bsz in all our use cases (hidden_dim / bsz <= 2 for bsz=256, d=4096 -> 16 rows).
    // We allocate a fixed-size array; unused slots remain at zero.
    constexpr int kMaxRowsPerThread = 64;
    ACC_T acc[kMaxRowsPerThread] = {};
    int num_owned = (rows + bsz - 1) / bsz; // rows owned by this thread

    auto load_tile = [&](int tile_idx, __half* dst) {
        if (tile_idx >= num_tiles) return;
        int k_start = tile_idx * TILE_K;
        int k_end   = min(k_start + TILE_K, cols);
        int tile_w  = k_end - k_start;

        // Each thread loads its own row(s) slice from global W.
        for (int r_local = 0; r_local < num_owned; ++r_local) {
            int row = tid + r_local * bsz;
            if (row >= rows) break;
            const __half* src = W + (size_t)row * cols + k_start;
            __half* d = dst + r_local * TILE_K;
            for (int k = 0; k < tile_w; k += 1) {
                // Use cp.async for each element (minimum granularity is 4 bytes = 2 halves).
                // For simplicity of addressing we fall back to vectorized 4B copies.
                // Pair-wise to meet cp.async 4-byte minimum.
                if ((k & 1) == 0 && k + 1 < tile_w) {
                    __pipeline_memcpy_async(d + k, src + k, sizeof(int)); // 4 bytes = 2 halves
                } else if ((k & 1) == 0) {
                    // Last element and odd tile_w: can't use cp.async for 2B, load normally.
                    d[k] = src[k];
                }
            }
        }
        __pipeline_commit();
    };

    // Prefetch first tile.
    load_tile(0, bufs[0]);
    int cur = 0;

    for (int t = 0; t < num_tiles; ++t) {
        // Prefetch next tile while we wait for the current one.
        if (t + 1 < num_tiles) {
            load_tile(t + 1, bufs[cur ^ 1]);
        }

        // Wait for current tile to be ready (allow at most 1 outstanding if next was issued).
        __pipeline_wait_prior(t + 1 < num_tiles ? 1 : 0);
        __syncthreads();

        int k_start = t * TILE_K;
        int k_end   = min(k_start + TILE_K, cols);
        int tile_w  = k_end - k_start;

        __half* cur_buf = bufs[cur];

        for (int r_local = 0; r_local < num_owned; ++r_local) {
            int row = tid + r_local * bsz;
            if (row >= rows) break;
            const __half* w_tile = cur_buf + r_local * TILE_K;
            ACC_T sum = acc[r_local];
            for (int k = 0; k < tile_w; ++k) {
                sum += (ACC_T)__half2float(w_tile[k]) *
                       (ACC_T)__half2float(x_smem[k_start + k]);
            }
            acc[r_local] = sum;
        }

        cur ^= 1;
        __syncthreads();
    }

    // Write outputs.
    for (int r_local = 0; r_local < num_owned; ++r_local) {
        int row = tid + r_local * bsz;
        if (row >= rows) break;
        y_smem[row] = __float2half((float)acc[r_local]);
    }
    __syncthreads();
}
