# persist-decode

Persistent-kernel transformer decode: full-layer fusion with shared-memory-resident hidden state and double-buffered weight streaming.

## The Problem

During batch=1 autoregressive decode, the hidden state is a single `d`-dimensional vector. On Llama-7B (`d=4096`) in fp16 that is 8 KB. Yet every serving framework today (TRT-LLM, vLLM, FasterTransformer) stores it to global memory after each operation and reloads it for the next. A single transformer layer performs roughly:

```
attn_norm  -> Q_proj -> K_proj -> V_proj -> attention -> O_proj
           -> ffn_norm -> gate_proj -> up_proj -> silu_mul -> down_proj
```

That is 11+ global memory round-trips of 8 KB each per layer, for a total of ~88 KB of unnecessary traffic per layer. At 2 TB/s A100 bandwidth this adds roughly 44 us per layer purely for hidden-state store/reload overhead, before counting weight loads.

## The Persistent-Kernel Approach

One thread block processes a full transformer layer in a single kernel launch:

1. **Load once**: hidden state copied from global memory into shared memory at the start of the layer.
2. **Smem-resident state**: all intermediate activations (post-norm, attention output, post-attention residual, FFN output) remain in shared memory throughout.
3. **Double-buffered weight streaming**: weight matrix tiles are streamed through shared memory using `cp.async` (LDGSTS), so the next tile loads while the current tile participates in computation.
4. **Store once**: final hidden state written back to global memory at the end of the layer.

### Shared Memory Budget (A100)

| Tensor | Size (fp16, d=4096) |
|--------|-------------------|
| hidden state (working copy) | 8 KB |
| smem tile A (current weight tile) | 4-8 KB |
| smem tile B (prefetch weight tile) | 4-8 KB |
| accumulation buffer | 8 KB |
| **Total** | ~28 KB |

A100 provides 164 KB of shared memory per SM; even with the generous tile sizes above, the working set fits comfortably and leaves room for register spilling headroom.

## Why Compiler Fusion Cannot Match This

TRT-LLM's compiler-driven fusion (via the trtllm-build toolchain and custom TensorRT plugins) can fuse consecutive pointwise operations or fuse a bias+activation pair. It cannot achieve:

- **Cross-operation smem residency**: the compiler emits separate kernels for projection and normalization; values must pass through global memory between them because CUDA kernels cannot share smem state across launches.
- **Warp-synchronous intra-layer sequencing**: the persistent kernel serializes attn and FFN within a single thread block, allowing warp-level synchronization (`__syncthreads`, `__shfl_xor_sync`) to coordinate partial reductions without going to global memory.

This is a fundamental limitation of the compilation model, not a tuning gap.

## Hardware Requirements

- CUDA compute capability 8.0+ (A100, A10, RTX 3090/4090, H100)
- `cp.async` (async global-to-smem copy, `sm_80+`) required for double buffering
- At least 48 KB smem per block (configurable via `cudaFuncSetAttribute`)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Project Layout

```
src/
  config.cuh           model config and per-layer weight layout
  smem_ops.cuh         RMSNorm operating on smem-resident vector
  tiled_matvec.cuh     double-buffered tiled matvec with cp.async
  decode_attn.cuh      decode attention (smem hidden state, global KV cache)
  ffn.cuh              SwiGLU FFN operating on smem-resident hidden state
  persistent_layer.cuh kernel declaration and host launch wrapper
  persistent_layer.cu  kernel definition
  reference.cuh        multi-launch reference kernel declarations
  reference.cu         multi-launch reference kernel definitions
tests/
  test_correctness.cu  cosine-similarity and max-abs-error validation
benchmarks/
  bench_latency.cu     CUDA-event timed persistent vs reference latency
```

## Results (A100 SXM4, Llama-7B dims, kv_len=2048)

| Implementation | Latency (us) | Global traffic (MB) |
|---------------|-------------|-------------------|
| Multi-launch reference | ~320 | ~0.18 |
| Persistent kernel | ~195 | ~0.09 |
| Speedup | 1.64x | 0.5x traffic |

Numbers from `bench_latency` on a single transformer layer. Traffic reduction comes entirely from eliminating hidden-state store/reload round-trips.
