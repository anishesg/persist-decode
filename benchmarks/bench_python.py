"""Python CUDA-event latency benchmark at Llama-7B dimensions.

Config: d=4096, intermediate=11008, 32 heads, 32 kv_heads, head_dim=128
kv_len in {512, 2048, 8192}

Reports per-variant latency (microseconds), speedup, and estimated global
memory traffic saved by the persistent kernel versus the multi-launch reference.
"""

import sys
import math
import torch

sys.path.insert(0, ".")

from persist_decode import _C  # noqa: E402


def make_weights(d: int, inter: int, nh: int, nkv: int, device: str):
    head_dim = d // nh
    q  = nh    * head_dim
    kv = nkv   * head_dim
    rh = lambda r, c: torch.randn(r, c, dtype=torch.float16, device=device) * 0.02
    rf = lambda n:    torch.ones(n, dtype=torch.float32, device=device)
    return {
        "q_proj":    rh(q,    d).contiguous(),
        "k_proj":    rh(kv,   d).contiguous(),
        "v_proj":    rh(kv,   d).contiguous(),
        "o_proj":    rh(d,    q).contiguous(),
        "gate_proj": rh(inter, d).contiguous(),
        "up_proj":   rh(inter, d).contiguous(),
        "down_proj": rh(d,  inter).contiguous(),
        "attn_norm": rf(d),
        "ffn_norm":  rf(d),
    }


def cuda_timed(fn, warmup: int = 10, iters: int = 50) -> float:
    """Returns median latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # ms -> us
    times.sort()
    return times[len(times) // 2]


def bench_kv_len(kv_len: int, device: str):
    D, INTER, NH, NKV = 4096, 11008, 32, 32
    HD   = D // NH
    KVDIM = NKV * HD

    w = make_weights(D, INTER, NH, NKV, device)
    x = torch.randn(D, dtype=torch.float16, device=device)

    max_seq = kv_len + 4
    kv_k_ref = torch.zeros(max_seq, KVDIM, dtype=torch.float16, device=device)
    kv_v_ref = torch.zeros(max_seq, KVDIM, dtype=torch.float16, device=device)
    kv_k_per = kv_k_ref.clone()
    kv_v_per = kv_v_ref.clone()

    def call_ref():
        return _C.reference_layer_forward(
            x, kv_k_ref, kv_v_ref, kv_len,
            w["q_proj"], w["k_proj"], w["v_proj"], w["o_proj"],
            w["gate_proj"], w["up_proj"], w["down_proj"],
            w["attn_norm"], w["ffn_norm"],
            D, INTER, NH, NKV,
        )

    def call_per():
        return _C.persistent_layer_forward(
            x, kv_k_per, kv_v_per, kv_len,
            w["q_proj"], w["k_proj"], w["v_proj"], w["o_proj"],
            w["gate_proj"], w["up_proj"], w["down_proj"],
            w["attn_norm"], w["ffn_norm"],
            D, INTER, NH, NKV,
        )

    lat_ref = cuda_timed(call_ref)
    lat_per = cuda_timed(call_per)
    speedup = lat_ref / lat_per if lat_per > 0 else float("nan")

    # Global memory traffic saved: the reference allocates and frees 8 temporary
    # buffers (norm_out, q_buf, k_buf, v_buf, attn_out, gate_buf, up_buf, residual_buf).
    # Approximate total R+W in bytes across all intermediate tensors.
    # persistent kernel reads hidden once from GMEM and writes once; no intermediate gmem.
    intermediates = (
        D     +   # norm_out read + write = 2D, but written once from RMSNorm
        D * 2 +   # q_proj read W, and also write q_buf
        KVDIM * 4 +  # k_buf, v_buf, k_cache_write, v_cache_write
        D     +   # attn_out
        INTER * 2 +  # gate_buf, up_buf
        D         # down_proj output (residual_buf + final)
    )
    traffic_saved_mb = intermediates * 2 / 1e6  # fp16 = 2 bytes each

    print(f"  kv_len={kv_len:5d}:  ref={lat_ref:8.1f} us  persist={lat_per:8.1f} us  "
          f"speedup={speedup:.2f}x  gmem_saved~{traffic_saved_mb:.2f} MB")


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)

    device = "cuda"
    prop = torch.cuda.get_device_properties(device)
    print(f"=== Llama-7B Python latency benchmark on {prop.name} ===")
    print(f"    d=4096  inter=11008  heads=32\n")

    for kv_len in [512, 2048, 8192]:
        bench_kv_len(kv_len, device)

    print()


if __name__ == "__main__":
    main()
