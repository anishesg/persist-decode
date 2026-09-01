"""Multi-layer latency benchmark at Llama-7B config.

Config: 32 layers, d=4096, intermediate=11008, 32 heads, 32 kv_heads, kv_len=512

Measures three paths:
  1. Single-launch multi-layer persistent kernel (all 32 layers in one launch)
  2. 32 sequential single-layer persistent calls
  3. 32 sequential multi-launch reference calls

Reports:
  - Total latency for each path (microseconds)
  - Speedup of path 1 vs paths 2 and 3
  - Global memory traffic saved by eliminating inter-layer round-trips
  - Kernel launch count reduction
"""

import sys
import torch

sys.path.insert(0, ".")

from persist_decode import _C  # noqa: E402

NUM_LAYERS   = 32
D            = 4096
INTER        = 11008
NH           = 32
NKV          = 32
HD           = D // NH
KVDIM        = NKV * HD
KV_LEN       = 512
MAX_SEQ      = KV_LEN + NUM_LAYERS + 4
WARMUP       = 5
ITERS        = 30


def make_weights(device: str, seed: int):
    q  = NH  * HD
    kv = NKV * HD
    g  = torch.Generator(device=device)
    g.manual_seed(seed)

    def rh(r, c):
        return (torch.rand(r, c, generator=g, device=device, dtype=torch.float16) - 0.5) * 0.02

    def rf(n):
        return torch.ones(n, device=device, dtype=torch.float32)

    return {
        "q_proj":    rh(q,     D).contiguous(),
        "k_proj":    rh(kv,    D).contiguous(),
        "v_proj":    rh(kv,    D).contiguous(),
        "o_proj":    rh(D,     q).contiguous(),
        "gate_proj": rh(INTER, D).contiguous(),
        "up_proj":   rh(INTER, D).contiguous(),
        "down_proj": rh(D,  INTER).contiguous(),
        "attn_norm": rf(D),
        "ffn_norm":  rf(D),
    }


def fresh_kv(device: str, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    k = (torch.rand(MAX_SEQ, KVDIM, generator=g, device=device, dtype=torch.float16) - 0.5) * 0.1
    v = (torch.rand(MAX_SEQ, KVDIM, generator=g, device=device, dtype=torch.float16) - 0.5) * 0.1
    k[KV_LEN:] = 0
    v[KV_LEN:] = 0
    return k.contiguous(), v.contiguous()


def timed_median(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
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
        times.append(start.elapsed_time(end) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)

    device = "cuda"
    prop   = torch.cuda.get_device_properties(device)
    print(f"=== Multi-layer persistent kernel benchmark on {prop.name} ===")
    print(f"    layers={NUM_LAYERS}  d={D}  inter={INTER}  "
          f"heads={NH}  kv_len={KV_LEN}\n")

    all_w = [make_weights(device, seed=i * 17 + 3) for i in range(NUM_LAYERS)]
    x = torch.randn(D, dtype=torch.float16, device=device)

    # Pre-build all lists once (avoids Python list comprehension overhead inside timing loops).
    q_projs    = [w["q_proj"]    for w in all_w]
    k_projs    = [w["k_proj"]    for w in all_w]
    v_projs    = [w["v_proj"]    for w in all_w]
    o_projs    = [w["o_proj"]    for w in all_w]
    gate_projs = [w["gate_proj"] for w in all_w]
    up_projs   = [w["up_proj"]   for w in all_w]
    down_projs = [w["down_proj"] for w in all_w]
    attn_norms = [w["attn_norm"] for w in all_w]
    ffn_norms  = [w["ffn_norm"]  for w in all_w]

    # Fresh KV caches (cloned each iteration inside fn to prevent cache growth).
    base_ks = [fresh_kv(device, seed=i * 5 + 1)[0] for i in range(NUM_LAYERS)]
    base_vs = [fresh_kv(device, seed=i * 5 + 2)[1] for i in range(NUM_LAYERS)]

    # --- Path 1: single-launch multi-layer persistent kernel ---
    def run_multi_layer():
        kv_k = [k.clone() for k in base_ks]
        kv_v = [v.clone() for v in base_vs]
        return _C.persistent_multi_layer_forward(
            x, kv_k, kv_v, [KV_LEN] * NUM_LAYERS,
            q_projs, k_projs, v_projs, o_projs,
            gate_projs, up_projs, down_projs,
            attn_norms, ffn_norms,
            D, INTER, NH, NKV, NUM_LAYERS,
        )

    lat_ml = timed_median(run_multi_layer)

    # --- Path 2: N sequential single-layer persistent calls ---
    def run_sequential_persist():
        kv_k = [k.clone() for k in base_ks]
        kv_v = [v.clone() for v in base_vs]
        cur = x
        for l in range(NUM_LAYERS):
            cur = _C.persistent_layer_forward(
                cur, kv_k[l], kv_v[l], KV_LEN,
                q_projs[l], k_projs[l], v_projs[l], o_projs[l],
                gate_projs[l], up_projs[l], down_projs[l],
                attn_norms[l], ffn_norms[l],
                D, INTER, NH, NKV,
            )
        return cur

    lat_seq_per = timed_median(run_sequential_persist)

    # --- Path 3: N sequential multi-launch reference calls ---
    def run_sequential_ref():
        kv_k = [k.clone() for k in base_ks]
        kv_v = [v.clone() for v in base_vs]
        cur = x
        for l in range(NUM_LAYERS):
            cur = _C.reference_layer_forward(
                cur, kv_k[l], kv_v[l], KV_LEN,
                q_projs[l], k_projs[l], v_projs[l], o_projs[l],
                gate_projs[l], up_projs[l], down_projs[l],
                attn_norms[l], ffn_norms[l],
                D, INTER, NH, NKV,
            )
        return cur

    lat_seq_ref = timed_median(run_sequential_ref)

    # --- Analysis ---
    # Inter-layer hidden state round-trips eliminated:
    #   each layer boundary would have written hidden_dim fp16 to GMEM and read it back.
    #   With 32 layers that is 31 write+read pairs.
    inter_layer_trips  = NUM_LAYERS - 1
    bytes_per_trip     = D * 2 * 2  # write + read, 2 bytes per fp16
    traffic_saved_kb   = inter_layer_trips * bytes_per_trip / 1024

    # Kernel launch count: multi-layer=1, sequential persist=N*1=N,
    # reference=N*(2 norms+3 linears_attn+1 attn+1 o_proj+1 res_add+
    #             1 rms_ffn+2 ffn_linears+1 silu+1 down+1 res_add) = N*~14
    launches_ml   = 1
    launches_sper = NUM_LAYERS
    launches_sref = NUM_LAYERS * 14  # approximate ops per reference layer

    print(f"  Latency (median over {ITERS} iterations):")
    print(f"    single-launch multi-layer  : {lat_ml:8.1f} us")
    print(f"    sequential persistent (x{NUM_LAYERS}) : {lat_seq_per:8.1f} us  "
          f"speedup vs ml: {lat_seq_per/lat_ml:.2f}x")
    print(f"    sequential reference  (x{NUM_LAYERS}) : {lat_seq_ref:8.1f} us  "
          f"speedup vs ml: {lat_seq_ref/lat_ml:.2f}x")
    print()
    print(f"  Inter-layer GMEM traffic eliminated : {traffic_saved_kb:.1f} KB "
          f"({inter_layer_trips} x {bytes_per_trip} B)")
    print(f"  Kernel launches: ml={launches_ml}  seq_persist={launches_sper}  "
          f"seq_ref~{launches_sref}")


if __name__ == "__main__":
    main()
