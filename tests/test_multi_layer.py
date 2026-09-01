"""Correctness test for the multi-layer persistent kernel.

Compares persistent_multi_layer_forward against N sequential reference_layer_forward
calls at two configurations:
  - 2 layers, d=2048, heads=16, kv_len=512
  - 4 layers, d=4096, heads=32, kv_len=1024

Asserts cosine similarity > 0.998 and reports max absolute error per config.
"""

import sys
import torch

sys.path.insert(0, ".")

from persist_decode import _C  # noqa: E402


def make_weights(d: int, inter: int, nh: int, nkv: int, device: str, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    hd  = d // nh
    q   = nh  * hd
    kv  = nkv * hd

    def rh(r, c):
        return (torch.rand(r, c, generator=g, device=device, dtype=torch.float16) - 0.5) * 0.04

    def rf(n):
        return torch.rand(n, generator=g, device=device, dtype=torch.float32) * 0.4 + 0.8

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


def run_case(num_layers: int, d: int, inter: int, nh: int, nkv: int,
             kv_len: int, device: str) -> bool:
    print(f"  layers={num_layers} d={d} inter={inter} heads={nh} kv_len={kv_len}")
    hd   = d // nh
    kvdim = nkv * hd
    max_seq = kv_len + num_layers + 2

    all_w = [make_weights(d, inter, nh, nkv, device, seed=i * 13 + 7)
             for i in range(num_layers)]

    g = torch.Generator(device=device)
    g.manual_seed(42)
    x = (torch.rand(d, generator=g, device=device, dtype=torch.float16) - 0.5)

    # Pre-fill identical KV caches for both paths.
    def fresh_kv():
        g2 = torch.Generator(device=device)
        g2.manual_seed(99)
        k = (torch.rand(max_seq, kvdim, generator=g2, device=device, dtype=torch.float16) - 0.5) * 0.1
        v = (torch.rand(max_seq, kvdim, generator=g2, device=device, dtype=torch.float16) - 0.5) * 0.1
        k[kv_len:] = 0
        v[kv_len:] = 0
        return k.contiguous(), v.contiguous()

    ref_ks   = [fresh_kv()[0] for _ in range(num_layers)]
    ref_vs   = [fresh_kv()[1] for _ in range(num_layers)]
    ml_ks    = [k.clone() for k in ref_ks]
    ml_vs    = [v.clone() for v in ref_vs]

    # Reference: N sequential reference_layer_forward calls.
    cur = x
    for l in range(num_layers):
        w = all_w[l]
        cur = _C.reference_layer_forward(
            cur, ref_ks[l], ref_vs[l], kv_len,
            w["q_proj"], w["k_proj"], w["v_proj"], w["o_proj"],
            w["gate_proj"], w["up_proj"], w["down_proj"],
            w["attn_norm"], w["ffn_norm"],
            d, inter, nh, nkv,
        )
    out_ref = cur

    # Multi-layer persistent kernel.
    kv_seq_lens = [kv_len] * num_layers
    out_ml = _C.persistent_multi_layer_forward(
        x, ml_ks, ml_vs, kv_seq_lens,
        [w["q_proj"]    for w in all_w],
        [w["k_proj"]    for w in all_w],
        [w["v_proj"]    for w in all_w],
        [w["o_proj"]    for w in all_w],
        [w["gate_proj"] for w in all_w],
        [w["up_proj"]   for w in all_w],
        [w["down_proj"] for w in all_w],
        [w["attn_norm"] for w in all_w],
        [w["ffn_norm"]  for w in all_w],
        d, inter, nh, nkv, num_layers,
    )

    torch.cuda.synchronize()

    a = out_ref.float()
    b = out_ml.float()
    cos = float(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12))
    max_err = float((a - b).abs().max())
    passed = cos > 0.998
    status = "PASS" if passed else "FAIL"
    print(f"    cosine_similarity={cos:.6f}  max_abs_error={max_err:.6f}  {status}")
    return passed


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)

    device = "cuda"
    print("=== multi-layer persistent kernel correctness test ===\n")

    cases = [
        dict(num_layers=2, d=2048, inter=5504,  nh=16, nkv=16, kv_len=512),
        dict(num_layers=4, d=4096, inter=11008, nh=32, nkv=32, kv_len=1024),
    ]

    results = [run_case(**c, device=device) for c in cases]
    passed  = sum(results)
    total   = len(results)
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
