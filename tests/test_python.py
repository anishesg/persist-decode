"""Python correctness test: compare persistent_layer against reference_layer.

Tests two configurations:
  - d=2048, heads=16, kv_len=512
  - d=4096, heads=32, kv_len=2048

Asserts cosine similarity > 0.998 and reports max absolute error.
"""

import sys
import math
import torch

sys.path.insert(0, ".")

from persist_decode import _C  # noqa: E402 -- requires built extension


def make_weights(hidden_dim: int, intermediate_dim: int,
                 num_heads: int, num_kv_heads: int,
                 device: str, seed: int = 42):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    head_dim = hidden_dim // num_heads
    q_cols   = num_heads    * head_dim
    kv_cols  = num_kv_heads * head_dim

    def rh(*shape):
        return (torch.rand(*shape, generator=g, device=device, dtype=torch.float16) - 0.5) * 0.04

    def rf(n):
        return torch.rand(n, generator=g, device=device, dtype=torch.float32) * 0.4 + 0.8

    return {
        "q_proj":    rh(q_cols,              hidden_dim),
        "k_proj":    rh(kv_cols,             hidden_dim),
        "v_proj":    rh(kv_cols,             hidden_dim),
        "o_proj":    rh(hidden_dim,          q_cols),
        "gate_proj": rh(intermediate_dim,    hidden_dim),
        "up_proj":   rh(intermediate_dim,    hidden_dim),
        "down_proj": rh(hidden_dim,          intermediate_dim),
        "attn_norm": rf(hidden_dim),
        "ffn_norm":  rf(hidden_dim),
    }


def make_kv_cache(kv_len: int, num_kv_heads: int, head_dim: int,
                  device: str, max_extra: int = 2, seed: int = 7):
    kv_dim   = num_kv_heads * head_dim
    max_seq  = kv_len + max_extra
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    k = (torch.rand(max_seq, kv_dim, generator=g, device=device, dtype=torch.float16) - 0.5) * 0.2
    v = (torch.rand(max_seq, kv_dim, generator=g, device=device, dtype=torch.float16) - 0.5) * 0.2
    # Zero out the positions past kv_len so they are not real cache entries.
    k[kv_len:] = 0
    v[kv_len:] = 0
    return k.contiguous(), v.contiguous()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float(torch.dot(a.flatten(), b.flatten()) /
                 (torch.norm(a) * torch.norm(b) + 1e-12))


def run_case(hidden_dim: int, intermediate_dim: int,
             num_heads: int, num_kv_heads: int, kv_len: int,
             device: str = "cuda") -> bool:
    head_dim = hidden_dim // num_heads
    print(f"  d={hidden_dim} inter={intermediate_dim} heads={num_heads} "
          f"kv_heads={num_kv_heads} kv_len={kv_len}")

    weights = make_weights(hidden_dim, intermediate_dim, num_heads, num_kv_heads, device)
    kv_k_ref, kv_v_ref = make_kv_cache(kv_len, num_kv_heads, head_dim, device, seed=7)
    kv_k_per, kv_v_per = kv_k_ref.clone(), kv_v_ref.clone()

    g = torch.Generator(device=device)
    g.manual_seed(99)
    x = (torch.rand(hidden_dim, generator=g, device=device, dtype=torch.float16) - 0.5)

    def _call(fn, kv_k, kv_v):
        return fn(
            x, kv_k, kv_v, kv_len,
            weights["q_proj"], weights["k_proj"],
            weights["v_proj"], weights["o_proj"],
            weights["gate_proj"], weights["up_proj"],
            weights["down_proj"], weights["attn_norm"],
            weights["ffn_norm"],
            hidden_dim, intermediate_dim, num_heads, num_kv_heads,
        )

    out_ref = _call(_C.reference_layer_forward,  kv_k_ref, kv_v_ref)
    out_per = _call(_C.persistent_layer_forward, kv_k_per, kv_v_per)

    torch.cuda.synchronize()

    cos = cosine_similarity(out_ref, out_per)
    max_err = float((out_ref.float() - out_per.float()).abs().max())
    passed = cos > 0.998
    status = "PASS" if passed else "FAIL"
    print(f"    cosine_similarity={cos:.6f}  max_abs_error={max_err:.6f}  {status}")
    return passed


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)

    device = "cuda"
    print("=== Python correctness test ===\n")

    cases = [
        dict(hidden_dim=2048, intermediate_dim=5504,  num_heads=16, num_kv_heads=16, kv_len=512),
        dict(hidden_dim=4096, intermediate_dim=11008, num_heads=32, num_kv_heads=32, kv_len=2048),
    ]

    results = [run_case(**c, device=device) for c in cases]
    passed  = sum(results)
    total   = len(results)
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
