"""Stateless functional API for persistent-kernel transformer decode.

Both functions accept the same set of arguments and return the post-layer
hidden state as a float16 CUDA tensor. The KV-cache tensors are mutated
in-place by the underlying C kernel (new K/V appended at position kv_seq_len).
"""

from __future__ import annotations
from typing import Dict
import torch


def _ext():
    try:
        from persist_decode import _C
        return _C
    except ImportError as e:
        raise ImportError(
            "persist_decode C extension not found. Run: pip install -e ."
        ) from e


def persistent_layer(
    x: torch.Tensor,
    kv_cache: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    hidden_dim: int,
    intermediate_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Run one transformer layer using the persistent single-launch kernel.

    Args:
        x: float16 CUDA tensor of shape [hidden_dim], the current token's
           hidden state before this layer.
        kv_cache: dict with keys 'k' and 'v' (float16 CUDA tensors of shape
                  [max_seq_len, num_kv_heads * head_dim]) and 'seq_len' (int).
                  The cache is mutated in-place.
        weights: dict with keys q_proj, k_proj, v_proj, o_proj, gate_proj,
                 up_proj, down_proj (float16), attn_norm, ffn_norm (float32).
        hidden_dim, intermediate_dim, num_heads, num_kv_heads: model dimensions.

    Returns:
        float16 CUDA tensor of shape [hidden_dim].
    """
    _C = _ext()
    kv_k = kv_cache["k"].contiguous()
    kv_v = kv_cache["v"].contiguous()
    seq_len = int(kv_cache["seq_len"])

    out = _C.persistent_layer_forward(
        x.contiguous().half(),
        kv_k, kv_v, seq_len,
        weights["q_proj"].contiguous(),
        weights["k_proj"].contiguous(),
        weights["v_proj"].contiguous(),
        weights["o_proj"].contiguous(),
        weights["gate_proj"].contiguous(),
        weights["up_proj"].contiguous(),
        weights["down_proj"].contiguous(),
        weights["attn_norm"].contiguous(),
        weights["ffn_norm"].contiguous(),
        hidden_dim, intermediate_dim, num_heads, num_kv_heads,
    )
    kv_cache["seq_len"] = seq_len + 1
    return out


def reference_layer(
    x: torch.Tensor,
    kv_cache: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    hidden_dim: int,
    intermediate_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Run one transformer layer using the multi-launch reference kernel.

    Arguments and return value are identical to persistent_layer().
    """
    _C = _ext()
    kv_k = kv_cache["k"].contiguous()
    kv_v = kv_cache["v"].contiguous()
    seq_len = int(kv_cache["seq_len"])

    out = _C.reference_layer_forward(
        x.contiguous().half(),
        kv_k, kv_v, seq_len,
        weights["q_proj"].contiguous(),
        weights["k_proj"].contiguous(),
        weights["v_proj"].contiguous(),
        weights["o_proj"].contiguous(),
        weights["gate_proj"].contiguous(),
        weights["up_proj"].contiguous(),
        weights["down_proj"].contiguous(),
        weights["attn_norm"].contiguous(),
        weights["ffn_norm"].contiguous(),
        hidden_dim, intermediate_dim, num_heads, num_kv_heads,
    )
    kv_cache["seq_len"] = seq_len + 1
    return out
