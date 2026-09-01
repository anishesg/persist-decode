from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class DecodeConfig:
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int = 1

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


def _load_ext():
    try:
        from persist_decode import _C
        return _C
    except ImportError as e:
        raise ImportError(
            "persist_decode C extension not found. Run: pip install -e ."
        ) from e


class PersistentDecodeLayer(nn.Module):
    """Single transformer decoder layer backed by the persistent kernel.

    All weight tensors are stored as float16. The forward() path dispatches to the
    fused single-launch persistent kernel; reference_forward() dispatches to the
    multi-launch reference implementation for correctness comparison.
    """

    def __init__(self, cfg: DecodeConfig, max_seq_len: int = 4096) -> None:
        super().__init__()
        d   = cfg.hidden_dim
        i   = cfg.intermediate_dim
        q   = cfg.num_heads    * cfg.head_dim
        kv  = cfg.num_kv_heads * cfg.head_dim

        self.cfg         = cfg
        self.max_seq_len = max_seq_len

        # Attention projection weights [out_dim, in_dim] stored row-major.
        self.q_proj    = nn.Parameter(torch.zeros(q,  d,  dtype=torch.float16))
        self.k_proj    = nn.Parameter(torch.zeros(kv, d,  dtype=torch.float16))
        self.v_proj    = nn.Parameter(torch.zeros(kv, d,  dtype=torch.float16))
        self.o_proj    = nn.Parameter(torch.zeros(d,  q,  dtype=torch.float16))

        # FFN weights.
        self.gate_proj = nn.Parameter(torch.zeros(i,  d,  dtype=torch.float16))
        self.up_proj   = nn.Parameter(torch.zeros(i,  d,  dtype=torch.float16))
        self.down_proj = nn.Parameter(torch.zeros(d,  i,  dtype=torch.float16))

        # Norm weights (float32).
        self.attn_norm = nn.Parameter(torch.ones(d,  dtype=torch.float32))
        self.ffn_norm  = nn.Parameter(torch.ones(d,  dtype=torch.float32))

        self._init_kv_cache()

    def _init_kv_cache(self) -> None:
        cfg = self.cfg
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        self.register_buffer(
            "kv_k", torch.zeros(self.max_seq_len, kv_dim, dtype=torch.float16)
        )
        self.register_buffer(
            "kv_v", torch.zeros(self.max_seq_len, kv_dim, dtype=torch.float16)
        )
        self.kv_seq_len: int = 0

    def reset_kv_cache(self) -> None:
        self.kv_k.zero_()
        self.kv_v.zero_()
        self.kv_seq_len = 0

    def _common_args(self):
        cfg = self.cfg
        return dict(
            hidden_dim=cfg.hidden_dim,
            intermediate_dim=cfg.intermediate_dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the persistent single-launch kernel."""
        _C = _load_ext()
        x = x.contiguous().half()
        kv_k = self.kv_k.contiguous()
        kv_v = self.kv_v.contiguous()
        out = _C.persistent_layer_forward(
            x, kv_k, kv_v, self.kv_seq_len,
            self.q_proj.contiguous(),
            self.k_proj.contiguous(),
            self.v_proj.contiguous(),
            self.o_proj.contiguous(),
            self.gate_proj.contiguous(),
            self.up_proj.contiguous(),
            self.down_proj.contiguous(),
            self.attn_norm.contiguous(),
            self.ffn_norm.contiguous(),
            **self._common_args(),
        )
        self.kv_seq_len += 1
        return out

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the multi-launch reference kernel."""
        _C = _load_ext()
        x = x.contiguous().half()
        kv_k = self.kv_k.contiguous()
        kv_v = self.kv_v.contiguous()
        out = _C.reference_layer_forward(
            x, kv_k, kv_v, self.kv_seq_len,
            self.q_proj.contiguous(),
            self.k_proj.contiguous(),
            self.v_proj.contiguous(),
            self.o_proj.contiguous(),
            self.gate_proj.contiguous(),
            self.up_proj.contiguous(),
            self.down_proj.contiguous(),
            self.attn_norm.contiguous(),
            self.ffn_norm.contiguous(),
            **self._common_args(),
        )
        self.kv_seq_len += 1
        return out

    @classmethod
    def from_config(cls, cfg: DecodeConfig, max_seq_len: int = 4096,
                    device: Optional[str] = None) -> "PersistentDecodeLayer":
        layer = cls(cfg, max_seq_len=max_seq_len)
        if device is not None:
            layer = layer.to(device)
        return layer


class PersistentDecodeModel(nn.Module):
    """N-layer transformer decoder using the fused multi-layer persistent kernel.

    All layers share the same config. The multi_layer_forward() path processes all
    N layers in a single kernel launch, keeping the hidden state in shared memory
    across layers. sequential_forward() runs N single-layer persistent calls for
    comparison.
    """

    def __init__(self, cfg: DecodeConfig, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.cfg         = cfg
        self.max_seq_len = max_seq_len
        self.layers      = nn.ModuleList(
            [PersistentDecodeLayer(cfg, max_seq_len=max_seq_len)
             for _ in range(cfg.num_layers)]
        )

    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            layer.reset_kv_cache()

    def multi_layer_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all layers in a single persistent kernel launch."""
        _C = _load_ext()
        cfg = self.cfg
        x = x.contiguous().half()

        all_kv_k = [l.kv_k.contiguous() for l in self.layers]
        all_kv_v = [l.kv_v.contiguous() for l in self.layers]
        kv_seq_lens = [l.kv_seq_len for l in self.layers]

        q_projs    = [l.q_proj.contiguous()    for l in self.layers]
        k_projs    = [l.k_proj.contiguous()    for l in self.layers]
        v_projs    = [l.v_proj.contiguous()    for l in self.layers]
        o_projs    = [l.o_proj.contiguous()    for l in self.layers]
        gate_projs = [l.gate_proj.contiguous() for l in self.layers]
        up_projs   = [l.up_proj.contiguous()   for l in self.layers]
        down_projs = [l.down_proj.contiguous() for l in self.layers]
        attn_norms = [l.attn_norm.contiguous() for l in self.layers]
        ffn_norms  = [l.ffn_norm.contiguous()  for l in self.layers]

        out = _C.persistent_multi_layer_forward(
            x, all_kv_k, all_kv_v, kv_seq_lens,
            q_projs, k_projs, v_projs, o_projs,
            gate_projs, up_projs, down_projs,
            attn_norms, ffn_norms,
            cfg.hidden_dim, cfg.intermediate_dim,
            cfg.num_heads, cfg.num_kv_heads,
            cfg.num_layers,
        )
        for l in self.layers:
            l.kv_seq_len += 1
        return out

    def sequential_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run N sequential single-layer persistent calls."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    @classmethod
    def from_config(cls, cfg: DecodeConfig, max_seq_len: int = 4096,
                    device: Optional[str] = None) -> "PersistentDecodeModel":
        model = cls(cfg, max_seq_len=max_seq_len)
        if device is not None:
            model = model.to(device)
        return model
