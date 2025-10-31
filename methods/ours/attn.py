# methods/ours/attn.py
# OursAttention skeleton (Python 3.9). Wraps timm Attention; currently identity-safe.
# TODO: insert selector/merge/norm logic (merge -> attend -> unmerge).

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class OursAttention(nn.Module):
    """Shape-preserving wrapper around timm Attention (identity until logic is wired)."""

    def __init__(self, inner_attn: nn.Module, cfg: Optional[Dict[str, Any]] = None):
        super(OursAttention, self).__init__()
        self.inner = inner_attn
        self.cfg = dict(cfg or {})

        # Core knobs
        self.r_global = int(self.cfg.get("r", 0))
        self.match_feature = str(self.cfg.get("match_feature", self.cfg.get("match-feature", "k")))
        self.merge_space = str(self.cfg.get("merge", "kv"))  # 'kv' or 'v'

        # Selector knobs (placeholders)
        self.selector = str(self.cfg.get("selector", "hquota_ff"))
        self.hq_q = float(self.cfg.get("hq_q", 0.3))
        self.gamma = float(self.cfg.get("gamma", 0.0))
        self.cand_extra = int(self.cfg.get("cand_extra", 128))

        # Norm/merge knobs (placeholders)
        self.alpha = float(self.cfg.get("alpha", 0.0))
        self.beta0 = float(self.cfg.get("beta0", 0.0))
        self.top_r = int(self.cfg.get("top_r", 0))
        self.l2_clip_tau = float(self.cfg.get("l2_clip_tau", 0.0))
        self.temp_eta = float(self.cfg.get("temp_eta", 0.0))

        # Detect timm attributes (robust across versions)
        self.num_heads = int(getattr(self.inner, "num_heads", 0))
        self.head_dim = getattr(self.inner, "head_dim", None)
        if self.head_dim is None:
            # Fallback infer: embed_dim / num_heads
            embed_dim = getattr(self.inner, "dim", None)
            if embed_dim is None and hasattr(self.inner, "qkv") and hasattr(self.inner.qkv, "weight"):
                embed_dim = int(self.inner.qkv.weight.shape[1])
            if embed_dim is not None and self.num_heads:
                self.head_dim = max(1, int(embed_dim // self.num_heads))
        self.scale = getattr(self.inner, "scale", 1.0 if self.head_dim is None else 1.0 / (self.head_dim ** 0.5))
        self.proj = getattr(self.inner, "proj", nn.Identity())
        self.proj_drop = getattr(self.inner, "proj_drop", nn.Identity())
        self.attn_drop = getattr(self.inner, "attn_drop", nn.Identity())

        # qkv path
        self.has_fused_qkv = hasattr(self.inner, "qkv")
        if not self.has_fused_qkv:
            # Split projections must exist if fused is absent
            if not (hasattr(self.inner, "q") and hasattr(self.inner, "k") and hasattr(self.inner, "v")):
                raise RuntimeError("OursAttention: inner_attn must expose qkv or q/k/v projections")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity-safe fallback for now (until reduction logic is wired in step 3+).
        # Preserve original shapes and residual compatibility.
        return self.inner(x)

    # --- Below are helpers reserved for later steps (not used yet) ---

    def _qkv(self, x: torch.Tensor):
        """Return q,k,v as [B,H,T,Hd]."""
        B, T, C = x.shape
        H = self.num_heads if self.num_heads else 1
        Hd = self.head_dim if self.head_dim is not None else max(1, C // H)
        if self.has_fused_qkv:
            qkv = self.inner.qkv(x)                              # [B, T, 3*C]
            qkv = qkv.reshape(B, T, 3, H, Hd).permute(2, 0, 3, 1, 4)  # [3, B, H, T, Hd]
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = self.inner.q(x).view(B, T, H, Hd).permute(0, 2, 1, 3)
            k = self.inner.k(x).view(B, T, H, Hd).permute(0, 2, 1, 3)
            v = self.inner.v(x).view(B, T, H, Hd).permute(0, 2, 1, 3)
        return q, k, v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Standard attention on possibly reduced tokens; returns [B,T',H*Hd] before proj."""
        import torch.nn.functional as F
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v)  # [B,H,T',Hd]
        y = y.transpose(1, 2).contiguous().view(y.shape[0], y.shape[2], -1)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

    # Placeholders for future: selector, merge, unmerge interfaces
    def _select(self, metric: torch.Tensor, r_block: int):
        """Return keep_idx, assign map (placeholder)."""
        return None, None

    def _merge_kv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep_idx, assign):
        """Return merged q,k,v and mapping needed for unmerge (placeholder)."""
        return q, k, v, None

    def _unmerge(self, y_reduced: torch.Tensor, mapping, T: int):
        """Scatter reduced tokens back to length T (placeholder)."""
        return y_reduced
