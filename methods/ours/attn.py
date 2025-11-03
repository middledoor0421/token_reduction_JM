# methods/ours/attn.py
# OursAttention: merge -> attend -> unmerge (Python 3.9)
# Comments in English only.

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .selectors import get_selector
from .merges import (
    size_weighted_merge_v,
    mean_merge_k,
    build_unmerge_map,
    apply_unmerge,
    apply_pushlite,
)
from .sched import get_scheduler
from .norm.l2_clip import l2_clip_
from .norm.temp_scale import apply_size_temperature
from .norm.recenter import recenter_


class OursAttention(nn.Module):
    """Wrap timm Attention, reduce tokens by our policy, run attention on reduced set,
    then unmerge back to original length [B,T,C] so residual add is shape-safe.
    """

    def __init__(self, inner_attn: nn.Module, cfg: Dict[str, Any]):
        super(OursAttention, self).__init__()
        self.inner = inner_attn
        self.cfg = dict(cfg or {})

        # Core knobs
        self.match_feature = str(self.cfg.get("match_feature", self.cfg.get("match-feature", "k")))  # 'k'|'xnorm'
        self.merge_space = str(self.cfg.get("merge", "kv"))  # 'kv' or 'v'
        self.keep_str = self.cfg.get("keep", None)           # optional string like "0.68,0.66,0.64"
        self.r_global = int(self.cfg.get("r", 0))

        # Selector
        self.selector_name = str(self.cfg.get("selector", "hquota_ff"))
        self.hq_q = float(self.cfg.get("hq_q", 0.3))
        self.gamma = float(self.cfg.get("gamma", 0.0))
        self.cand_extra = int(self.cfg.get("cand_extra", 128))
        self._selector = get_selector(self.selector_name)

        # Norm / push-lite
        self.l2_clip_tau = float(self.cfg.get("l2_clip_tau", 0.0))
        self.alpha = float(self.cfg.get("alpha", 0.0))
        self.beta0 = float(self.cfg.get("beta0", 0.5))
        self.top_r = int(self.cfg.get("top_r", 0))
        self.temp_eta = float(self.cfg.get("temp_eta", 0.0))

        # Scheduler
        self.schedule_name = str(self.cfg.get("schedule", "early_bias"))
        self._scheduler = get_scheduler(self.schedule_name)

        # Special tokens (DeiT has CLS, distill token optional)
        self.has_cls = True
        self.has_dist = False

        # Infer timm attention attributes robustly
        self.num_heads = int(getattr(self.inner, "num_heads", 0))
        self.head_dim = getattr(self.inner, "head_dim", None)
        if self.head_dim is None:
            embed_dim = getattr(self.inner, "dim", None)
            if embed_dim is None and hasattr(self.inner, "qkv") and hasattr(self.inner.qkv, "weight"):
                embed_dim = int(self.inner.qkv.weight.shape[1])
            if embed_dim is not None and self.num_heads:
                self.head_dim = max(1, int(embed_dim // self.num_heads))
        self.scale = getattr(self.inner, "scale", 1.0 if self.head_dim is None else 1.0 / (self.head_dim ** 0.5))
        self.attn_drop = getattr(self.inner, "attn_drop", nn.Identity())
        self.proj = getattr(self.inner, "proj", nn.Identity())
        self.proj_drop = getattr(self.inner, "proj_drop", nn.Identity())

        self.has_fused_qkv = hasattr(self.inner, "qkv")
        if not self.has_fused_qkv:
            if not (hasattr(self.inner, "q") and hasattr(self.inner, "k") and hasattr(self.inner, "v")):
                raise RuntimeError("OursAttention: inner_attn must expose qkv or q/k/v projections")

    def forward(self, x: torch.Tensor, attn_mask=None, layer_idx: int = 0, total_layers: int = 12, **unused) -> torch.Tensor:
        B, T, C = x.shape

        # Disabled guard: identity path if keep≈1 or r<=0
        if self._disabled(T, layer_idx, total_layers):
            return self.inner(x)

        # Build q,k,v: [B,H,T,Hd]
        q, k, v = self._qkv(x)

        # Metric for selection
        if self.match_feature == "k":
            metric = F.normalize(k, dim=-1)                           # [B,H,T,Hd]
        else:
            metric = F.normalize(x, dim=-1).unsqueeze(1)              # [B,1,T,C]

        # Size tracking (start from ones)
        size = x.new_ones(B, T)

        # Decide per-block keep and r
        keep_ratio, r_block = self._decide_keep_r(T, layer_idx, total_layers)
        if r_block <= 0 or keep_ratio >= 1.0:
            return self.inner(x)

        # 1) current token length and K (number of tokens to keep)
        n_tokens = x.size(1)
        K = max(1, n_tokens - int(r_block))  # ToMe-style r_block -> kept-count

        # 2) build head-profile phi: [B, T, H]
        if self.match_feature == "k":
            # k: [B,H,N,d] → per-head norm → φ: [B,N,H]
            h_mag = torch.linalg.norm(k, dim=-1)  # [B,H,N]
            phi = F.normalize(h_mag.transpose(1, 2).contiguous(), dim=-1)  # [B,N,H]
        else:
            # x: [B,N,C] → scalar profile per token → [B,N,1]
            phi = torch.linalg.norm(F.normalize(x, dim=-1), dim=-1, keepdim=True)

        # ---- SELECTOR: keep & assignment (absolute token indices) ----
        keep_abs, assign_abs = self._selector(phi, K)

        # Map absolute kept indices -> compact [0..K-1] for merging
        K = int(keep_abs.shape[1])
        idx_map = x.new_full((B, T), -1, dtype=torch.long)
        for b in range(B):
            idx_map[b, keep_abs[b]] = torch.arange(K, device=x.device, dtype=torch.long)
        assign_idx = torch.gather(idx_map, 1, assign_abs)            # [B,T] in [0..K-1]

        # ---- MERGE ----
        # Value: size-weighted average; also returns merged sizes [B,K]
        v_m, size_m = size_weighted_merge_v(v, size, assign_idx)     # [B,H,K,Hd], [B,K]

        # Key/Query: either mean (kv) or gather (v-only)
        if self.merge_space == "kv":
            k_m = mean_merge_k(k, assign_idx)                        # [B,H,K,Hd]
            q_m = mean_merge_k(q, assign_idx)                        # [B,H,K,Hd]
        else:
            k_m = torch.stack([k[b, :, keep_abs[b]] for b in range(B)], dim=0)  # [B,H,K,Hd]
            q_m = torch.stack([q[b, :, keep_abs[b]] for b in range(B)], dim=0)

        # Optional pre/post normalization
        if self.l2_clip_tau > 0.0:
            l2_clip_(v_m, self.l2_clip_tau)
            l2_clip_(k_m, self.l2_clip_tau)

        # Push-lite (light correction; if no reference delta, skip by passing None)
        v_m = apply_pushlite(
            v_merged=v_m, v_ref=None,
            alpha=self.alpha, beta0=self.beta0,
            size_merged=size_m, size_eta=self.temp_eta, top_r=self.top_r
        )

        # Optional size-adaptive temperature scaling (divide q magnitude)
        if self.temp_eta != 0.0:
            factor = apply_size_temperature(
                scale=torch.ones(B, K, device=x.device), size=size_m, eta=self.temp_eta
            )  # [B,K]
            q_m = q_m / factor.view(B, 1, K, 1).clamp_min(1e-12)

        # ---- ATTENTION on reduced tokens ----
        y_red = self._attend(q_m, k_m, v_m)                          # [B,K,C]

        # ---- UNMERGE back to T ----
        scatter_idx = build_unmerge_map(assign_idx)                   # [B,T,1]
        y_full = apply_unmerge(y_red, scatter_idx)                    # [B,T,C]
        return y_full

    # ---------- helpers ----------

    def _disabled(self, T: int, layer_idx: int, total_layers: int) -> bool:
        """Return True if reduction is disabled by keep/r settings."""
        if self.keep_str is not None:
            ks = [float(s.strip()) for s in str(self.keep_str).split(",") if s.strip() != ""]
            if len(ks) == 0:
                return True
            if len(ks) == 1:
                keep = ks[0]
            elif layer_idx < len(ks):
                keep = ks[layer_idx]
            else:
                keep = ks[-1]
            if keep >= 0.999:
                return True
        if self.keep_str is None and self.r_global <= 0:
            return True
        return False

    def _decide_keep_r(self, T: int, layer_idx: int, total_layers: int) -> Tuple[float, int]:
        """Choose keep ratio and r for this block."""
        if self.keep_str is not None:
            keep, r_blk = self._scheduler(
                T=T, layer_idx=layer_idx, total_layers=total_layers,
                cfg=self.cfg, class_token=self.has_cls, distill_token=self.has_dist
            )
            return keep, r_blk
        # r-driven mode (cap at 50% merges while protecting special tokens)
        protected = (1 if self.has_cls else 0) + (1 if self.has_dist else 0)
        max_r = max(0, (T - protected) // 2)
        r_blk = min(max(0, int(self.r_global)), max_r)
        keep = float(max(1, T - r_blk)) / float(max(1, T))
        return keep, r_blk

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q,k,v as [B,H,T,Hd]."""
        B, T, C = x.shape
        H = self.num_heads if self.num_heads else 1
        Hd = self.head_dim if self.head_dim is not None else max(1, C // H)
        if self.has_fused_qkv:
            qkv = self.inner.qkv(x)                                   # [B, T, 3*C]
            qkv = qkv.reshape(B, T, 3, H, Hd).permute(2, 0, 3, 1, 4)  # [3,B,H,T,Hd]
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = self.inner.q(x).view(B, T, H, Hd).permute(0, 2, 1, 3)
            k = self.inner.k(x).view(B, T, H, Hd).permute(0, 2, 1, 3)
            v = self.inner.v(x).view(B, T, H, Hd).permute(0, 2, 1, 3)
        return q, k, v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run attention on reduced tokens; returns [B,K,C] after proj."""
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v)                                     # [B,H,K,Hd]
        y = y.transpose(1, 2).contiguous().view(y.shape[0], y.shape[2], -1)  # [B,K,C]
        y = self.proj(y)
        y = self.proj_drop(y)
        return y
