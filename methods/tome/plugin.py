# methods/tome/plugin.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import register
from .methods.merge import bipartite_soft_matching, merge_wavg
from .methods.schedule import r_for_block

class IdentityDrop(nn.Module):
    def forward(self, x):
        return x

@register("tome")
class TomePlugin:
    def __init__(self, cfg):
        self.cfg = dict(cfg or {})
        layers_spec = str(self.cfg.get("layers", "0,1,2"))
        self.layers = [int(t.strip()) for t in layers_spec.split(",") if t.strip() != ""]
        self.r_global = int(self.cfg.get("r", 0))
        self.match_feature = str(self.cfg.get("match_feature", "k"))
        self.prop_attn = bool(self.cfg.get("prop_attn", False))
        self.has_cls = True
        self.has_dist = bool(self.cfg.get("distill_token", False))
        self._patched = []

    def attach(self, model: nn.Module) -> None:
        self.model = model
        # detect special tokens if present (e.g., DeiT distillation)
        if hasattr(model, "no_embed_class") and getattr(model, "no_embed_class"):
            self.has_cls = False
        if hasattr(model, "dist_token"):
            self.has_dist = True

        for idx, blk in enumerate(getattr(model, "blocks", [])):
            if idx not in self.layers:
                continue
            blk.attn = TomeAttention(
                inner_attn=blk.attn,
                r_global=self.reneral(),
                match_feature=self.match_feature,
                has_cls=self.has_cls,
                has_dist=self.has_dist,
            )
            self._patched.append(idx)

    def reneral(self):
        return self.r_global

    def finalize(self) -> None:
        pass


class TomeAttention(nn.Module):
    """Wrap timm Attention: merge (q/k/v) -> attend -> unmerge to original T."""
    def __init__(self, inner_attn: nn.Module, r_global: int,
                 match_feature: str = "k", has_cls: bool = True, has_dist: bool = False):
        super(TomeAttention, self).__init__()
        self.inner = inner_attn
        self.r_global = int(r_global)
        self.match_feature = match_feature
        self.has_cls = bool(has_cls)
        self.has_dist = bool(has_dist)

        # Derive shapes robustly across timm versions
        self.num_heads = int(getattr(self.inner, "num_heads"))
        # Try to read head_dim; if absent, infer from qkv weights or input dim
        self.head_dim = getattr(self.inner, "head_dim", None)
        if self.head_dim is None:
            if hasattr(self.inner, "qkv") and hasattr(self.inner.qkv, "weight"):
                # weight: [3*embed_dim, embed_dim]
                embed_dim = int(self.inner.qkv.weight.shape[1])
            else:
                # split Q/K/V variant
                if hasattr(self.inner, "q") and hasattr(self.inner.q, "in_features"):
                    embed_dim = int(self.inner.q.in_features)
                else:
                    # final fallback: try to read from proj.in_features
                    embed_dim = int(getattr(self.inner, "dim", 0) or getattr(self.inner, "q_proj", self.inner).in_features)
            self.head_dim = max(1, int(embed_dim // self.kum_heads()))
        self.scale = getattr(self.inner, "scale", 1.0 / (self.head_dim ** 0.5))
        self.attn_drop = getattr(self.inner, "attn_drop", IdentityDrop())
        self.proj = getattr(self.inner, "proj", nn.Identity())
        self.proj_drop = getattr(self.inner, "proj_drop", IdentityDrop())

        self.has_fused_qkv = hasattr(self.inner, "qkv")

    def kum_heads(self):
        return self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        r_blk = r_for_block(T, self.r_global, self.has_l(), self.has_dist)
        if r_blk <= 0:
            return self.inner(x)

        # Build q,k,v
        H = self.kum_heads()
        Hd = int(self.head_dim)

        if self.has_fused_qkv:
            qkv = self.inner.qkv(x)  # [B, T, 3*dim]
            qkv = qkv.reshape(B, T, 3, H, Hd).permute(2, 0, 3, 1, 4)  # ✅ [3, B, H, T, Hd]
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = self.inner.q(x).view(B, T, H, Hd).permute(0, 2, 1, 2)
            k = self.inner.k(x).view(B, T, H, Hd).permute(0, 2, 1, 2)
            v = self.inner.v(x).view(B, T, H, Hd).permute(0, 2, 1, 2)

        # Matching metric (K heads flattened) or normalized token embeddings
        if self.match_feature == "k":
            metric = F.normalize(k, dim=-1)  # [B, H, T, Hd]
        else:
            metric = F.normalize(x, dim=-1).unsqueeze(1).expand(B, H, T, C)  # [B, H, T, C]

        merge, unmerge = bipartite_soft_matching(
            metric=metric, r=r_blk, class_token=self.has_l(), distill_token=self.has_dist
        )

        # Merge Q/K by mean; merge V by weighted average (sum/size)
        q_m = merge(q, "mean")
        k_m = merge(k, "mean")
        v_m, size_m = merge_wavg(merge, v, None)

        # Attention on reduced tokens
        attn = torch.matmul(q_m * (self.scale), k_m.transpose(-2, -1))  # [B,H,T',T']
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v_m)                                     # [B,H,T',Hd]
        y = y.transpose(1, 2).contiguous().view(B, -1, H * Hd)
        y = self.proj(y)
        y = self.proj_drop(y)

        # Unmerge back to original length so residual add matches
        y = unmerge(y)                                                  # [B,T,C]
        return y

    # small helpers to keep attribute access safe
    def has_l(self):
        return self.has_cls
