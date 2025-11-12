# methods/ours/attn.py
# Python 3.9 compatible. Comments in English only.

from typing import Optional, Tuple, List, Dict, Any, Callable
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- external deps (safe imports) -------------------------------------------
try:
    from core import token_stats as tstats
except Exception:
    tstats = None

try:
    from core.measure import BlockMeter
except Exception:
    BlockMeter = None  # meter will be optional

try:
    # Selector (drop-r + vectorized head-quota)
    from .selectors.hquota_ff import select_hquota_ff
except Exception:
    def select_hquota_ff(phi, K, quota_frac=0.0, cand_extra=0, force_k=False,
                         cls_protect=True, scores=None, mix_alpha=0.5,
                         select_mode="keep"):
        base = phi.norm(p=2, dim=-1)
        if scores is not None:
            base = 0.5 * base + 0.5 * scores
        topk = min(K, base.shape[1])
        return torch.topk(base, k=topk, dim=-1, largest=True)[1]

try:
    # Merges
    from .merges.kv_merge import size_weighted_merge_v, size_weighted_merge_kv, apply_unmerge
    _HAS_EXT_MERGE = True
except Exception:
    _HAS_EXT_MERGE = False
    def size_weighted_merge_v(x, keep_idx, assign_idx, alpha=0.0, size_delta=0.0):
        # Fallback: simple mean merge
        B, T, C = x.shape
        K = int(keep_idx.shape[1])
        out = x.new_zeros((B, K, C))
        sizes = x.new_zeros((B, K))
        for b in range(B):
            idx = assign_idx[b]
            out_b = out[b]
            sizes_b = sizes[b]
            out_b.index_add_(0, idx, x[b])
            sizes_b.index_add_(0, idx, torch.ones(T, dtype=x.dtype, device=x.device))
            out[b] = out_b / sizes_b.clamp_min(1.0).unsqueeze(-1)
        info = {"assign_idx": assign_idx, "keep_idx": keep_idx, "sizes": sizes}
        return out, info
    def apply_unmerge(x_merged, keep_idx, assign_idx, info=None):
        return x_merged


# ============================ Ours core (reducer) ============================

class OursAttention(nn.Module):
    """
    Ours reduction stage to be applied between Attention and MLP (in-block).
    Training-free; supports token-cap semantics and per-layer r schedule.
    """

    def __init__(
        self,
        *,
        token_cap: str = "on",              # "on": allow <r; "off": force exactly r
        debug_token_stats: bool = False,
        tau_adapt: bool = True,             # reserved for advanced selectors
        quota_frac: float = 0.30,
        cand_extra: int = 16,
        merge_mode: str = "v",              # "v" or "kv"
        alpha: float = 0.15,
        size_delta: float = 0.0,
        select_mode: str = "keep"           # "keep" or "drop"
    ) -> None:
        super().__init__()
        self.token_cap = str(token_cap).lower()
        self.debug = bool(debug_token_stats)
        self.tau_adapt = bool(tau_adapt)
        self.quota_frac = float(quota_frac)
        self.cand_extra = int(cand_extra)
        self.merge_mode = str(merge_mode).lower()
        self.alpha = float(alpha)
        self.size_delta = float(size_delta)
        self.select_mode = "drop" if str(select_mode).lower() == "drop" else "keep"
        # Always preserve CLS
        self.cls_protect = True

    # ---- assignment by cosine ------------------------------------------------
    def _build_assignment(self, feat: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
        """
        Map each token to nearest kept center (cosine). feat: [B,T,H], keep_idx: [B,K] (long).
        Returns assign_idx: [B,T] with values in [0..K-1].
        """
        B, T, H = feat.shape
        K = keep_idx.shape[1]
        device = feat.device
        # Gather kept features
        gather_idx = keep_idx.unsqueeze(-1).expand(B, K, H)              # [B,K,H]
        kept_feat = torch.gather(feat, 1, gather_idx)                     # [B,K,H]
        feat_n = F.normalize(feat, dim=-1)                                # [B,T,H]
        kept_n = F.normalize(kept_feat, dim=-1)                           # [B,K,H]
        # Similarity [B,T,K]
        sim = torch.einsum("bth,bkh->btk", feat_n, kept_n)
        assign_idx = torch.argmax(sim, dim=-1)                            # [B,T]
        # ensure centers map to themselves
        for b in range(B):
            for k in range(K):
                center_tok = int(keep_idx[b, k].item())
                if 0 <= center_tok < T:
                    assign_idx[b, center_tok] = k
        return assign_idx

    # ---- forward -------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                    # [B, T, C] (post-Attention)
        *,
        layer_idx: int,
        requested_r: Optional[int],
        scores: Optional[torch.Tensor] = None,   # optional external strength [B,T]
        enable_unmerge: bool = False,
        meter: Optional["BlockMeter"] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        t_pre = T
        t0 = time.perf_counter()

        # Base scores (fallback)
        if scores is None:
            scores = x.pow(2).sum(dim=-1)  # [B,T]

        # Feature for selection/assignment (unit-normalized)
        feat = F.normalize(x, dim=-1)      # [B,T,C] treated as [B,T,H]

        # Target K from requested_r
        if isinstance(requested_r, int) and requested_r > 0:
            K_target = max(1, T - int(requested_r))
        else:
            K_target = T  # no reduction on this layer

        # Select kept indices
        keep_idx = select_hquota_ff(
            phi=feat,
            K=K_target,
            quota_frac=self.quota_frac,
            cand_extra=self.cand_extra,
            force_k=(self.token_cap == "off"),
            cls_protect=True,
            scores=scores,
            mix_alpha=0.5,
            select_mode=self.select_mode
        )  # [B,K_eff]
        if keep_idx.ndim != 2:
            keep_idx = keep_idx.view(B, -1)
        K_found = int(keep_idx.shape[1])

        # Assign each token to nearest kept token
        assign_idx = self._build_assignment(feat, keep_idx)  # [B,T]

        # Merge (V-merge default; KV when external keys available and requested)
        if self.merge_mode == "kv":
            # No key/value tensors wired here; fall back to V-merge for safety
            x_merged, size_info = size_weighted_merge_v(
                x=x, keep_idx=keep_idx, assign_idx=assign_idx, alpha=self.alpha, size_delta=self.size_delta
            )
        else:
            x_merged, size_info = size_weighted_merge_v(
                x=x, keep_idx=keep_idx, assign_idx=assign_idx, alpha=self.alpha, size_delta=self.size_delta
            )

        # Optional unmerge (rare for in-block usage)
        if enable_unmerge:
            try:
                x_out = apply_unmerge(x_merged, keep_idx, assign_idx, size_info)
            except Exception:
                x_out = x_merged
        else:
            x_out = x_merged

        # Stats / meter
        t_post = int(x_out.shape[1])
        ms = (time.perf_counter() - t0) * 1000.0

        if self.debug:
            print(f"[Ours][L{layer_idx}] before={t_pre}, after_merge={x_merged.shape[1]}, after_unmerge={t_post}, req_r={requested_r}, kept={K_found}")

        if tstats is not None:
            try:
                tstats.record(
                    layer_idx=layer_idx,
                    before_len=t_pre,
                    after_merge_len=int(x_merged.shape[1]),
                    after_unmerge_len=t_post,
                    requested_r=int(requested_r) if isinstance(requested_r, int) else None
                )
            except Exception:
                pass

        if (meter is not None) and (hasattr(meter, "add")):
            try:
                meter.add(layer=layer_idx, t_pre=t_pre, t_post=t_post, ms=ms)
            except Exception:
                pass

        return x_out


# ============================ In-block wrapper ===============================

class OursBlockWrapper(nn.Module):
    """
    Insert Ours reduction BETWEEN Attention and MLP in a timm ViT block:
      x = x + drop(attn(norm1(x)))
      x = ours.reduce(x)             # <--- here (token reduction)
      x = x + drop(mlp(norm2(x)))
    This wrapper assumes the block has attributes: norm1, attn, norm2, mlp,
    and optionally drop_path, ls1/ls2 or gamma_{1,2}/gamma{1,2}.
    """

    def __init__(
        self,
        orig_block: nn.Module,
        *,
        layer_idx: int,
        reducer: OursAttention,
        # reducer knobs
        token_cap: str = "on",
        debug_token_stats: bool = False,
        tau_adapt: bool = True,
        quota_frac: float = 0.30,
        cand_extra: int = 16,
        merge_mode: str = "v",
        alpha: float = 0.15,
        size_delta: float = 0.0,
        # schedule
        get_r_for_layer: Optional[Callable[[int], Optional[int]]] = None,
        # runtime
        enable_unmerge: bool = False,
        meter: Optional["BlockMeter"] = None
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.reducer = reducer
        self.token_cap = token_cap
        self.debug = bool(debug_token_stats)
        self.tau_adapt = bool(tau_adapt)
        self.quota_frac = float(quota_frac)
        self.cand_extra = int(cand_extra)
        self.merge_mode = str(merge_mode).lower()
        self.alpha = float(alpha)
        self.size_delta = float(size_delta)
        self.get_r_for_layer = get_r_for_layer
        self.enable_unmerge = bool(enable_unmerge)
        self.meter = meter

        # Capture original submodules
        self.norm1 = orig_block.norm1
        self.attn = orig_block.attn
        self.norm2 = orig_block.norm2
        self.mlp = orig_block.mlp
        self.drop_path = getattr(orig_block, "drop_path", nn.Identity())

        # Optional layer-scale variants
        self.ls1 = getattr(orig_block, "ls1", None)
        self.ls2 = getattr(orig_block, "ls2", None)
        self.gamma_1 = getattr(orig_block, "gamma_1", None)
        self.gamma_2 = getattr(orig_block, "gamma_2", None)
        self.gamma1 = getattr(orig_block, "gamma1", None)
        self.gamma2 = getattr(orig_block, "gamma2", None)

    def _apply_layer_scale1(self, x: torch.Tensor) -> torch.Tensor:
        if self.ls1 is not None:
            return self.ls1(x)
        if self.gamma_1 is not None:
            return self.gamma_1 * x
        if self.gamma1 is not None:
            return self.gamma1 * x
        return x

    def _apply_layer_scale2(self, x: torch.Tensor) -> torch.Tensor:
        if self.ls2 is not None:
            return self.ls2(x)
        if self.gamma_2 is not None:
            return self.gamma_2 * x
        if self.gamma2 is not None:
            return self.gamma2 * x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention branch
        residual = x
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x1 = self._apply_layer_scale1(x1)
        x1 = self.drop_path(x1)
        x = residual + x1

        # Ours reduction (between Attn and MLP)
        req_r = self.get_r_for_layer(self.layer_idx) if self.get_r_for_layer is not None else None
        x = self.reducer(
            x,
            layer_idx=self.layer_idx,
            requested_r=req_r,
            scores=None,  # you can pass attn key-norms here if you have them
            enable_unmerge=self.enable_unmerge,
            meter=self.meter
        )

        # MLP branch (on reduced tokens)
        residual2 = x
        x2 = self.norm2(x)
        x2 = self.mlp(x2)
        x2 = self._apply_layer_scale2(x2)
        x2 = self.drop_path(x2)
        x = residual2 + x2

        return x


# ========================== Installer for wrappers ===========================

def _find_blocks(model: nn.Module):
    for name in ["blocks", "stages"]:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def apply_ours_inblock(
    model: nn.Module,
    *,
    r: int,
    r_list: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    token_cap: str = "on",
    debug_token_stats: bool = False,
    tau_adapt: bool = True,
    enable_unmerge: bool = False,
    selector: str = "hquota_ff",
    hq_q: float = 0.30,
    cand_extra: int = 16,
    merges: str = "v",
    alpha: float = 0.15,
    size_delta: float = 0.0,
    match_feature: str = "xnorm",   # reserved (keys vs xnorm)
    prop_attn: bool = False,        # reserved (not used in this minimal file)
    select_mode: str = "keep"       # "keep" or "drop"
) -> nn.Module:
    """
    Replace timm ViT blocks so reduction happens in-block (Attn -> Ours -> MLP).
    Supports per-layer r_list schedule; if absent, uses scalar r and layers set.
    """
    blocks = _find_blocks(model)
    if blocks is None:
        if debug_token_stats:
            print("[Ours][WARN] no transformer blocks found; in-block wiring skipped.")
        return model

    # Build per-layer r map
    n_blocks = len(list(blocks))
    if r_list is not None and len(r_list) > 0:
        if len(r_list) < n_blocks:
            last = int(r_list[-1])
            r_map = [int(v) for v in r_list] + [last] * (n_blocks - len(r_list))
        else:
            r_map = [int(v) for v in r_list[:n_blocks]]
    else:
        target = set(layers) if layers is not None else None
        r_map = [int(r) if (target is None or i in target) else 0 for i in range(n_blocks)]

    # Shared reducer instance
    reducer = OursAttention(
        token_cap=token_cap,
        debug_token_stats=debug_token_stats,
        tau_adapt=tau_adapt,
        quota_frac=hq_q,
        cand_extra=cand_extra,
        merge_mode=merges,
        alpha=alpha,
        size_delta=size_delta,
        select_mode=select_mode
    )

    # Shared meter
    meter = BlockMeter() if BlockMeter is not None else None

    # Closure for per-layer r
    def _r_for_layer(L: int) -> Optional[int]:
        v = r_map[L] if (0 <= L < len(r_map)) else 0
        return v if v > 0 else None

    # Wrap each block
    new_blocks = []
    for i, blk in enumerate(list(blocks)):
        if r_map[i] > 0:
            wrapped = OursBlockWrapper(
                orig_block=blk,
                layer_idx=i,
                reducer=reducer,
                token_cap=token_cap,
                debug_token_stats=debug_token_stats,
                tau_adapt=tau_adapt,
                quota_frac=hq_q,
                cand_extra=cand_extra,
                merge_mode=merges,
                alpha=alpha,
                size_delta=size_delta,
                get_r_for_layer=_r_for_layer,
                enable_unmerge=enable_unmerge,
                meter=meter
            )
            new_blocks.append(wrapped)
        else:
            new_blocks.append(blk)

    # Reassign back
    if isinstance(blocks, nn.Sequential):
        model.blocks = nn.Sequential(*new_blocks)  # type: ignore
    else:
        try:
            for i, w in enumerate(new_blocks):
                blocks[i] = w  # type: ignore
        except Exception:
            for i, w in enumerate(new_blocks):
                setattr(blocks, str(i), w)

    # Expose meter for final summary
    if meter is not None:
        setattr(model, "_ours_meter", meter)

    return model
