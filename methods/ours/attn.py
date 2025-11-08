# methods/ours/attn.py
# Python 3.9 compatible. Comments in English only.

from typing import Optional, Tuple, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- project deps ----------------------------------

try:
    from core import token_stats as tstats
except Exception:
    tstats = None

# Selector (O-2)
from .selectors.hquota_ff import select_hquota_ff

# Merge backends (O-3)
try:
    from .merges.kv_merge import size_weighted_merge_v as ext_size_weighted_merge_v  # type: ignore
    from .merges.kv_merge import size_weighted_merge_kv as ext_size_weighted_merge_kv  # type: ignore
    from .merges.kv_merge import apply_unmerge as ext_apply_unmerge  # type: ignore

    _HAS_EXT_MERGE = True
except Exception:
    _HAS_EXT_MERGE = False


# ----------------------------- OursAttention ---------------------------------

class OursAttention(nn.Module):
    """
    Ours token reduction (training-free).
    Designed to be called BETWEEN Attention and MLP (in-block).
    """

    def __init__(
            self,
            token_cap: str = "on",  # "on": allow <r; "off": force exactly r
            debug_token_stats: bool = False,
            tau_adapt: bool = True,  # reserved for future threshold relaxation
            enable_unmerge: bool = False,  # keep False for in-block mode (O-1)
            # selector knobs
            selector: str = "hquota_ff",
            hq_quota: float = 0.0,
            cand_extra: int = 0,
            # merge knobs
            merge_mode: str = "v",  # "v" or "kv" (kv requires keys/values)
            alpha: float = 0.0,  # center bias: out=(1-alpha)*mean+alpha*center
            beta0: float = 0.0,  # reserved
            top_r: int = 0,  # reserved
            l2_clip_tau: float = 0.0,
            temp_eta: float = 1.0,
            size_delta: float = 0.0,  # reserved for custom size weighting
            match_feature: str = "xnorm"  # "xnorm" (default) or "k" if keys are passed as phi
    ) -> None:
        super().__init__()
        self.token_cap = str(token_cap).lower()
        self.debug_token_stats = bool(debug_token_stats)
        self.tau_adapt = bool(tau_adapt)
        self.enable_unmerge = bool(enable_unmerge)

        self.selector = str(selector)
        self.hq_quota = float(hq_quota)
        self.cand_extra = int(cand_extra)

        self.merge_mode = str(merge_mode).lower()
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.top_r = int(top_r)
        self.l2_clip_tau = float(l2_clip_tau)
        self.temp_eta = float(temp_eta)
        self.size_delta = float(size_delta)
        self.match_feature = str(match_feature)

        # Always preserve CLS
        self.cls_protect = True

        # Cache last log-sizes for proportional attention (O-4)
        self.last_log_sizes = None  # type: Optional[torch.Tensor]

    # ---- public API ----

    def forward(
            self,
            x: torch.Tensor,  # [B, T, C]  (post-Attention, pre-MLP)
            *,
            layer_idx: int,
            requested_r: Optional[int],
            phi: Optional[torch.Tensor] = None,  # [B, T, H] optional features (e.g., keys)
            scores: Optional[torch.Tensor] = None,  # [B, T] optional strength
            return_info: bool = False
    ):
        if x.dim() != 3:
            return (x, {}) if return_info else x
        B, T, C = int(x.size(0)), int(x.size(1)), int(x.size(2))
        before_len = T

        # 1) Build selection features and base scores
        feat, base_scores = self._build_features(x, phi, scores)  # feat: [B,T,Hf], base_scores: [B,T]

        # 2) Determine target keep count
        if requested_r is None:
            K = T
        else:
            K = max(1, T - int(requested_r))

        # 3) Select keep indices (CLS reserved, token-cap respected)
        force_exact = (self.token_cap == "off")
        if self.selector != "hquota_ff":
            # fallback: use hquota_ff anyway
            pass

        keep_idx = select_hquota_ff(
            phi=feat,
            K=K,
            quota_frac=self.hq_quota,
            cand_extra=max(0, self.cand_extra),
            force_k=force_exact,
            cls_protect=True,
            scores=base_scores,
            mix_alpha=0.5
        )  # [B, K_eff]
        K_found = int(keep_idx.size(1)) if keep_idx.numel() > 0 else 0
        if K_found == 0:
            # Should not happen, but be safe
            keep_idx = torch.zeros((B, 1), dtype=torch.long, device=x.device)
            K_found = 1

        # 4) Build assignment of all tokens to nearest kept center (cosine)
        assign_idx = self._build_assignment(feat, keep_idx)  # [B, T] in [0..K_found-1]

        # 5) Merge tokens (V-merge by default, size-weighted backend if available)
        if K_found == T:
            x_merged = x
            size_info = {"assign_idx": assign_idx, "keep_idx": keep_idx, "sizes": x.new_full((B, T), 1.0)}
        else:
            if _HAS_EXT_MERGE and self.merge_mode == "v":
                try:
                    x_merged, size_info = ext_size_weighted_merge_v(
                        x=x, keep_idx=keep_idx, assign_idx=assign_idx,
                        alpha=self.alpha, size_delta=self.size_delta
                    )
                except Exception:
                    x_merged, size_info = self._merge_v_mean(x, keep_idx, assign_idx)
            elif _HAS_EXT_MERGE and self.merge_mode == "kv" and phi is not None:
                # If keys (phi) are provided and a KV merge is desired, you could pass
                # keys=phi and values=x here. Otherwise fall back to V-merge.
                try:
                    k_merged, v_merged, size_info = ext_size_weighted_merge_kv(
                        k=phi, v=x, keep_idx=keep_idx, assign_idx=assign_idx,
                        alpha=self.alpha, size_delta=self.size_delta
                    )
                    x_merged = v_merged
                except Exception:
                    x_merged, size_info = self._merge_v_mean(x, keep_idx, assign_idx)
            else:
                x_merged, size_info = self._merge_v_mean(x, keep_idx, assign_idx)

        after_merge_len = int(x_merged.size(1))

        # 6) Optional unmerge (kept off for in-block; left for completeness)
        if self.enable_unmerge and _HAS_EXT_MERGE:
            try:
                x_out = ext_apply_unmerge(x_merged, keep_idx, size_info.get("assign_idx"), size_info)
            except Exception:
                x_out = x_merged
        else:
            x_out = x_merged

        after_unmerge_len = int(x_out.size(1))

        # 7) Stats + cache sizes for proportional attention (O-4)
        sizes = size_info.get("sizes", None)
        if isinstance(sizes, torch.Tensor):
            self.last_log_sizes = torch.log(sizes.clamp_min(1.0))
        else:
            self.last_log_sizes = None

        if self.debug_token_stats:
            print(
                "[Ours]"
                + f"[L{layer_idx}] before={before_len}, "
                + f"after_merge={after_merge_len}, after_unmerge={after_unmerge_len}, "
                + f"req_r={requested_r}, kept={K_found}, cap={self.token_cap}"
            )

        if tstats is not None:
            try:
                tstats.record(
                    layer_idx=int(layer_idx),
                    before_len=before_len,
                    after_merge_len=after_merge_len,
                    after_unmerge_len=after_unmerge_len if self.enable_unmerge else after_merge_len,
                    requested_r=int(requested_r) if requested_r is not None else None
                )
            except Exception:
                pass

        if return_info:
            return x_out, size_info
        return x_out

    # ---- helpers: features, assignment, merge ----

    def _build_features(
            self,
            x: torch.Tensor,  # [B,T,C]
            phi: Optional[torch.Tensor],
            scores: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          feat: [B,T,Hf] normalized feature for cosine assignment/diversity
          base: [B,T] base strength for ranking/backfill
        """
        if (phi is not None) and phi.numel() > 0:
            feat = F.normalize(phi, dim=-1)
            base = (phi.pow(2).sum(dim=-1) + 1e-6).sqrt()
        else:
            feat = F.normalize(x, dim=-1)
            base = (x.pow(2).sum(dim=-1) + 1e-6).sqrt()

        if self.l2_clip_tau > 0.0:
            base = torch.clamp(base, max=float(self.l2_clip_tau))
        if scores is not None:
            base = 0.5 * base + 0.5 * scores
        if self.temp_eta != 1.0:
            base = base / float(self.temp_eta)
        return feat, base

    def _build_assignment(
            self,
            feat: torch.Tensor,  # [B,T,Hf], normalized
            keep_idx: torch.Tensor  # [B,K]
    ) -> torch.Tensor:
        """Assign each token to nearest kept center by cosine similarity."""
        B, T, H = int(feat.size(0)), int(feat.size(1)), int(feat.size(2))
        K = int(keep_idx.size(1))
        assign_rows: List[torch.Tensor] = []

        for b in range(B):
            centers = feat[b, keep_idx[b]]  # [K,H]
            sims = torch.matmul(feat[b], centers.t())  # [T,K]
            idx = torch.argmax(sims, dim=1)  # [T]
            # ensure centers map to themselves
            idx_scatter = idx.clone()
            for k in range(K):
                center_tok = int(keep_idx[b, k].item())
                if 0 <= center_tok < T:
                    idx_scatter[center_tok] = k
            assign_rows.append(idx_scatter)

        return torch.stack(assign_rows, dim=0)  # [B,T]

    def _merge_v_mean(
            self,
            x: torch.Tensor,  # [B,T,C]
            keep_idx: torch.Tensor,  # [B,K]
            assign_idx: torch.Tensor  # [B,T]
    ) -> Tuple[torch.Tensor, dict]:
        """Simple mean aggregation in V space as a robust fallback."""
        B, T, C = int(x.size(0)), int(x.size(1)), int(x.size(2))
        K = int(keep_idx.size(1))
        out = x.new_zeros((B, K, C))
        sizes = x.new_zeros((B, K))

        for b in range(B):
            idx = assign_idx[b]  # [T]
            out_b = out[b]  # [K,C]
            sizes_b = sizes[b]  # [K]
            out_b.index_add_(0, idx, x[b])  # sum values per cluster
            ones = torch.ones(T, dtype=x.dtype, device=x.device)
            sizes_b.index_add_(0, idx, ones)  # count per cluster
            denom = sizes_b.clamp_min(1.0).unsqueeze(-1)  # [K,1]
            out[b] = out_b / denom

        size_info = {"assign_idx": assign_idx, "keep_idx": keep_idx, "sizes": sizes}
        return out, size_info


# ---------------------- proportional attention (O-4) -------------------------

class _SizeState(object):
    """Holds per-batch token sizes for proportional attention."""

    def __init__(self) -> None:
        self.current = None  # type: Optional[torch.Tensor]  # [B, N] or None


class ProportionalSelfAttention(nn.Module):
    """
    Wrap timm Attention to add log(size) bias to key dimension of attention logits.
    If size state is missing or mismatched, falls back to zero bias.
    """

    def __init__(self, attn: nn.Module, size_state: _SizeState) -> None:
        super().__init__()
        # copy essential parts from timm Attention
        self.qkv = attn.qkv
        self.num_heads = int(attn.num_heads)
        self.scale = float(getattr(attn, "scale", 1.0))
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop
        self._size_state = size_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3*C]
        head_dim = C // self.num_heads
        scale = self.scale if self.scale is not None else (1.0 / float(head_dim) ** 0.5)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, h, N, d]

        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, h, Nq, Nk]

        # Add log-size bias along key dimension
        sizes = self._size_state.current  # [B, Nk] or None
        if isinstance(sizes, torch.Tensor) and sizes.dim() == 2 and sizes.size(0) == B and sizes.size(1) == k.size(-2):
            log_s = torch.log(sizes.clamp_min(1e-6))
            logits = logits + log_s.view(B, 1, 1, sizes.size(1))

        attn = logits.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------- in-block wiring (O-1) ----------------------------

class OursBlockWrapper(nn.Module):
    """
    ViT Block wrapper (in-block reduction):
      1) x = x + Attn(norm1(x))                 (optionally with proportional bias)
      2) x = reducer(x, r)                      [Attn -> Reduce -> MLP]
      3) x = x + MLP(norm2(x))
    """

    def __init__(
            self,
            orig_block: nn.Module,
            layer_idx: int,
            reducer: OursAttention,
            r_per_layer: Callable[[int], Optional[int]],
            size_state: "_SizeState",
            prop_attn: bool = False,
            debug: bool = False
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.reducer = reducer
        self._r_fn = r_per_layer
        self.size_state = size_state
        self.prop_attn = bool(prop_attn)
        self.debug = bool(debug)

        # keep original submodules
        self.norm1 = orig_block.norm1
        self.norm2 = orig_block.norm2
        self.mlp = orig_block.mlp
        self.drop_path = getattr(orig_block, "drop_path", nn.Identity())

        # attention (wrap for proportional attention if requested)
        self.attn = ProportionalSelfAttention(orig_block.attn, size_state) if self.prop_attn else orig_block.attn

        # optional layer-scale variants
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
        # Attention branch (may include proportional bias from previous sizes)
        residual = x
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x1 = self._apply_layer_scale1(x1)
        x1 = self.drop_path(x1)
        x = residual + x1

        # Ours reduction (between Attn and MLP)
        req_r = self._r_fn(self.layer_idx)
        x, info = self.reducer(x, layer_idx=self.layer_idx, requested_r=req_r, return_info=True)

        # Update size state for NEXT block's attention
        sizes = info.get("sizes", None) if isinstance(info, dict) else None
        if isinstance(sizes, torch.Tensor):
            self.size_state.current = sizes  # [B, K]
        else:
            self.size_state.current = None

        # MLP branch on reduced tokens
        residual2 = x
        x2 = self.norm2(x)
        x2 = self.mlp(x2)
        x2 = self._apply_layer_scale2(x2)
        x2 = self.drop_path(x2)
        x = residual2 + x2
        return x


def _find_blocks(model: nn.Module):
    for name in ["blocks", "stages"]:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def apply_ours_inblock(
        model: nn.Module,
        *,
        r: int,
        layers: Optional[List[int]],
        # reducer knobs (must match OursAttention.__init__)
        token_cap: str = "on",
        debug_token_stats: bool = False,
        tau_adapt: bool = True,
        enable_unmerge: bool = False,
        selector: str = "hquota_ff",
        hq_quota: float = 0.0,
        cand_extra: int = 0,
        merge_mode: str = "v",
        alpha: float = 0.0,
        beta0: float = 0.0,
        top_r: int = 0,
        l2_clip_tau: float = 0.0,
        temp_eta: float = 1.0,
        size_delta: float = 0.0,
        match_feature: str = "xnorm",
        prop_attn: bool = False  # O-4 toggle
) -> nn.Module:
    """
    Replace timm blocks so reduction happens between Attention and MLP.
    Optionally enable proportional attention (log-size bias) for next block.
    """
    blocks = _find_blocks(model)
    if blocks is None:
        if debug_token_stats:
            print("[Ours][WARN] no transformer blocks found; in-block wiring skipped.")
        return model

    # Single reducer reused across blocks
    reducer = OursAttention(
        token_cap=token_cap,
        debug_token_stats=debug_token_stats,
        tau_adapt=tau_adapt,
        enable_unmerge=enable_unmerge,
        selector=selector,
        hq_quota=hq_quota,
        cand_extra=cand_extra,
        merge_mode=merge_mode,
        alpha=alpha,
        beta0=beta0,
        top_r=top_r,
        l2_clip_tau=l2_clip_tau,
        temp_eta=temp_eta,
        size_delta=size_delta,
        match_feature=match_feature
    )

    # Shared size state across blocks (for proportional attention)
    size_state = _SizeState()

    target = set(layers) if layers is not None else None

    def r_for_layer(L: int) -> Optional[int]:
        if target is None:
            return int(r)
        return int(r) if L in target else None

    # Wrap each block
    new_blocks = []
    for i, blk in enumerate(list(blocks)):
        wrapped = OursBlockWrapper(
            orig_block=blk,
            layer_idx=i,
            reducer=reducer,
            r_per_layer=r_for_layer,
            size_state=size_state,
            prop_attn=bool(prop_attn),
            debug=debug_token_stats
        )
        new_blocks.append(wrapped)

    # Assign back
    if isinstance(blocks, nn.Sequential):
        model.blocks = nn.Sequential(*new_blocks)  # type: ignore
    else:
        # try ModuleList-like assignment; fallback to setattr per index-name
        try:
            for i, w in enumerate(new_blocks):
                blocks[i] = w  # type: ignore
        except Exception:
            for i, w in enumerate(new_blocks):
                setattr(blocks, str(i), w)

    return model
