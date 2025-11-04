# methods/ours/attn.py
# Python 3.9 compatible. Comments in English only.

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Optional: if your project provides these modules, they will be used.
    from core import token_stats as tstats
except Exception:
    tstats = None

try:
    from methods.ours.selectors.hquota import select_hquota_ff
except Exception:
    # Fallback stub if selector is not available; selects top-K by norm.
    def select_hquota_ff(phi, K, quota_frac=0.0, cand_extra=0, force_k=False,
                         cls_protect=True, scores=None, mix_alpha=0.5):
        base = phi.norm(p=2, dim=-1)
        if scores is not None:
            base = (1.0 - mix_alpha) * base + mix_alpha * scores
        K = min(K, base.shape[1])
        return torch.topk(base, k=K, dim=-1, largest=True)[1]


class OursAttention(nn.Module):
    """
    Token-reduction wrapper for our method with token-cap control and selector/merge hooks.

    This module is designed to drop into an existing timm ViT block pipeline via a hook
    (see OursApplier in main.py). It performs:
      1) Build selection scores from provided features or x (fallback).
      2) Run head-quota + farthest-first selector (select_hquota_ff).
      3) If token_cap == "off": enforce exact r via simple backfill to reach K = T - r.
      4) Merge by gathering kept tokens (placeholder). If your pipeline uses KV merge,
         connect your merge function in _merge_tokens().
      5) Optionally unmerge (no-op by default).
      6) Record token stats (before, after_merge, after_unmerge, requested_r).

    Notes:
      - To preserve immediate run-ability without tight coupling to attention internals,
        this implementation falls back to x-based norms when specialized features are absent.
      - You can later wire in true head-profile (phi) and value/key merges without changing
        the call sites; see TODOs in _build_scores/_merge_tokens/_unmerge_tokens.
    """

    def __init__(
        self,
        token_cap: str = "on",           # "on": allow <r; "off": force exactly r
        debug_token_stats: bool = False,
        tau_adapt: bool = True,          # kept for compatibility (used if you pass tau)
        max_tau_iters: int = 5,
        cls_protect: bool = True,
        enable_unmerge: bool = True,

        # Selector/merge knobs (kept for backward compatibility with your CLI)
        selector: str = "hquota_ff",
        hq_quota: float = 0.0,           # fraction [0,1], reserved per head
        cand_extra: int = 0,             # extra candidate pool
        merge_mode: str = "v",           # "v" or "kv" (placeholder hook)
        alpha: float = 0.0,              # merge weight param (placeholder)
        beta0: float = 0.0,              # merge cap param (placeholder)
        top_r: int = 0,                  # sparsity in merge (placeholder)
        l2_clip_tau: float = 0.0,        # pre-score clipping tau (0 disables)
        temp_eta: float = 1.0,           # temperature scaling for scores
        size_delta: float = 0.0,         # optional size-based scaling (placeholder)
        match_feature: str = "xnorm"     # "k" or "xnorm" (for future wiring)
    ) -> None:
        super().__init__()
        self.token_cap = str(token_cap).lower()
        self.debug_token_stats = bool(debug_token_stats)
        self.tau_adapt = bool(tau_adapt)
        self.max_tau_iters = int(max_tau_iters)
        self.cls_protect = bool(cls_protect)
        self.enable_unmerge = bool(enable_unmerge)

        # keep knobs for compatibility; used where applicable
        self.selector_name = str(selector)
        self.hq_quota = float(hq_quota)
        self.cand_extra = int(cand_extra)
        self.merge_mode = str(merge_mode).lower()
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.top_r = int(top_r)
        self.l2_clip_tau = float(l2_clip_tau)
        self.temp_eta = float(max(1e-6, temp_eta))
        self.size_delta = float(size_delta)
        self.match_feature = str(match_feature)

    # ----------------------------- public API ---------------------------------

    def forward(
        self,
        x: torch.Tensor,                 # [B, T, C]
        layer_idx: int,
        requested_r: Optional[int],
        **feat_kwargs                     # optional: phi, scores, tau, etc.
    ) -> torch.Tensor:
        if x.dim() != 3:
            return x

        B, T, C = x.shape
        before_len = int(T)

        # 1) Build selection features/scores (phi and scores)
        phi, scores, tau_used = self._build_scores(x, **feat_kwargs)  # phi: [B,T,H], scores: [B,T]

        # 2) Determine target keep K from requested_r
        if requested_r is None:
            K_target = T
        else:
            K_target = max(1, T - int(requested_r))

        # 3) Run selector to get keep indices
        keep_idx = self._run_selector(phi, scores, K_target)

        # 4) token-cap enforcement: force exactly r (i.e., exactly K_target kept) if off
        if self.token_cap == "off":
            keep_idx = self._backfill_to_exact_K(keep_idx, scores, K_target)

        # 5) Merge (placeholder gather-keep; wire your KV/V merge here)
        x_merged, size_info = self._merge_tokens(x, keep_idx, layer_idx)

        after_merge_len = int(x_merged.shape[1])

        # 6) Unmerge (no-op by default; wire your unmerge if needed)
        x_out = self._unmerge_tokens(x_merged, keep_idx, size_info, layer_idx)
        after_unmerge_len = int(x_out.shape[1])

        # 7) Stats and debug
        if self.debug_token_stats:
            removed_est = before_len - after_unmerge_len
            print(
                "[Ours]"
                + f"[L{layer_idx}] before={before_len}, "
                + f"after_merge={after_merge_len}, "
                + f"after_unmerge={after_unmerge_len}, "
                + f"req_r={requested_r}, "
                + f"K_target={K_target}, "
                + f"removed_final={removed_est}, "
                + f"tau={tau_used}"
            )

        if tstats is not None:
            try:
                tstats.record(
                    layer_idx=int(layer_idx),
                    before_len=before_len,
                    after_merge_len=after_merge_len,
                    after_unmerge_len=after_unmerge_len,
                    requested_r=int(requested_r) if requested_r is not None else None
                )
            except Exception:
                pass

        return x_out

    # --------------------------- feature building -----------------------------

    def _build_scores(
        self,
        x: torch.Tensor,
        **feat_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[float]]:
        """
        Build (phi, scores, tau_used) for selection.
          - If feat_kwargs provides 'phi' ([B,T,H]) or 'scores' ([B,T]), use them.
          - Otherwise, fall back to x-based features: phi from split channels, scores from ||x||.
          - Apply optional L2 clipping and temperature scaling on scores.
        """
        B, T, C = x.shape
        device = x.device

        # Use provided head-profile if available; else create a proxy from x
        phi = feat_kwargs.get("phi", None)
        if phi is None:
            H = min(8, max(1, C // max(1, (C // 8))))  # heuristic small head count
            if C % H == 0:
                x_resh = x.view(B, T, H, C // H)
                phi = x_resh.norm(p=2, dim=-1)  # [B,T,H]
            else:
                # Fallback: replicate a single norm across H=1
                phi = x.norm(p=2, dim=-1, keepdim=True)  # [B,T,1]
        else:
            # ensure float
            phi = phi.float()

        # Base scores
        scores = feat_kwargs.get("scores", None)
        if scores is None:
            scores = x.pow(2).sum(dim=-1).sqrt()  # [B,T]
        else:
            scores = scores.float()

        # Optional L2 clipping on scores
        if self.l2_clip_tau is not None and self.l2_clip_tau > 0.0:
            tau = float(self.l2_clip_tau)
            scores = torch.clamp(scores, max=tau)

        # Temperature scaling
        if self.temp_eta is not None and self.temp_eta != 1.0:
            scores = scores / float(self.temp_eta)

        tau_used = feat_kwargs.get("tau", None)
        if tau_used is not None:
            try:
                tau_used = float(tau_used)
            except Exception:
                tau_used = None

        return phi, scores, tau_used

    # ------------------------------ selection ---------------------------------

    def _run_selector(
        self,
        phi: torch.Tensor,               # [B,T,H]
        scores: torch.Tensor,            # [B,T]
        K_target: int
    ) -> torch.Tensor:
        """
        Run head-quota + farthest-first selector to get keep indices.
        """
        K_target = max(1, min(K_target, phi.shape[1]))
        keep_idx = select_hquota_ff(
            phi=phi,
            K=K_target,
            quota_frac=self.hq_quota,
            cand_extra=self.cand_extra,
            force_k=(self.token_cap == "off"),
            cls_protect=self.cls_protect,
            scores=scores,
            mix_alpha=0.5
        )
        return keep_idx

    def _backfill_to_exact_K(
        self,
        keep_idx: torch.Tensor,          # [B,K_found]
        scores: torch.Tensor,            # [B,T]
        K_target: int
    ) -> torch.Tensor:
        """
        Ensure exactly K_target kept indices by backfilling with top scores,
        excluding CLS and already chosen, batch-wise.
        """
        B, T = scores.shape
        device = scores.device
        cur_K = int(keep_idx.shape[-1])
        need = K_target - cur_K
        if need <= 0:
            return keep_idx

        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        if self.cls_protect and T > 0:
            mask[:, 0] = False

        chosen = keep_idx
        if chosen.dim() == 1:
            chosen = chosen.view(1, -1).expand(B, -1)
        for b in range(B):
            mask[b, chosen[b]] = False

        base = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
        extra_k = min(need, int(mask.sum(dim=1).min().item()))
        if extra_k <= 0:
            return keep_idx

        extra_idx = torch.topk(base, k=extra_k, dim=-1, largest=True)[1]  # [B, extra_k]
        keep_idx = torch.cat([keep_idx, extra_idx], dim=-1)
        return keep_idx

    # ------------------------------ merge/unmerge -----------------------------

    def _merge_tokens(
        self,
        x: torch.Tensor,                 # [B,T,C]
        keep_idx: torch.Tensor,          # [B,K]
        layer_idx: int
    ) -> Tuple[torch.Tensor, dict]:
        """
        Placeholder merge: gather kept tokens only.
        If you have KV/V merge with size-weighted averaging, plug it here.
        """
        B, T, C = x.shape
        K = int(keep_idx.shape[-1])
        gather = keep_idx.unsqueeze(-1).expand(B, K, C)     # [B,K,C]
        x_merged = torch.gather(x, 1, gather)               # [B,K,C]
        size_info = {}  # fill with your merge meta if needed
        return x_merged, size_info

    def _unmerge_tokens(
        self,
        x_merged: torch.Tensor,
        keep_idx: torch.Tensor,
        size_info: dict,
        layer_idx: int
    ) -> torch.Tensor:
        """
        No-op unmerge by default. If your pipeline supports unmerge (e.g., for attention),
        connect it here using size_info and keep_idx.
        """
        if not self.enable_unmerge:
            return x_merged
        # Default behavior: return merged representation without restoring length.
        return x_merged
