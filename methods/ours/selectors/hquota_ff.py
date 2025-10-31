# methods/ours/selectors/hquota_ff.py
# Head-diversity selector: hub-quota seeds + farthest-first completion.
# Inputs:
#   metric:  [B, H, T, D]  (e.g., normalized K per head; if you pass [B,T,C], unsqueeze to [B,1,T,C])
#   size:    [B, T] or None (token sizes; default = 1)
#   r_block: int (number of tokens to merge in this block)
#   q:       float in (0,1], fraction of seeds from hub quota
#   gamma:   float, hub score exponent for size (score *= size^gamma)
#   cls_protect: bool (protect token index 0)
#   cand_extra: int, optional extra candidate pool for speed/quality trade-off
# Returns:
#   keep_idx:   LongTensor [B, K] (K = T - r_block), sorted by construction (CLS first if protected)
#   assign_idx: LongTensor [B, T] mapping each token -> its kept representative (self for kept)
#
# Notes:
# - Greedy farthest-first is O(K*T*H) in worst case; implemented in PyTorch with simple loops over K.
# - This is a selector only; pairing/merge is handled elsewhere.

from typing import Tuple
import torch
import torch.nn.functional as F
from . import register_selector


def _ensure_shapes(metric: torch.Tensor, size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # metric: [B,H,T,D] required
    if metric.dim() == 3:  # [B,T,C] -> [B,1,T,C]
        metric = metric.unsqueeze(1)
    assert metric.dim() == 4, "metric must be [B,H,T,D] or [B,T,C]"
    B, H, T, D = metric.shape
    if size is None:
        size = metric.new_ones(B, T)
    assert size.shape == (B, T), "size must be [B,T]"
    return metric, size


def _head_signature(metric: torch.Tensor) -> torch.Tensor:
    # phi: per-token head profile using L2 norm across D: [B,H,T,D] -> [B,T,H]
    # normalize per-head then across heads for stability.
    B, H, T, D = metric.shape
    m = torch.norm(metric, dim=-1)           # [B,H,T]
    m = F.normalize(m, dim=2, eps=1e-6)      # per-head token norm
    phi = m.transpose(1, 2).contiguous()     # [B,T,H]
    phi = F.normalize(phi, dim=2, eps=1e-6)  # across-head normalization
    return phi


def _strength(phi: torch.Tensor, size: torch.Tensor, gamma: float) -> torch.Tensor:
    # global hub strength with optional size^gamma
    # phi: [B,T,H], size: [B,T]
    s = torch.sum(phi, dim=2)  # [B,T]
    if gamma != 0.0:
        s = s * torch.clamp(size, min=1e-6).pow(gamma)
    return s


def _farthest_first(phi: torch.Tensor,
                    seeds: torch.Tensor,
                    K: int,
                    protect0: bool) -> torch.Tensor:
    # Greedy farthest-first in cosine distance on phi.
    # phi: [B,T,H], seeds: [B,S], return keep_idx [B,K] (CLS at front if protect0)
    B, T, H = phi.shape
    device = phi.device
    kept = torch.full((B, K), -1, dtype=torch.long, device=device)

    # Init kept with seeds (cap at K)
    S = min(seeds.shape[1], K)
    if S > 0:
        kept[:, :S] = seeds[:, :S]

    # Precompute cosine similarities between all tokens
    # sim[b, i, j] = cos(phi[b,i], phi[b,j])
    # We incrementally track min distance to current kept set.
    phi_b = phi  # [B,T,H]
    sim = torch.matmul(phi_b, phi_b.transpose(1, 2))  # [B,T,T]
    sim = torch.clamp(sim, -1.0, 1.0)
    dist = 1.0 - sim  # cosine distance

    # Initialize current min distance to inf; then update with seeds
    cur_min = torch.full((B, T), float("inf"), device=device)
    if S > 0:
        for s_idx in range(S):
            k_idx = kept[:, s_idx]  # [B]
            # gather distance to this kept token: dist[b, :, k_idx[b]]
            d = torch.stack([dist[b, :, k_idx[b]] for b in range(B)], dim=0)  # [B,T]
            cur_min = torch.minimum(cur_min, d)

    # If protecting CLS(0), mark it as already kept if not present and reserve slot 0
    if protect0:
        # If CLS not in first S, ensure it is included and positioned at 0.
        need_cls = (kept[:, :S] != 0).all(dim=1) if S > 0 else torch.ones(B, dtype=torch.bool, device=device)
        for b in range(B):
            if need_cls[b]:
                # shift right if needed
                if S < K:
                    if S > 0:
                        kept[b, 1:S+1] = kept[b, 0:S]
                    kept[b, 0] = 0
                    S = min(S + 1, K)
                else:
                    kept[b, 0] = 0
        # update cur_min with CLS
        d_cls = dist[:, :, 0]  # [B,T]
        cur_min = torch.minimum(cur_min, d_cls)

    # Greedy fill remaining
    start = S
    for tpos in range(start, K):
        # pick argmax of current min distance (farthest from kept set)
        nxt = torch.argmax(cur_min, dim=1)  # [B]
        kept[:, tpos] = nxt
        # update cur_min with new kept
        d_new = torch.stack([dist[b, :, nxt[b]] for b in range(B)], dim=0)  # [B,T]
        cur_min = torch.minimum(cur_min, d_new)

    # Deduplicate (rare edge): enforce uniqueness by stable set semantics
    for b in range(B):
        uniq, idx = torch.unique(kept[b], sorted=True, return_inverse=False, return_counts=False)
        if uniq.numel() < K:
            # fill missing by highest-strength non-selected
            mask = torch.ones(T, dtype=torch.bool, device=device)
            mask[uniq] = False
            extra = torch.nonzero(mask, as_tuple=False).squeeze(1)
            fill = extra[: (K - uniq.numel())]
            kept[b, :uniq.numel()] = uniq
            if fill.numel() > 0:
                kept[b, uniq.numel():K] = fill
        else:
            kept[b] = uniq[:K]
    return kept


def _assign_to_nearest(phi: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    # Assign every token to its nearest kept token in cosine distance.
    # phi: [B,T,H], keep_idx: [B,K] -> assign_idx: [B,T]
    B, T, H = phi.shape
    K = keep_idx.shape[1]
    device = phi.device
    # gather kept embeddings: [B,K,H]
    kept_phi = torch.stack([phi[b, keep_idx[b]] for b in range(B)], dim=0)
    # sim: [B,T,K]
    sim = torch.matmul(phi, kept_phi.transpose(1, 2))
    sim = torch.clamp(sim, -1.0, 1.0)
    assign = torch.argmax(sim, dim=2)  # [B,T], indices in [0..K-1]
    # map to kept token indices
    mapped = torch.stack([keep_idx[b, assign[b]] for b in range(B)], dim=0)  # [B,T]
    return mapped


@register_selector("hquota_ff")
def select_hquota_ff(metric: torch.Tensor,
                     size: torch.Tensor,
                     r_block: int,
                     q: float = 0.3,
                     gamma: float = 0.0,
                     cls_protect: bool = True,
                     cand_extra: int = 128):
    metric, size = _ensure_shapes(metric, size)          # metric: [B,H,T,D], size: [B,T]
    B, H, T, D = metric.shape
    K = max(0, T - int(r_block))
    if K >= T or K <= 0:
        # trivial: keep all or keep none (guard)
        keep_idx = torch.arange(T, device=metric.device).unsqueeze(0).expand(B, T)
        assign_idx = keep_idx.clone()
        if K < T:
            keep_idx = keep_idx[:, :K]
            assign_idx = assign_idx[:, :K]  # not used when K==0
        return keep_idx, assign_idx

    # Build head-signature
    phi = _head_signature(metric)                        # [B,T,H]
    strength = _strength(phi, size, gamma)               # [B,T]

    # Seed pool (hub quota)
    S = max(1, int(q * K))
    # Always include CLS if protected
    if cls_protect and T > 0:
        # Ensure CLS is in seed set by boosting its score
        strength = strength.clone()
        strength[:, 0] = strength[:, 0] + 1e6

    # top-S seeds per batch
    topv, topi = torch.topk(strength, k=S, dim=1, largest=True, sorted=True)  # [B,S]
    seeds = topi

    # Farthest-first to complete K
    keep_idx = _farthest_first(phi, seeds, K, protect0=cls_protect)  # [B,K]

    # Build assignment for all tokens to nearest kept
    assign_idx = _assign_to_nearest(phi, keep_idx)       # [B,T]

    return keep_idx, assign_idx
