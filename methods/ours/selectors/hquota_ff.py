# methods/ours/selectors/hquota.py
# Python 3.9 compatible. Comments in English only.
# Step-3: Head-quota vectorization + incremental farthest-first + fast drop-r mode.

from typing import Optional, List
import math
import torch
import torch.nn.functional as F


# ------------------------------ small utils ----------------------------------

def _l2norm_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 norm along last dimension."""
    return (x.pow(2).sum(dim=-1) + eps).sqrt()


def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Unit-normalize along given dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _dedup_preserve_order_1d(idxs: torch.Tensor) -> torch.Tensor:
    """Remove duplicates (1D LongTensor) while preserving order."""
    seen = set()
    kept: List[int] = []
    for v in idxs.tolist():
        if v not in seen:
            seen.add(v)
            kept.append(v)
    return torch.tensor(kept, dtype=idxs.dtype, device=idxs.device)


# ----------------------- incremental farthest-first (single) ------------------

def _ff_incremental_single(
    feat_b: torch.Tensor,          # [T,H] normalized features
    cand_mask_b: torch.Tensor,     # [T] bool (True=candidate)
    base_b: torch.Tensor,          # [T] base scores (for seeding/fallback)
    seed_idx: torch.Tensor,        # [S] initial kept indices (may be empty)
    need: int                      # how many to pick
) -> torch.Tensor:
    """
    Incremental farthest-first with candidate clamp and min-sim cache.
    Complexity ~ O((T_cand + need) * need * H); avoids full matmul each loop.
    """
    device = feat_b.device
    T = int(feat_b.size(0))
    if need <= 0 or T == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    cand_mask = cand_mask_b.clone()
    if seed_idx.numel() > 0:
        cand_mask[seed_idx] = False

    # If no seed, start from the strongest candidate by base score
    keep = seed_idx.clone()
    if keep.numel() == 0 and cand_mask.any():
        seed0 = int(torch.argmax(torch.where(cand_mask, base_b, torch.full_like(base_b, float("-inf")))).item())
        keep = torch.tensor([seed0], dtype=torch.long, device=device)
        cand_mask[seed0] = False

    # Pre-allocate min-sim cache for candidates
    cand_ids = torch.nonzero(cand_mask, as_tuple=False).view(-1)  # [Nc]
    if cand_ids.numel() == 0 or need <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Initialize min-sim with first center
    centers = feat_b[keep]                                  # [K0,H]
    sims0 = torch.matmul(feat_b[cand_ids], centers.t())     # [Nc,K0]
    min_sim = sims0.max(dim=1).values if sims0.numel() > 0 else torch.full((cand_ids.numel(),), 0.0, device=device)

    chosen: List[int] = []
    for _ in range(need):
        # pick farthest: smallest min_sim
        pick_local = int(torch.argmin(min_sim).item())
        pick = int(cand_ids[pick_local].item())
        chosen.append(pick)

        # remove picked from candidate set
        cand_mask[pick] = False
        # update caches incrementally with new center
        new_center = feat_b[pick:pick + 1]                  # [1,H]
        sims_new = torch.matmul(feat_b[cand_ids], new_center.t()).squeeze(-1)  # [Nc]
        min_sim = torch.minimum(min_sim, sims_new)          # update farthest cache

        # drop picked from arrays
        if cand_ids.numel() > 1:
            mask_keep = torch.ones(cand_ids.numel(), dtype=torch.bool, device=device)
            mask_keep[pick_local] = False
            cand_ids = cand_ids[mask_keep]
            min_sim = min_sim[mask_keep]
        else:
            break

    return torch.tensor(chosen, dtype=torch.long, device=device)


# -------------------------------- public API ---------------------------------

def select_hquota_ff(
    phi: torch.Tensor,                 # [B,T,H] head-profile per token
    K: int,                            # target keep count (T - r) in keep-mode
    quota_frac: float = 0.0,           # per-head reserved fraction (0~1)
    cand_extra: int = 0,               # extra candidate pool size
    force_k: bool = False,             # ensure exactly K in keep-mode (backfill)
    cls_protect: bool = True,          # always keep CLS (idx 0)
    scores: Optional[torch.Tensor] = None,  # [B,T] optional global strength
    mix_alpha: float = 0.5,            # blend weight between ||phi|| and scores
    select_mode: str = "keep"          # "keep" (default) or "drop" (fast)
) -> torch.Tensor:
    """
    Head-quota + farthest-first selector with:
      - CLS reservation (keep CLS at column-0 when protected).
      - Candidate pool clamp (K + cand_extra).
      - Incremental farthest-first (min-sim cache) for diversity.
      - Fast drop-r mode: drop r tokens, then return the complement as keep set.
    """
    assert phi.dim() == 3, "phi must be [B,T,H]"
    B, T, H = int(phi.size(0)), int(phi.size(1)), int(phi.size(2))
    device = phi.device

    if T == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Base strength and normalized features
    base = _l2norm_last(phi)  # [B,T]
    if scores is not None:
        a = float(mix_alpha)
        base = (1.0 - a) * base + a * scores
    feat = F.normalize(phi, dim=-1)  # [B,T,H]

    # ---------------- fast "drop-r" mode ----------------
    drop_mode = (str(select_mode).lower() == "drop")
    if drop_mode:
        K_eff = max(1, min(int(K), T))
        r = T - K_eff
        if r <= 0:
            # keep all (respect K_eff upper bound)
            keep_idx = torch.arange(T, device=device).view(1, T).repeat(B, 1)
            if cls_protect and T > 0:
                for b in range(B):
                    row = keep_idx[b]
                    zero_pos = (row == 0).nonzero(as_tuple=False).view(-1)
                    if zero_pos.numel() > 0 and int(zero_pos[0].item()) != 0:
                        pos = int(zero_pos[0].item())
                        row[0], row[pos] = row[pos], row[0]
            return keep_idx[:, :K_eff]

        # forbid dropping CLS
        drop_cand = torch.ones((B, T), dtype=torch.bool, device=device)
        if cls_protect and T > 0:
            drop_cand[:, 0] = False

        # smallest-by-base are dropped (vectorized)
        base_masked = torch.where(drop_cand, base, torch.full_like(base, float("+inf")))
        r_clamped = min(r, max(0, T - (1 if (cls_protect and T > 0) else 0)))
        if r_clamped == 0:
            keep_idx = torch.arange(T, device=device).view(1, T).repeat(B, 1)[:, :K_eff]
            return keep_idx

        drop_vals, drop_idx = torch.topk(-base_masked, k=r_clamped, dim=1, largest=True, sorted=False)
        keep_mask = torch.ones((B, T), dtype=torch.bool, device=device)
        for b in range(B):
            keep_mask[b, drop_idx[b]] = False
            if cls_protect and T > 0:
                keep_mask[b, 0] = True

        out = torch.zeros((B, K_eff), dtype=torch.long, device=device)
        arange_t = torch.arange(T, device=device)
        for b in range(B):
            row = arange_t[keep_mask[b]]
            # ensure CLS at column 0
            if cls_protect and T > 0 and row.numel() > 0 and int(row[0].item()) != 0:
                where0 = (row == 0).nonzero(as_tuple=False).view(-1)
                if where0.numel() > 0:
                    pos = int(where0[0].item())
                    row[0], row[pos] = row[pos], row[0]
            out[b, :min(K_eff, row.numel())] = row[:K_eff]
        return out

    # ---------------- keep-mode (vectorized + incremental FF) -----------------
    K_eff = max(1, min(int(K), T))
    out = torch.zeros((B, K_eff), dtype=torch.long, device=device)

    for b in range(B):
        keep_list: List[int] = []

        # reserve CLS
        if cls_protect and T > 0 and K_eff > 0:
            keep_list.append(0)

        need_more = K_eff - len(keep_list)
        if need_more <= 0:
            out[b] = torch.tensor(keep_list[:K_eff], dtype=torch.long, device=device)
            continue

        # candidate mask; exclude CLS
        cand_mask = torch.ones(T, dtype=torch.bool, device=device)
        if cls_protect and T > 0:
            cand_mask[0] = False

        # candidate pool clamp by base
        if cand_extra > 0:
            pool_k = min(T, max(K_eff, K_eff + int(cand_extra)))
            idx_pool = torch.topk(base[b], k=pool_k, dim=-1, largest=True).indices
            mask_pool = torch.zeros(T, dtype=torch.bool, device=device)
            mask_pool[idx_pool] = True
            cand_mask = cand_mask & mask_pool

        # head-quota (vectorized per head)
        chosen_quota: List[int] = []
        remaining = need_more
        if quota_frac > 0.0 and H > 0 and remaining > 0:
            per_head = int(math.ceil(float(need_more) * float(quota_frac) / float(H)))
            if per_head > 0:
                head_scores = phi[b].abs()                                 # [T,H]
                masked = torch.where(cand_mask.unsqueeze(1), head_scores,
                                     torch.full_like(head_scores, float("-inf")))
                take_k = min(per_head, int(cand_mask.sum().item()))
                if take_k > 0:
                    # take top-k per head, then unify with dedup until remaining exhausts
                    for h in range(H):
                        idx_h = torch.topk(masked[:, h], k=take_k, dim=0, largest=True).indices
                        for t in idx_h.tolist():
                            if cand_mask[t]:
                                chosen_quota.append(int(t))
                                cand_mask[t] = False
                                remaining -= 1
                                if remaining <= 0:
                                    break
                        if remaining <= 0:
                            break

        # incremental farthest-first for the rest
        chosen_div: List[int] = []
        if remaining > 0:
            seed_idx = torch.tensor(keep_list + chosen_quota, dtype=torch.long, device=device)
            extra = _ff_incremental_single(
                feat_b=feat[b], cand_mask_b=cand_mask, base_b=base[b], seed_idx=seed_idx, need=remaining
            )
            chosen_div = extra.tolist()

        chosen = chosen_quota + chosen_div

        # backfill (force_k) if short
        if len(chosen) < remaining and bool(force_k):
            missing = remaining - len(chosen)
            mask2 = cand_mask.clone()
            for t in chosen:
                if 0 <= t < T:
                    mask2[t] = False
            base_masked = torch.where(mask2, base[b], torch.full_like(base[b], float("-inf")))
            add_idx = torch.topk(base_masked, k=min(missing, int(mask2.sum().item())), dim=-1, largest=True).indices
            chosen.extend(add_idx.tolist())

        # compose keep row and finalize
        keep_b = keep_list + chosen
        keep_b = [t for t in keep_b if 0 <= t < T]
        keep_b = _dedup_preserve_order_1d(torch.tensor(keep_b, dtype=torch.long, device=device)).tolist()
        keep_b = keep_b[:K_eff]

        if cls_protect and T > 0 and K_eff > 0:
            if 0 not in keep_b:
                if len(keep_b) == 0:
                    keep_b = [0]
                else:
                    keep_b[-1] = 0
            zero_pos = keep_b.index(0)
            if zero_pos != 0:
                keep_b[0], keep_b[zero_pos] = keep_b[zero_pos], keep_b[0]

        out[b] = torch.tensor(keep_b, dtype=torch.long, device=device)

    return out
