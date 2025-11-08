# methods/ours/selectors/hquota.py
# Python 3.9 compatible. Comments in English only.

from typing import Optional, List
import math
import torch
import torch.nn.functional as F


# ------------------------------ small utils ----------------------------------

def _l2norm_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 norm along last dimension."""
    return (x.pow(2).sum(dim=-1) + eps).sqrt()


def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Unit-normalize along the given dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _dedup_preserve_order_1d(idxs: torch.Tensor) -> torch.Tensor:
    """Remove duplicates from a 1D LongTensor while preserving order."""
    seen = set()
    kept: List[int] = []
    for v in idxs.tolist():
        if v not in seen:
            seen.add(v)
            kept.append(v)
    return torch.tensor(kept, dtype=idxxs.dtype if False else idxs.dtype, device=idxs.device)


# --------------------------- farthest-first core ------------------------------

def _farthest_first_single(
    feat_b: torch.Tensor,          # [T,H], normalized features
    cand_mask_b: torch.Tensor,     # [T] bool (True = candidate)
    base_b: torch.Tensor,          # [T] base strength (for seeding)
    seed_idx: torch.Tensor,        # [S] indices already kept (may be empty)
    need: int                      # number of additional tokens to pick
) -> torch.Tensor:
    """
    Farthest-first selection on a single batch row.
    Picks `need` indices among candidates, maximizing cosine distance
    from the current kept set.
    """
    T = int(feat_b.size(0))
    if need <= 0 or T == 0:
        return torch.empty(0, dtype=torch.long, device=feat_b.device)

    cand_mask = cand_mask_b.clone()
    if seed_idx.numel() > 0:
        cand_mask[seed_idx] = False

    chosen: List[int] = []

    # If there is no seed yet, start from the strongest base among candidates.
    if seed_idx.numel() == 0 and cand_mask.any():
        seed0 = int(torch.argmax(torch.where(cand_mask, base_b, torch.full_like(base_b, float("-inf")))).item())
        seed_idx = torch.tensor([seed0], dtype=torch.long, device=feat_b.device)
        cand_mask[seed0] = False

    keep = seed_idx.clone()

    for _ in range(need):
        cand_ids = torch.nonzero(cand_mask, as_tuple=False).view(-1)  # [Nc]
        if cand_ids.numel() == 0:
            break
        centers = feat_b[keep]                                  # [K,H]
        sims = torch.matmul(feat_b[cand_ids], centers.t())      # [Nc,K]
        max_sim, _ = sims.max(dim=1)                            # [Nc]
        pick_local = int(torch.argmin(max_sim).item())          # farthest from kept
        pick = int(cand_ids[pick_local].item())
        chosen.append(pick)
        cand_mask[pick] = False
        keep = torch.cat([keep, torch.tensor([pick], dtype=torch.long, device=feat_b.device)], dim=0)

    return torch.tensor(chosen, dtype=torch.long, device=feat_b.device)


# ------------------------------- public API ----------------------------------

def select_hquota_ff(
    phi: torch.Tensor,                 # [B,T,H] head-profile per token
    K: int,                            # target number of kept tokens
    quota_frac: float = 0.0,           # fraction of K reserved by per-head picks (0~1)
    cand_extra: int = 0,               # extra candidate pool beyond K for diversity
    force_k: bool = False,             # if True, backfill to exactly K
    cls_protect: bool = True,          # always keep CLS (token 0) if T>0
    scores: Optional[torch.Tensor] = None,  # [B,T] optional global strength
    mix_alpha: float = 0.5             # weight for mixing base and scores
) -> torch.Tensor:
    """
    Head-quota + farthest-first selector with CLS reservation and optional backfill.

    Returns:
      keep_idx: LongTensor [B, K_eff] of kept token indices.
        * If cls_protect and T>0 and K_eff>0, keep_idx[:,0] == 0.
        * If force_k=True, K_eff == min(K, T). Otherwise K_eff can be smaller.
    """
    assert phi.dim() == 3, "phi must be [B,T,H]"
    B, T, H = int(phi.size(0)), int(phi.size(1)), int(phi.size(2))
    device = phi.device

    if K <= 0 or T == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Clamp K to feasible range
    K_eff = max(1, min(int(K), T))

    # Base strength and normalized features
    base = _l2norm_last(phi)  # [B,T]
    if scores is not None:
        a = float(mix_alpha)
        base = (1.0 - a) * base + a * scores

    feat = F.normalize(phi, dim=-1)  # [B,T,H]

    out = torch.zeros((B, K_eff), dtype=torch.long, device=device)

    for b in range(B):
        keep_list: List[int] = []

        # 1) Reserve CLS at the first position if requested
        if cls_protect and T > 0 and K_eff > 0:
            keep_list.append(0)

        need_more = K_eff - len(keep_list)
        if need_more <= 0:
            out[b] = torch.tensor(keep_list[:K_eff], dtype=torch.long, device=device)
            continue

        # 2) Build candidate mask; exclude CLS when protected
        cand_mask = torch.ones(T, dtype=torch.bool, device=device)
        if cls_protect and T > 0:
            cand_mask[0] = False

        # 3) Narrow candidate pool by base score if requested
        if cand_extra > 0:
            pool_k = min(T, max(K_eff, K_eff + int(cand_extra)))
            idx_pool = torch.topk(base[b], k=pool_k, dim=-1, largest=True).indices  # [pool_k]
            mask_pool = torch.zeros(T, dtype=torch.bool, device=device)
            mask_pool[idx_pool] = True
            cand_mask = cand_mask & mask_pool

        # 4) Head quota picks (greedy by per-head magnitude)
        chosen_quota: List[int] = []
        remaining = need_more
        if quota_frac > 0.0 and H > 0 and remaining > 0:
            per_head = int(math.ceil(float(need_more) * float(quota_frac) / float(H)))
            for h in range(H):
                if remaining <= 0:
                    break
                head_scores = torch.where(cand_mask, phi[b, :, h].abs(), torch.full_like(base[b], float("-inf")))
                take_h = min(per_head, int(cand_mask.sum().item()), remaining)
                if take_h <= 0:
                    continue
                idx_h = torch.topk(head_scores, k=take_h, dim=-1, largest=True).indices
                for t in idx_h.tolist():
                    if cand_mask[t]:
                        chosen_quota.append(int(t))
                        cand_mask[t] = False
                        remaining -= 1
                        if remaining <= 0:
                            break

        # 5) Diversity fill for the rest via farthest-first
        chosen_div: List[int] = []
        if remaining > 0:
            seed_idx = torch.tensor(keep_list + chosen_quota, dtype=torch.long, device=device)
            extra = _farthest_first_single(
                feat_b=feat[b], cand_mask_b=cand_mask, base_b=base[b], seed_idx=seed_idx, need=remaining
            )
            chosen_div = extra.tolist()

        chosen = chosen_quota + chosen_div

        # 6) Backfill by base if still short or when force_k=True
        if len(chosen) < need_more:
            missing = need_more - len(chosen)
            mask2 = cand_mask.clone()
            for t in chosen:
                if 0 <= t < T:
                    mask2[t] = False
            base_masked = torch.where(mask2, base[b], torch.full_like(base[b], float("-inf")))
            add_idx = torch.topk(base_masked, k=min(missing, int(mask2.sum().item())), dim=-1, largest=True).indices
            chosen.extend(add_idx.tolist())

        # 7) Compose keep row, dedup, and final padding if still short
        keep_b = keep_list + chosen
        keep_b = [t for t in keep_b if 0 <= t < T]
        keep_b = _dedup_preserve_order_1d(torch.tensor(keep_b, dtype=torch.long, device=device)).tolist()

        if len(keep_b) < K_eff and bool(force_k):
            pad_need = K_eff - len(keep_b)
            mask3 = torch.ones(T, dtype=torch.bool, device=device)
            for t in keep_b:
                mask3[t] = False
            if cls_protect and T > 0:
                mask3[0] = False
            base_masked2 = torch.where(mask3, base[b], torch.full_like(base[b], float("-inf")))
            pad_idx = torch.topk(base_masked2, k=min(pad_need, int(mask3.sum().item())), dim=-1, largest=True).indices
            keep_b.extend(pad_idx.tolist())

        # 8) Clamp to K_eff and enforce CLS at position 0 when protected
        keep_b = keep_b[:K_eff]
        if cls_protect and T > 0 and K_eff > 0:
            if 0 not in keep_b:
                # ensure CLS exists by replacing last slot
                if len(keep_b) == 0:
                    keep_b = [0]
                else:
                    keep_b[-1] = 0
            # move CLS to front
            zero_pos = keep_b.index(0)
            if zero_pos != 0:
                keep_b[0], keep_b[zero_pos] = keep_b[zero_pos], keep_b[0]

        out[b] = torch.tensor(keep_b, dtype=torch.long, device=device)

    return out
