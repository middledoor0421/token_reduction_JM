# methods/ours/selectors/hquota.py
# Python 3.9 compatible. Comments in English only.

from typing import Optional
import math
import torch


def select_hquota_ff(
    phi: torch.Tensor,          # [B, T, H] head-profile per token
    K: int,                     # target keep count
    quota_frac: float = 0.0,    # fraction of K reserved for head-wise quotas (0.0~1.0)
    cand_extra: int = 0,        # extra candidate pool size to mitigate shortage
    force_k: bool = False,      # when True, backfill to exactly K
    cls_protect: bool = True,   # exclude CLS (idx 0) from removal
    scores: Optional[torch.Tensor] = None,  # [B, T] optional global scores to mix
    mix_alpha: float = 0.5      # weight to mix base-norm and scores: base*(1-alpha)+scores*alpha
) -> torch.Tensor:
    """
    Head-quota + farthest-first selector.

    Steps:
      1) Build base strength per token: base = ||phi||_2 (optionally mixed with provided scores).
      2) Reserve per-head quotas if quota_frac > 0: each head contributes ~ceil((quota_frac*K)/H) tokens.
      3) Diversity fill: farthest-first over normalized head-profile features to reach K.
      4) If force_k and still < K, backfill by global base score, respecting CLS-protect and no-dup.

    Returns:
      keep_idx: LongTensor [B, K_found]
    """
    assert phi.dim() == 3, "phi must be [B,T,H]"
    B, T, H = phi.shape
    device = phi.device
    K = int(max(1, K))

    # Base score
    base = phi.norm(p=2, dim=-1)  # [B,T]
    if scores is not None:
        a = float(mix_alpha)
        base = (1.0 - a) * base + a * scores

    # Candidate mask (exclude CLS if requested)
    cand_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    if cls_protect and T > 0:
        cand_mask[:, 0] = False

    # Optional candidate pool trimming for speed
    pool_k = min(T, max(K, K + int(cand_extra)))
    pool_idx = torch.topk(base, k=pool_k, dim=-1, largest=True)[1]  # [B, pool_k]
    pool_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        pool_mask[b, pool_idx[b]] = True
    cand_mask = cand_mask & pool_mask

    # Normalize head-profile for cosine distance
    eps = 1e-6
    feat = phi / (phi.pow(2).sum(dim=-1, keepdim=True).sqrt() + eps)  # [B,T,H]

    # Selection container
    selected = [[] for _ in range(B)]

    # 1) Per-head quota reservation
    if quota_frac > 0.0 and H > 0:
        per_head = int(math.ceil((float(quota_frac) * float(K)) / float(H)))
        if per_head > 0:
            for h in range(H):
                head_score = phi[:, :, h]  # [B,T]
                masked = torch.where(cand_mask, head_score, torch.full_like(head_score, float("-inf")))
                topk_h = min(per_head, int(cand_mask.sum(dim=1).min().item()))
                if topk_h <= 0:
                    continue
                idx_h = torch.topk(masked, k=topk_h, dim=-1, largest=True)[1]  # [B, topk_h]
                for b in range(B):
                    selected[b].extend(idx_h[b].tolist())

            # Deduplicate and validate
            for b in range(B):
                uniq = []
                seen = set()
                for t in selected[b]:
                    if t < 0 or t >= T:
                        continue
                    if not cand_mask[b, t]:
                        continue
                    if t in seen:
                        continue
                    seen.add(t)
                    uniq.append(t)
                selected[b] = uniq

    # 2) Diversity fill: farthest-first by cosine distance
    def farthest_first_fill(b: int, need: int) -> None:
        if need <= 0:
            return
        mask = cand_mask[b].clone()
        for t in selected[b]:
            if 0 <= t < T:
                mask[t] = False
        cand_ids = torch.nonzero(mask, as_tuple=False).view(-1)
        if cand_ids.numel() == 0:
            return

        # Seed by strongest base score if empty
        if len(selected[b]) == 0:
            base_b = torch.where(mask, base[b], torch.full_like(base[b], float("-inf")))
            seed = int(torch.argmax(base_b).item())
            selected[b].append(seed)
            need -= 1
            if need <= 0:
                return
            mask[seed] = False
            cand_ids = torch.nonzero(mask, as_tuple=False).view(-1)

        feat_b = feat[b]  # [T,H]
        chosen = torch.tensor(selected[b], device=device, dtype=torch.long)

        for _ in range(need):
            if cand_ids.numel() == 0:
                break
            cand_feat = feat_b[cand_ids]                  # [Nc,H]
            chosen_feat = feat_b[chosen]                  # [M,H]
            sim = torch.matmul(cand_feat, chosen_feat.t())  # [Nc,M]
            max_sim, _ = sim.max(dim=1)                  # [Nc]
            pick_idx = int(torch.argmin(max_sim).item())
            pick = int(cand_ids[pick_idx].item())
            selected[b].append(pick)

            chosen = torch.tensor(selected[b], device=device, dtype=torch.long)
            mask[pick] = False
            cand_ids = torch.nonzero(mask, as_tuple=False).view(-1)

    for b in range(B):
        need = K - len(selected[b])
        if need > 0:
            farthest_first_fill(b, need)

    # 3) Optional backfill to exactly K
    if force_k:
        for b in range(B):
            if len(selected[b]) < K:
                mask = cand_mask[b].clone()
                for t in selected[b]:
                    if 0 <= t < T:
                        mask[t] = False

                if not mask.any():
                    mask = torch.ones(T, dtype=torch.bool, device=device)
                    if cls_protect and T > 0:
                        mask[0] = False
                    for t in selected[b]:
                        if 0 <= t < T:
                            mask[t] = False

                base_b = torch.where(mask, base[b], torch.full_like(base[b], float("-inf")))
                need = K - len(selected[b])
                extra_k = min(need, int(mask.sum().item()))
                if extra_k > 0:
                    extra_idx = torch.topk(base_b, k=extra_k, dim=-1, largest=True)[1]
                    selected[b].extend(extra_idx.tolist())

    # Pack result to tensor with consistent width
    keep_list = []
    for b in range(B):
        uniq = []
        seen = set()
        for t in selected[b]:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        if len(uniq) > K:
            uniq = uniq[:K]
        keep_list.append(torch.tensor(uniq, device=device, dtype=torch.long) if len(uniq) > 0
                         else torch.zeros(0, dtype=torch.long, device=device))

    K_found = 0
    for v in keep_list:
        if v.numel() > K_found:
            K_found = int(v.numel())
    if K_found == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=device)

    out = torch.zeros(B, K_found, dtype=torch.long, device=device)
    for b in range(B):
        v = keep_list[b]
        if v.numel() == 0:
            continue
        out[b, : v.numel()] = v

    return out
