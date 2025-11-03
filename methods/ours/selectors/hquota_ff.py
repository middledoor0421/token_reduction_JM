# methods/ours/selectors/hquota.py
# Head-diversity selectors for "Ours".
# Final signature for exported selector:  select(phi, K) -> (keep_idx[B,K], assign_idx[B,T])
#   - phi : [B, T, H]  (L2-normalized head-profile per token; e.g., attn-in per head)
#   - K   : int, number of tokens to keep at this layer
#
# We provide:
#   * _farthest_first(phi, K) -> (keep_idx, assign_idx)
#   * select_hquota_ff(phi, K, q=0.3) -> (keep_idx, assign_idx)
#   * get_selector(name, q=0.3) -> callable `(phi, K) -> (keep_idx, assign_idx)`

from typing import Tuple, Callable
import torch
import torch.nn.functional as F


def _farthest_first(phi: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy farthest-first on token head-profiles.
    Args:
        phi: [B,T,H] L2-normalized along last dim.
        K:   number of tokens to keep (>=1, <=T).
    Returns:
        keep_idx:   [B,K] absolute indices of kept tokens
        assign_idx:[B,T] for each token, index (0..K-1) of its nearest kept token
    """
    B, T, H = phi.shape
    K = max(1, min(int(K), T))
    device = phi.device

    # Cosine similarity among tokens
    # S[b,i,j] = <phi[b,i], phi[b,j]>
    S = torch.einsum("bih,bjh->bij", phi, phi)  # [B,T,T]

    # Always keep CLS at position 0
    keep = torch.empty((B, K), dtype=torch.long, device=device)
    keep[:, 0] = 0

    # Track chosen set per batch
    chosen = torch.zeros((B, T), dtype=torch.bool, device=device)
    chosen[:, 0] = True

    # best similarity of each token to current kept set
    best_sim = S[:, 0, :]  # [B,T]

    for m in range(1, K):
        cands = best_sim.masked_fill(chosen, float("inf"))
        nxt = torch.argmin(cands, dim=1)        # farthest (min cosine) from current set
        keep[:, m] = nxt
        chosen.scatter_(1, nxt.unsqueeze(1), True)
        # update best similarity with new center
        upd = S[torch.arange(B, device=device), nxt][:, :]
        best_sim = torch.maximum(best_sim, upd)

    # assign each token to nearest kept token
    kept = torch.gather(phi, 1, keep.unsqueeze(-1).expand(-1, K, H))  # [B,K,H]
    sim = torch.einsum("bih,bkh->bik", phi, kept)  # [B,T,K]
    assign = sim.argmax(dim=2)                     # [B,T]
    # force CLS token (position 0) to map to first kept
    assign[:, 0] = 0
    return keep, assign


def select_hquota_ff(
    phi: torch.Tensor,
    K: int,
    q: float = 0.30
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Head-quota(top-q by head-sum) + farthest-first selection with CLS protection.

    Args:
        phi: [B, T, H]  L2-normalized head-profile per token (b= batch, t=token, h=head)
        K  : int, number of tokens to keep (>=1, <=T)
        q  : fraction in [0,1], number of "seed" tokens taken by global head-sum (excl. CLS)

    Returns:
        keep_idx   : LongTensor [B, K] of absolute token indices to keep.
                     Column 0 is always CLS (index 0).
        assign_idx : LongTensor [B, T] for mapping each token to its assigned kept token index.
    """
    # ---- normalize features for cosine geometry (defensive) ----
    phi = F.normalize(phi, dim=-1, eps=1e-6)  # [B,T,H]
    B, T, H = phi.shape
    K = int(max(1, min(K, T)))                # never exceed T, never be < 1
    dev = phi.device

    # allocate output container for kept absolute indices
    keep = torch.zeros((B, K), dtype=torch.long, device=dev)

    # always keep CLS in column 0
    keep[:, 0] = 0

    # nothing more to pick if K==1 or no non-CLS tokens
    noncls = max(0, T - 1)
    if K == 1 or noncls == 0:
        # everyone maps to CLS column 0
        assign_col = torch.zeros((B, T), dtype=torch.long, device=dev)
        return keep, assign_col

    # ----- hub-quota seeds from non-CLS pool (indices 1..T-1) -----
    # how many seeds from (K-1) slots, bounded by available non-CLS tokens
    seed_cnt = min(noncls, max(0, int(q * K)))
    # gather top 'seed_cnt' most dissimilar heads by global strength (sum over heads)
    if seed_cnt > 0:
        scores = phi[:, 1:, :].sum(dim=-1)  # [B, noncls]
        _, top_rel = torch.topk(scores, k=seed_cnt, dim=1, largest=True, sorted=True)  # [B, seed_cnt]
        keep[:, 1:1 + seed_cnt] = top_rel + 1  # +1 to shift back to absolute token indices

    # remaining slots to fill via farthest-first (diversity) among non-CLS tokens
    rem = K - 1 - seed_cnt
    if rem > 0:
        # pairwise cosine similarity per batch: S[b] = phi_b @ phi_b^T, shape [T,T]
        S = torch.bmm(phi, phi.transpose(1, 2))  # [B, T, T]

        # for each batch, maintain a boolean mask of which token indices are already kept
        taken = torch.zeros((B, T), dtype=torch.bool, device=dev)
        taken[:, 0] = True  # CLS is always kept
        if seed_cnt > 0:
            # mark seeded columns as taken (shift by +1 for non-CLS indices)
            cols = keep[:, 1:1 + seed_cnt]
            # scatter True into taken mask at chosen columns per batch
            # loop over batch for clarity/robustness
            for b in range(B):
                if cols.size(1) > 0:
                    taken[b].scatter_(0, cols[b], True)

        next_col = 1 + seed_cnt  # next free column to write in `keep`
        # Greedy farthest-first fill for remaining slots
        for _ in range(rem):
            # for each batch: compute "best(sim)" to current kept set for every token j
            # best_sim[b, j] = max_{i in kept(b)} S[b, j, i]
            best_sim = torch.full((B, T), float('-inf'), device=dev)
            for b in range(B):
                kept_cols = keep[b, :next_col]  # [next_col]
                # similarity of all tokens (rows) to currently kept columns
                sim_to_kept = S[b][:, kept_cols]                       # [T, next_col]
                # best (max) similarity to any kept token
                best = sim_to_kept.max(dim=1).values                   # [T]
                # forbid reselecting already kept tokens
                best = best.masked_fill(taken[b], float('inf'))
                # pick token with smallest "best similarity" (i.e., farthest from current set)
                nxt = torch.argmin(best).item()
                keep[b, next_col] = int(nxt)
                taken[b, nxt] = True
            next_col += 1
            if next_col >= K:  # safety guard
                break

    # ----- Assign each token to nearest kept column (keep CLS self-mapping) -----
    kept_phi = torch.gather(phi, 1, keep.unsqueeze(-1).expand(-1, K, H))  # [B,K,H]
    sim = torch.einsum("bth,bkh->btk", phi, kept_phi)                      # [B,T,K]
    # prevent non-CLS tokens from mapping to CLS column (index 0) if 원치 않으면 주석 처리
    sim[:, 1:, 0] = -1e9
    assign_col = sim.argmax(dim=2)                                         # [B,T]
    assign_col[:, 0] = 0                                                   # CLS → column 0
    # translate kept-column id → absolute token index
    assign_idx = torch.stack([keep[b, assign_col[b]] for b in range(B)], dim=0)  # [B,T]

    return keep, assign_idx

def get_selector(name: str, q: float = 0.30) -> Callable[[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]:
    name = (name or "hquota_ff").lower()
    if name in ("hquota", "hquota_ff"):
        return lambda phi, K: select_hquota_ff(phi, K, q)
    if name in ("ff", "farthest_first"):
        return _farthest_first
    raise ValueError(f"Unknown selector: {name}")
