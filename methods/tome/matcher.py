# methods/tome/matcher.py
# ToMe-style bipartite matching:
# - Alternating partition on token order (CLS excluded by caller)
# - Cosine similarity between the two partitions
# - Greedy 1:1 selection of the top-r pairs (no Sinkhorn)

import torch
import torch.nn.functional as F

def _cosine_sim(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [Ne, C], B: [No, C] -> [Ne, No]
    A = F.normalize(A, dim=-1, eps=1e-6)
    B = F.normalize(B, dim=-1, eps=1e-6)
    return A @ B.t()

def match_greedy_bipartite(x_tokens: torch.Tensor, r: int, offset_swap: bool):
    """
    x_tokens: [N, C]  (CLS excluded; tokens 1..N are passed by caller)
    r: number of pairs to select
    offset_swap: if True, swap the alternating partitions (odd/even) for this block

    Returns:
      left_idx:  [R_eff] indices in [0..N-1] (relative to x_tokens)
      right_idx: [R_eff] indices in [0..N-1] (relative to x_tokens)
    """
    device = x_tokens.device
    N = x_tokens.size(0)
    if r <= 0 or N < 2:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

    # Alternating partition with per-block offset
    even = torch.arange(0, N, 2, device=device)
    odd  = torch.arange(1, N, 2, device=device)
    if offset_swap:
        left_ids, right_ids = odd, even
    else:
        left_ids, right_ids = even, odd

    if left_ids.numel() == 0 or right_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

    L = x_tokens[left_ids]  # [Nl, C]
    R = x_tokens[right_ids] # [Nr, C]
    S = _cosine_sim(L, R)   # [Nl, Nr]

    used_l = torch.zeros(L.size(0), dtype=torch.bool, device=device)
    used_r = torch.zeros(R.size(0), dtype=torch.bool, device=device)

    sel_L, sel_R = [], []
    for _ in range(int(r)):
        S_masked = S.clone()
        if used_l.any():
            S_masked[used_l] = float("-inf")
        if used_r.any():
            S_masked[:, used_r] = float("-inf")

        val, flat_idx = torch.max(S_masked.view(-1), dim=0)
        if torch.isinf(val) or torch.isnan(val):
            break

        li = flat_idx // R.size(0)
        ri = flat_idx %  R.size(0)

        sel_L.append(int(left_ids[int(li)].item()))
        sel_R.append(int(right_ids[int(ri)].item()))
        used_l[int(li)] = True
        used_r[int(ri)] = True

    if len(sel_L) == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

    return (torch.tensor(sel_L, dtype=torch.long, device=device),
            torch.tensor(sel_R, dtype=torch.long, device=device))
