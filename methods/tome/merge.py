# methods/tome/merge.py
# Size-weighted token merging (convex). CLS (index 0) must be kept by caller.

import torch

def merge_pairs_size_weighted(x, sizes, left_idx, right_idx):
    """
    x:        [B, N, C]
    sizes:    [B, N]
    left_idx: [B, R]
    right_idx:[B, R]

    Returns:
      x_new:     [B, N-R, C]
      sizes_new: [B, N-R]
      keep_mask: [B, N] boolean mask of kept tokens (same mask must be applied to residual)

    Notes:
      - Merge right -> left with convex weights (size-weighted average).
      - Preserve original order; merged right indices are removed.
      - Caller must ensure CLS (index 0) is not in right_idx.
    """
    B, N, C = x.shape
    device = x.device
    keep = torch.ones(B, N, dtype=torch.bool, device=device)  # will drop merged-right positions

    for b in range(B):
        L = left_idx[b]
        R = right_idx[b]
        for li, ri in zip(L.tolist(), R.tolist()):
            if li < 0 or ri < 0 or li == ri:
                continue
            # size-weighted convex merge into left
            sl = sizes[b, li]
            sr = sizes[b, ri]
            ssum = sl + sr
            w_l = sl / (ssum + 1e-6)
            w_r = 1.0 - w_l
            x[b, li] = w_l * x[b, li] + w_r * x[b, ri]
            sizes[b, li] = ssum
            keep[b, ri] = False  # drop the right index

    x_new = x[keep].view(B, -1, C)
    s_new = sizes[keep].view(B, -1)
    return x_new, s_new, keep
