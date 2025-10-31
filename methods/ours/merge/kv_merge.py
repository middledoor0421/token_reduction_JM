# methods/ours/merges/kv_merge.py
# Minimal KV merge utilities (Python 3.9). Comments in English.

from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def size_weighted_merge_v(v: torch.Tensor,
                          size: torch.Tensor,
                          assign_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge Value by size-weighted average.
    Args:
      v:    [B, H, T, Hd]
      size: [B, T]
      assign_idx: [B, T] mapping each token -> kept token index in [0..K-1]
    Returns:
      v_merged: [B, H, K, Hd]
      size_merged: [B, K]
    """
    B, H, T, Hd = v.shape
    K = int(assign_idx.max().item()) + 1
    device = v.device
    # Expand size to match v for weighted sum
    s = size.view(B, 1, T, 1)  # [B,1,T,1]
    v_w = v * s                # [B,H,T,Hd]

    # Build one-hot assignment: [B,T,K]
    one_hot = torch.zeros(B, T, K, device=device, dtype=v.dtype)
    one_hot.scatter_(dim=2, index=assign_idx.unsqueeze(2), value=1.0)  # [B,T,K]

    # Sum per kept group
    # v_sum: [B,H,K,Hd] = (B,H,T,Hd) x (B,T,K) along T
    v_sum = torch.einsum("bhtd,btk->bhkd", v_w, one_hot)
    s_sum = torch.einsum("btk,bt->bk", one_hot, size)  # [B,K]

    # Avoid division by zero
    eps = 1e-12
    v_merged = v_sum / (s_sum.view(B, 1, K, 1).clamp_min(eps))
    return v_merged, s_sum


def mean_merge_k(k: torch.Tensor,
                 assign_idx: torch.Tensor) -> torch.Tensor:
    """
    Merge Key by simple mean per kept group.
    Args:
      k: [B, H, T, Hd]
      assign_idx: [B, T] in [0..K-1]
    Returns:
      k_merged: [B, H, K, Hd]
    """
    B, H, T, Hd = k.shape
    K = int(assign_idx.max().item()) + 1
    device = k.device

    one_hot = torch.zeros(B, T, K, device=device, dtype=k.dtype)
    one_hot.scatter_(dim=2, index=assign_idx.unsqueeze(2), value=1.0)  # [B,T,K]

    k_sum = torch.einsum("bhtd,btk->bhkd", k, one_hot)  # [B,H,K,Hd]
    cnt = one_hot.sum(dim=1).clamp_min(1.0)            # [B,K]
    k_merged = k_sum / cnt.view(B, 1, K, 1)
    return k_merged


def build_unmerge_map(assign_idx: torch.Tensor) -> torch.Tensor:
    """
    Build unmerge scatter indices mapping reduced -> full length.
    Args:
      assign_idx: [B, T] values in [0..K-1] (kept representative per original token)
    Returns:
      scatter_idx: [B, T, 1] indices into reduced axis (K) for gathering/scattering.
    """
    return assign_idx.unsqueeze(-1)  # [B,T,1]


def apply_unmerge(y_reduced: torch.Tensor,
                  scatter_idx: torch.Tensor) -> torch.Tensor:
    """
    Expand reduced tokens back to length T using scatter indices.
    Args:
      y_reduced:  [B, T', C] where T' == K
      scatter_idx:[B, T, 1] with values in [0..K-1]
    Returns:
      y_full:     [B, T, C]
    """
    B, K, C = y_reduced.shape
    T = scatter_idx.shape[1]
    # Gather along K -> T
    y_full = torch.gather(y_reduced, dim=1, index=scatter_idx.expand(B, T, C))
    return y_full


def apply_pushlite(v_merged: torch.Tensor,
                   v_ref: Optional[torch.Tensor],
                   alpha: float,
                   beta0: float,
                   size_merged: Optional[torch.Tensor] = None,
                   size_eta: float = 0.0,
                   top_r: int = 0) -> torch.Tensor:
    """
    Apply light push correction on merged Value.
    v_merged:    [B,H,K,Hd] merged value
    v_ref:       [B,H,K,Hd] reference delta source (e.g., pre-merge diff), or None to skip
    alpha:       strength
    beta0:       base cap
    size_merged: [B,K] sizes for adaptive cap, or None (no adaptation)
    size_eta:    exponent for size adaptation (beta = beta0 * (size/mean)^eta)
    top_r:       apply only to top-r channels per token by magnitude (0 disables sparsity)
    """
    if v_ref is None or alpha == 0.0:
        return v_merged
    B, H, K, Hd = v_ref.shape
    if size_merged is not None and size_eta != 0.0:
        mean_s = size_merged.mean(dim=1, keepdim=True)  # [B,1]
        adapt = (size_merged / mean_s.clamp_min(1e-12)).pow(size_eta)  # [B,K]
        beta = beta0 * adapt  # [B,K]
        beta = beta.view(B, 1, K, 1)
    else:
        beta = v_ref.new_full((B, 1, K, 1), float(beta0))

    delta = torch.clamp(v_ref, min=-1.0, max=1.0)  # basic safe bound
    if top_r and top_r > 0 and top_r < Hd:
        # keep only top-r channels per token by |delta|
        mag = delta.abs()                            # [B,H,K,Hd]
        kth = torch.kthvalue(mag, k=Hd - top_r + 1, dim=-1).values  # [B,H,K]
        mask = (mag >= kth.unsqueeze(-1)).to(delta.dtype)           # [B,H,K,Hd]
        delta = delta * mask

    v_corr = v_merged + alpha * torch.clamp(delta, min=-beta, max=beta)
    return v_corr
