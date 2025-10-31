# methods/ours/norm/recenter.py
# Simple recenter utility.

import torch

def recenter_(x: torch.Tensor) -> torch.Tensor:
    """
    In-place zero-mean recenter per batch (and optionally per head if present).
    Works for [..., T, C] or [B,H,T,D].
    """
    if x.dim() >= 3:
        # subtract mean over token axis
        m = x.mean(dim=-2, keepdim=True)
        x.sub_(m)
    else:
        x.sub_(x.mean(dim=-1, keepdim=True))
    return x
