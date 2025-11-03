# methods/ours/norm/l2_clip.py
# In-place L2 clipping utilities.

import torch

def l2_clip_(x: torch.Tensor, tau: float) -> torch.Tensor:
    """
    In-place L2 norm clipping per token vector (last dim).
    Args:
      x:   [..., C]
      tau: clip threshold (>0). If tau<=0, no-op.
    Returns:
      x after possible in-place clipping.
    """
    if tau is None or tau <= 0.0:
        return x
    eps = 1e-12
    n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(tau / n, max=1.0)
    x.mul_(scale)
    return x
