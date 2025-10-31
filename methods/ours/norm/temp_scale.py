# methods/ours/norm/temp_scale.py
# Size-adaptive temperature scaling helper.

import torch

def apply_size_temperature(scale: torch.Tensor,
                           size: torch.Tensor,
                           eta: float) -> torch.Tensor:
    """
    Adjust attention scale by token size.
    scale: scalar or tensor broadcastable to attention logits
    size:  [B, K] merged sizes (or [B,T] if applied before merge)
    eta:   exponent; if 0, no-op
    Returns:
      adjusted scale (same shape as scale)
    """
    if eta == 0.0:
        return scale
    mean_s = size.mean(dim=1, keepdim=True).clamp_min(1e-12)
    adj = (size / mean_s).pow(eta)  # [B,K]
    while adj.dim() < scale.dim():
        adj = adj.unsqueeze(-1)
    return scale / adj
