# core/flops.py
# FLOPs estimation using fvcore (single resolution, inference-mode).

from typing import Tuple
import torch

def flops_gmacs(model, resolution=(3, 224, 224)) -> Tuple[float, int]:
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
    except Exception:
        return -1.0, sum(p.numel() for p in model.parameters())
    model.eval()
    HWC = resolution
    B = 1
    x = torch.zeros(B, *HWC, device=next(model.parameters()).device)
    with torch.inference_mode():
        flops = FlopCountAnalysis(model, x).total()
    params = sum(p.numel() for p in model.parameters())
    gmacs = flops / 1e9
    return float(gmacs), int(params)
