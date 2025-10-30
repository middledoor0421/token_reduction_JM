# methods/ours/selectors/topk.py
import torch
from .registry import register

@register("topk")
def topk(sig, keep_k, sizes=None, cand_extra=0, hq_q=0.0, gamma=0.0):
    mag = sig.norm(p=1, dim=-1)
    return mag.topk(keep_k, largest=True, sorted=False).indices
