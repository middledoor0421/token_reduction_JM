# methods/ours/selectors/random.py
import torch
from .registry import register

@register("random")
def random(sel_sig, keep_k, sizes=None, cand_extra=0, hq_q=0.0, gamma=0.0):
    N = sel_sig.size(0)
    perm = torch.randperm(N, device=sel_sig.device)
    return perm[:keep_k]
