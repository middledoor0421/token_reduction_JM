# methods/ours/selectors/kdpp.py
import torch
import torch.nn.functional as F
from .registry import register

@register("kdpp")
def kdpp(sig, keep_k, sizes=None, cand_extra=0, hq_q=0.0, gamma=0.0):
    N = sig.size(0)
    if keep_k >= N:
        return torch.arange(N, device=sig.device)
    X = F.normalize(sig, dim=-1, eps=1e-6)
    idx_map = torch.arange(N, device=sig.device)
    if cand_extra > 0 and cand_extra < N:
        norms = sig.norm(p=2, dim=-1)
        cand_k = max(keep_k, cand_extra)
        cand = norms.topk(cand_k, largest=True, sorted=False).indices
        X = X[cand]; idx_map = cand
    K = (X @ X.t()).clamp(min=0.0, max=1.0)
    evals, evecs = torch.linalg.eigh(K)
    evals = evals.clamp(min=0.0, max=1.0)
    idx = torch.argsort(evals, descending=True)[:keep_k]
    V = evecs[:, idx]
    lev = (V ** 2).sum(dim=1)
    chosen = torch.topk(lev, k=keep_k, largest=True, sorted=False).indices
    return idx_map[chosen]
