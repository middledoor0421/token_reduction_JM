# methods/ours/selectors/ff.py
import torch
import torch.nn.functional as F
from .registry import register, pairwise_cosine_distance

@register("ff")
def ff(sig, keep_k, sizes=None, cand_extra=0, hq_q=0.0, gamma=0.0):
    # sig: [N,H]
    N = sig.size(0)
    if keep_k >= N:
        return torch.arange(N, device=sig.device)
    idx_map = torch.arange(N, device=sig.device)
    S = sig
    if cand_extra > 0 and cand_extra < N:
        norms = sig.norm(p=2, dim=-1)
        cand_k = max(keep_k, cand_extra)
        cand = norms.topk(cand_k, largest=True, sorted=False).indices
        S = sig[cand]; idx_map = cand
    D = pairwise_cosine_distance(S)
    sel = []
    seed = D.sum(dim=-1).argmax().item()
    sel.append(seed)
    mind = D[seed].clone()
    for _ in range(1, keep_k):
        j = mind.argmax().item()
        sel.append(j)
        mind = torch.minimum(mind, D[j])
    return idx_map[torch.tensor(sel, device=sig.device)]
