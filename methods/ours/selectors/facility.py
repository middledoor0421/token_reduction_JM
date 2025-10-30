# methods/ours/selectors/facility.py
import torch
import torch.nn.functional as F
from .registry import register

@register("facility")
def facility(sig, keep_k, sizes=None, cand_extra=0, hq_q=0.0, gamma=0.0):
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
    sims = X @ X.t()
    chosen = []
    max_sim = torch.full((X.size(0),), -1e9, device=X.device, dtype=X.dtype)
    y0 = sims.sum(dim=0).argmax().item()
    chosen.append(y0)
    max_sim = torch.maximum(max_sim, sims[:, y0])
    for _ in range(1, keep_k):
        gain = (sims - max_sim.unsqueeze(1)).clamp_min(0.0).sum(dim=0)
        gain[chosen] = -1e9
        y = gain.argmax().item()
        chosen.append(y)
        max_sim = torch.maximum(max_sim, sims[:, y])
    return idx_map[torch.tensor(chosen, device=X.device)]
