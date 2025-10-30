# methods/ours/selectors/hquota_ff.py
import torch
from .registry import register, _hub_mask
from .ff import ff as _ff

@register("hquota_ff")
def hquota_ff(sig, keep_k, sizes=None, cand_extra=0, hq_q=0.0, gamma=0.0):
    N = sig.size(0)
    if keep_k >= N:
        return torch.arange(N, device=sig.device)
    hubs = _hub_mask(sig, sizes, q=float(hq_q), gamma=float(gamma)) if hq_q > 0 else torch.zeros(N, dtype=torch.bool, device=sig.device)
    hub_idx = torch.nonzero(hubs, as_tuple=False).flatten()
    remain = max(0, keep_k - hub_idx.numel())
    if remain == 0:
        return hub_idx
    non_idx = torch.nonzero(~hubs, as_tuple=False).flatten()
    sel_non = _ff(sig[non_idx], remain, sizes=None, cand_extra=cand_extra)
    return torch.cat([hub_idx, non_idx[sel_non]], dim=0)
