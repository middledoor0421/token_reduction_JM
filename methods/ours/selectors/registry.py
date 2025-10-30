# methods/ours/selectors/registry.py
# Selector registry and common helpers.

import torch
import torch.nn.functional as F

_SELECTORS = {}

def register(name):
    def _wrap(fn):
        if name in _SELECTORS:
            raise ValueError("Selector already registered: %s" % name)
        _SELECTORS[name] = fn
        return fn
    return _wrap

def list_selectors():
    return sorted(_SELECTORS.keys())

def select(name, sig, keep_k, sizes=None, cand_extra=0, hq_q=0.0, gamma=0.0):
    if name not in _SELECTORS:
        raise ValueError("Unknown selector: %s (available=%s)" % (name, list_selectors()))
    return _SELECTORS[name](sig=sig, keep_k=keep_k, sizes=sizes, cand_extra=cand_extra, hq_q=hq_q, gamma=gamma)

# ---- helpers ----

def pairwise_cosine_distance(X):
    X = F.normalize(X, dim=-1, eps=1e-6)
    sim = X @ X.t()
    return (1.0 - sim).clamp(min=0.0, max=2.0)

def _hub_mask(sig, sizes, q, gamma):
    # sig: [N,H], sizes: [N]
    inflow = sig.norm(p=1, dim=-1)
    s = sizes if sizes is not None else torch.ones_like(inflow)
    score = inflow * (s ** gamma)
    k = max(1, int(round(q * sig.size(0))))
    idx = score.topk(k, largest=True, sorted=False).indices
    m = torch.zeros(sig.size(0), dtype=torch.bool, device=sig.device)
    m[idx] = True
    return m
