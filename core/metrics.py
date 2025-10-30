# core/metrics.py
# Basic metrics: Top-1, margin, KL(P||Q), coverage/diversity.

from typing import Tuple
import torch
import torch.nn.functional as F

def top1_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).float().mean().item())

def margin_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # margin = logit_y - max_{c!=y} logit_c
    y = target.view(-1, 1)
    correct = logits.gather(1, y).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, y, float('-inf'))
    other = tmp.max(dim=1).values
    return (correct - other).detach()

def kl_pq_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    # KL(P||Q) where inputs are logits
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    # Note: torch.kl_div expects log_target=False for probabilities. We use manual formula.
    kl = (p.exp() * (p - q)).sum(dim=-1).mean()
    return float(kl.item())

# ---- Coverage on signature space ----

def kcenter_radius(sig: torch.Tensor, chosen_idx: torch.Tensor) -> float:
    # sig: [N,H], chosen_idx: [K]
    if chosen_idx.numel() == 0:
        return 0.0
    X = F.normalize(sig, dim=-1, eps=1e-6)
    Y = X[chosen_idx]
    dist = 1.0 - (X @ Y.t())
    min_d, _ = dist.min(dim=-1)
    return float(min_d.mean().item())

def facility_location(sig: torch.Tensor, chosen_idx: torch.Tensor) -> float:
    if chosen_idx.numel() == 0:
        return 0.0
    X = F.normalize(sig, dim=-1, eps=1e-6)
    Y = X[chosen_idx]
    sim = X @ Y.t()
    max_sim, _ = sim.max(dim=-1)
    return float(max_sim.sum().item())

def pairwise_diversity(sig: torch.Tensor, chosen_idx: torch.Tensor) -> float:
    if chosen_idx.numel() <= 1:
        return 0.0
    X = F.normalize(sig[chosen_idx], dim=-1, eps=1e-6)
    sim = X @ X.t()
    M = sim.size(0)
    mask = ~torch.eye(M, dtype=torch.bool, device=sim.device)
    vals = (1.0 - sim)[mask]
    return float(vals.mean().item())
