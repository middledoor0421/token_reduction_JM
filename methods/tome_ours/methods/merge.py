# methods/merge.py
# Core ToMe matching + merge utilities (ported from Meta's ToMe; Python 3.9 typing)

import math
from typing import Callable, Tuple
import torch
import torch.nn.functional as F

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable[[torch.Tensor, str], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """
    Balanced bipartite soft matching (A: even positions, B: odd positions).
    Returns (merge, unmerge) callables operating on tensors that have token
    dimension in the penultimate axis (..., T, C_like).
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = int(metric.shape[1])
    r = min(int(r), max(0, (t - protected) // 2))
    if r <= 0:
        return do_nothing, do_nothing # no-op

    with torch.no_grad():
        m = F.normalize(metric, dim=-1)
        a, b = m[..., ::2, :], m[..., 1::2, :]
        scores = torch.matmul(a, b.transpose(-1, -2))  # [..., Ta, Tb]

        if class_token:
            # Protect CLS at a[...,0,:]; forbid pairing it as src
            scores[..., 0, :] = -math.inf
        if distill_token:
            # If distill token is first in B, forbid pairing into it
            scores[..., :, 0] = -math.inf

        node_max, node_idx = torch.max(scores, dim=-1)             # [..., Ta]
        edge_idx = torch.argsort(node_max, dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # kept from A
        src_idx = edge_idx[..., :r, :]  # from A merged into B
        dst_idx = torch.gather(node_idx[..., None], dim=-2, index=src_idx)

        if class_token:
            # Ensure CLS (a[...,0,:]) stays in the first position of 'unm'
            unm_idx = torch.sort(unm_idx, dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n = src.shape[-2]
        c = x.shape[-1]
        # indices from bipartite have shape [..., n, 1] where ... = (B,H)
        keep = torch.gather(src, -2, unm_idx.expand_as(src[..., : n - r, :]))
        moved = torch.gather(src, -2, src_idx.expand_as(src[..., : r, :]))
        dst = dst.scatter_reduce(dim=-2, index=dst_idx.expand_as(moved), src=moved, reduce=mode, include_self=True)
        if distill_token:
            head_keep = keep[..., :1, :]
            rest_keep = keep[..., 1:, :]
            head_dst = dst[..., :1, :]
            tail_dst = dst[..., 1:, :]
            out = torch.cat([head_keep, head_dst, rest_keep, tail_dst], dim=-2)
        else:
            out = torch.cat([keep, dst], dim=-2)
        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # x has token axis length T' = T - r
        keep_len = int(unm_idx.shape[1])
        keep = x[..., :keep_len, :]
        dst = x[...,   keep_len:, :]
        c = x.shape[-1]
        moved = torch.gather(dst, dim=-2, index=dst_idx.expand(x.shape[:-2] + (dst.shape[-2], c)))

        out = x.new_zeros(x.shape[:-2] + (t, c))
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(out.shape[:-2] + (keep_len, c)), src=keep)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(out.shape[:-2] + (r, c)),     src=moved)
        return out

    return merge, unmerge

def merge_wavg(merge: Callable[[torch.Tensor, str], torch.Tensor],
               x: torch.Tensor,
               size: torch.Tensor = None):
    if size is None:
        shp = list(x.shape)
        shp[-1] = 1
        size = x.new_ones(*shp)
    x_sum = merge(x * size, "sum")
    s_sum = merge(size, "sum")
    eps = 1e-12
    x_out = x_sum / (s_sum.clamp_min(eps))
    return x_out, s_sum

def s_rev(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Safe reciprocal for broadcasted division."""
    return torch.clamp(x, min=eps)
