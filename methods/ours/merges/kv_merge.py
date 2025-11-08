# methods/ours/merges/kv_merge.py
# Python 3.9 compatible. Comments in English only.

from typing import Tuple, Dict, Any, Optional
import torch


@torch.no_grad()
def size_weighted_merge_v(
    x: torch.Tensor,                 # [B, T, C]
    keep_idx: torch.Tensor,          # [B, K] indices of kept centers
    assign_idx: torch.Tensor,        # [B, T] cluster id in [0..K-1] for each token
    alpha: float = 0.0,              # center bias: out = (1-alpha)*mean + alpha*center_vec
    size_delta: float = 0.0          # reserved (hook for custom size weighting)
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Merge tokens in V-space by size-weighted averaging per cluster.
    Ensures each center maps to itself (assumed by caller).

    Returns:
      x_merged: [B, K, C]
      info: {"assign_idx":[B,T], "keep_idx":[B,K], "sizes":[B,K]}
    """
    assert x.dim() == 3, "x must be [B,T,C]"
    B, T, C = int(x.size(0)), int(x.size(1)), int(x.size(2))
    K = int(keep_idx.size(1))
    device = x.device
    dtype = x.dtype

    if K == 0 or T == 0:
        return x, {"assign_idx": assign_idx, "keep_idx": keep_idx, "sizes": torch.zeros((B, 0), dtype=dtype, device=device)}

    out = torch.zeros((B, K, C), dtype=dtype, device=device)
    sizes = torch.zeros((B, K), dtype=dtype, device=device)

    one_vec = torch.ones((T,), dtype=dtype, device=device)

    for b in range(B):
        idx = assign_idx[b]  # [T] -> cluster id in [0..K-1]

        # Sum values per cluster (scatter-add along cluster axis)
        out_b = out[b]                    # [K,C]
        sizes_b = sizes[b]                # [K]
        out_b.index_add_(0, idx, x[b])   # sum of vectors
        sizes_b.index_add_(0, idx, one_vec)

        # Mean per cluster
        denom = sizes_b.clamp_min(1.0).unsqueeze(-1)  # [K,1]
        mean_b = out_b / denom                        # [K,C]

        # Optional center bias blending
        if alpha != 0.0:
            centers_b = x[b, keep_idx[b]]            # [K,C]
            out[b] = (1.0 - float(alpha)) * mean_b + float(alpha) * centers_b
        else:
            out[b] = mean_b

        # (reserved) custom size weighting via size_delta could be inserted here

    info: Dict[str, Any] = {
        "assign_idx": assign_idx,
        "keep_idx": keep_idx,
        "sizes": sizes
    }
    return out, info


@torch.no_grad()
def size_weighted_merge_kv(
    k: torch.Tensor,                 # [B, T, Ck]
    v: torch.Tensor,                 # [B, T, Cv]
    keep_idx: torch.Tensor,          # [B, K]
    assign_idx: torch.Tensor,        # [B, T] in [0..K-1]
    alpha: float = 0.0,
    size_delta: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Merge keys and values jointly (mean per cluster, optional center bias).
    Returns:
      k_merged: [B, K, Ck], v_merged: [B, K, Cv], info: {..., "sizes":[B,K]}
    """
    assert k.dim() == 3 and v.dim() == 3, "k,v must be [B,T,C]"
    B, T, Ck = int(k.size(0)), int(k.size(1)), int(k.size(2))
    _, _, Cv = v.size()
    K = int(keep_idx.size(1))
    device = k.device
    dtype_k = k.dtype
    dtype_v = v.dtype

    if K == 0 or T == 0:
        info = {"assign_idx": assign_idx, "keep_idx": keep_idx, "sizes": torch.zeros((B, 0), dtype=k.dtype, device=device)}
        return k, v, info

    k_out = torch.zeros((B, K, Ck), dtype=dtype_k, device=device)
    v_out = torch.zeros((B, K, Cv), dtype=dtype_v, device=device)
    sizes = torch.zeros((B, K), dtype=dtype_k, device=device)

    one_vec_k = torch.ones((T,), dtype=dtype_k, device=device)
    one_vec_v = torch.ones((T,), dtype=dtype_v, device=device)

    for b in range(B):
        idx = assign_idx[b]

        # scatter-add for keys
        k_out_b = k_out[b]
        sizes_b = sizes[b]
        k_out_b.index_add_(0, idx, k[b])
        sizes_b.index_add_(0, idx, one_vec_k)

        # scatter-add for values
        v_out_b = v_out[b]
        v_out_b.index_add_(0, idx, v[b])

        denom = sizes_b.clamp_min(1.0).unsqueeze(-1)
        k_mean = k_out_b / denom
        v_mean = v_out_b / denom

        if alpha != 0.0:
            centers_k = k[b, keep_idx[b]]
            centers_v = v[b, keep_idx[b]]
            k_out[b] = (1.0 - float(alpha)) * k_mean + float(alpha) * centers_k
            v_out[b] = (1.0 - float(alpha)) * v_mean + float(alpha) * centers_v
        else:
            k_out[b] = k_mean
            v_out[b] = v_mean

        # (reserved) insert size_delta weighting if needed

    info = {"assign_idx": assign_idx, "keep_idx": keep_idx, "sizes": sizes}
    return k_out, v_out, info


@torch.no_grad()
def apply_unmerge(
    x_merged: torch.Tensor,           # [B, K, C]
    keep_idx: torch.Tensor,           # [B, K]
    assign_idx: torch.Tensor,         # [B, T] cluster id in [0..K-1]
    info: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Unmerge by broadcasting cluster representatives back to original positions.
    Simple nearest-center "unpool": x_out[b, t] = x_merged[b, assign_idx[b, t]].
    """
    assert x_merged.dim() == 3, "x_merged must be [B,K,C]"
    B, K, C = int(x_merged.size(0)), int(x_merged.size(1)), int(x_merged.size(2))
    T = int(assign_idx.size(1))
    device = x_merged.device
    dtype = x_merged.dtype

    x_out = torch.zeros((B, T, C), dtype=dtype, device=device)
    for b in range(B):
        idx = assign_idx[b]                     # [T]
        x_out[b] = x_merged[b].index_select(0, idx)  # gather per original position
    return x_out
