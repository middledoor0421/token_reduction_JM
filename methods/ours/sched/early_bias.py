# core/sched/early_bias.py
# Python 3.9 compatible. Comments in English only.

from typing import List, Optional


def _coerce_bounds(keep: int, T: int) -> int:
    """
    Clamp the keep count to [1, T].
    """
    if keep < 1:
        keep = 1
    if keep > T:
        keep = T
    return keep


def r_to_keep(T: int, r: int, token_cap: str = "on", min_keep_cap: float = 0.0) -> int:
    """
    Convert requested removal count r to keep count for a given sequence length T.

    Arguments:
      T            : current token length before reduction (>= 1)
      r            : requested number of tokens to remove (>= 0)
      token_cap    : "on" or "off"
                     - "on"  : allow cap (minimum keep ratio) to apply
                     - "off" : ignore any lower bound cap
      min_keep_cap : minimum keep ratio in [0.0, 1.0]; only applied when token_cap == "on"

    Returns:
      keep count after applying r and optional lower bound cap.
    """
    T = int(T)
    r = int(r)
    cap_on = str(token_cap).lower() == "on"

    # Base keep by requested r
    keep = T - r

    # Apply minimum keep cap only when token_cap == "on"
    cap = float(min_keep_cap) if cap_on else 0.0
    if cap < 0.0:
        cap = 0.0
    if cap > 1.0:
        cap = 1.0

    min_keep = int(T * cap)
    if keep < min_keep:
        keep = min_keep

    # Enforce bounds
    keep = _coerce_bounds(keep, T)
    return keep


def keep_to_r(T: int, keep: int) -> int:
    """
    Convert keep count to removal count r for a given sequence length T.
    Result is clamped to [0, T-1] to ensure at least one token remains.

    Arguments:
      T    : current token length before reduction (>= 1)
      keep : desired keep count (will be clamped to [1, T])

    Returns:
      r = T - keep, clamped to [0, T-1].
    """
    T = int(T)
    keep = _coerce_bounds(int(keep), T)
    r = T - keep
    if r < 0:
        r = 0
    if r > T - 1:
        r = T - 1
    return r


def compute_keep_for_layers(
    T_per_layer: List[int],
    r: int,
    target_layers: Optional[List[int]] = None,
    token_cap: str = "on",
    min_keep_cap: float = 0.0
) -> List[int]:
    """
    Compute per-layer keep counts from a single removal target r and per-layer lengths.

    Arguments:
      T_per_layer  : list of token lengths for each layer before reduction, len = #layers
      r            : requested removal per targeted layer
      target_layers: list of layer indices where reduction is applied;
                     if None, apply to all layers
      token_cap    : "on" or "off" (see r_to_keep)
      min_keep_cap : minimum keep ratio applied when token_cap == "on"

    Returns:
      keep_per_layer: list of keep counts with the same length as T_per_layer.
                      Non-target layers return keep = T (no reduction).
    """
    L = len(T_per_layer)
    keeps: List[int] = [0 for _ in range(L)]
    target = set(target_layers) if target_layers is not None else None

    for i in range(L):
        T = int(T_per_layer[i])
        if target is None or i in target:
            keep = r_to_keep(T=T, r=r, token_cap=token_cap, min_keep_cap=min_keep_cap)
        else:
            keep = T  # no reduction on non-target layers
        keeps[i] = _coerce_bounds(keep, T)

    return keeps


def compute_r_for_layers(
    T_per_layer: List[int],
    keep_per_layer: List[int]
) -> List[int]:
    """
    Compute per-layer removal counts r from per-layer keep counts.

    Arguments:
      T_per_layer   : list of token lengths for each layer before reduction
      keep_per_layer: list of target keep counts for each layer

    Returns:
      r_per_layer: list of removal counts per layer, clamped to [0, T-1].
    """
    if len(T_per_layer) != len(keep_per_layer):
        raise ValueError("Length mismatch: T_per_layer and keep_per_layer must have the same length.")

    L = len(T_per_layer)
    r_out: List[int] = [0 for _ in range(L)]
    for i in range(L):
        T = int(T_per_layer[i])
        keep = int(keep_per_layer[i])
        r_out[i] = keep_to_r(T=T, keep=keep)
    return r_out
