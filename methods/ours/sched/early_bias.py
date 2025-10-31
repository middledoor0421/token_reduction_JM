# methods/ours/sched/early_bias.py
# Early-biased keep→r conversion with 50% cap and special-token protection.

from math import ceil
from typing import Tuple, Optional, Sequence
from . import register_scheduler


def _protected_count(class_token: bool, distill_token: bool) -> int:
    n = 0
    if class_token:
        n += 1
    if distill_token:
        n += 1
    return n


def _cap_r(T: int, r: int, class_token: bool, distill_token: bool) -> int:
    if T <= 0 or r <= 0:
        return 0
    max_r = max(0, (T - _protected_count(class_token, distill_token)) // 2)
    if r > max_r:
        r = max_r
    if r < 0:
        r = 0
    return r


def _parse_int_list(spec: Optional[str], total_layers: int) -> Sequence[int]:
    if spec is None:
        return []
    s = str(spec).strip().lower()
    if s == "" or s == "none":
        return []
    if s == "all":
        return list(range(total_layers))
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


@register_scheduler("early_bias")
def compute_keep_and_r(T: int,
                       layer_idx: int,
                       total_layers: int,
                       cfg: dict,
                       class_token: bool = True,
                       distill_token: bool = False) -> Tuple[float, int]:
    """
    Decide per-block keep ratio and r_block with early-biased policy.

    Inputs:
      T            : input token length (incl. special tokens)
      layer_idx    : 0-based block index
      total_layers : number of transformer blocks
      cfg          : dict with keys (all optional; defaults shown)
         keep_early: float, default 0.68
         keep_mid  : float, default 0.70
         keep_late : float, default 0.80
         early_set : str, e.g. "0,1,2" (override automatic early range)
         mid_set   : str, e.g. "3,4,5,6,7"
         late_set  : str, e.g. "8,9,10,11"
         cap_50    : bool, default True  (enforce r <= floor((T-protected)/2))
         min_keep  : float, default 0.50 (floor on keep to avoid >50% merges)
      class_token  : protect CLS from merging
      distill_token: protect distillation token if present

    Outputs:
      keep_ratio   : float in (0,1]
      r_block      : int, merges to perform in this block (after cap)
    """
    # defaults
    keep_early = float(cfg.get("keep_early", 0.68))
    keep_mid   = float(cfg.get("keep_mid",   0.70))
    keep_late  = float(cfg.get("keep_late",  0.80))
    min_keep   = float(cfg.get("min_keep",   0.50))
    cap_50     = bool(cfg.get("cap_50",      True))

    # sets
    early_set = _parse_int_list(cfg.get("early_set", None), total_layers)
    mid_set   = _parse_int_list(cfg.get("mid_set",   None), total_layers)
    late_set  = _parse_int_list(cfg.get("late_set",  None), total_layers)

    # if sets not provided, derive simple thirds with early bias on first 3 layers
    if len(early_set) == 0 and len(mid_set) == 0 and len(late_set) == 0:
        if total_layers >= 3:
            early_set = [0, 1, 2]
            mid_set   = list(range(3, max(3, total_layers - 2)))
            late_set  = list(range(max(3, total_layers - 2), total_layers))
        else:
            # fall back to simple split
            split = max(1, total_layers // 3)
            early_set = list(range(0, split))
            mid_set   = list(range(split, min(2 * split, total_layers)))
            late_set  = list(range(min(2 * split, total_layers), total_layers))

    if layer_idx in early_set:
        keep = keep_early
    elif layer_idx in mid_set:
        keep = keep_mid
    else:
        keep = keep_late

    # clamp keep and compute raw r
    if keep < min_keep:
        keep = min_keep
    if keep > 1.0:
        keep = 1.0

    target = int(ceil(keep * float(T)))
    r_raw = max(0, T - target)

    # optional 50% cap with special-token protection
    r_block = _cap_r(T, r_raw, class_token, distill_token) if cap_50 else r_raw
    return keep, r_block
