# core/schedule.py
# Build a unified per-layer r_map from CLI-style options.
# Python 3.9 compatible. Comments in English only.

from typing import List, Optional, Any

# ------------------------------
# CSV parser (added)
# ------------------------------
def parse_csv_int_list(x: Optional[Any]) -> Optional[List[int]]:
    """
    Parse CLI-like CSV into a list of ints.
    Accepts: None, str ("0,1, 2"), list/tuple of str/ints.
    Returns None if input is None or becomes empty after parsing.
    Invalid tokens are ignored.
    """
    if x is None:
        return None

    # Already a sequence
    if isinstance(x, (list, tuple)):
        out: List[int] = []
        for v in x:
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            try:
                out.append(int(s))
            except Exception:
                continue
        return out if len(out) > 0 else None

    # Treat as string
    s = str(x).strip()
    if not s or s.lower() == "none":
        return None

    out: List[int] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t or t.lower() == "none":
            continue
        try:
            out.append(int(t))
        except Exception:
            continue
    return out if len(out) > 0 else None


# ------------------------------
# Core builder (existing)
# ------------------------------
def make_r_map(
    n_blocks: int,
    r: Optional[int],
    r_list: Optional[List[int]],
    layers: Optional[List[int]],
) -> List[int]:
    """
    Return an r_map of length n_blocks. Single canonical path:
    1) If r_list is provided, use it; if shorter than n_blocks, pad with zeros.
    2) Else if layers is provided, fill r at those indices and zeros elsewhere.
    3) Else return all zeros.
    All negative values are clamped to 0.
    """
    nb = int(n_blocks)
    if nb <= 0:
        return []

    # 1) r_list path
    if r_list is not None and len(r_list) > 0:
        rl = [int(v) if int(v) > 0 else 0 for v in r_list]
        if len(rl) >= nb:
            return rl[:nb]
        return rl + [0] * (nb - len(rl))

    # 2) layers + r path
    out = [0] * nb
    if layers is not None and len(layers) > 0:
        rv = int(r) if r is not None else 0
        if rv < 0:
            rv = 0
        for idx in layers:
            if 0 <= int(idx) < nb:
                out[int(idx)] = rv
        return out

    # 3) default: no reduction
    return out


# ------------------------------
# CLI-friendly wrapper (added)
# ------------------------------
def make_r_map_from_cli(
    n_blocks: int,
    r: Optional[int],
    r_list_csv: Optional[Any],
    layers_csv: Optional[Any],
) -> List[int]:
    """
    Convenience wrapper that accepts CSV strings or lists directly from argparse
    and builds the final r_map using make_r_map.
    """
    rl = parse_csv_int_list(r_list_csv)
    ls = parse_csv_int_list(layers_csv)
    return make_r_map(n_blocks=n_blocks, r=r, r_list=rl, layers=ls)


# ------------------------------
# Pretty printer (existing)
# ------------------------------
def pretty_r_map(r_map: List[int]) -> str:
    """Compact string for logging, e.g., [13,13,13,0,0,...]."""
    return "[" + ",".join(str(int(v if v > 0 else 0)) for v in r_map) + "]"
