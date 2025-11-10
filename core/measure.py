# core/measure.py
# Python 3.9 compatible. Comments in English only.

import time
from typing import Dict, Any, List, Optional
import torch

class BlockMeter(object):
    """Measure per-block token lengths and elapsed time (ms)."""
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, layer: int, t_pre: int, t_post: int, ms: float) -> None:
        self.rows.append({"layer": layer, "T_pre": t_pre, "T_post": t_post, "delta": t_pre - t_post, "ms": ms})

    def summary(self) -> str:
        if not self.rows:
            return "[Measure] No data."
        rows = sorted(self.rows, key=lambda x: x["layer"])
        header = ["Layer", "T_pre", "T_post", "Δ", "ms"]
        widths = [max(len(h), 5) for h in header]
        def fmt(v): return str(v)
        data = []
        for r in rows:
            row = [fmt(r["layer"]), fmt(r["T_pre"]), fmt(r["T_post"]), fmt(r["delta"]), f"{r['ms']:.3f}"]
            data.append(row)
            for i, c in enumerate(row):
                widths[i] = max(widths[i], len(c))
        def join(line): return " | ".join(line[i].ljust(widths[i]) for i in range(len(line)))
        lines = [join(header), "-+-".join("-"*w for w in widths)]
        lines += [join(d) for d in data]
        tot_ms = sum(r["ms"] for r in rows)
        lines.append(f"\nTotal block time: {tot_ms:.3f} ms")
        return "\n".join(lines)
