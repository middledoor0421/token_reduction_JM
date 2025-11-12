# core/token_stats.py
# Python 3.9 compatible. Comments in English only.

from typing import Dict, Optional, Any, List
import json

class _Agg:
    """Running sums for per-layer aggregation."""
    __slots__ = (
        "before_sum", "merge_sum", "unmerge_sum",
        "req_sum", "count"
    )
    def __init__(self) -> None:
        self.before_sum = 0
        self.merge_sum = 0
        self.unmerge_sum = 0
        self.req_sum = 0
        self.count = 0

    def add(self, before_len: int, after_merge_len: int,
            after_unmerge_len: Optional[int], req_r: Optional[int]) -> None:
        self.before_sum += int(before_len)
        self.merge_sum += int(after_merge_len)
        if after_unmerge_len is None:
            # If unmerge did not happen, use merge value for final
            self.unmerge_sum += int(after_merge_len)
        else:
            self.unmerge_sum += int(after_unmerge_len)
        if isinstance(req_r, int):
            self.req_sum += int(req_r)
        self.count += 1

    def avg_tuple(self) -> Any:
        if self.count <= 0:
            # No data: return zeros
            return (0, 0, 0, 0)
        before = round(self.before_sum / self.count)
        after_m = round(self.merge_sum / self.count)
        after_u = round(self.unmerge_sum / self.count)
        req = round(self.req_sum / self.count) if self.req_sum > 0 else 0
        return (before, after_m, after_u, req)


class _TokenStats:
    """Dataset-averaged token statistics across layers."""
    def __init__(self) -> None:
        self._layers: Dict[int, _Agg] = {}
        self._total_layers: Optional[int] = None

    # -------- lifecycle --------
    def reset(self) -> None:
        self._layers = {}
        self._total_layers = None

    def set_total_layers(self, n: Optional[int]) -> None:
        """Set total number of transformer blocks; used for fixed-row reporting."""
        if n is None:
            self._total_layers = None
            return
        self._total_layers = max(0, int(n))

    # -------- record API --------
    def record(
        self,
        *,
        layer_idx: int,
        before_len: int,
        after_merge_len: int,
        after_unmerge_len: Optional[int] = None,
        requested_r: Optional[int] = None
    ) -> None:
        """Accumulate sums for a given layer; no per-batch raw storage."""
        lid = int(layer_idx)
        if lid not in self._layers:
            self._layers[lid] = _Agg()
        self._layers[lid].add(
            before_len=before_len,
            after_merge_len=after_merge_len,
            after_unmerge_len=after_unmerge_len,
            req_r=requested_r
        )

    # -------- report API --------
    def _iter_layer_indices(self) -> List[int]:
        if self._total_layers is not None:
            return list(range(self._total_layers))
        if not self._layers:
            return []
        return sorted(self._layers.keys())

    def report_table_str(self) -> str:
        """Return dataset-averaged table as a string. One row per layer."""
        rows: List[str] = []
        header = ["Layer", "Before", "AfterMerge", "AfterUnmerge", "Removed@Merge", "Removed@Final", "Req_r(avg)"]
        widths = [len(h) for h in header]

        # Build rows
        layer_ids = self._iter_layer_indices()
        for lid in layer_ids:
            agg = self._layers.get(lid, None)
            if agg is None or agg.count == 0:
                # No data recorded for this layer
                row = [str(lid), "0", "0", "0", "0", "0", "0"]
            else:
                before, after_m, after_u, req = agg.avg_tuple()
                removed_m = max(0, before - after_m)
                removed_u = max(0, before - after_u)
                row = [
                    str(lid),
                    str(before),
                    str(after_m),
                    str(after_u),
                    str(removed_m),
                    str(removed_u),
                    str(req),
                ]
            for i, c in enumerate(row):
                if len(c) > widths[i]:
                    widths[i] = len(c)
            rows.append(row)

        # Pretty print
        def join_line(cols: List[str]) -> str:
            return " | ".join(cols[i].rjust(widths[i]) for i in range(len(cols)))

        sep = "-+-".join("-" * w for w in widths)
        out = [join_line(header), sep]
        out += [join_line(r) for r in rows]
        return "\n".join(out)

    def dump_json(self, path: str) -> None:
        """Dump dataset-averaged stats to JSON."""
        data: List[Dict[str, Any]] = []
        for lid in self._iter_layer_indices():
            agg = self._layers.get(lid, None)
            if agg is None or agg.count == 0:
                data.append({
                    "layer": lid,
                    "count": 0,
                    "before_avg": 0,
                    "after_merge_avg": 0,
                    "after_unmerge_avg": 0,
                    "removed_merge_avg": 0,
                    "removed_final_avg": 0,
                    "req_r_avg": 0
                })
            else:
                before, after_m, after_u, req = agg.avg_tuple()
                data.append({
                    "layer": lid,
                    "count": agg.count,
                    "before_avg": before,
                    "after_merge_avg": after_m,
                    "after_unmerge_avg": after_u,
                    "removed_merge_avg": max(0, before - after_m),
                    "removed_final_avg": max(0, before - after_u),
                    "req_r_avg": req
                })
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"layers": data}, f, ensure_ascii=False, indent=2)


# Singleton-style module API
_stats = _TokenStats()

def reset() -> None:
    _stats.reset()

def set_total_layers(n: Optional[int]) -> None:
    _stats.set_total_layers(n)

def record(
    *,
    layer_idx: int,
    before_len: int,
    after_merge_len: int,
    after_unmerge_len: Optional[int] = None,
    requested_r: Optional[int] = None
) -> None:
    _stats.record(
        layer_idx=layer_idx,
        before_len=before_len,
        after_merge_len=after_merge_len,
        after_unmerge_len=after_unmerge_len,
        requested_r=requested_r
    )

def report_table_str() -> str:
    return _stats.report_table_str()

def dump_json(path: str) -> None:
    _stats.dump_json(path)
