# core/token_stats.py
# Python 3.9 compatible. Comments in English only.

from typing import Any, Dict, List, Optional
import json
import datetime


class _TokenStatsRecorder:
    """
    A lightweight recorder to track per-layer token-length changes.
    Typical call sequence per block:
      - record(layer_idx, before_len, after_merge_len, after_unmerge_len, requested_r)
    At the end of a run:
      - report_table_str() for console table
      - dump_json(path) to persist structured stats
    """

    def __init__(self) -> None:
        self.total_layers: Optional[int] = None   # NEW
        self.reset()
    def reset(self) -> None:
        # Reset records but keep total_layers as-is so caller can set once per run.
        self.records: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {
            "created_at": datetime.datetime.utcnow().isoformat() + "Z"
        }

    def set_total_layers(self, n: Optional[int]) -> None:
        """
        Optionally set total number of layers so that the final report prints
        exactly this many rows (filling missing layers with zeros).
        """
        if n is None:
            self.total_layers = None
        else:
            self.total_layers = int(n)

    def record(
        self,
        layer_idx: int,
        before_len: int,
        after_merge_len: Optional[int],
        after_unmerge_len: Optional[int],
        requested_r: Optional[int],
    ) -> None:
        """
        Store one layer's token length info.
        - If a stage is not applicable, pass None and it will be stored as None.
        - If there is no reduction on a layer, after_* may equal before_len.
        """
        rec: Dict[str, Any] = {
            "layer": int(layer_idx),
            "before": int(before_len),
            "after_merge": int(after_merge_len) if after_merge_len is not None else None,
            "after_unmerge": int(after_unmerge_len) if after_unmerge_len is not None else None,
            "requested_r": int(requested_r) if requested_r is not None else None,
        }

        # Compute actual removed counts if possible
        removed_merge = None
        removed_final = None
        if after_merge_len is not None:
            removed_merge = before_len - after_merge_len
        if after_unmerge_len is not None:
            removed_final = before_len - after_unmerge_len

        rec["removed_merge"] = removed_merge
        rec["removed_final"] = removed_final
        self.records.append(rec)

    def summarize_per_layer(self) -> List[Dict[str, Any]]:
        """
        Aggregate all recorded entries across all batches and return
        per-layer averages. If total_layers is set, include rows for every
        layer in [0, total_layers-1], filling missing ones with zeros.
        """
        # Accumulate sums per layer
        acc: Dict[int, Dict[str, Any]] = {}
        for r in self.records:
            L = int(r["layer"])
            if L not in acc:
                acc[L] = {
                    "n": 0,
                    "before_sum": 0,
                    "after_merge_sum": 0,
                    "after_unmerge_sum": 0,
                    "have_after_unmerge": False,
                    "req": None,
                    "rm_merge_sum": 0,
                    "rm_final_sum": 0,
                }
            acc[L]["n"] += 1
            acc[L]["before_sum"] += int(r["before"])
            am = r.get("after_merge")
            if am is None:
                am = int(r["before"])  # treat as no change when missing
            acc[L]["after_merge_sum"] += int(am)
            au = r.get("after_unmerge")
            if au is not None:
                acc[L]["after_unmerge_sum"] += int(au)
                acc[L]["have_after_unmerge"] = True
            rm_m = r.get("removed_merge")
            if rm_m is not None:
                acc[L]["rm_merge_sum"] += int(rm_m)
            rm_f = r.get("removed_final")
            if rm_f is not None:
                acc[L]["rm_final_sum"] += int(rm_f)
            if acc[L]["req"] is None and r.get("requested_r") is not None:
                acc[L]["req"] = int(r["requested_r"])

        # Determine how many rows to output
        if self.total_layers is not None:
            max_L = self.total_layers - 1
        else:
            max_L = max(acc.keys()) if acc else -1

        rows: List[Dict[str, Any]] = []
        for L in range(max_L + 1):
            if L in acc:
                n = max(1, acc[L]["n"])
                row = {
                    "layer": L,
                    "before": acc[L]["before_sum"] // n,
                    "after_merge": acc[L]["after_merge_sum"] // n,
                    "after_unmerge": (acc[L]["after_unmerge_sum"] // n) if acc[L]["have_after_unmerge"] else None,
                    "requested_r": acc[L]["req"],
                    "removed_merge": acc[L]["rm_merge_sum"] // n,
                    "removed_final": (acc[L]["rm_final_sum"] // n) if acc[L]["have_after_unmerge"] else None,
                }
            else:
                # No records for this layer: report zeros so user can see no reduction occurred
                row = {
                    "layer": L,
                    "before": None,
                    "after_merge": None,
                    "after_unmerge": None,
                    "requested_r": None,
                    "removed_merge": 0,
                    "removed_final": 0,
                }
            rows.append(row)
        return rows

    def as_table_str(self) -> str:
        rows = self.summarize_per_layer()
        if not rows:
            return "[TokenStats] No records."

        # Header
        header = [
            "Layer",
            "Before",
            "AfterMerge",
            "AfterUnmerge",
            "Requested_r",
            "Removed@Merge",
            "Removed@Final",
        ]
        col_widths = [len(h) for h in header]

        def to_str(v: Any) -> str:
            return "None" if v is None else str(v)

        # Collect rows and compute column widths
        data: List[List[str]] = []
        for r in rows:
            row = [
                to_str(r.get("layer")),
                to_str(r.get("before")),
                to_str(r.get("after_merge")),
                to_str(r.get("after_unmerge")),
                to_str(r.get("requested_r")),
                to_str(r.get("removed_merge")),
                to_str(r.get("removed_final")),
            ]
            data.append(row)
            for i, cell in enumerate(row):
                if len(cell) > col_widths[i]:
                    col_widths[i] = len(cell)

        def fmt_row(cells: List[str]) -> str:
            return " | ".join(cells[i].ljust(col_widths[i]) for i in range(len(cells)))

        sep = "-+-".join("-" * w for w in col_widths)

        lines: List[str] = []
        lines.append(fmt_row(header))
        lines.append(sep)
        for row in data:
            lines.append(fmt_row(row))

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "records": self.summarize_per_layer(),
        }

    def dump_json(self, path: str) -> None:
        payload = self.to_json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# Module-level singleton
_recorder = _TokenStatsRecorder()

# 모듈 레벨 API (새 함수 추가)
def set_total_layers(n: Optional[int]) -> None:
    _recorder.set_total_layers(n)

# Public API
def reset() -> None:
    _recorder.reset()


def record(
    layer_idx: int,
    before_len: int,
    after_merge_len: Optional[int],
    after_unmerge_len: Optional[int],
    requested_r: Optional[int],
) -> None:
    _recorder.record(layer_idx, before_len, after_merge_len, after_unmerge_len, requested_r)


def report_table_str() -> str:
    return _recorder.as_table_str()


def dump_json(path: str) -> None:
    _recorder.dump_json(path)
