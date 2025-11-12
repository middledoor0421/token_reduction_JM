# methods/ours/selectors/__init__.py
# Scalable one-place registration for selector plugins.
# Python 3.9 compatible. Comments in English only.

import importlib
from typing import Callable
from methods.ours.registry import register_selector

# Only keep names here. Add one more string when you add a new selector module.
SELECTOR_PLUGINS = (
    "hquota_ff",  # only this one for now
)

def _make_entry(mod) -> Callable:
    """
    Build a canonical entry: fn(x, mode, k, r, cls_protect, token_cap, debug, **kwargs)
    It adapts to whatever API the module exposes.
    """
    has_select  = hasattr(mod, "select")
    has_keep    = hasattr(mod, "select_keep_k")
    has_drop    = hasattr(mod, "select_drop_r")
    has_low     = hasattr(mod, "select_hquota_ff")  # legacy low-level

    def entry(x, mode, k, r, cls_protect=True, token_cap=True, debug=False, **kwargs):
        m = str(mode).lower()

        # Prefer a generic 'select'
        if has_select:
            return mod.select(
                x=x, mode=m, k=int(k), r=int(r),
                cls_protect=bool(cls_protect), token_cap=bool(token_cap),
                debug=bool(debug), **kwargs
            )

        # Mode-specific fallbacks
        if m == "keep" and has_keep:
            return mod.select_keep_k(
                x=x, k=int(k),
                cls_protect=bool(cls_protect), token_cap=bool(token_cap),
                debug=bool(debug), **kwargs
            )
        if m == "drop" and has_drop:
            return mod.select_drop_r(
                x=x, r=int(r),
                cls_protect=bool(cls_protect), token_cap=bool(token_cap),
                debug=bool(debug), **kwargs
            )

        # Low-level legacy (keep only)
        if has_low and m == "keep":
            K = int(k)
            return mod.select_hquota_ff(
                phi=x, K=K,
                quota_frac=float(kwargs.get("hq_q", 0.0)),
                cand_extra=int(kwargs.get("cand_extra", 0)),
                force_k=(not bool(token_cap)),
                cls_protect=bool(cls_protect),
                scores=kwargs.get("scores", None),
                mix_alpha=float(kwargs.get("mix_alpha", 0.5)),
                select_mode="keep",
            )

        raise RuntimeError(
            "Selector entry not found. Expected one of {select, select_keep_k/select_drop_r, select_hquota_ff}."
        )

    return entry

_pkg = __name__  # "methods.ours.selectors"
for _name in SELECTOR_PLUGINS:
    _mod = importlib.import_module(f"{_pkg}.{_name}")
    _entry = _make_entry(_mod)
    # Most selectors can optionally return (keep_idx, assign_idx).
    register_selector(_name, returns_assign=True)(_entry)
