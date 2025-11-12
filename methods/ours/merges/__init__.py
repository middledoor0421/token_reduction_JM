# methods/ours/merges/__init__.py
# Scalable one-place registration for merge plugins.
# Comments in English only. Python 3.9 compatible.

from importlib import import_module
from typing import Callable
from methods.ours.registry import register_merge

# Only keep names here. Add one more string when you add a new merge module.
MERGE_PLUGINS = [
    "kv_merge",   # only this one for now
    # "v_merge",  # example: add later
]

def _make_entry(mod) -> Callable:
    """
    Build a canonical entry:
      fn(x, keep_idx, assign_idx, alpha=0.15, size=None, **kwargs) -> x_merged
    It adapts to whatever function names the module exposes.
    """
    # Candidate function names to search inside the module
    fn_names = ("merge", "kv_merge", "merge_once", "merge_tokens", "run_merge")
    target = None
    for nm in fn_names:
        if hasattr(mod, nm):
            target = getattr(mod, nm)
            break
    if target is None:
        raise ImportError(
            f"[merges] No merge entry found in module {mod.__name__}. "
            f"Expected one of: {fn_names}"
        )

    def _entry(x, keep_idx, assign_idx, alpha=0.15, size=None, **kwargs):
        # Try several common signatures without inspecting function spec.
        # 1) Named 'assign'
        try:
            out = target(
                x=x, keep_idx=keep_idx, assign=assign_idx,
                alpha=float(alpha), **kwargs
            )
            if out is not None:
                return out
        except TypeError:
            pass

        # 2) Named 'assign_idx'
        try:
            out = target(
                x=x, keep_idx=keep_idx, assign_idx=assign_idx,
                alpha=float(alpha), **kwargs
            )
            if out is not None:
                return out
        except TypeError:
            pass

        # 3) Positional with alpha
        try:
            out = target(x, keep_idx, assign_idx, float(alpha))
            if out is not None:
                return out
        except TypeError:
            pass

        # 4) Minimal positional
        out = target(x, keep_idx, assign_idx)
        return out

    return _entry


# Register all MERGE_PLUGINS under convenient keys.
_pkg = __name__  # "methods.ours.merges"
for _name in MERGE_PLUGINS:
    _mod = import_module(f"{_pkg}.{_name}")
    _entry = getattr(_mod, "MERGE_ENTRY", None)  # optional explicit entry
    if callable(_entry):
        # If the module provides an explicit entry, use it.
        register_merge(_name, kind="kv")(_entry)
    else:
        # Otherwise, adapt whatever it exposes and register.
        entry = _make_entry(_mod)
        # Register with two aliases when the plugin is 'kv_merge'.
        if _name == "kv_merge":
            register_merge("kv", kind="kv")(entry)
            register_merge("kv_merge", kind="kv")(entry)
        else:
            # Generic key equals the module name.
            register_merge(_name, kind="v")(entry)
