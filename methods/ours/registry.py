# methods/ours/registry.py
# Python 3.9 compatible. Comments in English only.

from typing import Dict, Any, Callable, Optional

_SELECTOR_REG: Dict[str, Dict[str, Any]] = {}
_MERGE_REG: Dict[str, Dict[str, Any]] = {}

def register_selector(name: str, *, returns_assign: bool = True) -> Callable:
    """
    Decorator to register a selector under a canonical interface.
    Expected callable signature:
      fn(x, mode, k, r, cls_protect, token_cap, debug, **kwargs)
    It may also internally wrap keep/drop specialized APIs.
    """
    key = str(name).lower()

    def _wrap(fn: Callable) -> Callable:
        _SELECTOR_REG[key] = {"fn": fn, "returns_assign": bool(returns_assign)}
        return fn

    return _wrap


def register_merge(name: str, *, kind: str = "kv") -> Callable:
    """
    Decorator to register a merger under a canonical interface.
    Expected callable signature:
      fn(x, keep_idx, assign_idx, alpha=0.15, size=None, **kwargs) -> x_merged
    'kind' is informative ("kv" or "v"); not enforced here.
    """
    key = str(name).lower()

    def _wrap(fn: Callable) -> Callable:
        _MERGE_REG[key] = {"fn": fn, "kind": str(kind).lower()}
        return fn

    return _wrap


def get_selector(name: Optional[str]) -> Optional[Dict[str, Any]]:
    if name is None:
        return None
    return _SELECTOR_REG.get(str(name).lower(), None)


def get_merge(name: Optional[str]) -> Optional[Dict[str, Any]]:
    if name is None:
        return None
    return _MERGE_REG.get(str(name).lower(), None)
