# methods/ours/selectors/__init__.py
# Lightweight selector registry for "ours" method.

from typing import Callable, Dict

_SELECTORS: Dict[str, Callable] = {}

def register_selector(name: str):
    def _wrap(fn: Callable):
        if name in _SELECTORS:
            raise ValueError("Selector already registered: %s" % name)
        _SELECTORS[name] = fn
        return fn
    return _wrap

def get_selector(name: str) -> Callable:
    if name not in _SELECTORS:
        raise ValueError("Unknown selector: %s (available: %s)" % (name, list(_SELECTORS.keys())))
    return _SELECTORS[name]

# bring built-ins
from .hquota_ff import select_hquota_ff  # noqa: F401
