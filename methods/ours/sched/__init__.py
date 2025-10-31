# methods/ours/sched/__init__.py
# Scheduler registry for "ours" method (Python 3.9).

from typing import Callable, Dict

_SCHED: Dict[str, Callable] = {}

def register_scheduler(name: str):
    def _wrap(fn: Callable):
        if name in _SCHED:
            raise ValueError("Scheduler already registered: %s" % name)
        _SCHED[name] = fn
        return fn
    return _wrap

def get_scheduler(name: str) -> Callable:
    if name not in _SCHED:
        raise ValueError("Unknown scheduler: %s (available: %s)" % (name, list(_SCHED.keys())))
    return _SCHED[name]

# bring built-ins
from .early_bias import compute_keep_and_r  # noqa: F401
