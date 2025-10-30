# core/registry.py
# Minimal plugin registry for token-reduction methods.

from typing import Dict, Type

class TokenReducerPlugin:
    """Base interface for token-reduction plugins."""
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})

    def attach(self, model):
        """Install hooks / patch modules on the given model."""
        raise NotImplementedError

    def finalize(self):
        """Cleanup or print summary after evaluation."""
        pass


_REGISTRY: Dict[str, Type[TokenReducerPlugin]] = {}

def register(name: str):
    """Decorator to register a plugin class under a method name."""
    def _wrap(cls):
        if name in _REGISTRY:
            raise ValueError("Plugin already registered: %s" % name)
        _REGISTRY[name] = cls
        return cls
    return _wrap

def create_plugin(name: str, cfg: dict) -> TokenReducerPlugin:
    if name not in _REGISTRY:
        raise ValueError("Unknown method: %s (registered: %s)" % (name, list(_REGISTRY.keys())))
    return _REGISTRY[name](cfg)

def available_methods():
    return sorted(_REGISTRY.keys())
