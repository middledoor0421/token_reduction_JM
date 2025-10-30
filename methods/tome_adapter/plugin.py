# methods/tome_adapter/plugin.py
from typing import List
import importlib
from core.registry import TokenReducerPlugin, register   # ensure base name matches your registry

def _parse_layers(spec, total):
    s = str(spec).strip().lower() if spec is not None else ""
    if s == "" or s == "all":
        return list(range(total))
    out = []
    for t in s.split(","):
        t = t.strip()
        if t:
            out.append(int(t))
    return out

@register("tome")
class TomePlugin(TokenReducerPlugin):  # fix base class name
    def __init__(self, cfg):
        super(TomePlugin, self).__init__(cfg)
        self.cfg = dict(cfg or {})
        self.r = int(self.cfg.get("r", 0))
        self.layer_spec = str(self.cfg.get("layers", "all"))
        self.provider = str(self.cfg.get("match-feature", "k"))  # 'k' or 'xnorm'

    def attach(self, model):
        tome = importlib.import_module("tome")
        patch = getattr(tome, "patch", None)
        if patch is None or not hasattr(patch, "timm"):
            raise RuntimeError("Upstream 'tome.patch.timm' not found. Ensure ./tome is present.")
        # Apply upstream patch
        patch.timm(model, trace_source=False)

        # Global r
        setattr(model, "r", self.r)

        # Per-layer r
        blocks = getattr(model, "blocks", [])
        for i, blk in enumerate(blocks):
            attn = getattr(blk, "attn", None)
            if hasattr(attn, "r"):
                setattr(attn, "r", self.r if i in _parse_layers(self.layer_spec, len(blocks)) else 0)

        # Optional provider hint (ignored if not used upstream)
        info = getattr(model, "tome_info", None) or {}
        info["provider"] = "k" if self.provider == "k" else "xnorm"
        setattr(model, "tome_info", info)

    def finalize(self):
        return
