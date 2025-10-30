# Python 3.9 compatible (no '|'), comments in English
from typing import List
import importlib
from core.registry import TokenReducerPlugin, register

def _parse_layers(spec: str, total: int) -> List[int]:
    if spec is None:
        return []
    s = str(spec).strip().lower()
    if s == "" or s == "all":
        return list(range(total))
    out = []
    for t in s.split(","):
        t = t.strip()
        if t:
            out.append(int(t))
    return out

@register("tome")
class TomePlugin(TokenReducerPlugin):  # if your base is TokenReducerPlugin, use that
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = dict(cfg or {})
        self.r = int(self.cfg.get("r", 0))
        self.layer_spec = str(self.cfg.get("layers", "all"))
        self.provider = str(self.cfg.get("match-feature", "k"))  # 'k' or 'xnorm'

    def attach(self, model):
        # import upstream `tome` from project root (copied from facebookresearch/ToMe)
        tome = importlib.import_module("tome")

        # find patch entrypoint; upstream exposes `tome.patch.timm`
        patch_mod = getattr(tome, "patch", None)
        if patch_mod is None or not hasattr(patch_mod, "timm"):
            raise RuntimeError("Upstream 'tome.patch.timm' not found. Make sure ./tome is copied.")

        # apply in-place patch to vendored timm model
        # some versions accept `trace_source`; pass False by default
        patch_mod.timm(model, trace_source=False)

        # global r flag (upstream ToMe reads `model.r`)
        setattr(model, "r", self.r)

        # per-block r (only enable on selected `layers`)
        blocks = getattr(model, "blocks", [])
        use = _parse_layers(self.layer_spec, len(blocks))
        for i, blk in enumerate(blocks):
            attn = getattr(blk, "attn", None)
            if hasattr(attn, "r"):
                setattr(attn, "r", self.r if i in use else 0)

        # best-effort provider hint (some ToMe versions read this dict)
        try:
            info = getattr(model, "tome_info", None)
            if info is None:
                info = {}
                setattr(model, "tome_info", info)
            info["provider"] = "k" if self.provider == "k" else "xnorm"
        except Exception:
            pass

    def finalize(self):
        # no-op; upstream ToMe already modified the model
        return
