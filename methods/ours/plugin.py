# methods/ours/plugin.py
# Skeleton for "ours" method plugin (Python 3.9). Comments in English.

from typing import Dict, Any, List
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
    return sorted(set(out))


@register("ours")
class OursPlugin(TokenReducerPlugin):
    def __init__(self, cfg: Dict[str, Any]):
        super(OursPlugin, self).__init__(cfg)
        c = dict(cfg or {})

        # core knobs
        self.layer_spec = str(c.get("layers", "all"))
        self.r = int(c.get("r", 0))
        self.keep = c.get("keep", None)  # optional: string like "0.68,0.66,0.64"
        self.match_feature = str(c.get("match_feature", c.get("match-feature", "k")))  # 'k' or 'xnorm'

        # selector knobs (placeholders; wire real logic later)
        self.selector = str(c.get("selector", "hquota_ff"))
        self.hq_q = float(c.get("hq_q", 0.3))
        self.gamma = float(c.get("gamma", 0.0))
        self.cand_extra = int(c.get("cand_extra", 128))

        # merge/norm knobs (placeholders)
        self.merge_space = str(c.get("merge", "kv"))          # 'kv' or 'v'
        self.alpha = float(c.get("alpha", 0.15))              # push-lite strength
        self.beta0 = float(c.get("beta0", 0.5))               # cap base
        self.top_r = int(c.get("top_r", 0))                   # sparse push count
        self.l2_clip_tau = float(c.get("l2_clip_tau", 0.0))   # 0 to disable
        self.temp_eta = float(c.get("temp_eta", 0.0))
        self.size_delta = float(c.get("size_delta", 0.0))

        # schedule/logging
        self.schedule = str(c.get("schedule", "early_bias"))
        self.log_coverage = bool(c.get("log_coverage", False))

        # resolved at attach()
        self._layers = []
        self._installed = []
        self.model = None

    def attach(self, model):
        """Install identity-safe placeholders; real reduction will be added later."""
        self.model = model
        blocks = getattr(model, "blocks", [])
        self._layers = _parse_layers(self.layer_spec, len(blocks))

        # Identity guard: if no target layers or r<=0 and keep is None/1.0, do nothing.
        if len(self._layers) == 0:
            return
        if self.keep is None and self.r <= 0:
            return
        if isinstance(self.keep, str):
            ks = [float(x.strip()) for x in self.keep.split(",") if x.strip() != ""]
            if len(ks) == 0 or all(abs(k - 1.0) < 1e-9 for k in ks):
                return

        # Placeholder only: no-op to keep shapes; will be replaced by OursAttention later.
        for i, blk in enumerate(blocks):
            if i not in self._layers or not hasattr(blk, "attn"):
                continue
            # no-op wrapper
            original_attn = blk.attn

            class _NoOp(nn.Module):
                def __init__(self, inner):
                    super(_NoOp, self).__init__()
                    self.inner = inner
                def forward(self, x):
                    return self.inner(x)

            import torch.nn as nn  # local import to avoid unused at module load
            wrapper = _NoOp(original_attn)
            setattr(blk, "attn", wrapper)
            self._installed.append(i)

    def finalize(self):
        # No side-effects required for now.
        return
