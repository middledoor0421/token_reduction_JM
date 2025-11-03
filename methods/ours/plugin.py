# methods/ours/plugin.py
# Attach OursAttention to selected blocks. Comments in English only.

from typing import Dict, Any, List
import torch.nn as nn
from core.registry import TokenReducerPlugin, register
from .attn import OursAttention


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
        self.layer_spec = str(c.get("layers", "all"))
        self.cfg = c
        self._layers = []
        self._installed = []

    def attach(self, model: nn.Module):
        self.model = model
        blocks = getattr(model, "blocks", [])
        self._layers = _parse_layers(self.layer_spec, len(blocks))
        if len(self._layers) == 0:
            return

        for i, blk in enumerate(blocks):
            if i not in self._layers or not hasattr(blk, "attn"):
                continue
            blk.attn = _WrapWithLayerIndex(blk.attn, i, len(blocks), self.cfg)
            self._installed.append(i)

    def finalize(self):
        return


class _WrapWithLayerIndex(nn.Module):
    """Small shim to pass (layer_idx, total_layers) into OursAttention forward."""
    def __init__(self, inner_attn: nn.Module, layer_idx: int, total_layers: int, cfg: Dict[str, Any]):
        super(_WrapWithLayerIndex, self).__init__()
        self.attn = OursAttention(inner_attn, cfg)
        self.layer_idx = layer_idx
        self.total_layers = total_layers

    def forward(self, x, attn_mask=None, **kwargs):
        return self.attn(x,
                         attn_mask=attn_mask,  # ← 전달
                         layer_idx=self.layer_idx,
                         total_layers=self.total_layers,
                         **kwargs)