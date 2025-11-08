# methods/tome_adapter/plugin.py
# Strict ToMe adapter using local upstream `tome`:
# - Preserve Attn -> Merge -> MLP order (no extra merges here)
# - Inject r per block robustly (model / block / attn)
# - Hooks/logging only
from typing import Optional, List
import types
import torch

try:
    from core import token_stats as tstats
except Exception:
    tstats = None

# ---- registry import (our project) ------------------------------------------
try:
    from core.registry import register, TokenReducerPlugin
except Exception:
    # Fallback shims if registry is not importable
    def register(_name):
        def deco(cls):
            return cls
        return deco
    class TokenReducerPlugin(object):
        def __init__(self, cfg):
            self.cfg = cfg
        def attach(self, model):
            return model
        def finalize(self):
            pass
# ----------------------------------------------------------------------------


def apply_tome_with_hooks(
    model: torch.nn.Module,
    r: int,
    layers: Optional[List[int]],
    token_cap: str = "on",             # ignored for parity
    debug_token_stats: bool = False,
    cls_protect: bool = True,          # tagging only
    match_feature: str = "xnorm",      # tagging only
    prop_attn: bool = False            # forwarded to tome.patch if supported
) -> torch.nn.Module:
    """
    Patch timm ViT with upstream ToMe (in-place), inject per-block r, and
    log token lengths. No extra merges happen here.
    """
    import tome  # local upstream ToMe package

    # 1) Upstream ToMe patch (IN-PLACE; do NOT assign return)
    try:
        tome.patch.timm(model, prop_attn=bool(prop_attn))
    except TypeError:
        tome.patch.timm(model)

    # 2) Locate blocks (timm)
    blocks = None
    for name in ["blocks", "stages"]:
        if hasattr(model, name):
            blocks = getattr(model, name)
            break
    if blocks is None:
        if debug_token_stats:
            print("[ToMe][WARN] no transformer blocks found; nothing to patch.")
        return model

    # 3) Target layer set and per-layer r
    layer_set = set(layers) if layers is not None else None
    def _r_for_layer(L: int) -> int:
        return int(r) if (layer_set is None or L in layer_set) else 0

    # Helper: set r on possible attribute layouts
    def _set_r_attr(obj: object, r_val: int) -> bool:
        ok = False
        if obj is None:
            return ok
        for nm in ("r", "tome_r", "_tome_r"):
            if hasattr(obj, nm):
                try:
                    setattr(obj, nm, int(r_val))
                    ok = True
                except Exception:
                    pass
        for nm in ("tome_info", "_tome_info"):
            info = getattr(obj, nm, None)
            if info is not None:
                try:
                    setattr(info, "r", int(r_val))
                    ok = True
                except Exception:
                    pass
        return ok

    # 4) Wrap each block.forward: set r and log lengths (LOGGING ONLY)
    for L, blk in enumerate(list(blocks)):
        orig_forward = blk.forward

        def make_forward(_blk, _L, _orig):
            def new_forward(self, x):
                # before length
                T_prev = None
                if isinstance(x, torch.Tensor) and x.dim() == 3:
                    T_prev = int(x.shape[1])

                # set r for this block (robust: model / block / attn)
                r_here = _r_for_layer(_L)
                try:
                    setattr(model, "r", int(r_here))    # many ToMe versions read model.r
                except Exception:
                    pass
                _set_r_attr(_blk, r_here)
                if hasattr(_blk, "attn"):
                    _set_r_attr(_blk.attn, r_here)

                # run original forward (ToMe does Attn->Merge->MLP inside)
                out = _orig(x)

                # log after length
                if isinstance(out, torch.Tensor) and out.dim() == 3 and T_prev is not None:
                    T_cur = int(out.shape[1])
                    req = int(r) if r_here > 0 else None
                    if debug_token_stats:
                        print(f"[ToMe][L{_L}] before={T_prev}, after={T_cur}, delta={T_prev - T_cur}, req_r={req}")
                    if tstats is not None:
                        try:
                            tstats.record(
                                layer_idx=_L,
                                before_len=T_prev,
                                after_merge_len=T_cur,
                                after_unmerge_len=None,
                                requested_r=req
                            )
                        except Exception:
                            pass
                return out
            return new_forward

        # bind method (expects self,x); capture original bound method for call
        def _orig_bound(x, _orig=orig_forward):
            return _orig(x)

        wrapper = make_forward(blk, L, _orig_bound)
        blk.forward = types.MethodType(wrapper, blk)

        # initial tag (optional)
        r_init = _r_for_layer(L)
        _set_r_attr(blk, r_init)
        if hasattr(blk, "attn"):
            _set_r_attr(blk.attn, r_init)
        if debug_token_stats:
            rb = getattr(blk, "r", None)
            ra = getattr(getattr(blk, "attn", None), "r", None)
            print(f"[ToMe] init L{L}: r_blk={rb}, r_attn={ra}")

    # tags (for reference only)
    model.tome_requested_r = int(r)
    model.tome_layers = layer_set
    model.tome_debug = bool(debug_token_stats)
    model.tome_cls_protect = bool(cls_protect)
    model.tome_match_feature = str(match_feature)
    model.tome_prop_attn = bool(prop_attn)

    return model


# ---------------- registry plugin (so main/registry path works) --------------
@register("tome")
class TomePlugin(TokenReducerPlugin):
    """Registry-compatible ToMe plugin that calls the adapter above."""
    name = "tome"

    def __init__(self, cfg):
        super().__init__(cfg)

    def _parse_layers(self, v):
        if v is None:
            return None
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, str):
            s = v.strip()
            if len(s) == 0:
                return None
            return [int(p.strip()) for p in s.split(",") if p.strip() != ""]
        return None

    def attach(self, model):
        cfg = self.cfg if isinstance(self.cfg, dict) else {}
        return apply_tome_with_hooks(
            model=model,
            r=int(cfg.get("r", 13)),
            layers=self._parse_layers(cfg.get("layers", None)),
            token_cap=str(cfg.get("token_cap", "on")),
            debug_token_stats=bool(cfg.get("debug_token_stats", False)),
            cls_protect=bool(cfg.get("cls_protect", True)),
            match_feature=str(cfg.get("match_feature", "xnorm")),
            prop_attn=bool(cfg.get("prop_attn", False))
        )

    def finalize(self):
        pass
