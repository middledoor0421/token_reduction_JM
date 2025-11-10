# methods/tome_adapter/plugin.py
# Python 3.9 compatible. Comments in English only.
# Upstream ToMe parity adapter:
# - Preserve Attn -> Merge -> MLP inside upstream ToMe (no extra merges here)
# - Support per-layer r schedule (r_list) and/or scalar r + layers
# - Register LOGGING-ONLY hooks (dataset-avg stats are handled elsewhere)

from typing import Optional, List, Dict, Any
import types
import torch

try:
    from core import token_stats as tstats
except Exception:
    tstats = None

# ---- project registry (for legacy launcher compatibility) -------------------
try:
    from core.registry import register, TokenReducerPlugin
except Exception:
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
# -----------------------------------------------------------------------------


def apply_tome_with_hooks(
    model: torch.nn.Module,
    r: int,
    r_list: Optional[List[int]] = None,   # NEW: per-layer schedule overrides scalar r
    layers: Optional[List[int]] = None,   # optional target layer indices (ignored if r_list provided)
    token_cap: str = "on",                # ignored (keep upstream parity)
    debug_token_stats: bool = False,
    cls_protect: bool = True,             # tagging only
    match_feature: str = "xnorm",         # tagging only
    prop_attn: bool = False               # forwarded to tome.patch if supported
) -> torch.nn.Module:
    """
    Patch timm ViT with upstream ToMe (in-place), inject per-block r, and
    log token lengths per block. No extra merges or behavior changes.

    Args:
      model: timm ViT model.
      r: scalar merges per targeted block (used when r_list is None).
      r_list: per-layer merges list. If provided, overrides r/layers.
      layers: target layer indices for scalar-r mode. None means all blocks.
      token_cap: ignored to preserve parity.
      debug_token_stats: print before/after lengths per block if True.
      cls_protect/match_feature/prop_attn: kept for CLI parity.

    Returns:
      model (same object), patched and hooked.
    """
    import tome  # local upstream ToMe package

    # 1) Upstream ToMe patch (IN-PLACE). Do NOT assign return value.
    try:
        tome.patch.timm(model, prop_attn=bool(prop_attn))
    except TypeError:
        tome.patch.timm(model)

    # 2) Locate transformer blocks
    blocks = _find_blocks(model)
    if blocks is None:
        if debug_token_stats:
            print("[ToMe][WARN] no transformer blocks found ('blocks'/'stages').")
        return model

    n_blocks = len(list(blocks))

    # 3) Build per-layer r_map
    r_map: List[int] = []
    if r_list is not None and len(r_list) > 0:
        if len(r_list) < n_blocks:
            last = int(r_list[-1])
            r_map = [int(v) for v in r_list] + [last] * (n_blocks - len(r_list))
        else:
            r_map = [int(v) for v in r_list[:n_blocks]]
    else:
        layer_set = set(layers) if layers is not None else None
        for i in range(n_blocks):
            r_map.append(int(r) if (layer_set is None or i in layer_set) else 0)

    # 4) Initial r tagging (broad compatibility)
    _set_r_attr(model, 0)
    for i, blk in enumerate(list(blocks)):
        r_i = int(r_map[i])
        _set_r_attr(blk, r_i)
        attn = getattr(blk, "attn", None)
        if attn is not None:
            _set_r_attr(attn, r_i)
        if debug_token_stats:
            rb = getattr(blk, "r", None)
            ra = getattr(attn, "r", None) if attn is not None else None
            print(f"[ToMe][Init] L{i}: r_blk={rb}, r_attn={ra}")

    # 5) (Optional) verify patching
    if debug_token_stats:
        _verify_tome_patch(model, blocks)

    # 6) Wrap each block.forward to (a) set model.r for this block, (b) log lengths
    def make_forward(_blk, _L, _orig):
        def new_forward(self, x):
            # before length
            t_prev = None
            if isinstance(x, torch.Tensor) and x.dim() == 3:
                t_prev = int(x.shape[1])

            # set r for this block (some ToMe versions read model.r at runtime)
            r_here = int(r_map[_L])
            _set_r_attr(model, r_here)   # global
            _set_r_attr(_blk, r_here)    # block
            attn_local = getattr(_blk, "attn", None)
            if attn_local is not None:
                _set_r_attr(attn_local, r_here)

            # original forward (ToMe runs Attn -> Merge -> MLP inside)
            out = _orig(x)

            # after length + logging
            if isinstance(out, torch.Tensor) and out.dim() == 3 and t_prev is not None:
                t_cur = int(out.shape[1])
                req = r_here if r_here > 0 else None
                if debug_token_stats:
                    print(f"[ToMe][L{_L}] before={t_prev}, after={t_cur}, delta={t_prev - t_cur}, req_r={req}")
                if tstats is not None:
                    try:
                        tstats.record(
                            layer_idx=_L,
                            before_len=t_prev,
                            after_merge_len=t_cur,
                            after_unmerge_len=None,
                            requested_r=req
                        )
                    except Exception:
                        pass
            return out
        return new_forward

    for i, blk in enumerate(list(blocks)):
        orig_forward = blk.forward
        # bind wrapper as a method; call orig via closure to avoid recursion
        def _orig_bound(x, _orig=orig_forward):
            return _orig(x)
        wrapped = make_forward(blk, i, _orig_bound)
        blk.forward = types.MethodType(wrapped, blk)

    # Expose schedule for inspection (optional)
    model._tome_r_map = r_map
    model._tome_debug = bool(debug_token_stats)

    return model


# ------------------------------ helpers --------------------------------------

def _find_blocks(model: torch.nn.Module):
    for name in ["blocks", "stages"]:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def _set_r_attr(obj: object, r_val: int) -> None:
    """Set 'r' on various ToMe versions (model/block/attn) safely."""
    if obj is None:
        return
    for nm in ("r", "tome_r", "_tome_r"):
        try:
            setattr(obj, nm, int(r_val))
        except Exception:
            pass
    for nm in ("tome_info", "_tome_info"):
        info = getattr(obj, nm, None)
        if info is not None:
            try:
                setattr(info, "r", int(r_val))
            except Exception:
                pass


def _verify_tome_patch(model: torch.nn.Module, blocks) -> None:
    try:
        b0 = list(blocks)[0]
        attn0 = getattr(b0, "attn", None)
        print("[ToMe][Check] block0.attn =", type(attn0))
        print("[ToMe][Check] model.r   =", getattr(model, "r", None),
              " block0.r =", getattr(b0, "r", None))
    except Exception:
        pass


# ------------------------- registry plugin (legacy path) ---------------------

@register("tome")
class TomePlugin(TokenReducerPlugin):
    """Registry-compatible ToMe plugin that calls the adapter above."""
    name = "tome"

    def __init__(self, cfg):
        super().__init__(cfg)

    @staticmethod
    def _parse_layers(v):
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

    @staticmethod
    def _parse_r_list(v):
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
            r_list=self._parse_r_list(cfg.get("r_list", None)),     # NEW
            layers=self._parse_layers(cfg.get("layers", None)),
            token_cap=str(cfg.get("token_cap", "on")),
            debug_token_stats=bool(cfg.get("debug_token_stats", False)),
            cls_protect=bool(cfg.get("cls_protect", True)),
            match_feature=str(cfg.get("match_feature", "xnorm")),
            prop_attn=bool(cfg.get("prop_attn", False))
        )

    def finalize(self):
        pass
