# methods/tome_adapter/plugin.py
# Python 3.9 compatible. Comments in English only.

from typing import Optional, List, Any, Callable
import torch
import torch.nn as nn

# Origin ToMe merge operators
from tome.merge import (
    bipartite_soft_matching,  # (merge, unmerge)
    merge_wavg,               # size-weighted merge
    merge_source              # optional trace
)

# Optional dataset-avg token stats (your repo)
try:
    from core import token_stats as tstats
except Exception:
    tstats = None

# Unified schedule builder (main.py should set model._r_map; this is fallback)
try:
    from core.schedule import make_r_map
except Exception:
    make_r_map = None


# ---------------- utilities ----------------

def _find_transformer_blocks(model: nn.Module):
    """Return container holding transformer blocks (timm ViT/DeiT)."""
    for name in ["blocks", "stages"]:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def _ensure_r_map(model: nn.Module,
                  n_blocks: int,
                  r: Optional[int],
                  r_list: Optional[List[int]],
                  layers: Optional[List[int]]) -> List[int]:
    """Prefer model._r_map; else build from args."""
    r_map = getattr(model, "_r_map", None)
    if isinstance(r_map, list) and len(r_map) == n_blocks:
        return r_map
    if make_r_map is None:
        return [0] * n_blocks
    return make_r_map(n_blocks=n_blocks, r=r, r_list=r_list, layers=layers)


# ------------- patched Attention / Block / VT -------------

class _ToMeAttention(nn.Module):
    """Drop-in replacement for timm.models.vision_transformer.Attention."""
    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.__dict__["_wrapped"] = attn
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.proj_drop = getattr(attn, "proj_drop", nn.Identity())
        self.attn_drop = getattr(attn, "attn_drop", nn.Identity())
        self.num_heads = attn.num_heads
        self.scale = attn.scale

    def forward(self, x: torch.Tensor, size: torch.Tensor = None):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,h,N,d]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # proportional attention (size bias)
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # ToMe uses k-mean across heads as matching metric
        return x, k.mean(1)


class _ToMeBlock(nn.Module):
    """Drop-in replacement for timm.models.vision_transformer.Block.
    Apply ToMe between attn and mlp; record stats also when r==0.
    """
    def __init__(self, block: nn.Module, shared_info: dict) -> None:
        super().__init__()
        self.__dict__["_wrapped"] = block
        self._tome_info = shared_info

        self.norm1 = block.norm1
        self.attn = block.attn
        self.norm2 = block.norm2
        self.mlp = block.mlp

        self.drop_path = getattr(block, "drop_path", None)
        self.drop_path1 = getattr(block, "drop_path1", None)
        self.drop_path2 = getattr(block, "drop_path2", None)

    def _dp1(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path1 is not None:
            return self.drop_path1(x)
        if self.drop_path is not None:
            return self.drop_path(x)
        return x

    def _dp2(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path2 is not None:
            return self.drop_path2(x)
        if self.drop_path is not None:
            return self.drop_path(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Attention (metric for matching)
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._dp1(x_attn)

        # 2) Per-layer r from queue
        r_queue = self._tome_info["r"]
        req_r = int(r_queue.pop(0)) if isinstance(r_queue, list) and len(r_queue) > 0 else 0

        # 2.5) Length before merge (for stats)
        T0 = 0
        if isinstance(x, torch.Tensor) and x.dim() == 3:
            try:
                T0 = int(x.shape[1])
            except Exception:
                T0 = 0

        # 3) Merge (or baseline record if r==0)
        if req_r > 0:
            merge, _ = bipartite_soft_matching(
                metric,
                r=req_r,
                class_token=self._tome_info["class_token"],
                distill_token=self._tome_info["distill_token"],
            )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )

            x, self._tome_info["size"] = merge_wavg(
                merge, x, self._tome_info["size"]
            )

            # after length
            T1 = int(x.shape[1]) if isinstance(x, torch.Tensor) and x.dim() == 3 else T0

            if tstats is not None:
                try:
                    tstats.record(
                        layer_idx=self._tome_info["layer_ptr"],
                        before_len=T0,
                        after_merge_len=T1,
                        after_unmerge_len=T1,
                        requested_r=req_r
                    )
                except Exception:
                    pass
        else:
            # >>> Added: record baseline when no reduction on this layer <<<
            if tstats is not None:
                try:
                    tstats.record(
                        layer_idx=self._tome_info["layer_ptr"],
                        before_len=T0,
                        after_merge_len=T0,
                        after_unmerge_len=T0,
                        requested_r=0
                    )
                except Exception:
                    pass

        # 4) MLP
        x = x + self._dp2(self.mlp(self.norm2(x)))
        return x


def _make_tome_vt_class(base_cls):
    """Subclass VT to reset r-queue/size/source per forward (origin ToMe style)."""
    class _ToMeVisionTransformer(base_cls):
        def forward(self, *args: Any, **kw: Any) -> torch.Tensor:
            blocks = _find_transformer_blocks(self)
            n_blocks = len(list(blocks)) if blocks is not None else 0
            r_map = getattr(self, "_r_map", [0] * n_blocks)

            # init ToMe state for this forward
            self._tome_info["r"] = list(r_map)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["layer_ptr"] = 0

            out = super().forward(*args, **kw)

            # clear queue
            self._tome_info["r"] = []
            return out
    return _ToMeVisionTransformer


# ------------- public entry -------------

def apply_tome_with_hooks(
    model: nn.Module,
    *,
    r: Optional[int] = None,
    r_list: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    token_cap: str = "off",          # ToMe merges exactly r (50% cap inside)
    cls_protect: bool = True,
    match_feature: str = "kmean",
    prop_attn: bool = True,
    trace_source: bool = False,
    debug_token_stats: bool = False
) -> nn.Module:
    """
    Patch timm VisionTransformer to run ToMe in-block (attn->merge->mlp).
    Layer-wise r is taken from model._r_map, set by main via core.schedule.
    """
    blocks = _find_transformer_blocks(model)
    if blocks is None:
        if debug_token_stats:
            print("[ToMe] WARN: no transformer blocks found; skipping patch.")
        return model
    n_blocks = len(list(blocks))

    # Ensure r_map exists on the model
    r_map = _ensure_r_map(model, n_blocks, r, r_list, layers)
    setattr(model, "_r_map", r_map)

    # Shared ToMe state
    class_token = bool(getattr(model, "cls_token", None) is not None)
    distill_token = bool(getattr(model, "dist_token", None) is not None)
    model._tome_info = {
        "r": [],
        "size": None,
        "source": None,
        "trace_source": bool(trace_source),
        "prop_attn": bool(prop_attn),
        "class_token": bool(cls_protect and class_token),
        "distill_token": bool(cls_protect and distill_token),
        "layer_ptr": 0,
    }

    # Swap model class to reset ToMe state each forward (if VT)
    try:
        from timm.models.vision_transformer import VisionTransformer as _VT
        base_cls = model.__class__
        if issubclass(base_cls, _VT):
            model.__class__ = _make_tome_vt_class(base_cls)
    except Exception:
        pass

    # Replace Attention modules first
    try:
        from timm.models.vision_transformer import Attention as _Attn
        for m in model.modules():
            if isinstance(m, _Attn) and not isinstance(m, _ToMeAttention):
                parent = _find_parent(model, m)
                if parent is not None:
                    name = _child_name(parent, m)
                    setattr(parent, name, _ToMeAttention(m))
    except Exception:
        pass

    # Replace Blocks with ToMe blocks
    try:
        from timm.models.vision_transformer import Block as _Block
        for m in model.modules():
            if isinstance(m, _Block) and not isinstance(m, _ToMeBlock):
                parent = _find_parent(model, m)
                if parent is not None:
                    name = _child_name(parent, m)
                    setattr(parent, name, _ToMeBlock(m, model._tome_info))
    except Exception:
        pass

    # After-block hook to advance layer_ptr
    def _after_block_hook(module: nn.Module, inputs: Any, output: Any):
        if isinstance(module, _ToMeBlock):
            try:
                module._tome_info["layer_ptr"] += 1
            except Exception:
                pass
        return None

    handles = []
    for blk in model.modules():
        if isinstance(blk, _ToMeBlock):
            handles.append(blk.register_forward_hook(_after_block_hook))
    setattr(model, "_tome_handles", handles)

    if debug_token_stats:
        print("[ToMe] patched with origin operators; r_map len =", len(r_map))

    return model


# -------- helpers to navigate module tree --------

def _find_parent(root: nn.Module, child: nn.Module) -> Optional[nn.Module]:
    for m in root.modules():
        for name, sub in m.named_children():
            if sub is child:
                return m
    return None


def _child_name(parent: nn.Module, child: nn.Module) -> Optional[str]:
    for name, sub in parent.named_children():
        if sub is child:
            return name
    return None
