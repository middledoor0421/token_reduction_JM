# methods/ours/inblock_patch.py
# Python 3.9 compatible. Comments in English only.

from typing import List, Optional, Callable, Any, Dict
import types
import torch
import torch.nn as nn

from .attn import OursAttention


class _PropAttnWrapper(nn.Module):
    """
    Thin wrapper around timm Attention to optionally add a key-side log-size bias
    to attention logits (proportional attention). If `model_ctx["_key_log_bias"]`
    is set to a [B, Nk] tensor, we add it to logits along the key dimension.
    """
    def __init__(self, attn_mod: nn.Module, model_ctx: Dict[str, Any]) -> None:
        super().__init__()
        self.attn = attn_mod
        self.ctx = model_ctx

        # Cache handles to original submodules/params used by timm Attention
        # We expect attributes: qkv, num_heads, head_dim, attn_drop, proj, proj_drop, scale
        for name in ["qkv", "num_heads", "attn_drop", "proj", "proj_drop"]:
            if not hasattr(self.attn, name):
                raise AttributeError("Unsupported Attention module: missing '{}'".format(name))
        # head_dim / scale may differ across timm versions
        if hasattr(self.attn, "head_dim"):
            self._get_head_dim = lambda C: int(self.attn.head_dim)
            self._scale = float(getattr(self.attn, "scale", 1.0 / (float(self.attn.head_dim) ** 0.5)))
        else:
            # Fallback: infer head_dim from qkv weight shape at runtime
            self._get_head_dim = None
            self._scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.attn.qkv(x)  # [B, N, 3*C]
        # reshape to [3, B, heads, N, head_dim]
        # infer head_dim if needed
        if self._get_head_dim is None:
            # assume split equally into 3, then per-head
            head_dim = C // getattr(self.attn, "num_heads")
            scale = 1.0 / float(head_dim) ** 0.5
        else:
            head_dim = self._get_head_dim(C)
            scale = self._scale if self._scale is not None else 1.0 / float(head_dim) ** 0.5

        qkv = qkv.reshape(B, N, 3, self.attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, heads, Nq, Nk]

        # Optional proportional attention: add log-size bias on the key dimension
        key_log_bias = self.ctx.get("_key_log_bias", None)
        if key_log_bias is not None:
            # key_log_bias: [B, Nk] -> [B, 1, 1, Nk]
            if isinstance(key_log_bias, torch.Tensor) and key_log_bias.dim() == 2 and key_log_bias.size(0) == B and key_log_bias.size(1) == k.size(-2):
                attn_logits = attn_logits + key_log_bias.unsqueeze(1).unsqueeze(1)
            # Consume once
            self.ctx["_key_log_bias"] = None

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)

        x_out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x_out = self.attn.proj(x_out)
        x_out = self.attn.proj_drop(x_out)
        return x_out


def _wrap_block_forward(
    block: nn.Module,
    layer_idx: int,
    reducer: OursAttention,
    model_ctx: Dict[str, Any],
    requested_r_fn: Callable[[int], Optional[int]],
    enable_prop_attn: bool
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a new forward function for timm Block:
      x = x + drop_path(attn(norm1(x)))
      x = reduce(x)          # <--- inserted here (Attn -> Merge -> MLP)
      x = x + drop_path(mlp(norm2(x)))
    Also, if proportional attention is enabled, set model_ctx["_key_log_bias"]
    from reducer's last sizes so that next block's Attention can read it.
    """
    # Optional layer-scale params supported by certain timm blocks
    gamma1 = getattr(block, "gamma_1", None)
    gamma2 = getattr(block, "gamma_2", None)

    # DropPath or Identity
    drop_path = getattr(block, "drop_path", None)

    # Ensure Attention wrapper when proportional attention is requested
    if enable_prop_attn and not isinstance(block.attn, _PropAttnWrapper):
        block.attn = _PropAttnWrapper(block.attn, model_ctx)

    def new_forward(x: torch.Tensor) -> torch.Tensor:
        # Branch 1: Attention
        residual = x
        x_norm = block.norm1(x)
        x_attn = block.attn(x_norm)  # possibly wrapped
        if gamma1 is not None:
            x_attn = gamma1 * x_attn
        if drop_path is not None:
            x_attn = drop_path(x_attn)
        x = residual + x_attn  # post-attn residual output

        # Insert Ours reduction right here (Attn -> Reduce -> MLP)
        req_r = requested_r_fn(layer_idx)
        x = reducer(
            x,
            layer_idx=layer_idx,
            requested_r=req_r
        )

        # If proportional attention is enabled, set bias for NEXT block
        if enable_prop_attn:
            last_log_sizes = getattr(reducer, "last_log_sizes", None)
            if isinstance(last_log_sizes, torch.Tensor):
                # Shape [B, K]; store in model context to be consumed by next attention
                model_ctx["_key_log_bias"] = last_log_sizes

        # Branch 2: MLP
        residual2 = x
        x_norm2 = block.norm2(x)
        x_mlp = block.mlp(x_norm2)
        if gamma2 is not None:
            x_mlp = gamma2 * x_mlp
        if drop_path is not None:
            x_mlp = drop_path(x_mlp)
        x = residual2 + x_mlp
        return x

    return new_forward


def apply_ours_inblock(
    model: nn.Module,
    *,
    r: int,
    layers: Optional[List[int]],
    token_cap: str = "on",
    debug_token_stats: bool = False,
    # selector/merge knobs (same meaning as in OursAttention)
    hq_quota: float = 0.0,
    cand_extra: int = 0,
    merge_kind: str = "v",
    alpha: float = 0.0,
    size_delta: float = 0.0,
    match_feature: str = "xnorm",
    enable_unmerge: bool = False,
    prop_attn: bool = False
) -> nn.Module:
    """
    Patch timm ViT to reduce tokens inside each block between Attention and MLP.
    """
    # Build a single reducer instance reused across blocks
    reducer = OursAttention(
        token_cap=token_cap,
        debug_token_stats=debug_token_stats,
        cls_protect=True,
        enable_unmerge=enable_unmerge,
        quota_frac=hq_quota,
        cand_extra=cand_extra,
        merge_kind=merge_kind,
        alpha=alpha,
        size_delta=size_delta,
        match_feature=match_feature
    )

    # Shared model-level context to pass proportional attention bias to next block
    model_ctx: Dict[str, Any] = {"_key_log_bias": None}

    # Helper to decide r for a given layer
    layer_set = set(layers) if layers is not None else None
    def _req_r_fn(L: int) -> Optional[int]:
        if layer_set is None:
            return int(r)
        return int(r) if L in layer_set else None

    # Find transformer blocks (timm style: model.blocks or model.stages)
    blocks = None
    for name in ["blocks", "stages"]:
        if hasattr(model, name):
            blocks = getattr(model, name)
            break
    if blocks is None:
        raise AttributeError("Could not find transformer blocks on the model (expected 'blocks' or 'stages').")

    # Monkey-patch each block's forward with inserted reduction point
    for L, blk in enumerate(blocks):
        # Sanity check: block should have norm1, attn, norm2, mlp
        for comp in ["norm1", "attn", "norm2", "mlp"]:
            if not hasattr(blk, comp):
                raise AttributeError("Block {} is missing required component '{}'".format(L, comp))

        new_fwd = _wrap_block_forward(
            block=blk,
            layer_idx=L,
            reducer=reducer,
            model_ctx=model_ctx,
            requested_r_fn=_req_r_fn,
            enable_prop_attn=bool(prop_attn)
        )
        blk.forward = types.MethodType(new_fwd, blk) if False else new_fwd  # bound-like replacement

    return model
