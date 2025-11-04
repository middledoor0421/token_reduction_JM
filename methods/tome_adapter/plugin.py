# methods/tome_adapter/plugin.py
# Python 3.9 compatible. Comments in English only.

from typing import List, Optional
import torch
import torch.nn.functional as F
from core.registry import register, TokenReducerPlugin  # 핵심: core 경로 사용

try:
    from core import token_stats as tstats
except Exception:
    tstats = None


def apply_tome_with_hooks(
    model: torch.nn.Module,
    r: int,
    layers: Optional[List[int]],
    token_cap: str = "on",            # "on": allow <r, "off": force exactly r
    debug_token_stats: bool = False,  # print per-block lengths
    cls_protect: bool = True,         # exclude CLS (idx 0) from extra merges
    match_feature: str = "xnorm",     # "k" or "xnorm" (kept for parity with your CLI)
    prop_attn: bool = False           # propagate attention flag (if used by upstream)
) -> torch.nn.Module:
    """
    Patch timm model with upstream ToMe and register per-block hooks.
    When token_cap == "off", if a layer removed < r tokens, we greedily merge
    extra pairs post-forward until the delta equals r (best-effort).
    Also stores match_feature/prop_attn on model for parity with your previous CLI.
    """
    import tome  # upstream ToMe
    # Standard upstream patch: modify timm blocks in-place
    tome.patch.timm(model)

    # Persist basic configs on model for reference and potential downstream use
    model.tome_requested_r = int(r)
    model.tome_layers = set(layers) if layers is not None else None
    model.tome_token_cap = str(token_cap).lower()
    model.tome_debug = bool(debug_token_stats)
    model.tome_cls_protect = bool(cls_protect)

    # Keep compatibility flags your code may read elsewhere
    model.tome_match_feature = str(match_feature)  # "k" or "xnorm"
    model.tome_prop_attn = bool(prop_attn)

    # Discover transformer blocks (timm style)
    blocks = None
    for name in ["blocks", "stages"]:
        if hasattr(model, name):
            blocks = getattr(model, name)
            break
    if blocks is None:
        if model.tome_debug:
            print("[ToMe][WARN] no transformer blocks found for hooks.")
        return model

    # Hook state
    state = {"prev_len": None, "layer_idx": -1}

    def _is_target_layer(L: int) -> bool:
        if model.tome_layers is None:
            return True
        return L in model.tome_layers

    def pre_hook(mod, inp):
        # inp: tuple, token tensor at inp[0] with shape [B, T, C]
        x = inp[0] if isinstance(inp, tuple) else inp
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return None
        _, T, _ = x.shape
        state["prev_len"] = int(T)
        state["layer_idx"] += 1
        return None

    def post_hook(mod, _inp, out):
        # out: token tensor [B, T2, C]
        if not isinstance(out, torch.Tensor) or out.dim() != 3:
            return out

        L = state["layer_idx"]
        T_prev = int(state["prev_len"])
        T_cur = int(out.shape[1])
        req_r = model.tome_requested_r if _is_target_layer(L) else None
        delta = T_prev - T_cur

        # Debug print before any enforcement
        if model.tome_debug:
            print(f"[ToMe][L{L}] before={T_prev}, after={T_cur}, delta={delta}, req_r={req_r}")

        # If token-cap is off, try to enforce exact r on target layers
        if req_r is not None and model.tome_token_cap == "off":
            need = int(req_r) - int(delta)
            if need > 0 and T_cur > 1:
                out = _greedy_extra_merge(
                    out,
                    need=need,
                    cls_protect=model.tome_cls_protect,
                )
                if model.tome_debug:
                    T_new = int(out.shape[1])
                    print(f"[ToMe][L{L}] enforced extra_merge need={need} -> after={T_new}, delta={T_prev - T_new}")

        # Record stats (after enforcement)
        if tstats is not None:
            try:
                T_final = int(out.shape[1])
                tstats.record(
                    layer_idx=L,
                    before_len=T_prev,
                    after_merge_len=T_final,      # ToMe path has no unmerge
                    after_unmerge_len=None,
                    requested_r=req_r,
                )
            except Exception:
                pass

        return out

    # Register hooks for each block
    for blk in blocks:
        blk.register_forward_pre_hook(lambda m, i: pre_hook(m, i))
        blk.register_forward_hook(lambda m, i, o: post_hook(m, i, o))

    return model


def _greedy_extra_merge(
    x: torch.Tensor,
    need: int,
    cls_protect: bool = True,
) -> torch.Tensor:
    """
    Perform additional greedy merges on the token sequence x to remove `need` tokens.
    Simple V-merge approximation: choose the weakest token (low L2 norm),
    merge it into its most similar neighbor (cosine), and drop the victim.
    This runs per batch independently and repeats `need` times.
    """
    # x: [B, T, C]
    B, T, C = x.shape
    if need <= 0 or T <= 1:
        return x
    need = min(need, T - 1)  # cannot remove more than T-1

    xs = x
    for _ in range(need):
        # 1) score by L2 norm (lower is weaker)
        norm = xs.pow(2).sum(dim=-1).sqrt()  # [B, T]

        # 2) choose a victim idx per batch (exclude CLS if needed)
        idx_victim = []
        for b in range(B):
            candidate = norm[b].clone()
            if cls_protect and candidate.shape[0] > 0:
                candidate[0] = float("inf")
            v = int(torch.argmin(candidate).item())
            idx_victim.append(v)
        idx_victim = torch.tensor(idx_victim, device=xs.device, dtype=torch.long)  # [B]

        # 3) find nearest neighbor by cosine similarity
        feat = F.normalize(xs, dim=-1)  # [B, T, C]
        vf = torch.gather(feat, dim=1, index=idx_victim.view(B, 1, 1).expand(B, 1, C))  # [B,1,C]
        sim = (feat * vf).sum(dim=-1)
        #sim = torch.einsum("btc,b1c->bt", feat, vf)  # [B, T]
        for b in range(B):
            sim[b, idx_victim[b]] = -1e9
            # Allow CLS as neighbor; merging into CLS is typically safe.
        idx_neighbor = torch.argmax(sim, dim=1)  # [B]

        # 4) size-weighted average in V-space (approx; no explicit size buffer)
        victim_w = torch.gather(norm, 1, idx_victim.view(B, 1)).view(B, 1, 1) + 1e-6
        neigh_w = torch.gather(norm, 1, idx_neighbor.view(B, 1)).view(B, 1, 1) + 1e-6
        total_w = victim_w + neigh_w

        vf_raw = torch.gather(xs, 1, idx_victim.view(B, 1, 1).expand(B, 1, C))
        nb_raw = torch.gather(xs, 1, idx_neighbor.view(B, 1, 1).expand(B, 1, C))
        merged = (victim_w * vf_raw + neigh_w * nb_raw) / total_w  # [B,1,C]

        # 5) build new sequence: replace neighbor with merged, drop victim (order-stable)
        keep_mask = torch.ones(B, xs.shape[1], dtype=torch.bool, device=xs.device)
        for b in range(B):
            keep_mask[b, idx_victim[b]] = False

        xs_new = []
        for b in range(B):
            row = xs[b][keep_mask[b]]  # [T-1, C]
            nb = int(idx_neighbor[b].item())
            shift = 1 if nb > int(idx_victim[b].item()) else 0
            nb_new = nb - shift
            row = row.clone()
            row[nb_new:nb_new+1] = merged[b]
            xs_new.append(row)
        xs = torch.stack(xs_new, dim=0)  # [B, T-1, C]

    return xs

# 중복 등록 시 에러를 피하고 싶다면 안전 래퍼를 써도 됨.
def _safe_register(name):
    def deco(cls):
        try:
            return register(name)(cls)
        except ValueError:
            # Already registered; just return the class as-is.
            return cls
    return deco

@_safe_register("tome")
class TomePlugin(TokenReducerPlugin):
    """
    Backward-compatible wrapper so that legacy registry-based launchers keep working.
    """
    name = "tome"

    def __init__(self, cfg):
        # cfg는 dict, core/registry.TokenReducerPlugin 시그니처에 맞춤
        super().__init__(cfg)

    def attach(self, model):
        """
        Install ToMe on the given model according to cfg, then return the model.
        """
        r = int(self.cfg.get("r", 13))
        layers = self.cfg.get("layers", None)  # e.g., [0,1,2]
        token_cap = str(self.cfg.get("token_cap", "on")).lower()
        debug = bool(self.cfg.get("debug_token_stats", False))
        cls_protect = bool(self.cfg.get("cls_protect", True))
        match_feature = str(self.cfg.get("match_feature", "xnorm"))
        prop_attn = bool(self.cfg.get("prop_attn", False))

        return apply_tome_with_hooks(
            model=model,
            r=r,
            layers=layers,
            token_cap=token_cap,
            debug_token_stats=debug,
            cls_protect=cls_protect,
            match_feature=match_feature,
            prop_attn=prop_attn,
        )

    def finalize(self):
        # 선택: 요약 출력이나 정리 작업이 필요하면 여기에 추가
        pass