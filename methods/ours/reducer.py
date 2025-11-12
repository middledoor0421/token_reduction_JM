# methods/ours/reducer.py
# Orchestrator for one-step token reduction: selection -> assignment -> merge.
# Python 3.9 compatible. Comments in English only.

from typing import Optional, Tuple, Any, List
import importlib
import torch

# Registry (preferred)
try:
    from methods.ours.registry import get_selector, get_merge
except Exception:
    get_selector = None
    get_merge = None


# ------------------------------
# Public API
# ------------------------------
@torch.no_grad()
def reduce_once(
    x: torch.Tensor,
    *,
    r: int,
    selector: str = "hquota_ff",   # module or registry key under methods.ours.selectors
    merges: str = "kv",            # module or registry key under methods.ours.merges
    alpha: float = 0.15,           # center-preserve factor
    select_mode: str = "keep",     # "keep" or "drop"
    token_cap: str = "on",         # "off" => force exactly r, "on" => <= r allowed
    prop_attn: bool = False,       # reserved for merger implementations
    cls_protect: bool = True,
    layer_idx: Optional[int] = None,
    debug: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    x: [B, T, C]
    Returns: x_merged [B, T', C]
    """
    assert isinstance(x, torch.Tensor) and x.dim() == 3, "x must be [B, T, C]"
    B, T, C = x.shape
    if r <= 0 or T <= 1:
        return x

    K_target = max(1, T - int(r))

    # 1) Selection (registry first, then module fallback)
    keep_idx, assign_idx = _run_selector(
        x=x,
        K_target=K_target,
        r=r,
        selector=selector,
        mode=select_mode,
        token_cap=token_cap,
        cls_protect=cls_protect,
        debug=debug,
        **kwargs
    )

    # 2) If selector did not provide assignment, build nearest by cosine
    if assign_idx is None:
        assign_idx = _nearest_assign_cosine(x, keep_idx)  # [B, T] in [0..K-1]

    # 3) Merge (registry first, then module fallback; fallback to scatter-mean)
    x_merged = _run_merge(
        x=x,
        keep_idx=keep_idx,
        assign_idx=assign_idx,
        merges=merges,
        alpha=alpha,
        prop_attn=prop_attn,
        debug=debug
    )
    return x_merged


# ------------------------------
# Selector bridge
# ------------------------------
def _safe_import(name: str, debug: bool = False):
    try:
        return importlib.import_module(name)
    except Exception as e:
        if debug:
            print("[OURS][import] failed:", name, str(e))
        class _Dummy:
            pass
        return _Dummy()


def _run_selector(
    *,
    x: torch.Tensor,
    K_target: int,
    r: int,
    selector: str,
    mode: str,
    token_cap: str,
    cls_protect: bool,
    debug: bool,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      keep_idx: LongTensor [B, K]
      assign_idx: Optional LongTensor [B, T] mapping tokens to [0..K-1]
    """
    B, T, C = x.shape
    device = x.device
    mode = str(mode).lower()
    token_cap_flag = (str(token_cap).lower() == "on")

    # --- 0) Registry path (preferred) ---
    reg = get_selector(selector) if get_selector is not None else None
    if reg is not None and callable(reg.get("fn", None)):
        try:
            fn = reg["fn"]
            out = fn(
                x=x,
                mode=mode,
                k=int(K_target),
                r=int(r),
                cls_protect=bool(cls_protect),
                token_cap=bool(token_cap_flag),
                debug=bool(debug),
                **kwargs
            )
            if isinstance(out, tuple):
                keep_idx, assign_idx = out[0], (out[1] if len(out) > 1 else None)
            else:
                keep_idx, assign_idx = out, None
            keep_idx = _post_select_fix(keep_idx, K_target, T, x, cls_protect, token_cap_flag)
            return keep_idx.to(device, dtype=torch.long), (None if assign_idx is None else assign_idx.to(device, dtype=torch.long))
        except Exception as e:
            if debug:
                print("[OURS][selector registry] failed:", str(e))
            # fall through to module path

    # --- 1) Module path (dynamic import + multiple entry names) ---
    sel_mod = _safe_import(f"methods.ours.selectors.{selector.split(':')[0]}", debug=debug)

    candidates = []
    # keep-K style
    for fn in ["select_keep_k", "keep_k", "select_keep", "run_select_keep"]:
        if hasattr(sel_mod, fn):
            candidates.append(getattr(sel_mod, fn))
    # drop-r style
    for fn in ["select_drop_r", "drop_r", "select_drop", "run_select_drop"]:
        if hasattr(sel_mod, fn):
            candidates.append(getattr(sel_mod, fn))
    # generic
    if hasattr(sel_mod, "select"):
        candidates.append(getattr(sel_mod, "select"))

    keep_idx = None
    assign_idx = None
    last_err = None

    for fn in candidates:
        try:
            name = fn.__name__
            if name in ("select_keep_k", "keep_k", "select_keep", "run_select_keep"):
                out = fn(
                    x=x, k=K_target,
                    cls_protect=cls_protect,
                    token_cap=token_cap_flag,
                    debug=debug, **kwargs
                )
            elif name in ("select_drop_r", "drop_r", "select_drop", "run_select_drop"):
                out = fn(
                    x=x, r=r,
                    cls_protect=cls_protect,
                    token_cap=token_cap_flag,
                    debug=debug, **kwargs
                )
            else:
                out = fn(
                    x=x, mode=mode, k=K_target, r=r,
                    cls_protect=cls_protect, token_cap=token_cap_flag,
                    debug=debug, **kwargs
                )

            if isinstance(out, tuple):
                keep_idx, assign_idx = out[0], (out[1] if len(out) > 1 else None)
            else:
                keep_idx, assign_idx = out, None

            keep_idx = _post_select_fix(keep_idx, K_target, T, x, cls_protect, token_cap_flag)
            return keep_idx.to(device, dtype=torch.long), (None if assign_idx is None else assign_idx.to(device, dtype=torch.long))

        except Exception as e:
            last_err = e
            continue

    # --- 2) Hard fallback: L2 Top-K (CLS boosted) ---
    if debug:
        print("[OURS][selector] Fallback to L2 Top-K due to:", last_err)
    base = torch.norm(x, p=2, dim=-1)  # [B, T]
    base[:, 0] = base.max(dim=1, keepdim=True)[0].squeeze(1) + 1e6
    keep_idx = torch.topk(base, k=min(K_target, T), dim=1, largest=True, sorted=True).indices
    keep_idx = _post_select_fix(keep_idx, K_target, T, x, cls_protect, token_cap_flag)
    return keep_idx.to(device, dtype=torch.long), None


def _post_select_fix(keep_idx: torch.Tensor, K_target: int, T: int, x: torch.Tensor,
                     cls_protect: bool, token_cap_flag: bool) -> torch.Tensor:
    """CLS(0) enforcement and exact-K backfill if token_cap is off."""
    if cls_protect:
        keep_idx = _ensure_cls0(keep_idx, T)
    if not token_cap_flag:
        keep_idx = _force_exact_k(keep_idx, K_target, T, x)
    return keep_idx


def _ensure_cls0(keep_idx: torch.Tensor, T: int) -> torch.Tensor:
    """Ensure index-0 is present in each row; replace the last if missing; keep order stable."""
    B, K = keep_idx.shape
    dev = keep_idx.device
    has = (keep_idx == 0).any(dim=1)
    if has.all():
        return keep_idx
    fixed = keep_idx.clone()
    for b in range(B):
        if not bool(has[b].item()):
            fixed[b, -1] = 0
    # Stable unique
    rows: List[torch.Tensor] = []
    for b in range(B):
        seen = set()
        row = []
        for j in range(K):
            v = int(fixed[b, j].item())
            if 0 <= v < T and v not in seen:
                row.append(v)
                seen.add(v)
        while len(row) < K:
            row.append(0)
        rows.append(torch.tensor(row, device=dev, dtype=torch.long))
    return torch.stack(rows, dim=0)


def _force_exact_k(keep_idx: torch.Tensor, K_target: int, T: int, x: torch.Tensor) -> torch.Tensor:
    """Backfill using L2 score if returned K < K_target."""
    B, K = keep_idx.shape
    if K == K_target:
        return keep_idx
    if K > K_target:
        return keep_idx[:, :K_target]

    dev = x.device
    base = torch.norm(x, p=2, dim=-1)  # [B, T]
    rows: List[torch.Tensor] = []
    for b in range(B):
        chosen = set(int(v.item()) for v in keep_idx[b])
        scores = []
        for t in range(T):
            if t in chosen:
                continue
            scores.append((float(base[b, t].item()), t))
        scores.sort(key=lambda z: z[0], reverse=True)
        add = [idx for _, idx in scores[: max(0, K_target - K)]]
        row = list(chosen) + add
        row = row[:K_target]
        rows.append(torch.tensor(row, device=dev, dtype=torch.long))
    return torch.stack(rows, dim=0)


# ------------------------------
# Assignment (cosine, vectorized)
# ------------------------------
def _nearest_assign_cosine(x: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    """Return assign_idx [B, T] mapping each token to nearest keep (cosine)."""
    B, T, C = x.shape
    dev = x.device
    K = int(keep_idx.shape[1])

    batch = torch.arange(B, device=dev).unsqueeze(1).expand(B, K)
    keep_feat = x[batch, keep_idx]  # [B, K, C]

    x_n = torch.nn.functional.normalize(x, p=2, dim=-1)           # [B, T, C]
    k_n = torch.nn.functional.normalize(keep_feat, p=2, dim=-1)   # [B, K, C]
    sims = torch.einsum("btc,bkc->btk", x_n, k_n)                 # [B, T, K]
    assign = sims.argmax(dim=-1).to(torch.long)                   # [B, T]
    return assign


# ------------------------------
# Merge bridge (registry first, then module; fallback scatter-mean)
# ------------------------------
def _run_merge(
    *,
    x: torch.Tensor,
    keep_idx: torch.Tensor,     # [B, K]
    assign_idx: torch.Tensor,   # [B, T] -> [0..K-1]
    merges: str,
    alpha: float,
    prop_attn: bool,
    debug: bool
) -> torch.Tensor:
    """
    Return x_merged [B, K, C] (keep order preserved).
    """
    B, T, C = x.shape
    dev = x.device
    K = int(keep_idx.shape[1])

    # --- 0) Registry path ---
    reg = get_merge(merges) if get_merge is not None else None
    if reg is not None and callable(reg.get("fn", None)):
        try:
            fn = reg["fn"]
            out = fn(
                x=x,
                keep_idx=keep_idx,
                assign_idx=assign_idx,
                alpha=float(alpha),
                size=None
            )
            if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                return out[0]
            if isinstance(out, torch.Tensor):
                return out
        except Exception as e:
            if debug:
                print("[OURS][merge registry] failed:", str(e))
            # fall through to module path

    # --- 1) Module path ---
    if merges.lower() == "kv":
        module_names = ["methods.ours.merges.kv_merge", "methods.ours.merges.merge_kv"]
    else:
        module_names = ["methods.ours.merges.v_merge", "methods.ours.merges.merge_v"]

    for mn in module_names:
        mod = _safe_import(mn, debug=debug)
        for fn_name in ["merge", "merge_once", "merge_tokens", "kv_merge", "v_merge", "run_merge"]:
            if hasattr(mod, fn_name):
                fn = getattr(mod, fn_name)
                try:
                    out = fn(
                        x=x,
                        keep_idx=keep_idx,
                        assign=assign_idx,
                        alpha=alpha,
                        prop_attn=prop_attn,
                        debug=debug
                    )
                    if isinstance(out, torch.Tensor):
                        return out
                    if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                        return out[0]
                except TypeError:
                    out = fn(x, keep_idx, assign_idx)
                    if isinstance(out, torch.Tensor):
                        return out
                    if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                        return out[0]
                except Exception as e:
                    if debug:
                        print("[OURS][merge module] failed:", str(e))
                # try next module name
    # --- 2) Fallback: scatter-mean with center preserve ---
    batch = torch.arange(B, device=dev).unsqueeze(1).expand(B, K)
    centers = x[batch, keep_idx]  # [B, K, C]

    sums = x.new_zeros(B * K, C)
    counts = x.new_zeros(B * K)
    idx = assign_idx + (torch.arange(B, device=dev).unsqueeze(1) * K)  # [B, T]
    idx_flat = idx.reshape(B * T)
    x_flat = x.reshape(B * T, C)

    sums.index_add_(0, idx_flat, x_flat)
    ones = torch.ones(B * T, device=dev, dtype=counts.dtype)
    counts.index_add_(0, idx_flat, ones)

    sums = sums.view(B, K, C)
    counts = counts.view(B, K).clamp_min(1.0)
    mean = sums / counts.unsqueeze(-1)

    out = (1.0 - float(alpha)) * centers + float(alpha) * mean
    return out
