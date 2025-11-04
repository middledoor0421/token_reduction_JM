#!/usr/bin/env python3
# main.py
# Python 3.9 compatible. Comments in English only.

import argparse
from typing import List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import timm
from timm.data import resolve_data_config, create_transform
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Token stats recorder
from core import token_stats as tstats

# ToMe adapter
from methods.tome_adapter.plugin import apply_tome_with_hooks

# Ours reducer wrapper
from methods.ours.attn import OursAttention
from tqdm import tqdm

# ------------------------------ arg parsing ----------------------------------

def parse_layers(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if len(s) == 0:
        return None
    parts = s.split(",")
    out = []
    for p in parts:
        p = p.strip()
        if p == "":
            continue
        out.append(int(p))
    return out if len(out) > 0 else None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Token Reduction Runner (ToMe / Ours)")

    # Core model/run args
    p.add_argument("--model", type=str, default="deit_small_patch16_224", help="timm model name")
    p.add_argument("--pretrained", type=str, default="true", choices=["true", "false"], help="load pretrained weights")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--batch-size", type=int, default=64, help="batch size for evaluation")
    p.add_argument("--workers", type=int, default=4, help="dataloader workers")
    p.add_argument("--pin-memory", type=str, default="true", choices=["true", "false"], help="pin_memory for DataLoader")

    # Method selection
    p.add_argument("--method", type=str, default="tome", choices=["tome", "ours"], help="token reduction method")
    p.add_argument("--layers", type=str, default="0,1,2", help="comma-separated target layers; empty means all")
    p.add_argument("--r", type=int, default=13, help="requested removal per targeted layer")

    # Token-cap toggle and stats
    p.add_argument("--token-cap", type=str, default="on", choices=["on", "off"], help="on: allow <r, off: force exactly r")
    p.add_argument("--token-stats", action="store_true", help="print per-layer token reduction stats at the end")
    p.add_argument("--token-stats-json", type=str, default=None, help="optional path to save token stats JSON")
    p.add_argument("--debug-token-stats", action="store_true", help="print debug before/after token lengths per block")

    # ToMe-specific toggles (kept for parity with your previous runs)
    p.add_argument("--cls-protect", type=str, default="true", choices=["true", "false"], help="protect CLS from extra merges in ToMe enforcement")
    p.add_argument("--match-feature", type=str, default="xnorm", choices=["k", "xnorm"], help="ToMe match feature flag (parity only)")
    p.add_argument("--prop-attn", type=str, default="false", choices=["true", "false"], help="propagate attention flag (parity only)")

    # Ours-specific knobs (kept as before; now actually threaded through)
    p.add_argument("--selector", type=str, default="hquota_ff", help="selector name")
    p.add_argument("--hq-q", type=float, default=0.0, help="head-quota fraction [0,1]")
    p.add_argument("--cand-extra", type=int, default=0, help="extra candidate pool size")
    p.add_argument("--merges", type=str, default="v", choices=["v", "kv"], help="merge mode")
    p.add_argument("--alpha", type=float, default=0.0, help="merge alpha")
    p.add_argument("--beta0", type=float, default=0.0, help="merge beta0")
    p.add_argument("--top-r", type=int, default=0, help="merge sparsity top-r")
    p.add_argument("--l2-clip-tau", type=float, default=0.0, help="pre-score clipping tau (0 disables)")
    p.add_argument("--temp-eta", type=float, default=1.0, help="temperature scaling for scores")
    p.add_argument("--size-delta", type=float, default=0.0, help="size delta for merge weighting")
    p.add_argument("--tau-adapt", type=str, default="true", choices=["true", "false"], help="enable tau relaxation before backfill when token-cap=off")
    p.add_argument("--enable-unmerge", type=str, default="true", choices=["true", "false"], help="enable unmerge in our pipeline")

    # Dataset/eval
    p.add_argument("--data", type=str, default="./data/imagenet", help="dataset root; expects a 'val' subfolder")
    p.add_argument("--max-samples", type=int, default=0, help="0 for full eval; otherwise limit samples")

    return p


# ------------------------------ model helpers --------------------------------

def create_model(args: argparse.Namespace) -> torch.nn.Module:
    pretrained = (args.pretrained.lower() == "true")
    model = timm.create_model(args.model, pretrained=pretrained, num_classes=1000)
    return model


def find_transformer_blocks(model: torch.nn.Module):
    for name in ["blocks", "stages"]:
        if hasattr(model, name):
            return getattr(model, name)
    return None


# ------------------------------ Ours wrapper ---------------------------------

class OursApplier(object):
    """
    Attach OursAttention to a timm ViT via per-block hooks.
    The reducer runs after each block to apply token reduction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        r: int,
        layers: Optional[List[int]],
        token_cap: str,
        debug_token_stats: bool,
        tau_adapt: bool,
        enable_unmerge: bool,
        selector: str,
        hq_q: float,
        cand_extra: int,
        merges: str,
        alpha: float,
        beta0: float,
        top_r: int,
        l2_clip_tau: float,
        temp_eta: float,
        size_delta: float,
        match_feature: str
    ):
        self.model = model
        self.r = int(r)
        self.layers = set(layers) if layers is not None else None
        self.token_cap = str(token_cap).lower()
        self.debug_token_stats = bool(debug_token_stats)
        self.tau_adapt = bool(tau_adapt)
        self.enable_unmerge = bool(enable_unmerge)

        # Build reducer with full knobs
        self.reducer = OursAttention(
            token_cap=self.token_cap,
            debug_token_stats=self.debug_token_stats,
            tau_adapt=self.tau_adapt,
            enable_unmerge=self.enable_unmerge,
            selector=selector,
            hq_quota=hq_q,
            cand_extra=cand_extra,
            merge_mode=merges,
            alpha=alpha,
            beta0=beta0,
            top_r=top_r,
            l2_clip_tau=l2_clip_tau,
            temp_eta=temp_eta,
            size_delta=size_delta,
            match_feature=match_feature
        )

        self.state = {"prev_len": None, "layer_idx": -1}
        self._register_hooks()

    def _is_target_layer(self, L: int) -> bool:
        if self.layers is None:
            return True
        return L in self.layers

    def _register_hooks(self) -> None:
        blocks = find_transformer_blocks(self.model)
        if blocks is None:
            if self.debug_token_stats:
                print("[Ours][WARN] no transformer blocks found for hooks.")
            return

        def pre_hook(mod, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return None
            self.state["prev_len"] = int(x.shape[1])
            self.state["layer_idx"] += 1
            return None

        def post_hook(mod, _inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 3:
                return out
            L = self.state["layer_idx"]
            T_prev = int(self.state["prev_len"])
            req_r = self.r if self._is_target_layer(L) else None

            out2 = self.reducer(
                out,
                layer_idx=L,
                requested_r=req_r
                # If you have extra selector features (phi/scores), pass them here.
            )

            if self.debug_token_stats:
                T_cur = int(out2.shape[1])
                delta = T_prev - T_cur
                print(f"[Ours][L{L}] before={T_prev}, after={T_cur}, delta={delta}, req_r={req_r}")

            return out2

        for blk in blocks:
            blk.register_forward_pre_hook(lambda m, i: pre_hook(m, i))
            blk.register_forward_hook(lambda m, i, o: post_hook(m, i, o))


# ------------------------------ eval utilities -------------------------------

class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()
    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Tuple[torch.Tensor, ...]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)   # [B,maxk]
        pred = pred.t()                               # [maxk,B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res: List[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return tuple(res)


def build_eval_loader(args: argparse.Namespace, model: nn.Module) -> DataLoader:
    pretrained = (args.pretrained.lower() == "true")
    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    transform = create_transform(
        input_size=data_config["input_size"],
        interpolation=data_config.get("interpolation", "bilinear"),
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config.get("crop_pct", 0.875)
    )
    val_root = args.data.rstrip("/\\") + "/val"
    dataset = datasets.ImageFolder(val_root, transform=transform)
    if args.max_samples and args.max_samples > 0:
        indices = list(range(min(args.max_samples, len(dataset))))
        dataset = Subset(dataset, indices)
    pin_mem = (args.pin_memory.lower() == "true")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_mem)
    return loader



@torch.no_grad()
def run_inference(model: torch.nn.Module, args: argparse.Namespace) -> Tuple[float, float]:
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()
    cudnn.benchmark = True if device.type == "cuda" else False

    loader = build_eval_loader(args, model)
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    pbar = tqdm(total=len(loader), desc="Infer", leave=True)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        bs = images.size(0)
        top1_meter.update(acc1.item(), n=bs)
        top5_meter.update(acc5.item(), n=bs)

        # show both current and running average in the bar
        pbar.set_postfix({
            "acc1(cur)": f"{acc1.item():.2f}",
            "acc5(cur)": f"{acc5.item():.2f}",
            "acc1(avg)": f"{top1_meter.avg:.2f}",
            "acc5(avg)": f"{top5_meter.avg:.2f}"
        })
        pbar.update(1)
    pbar.close()

    return top1_meter.avg, top5_meter.avg


# ----------------------------------- main ------------------------------------

def main():
    args = build_argparser().parse_args()

    # Reset token stats at the beginning of a run
    tstats.reset()

    # Build model
    model = create_model(args)

    blocks = find_transformer_blocks(model)
    tstats.set_total_layers(len(blocks) if blocks is not None else None)

    # Apply method
    layers = parse_layers(args.layers)

    if args.method == "tome":
        cls_protect = (args.cls_protect.lower() == "true")
        prop_attn = (args.prop_attn.lower() == "true")
        model = apply_tome_with_hooks(
            model=model,
            r=args.r,
            layers=layers,
            token_cap=args.token_cap,
            debug_token_stats=args.debug_token_stats,
            cls_protect=cls_protect,
            match_feature=args.match_feature,
            prop_attn=prop_attn
        )
    elif args.method == "ours":
        tau_adapt = (args.tau_adapt.lower() == "true")
        enable_unmerge = (args.enable_unmerge.lower() == "true")
        _ = OursApplier(
            model=model,
            r=args.r,
            layers=layers,
            token_cap=args.token_cap,
            debug_token_stats=args.debug_token_stats,
            tau_adapt=tau_adapt,
            enable_unmerge=enable_unmerge,
            selector=args.selector,
            hq_q=args.hq_q,
            cand_extra=args.cand_extra,
            merges=args.merges,
            alpha=args.alpha,
            beta0=args.beta0,
            top_r=args.top_r,
            l2_clip_tau=args.l2_clip_tau,
            temp_eta=args.temp_eta,
            size_delta=args.size_delta,
            match_feature=args.match_feature
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Run evaluation
    top1, top5 = run_inference(model, args)

    # Print/save token stats
    if args.token_stats:
        print("\n[TokenStats] Per-layer token reduction summary:")
        print(tstats.report_table_str())

    if args.token_stats_json is not None:
        tstats.dump_json(args.token_stats_json)
        print(f"[TokenStats] Saved JSON to: {args.token_stats_json}")

    # Print metrics
    print(f"[Eval] Top-1: {top1:.3f}  Top-5: {top5:.3f}")


if __name__ == "__main__":
    main()
