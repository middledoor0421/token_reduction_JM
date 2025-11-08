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
from tqdm import tqdm

# Token stats (per-layer averages over all batches)
from core import token_stats as tstats

# ToMe adapter (logging-only hooks; merge inside blocks: Attn -> Merge -> MLP)
from methods.tome_adapter.plugin import apply_tome_with_hooks

# Ours in-block reducer (Attn -> Merge -> MLP)
from methods.ours.attn import apply_ours_inblock


# ------------------------------ arg parsing ----------------------------------

def parse_layers(s: Optional[str]) -> Optional[List[int]]:
    """Return None for all layers, or a list of layer indices."""
    if s is None:
        return None
    s = s.strip()
    if len(s) == 0:
        return None
    parts = [p.strip() for p in s.split(",")]
    out: List[int] = []
    for p in parts:
        if p == "":
            continue
        out.append(int(p))
    return out if len(out) > 0 else None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Token Reduction Runner (ToMe / Ours in-block)")

    # Core model/run args
    p.add_argument("--model", type=str, default="deit_small_patch16_224", help="timm model name")
    p.add_argument("--pretrained", type=str, default="true", choices=["true", "false"], help="load pretrained weights")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--batch-size", type=int, default=64, help="batch size for evaluation")
    p.add_argument("--workers", type=int, default=4, help="dataloader workers")
    p.add_argument("--pin-memory", type=str, default="true", choices=["true", "false"], help="pin_memory for DataLoader")

    # Method selection
    p.add_argument("--method", type=str, default="tome", choices=["tome", "ours"], help="token reduction method")

    # Reduction schedule
    p.add_argument("--layers", type=str, default="0,1,2", help="comma-separated target layers; empty means all")
    p.add_argument("--r", type=int, default=13, help="requested removal per targeted layer")

    # Token-cap toggle and stats
    p.add_argument("--token-cap", type=str, default="on", choices=["on", "off"], help="on: allow <r, off: force exactly r")
    p.add_argument("--token-stats", action="store_true", help="print per-layer token reduction stats at the end")
    p.add_argument("--token-stats-json", type=str, default=None, help="optional path to save token stats JSON")
    p.add_argument("--debug-token-stats", action="store_true", help="print debug token lengths per block")

    # ToMe-specific (parity tagging only; token-cap ignored in adapter)
    p.add_argument("--match-feature", type=str, default="xnorm", choices=["k", "xnorm"], help="ToMe match feature tag")
    p.add_argument("--prop-attn", type=str, default="false", choices=["true", "false"], help="ToMe proportional attention tag")
    p.add_argument("--cls-protect", type=str, default="true", choices=["true", "false"], help="ToMe CLS policy tag")

    # Ours-specific knobs (passed to in-block reducer)
    p.add_argument("--selector", type=str, default="hquota_ff", help="selector name")
    p.add_argument("--hq-q", type=float, default=0.0, help="head-quota fraction [0,1]")
    p.add_argument("--cand-extra", type=int, default=0, help="extra candidate pool size")
    p.add_argument("--merges", type=str, default="v", choices=["v", "kv"], help="merge mode")
    p.add_argument("--alpha", type=float, default=0.0, help="merge alpha")
    p.add_argument("--beta0", type=float, default=0.0, help="reserved")
    p.add_argument("--top-r", type=int, default=0, help="reserved")
    p.add_argument("--l2-clip-tau", type=float, default=0.0, help="pre-score clipping tau (0 disables)")
    p.add_argument("--temp-eta", type=float, default=1.0, help="temperature scaling for scores")
    p.add_argument("--size-delta", type=float, default=0.0, help="size delta for merge weighting")
    p.add_argument("--tau-adapt", type=str, default="true", choices=["true", "false"], help="reserved for threshold relaxation")
    p.add_argument("--enable-unmerge", type=str, default="false", choices=["true", "false"], help="enable unmerge in our pipeline")

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

    # Reset stats for this run
    tstats.reset()

    # Build base model
    model = create_model(args)

    # Apply method
    layers = parse_layers(args.layers)

    if args.method == "tome":
        cls_protect = (args.cls_protect.lower() == "true")
        prop_attn = (args.prop_attn.lower() == "true")
        # Note: token_cap is ignored inside the ToMe adapter to preserve parity (logging-only).
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
        # In-block reduction: Attention -> Merge -> MLP
        model = apply_ours_inblock(
            model=model,
            r=args.r,
            layers=layers,
            token_cap=args.token_cap,
            debug_token_stats=args.debug_token_stats,
            tau_adapt=(args.tau_adapt.lower() == "true"),
            enable_unmerge=(args.enable_unmerge.lower() == "true"),
            selector=args.selector,
            hq_quota=args.hq_q,
            cand_extra=args.cand_extra,
            merge_mode=args.merges,
            alpha=args.alpha,
            beta0=args.beta0,
            top_r=args.top_r,
            l2_clip_tau=args.l2_clip_tau,
            temp_eta=args.temp_eta,
            size_delta=args.size_delta,
            match_feature=args.match_feature
        )

    else:
        raise ValueError("Unknown method: {}".format(args.method))

    # Inform stats module about total number of layers (for fixed-row reporting)
    blocks = find_transformer_blocks(model)
    tstats.set_total_layers(len(blocks) if blocks is not None else None)

    # Run evaluation
    top1, top5 = run_inference(model, args)

    # Print/save token stats
    if args.token_stats:
        print("\n[TokenStats] Per-layer token reduction summary (dataset-avg):")
        print(tstats.report_table_str())

    if args.token_stats_json is not None:
        tstats.dump_json(args.token_stats_json)
        print("[TokenStats] Saved JSON to: {}".format(args.token_stats_json))

    # Print metrics
    print("[Eval] Top-1: {:.3f}  Top-5: {:.3f}".format(top1, top5))


if __name__ == "__main__":
    main()
