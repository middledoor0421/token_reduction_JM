# main.py
# Entrypoint with plugin registry routing.
# - Supports identity / tome / ours
# - Progress bar + final Top-1 (%)
# - Optional KL/Δmargin vs identity (subset), FLOPs print

import os
import sys
import argparse
from typing import Optional

import torch
import torchvision as tv
import torchvision.transforms as T
from tqdm import tqdm

# prefer vendored timm at project root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from timm._vendor_version import __version__ as TIMM_VENDOR_VERSION
    print(f"[timm] vendored snapshot: {TIMM_VENDOR_VERSION}")
except Exception:
    import timm  # fallback
    print(f"[timm] site-packages: {getattr(timm, '__version__', 'unknown')} @ {getattr(timm, '__file__', 'n/a')}")

import timm

from core.registry import available_methods, create_plugin
from core.metrics import margin_from_logits, kl_pq_from_logits
from core.flops import flops_gmacs
import methods  # noqa: F401  # trigger plugin registrations


# ---------- utils ----------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_val_loader(data_root: str, batch_size: int, workers: int) -> torch.utils.data.DataLoader:
    tfm = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    ds = tv.datasets.ImageFolder(data_root, transform=tfm)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    return dl

def evaluate(model: torch.nn.Module,
             dl: torch.utils.data.DataLoader,
             device: str,
             max_samples: Optional[int] = None) -> float:
    model.eval()
    n_total = len(dl.dataset)
    target = n_total if (not max_samples or max_samples <= 0) else min(max_samples, n_total)
    seen, correct = 0, 0
    pbar = tqdm(total=target, unit="img", desc="eval", leave=True)
    with torch.inference_mode():
        for imgs, labels in dl:
            if max_samples and seen >= target:
                break
            if max_samples and seen + imgs.size(0) > target:
                cut = target - seen
                imgs, labels = imgs[:cut], labels[:cut]

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)

            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            seen += imgs.size(0)

            pbar.update(imgs.size(0))
            pbar.set_postfix(acc=f"{(100.0*correct/max(1,seen)):.2f}%")
            if max_samples and seen >= target:
                break
    pbar.close()
    return float(correct / max(1, seen))

def evaluate_collect_logits(model, dl, device, limit=None):
    model.eval()
    seen, logits_all, labels_all = 0, [], []
    with torch.inference_mode():
        for imgs, labels in dl:
            if limit and seen >= limit:
                break
            if limit and seen + imgs.size(0) > limit:
                cut = limit - seen
                imgs, labels = imgs[:cut], labels[:cut]
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())
            seen += imgs.size(0)
            if limit and seen >= limit:
                break
    return torch.cat(logits_all, 0), torch.cat(labels_all, 0)

# ---------- main ----------

def parse_args():
    ap = argparse.ArgumentParser(description="myproject — token reduction driver")
    # data/model
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--model", type=str, default="deit_small_patch16_224")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # methods
    from core.registry import available_methods
    ap.add_argument("--method", type=str, default="identity", choices=available_methods())

    # 공통 옵션
    ap.add_argument("--layers", type=str, default="0,1,2",
                    help="Comma-separated block indices to apply reduction ('' for all)")
    # NOTE: keep을 문자열로 받아 레이어별 비율을 지원 (예: '0.90,0.85,0.80' 또는 '0.68')
    ap.add_argument("--keep", type=str, default="0.68",
                    help="Keep ratio. Either a single float '0.68' or per-layer list like '0.90,0.85,0.80'")
    ap.add_argument("--max-samples", type=int, default=0)

    # ToMe 전용
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--match-feature", type=str, default="k", choices=["k", "xnorm"])
    ap.add_argument("--prop-attn", action="store_true", default=False)

    # Ours 전용
    ap.add_argument("--selector", type=str, default="hquota_ff",
                    help="ff | topk | random | facility | kdpp | hquota_ff")
    ap.add_argument("--hq-q", type=float, default=0.3, dest="hq_q")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--cand-extra", type=int, default=128, dest="cand_extra")
    ap.add_argument("--drop", action="store_true", default=False)

    # metrics/flops
    ap.add_argument("--metrics-samples", type=int, default=1000)
    ap.add_argument("--print-flops", action="store_true", default=False)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    print(f"[env] device={device}, seed={args.seed}")
    print(f"[data] root={args.data_root}")
    print(f"[method] {args.method}")

    # build model
    model = timm.create_model(args.model, pretrained=True).to(device)

    if args.print_flops:
        gmacs, params = flops_gmacs(model)
        print(f"[flops] ~{gmacs:.2f} GMACs, params={params/1e6:.1f}M")

    # ...
    plugin_cfg = {
        "layers": args.layers,
        "keep": args.keep,  # 문자열 그대로 전달 (단일/CSV 모두 허용)
        "r": args.r,
    }
    if args.method == "tome":
        plugin_cfg.update({
            "match_feature": args.match_feature,
            "prop_attn": bool(args.prop_attn),
        })
    if args.method == "ours":
        plugin_cfg.update({
            "selector": args.selector,
            "hq_q": args.hq_q,
            "gamma": args.gamma,
            "cand_extra": args.cand_extra,
            "drop": bool(args.drop),
            "log_coverage": True
        })

    plugin = create_plugin(args.method, plugin_cfg)
    plugin.attach(model)

    # eval
    dl = build_val_loader(args.data_root, args.batch_size, args.workers)
    max_samples = args.max_samples if args.max_samples > 0 else None
    acc = evaluate(model, dl, device, max_samples=max_samples)
    print(f"[result] Top-1@val = {acc * 100.0:.2f}%")

    # optional KL/Δmargin vs identity for non-identity methods
    if args.method != "identity" and args.metrics_samples > 0:
        base = timm.create_model(args.model, pretrained=True).to(device)
        base_logits, base_labels = evaluate_collect_logits(base, dl, device, limit=args.metrics_samples)
        red_logits, red_labels = evaluate_collect_logits(model, dl, device, limit=args.metrics_samples)
        if base_labels.shape[0] == red_labels.shape[0]:
            kl = kl_pq_from_logits(base_logits, red_logits)
            dm = margin_from_logits(red_logits, red_labels) - margin_from_logits(base_logits, base_labels)
            worst_k = max(1, int(0.05 * dm.numel()))
            print(f"[metrics] KL(base||reduced)={kl:.4f}  "
                  f"Δmargin(mean)={float(dm.mean().item()):+.4f}  "
                  f"worst5%={float(torch.kthvalue(dm, worst_k).values.item()):+.4f}")
        else:
            print("[metrics] skipped (label length mismatch)")

    plugin.finalize()

if __name__ == "__main__":
    main()
