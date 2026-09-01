#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诚实基线统一评估脚本（阶段 0.3）

对单个模型 checkpoint 执行「val 集 Youden 定阈值 → test 集评估」的权威协议，
保存 raw_predictions.npz（preds/labels/video_ids）与 eval_metrics.json。

支持 CAR（sigmoid 评分，优先 EMA 权重）与 baseline（softmax 评分）两类模型。

用法：
    python scripts/eval_honest.py --model car \
        --checkpoint results/baseline_honest/car/seed_42/checkpoints/best_model.pt \
        --output_dir results/baseline_honest/car/seed_42
    python scripts/eval_honest.py --model efficientnet_b0 --seed 42 \
        --checkpoint results/baseline_honest/efficientnet_b0/seed_42/best_model.pt \
        --output_dir results/baseline_honest/efficientnet_b0/seed_42
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import DeepfakeDataset
from src.data.dataloader import create_dataloader
from src.models.car import CAR
from src.utils.metrics import (
    find_optimal_threshold,
    compute_auc,
    compute_accuracy,
    compute_f1,
    compute_ap,
    compute_eer,
    compute_tpr_at_fpr,
)

BASELINE_MODELS = {"efficientnet_b0", "xception", "mesonet"}
DISPLAY_NAMES = {
    "car": "CAR",
    "mesonet": "MesoNet",
    "efficientnet_b0": "EfficientNet-B0",
    "xception": "Xception",
}


def log(msg):
    print(f"[INFO] {msg}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_cutmix(config):
    if "training" in config._data and isinstance(config._data["training"], dict):
        config._data["training"]["cutmix_p"] = 0.0


def build_car_loader(config, split):
    """CAR：复用 create_dataloader（评估无增强），强制 num_workers=0（避免 lambda 序列化）。"""
    config._data["data"]["num_workers"] = 0
    return create_dataloader(config, split=split)


def _baseline_collate(batch):
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "video_path": [b["video_path"] for b in batch],
    }


def build_baseline_loader(config, split):
    ds = DeepfakeDataset(
        config.data.data_root, split=split,
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
    )
    return DataLoader(ds, batch_size=config.data.batch_size, shuffle=False,
                      num_workers=4, collate_fn=_baseline_collate,
                      pin_memory=True, persistent_workers=True, prefetch_factor=4)


def load_car(config, checkpoint_path, device):
    model = CAR(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    used_ema = False
    if isinstance(ckpt, dict) and ckpt.get("ema_shadow") is not None:
        try:
            model.load_state_dict(ckpt["ema_shadow"], strict=True)
            used_ema = True
        except RuntimeError:
            model.load_state_dict(ckpt["ema_shadow"], strict=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            used_ema = True
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, used_ema


def load_baseline(model_name, num_frames, checkpoint_path, device):
    from baseline_full import build_model as _build
    model = _build(model_name, num_frames).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def car_forward(model, frames):
    outputs = model(frames)
    logits = outputs["logits"]
    if logits.size(1) > 1:
        return torch.sigmoid(logits[:, 1])
    return torch.sigmoid(logits.squeeze(-1))


def baseline_forward(model, frames):
    logits = model(frames)
    return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def run_inference(model, loader, device, forward_fn, desc="Evaluating"):
    model.eval()
    all_preds, all_labels, all_ids = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        frames = batch["frames"].to(device)
        labels = batch["label"]
        preds = forward_fn(model, frames)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if "video_path" in batch:
            all_ids.extend([str(v) for v in batch["video_path"]])
    preds = np.concatenate(all_preds).astype(np.float64)
    labels = np.concatenate(all_labels).astype(int)
    return preds, labels, all_ids[: len(preds)]


def compute_metrics_with_threshold(preds, labels, threshold):
    return {
        "auc": float(compute_auc(preds, labels)),
        "accuracy": float(compute_accuracy(preds, labels, threshold)),
        "f1": float(compute_f1(preds, labels, threshold)),
        "ap": float(compute_ap(preds, labels)),
        "eer": float(compute_eer(preds, labels)),
        "tpr_at_fpr_1": float(compute_tpr_at_fpr(preds, labels, 0.01)),
        "tpr_at_fpr_01": float(compute_tpr_at_fpr(preds, labels, 0.001)),
        "threshold": float(threshold),
    }


def compute_confusion_matrix(preds, labels, threshold):
    pred_binary = (np.asarray(preds).flatten() >= threshold).astype(int)
    labels = np.asarray(labels).flatten().astype(int)
    return {
        "tp": int(np.sum((pred_binary == 1) & (labels == 1))),
        "tn": int(np.sum((pred_binary == 0) & (labels == 0))),
        "fp": int(np.sum((pred_binary == 1) & (labels == 0))),
        "fn": int(np.sum((pred_binary == 0) & (labels == 1))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    choices=["car", "efficientnet_b0", "xception", "mesonet"])
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    log(f"Device: {device}, model={args.model}, seed={args.seed}")

    config = load_config(args.config)
    disable_cutmix(config)
    set_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        log(f"[SKIP] checkpoint 不存在: {args.checkpoint}")
        return

    is_car = (args.model == "car")
    if is_car:
        model, used_ema = load_car(config, args.checkpoint, device)
        val_loader = build_car_loader(config, "val")
        test_loader = build_car_loader(config, "test")
    else:
        model = load_baseline(args.model, config.data.num_frames, args.checkpoint, device)
        used_ema = False
        val_loader = build_baseline_loader(config, "val")
        test_loader = build_baseline_loader(config, "test")

    forward_fn = car_forward if is_car else baseline_forward
    scoring = "sigmoid" if is_car else "softmax"

    log("Running inference on val (for threshold)...")
    val_preds, val_labels, _ = run_inference(model, val_loader, device, forward_fn, "Val(thr)")
    threshold = find_optimal_threshold(val_preds, val_labels)
    log(f"optimal threshold (val Youden): {threshold:.4f}")

    log("Running inference on test (final)...")
    preds, labels, video_ids = run_inference(model, test_loader, device, forward_fn, "Test")
    metrics = compute_metrics_with_threshold(preds, labels, threshold)

    record = {
        "model": args.model,
        "display_name": DISPLAY_NAMES.get(args.model, args.model),
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "threshold_source": "val",
        "scoring": scoring,
        "used_ema": used_ema,
        "num_samples": int(len(labels)),
        "val_threshold_info": {
            "val_num_samples": int(len(val_labels)),
            "val_auc": float(compute_auc(val_preds, val_labels)),
        },
        "confusion_matrix": compute_confusion_matrix(preds, labels, threshold),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    save_dict = {"preds": preds, "labels": labels}
    if video_ids and len(video_ids) > 0:
        save_dict["video_ids"] = np.asarray(video_ids, dtype=object)
    np.savez(os.path.join(args.output_dir, "raw_predictions.npz"), **save_dict)

    log(f"{record['display_name']} (seed={args.seed}): AUC={metrics['auc']:.4f} "
        f"Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} EER={metrics['eer']:.4f} "
        f"AP={metrics['ap']:.4f}")
    log(f"Saved eval_metrics.json + raw_predictions.npz to {args.output_dir}")


if __name__ == "__main__":
    main()