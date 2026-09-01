#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CAR-v2 在诚实协议下的噪声重评估（v2→v3 trade-off 声明的协议匹配）

背景：tables.json 的 T7 中，v2 的噪声数字来自审计前的 500 样本子集协议
（不可信），与 v3 的全量诚实协议不可比。本脚本用与 robustness_honest.py
完全一致的协议（全量 5418、冻结 val 阈值、uint8 退化、noise_seed=2024）
重评 v2 的 clean + 三档噪声，使 trade-off 表格协议一致。

用法：
    python -u scripts/eval_v2_noise_honest.py
输出：
    results/final_car_v2/noise_honest.json
"""
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from src.config import load_config
from robustness_honest import (
    apply_degradation_batch, build_test_cache, build_val_cache, load_car,
    forward_probs, compute_metrics, log,
)
from src.utils.metrics import find_optimal_threshold, compute_auc

CKPT = os.path.join(PROJECT_ROOT, "results", "final_car_v2", "checkpoints", "best_model.pt")
OUT = os.path.join(PROJECT_ROOT, "results", "final_car_v2", "noise_honest.json")
NOISE_SEED = 2024

CONDITIONS = {
    "clean": None,
    "noise_std=0.01": ("noise", 0.01),
    "noise_std=0.02": ("noise", 0.02),
    "noise_std=0.05": ("noise", 0.05),
}


@torch.no_grad()
def infer(model, frames_mmap, device, deg_type, severity, batch_size=16):
    n = frames_mmap.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    for i in range(0, n, batch_size):
        batch_u8 = np.asarray(frames_mmap[i:i + batch_size])
        batch_u8 = apply_degradation_batch(batch_u8, deg_type, severity)
        x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
        x = ((x / 255.0 - 0.5) / 0.5).to(device)
        p = forward_probs(model, x, is_car=True)
        preds[i:i + batch_u8.shape[0]] = p.cpu().numpy()
    return preds


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    frames, labels, _, _ = build_test_cache(config)
    val_frames, val_labels = build_val_cache(config)

    model, used_ema = load_car(config, CKPT, device)

    # 冻结 val 阈值（Youden）
    log("计算 v2 val 阈值 ...")
    val_preds = infer(model, val_frames, device, None, None)
    threshold = float(find_optimal_threshold(val_preds, val_labels))
    val_auc = float(compute_auc(val_preds, val_labels))
    log(f"v2 val AUC={val_auc:.4f}, threshold={threshold:.4f}")

    results = {}
    for cond, deg in CONDITIONS.items():
        deg_type, severity = (deg if deg else (None, None))
        np.random.seed(NOISE_SEED)
        log(f"评估 {cond} ...")
        preds = infer(model, frames, device, deg_type, severity)
        results[cond] = compute_metrics(preds, labels, threshold)
        log(f"[car_v2] {cond:<16} AUC={results[cond]['auc']:.4f}")

    out = {
        "model": "car_v2",
        "checkpoint": CKPT,
        "used_ema": used_ema,
        "threshold": threshold,
        "val_auc": val_auc,
        "num_samples": int(frames.shape[0]),
        "noise_seed": NOISE_SEED,
        "protocol": "full-test, frozen val threshold, uint8 degradation identical to robustness_honest.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **results,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"已保存: {OUT}")


if __name__ == "__main__":
    main()
