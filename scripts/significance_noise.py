#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""σ=0.05 退化条件下的配对显著性检验（论文最关键统计主张）

背景：robustness_honest.py 只保存 clean 条件的原始分数。本脚本对
关键退化条件（noise_std=0.05）重推理并缓存分数，然后做配对 bootstrap。

比较对（论文核心主张的统计支撑）：
    CAR-v3 (3 seeds) vs B0 / B0+QAug —— 噪声鲁棒性的显著性

协议：与 robustness_honest.py 完全一致（同一 uint8 缓存、同一退化实现、
noise_seed=2024、全量 5418、同一顺序 → 天然配对）。

用法（GPU 空闲时）：
    python -u scripts/significance_noise.py

输出：
    results/significance/significance_noise005.json
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
    apply_degradation_batch, build_test_cache, load_car, load_baseline,
    forward_probs,
)
from significance_test import paired_bootstrap, auc_score, log

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "significance")
NOISE_SEED = 2024
COND = ("noise", 0.05)

MODELS = {
    "car": ("car", os.path.join(PROJECT_ROOT, "results", "final_car_v3", "checkpoints", "best_model.pt")),
    "car_s43": ("car", os.path.join(PROJECT_ROOT, "results", "cloud_recovery", "final_car_v3_s43_best.pt")),
    "car_s44": ("car", os.path.join(PROJECT_ROOT, "results", "cloud_recovery", "final_car_v3_s44_best.pt")),
    "efficientnet_b0": ("baseline", os.path.join(PROJECT_ROOT, "results", "baseline_honest", "efficientnet_b0", "seed_42", "best_model.pt")),
    "efficientnet_b0_qaug": ("baseline", os.path.join(PROJECT_ROOT, "results", "baseline_qaug", "efficientnet_b0", "last_model.pt")),
}


@torch.no_grad()
def infer_noise005(model, frames_mmap, device, is_car, batch_size=16):
    n = frames_mmap.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    np.random.seed(NOISE_SEED)
    for i in range(0, n, batch_size):
        batch_u8 = np.asarray(frames_mmap[i:i + batch_size])
        batch_u8 = apply_degradation_batch(batch_u8, COND[0], COND[1])
        x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
        x = ((x / 255.0 - 0.5) / 0.5).to(device)
        p = forward_probs(model, x, is_car)
        preds[i:i + batch_u8.shape[0]] = p.cpu().numpy()
    return preds


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(OUT_DIR, "noise005_preds")
    os.makedirs(cache_dir, exist_ok=True)

    frames, labels, _, _ = build_test_cache(config)

    # ---- 推理并缓存分数 ----
    preds_all = {}
    for name, (mtype, ckpt) in MODELS.items():
        npz_path = os.path.join(cache_dir, f"{name}.npz")
        if os.path.exists(npz_path):
            with np.load(npz_path) as d:
                preds_all[name] = d["preds"]
            log(f"{name}: 噪声分数缓存命中")
            continue
        if mtype == "car":
            model, _ = load_car(config, ckpt, device)
        else:
            model = load_baseline(name.replace("_qaug", ""), config.data.num_frames, ckpt, device)
        model.eval()
        log(f"推理 {name} @ noise_std=0.05 ...")
        preds_all[name] = infer_noise005(model, frames, device, mtype == "car")
        np.savez(npz_path, preds=preds_all[name], labels=labels)
        del model
        torch.cuda.empty_cache()
        log(f"{name}: AUC={auc_score(preds_all[name], labels):.4f}")

    # ---- 配对检验 ----
    pairs = [
        ("car", "efficientnet_b0", "CAR-v3 vs B0 @ sigma=0.05"),
        ("car", "efficientnet_b0_qaug", "CAR-v3 vs B0+QAug @ sigma=0.05"),
        ("car_s43", "efficientnet_b0_qaug", "CAR-v3 s43 vs B0+QAug @ sigma=0.05"),
        ("car_s44", "efficientnet_b0_qaug", "CAR-v3 s44 vs B0+QAug @ sigma=0.05"),
    ]
    results = []
    for a, b, desc in pairs:
        obs, p, lo, hi, _ = paired_bootstrap(preds_all[a], preds_all[b], labels)
        row = {
            "pair": f"{a} vs {b}",
            "desc": desc,
            "auc_a": round(auc_score(preds_all[a], labels), 4),
            "auc_b": round(auc_score(preds_all[b], labels), 4),
            "delta_auc": round(obs, 4),
            "bootstrap_p": round(p, 6),
            "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
            "significant_0.05": bool(p < 0.05),
        }
        results.append(row)
        log(f"{desc:<40} Δ={obs:+.4f}  p={p:.2e}  CI=[{lo:+.4f},{hi:+.4f}]  "
            f"{'**SIG**' if p < 0.05 else 'n.s.'}")

    out = {
        "condition": "noise_std=0.05, seed=2024, full test 5418, same degradation order",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    out_path = os.path.join(OUT_DIR, "significance_noise005.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
