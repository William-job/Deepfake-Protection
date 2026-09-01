#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单专家×退化交叉实验（Oracle-best-single-expert 对照，审稿人 Major#3 专用）

问题：CAR 的鲁棒性是来自"组合多专家"（compositional），还是某个最强单专家
      就足够（best-single-expert suffices）？

背景：专家 standalone AUC（T4）显示 temporal 单专家 clean 0.870 ≥ full 0.867，
      这个事实必须正面处理——clean 上单专家追平组合，退化下是否仍追平？

本实验把 4 个 only:<idx> 变体（仅保留单专家，权重=1）放到 4 个退化条件
下评估，与 full 对照：
- 若 only:temporal 在退化下崩而 full 不崩 → 组合价值成立
- 若单专家全面追平 full → 路由组合性叙事必须弱化（如实报告）

协议与 ablation_robustness.py 完全一致（一次前向算全部变体，
uint8 缓存、noise_seed=2024、阈值无关指标）。

用法：
    python -u scripts/expert_only_robustness.py
输出：
    results/ablation_robustness/expert_only_robustness.json
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
from src.models.car import CAR
from src.utils.metrics import compute_auc, compute_ap
from robustness_honest import apply_degradation_batch, build_test_cache, log, load_car
from robustness_transcode import transcode_roundtrip, TMP_DIR

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "ablation_robustness")
NOISE_SEED = 2024
CKPT = os.path.join(PROJECT_ROOT, "results", "final_car_v3", "checkpoints", "best_model.pt")

EXPERT_NAMES = ["motion", "temporal", "spectral", "boundary"]

VARIANTS = {
    "full": "none",
    "only_motion": "only:0",
    "only_temporal": "only:1",
    "only_spectral": "only:2",
    "only_boundary": "only:3",
}

CONDITIONS = {
    "clean": None,
    "noise_std=0.05": ("noise", 0.05),
    "blur_kernel=7": ("blur", 7),
    "jpeg_quality=30": ("jpeg", 30),
}


def modify_weights(w, rule):
    if rule == "none":
        return w
    if rule.startswith("only:"):
        idx = int(rule.split(":")[1])
        w2 = torch.zeros_like(w)
        w2[:, idx] = 1.0
        return w2
    raise ValueError(rule)


@torch.no_grad()
def eval_all_variants(model, frames_mmap, labels, device, deg_type=None, severity=None,
                      batch_size=16):
    n = frames_mmap.shape[0]
    preds = {v: np.zeros(n, dtype=np.float64) for v in VARIANTS}

    for i in range(0, n, batch_size):
        batch_u8 = np.asarray(frames_mmap[i:i + batch_size])
        batch_u8 = apply_degradation_batch(batch_u8, deg_type, severity)
        x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
        x = ((x / 255.0 - 0.5) / 0.5).to(device)

        out = model(x)
        head_outputs, z, w = out["head_outputs"], out["z"], out["w"]

        logit_list = []
        for name in model.expert_names:
            expert_out = model.experts[name](head_outputs[name])
            if expert_out.dim() == 2 and expert_out.size(0) != z.size(0):
                if expert_out.size(0) % z.size(0) == 0:
                    T = expert_out.size(0) // z.size(0)
                    expert_out = expert_out.view(z.size(0), T, -1).mean(dim=1)
                else:
                    rf = z.size(0) // expert_out.size(0)
                    if rf > 0:
                        expert_out = expert_out.repeat(rf, 1)
            logit_list.append(expert_out)
        stacked = torch.stack(logit_list, dim=1).clamp(-100.0, 100.0)

        for vname, rule in VARIANTS.items():
            w_mod = modify_weights(w, rule)
            y = (w_mod.unsqueeze(-1) * stacked).sum(dim=1)
            p = torch.sigmoid(y[:, 1] if y.size(1) > 1 else y.squeeze(-1))
            preds[vname][i:i + batch_u8.shape[0]] = p.cpu().numpy()

    return {v: {"auc": float(compute_auc(p, labels)),
                "ap": float(compute_ap(p, labels))} for v, p in preds.items()}


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    os.makedirs(TMP_DIR, exist_ok=True)

    frames, labels, _, _ = build_test_cache(config)
    model, _ = load_car(config, CKPT, device)
    model.eval()

    results = {}
    for cond, deg in CONDITIONS.items():
        deg_type, severity = (deg if deg else (None, None))
        np.random.seed(NOISE_SEED)
        log(f"评估条件 {cond} ...")
        results[cond] = eval_all_variants(model, frames, labels, device,
                                          deg_type=deg_type, severity=severity)
        for v, m in results[cond].items():
            log(f"  [{cond:<16}] {v:<15} AUC={m['auc']:.4f}  AP={m['ap']:.4f}")
        with open(os.path.join(OUT_DIR, "expert_only_robustness.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "checkpoint": CKPT,
                "protocol": ("full test 5418, uint8 cache, noise_seed=2024, "
                             "threshold-free (AUC/AP); only:X = weight 1 on expert X"),
                "expert_order": EXPERT_NAMES,
                "results": results,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2, ensure_ascii=False)

    log("---- 单专家 vs full 摘要（AUC） ----")
    for cond in CONDITIONS:
        row = "  ".join(f"{v}:{results[cond][v]['auc']:.4f}" for v in VARIANTS)
        log(f"  [{cond:<16}] {row}")


if __name__ == "__main__":
    main()
