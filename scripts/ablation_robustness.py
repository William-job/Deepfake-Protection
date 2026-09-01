#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消融×鲁棒性交叉实验（解决 -spectral 悖论，论文 Ablation×Robustness 表专用）

动机：clean 消融显示 -spectral 的 AUC 反而 +0.64 点（负贡献），但审稿人
必然追问："spectral 专家是否在退化条件下有回报？" 本实验正面回答：
把 6 个门控消融变体放到 4 个关键退化条件下重新评估。

协议（与 robustness_honest.py / ablation_v3.py 完全一致）：
- 全量 test 集 5418（uint8 帧缓存，mmap）；
- 退化实现与 robustness_honest.py 逐字节一致（uint8 空间）；
- 每条件前 np.random.seed(2024)；
- 阈值无关指标（AUC/AP）——消融变体的阈值无意义；
- 效率优化：一次前向产出全部 6 个变体的概率（主干共享，门控重组合
  的开销可忽略），GPU 前向次数 = 条件数而非 条件×变体数。

变体：
    full / -motion / -temporal / -spectral / -boundary / uniform

条件：
    clean, noise_std=0.05, blur_kernel=7, jpeg_quality=30

用法（等 B0+QAug 训练结束后运行，避免 GPU 竞争）：
    python -u scripts/ablation_robustness.py

输出：
    results/ablation_robustness/ablation_robustness.json
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
from robustness_honest import apply_degradation_batch, build_test_cache, log

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "ablation_robustness")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")
NOISE_SEED = 2024

EXPERT_NAMES = ["motion", "temporal", "spectral", "boundary"]  # 与 car.expert_names 一致

# 消融变体：name -> weight 修改规则
VARIANTS = {
    "full": "none",
    "-motion": "drop:0",
    "-temporal": "drop:1",
    "-spectral": "drop:2",
    "-boundary": "drop:3",
    "uniform_gating": "uniform",
}

# 关键退化条件（noise/blur/jpeg 的代表级）
CONDITIONS = {
    "clean": None,
    "noise_std=0.05": ("noise", 0.05),
    "blur_kernel=7": ("blur", 7),
    "jpeg_quality=30": ("jpeg", 30),
}


def modify_weights(w, rule):
    """按规则修改门控权重（与 ablation_v3.GatedVariant 逻辑逐行一致）。"""
    if rule == "none":
        return w
    if rule == "uniform":
        return torch.full_like(w, 0.25)
    if rule.startswith("drop:"):
        idx = int(rule.split(":")[1])
        w = w.clone()
        w[:, idx] = 0
        w_sum = w.sum(dim=1, keepdim=True)
        w_sum = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
        return w / w_sum
    raise ValueError(rule)


@torch.no_grad()
def eval_all_variants(model, frames_mmap, labels, device, deg_type=None, severity=None,
                      batch_size=16):
    """一次前向同时计算全部变体的 AUC/AP（退化在 uint8 空间逐 batch 应用）。

    返回 {variant_name: {"auc":..., "ap":...}}。
    preds 缓存到内存（5418×6 float64 ≈ 260KB，无压力）。
    """
    n = frames_mmap.shape[0]
    preds = {v: np.zeros(n, dtype=np.float64) for v in VARIANTS}

    for i in range(0, n, batch_size):
        batch_u8 = np.asarray(frames_mmap[i:i + batch_size])
        batch_u8 = apply_degradation_batch(batch_u8, deg_type, severity)
        x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
        x = ((x / 255.0 - 0.5) / 0.5).to(device)

        out = model(x)
        head_outputs, z, w = out["head_outputs"], out["z"], out["w"]

        # 重算 expert logits（与 ablation_v3.GatedVariant 一致）
        logit_list = []
        for name in model.expert_names:
            expert_out = model.experts[name](head_outputs[name])
            if expert_out.dim() == 2 and expert_out.size(0) != z.size(0):
                if expert_out.size(0) % z.size(0) == 0:
                    T = expert_out.size(0) // z.size(0)
                    expert_out = expert_out.view(z.size(0), T, -1).mean(dim=1)
                else:
                    repeat_factor = z.size(0) // expert_out.size(0)
                    if repeat_factor > 0:
                        expert_out = expert_out.repeat(repeat_factor, 1)
            logit_list.append(expert_out)
        stacked = torch.stack(logit_list, dim=1).clamp(-100.0, 100.0)  # (B,4,1)

        for vname, rule in VARIANTS.items():
            w_mod = modify_weights(w, rule)
            y_combined = (w_mod.unsqueeze(-1) * stacked).sum(dim=1)  # (B,2) 或 (B,1)
            if y_combined.size(1) > 1:
                p = torch.sigmoid(y_combined[:, 1])
            else:
                p = torch.sigmoid(y_combined.squeeze(-1))
            preds[vname][i:i + batch_u8.shape[0]] = p.cpu().numpy()

    results = {}
    for vname, p in preds.items():
        results[vname] = {
            "auc": float(compute_auc(p, labels)),
            "ap": float(compute_ap(p, labels)),
        }
    return results


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    frames, labels, video_ids, _ = build_test_cache(config)
    log(f"test 缓存: {frames.shape[0]} 样本")

    ckpt_path = os.path.join(PROJECT_ROOT, "results", "final_car_v3", "checkpoints", "best_model.pt")
    model = CAR(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("ema_shadow") is not None:
        try:
            model.load_state_dict(ckpt["ema_shadow"], strict=True)
        except RuntimeError:
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ablation_robustness.json")

    results = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f).get("results", {})

    for cond, deg in CONDITIONS.items():
        if cond in results:
            log(f"[{cond}] 已完成，跳过")
            continue
        deg_type, severity = (deg if deg else (None, None))
        np.random.seed(NOISE_SEED)  # 与 robustness_honest 相同噪声实现
        log(f"评估条件 {cond} ...")
        results[cond] = eval_all_variants(model, frames, labels, device,
                                          deg_type=deg_type, severity=severity)
        for vname, m in results[cond].items():
            log(f"  [{cond}] {vname:<16} AUC={m['auc']:.4f}  AP={m['ap']:.4f}")
        # 增量写入（崩溃安全）
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": ckpt_path,
                "protocol": ("full test 5418, uint8 cache, noise_seed=2024, "
                             "threshold-free (AUC/AP), variants share one forward pass"),
                "variants": VARIANTS,
                "conditions": list(CONDITIONS.keys()),
                "results": results,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2, ensure_ascii=False)

    # ---- 汇总：spectral 回报分析 ----
    if all(c in results for c in CONDITIONS):
        log("---- 消融×鲁棒性摘要（ΔAUC vs full，各条件） ----")
        for cond in CONDITIONS:
            base = results[cond]["full"]["auc"]
            deltas = {v: results[cond][v]["auc"] - base for v in VARIANTS if v != "full"}
            ds = "  ".join(f"{v}:{d:+.4f}" for v, d in deltas.items())
            log(f"  [{cond:<16}] {ds}")

    log(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
