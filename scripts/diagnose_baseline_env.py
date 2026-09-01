#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基线环境一致性诊断

问题：robustness_honest.py 本地实测 B0 clean AUC=0.8348，
但 baseline_honest 报告 0.9060（eval_metrics.json 的 checkpoint 路径为
/root/trae/... —— Linux 环境产物；本地 best_model.pt mtime 晚于评估时间）。

诊断：在本地 val 集上重算各基线 checkpoint 的 AUC，与 eval_metrics.json 中的
val_auc 对比：
  - 一致 → checkpoint 相同，差异来自 test 数据/解码；
  - 不一致 → 本地 checkpoint ≠ 被评估权重（本地被覆写）。

顺带构建 val 帧缓存（后续 qaug 阈值冻结复用）。
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from robustness_honest import (
    build_val_cache, load_baseline, forward_probs, MODEL_SPECS, log,
)
from src.utils.metrics import compute_auc

OUT_PATH = os.path.join(PROJECT_ROOT, "results", "robustness_honest", "baseline_env_diagnosis.json")


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    log(f"Device: {device}")

    val_frames, val_labels = build_val_cache(config)
    n = val_frames.shape[0]
    log(f"val 缓存: {n} 样本, {len(val_labels)} 标签")

    results = {}
    for name in ["efficientnet_b0", "xception", "mesonet"]:
        spec = MODEL_SPECS[name]
        with open(spec["threshold"], "r", encoding="utf-8") as f:
            reported = json.load(f)
        reported_val_auc = reported.get("val_threshold_info", {}).get("val_auc")
        reported_test_auc = reported.get("auc")
        reported_ckpt_path = reported.get("checkpoint", "?")
        reported_time = reported.get("timestamp", "?")
        ckpt_mtime = os.path.getmtime(spec["ckpt"])

        model = load_baseline(name, config.data.num_frames, spec["ckpt"], device)
        model.eval()
        preds_all = np.zeros(n, dtype=np.float64)
        for i in tqdm(range(0, n, 16), desc=f"{name}:val", ncols=80):
            batch_u8 = np.asarray(val_frames[i:i + 16])
            x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
            x = (x / 255.0 - 0.5) / 0.5
            p = forward_probs(model, x.to(device), is_car=False)
            preds_all[i:i + batch_u8.shape[0]] = p.cpu().numpy()
        local_val_auc = float(compute_auc(preds_all, val_labels))

        consistent = (reported_val_auc is not None
                      and abs(local_val_auc - reported_val_auc) < 0.01)
        results[name] = {
            "local_val_auc": local_val_auc,
            "reported_val_auc": reported_val_auc,
            "reported_test_auc": reported_test_auc,
            "val_consistent": bool(consistent),
            "reported_checkpoint_path": reported_ckpt_path,
            "reported_eval_timestamp": reported_time,
            "local_checkpoint_mtime": ckpt_mtime,
            "verdict": ("一致：权重相同，差异在 test 侧" if consistent else
                        "不一致：本地 checkpoint ≠ 被评估权重（或数据不同）"),
        }
        log(f"[{name}] local val AUC={local_val_auc:.4f} vs reported {reported_val_auc} "
            f"({results[name]['verdict']})")
        del model
        torch.cuda.empty_cache()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
