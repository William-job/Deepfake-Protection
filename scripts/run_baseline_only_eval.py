#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""精简版 Baseline 评估脚本 (仅 4 个 baseline, 跳过 CAR)

背景:
  scripts/run_table2_baseline.py 评估 5 个模型 (4 baseline + CAR v5), 但使用
  num_workers=0 导致极慢 (约 30 小时). 本脚本为精简版:
    - 仅评估 4 个 baseline (mesonet / efficientnet_b0 / efficientnet_b3 / xception)
    - 关键修改: num_workers=0 -> num_workers=4 (与训练时一致, 大幅加速)

评估协议 (与 run_table2_baseline.py / CAR 严格一致, 保证公平对比):
  1. 在 val 集用 Youden's Index (最大化 TPR - FPR) 确定阈值
  2. 阈值应用到 test 集计算 Acc / F1
  3. 阈值无关指标 (AUC / AP / EER / TPR@FPR) 直接在 test 集计算
  4. 强制 model.eval() + torch.no_grad()
  5. 评估时禁用 CutMix (config._data["training"]["cutmix_p"] = 0.0)
  6. DataLoader: num_workers=4 + pin_memory=True + persistent_workers=True
     + prefetch_factor=4 (非 num_workers=0!)
  7. 数据划分与 CAR 完全一致 (38,934 / 9,734 / 5,418)
  8. 使用 src.data.dataloader.create_dataloader 加载数据 (强制 num_workers=4)

评分方式: softmax(logits)[:, 1]  (CrossEntropy 训练, 与 baseline_full.py 一致)

报告 7 项指标: AUC, Accuracy, F1, AP, EER, TPR@FPR=1%, TPR@FPR=0.1%

输出:
  - results/baselines/<model>/eval_metrics.json
  - results/baselines/<model>/raw_predictions.npz  (preds, labels [, video_ids])

特性:
  - 支持跳过已完成模型 (检测 eval_metrics.json 存在则跳过)
  - 控制台打印评估进度和结果摘要
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# 确保项目根目录在 sys.path 中, 支持 `python scripts/run_baseline_only_eval.py` 直接运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import DeepfakeDataset
from src.utils.metrics import (
    find_optimal_threshold,
    compute_auc,
    compute_accuracy,
    compute_f1,
    compute_ap,
    compute_eer,
    compute_tpr_at_fpr,
)
from baseline_full import build_model as build_baseline_model, collate_fn

# ----------------------------------------------------------------------------
# 路径与常量
# ----------------------------------------------------------------------------
BASELINE_MODELS = ["mesonet", "efficientnet_b0", "efficientnet_b3", "xception"]
BASELINE_CKPT_TMPL = os.path.join("results", "baselines", "{model}", "best_model.pt")
BASELINE_RESULTS_TMPL = os.path.join("results", "baselines", "{model}", "results.json")
DEFAULT_CONFIG = os.path.join("configs", "default.yaml")

# DataLoader 加速参数 (与训练时一致, 替代原 num_workers=0)
EVAL_NUM_WORKERS = 4
EVAL_PIN_MEMORY = True
EVAL_PERSISTENT_WORKERS = True
EVAL_PREFETCH_FACTOR = 4

DISPLAY_NAMES = {
    "mesonet": "MesoNet",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3",
    "xception": "Xception",
}


def log(msg):
    """统一带 [INFO] 前缀的日志输出"""
    print(f"[INFO] {msg}")


def set_seed(seed):
    """固定随机种子, 保证评估可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    """统计模型总参数量"""
    return int(sum(p.numel() for p in model.parameters()))


def disable_cutmix(config):
    """评估时禁用 CutMix 等随机增强 (运行时修改内存对象, 不改动配置文件)"""
    if "training" in config._data and isinstance(config._data["training"], dict):
        config._data["training"]["cutmix_p"] = 0.0


def build_eval_loader(config, split):
    """构建评估 DataLoader (num_workers=4, shuffle=False, 无增强).

    直接构建 DataLoader (绕过 create_dataloader, 因其内部使用 lambda
    worker_init_fn, 在 Windows + num_workers>0 时无法 pickle).

    与 baseline_full.py 训练时 DataLoader 设置一致:
      num_workers=4 + pin_memory=True + persistent_workers=True + prefetch_factor=4
    """
    ds = DeepfakeDataset(
        config.data.data_root,
        split=split,
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
    )
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=EVAL_NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=EVAL_PIN_MEMORY,
        persistent_workers=EVAL_PERSISTENT_WORKERS,
        prefetch_factor=EVAL_PREFETCH_FACTOR,
    )
    return loader


@torch.no_grad()
def run_inference(model, loader, device, forward_fn, desc="Evaluating"):
    """通用推理: forward_fn(model, frames) -> preds tensor [B] (fake 概率).

    强制 model.eval() + torch.no_grad(). 收集 preds / labels / video_ids.
    """
    model.eval()  # 双重保险: 确保评估模式
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


def baseline_forward(model, frames):
    """baseline 评分: softmax(logits)[:, 1] (CrossEntropy 训练, 与 baseline_full.py 一致)"""
    logits = model(frames)  # [B, 2]
    return torch.softmax(logits, dim=1)[:, 1]


def compute_metrics_with_threshold(preds, labels, threshold):
    """用指定阈值计算 Acc/F1, 其余阈值无关指标直接计算 (与 evaluate_v2.py 一致)"""
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
    """计算混淆矩阵 (label=1 为 fake)"""
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten().astype(int)
    pred_binary = (preds >= threshold).astype(int)
    tp = int(np.sum((pred_binary == 1) & (labels == 1)))
    tn = int(np.sum((pred_binary == 0) & (labels == 0)))
    fp = int(np.sum((pred_binary == 1) & (labels == 0)))
    fn = int(np.sum((pred_binary == 0) & (labels == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_model(model, forward_fn, config, device, desc_prefix=""):
    """val 集确定阈值 -> test 集计算最终指标 (与 CAR 评估协议完全一致)."""
    log(f"{desc_prefix}Running inference on val split (for threshold)...")
    val_loader = build_eval_loader(config, "val")
    val_preds, val_labels, _ = run_inference(
        model, val_loader, device, forward_fn, desc="Val(thr)"
    )
    threshold = find_optimal_threshold(val_preds, val_labels)
    log(f"{desc_prefix}Optimal threshold from val (Youden): {threshold:.4f}")

    log(f"{desc_prefix}Running inference on test split (final)...")
    test_loader = build_eval_loader(config, "test")
    preds, labels, video_ids = run_inference(
        model, test_loader, device, forward_fn, desc="Test"
    )
    metrics = compute_metrics_with_threshold(preds, labels, threshold)
    val_info = {
        "val_num_samples": int(len(val_labels)),
        "val_auc": float(compute_auc(val_preds, val_labels)),
    }
    return preds, labels, video_ids, metrics, val_info


def save_model_results(output_dir, record, preds, labels, video_ids):
    """保存 eval_metrics.json 与 raw_predictions.npz"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    save_dict = {"preds": preds, "labels": labels}
    if video_ids is not None and len(video_ids) > 0:
        save_dict["video_ids"] = np.asarray(video_ids, dtype=object)
    np.savez(os.path.join(output_dir, "raw_predictions.npz"), **save_dict)
    log(f"Saved eval_metrics.json and raw_predictions.npz to {output_dir}")


def get_baseline_params(model_name, model):
    """优先从 results/baselines/<model>/results.json 读取参数量, 否则从模型重新计算."""
    results_path = os.path.join(
        PROJECT_ROOT, BASELINE_RESULTS_TMPL.format(model=model_name)
    )
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                r = json.load(f)
            if "params" in r:
                return int(r["params"])
        except Exception:
            pass
    return count_params(model)


def evaluate_baseline(model_name, config, device):
    """评估单个 baseline 模型 (val 定阈值 -> test).

    - 缺失 checkpoint 则跳过
    - 已存在 eval_metrics.json 则跳过 (支持断点续跑)
    """
    out_dir = os.path.join(PROJECT_ROOT, "results", "baselines", model_name)
    metrics_path = os.path.join(out_dir, "eval_metrics.json")
    if os.path.exists(metrics_path):
        log(f"[SKIP] {model_name}: eval_metrics.json 已存在 -> {metrics_path}")
        return None

    ckpt_path = os.path.join(
        PROJECT_ROOT, BASELINE_CKPT_TMPL.format(model=model_name)
    )
    if not os.path.exists(ckpt_path):
        log(f"[SKIP] baseline checkpoint 不存在 (训练可能尚未完成): {ckpt_path}")
        return None

    log(f"=== Baseline: {model_name} ===")
    model = build_baseline_model(model_name, config.data.num_frames).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # baseline 不使用 EMA, 直接加载 model_state_dict
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ckpt_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    ckpt_val_auc = ckpt.get("val_auc") if isinstance(ckpt, dict) else None
    params = get_baseline_params(model_name, model)

    preds, labels, video_ids, metrics, val_info = evaluate_model(
        model, baseline_forward, config, device, desc_prefix=f"[{model_name}] "
    )

    record = {
        "model": model_name,
        "display_name": DISPLAY_NAMES.get(model_name, model_name),
        "params": params,
        "training_data": "Celeb-DF++ only",
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_val_auc": ckpt_val_auc,
        "used_ema": False,
        "threshold_source": "val",
        "scoring": "softmax",
        "num_samples": int(len(labels)),
        "val_threshold_info": val_info,
        "confusion_matrix": compute_confusion_matrix(
            preds, labels, metrics["threshold"]
        ),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }
    save_model_results(out_dir, record, preds, labels, video_ids)
    return record


def print_result_summary(record):
    """打印单个模型结果摘要到控制台"""
    print(
        f"  -> {record['display_name']}: "
        f"AUC={record['auc']:.4f}  Acc={record['accuracy']:.4f}  "
        f"F1={record['f1']:.4f}  AP={record['ap']:.4f}  "
        f"EER={record['eer']:.4f}  "
        f"TPR@1%={record['tpr_at_fpr_1']:.4f}  "
        f"TPR@0.1%={record['tpr_at_fpr_01']:.4f}  "
        f"(thr={record['threshold']:.4f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="精简版 Baseline 评估 (仅 4 baseline, num_workers=4 加速)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="配置文件路径 (与 baseline 训练 / CAR 评估同配置, 保证数据划分一致)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()

    # 设备解析
    if args.device == "cuda" and not torch.cuda.is_available():
        log("CUDA 不可用, 回退到 CPU")
        device = "cpu"
    else:
        device = args.device

    # 加载配置 (与 baseline 训练 / CAR 评估同配置: 数据划分 38934/9734/5418 一致)
    log(f"Loading config from {args.config}")
    config = load_config(args.config)

    # 评估严谨性: 禁用 CutMix; 强制 num_workers=4 (加速)
    disable_cutmix(config)
    config._data["data"]["num_workers"] = EVAL_NUM_WORKERS
    set_seed(getattr(config.training, "seed", 42))

    log(
        f"DataLoader: num_workers={EVAL_NUM_WORKERS}, pin_memory={EVAL_PIN_MEMORY}, "
        f"persistent_workers={EVAL_PERSISTENT_WORKERS}, prefetch_factor={EVAL_PREFETCH_FACTOR}"
    )
    log(f"评估 {len(BASELINE_MODELS)} 个 baseline: {BASELINE_MODELS}")

    results = []
    for name in BASELINE_MODELS:
        rec = evaluate_baseline(name, config, device)
        if rec is not None:
            print_result_summary(rec)
            results.append(rec)

    if not results:
        log("[WARN] 没有任何模型被评估 (可能均已完成或 checkpoint 缺失).")
        return

    # 打印汇总摘要
    print("\n" + "=" * 100)
    print("Baseline 评估结果汇总 (Celeb-DF++ test)")
    print("=" * 100)
    header = (
        f"{'Model':<18} {'Params':>14} "
        f"{'AUC':>8} {'Acc':>8} {'F1':>8} {'AP':>8} "
        f"{'EER':>8} {'TPR@1%':>8} {'TPR@0.1%':>10}"
    )
    print(header)
    print("-" * 100)
    for r in results:
        params = r.get("params")
        params_str = f"{params:,}" if params is not None else "N/A"
        print(
            f"{r['display_name']:<18} {params_str:>14} "
            f"{r['auc']:>8.4f} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['ap']:>8.4f} "
            f"{r['eer']:>8.4f} {r['tpr_at_fpr_1']:>8.4f} {r['tpr_at_fpr_01']:>10.4f}"
        )
    print("=" * 100)
    print("注: Acc/F1 使用 val 集最优阈值 (Youden); AUC/AP/EER/TPR@FPR 为 test 集阈值无关指标")
    print(f"共评估 {len(results)} 个模型. 详细结果见 results/baselines/<model>/eval_metrics.json")


if __name__ == "__main__":
    main()
