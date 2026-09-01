#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成论文 Table 2: Baseline 对比实验

数据来源:
  - 4 个 baseline 模型 (mesonet / efficientnet_b0 / efficientnet_b3 / xception):
      checkpoint: results/baselines/<model>/best_model.pt
      训练数据:   Celeb-DF++ only (38,934 train, 与 CAR 完全相同划分)
      架构:       baseline_full.py 的 build_model() (MesoNet / FrameAvgModel)
      训练脚本:   baseline_full.py (CrossEntropyLoss, softmax 输出)
  - CAR v5 (单专家架构单独训练):
      checkpoint: checkpoints_v5/best_model.pt (epoch 23, val AUC 0.8625)
      config:     configs/default.yaml (v5 训练用配置)
      训练数据:   Celeb-DF++ only
  - CAR joint (联合训练主结果):
      直接引用 results/table1/config_0_Celeb-DF____test_/metrics.json (已有结果, 不重新评估)
      训练数据:   Celeb-DF++ + FF++

评估协议 (与 CAR 严格一致, 保证公平对比):
  1. 在 val 集用 Youden's Index (最大化 TPR - FPR) 确定阈值
  2. 阈值应用到 test 集计算 Acc / F1
  3. 阈值无关指标 (AUC / AP / EER / TPR@FPR) 直接在 test 集计算
  4. 强制 model.eval() + torch.no_grad()
  5. 评估时禁用 CutMix (config._data["training"]["cutmix_p"] = 0.0)
  6. num_workers = 0 (Windows 兼容)
  7. 数据划分与 CAR 完全一致 (38,934 / 9,734 / 5,418)
  8. 使用 src.data.dataloader.create_dataloader 加载数据, 与 CAR 评估同接口

报告 7 项指标: AUC, Accuracy, F1, AP, EER, TPR@FPR=1%, TPR@FPR=0.1%

评分方式 (各模型按其训练时所用损失一致):
  - baseline: softmax(logits)[:, 1]   (CrossEntropy 训练, 与 baseline_full.py evaluate 一致)
  - CAR:      sigmoid(logits[:, 1])   (BCE 训练, 与 evaluate_v2.py run_inference 一致)

严谨性:
  - 若 baseline 优于 CAR, 如实记录 (不隐藏)
  - 保存原始预测分数 raw_predictions.npz (含 preds, labels) 以支持复现与作图
  - 记录每个模型参数量 (baseline 优先从 results.json 读取, 否则重新计算; CAR 从已加载模型计算)
  - 因 baseline 架构与 CAR 不同, 不能用 subprocess 调用 evaluate_v2.py, 故在本脚本内实现评估逻辑

输出:
  - results/table2/<model>/raw_predictions.npz  (preds, labels [, video_ids])
  - results/table2/<model>/metrics.json
  - results/table2/comparison_results.json      (汇总所有模型)
  - 控制台打印 Table 2 预览
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

# 确保项目根目录在 sys.path 中, 支持 `python scripts/run_table2_baseline.py` 直接运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src.config import load_config
from src.data.dataloader import create_dataloader
from src.utils.metrics import (
    find_optimal_threshold,
    compute_auc,
    compute_accuracy,
    compute_f1,
    compute_ap,
    compute_eer,
    compute_tpr_at_fpr,
)
from baseline_full import build_model as build_baseline_model
from evaluate_v2 import load_model as load_car_model

# ----------------------------------------------------------------------------
# 路径与常量
# ----------------------------------------------------------------------------
BASELINE_MODELS = ["mesonet", "efficientnet_b0", "efficientnet_b3", "xception"]
BASELINE_CKPT_TMPL = os.path.join("results", "baselines", "{model}", "best_model.pt")
BASELINE_RESULTS_TMPL = os.path.join("results", "baselines", "{model}", "results.json")
CAR_V5_CKPT = os.path.join("checkpoints_v5", "best_model.pt")
CAR_V5_CONFIG = os.path.join("configs", "default.yaml")
CAR_JOINT_METRICS = os.path.join(
    "results", "table1", "config_0_Celeb-DF____test_", "metrics.json"
)
DEFAULT_OUTPUT_DIR = os.path.join("results", "table2")

# 训练数据范围标注 (Table 2 必须注明)
TRAIN_DATA_BASELINE = "Celeb-DF++ only"
TRAIN_DATA_CAR_V5 = "Celeb-DF++ only"
TRAIN_DATA_CAR_JOINT = "Celeb-DF++ + FF++"

DISPLAY_NAMES = {
    "mesonet": "MesoNet",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3",
    "xception": "Xception",
    "car_v5": "CAR (v5)",
    "car_joint": "CAR (joint)",
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
    """构建评估 DataLoader (num_workers=0, shuffle=False, 无增强).

    与 evaluate_v2.build_dataloader(celebdf) 一致: 复用项目统一接口
    create_dataloader, 仅强制 num_workers=0 (Windows 兼容, 避免序列化 lambda).
    val/test split 时 create_dataloader 内部 shuffle=False, transform=None, 无 sampler.
    """
    config._data["data"]["num_workers"] = 0
    return create_dataloader(config, split=split)


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


def car_forward(model, frames):
    """CAR 评分: sigmoid(logits[:, 1]) (BCE 训练, 与 evaluate_v2.py run_inference 一致)"""
    outputs = model(frames)
    logits = outputs["logits"]
    if logits.size(1) > 1:
        return torch.sigmoid(logits[:, 1])
    return torch.sigmoid(logits.squeeze(-1))


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
    """保存 metrics.json 与 raw_predictions.npz"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    save_dict = {"preds": preds, "labels": labels}
    if video_ids is not None and len(video_ids) > 0:
        save_dict["video_ids"] = np.asarray(video_ids, dtype=object)
    np.savez(os.path.join(output_dir, "raw_predictions.npz"), **save_dict)
    log(f"Saved metrics.json and raw_predictions.npz to {output_dir}")


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
    """评估单个 baseline 模型 (val 定阈值 -> test). 缺失 checkpoint 则跳过."""
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
        "training_data": TRAIN_DATA_BASELINE,
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
    out_dir = os.path.join(PROJECT_ROOT, DEFAULT_OUTPUT_DIR, model_name)
    save_model_results(out_dir, record, preds, labels, video_ids)
    return record


def evaluate_car_v5(config, device):
    """评估 CAR v5 单独训练模型 (val 定阈值 -> test). 缺失 checkpoint 则跳过."""
    ckpt_path = os.path.join(PROJECT_ROOT, CAR_V5_CKPT)
    if not os.path.exists(ckpt_path):
        log(f"[SKIP] CAR v5 checkpoint 不存在: {ckpt_path}")
        return None

    log("=== CAR v5 (single) ===")
    # 使用 evaluate_v2.py 的 load_model (优先 EMA 权重)
    model, ckpt_epoch, used_ema = load_car_model(config, ckpt_path, device)
    params = count_params(model)

    preds, labels, video_ids, metrics, val_info = evaluate_model(
        model, car_forward, config, device, desc_prefix="[CAR v5] "
    )

    record = {
        "model": "car_v5",
        "display_name": DISPLAY_NAMES["car_v5"],
        "params": params,
        "training_data": TRAIN_DATA_CAR_V5,
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ckpt_epoch,
        "used_ema": used_ema,
        "threshold_source": "val",
        "scoring": "sigmoid",
        "num_samples": int(len(labels)),
        "val_threshold_info": val_info,
        "confusion_matrix": compute_confusion_matrix(
            preds, labels, metrics["threshold"]
        ),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }
    out_dir = os.path.join(PROJECT_ROOT, DEFAULT_OUTPUT_DIR, "car_v5")
    save_model_results(out_dir, record, preds, labels, video_ids)
    return record


def load_car_joint(car_v5_params):
    """引用 CAR 联合训练已有结果 (Table 1), 不重新评估.

    raw_predictions.npz 已存在于 table1 目录, 供 figure2 直接读取.
    参数量: CAR joint 与 CAR v5 架构相同 (configs/default.yaml 与 joint_train.yaml
    的 model 段完全一致), 引用 v5 参数量.
    """
    metrics_path = os.path.join(PROJECT_ROOT, CAR_JOINT_METRICS)
    if not os.path.exists(metrics_path):
        log(f"[SKIP] CAR joint metrics 不存在: {metrics_path}")
        return None

    log("=== CAR (joint) === 引用已有 Table 1 结果 (不重新评估)")
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    record = {
        "model": "car_joint",
        "display_name": DISPLAY_NAMES["car_joint"],
        "params": car_v5_params,
        "params_note": (
            "架构与 CAR v5 一致 (configs/default.yaml 与 joint_train.yaml 的 "
            "model 段相同), 引用 v5 参数量"
        ),
        "training_data": TRAIN_DATA_CAR_JOINT,
        "checkpoint": m.get("checkpoint"),
        "checkpoint_epoch": m.get("checkpoint_epoch"),
        "used_ema": m.get("used_ema"),
        "threshold_source": m.get("threshold_source", "val"),
        "scoring": "sigmoid",
        "num_samples": m.get("num_samples"),
        "val_threshold_info": m.get("val_threshold_info"),
        "confusion_matrix": m.get("confusion_matrix"),
        "auc": m.get("auc"),
        "accuracy": m.get("accuracy"),
        "f1": m.get("f1"),
        "ap": m.get("ap"),
        "eer": m.get("eer"),
        "tpr_at_fpr_1": m.get("tpr_at_fpr_1"),
        "tpr_at_fpr_01": m.get("tpr_at_fpr_01"),
        "threshold": m.get("threshold"),
        "source_metrics": CAR_JOINT_METRICS,
    }
    log(f"Loaded CAR joint metrics from {metrics_path}")
    return record


def print_table2(results):
    """打印 Table 2 预览到控制台 (模型名 / 参数量 / 训练数据 / 7 项指标)."""
    print("\n" + "=" * 122)
    print("Table 2: Baseline Comparison on Celeb-DF++ Test Set")
    print("=" * 122)
    header = (
        f"{'Model':<18} {'Params':>14} {'TrainData':<22} "
        f"{'AUC':>8} {'Acc':>8} {'F1':>8} {'AP':>8} "
        f"{'EER':>8} {'TPR@1%':>8} {'TPR@0.1%':>10}"
    )
    print(header)
    print("-" * 122)
    for r in results:
        params = r.get("params")
        params_str = f"{params:,}" if params is not None else "N/A"
        print(
            f"{r['display_name']:<18} {params_str:>14} {r['training_data']:<22} "
            f"{r['auc']:>8.4f} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['ap']:>8.4f} "
            f"{r['eer']:>8.4f} {r['tpr_at_fpr_1']:>8.4f} {r['tpr_at_fpr_01']:>10.4f}"
        )
    print("=" * 122)
    print("注: Acc/F1 使用 val 集最优阈值 (Youden); AUC/AP/EER/TPR@FPR 为 test 集阈值无关指标")
    print("    baseline 与 CAR v5 训练数据: Celeb-DF++ only; CAR joint: Celeb-DF++ + FF++")
    print("    若 baseline 优于 CAR, 如实记录 (不隐藏)")


def main():
    parser = argparse.ArgumentParser(description="生成论文 Table 2: Baseline 对比")
    parser.add_argument(
        "--config",
        type=str,
        default=CAR_V5_CONFIG,
        help="配置文件路径 (v5 训练配置, 同时用于 baseline 与 CAR v5 的数据加载)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="汇总结果输出目录",
    )
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()

    # 设备解析
    if args.device == "cuda" and not torch.cuda.is_available():
        log("CUDA 不可用, 回退到 CPU")
        device = "cpu"
    else:
        device = args.device

    # 加载配置 (v5 训练配置: 数据划分与 baseline 训练一致)
    log(f"Loading config from {args.config}")
    config = load_config(args.config)

    # 评估严谨性: 禁用 CutMix, num_workers=0
    disable_cutmix(config)
    config._data["data"]["num_workers"] = 0
    set_seed(getattr(config.training, "seed", 42))

    output_root = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    results = []
    car_v5_params = None

    # 1. 评估 4 个 baseline
    log("Evaluating baselines on Celeb-DF++ test (protocol identical to CAR)...")
    for name in BASELINE_MODELS:
        rec = evaluate_baseline(name, config, device)
        if rec is not None:
            results.append(rec)

    # 2. 评估 CAR v5 (单独训练)
    rec = evaluate_car_v5(config, device)
    if rec is not None:
        car_v5_params = rec["params"]
        results.append(rec)

    # 3. 引用 CAR joint (联合训练已有结果)
    rec = load_car_joint(car_v5_params)
    if rec is not None:
        results.append(rec)

    if not results:
        log("[WARN] 没有任何模型可评估 (所有 checkpoint 均缺失). 请先完成 baseline 训练.")
        return

    # 保存汇总 JSON
    summary_path = os.path.join(output_root, "comparison_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"table": "Table 2: Baseline Comparison", "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    log(f"汇总结果已保存到 {summary_path}")

    # 打印表格预览
    print_table2(results)


if __name__ == "__main__":
    main()
