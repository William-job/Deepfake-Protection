"""顶刊标准评估引擎 v2

支持 Celeb-DF++ 与 FF++ 两个数据集的严格评估，输出符合顶刊投稿要求的
指标（AUC / AP / F1 / Acc / EER / TPR@FPR=1e-2 / TPR@FPR=1e-3）、
混淆矩阵及原始预测，便于复现与作图。

严谨性保证：
  - 强制 model.eval() + torch.no_grad()
  - 评估时禁用 CutMix 等一切随机增强
  - 若 checkpoint 含 ema_shadow，优先加载 EMA 权重
  - 支持 val 集确定阈值后泛化到 test 集（阈值无关指标仍在 test 集计算）
"""
import argparse
import json
import os
import sys
from datetime import datetime

# 确保项目根目录在 sys.path 中，支持 `python scripts/evaluate_v2.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.dataloader import create_dataloader
from src.data.ff_frame_dataset import FFFrameDataset
from src.models.car import CAR
from src.utils.metrics import (
    compute_all_metrics,
    compute_accuracy,
    compute_auc,
    compute_ap,
    compute_eer,
    compute_f1,
    compute_tpr_at_fpr,
    find_optimal_threshold,
)


def log(msg):
    """统一带 [INFO] 前缀的日志输出"""
    print(f"[INFO] {msg}")


def set_seed(seed):
    """固定随机种子，保证评估可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(config, checkpoint_path, device):
    """加载 CAR 模型与 checkpoint，优先使用 EMA 权重

    严谨性：若 EMA shadow 缺少部分键（如 freeze_stem=true 时 stem.backbone 未被 EMA 跟踪），
    回退到 model_state_dict，并记录原因。
    """
    model = CAR(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    used_ema = False
    # 若 checkpoint 含 ema_shadow，优先加载 EMA 权重（顶刊标准：EMA 通常更稳）
    if isinstance(ckpt, dict) and ckpt.get("ema_shadow") is not None:
        try:
            model.load_state_dict(ckpt["ema_shadow"], strict=True)
            log("Using EMA weights (strict load)")
            used_ema = True
        except RuntimeError as e:
            # EMA shadow 可能缺少 stem.backbone 键（freeze_stem=true 时 EMA 未跟踪 stem）
            log(f"EMA strict load failed: {str(e)[:100]}...，回退到 model_state_dict")
            # 尝试非严格加载 EMA，再用 model_state_dict 补齐缺失键
            model.load_state_dict(ckpt["ema_shadow"], strict=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            log("Loaded: EMA shadow (non-strict) + model_state_dict 补齐缺失键")
            used_ema = True
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()  # 强制评估模式
    ckpt_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, ckpt_epoch, used_ema


def disable_train_augmentation(config):
    """评估时禁用 CutMix 等随机增强（运行时修改 config 内存对象，不改动文件）。

    create_dataloader 在 cutmix_p > 0 时会注入 CutMixCollate，既会引入随机
    增强又会丢弃 video_path 字段，评估时必须关闭。
    """
    if "training" in config._data and isinstance(config._data["training"], dict):
        config._data["training"]["cutmix_p"] = 0.0


class FFEvalCollate:
    """FF++ 评估专用 collate：堆叠帧/标签，保留 frame_dir 用于 per-video 聚合"""

    def __call__(self, batch):
        return {
            "frames": torch.stack([b["frames"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "frame_dir": [b["frame_dir"] for b in batch],
            "method": [b.get("method") for b in batch],
        }


def build_dataloader(config, dataset, split, compression):
    """根据数据集类型构建评估用 DataLoader（shuffle=False，无增强）"""
    if dataset == "celebdf":
        # celebdf：用项目统一接口，注意 shuffle=False
        # 评估时强制 num_workers=0，避免 Windows 多进程无法序列化 lambda worker_init_fn
        original_num_workers = config._data.get("data", {}).get("num_workers", 4)
        config._data["data"]["num_workers"] = 0
        loader = create_dataloader(config, split=split)
        config._data["data"]["num_workers"] = original_num_workers  # 恢复
        return loader
    elif dataset == "ffpp":
        # ffpp：FFFrameDataset + 手动 DataLoader 包装
        ds = FFFrameDataset(
            data_root=config.data.ff_root,
            num_frames=config.data.num_frames,
            frame_stride=config.data.frame_stride,
            image_size=config.data.image_size,
            split=split,
            compression=compression,
        )
        loader = DataLoader(
            ds,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,  # 评估时单进程，避免序列化问题
            pin_memory=True,
            drop_last=False,
            collate_fn=FFEvalCollate(),
        )
        return loader
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


@torch.no_grad()
def run_inference(model, loader, device, desc="Evaluating"):
    """在指定 loader 上推理，收集 preds / labels / video_ids

    返回:
        preds:      np.ndarray (N,) fake 类概率
        labels:     np.ndarray (N,) int
        video_ids:  list[str] 长度 N（celebdf 用 video_path，ffpp 用 frame_dir）
    """
    model.eval()  # 双重保险：确保评估模式
    all_preds, all_labels, all_ids = [], [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        outputs = model(frames)
        logits = outputs["logits"]
        # 取 fake 类概率（logits 第二列）
        if logits.size(1) > 1:
            preds = torch.sigmoid(logits[:, 1])
        else:
            preds = torch.sigmoid(logits.squeeze(-1))

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # 收集 video 标识用于 per-video 聚合
        if "video_path" in batch:
            ids = batch["video_path"]
            all_ids.extend([str(v) for v in ids])
        elif "frame_dir" in batch:
            ids = batch["frame_dir"]
            all_ids.extend([str(v) for v in ids])
        else:
            # 无标识时用样本序号占位
            all_ids.extend([str(i) for i in range(len(preds))])

    preds = np.concatenate(all_preds).astype(np.float64)
    labels = np.concatenate(all_labels).astype(int)
    return preds, labels, all_ids[: len(preds)]


def compute_confusion_matrix(preds, labels, threshold):
    """计算混淆矩阵（label=1 为 fake）"""
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten().astype(int)
    pred_binary = (preds >= threshold).astype(int)
    tp = int(np.sum((pred_binary == 1) & (labels == 1)))
    tn = int(np.sum((pred_binary == 0) & (labels == 0)))
    fp = int(np.sum((pred_binary == 1) & (labels == 0)))
    fn = int(np.sum((pred_binary == 0) & (labels == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics_with_threshold(preds, labels, threshold):
    """用指定阈值计算 Acc/F1，其余阈值无关指标直接计算"""
    return {
        "accuracy": float(compute_accuracy(preds, labels, threshold)),
        "auc": float(compute_auc(preds, labels)),
        "ap": float(compute_ap(preds, labels)),
        "f1": float(compute_f1(preds, labels, threshold)),
        "threshold": float(threshold),
        "eer": float(compute_eer(preds, labels)),
        "tpr_at_fpr_1": float(compute_tpr_at_fpr(preds, labels, 0.01)),
        "tpr_at_fpr_01": float(compute_tpr_at_fpr(preds, labels, 0.001)),
    }


def print_results_table(record):
    """打印格式化结果表格到控制台"""
    print("\n" + "=" * 70)
    print(" " * 22 + "EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Dataset:      {record['dataset']}")
    print(f"  Split:        {record['split']}")
    print(f"  Checkpoint:   {record['checkpoint']} (epoch {record.get('checkpoint_epoch', '?')})")
    print(f"  EMA weights:  {'Yes' if record.get('used_ema') else 'No'}")
    thr_src = record.get("threshold_source", "youden")
    print(f"  Threshold:    {record['threshold']:.4f} (source: {thr_src})")
    print(f"  Samples:      {record['num_samples']}")
    print("-" * 70)
    print(f"  {'Metric':<28} {'Value':>10}")
    print("-" * 70)
    print(f"  {'AUC':<28} {record['auc']:>10.4f}")
    print(f"  {'Accuracy':<28} {record['accuracy']:>10.4f}")
    print(f"  {'F1':<28} {record['f1']:>10.4f}")
    print(f"  {'AP':<28} {record['ap']:>10.4f}")
    print(f"  {'EER':<28} {record['eer']:>10.4f}")
    print(f"  {'TPR@FPR=1e-2':<28} {record['tpr_at_fpr_1']:>10.4f}")
    print(f"  {'TPR@FPR=1e-3':<28} {record['tpr_at_fpr_01']:>10.4f}")
    print("-" * 70)
    cm = record["confusion_matrix"]
    print(f"  Confusion Matrix:  TP={cm['tp']:<6} FP={cm['fp']}")
    print(f"                     FN={cm['fn']:<6} TN={cm['tn']}")
    print("=" * 70)


def save_results(record, preds, labels, video_ids, output_dir):
    """保存 metrics.json 与 raw_predictions.npz"""
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    log(f"Metrics saved to {metrics_path}")

    npz_path = os.path.join(output_dir, "raw_predictions.npz")
    save_dict = {"preds": preds, "labels": labels}
    if video_ids is not None and len(video_ids) > 0:
        save_dict["video_ids"] = np.asarray(video_ids, dtype=object)
    np.savez(npz_path, **save_dict)
    log(f"Raw predictions saved to {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CAR 顶刊标准评估引擎 v2 (Celeb-DF++ / FF++)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--config", type=str, default="configs/joint_train.yaml",
                        help="配置文件路径")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["celebdf", "ffpp"],
                        help="评估数据集")
    parser.add_argument("--compression", type=str, default="c23",
                        choices=["c23", "c40"],
                        help="FF++ 压缩等级（仅 ffpp 使用）")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="评估 split")
    parser.add_argument("--output_dir", type=str, default="results/eval_temp",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--use_val_threshold", action="store_true",
                        help="在 val 集确定阈值后应用到 test 集")
    args = parser.parse_args()

    # 设备解析
    if args.device == "cuda" and not torch.cuda.is_available():
        log("CUDA 不可用，回退到 CPU")
        device = "cpu"
    else:
        device = args.device

    # 加载配置
    log(f"Loading config from {args.config}")
    config = load_config(args.config)

    # 固定随机种子
    seed = getattr(config.training, "seed", 42)
    set_seed(seed)

    # 禁用 CutMix 等训练期随机增强（评估严谨性）
    disable_train_augmentation(config)

    # 加载模型与 checkpoint（优先 EMA）
    log(f"Loading model from {args.checkpoint}")
    model, ckpt_epoch, used_ema = load_model(config, args.checkpoint, device)

    timestamp = datetime.now().isoformat(timespec="seconds")

    if args.use_val_threshold:
        # —— 流程 A：val 集确定阈值，泛化到 test 集 ——
        log("Mode: val-threshold → test (避免 test 集阈值过拟合)")
        val_loader = build_dataloader(config, args.dataset, "val", args.compression)
        log(f"Running inference on val split (for threshold)...")
        val_preds, val_labels, _ = run_inference(
            model, val_loader, device, desc="Val(threshold)")
        threshold = find_optimal_threshold(val_preds, val_labels)
        log(f"Optimal threshold from val (Youden): {threshold:.4f}")

        # 在 test 集计算最终指标：阈值无关指标直接在 test 算，Acc/F1 用 val 阈值
        test_loader = build_dataloader(config, args.dataset, "test", args.compression)
        log(f"Running inference on test split (final)...")
        preds, labels, video_ids = run_inference(
            model, test_loader, device, desc="Test")
        metrics = compute_metrics_with_threshold(preds, labels, threshold)

        final_split = "test"
        threshold_source = "val"
        val_info = {
            "val_num_samples": int(len(val_labels)),
            "val_auc": float(compute_auc(val_preds, val_labels)),
        }
    else:
        # —— 流程 B：直接在指定 split 上用 compute_all_metrics（内部 Youden 定阈值）——
        log(f"Mode: direct evaluation on split={args.split}")
        loader = build_dataloader(config, args.dataset, args.split, args.compression)
        log(f"Running inference on {args.split} split...")
        preds, labels, video_ids = run_inference(
            model, loader, device, desc=args.split)
        metrics = compute_all_metrics(preds, labels)
        final_split = args.split
        threshold_source = "youden"
        val_info = None

    # 混淆矩阵（用最终阈值）
    cm = compute_confusion_matrix(preds, labels, metrics["threshold"])

    # 组装结果记录
    record = {
        "dataset": args.dataset,
        "split": final_split,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt_epoch,
        "config_path": args.config,
        "timestamp": timestamp,
        "used_ema": used_ema,
        "threshold_source": threshold_source,
        "threshold": metrics["threshold"],
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "ap": metrics["ap"],
        "eer": metrics["eer"],
        "tpr_at_fpr_1": metrics["tpr_at_fpr_1"],
        "tpr_at_fpr_01": metrics["tpr_at_fpr_01"],
        "num_samples": int(len(labels)),
        "confusion_matrix": cm,
    }
    if args.dataset == "ffpp":
        record["compression"] = args.compression
    if val_info is not None:
        record["val_threshold_info"] = val_info

    # 控制台输出
    print_results_table(record)

    # 保存结果
    save_results(record, preds, labels, video_ids, args.output_dir)
    log(f"Evaluation finished. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
