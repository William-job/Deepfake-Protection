#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 4: Difficulty Score Distribution on Celeb-DF++ Val.

数据来源:
  - Checkpoint: checkpoints_v6/best_model_joint_celebdf.pt (epoch 8, AUC 0.8557)
  - 配置文件:  configs/joint_train.yaml
  - 数据集:    Celeb-DF++ val split (避免使用 test 集, 防止数据泄露)

功能:
  加载 CAR 模型在 Celeb-DF++ val 集上推理, 提取 outputs["difficulty"]
  (形状 [B], 由 DifficultyEstimator 输出, 衡量样本被正确分类的难度),
  按 real (label=0) / fake (label=1) 分组绘制叠加直方图, 并用全样本
  的 33% / 67% 分位数标注 easy / mid / hard 三档分界。

输出:
  - results/figures/figure4_difficulty.pdf  (矢量)
  - results/figures/figure4_difficulty.png  (300 dpi)

用法:
  python scripts/figures/gen_figure4_difficulty.py
  python scripts/figures/gen_figure4_difficulty.py --batch_size 2   # OOM 时降级
"""

import argparse
import os
import sys

# ----------------------------------------------------------------------------
# 路径: scripts/figures/ -> scripts/ -> project_root
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # 避免显示窗口
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import load_config
from src.data.dataloader import create_dataloader
from src.models.car import CAR

# ----------------------------------------------------------------------------
# 固定路径
# ----------------------------------------------------------------------------
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "joint_train.yaml")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints_v6", "best_model_joint_celebdf.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Figure 4: Difficulty Score Distribution on Celeb-DF++ Val"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="覆盖 config.data.batch_size; 默认用配置值(8), OOM 时降到 2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu), cuda 不可用时自动回退 cpu",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[INFO] CUDA 不可用, 回退到 CPU")
        return "cpu"
    return device_arg


def load_model(config, checkpoint_path, device):
    """加载 CAR 模型与 checkpoint, 优先使用 EMA 权重 (参考 evaluate_v2.py)."""
    model = CAR(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and ckpt.get("ema_shadow") is not None:
        try:
            model.load_state_dict(ckpt["ema_shadow"], strict=True)
            print("[INFO] Using EMA weights (strict load)")
        except RuntimeError:
            model.load_state_dict(ckpt["ema_shadow"], strict=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("[INFO] Loaded: EMA shadow (non-strict) + model_state_dict 补齐")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[INFO] Using model_state_dict (no EMA shadow)")

    model.eval()
    ckpt_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, ckpt_epoch


def disable_train_augmentation(config):
    """评估时禁用 CutMix 等随机增强 (参考 evaluate_v2.py)."""
    if "training" in config._data and isinstance(config._data["training"], dict):
        config._data["training"]["cutmix_p"] = 0.0


@torch.no_grad()
def collect_difficulty(model, loader, device):
    """在 loader 上推理, 收集 difficulty 与 label.

    返回:
        difficulties: np.ndarray (N,)
        labels:       np.ndarray (N,) int
    """
    model.eval()
    all_diff, all_labels = [], []

    for batch in tqdm(loader, desc="Difficulty inference", leave=False):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        outputs = model(frames)
        diff = outputs["difficulty"]  # [B]
        if diff.dim() > 1:
            diff = diff.view(-1)

        all_diff.append(diff.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    difficulties = np.concatenate(all_diff).astype(np.float64)
    labels = np.concatenate(all_labels).astype(int)
    return difficulties, labels


def plot_difficulty_distribution(difficulties, labels, output_dir):
    """绘制 real/fake 叠加直方图, 标注 easy/mid/hard 分界."""
    real_diff = difficulties[labels == 0]
    fake_diff = difficulties[labels == 1]

    # 全样本 33% / 67% 分位数作为三档分界
    q33 = float(np.percentile(difficulties, 33))
    q67 = float(np.percentile(difficulties, 67))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bins = 40
    ax.hist(
        real_diff,
        bins=bins,
        alpha=0.6,
        label=f"Real (n={len(real_diff)})",
        color="green",
        density=True,
    )
    ax.hist(
        fake_diff,
        bins=bins,
        alpha=0.6,
        label=f"Fake (n={len(fake_diff)})",
        color="red",
        density=True,
    )

    # easy / mid / hard 分界竖线
    y_lo, y_hi = ax.get_ylim()
    ax.axvline(q33, color="gray", linestyle="--", linewidth=1.3, alpha=0.8)
    ax.axvline(q67, color="gray", linestyle="--", linewidth=1.3, alpha=0.8)
    # 分界标注 (避开顶部图例区)
    annotate_y = y_hi * 0.92
    ax.text(q33, annotate_y, " 33%", fontsize=9, color="gray", va="top", ha="left")
    ax.text(q67, annotate_y, " 67%", fontsize=9, color="gray", va="top", ha="left")
    # 三档区间标签
    mid_x = (q33 + q67) / 2.0
    ax.text(
        q33 / 2.0,
        y_hi * 0.82,
        "easy",
        fontsize=10,
        color="dimgray",
        ha="center",
        va="top",
    )
    ax.text(
        mid_x,
        y_hi * 0.82,
        "mid",
        fontsize=10,
        color="dimgray",
        ha="center",
        va="top",
    )
    ax.text(
        (q67 + difficulties.max()) / 2.0,
        y_hi * 0.82,
        "hard",
        fontsize=10,
        color="dimgray",
        ha="center",
        va="top",
    )

    ax.set_xlabel("Difficulty Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Difficulty Score Distribution on Celeb-DF++ Val", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right", frameon=True)

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, "figure4_difficulty.pdf")
    png_path = os.path.join(output_dir, "figure4_difficulty.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 已保存 PDF: {pdf_path}")
    print(f"[OK] 已保存 PNG: {png_path}")
    print(
        f"[STATS] real: mean={real_diff.mean():.4f} std={real_diff.std():.4f}  "
        f"fake: mean={fake_diff.mean():.4f} std={fake_diff.std():.4f}"
    )
    print(f"[STATS] 分界: q33={q33:.4f}  q67={q67:.4f}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载配置
    print(f"[INFO] Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)

    # 禁用 CutMix 等训练增强
    disable_train_augmentation(config)
    # 评估时单进程, 避免 Windows 序列化问题
    config._data["data"]["num_workers"] = 0

    # 覆盖 batch_size
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = config.data.batch_size
    config._data["data"]["batch_size"] = batch_size
    print(f"[INFO] Using batch_size={batch_size}")

    # 加载模型
    print(f"[INFO] Loading model from {CKPT_PATH}")
    model, ckpt_epoch = load_model(config, CKPT_PATH, device)
    print(f"[INFO] Checkpoint epoch: {ckpt_epoch}")

    # 构建 Celeb-DF++ val loader
    print("[INFO] Building Celeb-DF++ val loader...")
    loader = create_dataloader(config, split="val")

    # 推理收集 difficulty
    print("[INFO] Running inference on val split...")
    difficulties, labels = collect_difficulty(model, loader, device)
    print(f"[INFO] Collected {len(difficulties)} samples "
          f"(real={int(np.sum(labels == 0))}, fake={int(np.sum(labels == 1))})")

    # 绘图
    plot_difficulty_distribution(difficulties, labels, OUTPUT_DIR)


if __name__ == "__main__":
    main()
