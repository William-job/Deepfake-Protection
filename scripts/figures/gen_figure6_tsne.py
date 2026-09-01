#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 6: t-SNE Visualization of CAR Latent Space (Celeb-DF++ Test).

数据来源:
  - Checkpoint: checkpoints_v6/best_model_joint_celebdf.pt (epoch 8, AUC 0.8557)
  - 配置文件:  configs/joint_train.yaml
  - 数据集:    Celeb-DF++ test split, 随机子集 ~1000 样本 (seed=42)

功能:
  加载 CAR 模型在 Celeb-DF++ test 子集上推理, 提取 outputs["z"]
  (形状 [B, latent_k], latent_k=4, fusion 输出的潜在特征), 用
  sklearn.manifold.TSNE 降维到 2D, 按 real (label=0, 蓝色) /
  fake (label=1, 红色) 着色绘制散点图, 直观展示 CAR 潜在空间对
  real/fake 的可分性。

t-SNE 参数:
  - perplexity=30 (若样本数 < 1000, 用 sqrt(N) 以满足 sklearn 约束
    perplexity < N)
  - random_state=42
  - max_iter=1000

输出:
  - results/figures/figure6_tsne.pdf  (矢量)
  - results/figures/figure6_tsne.png  (300 dpi)

用法:
  python scripts/figures/gen_figure6_tsne.py
  python scripts/figures/gen_figure6_tsne.py --batch_size 2   # OOM 时降级
  python scripts/figures/gen_figure6_tsne.py --num_samples 500  # 调整子集大小
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
from torch.utils.data import DataLoader, Subset

import matplotlib

matplotlib.use("Agg")  # 避免显示窗口
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.config import load_config
from src.data.dataloader import create_dataloader
from src.models.car import CAR

# ----------------------------------------------------------------------------
# 固定路径与常量
# ----------------------------------------------------------------------------
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "joint_train.yaml")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints_v6", "best_model_joint_celebdf.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

DEFAULT_NUM_SAMPLES = 1000
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Figure 6: t-SNE Visualization of CAR Latent Space"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="覆盖 config.data.batch_size; 默认用配置值(8), OOM 时降到 2",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"子集样本数 (默认 {DEFAULT_NUM_SAMPLES})",
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


def build_subset_loader(config, num_samples, batch_size):
    """构建 Celeb-DF++ test 子集 DataLoader (随机选 num_samples 个样本, seed=42)."""
    # 先用 create_dataloader 拿到底层 dataset (split=test, shuffle=False, 无增强)
    config._data["data"]["num_workers"] = 0
    config._data["data"]["batch_size"] = batch_size
    full_loader = create_dataloader(config, split="test")
    dataset = full_loader.dataset

    n_total = len(dataset)
    n_select = min(num_samples, n_total)
    print(f"[INFO] Total test samples: {n_total}, selecting {n_select} (seed={SEED})")

    rng = np.random.RandomState(SEED)
    indices = rng.choice(n_total, size=n_select, replace=False).tolist()
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader, n_select


@torch.no_grad()
def collect_latent_z(model, loader, device):
    """在 loader 上推理, 收集 z 与 label.

    返回:
        z:       np.ndarray (N, latent_k) 潜在特征
        labels:  np.ndarray (N,) int
    """
    model.eval()
    all_z, all_labels = [], []

    for batch in tqdm(loader, desc="Latent z inference", leave=False):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        outputs = model(frames)
        z = outputs["z"]  # [B, latent_k]
        if z.dim() == 1:
            z = z.unsqueeze(0)

        all_z.append(z.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    z_arr = np.concatenate(all_z, axis=0).astype(np.float64)
    labels = np.concatenate(all_labels).astype(int)
    return z_arr, labels


def run_tsne(z, n_samples):
    """用 t-SNE 降维到 2D.

    perplexity=30, 若 n_samples < 1000 则用 sqrt(n_samples) (sklearn 要求 perplexity < N).
    """
    # perplexity 必须严格小于样本数
    if n_samples < 1000:
        perplexity = float(np.sqrt(n_samples))
    else:
        perplexity = 30.0
    # 安全边界: perplexity 至少为 5, 且必须 < n_samples
    perplexity = max(5.0, min(perplexity, n_samples - 1))
    print(f"[INFO] t-SNE: n_samples={n_samples}, perplexity={perplexity:.2f}, "
          f"max_iter=1000, random_state={SEED}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=SEED,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    z_2d = tsne.fit_transform(z)
    return z_2d


def plot_tsne(z_2d, labels, output_dir):
    """绘制 real/fake 散点图."""
    fig, ax = plt.subplots(figsize=(8, 7))

    real_mask = labels == 0
    fake_mask = labels == 1

    ax.scatter(
        z_2d[real_mask, 0],
        z_2d[real_mask, 1],
        c="blue",
        alpha=0.6,
        s=15,
        label=f"Real (n={int(real_mask.sum())})",
        edgecolors="none",
    )
    ax.scatter(
        z_2d[fake_mask, 0],
        z_2d[fake_mask, 1],
        c="red",
        alpha=0.6,
        s=15,
        label=f"Fake (n={int(fake_mask.sum())})",
        edgecolors="none",
    )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("t-SNE Visualization of CAR Latent Space (Celeb-DF++ Test)", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=11, loc="best", frameon=True, markerscale=2)

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, "figure6_tsne.pdf")
    png_path = os.path.join(output_dir, "figure6_tsne.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 已保存 PDF: {pdf_path}")
    print(f"[OK] 已保存 PNG: {png_path}")
    print(f"[STATS] real={int(real_mask.sum())}, fake={int(fake_mask.sum())}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载配置
    print(f"[INFO] Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)

    # 禁用 CutMix 等训练增强
    disable_train_augmentation(config)

    # 覆盖 batch_size
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = config.data.batch_size
    print(f"[INFO] Using batch_size={batch_size}")

    # 加载模型
    print(f"[INFO] Loading model from {CKPT_PATH}")
    model, ckpt_epoch = load_model(config, CKPT_PATH, device)
    print(f"[INFO] Checkpoint epoch: {ckpt_epoch}")

    # 构建 Celeb-DF++ test 子集 loader
    print("[INFO] Building Celeb-DF++ test subset loader...")
    loader, n_select = build_subset_loader(config, args.num_samples, batch_size)

    # 推理收集 z
    print("[INFO] Running inference on test subset...")
    z, labels = collect_latent_z(model, loader, device)
    print(f"[INFO] Collected {len(z)} samples, z shape={z.shape}")

    # t-SNE 降维
    print("[INFO] Running t-SNE...")
    z_2d = run_tsne(z, len(z))

    # 绘图
    plot_tsne(z_2d, labels, OUTPUT_DIR)


if __name__ == "__main__":
    main()
