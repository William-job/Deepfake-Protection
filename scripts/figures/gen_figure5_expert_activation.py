#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 5: Per-Method Expert Activation on FF++ c23 Test.

数据来源:
  - Checkpoint: checkpoints_v6/best_model_joint_ffpp.pt (epoch 12, AUC 0.9143)
  - 配置文件:  configs/joint_train.yaml
  - 数据集:    FF++ c23 test split
  - 数据加载:  src.data.ff_frame_dataset.FFFrameDataset + 自定义 collate
               (样本含 "method" 字段: Deepfakes/Face2Face/FaceSwap/NeuralTextures)

功能:
  加载 CAR 模型在 FF++ c23 test 集上推理, 提取 outputs["w_dense"]
  (形状 [B, 4], top_k 稀疏化前的密集门控权重), 按 4 种篡改方法分组
  计算每组平均 w_dense, 绘制 4 行 × 4 列热力图 (行=篡改方法, 列=专家),
  直观展示不同篡改类型对各专家的激活偏好。

输出:
  - results/figures/figure5_expert_activation.pdf  (矢量)
  - results/figures/figure5_expert_activation.png  (300 dpi)

用法:
  python scripts/figures/gen_figure5_expert_activation.py
  python scripts/figures/gen_figure5_expert_activation.py --batch_size 2   # OOM 时降级
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
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")  # 避免显示窗口
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import load_config
from src.data.ff_frame_dataset import FFFrameDataset
from src.models.car import CAR

# ----------------------------------------------------------------------------
# 固定路径与常量
# ----------------------------------------------------------------------------
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "joint_train.yaml")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints_v6", "best_model_joint_ffpp.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

EXPERT_NAMES = ["temporal", "flow", "frequency", "blending"]
METHOD_ORDER = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
COMPRESSION = "c23"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Figure 5: Per-Method Expert Activation on FF++ c23 Test"
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


class FFEvalCollate:
    """FF++ 评估专用 collate: 堆叠帧/标签, 保留 frame_dir 与 method 字段."""

    def __call__(self, batch):
        return {
            "frames": torch.stack([b["frames"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "frame_dir": [b["frame_dir"] for b in batch],
            "method": [b.get("method") for b in batch],
        }


def build_ffpp_loader(config, batch_size):
    """构建 FF++ c23 test DataLoader (参考 evaluate_v2.py 的 ffpp 分支)."""
    ds = FFFrameDataset(
        data_root=config.data.ff_root,
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
        split="test",
        compression=COMPRESSION,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 评估时单进程, 避免 Windows 序列化问题
        pin_memory=True,
        drop_last=False,
        collate_fn=FFEvalCollate(),
    )
    return loader


@torch.no_grad()
def collect_w_dense(model, loader, device):
    """在 loader 上推理, 收集 w_dense 与 method.

    返回:
        w_dense:  np.ndarray (N, 4) 密集门控权重
        methods:  list[str] 长度 N, 每个样本的篡改方法名
    """
    model.eval()
    all_w, all_methods = [], []

    for batch in tqdm(loader, desc="w_dense inference", leave=False):
        frames = batch["frames"].to(device)
        outputs = model(frames)
        w_dense = outputs["w_dense"]  # [B, 4]
        if w_dense is None:
            raise RuntimeError(
                "outputs['w_dense'] 为 None, 请检查 GatingNetwork 是否返回 w_dense"
            )
        if w_dense.dim() == 1:
            w_dense = w_dense.unsqueeze(0)
        all_w.append(w_dense.cpu().numpy())
        all_methods.extend(batch["method"])

    w_dense = np.concatenate(all_w, axis=0).astype(np.float64)
    return w_dense, all_methods


def compute_per_method_mean(w_dense, methods):
    """按 4 种篡改方法分组计算平均 w_dense.

    返回:
        matrix: np.ndarray (4, 4) 行=方法, 列=专家
        counts: dict[method] -> int
    """
    matrix = np.zeros((len(METHOD_ORDER), len(EXPERT_NAMES)), dtype=np.float64)
    counts = {m: 0 for m in METHOD_ORDER}

    w_arr = np.asarray(w_dense)
    for i, method in enumerate(methods):
        if method not in METHOD_ORDER:
            continue
        row = METHOD_ORDER.index(method)
        matrix[row] += w_arr[i]
        counts[method] += 1

    for row, method in enumerate(METHOD_ORDER):
        if counts[method] > 0:
            matrix[row] /= counts[method]
        else:
            print(f"[WARN] 方法 {method} 无样本, 该行用 0 填充")
    return matrix, counts


def plot_heatmap(matrix, counts, output_dir):
    """绘制 4×4 热力图 (行=篡改方法, 列=专家)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    # 坐标轴
    ax.set_xticks(range(len(EXPERT_NAMES)))
    ax.set_xticklabels([n.capitalize() for n in EXPERT_NAMES], fontsize=11)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels(
        [f"{m}\n(n={counts[m]})" for m in METHOD_ORDER], fontsize=10
    )

    # 单元格数值标注
    vmax = matrix.max() if matrix.max() > 0 else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            # 数值颜色: 深色背景用白字, 浅色背景用黑字
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

    ax.set_title("Per-Method Expert Activation on FF++ c23 Test", fontsize=13)
    ax.set_xlabel("Expert", fontsize=12)
    ax.set_ylabel("Manipulation Method", fontsize=12)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Activation Weight", fontsize=10)

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, "figure5_expert_activation.pdf")
    png_path = os.path.join(output_dir, "figure5_expert_activation.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 已保存 PDF: {pdf_path}")
    print(f"[OK] 已保存 PNG: {png_path}")
    print("[STATS] 各方法样本数:")
    for m in METHOD_ORDER:
        print(f"  {m}: {counts[m]}")
    print("[STATS] 平均激活权重矩阵 (行=方法, 列=专家):")
    header = "  " + "  ".join(f"{n:>12}" for n in EXPERT_NAMES)
    print(header)
    for i, m in enumerate(METHOD_ORDER):
        row_str = "  " + "  ".join(f"{v:>12.4f}" for v in matrix[i])
        print(f"{m:<16}{row_str}")


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

    # 构建 FF++ c23 test loader
    print(f"[INFO] Building FF++ {COMPRESSION} test loader...")
    loader = build_ffpp_loader(config, batch_size)

    # 推理收集 w_dense
    print("[INFO] Running inference on FF++ c23 test...")
    w_dense, methods = collect_w_dense(model, loader, device)
    print(f"[INFO] Collected {len(w_dense)} samples")

    # 按方法分组计算平均
    matrix, counts = compute_per_method_mean(w_dense, methods)

    # 绘图
    plot_heatmap(matrix, counts, OUTPUT_DIR)


if __name__ == "__main__":
    main()
