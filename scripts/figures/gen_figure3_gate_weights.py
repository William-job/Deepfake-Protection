#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 3: Gate Weight Evolution During Joint Training.

数据来源: checkpoints_v6/checkpoint_joint_epoch_{0..14}.pt
        (联合训练 15 个 epoch 的 checkpoint, 每个包含 model_state_dict 与 ema_shadow)
        门控参数 (GatingNetwork, 见 src/models/gating.py):
          - gating.weight                 (4, latent_k)   主权重矩阵
          - gating.bias                   (4,)            偏置
          - gating.beta                   ()              logits 缩放
          - gating.difficulty_proj.weight (4, 1)          difficulty 条件偏置
          - gating.difficulty_proj.bias   (4,)

说明:
  门控输出 w = softmax((z W^T + b + diff_proj(d)) * beta / T) 依赖输入 z 与
  difficulty, 为动态量。直接静态提取参数无法得到 w 本身。本脚本采用 Monte Carlo
  采样估计每个专家的 *期望 dense 门控权重* E[w] 作为该 epoch 的代表性门控权重:
    - z ~ N(0, I_k), k = latent_k = 4 (fusion 输出, 经 LayerNorm 后近似归一化)
    - difficulty ~ Uniform(0, 1)
    - T = T_min + (T_max - T_min) * d   (difficulty-temperature, config: 0.2~2.0)
    - 对 N 个采样求 E[w] (dense softmax, 未做 top-k 稀疏化)
  使用同一组随机样本跨 epoch 比较, 相对趋势可复现且与 z 的真实分布无关。

输出:
  - results/figures/figure3_gate_weights.pdf  (矢量)
  - results/figures/figure3_gate_weights.png  (300 dpi)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_v6")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

NUM_EPOCHS = 15  # epoch 0..14
EXPERT_NAMES = ["temporal", "flow", "frequency", "blending"]

# difficulty-conditioned temperature 范围 (configs/joint_train.yaml)
#   difficulty_temperature_min: 0.2, difficulty_temperature_max: 2.0
T_MIN = 0.2
T_MAX = 2.0

# Monte Carlo 采样数 (固定 seed 保证可复现)
MC_SAMPLES = 20000
MC_SEED = 42
LATENT_K = 4  # model.latent_k


def load_gating_params(ckpt_path):
    """从 checkpoint 提取门控参数 (优先 model_state_dict, 缺失键给 0)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    weight = sd["gating.weight"].float()                  # (4, k)
    bias = sd["gating.bias"].float()                      # (4,)
    beta = sd["gating.beta"].float().reshape(())          # ()

    # difficulty_proj 在 gating_difficulty_conditioned=true 时存在
    if "gating.difficulty_proj.weight" in sd:
        dp_weight = sd["gating.difficulty_proj.weight"].float()  # (4, 1)
        dp_bias = sd["gating.difficulty_proj.bias"].float()      # (4,)
    else:
        dp_weight = torch.zeros(4, 1)
        dp_bias = torch.zeros(4)

    return weight, bias, beta, dp_weight, dp_bias


def expected_gate_weights(weight, bias, beta, dp_weight, dp_bias, n=MC_SAMPLES, seed=MC_SEED):
    """Monte Carlo 估计每个专家的期望 dense 门控权重 E[w]."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, weight.shape[1], generator=gen)        # (n, k)
    diff = torch.rand(n, 1, generator=gen)                    # (n, 1) in [0,1)

    T = T_MIN + (T_MAX - T_MIN) * diff                        # (n, 1)
    dp = diff @ dp_weight.T + dp_bias                         # (n, 4)
    logits = (z @ weight.T + bias + dp) * beta                # (n, 4)
    w = F.softmax(logits / T, dim=-1)                         # (n, 4)
    return w.mean(dim=0).numpy()                              # (4,)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    epochs = list(range(NUM_EPOCHS))
    curves = {name: [] for name in EXPERT_NAMES}

    print(f"加载 {NUM_EPOCHS} 个联合训练 checkpoint (epoch 0..14)")
    for epoch in epochs:
        ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_joint_epoch_{epoch}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] 缺失 {ckpt_path}, 该 epoch 用 NaN 填充")
            for name in EXPERT_NAMES:
                curves[name].append(np.nan)
            continue

        weight, bias, beta, dp_weight, dp_bias = load_gating_params(ckpt_path)
        ew = expected_gate_weights(weight, bias, beta, dp_weight, dp_bias)
        for i, name in enumerate(EXPERT_NAMES):
            curves[name].append(float(ew[i]))

        # 控制台逐 epoch 报告
        parts = "  ".join(f"{n}={v:.4f}" for n, v in zip(EXPERT_NAMES, ew))
        print(f"  epoch {epoch:2d}: {parts}  (beta={float(beta):.4f})")

    # ------------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------------
    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(8, 5.2))

    for i, name in enumerate(EXPERT_NAMES):
        ax.plot(
            epochs,
            curves[name],
            marker=markers[i],
            color=colors[i],
            linewidth=1.8,
            markersize=6,
            label=name.capitalize(),
        )

    # y 轴范围 (留出顶部空间给注释)
    y_min = min(np.nanmin(v) for v in curves.values())
    y_max = max(np.nanmax(v) for v in curves.values())
    pad = (y_max - y_min) * 0.18
    y_lo = max(0.0, y_min - pad)
    y_hi = y_max + pad * 1.6
    ax.set_ylim(y_lo, y_hi)

    # 课程切换点 (top_k 1 -> 2) 竖虚线
    switch_epoch = 4
    ax.axvline(
        switch_epoch,
        color="gray",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
    )
    ax.text(
        switch_epoch + 0.15,
        y_hi - pad * 0.1,
        "Curriculum switch\n(top_k 1→2)",
        fontsize=9,
        color="gray",
        va="top",
        ha="left",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gate Weight", fontsize=12)
    ax.set_title("Gate Weight Evolution During Joint Training", fontsize=14)
    ax.set_xticks(epochs)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="best", frameon=True)

    plt.tight_layout()

    pdf_path = os.path.join(OUTPUT_DIR, "figure3_gate_weights.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figure3_gate_weights.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"[OK] 已保存 PDF: {pdf_path}")
    print(f"[OK] 已保存 PNG: {png_path}")

    # ------------------------------------------------------------------------
    # 趋势报告
    # ------------------------------------------------------------------------
    print()
    print("门控权重趋势 (期望 dense 权重 E[w]):")
    for name in EXPERT_NAMES:
        vals = curves[name]
        start = vals[0]
        end = vals[-1]
        peak_epoch = int(np.nanargmax(vals))
        peak = vals[peak_epoch]
        print(
            f"  {name:<10} start={start:.4f}  end={end:.4f}  "
            f"peak={peak:.4f}@epoch{peak_epoch}  delta={end-start:+.4f}"
        )

    # 各 epoch 权重最高/最低专家
    print()
    print("各 epoch 权重最高 / 最低专家:")
    for epoch in epochs:
        vals = [curves[n][epoch] for n in EXPERT_NAMES]
        hi = EXPERT_NAMES[int(np.nanargmax(vals))]
        lo = EXPERT_NAMES[int(np.nanargmin(vals))]
        print(f"  epoch {epoch:2d}: 最高={hi}({max(vals):.4f})  最低={lo}({min(vals):.4f})")


if __name__ == "__main__":
    main()
