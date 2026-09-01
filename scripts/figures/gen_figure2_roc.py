#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 2: Multi-model ROC Curves on Celeb-DF++ Test Set.

数据来源 (各 npz 含 preds / labels 两个数组, preds 为 fake 类概率):
  - CAR (joint):  results/table1/config_0_Celeb-DF____test_/raw_predictions.npz
                  (联合训练主结果, 由 evaluate_v2.py 生成)
  - CAR (v5):     results/table2/car_v5/raw_predictions.npz
                  (CAR v5 单独训练, 由 run_table2_baseline.py 生成)
  - 4 个 baseline: results/table2/<model>/raw_predictions.npz
                  <model> in {mesonet, efficientnet_b0, efficientnet_b3, xception}
                  (由 run_table2_baseline.py 生成)

绘图:
  - sklearn.metrics.roc_curve 计算 FPR / TPR
  - sklearn.metrics.auc 计算 AUC
  - 所有模型 ROC 曲线叠加在同一图上
  - 图例标注 "Model: AUC=x.xxx"
  - 对角线参考线 (random classifier)
  - X 轴: False Positive Rate, Y 轴: True Positive Rate
  - 标题: ROC Curves on Celeb-DF++ Test Set

输出:
  - results/figures/figure2_roc.pdf  (矢量格式)
  - results/figures/figure2_roc.png  (300 dpi 预览)

严谨性:
  - 若某个 npz 文件不存在, 打印警告并跳过 (不中断其余模型绘图)
"""

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 避免显示窗口, 支持无界面环境
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# (display_name, npz_path) — 顺序即图例顺序
# CAR 变体放最前以突出主模型; baseline 随后
MODELS = [
    (
        "CAR (joint)",
        os.path.join(
            PROJECT_ROOT,
            "results",
            "table1",
            "config_0_Celeb-DF____test_",
            "raw_predictions.npz",
        ),
    ),
    (
        "CAR (v5)",
        os.path.join(PROJECT_ROOT, "results", "table2", "car_v5", "raw_predictions.npz"),
    ),
    (
        "MesoNet",
        os.path.join(
            PROJECT_ROOT, "results", "table2", "mesonet", "raw_predictions.npz"
        ),
    ),
    (
        "EfficientNet-B0",
        os.path.join(
            PROJECT_ROOT, "results", "table2", "efficientnet_b0", "raw_predictions.npz"
        ),
    ),
    (
        "EfficientNet-B3",
        os.path.join(
            PROJECT_ROOT, "results", "table2", "efficientnet_b3", "raw_predictions.npz"
        ),
    ),
    (
        "Xception",
        os.path.join(
            PROJECT_ROOT, "results", "table2", "xception", "raw_predictions.npz"
        ),
    ),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.get_cmap("tab10").colors
    # CAR 主模型用稍粗线宽以突出; 其余模型统一较细
    car_names = {"CAR (joint)", "CAR (v5)"}

    plotted = 0
    print("加载各模型 raw_predictions.npz 并计算 ROC...")
    for i, (name, path) in enumerate(MODELS):
        if not os.path.exists(path):
            print(f"  [WARN] 缺失 {path}, 跳过 {name}")
            continue

        data = np.load(path)
        preds = data["preds"].flatten().astype(np.float64)
        labels = data["labels"].flatten().astype(int)

        # 跳过无效样本 (NaN) 或单一类别 (无法计算 ROC)
        valid_mask = ~(np.isnan(preds) | np.isnan(labels))
        preds = preds[valid_mask]
        labels = labels[valid_mask]
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            print(f"  [WARN] {name} 样本无效或单一类别, 跳过")
            continue

        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = auc(fpr, tpr)

        lw = 2.4 if name in car_names else 1.6
        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            linewidth=lw,
            label=f"{name}: AUC={auc_val:.3f}",
        )
        print(f"  {name:<18} AUC={auc_val:.4f}  (N={len(labels)})")
        plotted += 1

    # 对角线参考线 (random classifier)
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="Random (AUC=0.500)",
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves on Celeb-DF++ Test Set", fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="lower right", frameon=True)

    plt.tight_layout()

    if plotted == 0:
        print("\n[WARN] 没有任何模型数据可绘制, 跳过保存. 请先运行 run_table2_baseline.py.")
        plt.close(fig)
        return

    pdf_path = os.path.join(OUTPUT_DIR, "figure2_roc.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figure2_roc.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"[OK] 已保存 PDF: {pdf_path}")
    print(f"[OK] 已保存 PNG: {png_path}")
    print(f"共绘制 {plotted} 条 ROC 曲线")


if __name__ == "__main__":
    main()
