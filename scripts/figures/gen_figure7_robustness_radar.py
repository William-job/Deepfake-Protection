#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 7: Robustness Evaluation Radar Chart.

数据来源: results/table5/robustness_results.json
        (Table 5: Robustness Evaluation, Celeb-DF test split,
         checkpoint checkpoints_v6/best_model_joint_celebdf.pt)

生成 4 维雷达图 (方案 A): 展示模型在四种扰动类型
(JPEG / Noise / Blur / Brightness) 下的"最差情况"AUC
(即每种类型中 AUC 最低的那个级别), 并以 baseline (clean) AUC
绘制参考圆, 直观显示模型对各扰动的脆弱程度。

H.264 因 skipped 不参与绘图。

输出:
  - results/figures/figure7_robustness_radar.pdf  (矢量)
  - results/figures/figure7_robustness_radar.png  (300 dpi)
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_PATH = os.path.join(
    PROJECT_ROOT, "results", "table5", "robustness_results.json"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# 纳入绘图的扰动类型及其显示名 (顺序即雷达图顺时针/逆时针顺序)
TYPE_DISPLAY = {
    "jpeg": "JPEG",
    "noise": "Noise",
    "blur": "Blur",
    "brightness": "Brightness",
}
ORDERED_TYPES = ["jpeg", "noise", "blur", "brightness"]


def level_label(ptype, level):
    """根据扰动类型与级别生成简短的轴标签描述。"""
    if ptype == "jpeg":
        return f"Q{int(level)}"
    if ptype == "noise":
        return f"σ={level}"
    if ptype == "blur":
        return f"k={int(level)}"
    if ptype == "brightness":
        return f"{level}"
    return str(level)


def main():
    # ---- 读取数据 ----------------------------------------------------------
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    baseline_auc = float(data["baseline"]["auc"])
    perturbations = data["perturbations"]

    # ---- 按类型分组 (跳过 skipped / h264 / auc 为 None) --------------------
    grouped = {}
    for p in perturbations:
        if p.get("status") != "ok":
            continue
        ptype = p.get("type")
        auc = p.get("auc")
        if ptype not in TYPE_DISPLAY or auc is None:
            continue
        grouped.setdefault(ptype, []).append(p)

    # ---- 每个类型取最差 (最低) AUC ----------------------------------------
    worst_entries = []
    for ptype in ORDERED_TYPES:
        items = grouped.get(ptype, [])
        if not items:
            print(f"[WARN] 扰动类型 {ptype} 无有效数据, 已跳过")
            continue
        worst = min(items, key=lambda x: x["auc"])
        worst_entries.append((ptype, worst))

    categories = [
        f"{TYPE_DISPLAY[pt]}\n({level_label(pt, w['level'])})"
        for pt, w in worst_entries
    ]
    worst_aucs = [float(w["auc"]) for _, w in worst_entries]

    # ---- 雷达图构造 --------------------------------------------------------
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles_closed = angles + angles[:1]
    worst_aucs_closed = worst_aucs + worst_aucs[:1]
    baseline_circle = [baseline_auc] * (N + 1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 最差 AUC 多边形
    worst_color = "#C44E52"
    ax.plot(
        angles_closed,
        worst_aucs_closed,
        "o-",
        linewidth=2.0,
        color=worst_color,
        markersize=7,
        label="Worst-case AUC",
    )
    ax.fill(angles_closed, worst_aucs_closed, alpha=0.25, color=worst_color)

    # Baseline 参考圆
    ax.plot(
        angles_closed,
        baseline_circle,
        "--",
        linewidth=1.5,
        color="gray",
        label=f"Baseline (AUC={baseline_auc:.4f})",
    )

    # 轴标签
    ax.set_xticks(angles_closed[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    # 半径范围与刻度
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="gray")

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.spines["polar"].set_alpha(0.3)

    # 在每个数据点旁标注 AUC 数值 (向圆心方向偏移)
    for ang, val in zip(angles, worst_aucs):
        ax.annotate(
            f"{val:.3f}",
            xy=(ang, val),
            xytext=(ang, max(val - 0.08, 0.05)),
            fontsize=10,
            ha="center",
            va="center",
            color="#333333",
            fontweight="bold",
        )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.18, 1.12),
        fontsize=11,
        frameon=True,
    )

    plt.title(
        "Robustness Evaluation (Worst-case AUC per Perturbation Type)",
        fontsize=14,
        pad=24,
        fontweight="bold",
    )

    plt.tight_layout()

    # ---- 保存 -------------------------------------------------------------
    pdf_path = os.path.join(OUTPUT_DIR, "figure7_robustness_radar.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figure7_robustness_radar.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 控制台报告 -------------------------------------------------------
    print(f"[OK] 已保存 PDF: {pdf_path}")
    print(f"[OK] 已保存 PNG: {png_path}")
    print()
    print(f"Baseline AUC (clean): {baseline_auc:.4f}")
    print()
    print("各扰动类型最差情况 AUC:")
    for (ptype, w), cat in zip(worst_entries, categories):
        cat_clean = cat.replace("\n", " ")
        print(
            f"  {cat_clean:<22} AUC={w['auc']:.4f}  "
            f"(name={w['name']}, delta_auc={w.get('delta_auc', float('nan')):+.4f})"
        )


if __name__ == "__main__":
    main()
