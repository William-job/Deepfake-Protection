#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文图 v2（审稿人审查修正版）

相对 v1 的修改（依据 scipilot-figure-skill 审查）：
  fig2: RdYlGn（红绿，色盲不友好）→ viridis（感知均匀、色盲安全、灰度单调）
        注记颜色按色图亮度自适应（<75 白字 / ≥75 黑字）
  fig3: 补 (a)/(b) 面板标签（多面板编号要求）
  fig4: 由 collapse_analysis 的数据重绘，补 (a)–(d) 标签与 ylabel "density"
  全部: 增加 PDF 矢量导出 + 灰度预览（色盲/黑白打印自检）
  全部: 经 visual_qa.audit_layout 程序自检（缺字/裁切/刻度重叠）

数据源与 v1 完全相同（冻结 JSON），不引入新计算。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_QA = r"c:\Users\86188\.trae-cn\skills\scipilot-figure-skill\scripts"
if os.path.isdir(SKILL_QA):
    sys.path.insert(0, SKILL_QA)
try:
    from visual_qa import audit_layout, print_report
    HAS_QA = True
except Exception as e:
    print(f"[warn] visual_qa 不可用: {e}")
    HAS_QA = False

OUT = os.path.join(PROJECT_ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "font.family": "DejaVu Sans", "savefig.dpi": 300,
})

OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


R = lambda *p: os.path.join(PROJECT_ROOT, "results", *p)


def export_all(fig, name):
    """PDF（矢量，投稿）+ PNG（300dpi）+ 灰度预览（色盲/黑白自检）。"""
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight")
    # 灰度版
    img = Image.open(os.path.join(OUT, f"{name}.png")).convert("L")
    img.save(os.path.join(OUT, f"{name}_grayscale.png"))


def qa(fig, name):
    if not HAS_QA:
        return
    issues = audit_layout(fig)
    print(f"--- QA {name} ---")
    print_report(issues)


# ===========================================================================
# Fig 1 — 架构示意图（v1 审查通过：字号≥7.5pt、框宽核算无裁切风险；保留）
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")


def box(x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=1.2, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=10,
                                 lw=lw, color="#555555"))


box(0.2, 2.9, 1.7, 1.2, "8-frame\nclip\n224×224", fc="#F0F0F0")
box(2.4, 2.9, 1.9, 1.2, "Shared stem\nEfficientNet-B0\n(3.60M, frozen)", fc="#E7F0FA")
heads = [(5.0, 5.3, "temporal head"), (5.0, 3.9, "motion head"),
         (5.0, 2.5, "spectral head"), (5.0, 1.1, "boundary head")]
for (x, y, t) in heads:
    box(x, y, 1.9, 0.95, t, fc="#FFF7E6", fs=8)
    arrow(4.3, 3.5, 5.0, y + 0.48)
box(7.4, 2.9, 1.9, 1.2, "Fusion\n2-layer transformer\n+ difficulty d", fc="#EAF7EE")
for (x, y, t) in heads:
    arrow(6.9, y + 0.48, 7.4, 3.5)
box(9.8, 4.6, 1.9, 1.1, "Top-k router\nτ = 0.2 + 1.8·d\nk = 1 + round(d·(kmax−1))",
    fc="#FBEFF2", fs=7.5)
arrow(9.3, 3.7, 9.8, 5.15)
box(9.8, 1.4, 1.9, 2.4, "Experts (4 × MLP)\ntemporal · motion\nspectral · boundary",
    fc="#FFF7E6", fs=8)
arrow(10.75, 4.6, 10.75, 3.8, lw=1.4)
box(9.8, 0.2, 1.9, 0.8, "Σ wᵢ · logitᵢ", fc="#F0F0F0", fs=9)
arrow(11.4, 1.4, 11.4, 1.0)
ax.text(6.0, 6.7, "CAR: Compositional Artifact Routing",
        ha="center", fontsize=11, weight="bold")
ax.text(6.0, 6.35, "5.19M parameters · 3.82 GFLOPs per 8-frame clip",
        ha="center", fontsize=8.5, color="#555555")
qa(fig, "fig1")
export_all(fig, "fig1_architecture")
plt.close(fig)
print("fig1 done")

# ===========================================================================
# Fig 2 — 保留率热图（v2：viridis + 亮度自适应注记）
# ===========================================================================
RH = {m: load(R("robustness_honest", f"{m}.json"))
      for m in ["car", "car_s43", "car_s44", "efficientnet_b0",
                "efficientnet_b0_qaug", "xception", "xception_qaug", "mesonet"]}
TR = load(R("robustness_transcode", "summary.json"))

models = [("CAR (3-seed)", ["car", "car_s43", "car_s44"]),
          ("MesoNet", ["mesonet"]),
          ("B0", ["efficientnet_b0"]),
          ("B0+QA", ["efficientnet_b0_qaug"]),
          ("Xception", ["xception"]),
          ("Xception+QA", ["xception_qaug"])]
conds = [("JPEG Q30", "jpeg_quality=30"), ("Noise σ=.01", "noise_std=0.01"),
         ("Noise σ=.02", "noise_std=0.02"), ("Noise σ=.05", "noise_std=0.05"),
         ("Blur k7", "blur_kernel=7"), ("Bright ×1.2", "brightness_factor=1.2"),
         ("Transc. ×0.5", "transcode_scale=0.5"), ("Transc. ×0.25", "transcode_scale=0.25")]

ret = np.zeros((len(models), len(conds)))
for i, (_, keys) in enumerate(models):
    for j, (_, cond) in enumerate(conds):
        if cond.startswith("transcode"):
            ret[i, j] = TR[keys[0]][cond] / TR[keys[0]]["clean_auc_ref"]
        else:
            ret[i, j] = np.mean([RH[k][cond]["auc"] / RH[k]["clean"]["auc"] for k in keys])

fig, ax = plt.subplots(figsize=(7.2, 3.2))
im = ax.imshow(ret * 100, cmap="viridis", vmin=50, vmax=100, aspect="auto")
ax.set_xticks(range(len(conds)))
# 两行式标签替代旋转（实测 45° 旋转仍有 +7.6px 包围盒重叠；两行式水平占用 ~40px << 70px 列距）
ax.set_xticklabels([c.replace(" ", "\n", 1) for c, _ in conds])
ax.set_yticks(range(len(models)))
ax.set_yticklabels([m for m, _ in models])
for i in range(len(models)):
    for j in range(len(conds)):
        v = ret[i, j] * 100
        # viridis 低值端暗（<75 → 白字），高值端亮（≥75 → 黑字）
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if v < 75 else "black")
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("AUC retention (%) = AUC(degraded) / AUC(clean)")
ax.set_title("Robustness retention across degradation families", pad=8)
qa(fig, "fig2")
export_all(fig, "fig2_retention_heatmap")
plt.close(fig)
print("fig2 done")

# ===========================================================================
# Fig 3 — 路由补偿（v2：补 (a)/(b) 面板标签）
# ===========================================================================
RC = load(R("routing_compensation", "routing_compensation.json"))
EXPERTS = ["motion", "temporal", "spectral", "boundary"]
show_conds = [("clean", "Clean"), ("noise_std=0.05", "Noise σ=.05"),
              ("blur_kernel=7", "Blur k7"), ("jpeg_quality=30", "JPEG Q30"),
              ("transcode_scale=0.5", "Transc. ×0.5")]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for axi, seed, label in [(axes[0], "s42", "(a) Seed 42: compensation"),
                         (axes[1], "s44", "(b) Seed 44: invariant alternative")]:
    x = np.arange(len(show_conds))
    width = 0.19
    for e, exp in enumerate(EXPERTS):
        vals = [RC["results"][seed][c]["mean_weights"][exp] for c, _ in show_conds]
        axi.bar(x + (e - 1.5) * width, vals, width, label=exp, color=OI[e])
    axi.set_xticks(x)
    # 两行式标签替代旋转（实测 40° 旋转仍有 +8.4px 重叠）
    axi.set_xticklabels([l.replace(" ", "\n", 1) for _, l in show_conds], fontsize=7.5)
    axi.set_title(label, fontsize=9)
    axi.set_ylim(0, 0.82)
axes[0].set_ylabel("mean gate weight")
axes[1].legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
qa(fig, "fig3")
export_all(fig, "fig3_routing")
plt.close(fig)
print("fig3 done")

# ===========================================================================
# Fig 4 — 崩溃直方图（v2：重绘自 npz，补 (a)–(d) 标签 + ylabel）
# ===========================================================================
RH_DIR = R("robustness_honest")
N_DIR = R("significance", "noise005_preds")

specs = [
    ("B0", os.path.join(RH_DIR, "efficientnet_b0_clean_preds.npz"),
     os.path.join(N_DIR, "efficientnet_b0.npz"),
     os.path.join(RH_DIR, "efficientnet_b0.json")),
    ("CAR", os.path.join(RH_DIR, "car_clean_preds.npz"),
     os.path.join(N_DIR, "car.npz"),
     os.path.join(RH_DIR, "car_val_threshold.json")),
]

from sklearn.metrics import roc_auc_score

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
panel = 0
for mi, (name, clean_npz, noise_npz, thr_json) in enumerate(specs):
    with np.load(clean_npz, allow_pickle=True) as d:
        clean_preds = d["preds"].astype(np.float64)
        labels = d["labels"].astype(np.int64)
    with np.load(noise_npz, allow_pickle=True) as d:
        noise_preds = d["preds"].astype(np.float64)
    with open(thr_json, "r", encoding="utf-8") as f:
        frozen_thr = float(json.load(f)["threshold"])
    for ci, (cond, preds) in enumerate([("clean", clean_preds),
                                        ("noise σ=0.05", noise_preds)]):
        ax = axes[mi][ci]
        ax.hist(preds[labels == 0], bins=60, alpha=0.6, label="real",
                color="#0072B2", density=True)
        ax.hist(preds[labels == 1], bins=60, alpha=0.6, label="fake",
                color="#E69F00", density=True)
        ax.axvline(frozen_thr, color="#D55E00", ls="--", lw=1.2, label="frozen thr")
        auc = roc_auc_score(labels, preds)
        ax.set_title(f"({'abcd'[panel]}) {name}, {cond} (AUC={auc:.3f})", fontsize=9)
        ax.set_xlabel("score")
        if ci == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=7)
        panel += 1
qa(fig, "fig4")
export_all(fig, "fig4_collapse_histograms")
plt.close(fig)
print("fig4 done")
print(f"figures -> {OUT}")
