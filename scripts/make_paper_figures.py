#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文图生成器（paper/figures/，出版级 matplotlib，色盲友好）

  fig1_architecture.png      CAR 架构示意图（stem → 4 heads → fusion → router → experts）
  fig2_retention_heatmap.png 保留率热图（6 模型 × 8 退化条件，AUC/clean AUC）
  fig3_routing.png           路由补偿（s42 权重迁移 + s44 替代解，双面板）
  fig4 已有：results/collapse_analysis/collapse_histograms.png（崩溃直方图）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight",
})

# 色盲友好（Okabe-Ito）
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
      "#F0E442", "#000000"]


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


R = lambda *p: os.path.join(PROJECT_ROOT, "results", *p)

# ===========================================================================
# Fig 1 — 架构示意图
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")


def box(x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=8.5, style="round,pad=0.12"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                fc=fc, ec=ec, lw=1.2, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=10,
                                 lw=lw, color="#555555"))


box(0.2, 2.9, 1.7, 1.2, "8-frame\nclip\n224×224", fc="#F0F0F0")
box(2.4, 2.9, 1.9, 1.2, "Shared stem\nEfficientNet-B0\n(3.60M, frozen)", fc="#E7F0FA")
heads = [
    (5.0, 5.3, "temporal head"),
    (5.0, 3.9, "motion head"),
    (5.0, 2.5, "spectral head"),
    (5.0, 1.1, "boundary head"),
]
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
fig.savefig(os.path.join(OUT, "fig1_architecture.png"))
plt.close(fig)
print("fig1 done")

# ===========================================================================
# Fig 2 — 保留率热图
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
            aucs = [RH[k][cond]["auc"] / RH[k]["clean"]["auc"] for k in keys]
            ret[i, j] = np.mean(aucs)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
im = ax.imshow(ret * 100, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([c for c, _ in conds], rotation=30, ha="right")
ax.set_yticks(range(len(models)))
ax.set_yticklabels([m for m, _ in models])
for i in range(len(models)):
    for j in range(len(conds)):
        v = ret[i, j] * 100
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                color="black" if v > 55 else "white")
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("AUC retention (%) = AUC(degraded) / AUC(clean)")
ax.set_title("Robustness retention across degradation families", pad=8)
fig.savefig(os.path.join(OUT, "fig2_retention_heatmap.png"))
plt.close(fig)
print("fig2 done")

# ===========================================================================
# Fig 3 — 路由补偿（双面板：s42 补偿 vs s44 替代解）
# ===========================================================================
RC = load(R("routing_compensation", "routing_compensation.json"))
EXPERTS = ["motion", "temporal", "spectral", "boundary"]
show_conds = [("clean", "Clean"), ("noise_std=0.05", "Noise σ=.05"),
              ("blur_kernel=7", "Blur k7"), ("jpeg_quality=30", "JPEG Q30"),
              ("transcode_scale=0.5", "Transc. ×0.5")]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, seed, title in [(axes[0], "s42", "Seed 42: compensation"),
                        (axes[1], "s44", "Seed 44: invariant alternative")]:
    x = np.arange(len(show_conds))
    width = 0.19
    for e, exp in enumerate(EXPERTS):
        vals = [RC["results"][seed][c]["mean_weights"][exp] for c, _ in show_conds]
        ax.bar(x + (e - 1.5) * width, vals, width, label=exp, color=OI[e])
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in show_conds], rotation=25, ha="right", fontsize=7.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 0.75)
axes[0].set_ylabel("mean gate weight")
axes[1].legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
fig.suptitle("Router weights under degradation", fontsize=10)
fig.savefig(os.path.join(OUT, "fig3_routing.png"))
plt.close(fig)
print("fig3 done")
print(f"figures -> {OUT}")
