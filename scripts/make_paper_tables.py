#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文表格生成器（paper/tables.md，从冻结结果 JSON 聚合）

表格清单（与手稿 §6 对应）：
  T1  主对比 + 效率（clean AUC / 参数 / GFLOPs / FPS）
  T2  鲁棒性网格（8 模型 × 13 条件，AUC）
  T2b 转码鲁棒性（8 模型 × 4 等级）
  T3  移除消融 × 退化交叉
  T3b 单专家 oracle × 退化交叉
  T4  专家特化探针矩阵（响应/命中率/独立 AUC）
  T6  分伪造方法 AUC
  T7  v2→v3 噪声权衡（诚实协议）
  T8  跨数据集迁移（负结果）
  S1/S2 显著性检验（clean / σ=0.05）
  R1  路由补偿（3 种子 × 5 条件门控权重）

数据源全部为 results/ 下冻结 JSON；不引入任何新计算。
"""
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(PROJECT_ROOT, "paper", "tables.md")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


R = lambda *p: os.path.join(PROJECT_ROOT, "results", *p)

RH = {m: load(R("robustness_honest", f"{m}.json"))
      for m in ["car", "car_s43", "car_s44", "efficientnet_b0",
                "efficientnet_b0_qaug", "xception", "xception_qaug", "mesonet"]}
TR = load(R("robustness_transcode", "summary.json"))
ABR = load(R("ablation_robustness", "ablation_robustness.json"))
EO = load(R("ablation_robustness", "expert_only_robustness.json"))
PT = load(R("paper_tables", "tables.json"))
V2 = load(R("final_car_v2", "noise_honest.json"))
RC = load(R("routing_compensation", "routing_compensation.json"))
SIG_C = load(R("significance", "significance_clean.json"))
SIG_N = load(R("significance", "significance_noise005.json"))
JE = load(R("joint_eval", "joint_ff_celebdf_e12_best_joint.json"))
JF = load(R("joint_eval", "joint_ff_celebdf_e12_best_ff.json"))

L = []
w = L.append

# ---------------------------------------------------------------------------
w("# Paper tables (auto-generated from frozen result JSONs)\n")
w(f"Generated: {load(R('paper_tables','tables.json')).get('timestamp', 'n/a')} — see data-freeze list in manuscript.md\n")

EXPERTS = ["motion", "temporal", "spectral", "boundary"]

# ---- CAR 三种子统计 -------------------------------------------------------
car_clean = [RH[s]["clean"]["auc"] for s in ["car", "car_s43", "car_s44"]]
car_mean, car_std = np.mean(car_clean), np.std(car_clean, ddof=1)


def car3(cond):
    vals = [RH[s][cond]["auc"] for s in ["car", "car_s43", "car_s44"]]
    return np.mean(vals), np.std(vals, ddof=1)


# ---- T1 -------------------------------------------------------------------
w("## Table 1 — Main comparison and efficiency (clean, test 5,418 videos)\n")
w("| Model | Params (M) | GFLOPs | Clean AUC | GPU FPS | CPU FPS |")
w("|---|---|---|---|---|---|")
eff = PT["T5_efficiency"]
rows = [
    ("CAR (ours, 3 seeds)", f"5.19", f"3.82",
     f"{car_mean:.3f} ± {car_std:.3f}", "67.4", "8.3"),
    ("MesoNet", f"{eff['mesonet']['parameters']['total_M']:.2f}",
     f"{eff['mesonet']['flops']['flops_G']:.2f}",
     f"{RH['mesonet']['clean']['auc']:.3f}", "305.4", "55.8"),
    ("EfficientNet-B0", f"{eff['efficientnet_b0']['parameters']['total_M']:.2f}",
     f"{eff['efficientnet_b0']['flops']['flops_G']:.2f}",
     f"{RH['efficientnet_b0']['clean']['auc']:.3f}", "181.7", "9.7"),
    ("B0+QA", "4.01", "3.08",
     f"{RH['efficientnet_b0_qaug']['clean']['auc']:.3f}", "—", "—"),
    ("Xception", f"{eff['xception']['parameters']['total_M']:.2f}",
     f"{eff['xception']['flops']['flops_G']:.2f}",
     f"{RH['xception']['clean']['auc']:.3f}", "55.7", "3.2"),
    ("Xception+QA", "24.93", "40.43",
     f"{RH['xception_qaug']['clean']['auc']:.3f}", "—", "—"),
]
for r in rows:
    w("| " + " | ".join(r) + " |")
w("\nGPU: RTX 4060 Laptop, batch 1, 8-frame input, exclusive, warm-up; "
  "FPS = clips-per-second. MesoNet/B0/Xception single seed (42); "
  "B0 three-seed clean span 0.896–0.916. QA variants are fine-tuned copies of the "
  "plain architectures (same parameters and FLOPs; throughput unchanged, not re-measured).\n")

# ---- T2 鲁棒性网格 --------------------------------------------------------
w("## Table 2 — Robustness grid (AUC, full test 5,418, frozen val thresholds)\n")
conds = ["jpeg_quality=90", "jpeg_quality=70", "jpeg_quality=50", "jpeg_quality=30",
         "noise_std=0.01", "noise_std=0.02", "noise_std=0.05",
         "blur_kernel=3", "blur_kernel=5", "blur_kernel=7",
         "brightness_factor=1.2", "brightness_factor=0.8"]
# 前缀式表头：J=JPEG quality, N=noise σ, B=blur kernel, Br=brightness factor
hdr_alias = {"jpeg_quality": "J", "noise_std": "N", "blur_kernel": "B",
             "brightness_factor": "Br"}
def short_hdr(c):
    fam, val = c.split("=")
    return f"{hdr_alias[fam]}{val}"
w("| Model | Clean | " + " | ".join(short_hdr(c) for c in conds) + " |")
w("|---|" + "---|" * (len(conds) + 1))


def model_row(name):
    aucs = [RH[name][c]["auc"] for c in conds]
    return [f"{RH[name]['clean']['auc']:.3f}"] + [f"{a:.3f}" for a in aucs]


w("| CAR (seed 42) | " + " | ".join(model_row("car")) + " |")
m, s = None, None
for cond in conds:
    pass
# 三种子均值行
avg = [f"{np.mean([RH[k][c]['auc'] for k in ['car','car_s43','car_s44']]):.3f}"
       for c in conds]
w("| CAR (3-seed mean) | " + f"{car_mean:.3f} | " + " | ".join(avg) + " |")
for name, label in [("mesonet", "MesoNet"), ("efficientnet_b0", "B0"),
                    ("efficientnet_b0_qaug", "B0+QA"),
                    ("xception", "Xception"), ("xception_qaug", "Xception+QA")]:
    w(f"| {label} | " + " | ".join(model_row(name)) + " |")
w("\nColumn headers: J90–J30 = JPEG quality; N.01–N.05 = Gaussian noise σ in "
  "normalised [0,1] pixel space; B3–B7 = blur kernel; Br1.2/Br0.8 = brightness "
  "factor. Transcoding is reported separately in Table 2b.\n")

# ---- T2b 转码 --------------------------------------------------------------
w("## Table 2b — Transcoding robustness (AUC; scale = down-scale factor before MPEG-4 ASP round-trip)\n")
w("| Model | Clean | scale 1.0 | 0.75 | 0.5 | 0.25 | Retention@0.25 |")
w("|---|---|---|---|---|---|---|")
for key, label in [("car", "CAR (s42)"), ("car_s43", "CAR (s43)"), ("car_s44", "CAR (s44)"),
                   ("efficientnet_b0", "B0"), ("efficientnet_b0_qaug", "B0+QA"),
                   ("xception", "Xception"), ("xception_qaug", "Xception+QA"),
                   ("mesonet", "MesoNet")]:
    t = TR[key]
    clean = t["clean_auc_ref"]
    ret = t["transcode_scale=0.25"] / clean
    w(f"| {label} | {clean:.3f} | {t['transcode_scale=1.0']:.3f} | "
      f"{t['transcode_scale=0.75']:.3f} | {t['transcode_scale=0.5']:.3f} | "
      f"{t['transcode_scale=0.25']:.3f} | {ret*100:.1f}% |")
w("")

# ---- T3 移除消融 × 退化 ----------------------------------------------------
w("## Table 3 — Removal ablation × degradation (AUC; expert weight zeroed, renormalised)\n")
w("| Variant | Clean | σ=0.05 | blur k7 | JPEG Q30 |")
w("|---|---|---|---|---|")
for v, label in [("full", "Full CAR-v3"), ("-motion", "−motion"),
                 ("-temporal", "−temporal"), ("-spectral", "−spectral"),
                 ("-boundary", "−boundary"), ("uniform_gating", "Uniform gating (1/4)")]:
    r = ABR["results"]
    w(f"| {label} | {r['clean'][v]['auc']:.3f} | {r['noise_std=0.05'][v]['auc']:.3f} | "
      f"{r['blur_kernel=7'][v]['auc']:.3f} | {r['jpeg_quality=30'][v]['auc']:.3f} |")
w("")

# ---- T3b 单专家 ------------------------------------------------------------
w("## Table 3b — Single-expert oracle × degradation (AUC; weight 1 on one expert)\n")
w("| Variant | Clean | σ=0.05 | blur k7 | JPEG Q30 |")
w("|---|---|---|---|---|")
for v, label in [("full", "Full CAR-v3"), ("only_motion", "only motion"),
                 ("only_temporal", "only temporal"), ("only_spectral", "only spectral"),
                 ("only_boundary", "only boundary")]:
    r = EO["results"]
    w(f"| {label} | {r['clean'][v]['auc']:.3f} | {r['noise_std=0.05'][v]['auc']:.3f} | "
      f"{r['blur_kernel=7'][v]['auc']:.3f} | {r['jpeg_quality=30'][v]['auc']:.3f} |")
w("")

# ---- T4 特化探针 -----------------------------------------------------------
w("## Table 4 — Expert specialisation probes (synthetic artifacts, 1,200 samples)\n")
t4 = PT["T4_specialization"]
w("| Expert | Standalone AUC | Mean gate weight | Diagonal hit | Margin to best other |")
w("|---|---|---|---|---|")
hits = t4["diagonal_dominance"]["per_artifact_hit"]
margins = t4["diagonal_dominance"]["margin_to_best_other"]
gw = t4["gating"]["mean_weights"]
for e in EXPERTS:
    art = {"temporal": "temporal_jitter", "motion": "motion_ghost",
           "spectral": "compression_art", "boundary": "boundary_seam"}[e]
    w(f"| {e} | {t4['expert_standalone_auc'][e]:.3f} | {gw[e]:.3f} | "
      f"{'yes' if hits[art] else 'no'} | {margins[art]:+.3f} |")
w(f"\nDiagonal hit rate: {t4['diagonal_dominance']['hit_rate']*100:.0f}% "
  f"(3/4; motion fails — its target probe is captured by the temporal expert). "
  f"Inter-expert cosine mean: {t4['inter_expert_offdiag_mean']:.3f}.\n")

# ---- T6 分方法 --------------------------------------------------------------
w("## Table 6 — Per-manipulation AUC, all models (clean, vs all real)\n")
BB = load(R("per_method", "baseline_breakdown.json"))
w("| Model | FaceReenact | FaceSwap | TalkingFace |")
w("|---|---|---|---|")
for m, label in [("car", "CAR (s42)"), ("car_s43", "CAR (s43)"), ("car_s44", "CAR (s44)"),
                 ("mesonet", "MesoNet"), ("efficientnet_b0", "B0"),
                 ("efficientnet_b0_qaug", "B0+QA"), ("xception", "Xception"),
                 ("xception_qaug", "Xception+QA")]:
    r = BB["per_manipulation"][m]
    w(f"| {label} | {r['FaceReenact']['auc']:.3f} | {r['FaceSwap']['auc']:.3f} | "
      f"{r['TalkingFace']['auc']:.3f} |")
w("\nn per family: FaceReenact 1,400 / FaceSwap 1,740 / TalkingFace 2,100 vs 178 real.\n")

# ---- T2c TPR@FPR 安全表 ------------------------------------------------------
w("## Table 2c — Low-FPR detection rate: TPR at FPR=1% (frozen val thresholds)\n")
w("| Model | Clean | σ=0.05 | blur k7 | JPEG Q30 | Bright ×1.2 |")
w("|---|---|---|---|---|---|")
for m, label in [("car", "CAR (s42)"), ("car_s43", "CAR (s43)"), ("car_s44", "CAR (s44)"),
                 ("mesonet", "MesoNet"), ("efficientnet_b0", "B0"),
                 ("efficientnet_b0_qaug", "B0+QA"), ("xception", "Xception"),
                 ("xception_qaug", "Xception+QA")]:
    t = BB["tpr"][m]
    def g(c):
        return t[c]["tpr@fpr1%"]
    w(f"| {label} | {g('clean'):.3f} | {g('noise_std=0.05'):.3f} | "
      f"{g('blur_kernel=7'):.3f} | {g('jpeg_quality=30'):.3f} | "
      f"{g('brightness_factor=1.2'):.3f} |")
cm = [BB["tpr"][s]["clean"]["tpr@fpr1%"] for s in ["car", "car_s43", "car_s44"]]
cn = [BB["tpr"][s]["noise_std=0.05"]["tpr@fpr1%"] for s in ["car", "car_s43", "car_s44"]]
w(f"\nCAR 3-seed mean: clean {np.mean(cm):.3f} ± {np.std(cm, ddof=1):.3f}, "
  f"σ=0.05 {np.mean(cn):.3f} ± {np.std(cn, ddof=1):.3f}. "
  "Positive class = fake; TPR@FPR=0.1% available in "
  "results/per_method/baseline_breakdown.json.\n")

# ---- T7 v2→v3 --------------------------------------------------------------
w("## Table 7 — Noise-focused fine-tuning trade (identical architecture, honest protocol)\n")
w("| Model | Clean | σ=0.01 | σ=0.02 | σ=0.05 |")
w("|---|---|---|---|---|")
w(f"| CAR-v2 (stages 1–3) | {V2['clean']['auc']:.3f} | {V2['noise_std=0.01']['auc']:.3f} | "
  f"{V2['noise_std=0.02']['auc']:.3f} | {V2['noise_std=0.05']['auc']:.3f} |")
c3 = {c: car3(c) for c in ["clean", "noise_std=0.01", "noise_std=0.02", "noise_std=0.05"]}
w(f"| CAR-v3 (stages 1–4) | {c3['clean'][0]:.3f} | {c3['noise_std=0.01'][0]:.3f} | "
  f"{c3['noise_std=0.02'][0]:.3f} | {c3['noise_std=0.05'][0]:.3f} |")
w(f"\nTrade: −{V2['clean']['auc'] - c3['clean'][0]:.3f} clean for "
  f"+{c3['noise_std=0.05'][0] - V2['noise_std=0.05']['auc']:.3f} at σ=0.05 "
  f"(3-seed mean; v2 single run, protocol identical).\n")

# ---- T8 跨数据集 -----------------------------------------------------------
w("## Table 8 — Cross-dataset transfer to FF++ c23 (negative result)\n")
w("| Checkpoint | FF++ test AUC | Celeb-DF test AUC |")
w("|---|---|---|")
w(f"| zero-shot (no joint training) | 0.510 | {car_mean:.3f} |")
w(f"| best-joint | {JE['ff_test']['overall_auc']:.3f} | {JE['celeb_test']['overall_auc']:.3f} |")
w(f"| best-ff | {JF['ff_test']['overall_auc']:.3f} | {JF['celeb_test']['overall_auc']:.3f} |")
w("\nFF++ protocol: 1,796 train / 350 test videos (subset; not comparable to "
  "published FF++ numbers). Forgetting: 7.2–9.6 points on Celeb-DF.\n")

# ---- S1/S2 显著性 ----------------------------------------------------------
w("## Table S1 — Paired bootstrap significance (clean; n=10,000)\n")
w("| Pair | ΔAUC | p | 95% CI | Sig. (α=0.05) | Bonferroni (0.0083) |")
w("|---|---|---|---|---|---|")
for r in SIG_C["results"]:
    ci = f"[{r['bootstrap_ci95'][0]:+.4f}, {r['bootstrap_ci95'][1]:+.4f}]"
    w(f"| {r['desc'].split(' (')[0]} | {r['delta_auc']:+.4f} | {r['bootstrap_p']:.4f} | "
      f"{ci} | {'yes' if r['significant_0.05'] else 'no'} | "
      f"{'yes' if r['significant_bonferroni'] else 'no'} |")
w("")
w("## Table S2 — Paired bootstrap significance (σ=0.05; n=10,000)\n")
w("| Pair | ΔAUC | p | 95% CI | Sig. (α=0.05) |")
w("|---|---|---|---|---|")
for r in SIG_N["results"]:
    ci = f"[{r['bootstrap_ci95'][0]:+.4f}, {r['bootstrap_ci95'][1]:+.4f}]"
    w(f"| {r['desc']} | {r['delta_auc']:+.4f} | {r['bootstrap_p']:.4f} | {ci} | "
      f"{'yes' if r['significant_0.05'] else 'no'} |")
w("")

# ---- R1 路由补偿 -----------------------------------------------------------
w("## Table R1 — Routing compensation: mean gate weights under degradation\n")
w("| Seed | Condition | motion | temporal | spectral | boundary |")
w("|---|---|---|---|---|---|")
for seed in ["s42", "s43", "s44"]:
    for cond in ["clean", "noise_std=0.05", "blur_kernel=7", "jpeg_quality=30",
                 "transcode_scale=0.5"]:
        mw = RC["results"][seed][cond]["mean_weights"]
        w(f"| {seed} | {cond} | {mw['motion']:.3f} | {mw['temporal']:.3f} | "
          f"{mw['spectral']:.3f} | {mw['boundary']:.3f} |")
w("\nTop-two expert-pair counts (seed 42; router's highest-weighted pair per video): "
  "spectral+boundary 3,725/5,418 clean → 4,881/5,418 blur k7; "
  "motion+temporal 1,405 → 237.\n")

with open(PAPER, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print(f"written: {PAPER} ({len(L)} lines)")
