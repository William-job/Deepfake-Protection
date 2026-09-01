#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B0 崩溃的阈值-无关性分析（Oracle Threshold Analysis，审稿人 Major#2 专用）

质疑：B0 在 σ=0.05 下 AUC 0.524 的"崩溃"，是否只是 frozen threshold 的
      错位（score distribution 平移导致阈值失效），而非真实排序退化？

本分析用已保存的原始分数（clean npz + σ=0.05 npz）直接回答：
1. oracle threshold：在退化 test 集上重新寻优（Youden）→ 对比 frozen
   threshold 下的 accuracy；
2. all-fake 基线：1:29 类失衡下 accuracy 的"作弊上界"（5240/5418=96.7%）
   ——证明 accuracy 在失衡下可被"全判 fake"刷高，故 AUC 才是诚实度量；
3. score histogram：real/fake 分布在 clean vs σ=0.05 下的形态——
   排序退化（分布重叠）vs 阈值错位（分布平移但可分）一图可辨；
4. 保存论文附录图（PNG）与 JSON。

预期结论：oracle accuracy 不会高于 all-fake 基线多少，AUC 0.524 的崩溃
是排序性的（分布完全重叠），frozen-threshold 协议没有制造假崩溃。
"""
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "collapse_analysis")
RH_DIR = os.path.join(PROJECT_ROOT, "results", "robustness_honest")
N_DIR = os.path.join(PROJECT_ROOT, "results", "significance", "noise005_preds")

MODELS = {
    "efficientnet_b0": {
        "clean_npz": os.path.join(RH_DIR, "efficientnet_b0_clean_preds.npz"),
        "noise_npz": os.path.join(N_DIR, "efficientnet_b0.npz"),
        "threshold_json": os.path.join(RH_DIR, "efficientnet_b0.json"),
    },
    "car": {
        "clean_npz": os.path.join(RH_DIR, "car_clean_preds.npz"),
        "noise_npz": os.path.join(N_DIR, "car.npz"),
        "threshold_json": os.path.join(RH_DIR, "car_val_threshold.json"),
    },
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def auc_score(preds, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, preds))


def youden_threshold(preds, labels):
    """在给定分数上找 Youden J 最优阈值（升序扫描，含两端）。

    threshold t ∈ (s_i, s_{i+1}] 时 predicted fake = score >= t：
      TPR = 1 - cum1[i]/n_pos,  FPR = 1 - cum0[i]/n_neg
      J = TPR - FPR = cum0[i]/n_neg - cum1[i]/n_pos
    """
    order = np.argsort(preds, kind="mergesort")
    s = preds[order]
    y = labels[order]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    cum1 = np.concatenate([[0], np.cumsum(y == 1)])  # #{fake <= s_i}
    cum0 = np.concatenate([[0], np.cumsum(y == 0)])  # #{real <= s_i}
    j = cum0 / max(n_neg, 1) - cum1 / max(n_pos, 1)  # 长度 n+1，含 i=n（阈值=+inf）
    best = int(np.argmax(j))
    if best == 0:
        return float(s[0] - 1e-6)
    if best >= len(s):
        return float(s[-1] + 1e-6)
    return float((s[best - 1] + s[best]) / 2.0)


def max_accuracy_threshold(preds, labels):
    """accuracy-最优阈值（真 oracle：穷举所有分割点）。"""
    order = np.argsort(preds, kind="mergesort")
    s = preds[order]
    y = labels[order]
    n = len(s)
    cum1 = np.concatenate([[0], np.cumsum(y == 1)])
    cum0 = np.concatenate([[0], np.cumsum(y == 0)])
    n_pos, n_neg = cum1[-1], cum0[-1]
    # threshold 在 (s_i, s_{i+1}]：pred fake = 上方 i..n-1
    # correct = (n_pos - cum1[i]) + cum0[i]
    correct = (n_pos - cum1) + cum0
    best = int(np.argmax(correct))
    acc = correct[best] / n
    if best == 0:
        thr = float(s[0] - 1e-6)
    elif best >= n:
        thr = float(s[-1] + 1e-6)
    else:
        thr = float((s[best - 1] + s[best]) / 2.0)
    return thr, float(acc)


def accuracy_at(preds, labels, thr):
    return float(((preds >= thr).astype(int) == labels).mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for mi, (name, spec) in enumerate(MODELS.items()):
        with np.load(spec["clean_npz"], allow_pickle=True) as d:
            clean_preds = d["preds"].astype(np.float64)
            labels = d["labels"].astype(np.int64)
        with np.load(spec["noise_npz"], allow_pickle=True) as d:
            noise_preds = d["preds"].astype(np.float64)
            labels_n = d["labels"].astype(np.int64)
        assert np.array_equal(labels, labels_n)

        with open(spec["threshold_json"], "r", encoding="utf-8") as f:
            frozen_thr = float(json.load(f)["threshold"])

        n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
        all_fake_acc = n_pos / len(labels)  # 全判 fake 的 accuracy 上界

        row = {}
        for cond, preds in [("clean", clean_preds), ("noise_sigma=0.05", noise_preds)]:
            oracle_thr = youden_threshold(preds, labels)
            acc_thr, acc_max = max_accuracy_threshold(preds, labels)
            row[cond] = {
                "auc": round(auc_score(preds, labels), 4),
                "frozen_threshold": round(frozen_thr, 4),
                "frozen_acc": round(accuracy_at(preds, labels, frozen_thr), 4),
                "youden_oracle_threshold": round(oracle_thr, 4),
                "youden_oracle_acc": round(accuracy_at(preds, labels, oracle_thr), 4),
                "acc_oracle_threshold": round(acc_thr, 4),
                "acc_oracle": round(acc_max, 4),
            }
        row["all_fake_acc_bound"] = round(all_fake_acc, 4)
        row["class_ratio"] = f"1:{n_pos / n_neg:.1f} (fake:real)"
        out[name] = row

        log(f"== {name} ==")
        for cond in ["clean", "noise_sigma=0.05"]:
            r = row[cond]
            log(f"  {cond:<18} AUC={r['auc']:.4f}  frozen_acc={r['frozen_acc']:.4f}  "
                f"youden_oracle_acc={r['youden_oracle_acc']:.4f}  acc_oracle={r['acc_oracle']:.4f}")
        log(f"  all-fake accuracy bound = {all_fake_acc:.4f}（类失衡 {row['class_ratio']}）")

        # 直方图：clean 与 σ=0.05 各一格
        for ci, (cond, preds) in enumerate([("clean", clean_preds), ("noise σ=0.05", noise_preds)]):
            ax = axes[mi][ci]
            ax.hist(preds[labels == 0], bins=60, alpha=0.6, label="real", color="#4878CF", density=True)
            ax.hist(preds[labels == 1], bins=60, alpha=0.6, label="fake", color="#EE854A", density=True)
            ax.axvline(frozen_thr, color="red", ls="--", lw=1.2, label="frozen thr")
            ax.set_title(f"{name}  {cond}  (AUC={row['clean' if ci == 0 else 'noise_sigma=0.05']['auc']:.3f})",
                         fontsize=10)
            ax.set_xlabel("score")
            ax.legend(fontsize=8)

    verdict = {}
    for name in out:
        r = out[name]["noise_sigma=0.05"]
        gap = r["acc_oracle"] - out[name]["all_fake_acc_bound"]
        verdict[name] = {
            "acc_oracle_minus_allfake": round(gap, 4),
            "interpretation": (
                "accuracy-oracle 不高于 all-fake 多数类基线 → 分数不携带可用于"
                "阈值恢复的类别信息，崩溃是排序性的"
                if gap < 0.02 else
                "accuracy-oracle 显著高于 all-fake → 存在阈值可恢复的判别信息"),
        }
        log(f"[verdict] {name}: acc_oracle - all_fake = {gap:+.4f}")

    result = {
        "protocol": ("frozen val threshold vs test-set oracle (Youden) on identical "
                     "degraded scores; all-fake bound = majority-class accuracy under 1:29 imbalance"),
        "results": out,
        "verdict": verdict,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT_DIR, "collapse_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "collapse_histograms.png"), dpi=150)
    log(f"已保存: {OUT_DIR}/collapse_analysis.json + collapse_histograms.png")


if __name__ == "__main__":
    main()
