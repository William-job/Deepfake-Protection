#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配对统计显著性检验（论文 Significance Table 专用）

审稿人要求：3 种子 mean±std 不足以支撑比较性主张，需要配对检验。
本脚本在同一 test 集（5418 样本，同一顺序）上对模型对的 AUC 差做：

1. Paired bootstrap（10000 次重采样，报告 p 值 + 95% CI）——主检验；
2. Bootstrap 方差配对 z 检验——辅助验证（等价大样本 DeLong 的作用）。

协议说明：
- 所有模型使用 robustness_honest.py 的同一 uint8 帧缓存与同退化实现，
  因此预测向量天然配对（同样本序）；
- clean 条件直接读已保存的 *_clean_preds.npz（零 GPU 开销）。

用法：
    python -u scripts/significance_test.py

输出：
    results/significance/significance_clean.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from scipy.stats import norm

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "significance")
RH_DIR = os.path.join(PROJECT_ROOT, "results", "robustness_honest")

BOOT_SEED = 2024


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 统计核心
# ---------------------------------------------------------------------------
def auc_score(preds, labels):
    """Rank-based AUC（含并列秩平均，等价 sklearn.roc_auc_score）。"""
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos_mask = labels == 1
    pos, neg = preds[pos_mask], preds[~pos_mask]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # 并列秩取平均
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    allv = np.concatenate([neg, pos])[order]
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, np.arange(1, len(order) + 1))
    avg_ranks = sums / counts
    ranks = avg_ranks[inv]
    ranks_back = np.empty_like(ranks)
    ranks_back[order] = ranks
    r_pos = ranks_back[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(neg) * len(pos)))


def paired_bootstrap(preds_a, preds_b, labels, n_boot=10000, seed=BOOT_SEED):
    """配对 bootstrap：同一重采样索引下比较两模型 AUC 差。

    返回 (observed_diff, p_value, ci_low, ci_high, diffs)。
    """
    n = len(labels)
    obs = auc_score(preds_a, labels) - auc_score(preds_b, labels)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = auc_score(preds_a[idx], labels[idx]) - auc_score(preds_b[idx], labels[idx])
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    p = min(1.0, max(p, 1.0 / n_boot))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), float(p), float(lo), float(hi), diffs


def bootstrap_z_test(diffs, obs):
    """基于 bootstrap 方差的配对 z 检验（大样本下与 DeLong 等价的作用）。"""
    se = float(diffs.std(ddof=1))
    z = obs / se if se > 0 else 0.0
    p = 2.0 * (1.0 - float(norm.cdf(abs(z))))
    return float(p), se, float(z)


# ---------------------------------------------------------------------------
# 分数加载
# ---------------------------------------------------------------------------
NPZ_MODELS = {
    "car": "car_clean_preds.npz",
    "car_s43": "car_s43_clean_preds.npz",
    "car_s44": "car_s44_clean_preds.npz",
    "efficientnet_b0": "efficientnet_b0_clean_preds.npz",
    "efficientnet_b0_qaug": "efficientnet_b0_qaug_clean_preds.npz",
    "xception": "xception_clean_preds.npz",
    "xception_qaug": "xception_qaug_clean_preds.npz",
    "mesonet": "mesonet_clean_preds.npz",
}

DEFAULT_PAIRS = [
    ("car", "efficientnet_b0", "CAR-v3 vs EfficientNet-B0 (same param tier)"),
    ("car", "xception", "CAR-v3 vs Xception (5.2M vs 24.9M)"),
    ("car", "xception_qaug", "CAR-v3 vs Xception+QAug (strongest robust baseline)"),
    ("car", "mesonet", "CAR-v3 vs MesoNet"),
    ("car_s43", "efficientnet_b0", "CAR-v3 s43 vs B0 (seed robustness)"),
    ("car_s44", "efficientnet_b0", "CAR-v3 s44 vs B0 (seed robustness)"),
]


def load_clean_preds(model):
    path = os.path.join(RH_DIR, NPZ_MODELS[model])
    with np.load(path, allow_pickle=True) as d:
        return d["preds"].astype(np.float64), d["labels"].astype(np.int64)


# ---------------------------------------------------------------------------
# 自检（先验证检验器本身，再验证数据）
# ---------------------------------------------------------------------------
def sanity_check():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 100 + [1] * 100)
    # 无分离（AUC≈0.5）与强分离（AUC≈1.0）构造可检验的对照
    p_weak = rng.normal(0.5, 0.2, 200)
    p_strong = np.concatenate([rng.normal(0.2, 0.05, 100), rng.normal(0.8, 0.05, 100)])
    _, p_null, lo_null, hi_null, _ = paired_bootstrap(p_weak, p_weak, labels,
                                                      n_boot=2000, seed=1)
    assert p_null > 0.05 and lo_null <= 0 <= hi_null, "自检失败：相同预测应无显著差异"
    _, p_alt, lo_alt, hi_alt, _ = paired_bootstrap(p_strong, p_weak, labels,
                                                    n_boot=2000, seed=1)
    assert p_alt < 0.05 and lo_alt > 0, "自检失败：强分离预测应显著"
    # auc_score 正确性：与已知完美分离 AUC=1 对照
    assert abs(auc_score(np.concatenate([rng.normal(0, 0.1, 50), np.ones(50) + 10]),
                         np.array([0] * 50 + [1] * 50)) - 1.0) < 1e-12
    log(f"检验器自检通过 (p_null={p_null:.3f}, p_alt={p_alt:.4f}, AUC=1.0 精确)")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=None,
                    help="自定义比较对，格式 modelA:modelB,modelA:modelB")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, "significance_clean.json")

    sanity_check()

    pairs = DEFAULT_PAIRS
    if args.pairs:
        pairs = []
        for spec in args.pairs.split(","):
            a, b = spec.split(":")
            pairs.append((a, b, f"{a} vs {b}"))

    cache = {}

    def get_preds(name):
        if name not in cache:
            cache[name] = load_clean_preds(name)
            log(f"已加载 {name} clean 分数 ({len(cache[name][0])} 样本)")
        return cache[name]

    results = []
    for a, b, desc in pairs:
        preds_a, labels_a = get_preds(a)
        preds_b, labels_b = get_preds(b)
        assert np.array_equal(labels_a, labels_b), f"{a} 与 {b} 的标签顺序不一致"
        obs, p, lo, hi, diffs = paired_bootstrap(preds_a, preds_b, labels_a,
                                                 n_boot=args.n_boot)
        p_z, se, z = bootstrap_z_test(diffs, obs)
        auc_a, auc_b = auc_score(preds_a, labels_a), auc_score(preds_b, labels_b)
        row = {
            "pair": f"{a} vs {b}",
            "desc": desc,
            "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
            "delta_auc": round(obs, 4),
            "bootstrap_p": round(p, 6),
            "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
            "z_test_p": round(p_z, 6),
            "significant_0.05": bool(p < 0.05),
            "significant_bonferroni": bool(p < 0.05 / len(pairs)),
        }
        results.append(row)
        log(f"{desc:<48} Δ={obs:+.4f}  p={p:.2e}  CI=[{lo:+.4f},{hi:+.4f}]  "
            f"{'**SIG**' if p < 0.05 else 'n.s.'}")

    out = {
        "protocol": (f"paired bootstrap n={args.n_boot}, seed={BOOT_SEED}, "
                     f"same test sample order (uint8 cache); z-test = bootstrap-variance paired z"),
        "n_test_samples": int(len(labels_a)),
        "n_pairs": len(pairs),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
