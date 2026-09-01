#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诚实基线汇总脚本（阶段 0.4）

从 results/baseline_honest/<model>/seed_<s>/eval_metrics.json 收集各模型多 seed 指标，
计算 mean ± std，输出 summary.json 与 summary.md。未完成全部 seed 时标记 complete=false。

用法：
    python scripts/aggregate_honest.py [--seeds 42 43 44] [--root results/baseline_honest]
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

METRICS = ["auc", "accuracy", "f1", "ap", "eer", "tpr_at_fpr_1", "tpr_at_fpr_01"]
METRIC_NAMES = {
    "auc": "AUC",
    "accuracy": "Acc",
    "f1": "F1",
    "ap": "AP",
    "eer": "EER",
    "tpr_at_fpr_1": "TPR@1%",
    "tpr_at_fpr_01": "TPR@0.1%",
}
MODELS = ["car", "efficientnet_b0", "xception", "mesonet"]
DISPLAY = {
    "car": "CAR",
    "efficientnet_b0": "EfficientNet-B0",
    "xception": "Xception",
    "mesonet": "MesoNet",
}


def fmt_mean_std(values):
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    std = var ** 0.5
    return f"{m:.4f}±{std:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/baseline_honest")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--models", type=str, nargs="+", default=MODELS)
    args = ap.parse_args()

    root = os.path.join(PROJECT_ROOT, args.root) if not os.path.isabs(args.root) else args.root
    summary = {"seeds": args.seeds, "models": [], "complete": True}

    for model in args.models:
        entry = {"model": model, "display_name": DISPLAY.get(model, model),
                 "seeds": {}, "mean_std": {}}
        present = []
        for s in args.seeds:
            path = os.path.join(root, model, f"seed_{s}", "eval_metrics.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                entry["seeds"][str(s)] = {k: rec[k] for k in METRICS if k in rec}
                present.append(s)
            else:
                summary["complete"] = False

        if present:
            for k in METRICS:
                vals = [entry["seeds"][str(s)][k] for s in present if k in entry["seeds"][str(s)]]
                m = sum(vals) / len(vals)
                std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
                entry["mean_std"][k] = {"mean": m, "std": std, "n": len(vals)}
        entry["n_seeds"] = len(present)
        summary["models"].append(entry)

    # 写 summary.json
    with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 写 summary.md
    lines = ["# 诚实基线汇总（阶段 0.3）", ""]
    lines.append(f"- seeds: {args.seeds}")
    lines.append(f"- complete: **{summary['complete']}**")
    lines.append("")
    header = "| Model | " + " | ".join(METRIC_NAMES[k] for k in METRICS) + " |"
    sep = "|---|" + "|".join(["---"] * len(METRICS)) + "|"
    lines.append(header)
    lines.append(sep)
    for entry in summary["models"]:
        cells = []
        complete = entry["n_seeds"] == len(args.seeds)
        name = entry["display_name"] + ("" if complete else f" ({entry['n_seeds']}/{len(args.seeds)})")
        cells.append(name)
        for k in METRICS:
            if k in entry["mean_std"]:
                ms = entry["mean_std"][k]
                cells.append(f"{ms['mean']:.4f}±{ms['std']:.4f}")
            else:
                cells.append("N/A")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("> 权威指标均来自 `eval_metrics.json`（val 集 Youden 阈值冻结到 test）。")
    lines.append("> 若 `complete=false`，上表仅统计已完成的 seed，未完成项以 (n/N) 标注。")
    with open(os.path.join(root, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 控制台打印
    print("\n" + header)
    print(sep)
    for entry in summary["models"]:
        cells = [entry["display_name"]]
        for k in METRICS:
            if k in entry["mean_std"]:
                ms = entry["mean_std"][k]
                cells.append(f"{ms['mean']:.4f}±{ms['std']:.4f}")
            else:
                cells.append("N/A")
        print("| " + " | ".join(cells) + " |")
    print(f"\ncomplete={summary['complete']}  -> {os.path.join(root, 'summary.md')}")


if __name__ == "__main__":
    main()