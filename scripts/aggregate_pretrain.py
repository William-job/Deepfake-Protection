#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 R1/R2/R3 初始化对照结果（阶段 2.4）。"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODES = ["R1", "R2", "R3"]
DESC = {
    "R1": "随机初始化（无 ImageNet）+ Level2/3",
    "R2": "ImageNet stem（无 artifact 预训练）",
    "R3": "ImageNet stem + 完整 Level2/3（CAR-aware）",
}


def main():
    rows = []
    for m in MODES:
        path = os.path.join(PROJECT_ROOT, "results", "pretrain", m, "pretrain_result.json")
        if not os.path.exists(path):
            rows.append({"mode": m, "desc": DESC[m], "val_auc": None, "missing": True})
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "mode": m,
            "desc": DESC[m],
            "val_auc": r.get("val_auc"),
            "router_kl": r.get("level3", {}).get("router_kl"),
            "level2": r.get("level2"),
            "missing": False,
        })

    out_dir = os.path.join(PROJECT_ROOT, "results", "pretrain")
    with open(os.path.join(out_dir, "r1r2r3_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "val_auc 为预训练后（未做下游微调）的零样本 val AUC，"
                            "用于对照'初始化对下游判别力的影响'，不代表最终 CAR 性能",
                   "rows": rows}, f, indent=2, ensure_ascii=False)

    lines = ["# R1/R2/R3 初始化对照（阶段 2.4）", ""]
    lines.append("> 说明：val_auc 为预训练后（未做下游 Celeb-DF 微调）的零样本 val AUC，"
                 "用于对照初始化对下游判别力的影响。")
    lines.append("")
    lines.append("| Mode | 初始化 | Router KL | val AUC (零样本) |")
    lines.append("|---|---|---|---|")
    for r in rows:
        auc = "N/A" if r["val_auc"] is None else f"{r['val_auc']:.4f}"
        kl = "-" if r.get("router_kl") is None else f"{r['router_kl']:.4f}"
        lines.append(f"| {r['mode']} | {r['desc']} | {kl} | {auc} |")
    lines.append("")
    lines.append("## Level 2 各专家伪任务 AUC（CAR-aware 预训练质量探针）")
    lines.append("")
    for r in rows:
        if r.get("level2"):
            lines.append(f"### {r['mode']}")
            lines.append("| 专家 | acc | auc | batches |")
            lines.append("|---|---|---|---|")
            for e, v in r["level2"].items():
                lines.append(f"| {e} | {v['acc']:.3f} | {v['auc']:.3f} | {v['batches']} |")
            lines.append("")
    with open(os.path.join(out_dir, "r1r2r3_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()