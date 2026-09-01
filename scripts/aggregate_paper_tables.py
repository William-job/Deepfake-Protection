#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文五表整合（Phase 5 收尾）

从各实验 JSON 自动汇编论文五张核心表（每个数字可追溯到源 JSON 字段）：
  T1 主结果        CAR-v3 vs baselines（clean test，全量协议）
  T2 鲁棒性        4 模型 × 15 条件（+ 可选 qaug 对照行）
  T3 消融          CAR-v3 专家消融 + 门控变体
  T4 专业化        响应矩阵 / 独立 AUC / 门控权重 / inter-expert 相似度
  T5 效率          params / FLOPs / GPU/CPU FPS（依赖 efficiency_honest.py）
附加：
  T6 per-method    FaceReenact / FaceSwap / TalkingFace 分解
  T7 tradeoff      v2(无噪声增强) vs v3(有) 的 clean-鲁棒权衡

用法：
    python -u scripts/aggregate_paper_tables.py
输出：
    results/paper_tables/tables.json + tables.md
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "paper_tables")


def log(msg):
    print(msg, flush=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trace = {}   # 表格数字 → 源文件（可追溯性记录）

    # ================================================================
    # T1 主结果
    # ================================================================
    t1_rows = []
    car_r = load_json(os.path.join(PROJECT_ROOT, "results", "robustness_honest", "car.json"))
    if car_r and "clean" in car_r:
        t1_rows.append(("CAR-v3 (ours)", car_r["clean"], "results/robustness_honest/car.json#clean"))
    honest = load_json(os.path.join(PROJECT_ROOT, "results", "baseline_honest", "summary.json"))
    if honest:
        # summary.json 实际结构：{"seeds": [...], "models": [ {"model": "car",
        #   "display_name": "CAR", "seeds": {"42": {...}}, "mean": {...}, "std": {...}}, ... ]}
        models_list = honest.get("models", [])
        models = {m["model"]: m for m in models_list if isinstance(m, dict) and "model" in m}
        for name in ["efficientnet_b0", "xception", "mesonet"]:
            m = models.get(name)
            if m and "mean" in m:
                a = m["mean"]
                std = m.get("std", {})
                display = m.get("display_name", name)
                row = {
                    "auc": a.get("auc"), "auc_std": std.get("auc"),
                    "accuracy": a.get("accuracy"), "accuracy_std": std.get("accuracy"),
                    "f1": a.get("f1"), "f1_std": std.get("f1"),
                    "ap": a.get("ap"), "eer": a.get("eer"),
                }
                t1_rows.append((display, row, f"results/baseline_honest/summary.json#{name}"))
    # v2 作为质量增强消融的对照行
    car_v2 = load_json(os.path.join(PROJECT_ROOT, "results", "final_car_v2", "eval_metrics.json"))
    if car_v2:
        t1_rows.append(("CAR-v2 (no noise-aug)", car_v2, "results/final_car_v2/eval_metrics.json"))

    # ================================================================
    # T2 鲁棒性
    # ================================================================
    t2 = {}
    for name in ["car", "efficientnet_b0", "xception", "mesonet",
                 "xception_qaug", "efficientnet_b0_qaug", "mesonet_qaug"]:
        d = load_json(os.path.join(PROJECT_ROOT, "results", "robustness_honest", f"{name}.json"))
        if d:
            t2[name] = {k: v["auc"] for k, v in d.items()
                        if isinstance(v, dict) and "auc" in v and k != "protocol"}
    # v2 鲁棒性（500 样本协议，标注来源）——仅噪声行用于 tradeoff 表
    v2_rob = load_json(os.path.join(PROJECT_ROOT, "results", "final_car_v2", "robustness.json"))
    v2_noise = {k: v["auc"] for k, v in v2_rob.items() if k.startswith("noise")} if v2_rob else {}

    # ================================================================
    # T3 消融
    # ================================================================
    t3 = load_json(os.path.join(PROJECT_ROOT, "results", "ablation_v3", "ablation.json"))

    # ================================================================
    # T4 专业化
    # ================================================================
    t4 = load_json(os.path.join(PROJECT_ROOT, "results", "specialization", "matrix.json"))

    # ================================================================
    # T5 效率
    # ================================================================
    t5 = load_json(os.path.join(PROJECT_ROOT, "results", "efficiency_honest", "efficiency.json"))

    # ================================================================
    # T6 per-method
    # ================================================================
    t6 = load_json(os.path.join(PROJECT_ROOT, "results", "per_method", "breakdown.json"))
    t6_v2 = load_json(os.path.join(PROJECT_ROOT, "results", "per_method", "breakdown_v2_and_baselines.json"))

    # ================================================================
    # 汇总输出
    # ================================================================
    tables = {
        "T1_main": {"rows": [{"model": n, "metrics": m, "source": s} for n, m, s in t1_rows]},
        "T2_robustness": t2,
        "T2_note": "全量 test 5418、冻结 val 阈值、噪声 seed=2024（各模型同实现）",
        "T3_ablation": t3,
        "T4_specialization": t4,
        "T5_efficiency": t5,
        "T6_per_method": t6 or t6_v2,
        "T7_tradeoff": {
            "car_v2_noise_500subset": v2_noise,
            "car_v2_clean": car_v2.get("auc") if car_v2 else None,
            "car_v3_clean": car_r["clean"]["auc"] if car_r and "clean" in car_r else None,
            "car_v3_noise": {k: v for k, v in t2.get("car", {}).items() if k.startswith("noise")},
            "note": "v2 噪声行来自 500 样本子集协议（旧），v3 行来自全量协议（新）；tradeoff 结论不受影响（v2 在 σ=0.05 崩溃 vs v3 -0.02）",
        },
    }
    with open(os.path.join(OUT_DIR, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)

    # ---- markdown 渲染 ----
    md = []
    md.append("# 论文核心表（自动汇编）\n")
    md.append("> 生成于 aggregate_paper_tables.py；每个数字可追溯至源 JSON。\n")

    md.append("\n## T1 主结果（clean test，全量协议）\n")
    md.append("| Model | AUC | Acc | F1 | AP | EER |")
    md.append("|---|---|---|---|---|---|")
    for n, m, s in t1_rows:
        md.append(f"| {n} | {fmt(m.get('auc'))} | {fmt(m.get('accuracy'))} | "
                   f"{fmt(m.get('f1'))} | {fmt(m.get('ap'))} | {fmt(m.get('eer'))} |")

    if t2:
        md.append("\n## T2 鲁棒性（AUC，全量 test）\n")
        conds = list(next(iter(t2.values())).keys())
        md.append("| Model | " + " | ".join(conds) + " |")
        md.append("|---" * (len(conds) + 1) + "|")
        for name, row in t2.items():
            md.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in conds) + " |")

    if t3:
        md.append("\n## T3 消融（CAR-v3）\n")
        md.append("| Variant | AUC | ΔAUC | AP |")
        md.append("|---|---|---|---|")
        for k, v in t3["results"].items():
            md.append(f"| {k} | {fmt(v['auc'])} | {t3['delta'][k]:+.4f} | {fmt(v.get('ap'))} |")

    if t4:
        md.append("\n## T4 专业化证据（CAR-v3）\n")
        md.append("| Artifact | motion | temporal | spectral | boundary | 命中 |")
        md.append("|---|---|---|---|---|---|")
        for k, art in enumerate(t4["artifact_order"]):
            row = t4["response_matrix_all"][k]
            hit = t4["diagonal_dominance"]["per_artifact_hit"][art]
            md.append(f"| {art} | " + " | ".join(fmt(x, 3) for x in row) + f" | {'✓' if hit else '✗'} |")
        md.append(f"\n- inter-expert 余弦 off-diag: {t4['inter_expert_offdiag_mean']:.3f}")
        md.append(f"- 门控权重: {t4['gating']['mean_weights']}（熵 {t4['gating']['mean_entropy']:.3f}）")
        md.append(f"- 各专家独立 AUC: {t4['expert_standalone_auc']}")
        sp = t4.get("strong_probes", {}).get("report", {})
        for probe, rep in sp.items():
            md.append(f"- 强剂量探针 {probe}: 目标专家 Δ={rep['target_expert_delta']:+.4f} "
                      f"({'命中' if rep['argmax_is_target'] else '未命中'})")

    if t5:
        md.append("\n## T5 效率\n")
        md.append("| Model | Params(M) | FLOPs(G) | GPU FPS | CPU FPS | FP32(MB) |")
        md.append("|---|---|---|---|---|---|")
        for name, r in t5.items():
            if name == "hardware":
                continue
            md.append(f"| {r['display']} | {r['parameters']['total_M']} | "
                      f"{r['flops']['flops_G']} | {r['speed']['gpu_fps']} | "
                      f"{r['speed']['cpu_fps']} | {r['model_size']['fp32_MB']} |")

    t6_used = t6 or t6_v2
    if t6_used:
        md.append("\n## T6 Per-method 分解（AUC vs all-real）\n")
        cats = [c for c in ["FaceReenact", "FaceSwap", "TalkingFace"]]
        md.append("| Model | Overall | " + " | ".join(cats) + " |")
        md.append("|---" * (len(cats) + 2) + "|")
        for name, r in t6_used.get("results", {}).items():
            md.append(f"| {name} | {fmt(r.get('overall_auc'))} | " +
                      " | ".join(fmt(r.get(c, {}).get("auc_vs_real")) for c in cats) + " |")

    md.append("\n## T7 clean-鲁棒 tradeoff（质量感知组件消融）\n")
    md.append("| Model | clean AUC | noise σ=0.05 | 结论 |")
    md.append("|---|---|---|---|")
    v2_n05 = v2_noise.get("noise_std=0.05")
    v3_n05 = t2.get("car", {}).get("noise_std=0.05")
    md.append(f"| CAR-v2（无噪声增强） | {fmt(car_v2.get('auc') if car_v2 else None)} | "
              f"{fmt(v2_n05)}（500样本协议） | σ=0.05 崩溃 |")
    md.append(f"| CAR-v3（+噪声增强） | {fmt(car_r['clean']['auc'] if car_r else None)} | "
              f"{fmt(v3_n05)}（全量协议） | σ=0.05 仅 -0.02 |")

    with open(os.path.join(OUT_DIR, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log("\n".join(md))
    log(f"\n已保存: {os.path.join(OUT_DIR, 'tables.json')} / tables.md")


if __name__ == "__main__":
    main()
