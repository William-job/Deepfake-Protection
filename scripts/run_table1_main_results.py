"""生成论文 Table 1 主结果

调用 evaluate_v2.py 评估 3 个配置：
  1. Celeb-DF++ 联合训练最佳模型 → Celeb-DF++ test（用 --use_val_threshold）
  2. FF++ 联合训练最佳模型 → FF++ c23 test（用 --use_val_threshold）
  3. FF++ 联合训练最佳模型 → FF++ c40 test（用 --use_val_threshold，--compression c40）

汇总 3 个 metrics.json 到 results/table1/main_results.json，并打印表格预览。
实现方式：用 subprocess 调用 evaluate_v2.py，避免重复加载模型与配置冲突。
"""
import argparse
import json
import os
import subprocess
import sys

# 确保项目根目录在 sys.path 中，支持 `python scripts/run_table1_main_results.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Table 1 的 3 个评估配置
TABLE1_CONFIGS = [
    {
        "name": "Celeb-DF++ (test)",
        "checkpoint": "checkpoints_v6/best_model_joint_celebdf.pt",
        "dataset": "celebdf",
        "compression": "c23",
        "split": "test",
        "use_val_threshold": True,
    },
    {
        "name": "FF++ c23 (test)",
        "checkpoint": "checkpoints_v6/best_model_joint_ffpp.pt",
        "dataset": "ffpp",
        "compression": "c23",
        "split": "test",
        "use_val_threshold": True,
    },
    {
        "name": "FF++ c40 (test)",
        "checkpoint": "checkpoints_v6/best_model_joint_ffpp.pt",
        "dataset": "ffpp",
        "compression": "c40",
        "split": "test",
        "use_val_threshold": True,
    },
]


def log(msg):
    """统一带 [INFO] 前缀的日志输出"""
    print(f"[INFO] {msg}")


def safe_dir_name(name):
    """将配置名转为安全的目录名"""
    for ch in [" ", "(", ")", "/", "+"]:
        name = name.replace(ch, "_")
    return name


def run_evaluate(cfg, config_path, output_dir):
    """用 subprocess 调用 evaluate_v2.py 完成单次评估"""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "evaluate_v2.py"),
        "--checkpoint", os.path.join(PROJECT_ROOT, cfg["checkpoint"]),
        "--config", config_path,
        "--dataset", cfg["dataset"],
        "--compression", cfg["compression"],
        "--split", cfg["split"],
        "--output_dir", output_dir,
    ]
    if cfg["use_val_threshold"]:
        cmd.append("--use_val_threshold")
    log(f"Running evaluate_v2.py: {cfg['name']}")
    log("  CMD: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="生成论文 Table 1 主结果")
    parser.add_argument("--config", type=str, default="configs/joint_train.yaml",
                        help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="results/table1",
                        help="汇总结果输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary = []
    for i, cfg in enumerate(TABLE1_CONFIGS):
        sub_dir = os.path.join(args.output_dir, f"config_{i}_{safe_dir_name(cfg['name'])}")
        os.makedirs(sub_dir, exist_ok=True)
        run_evaluate(cfg, args.config, sub_dir)

        metrics_path = os.path.join(sub_dir, "metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        summary.append({
            "config": cfg["name"],
            "checkpoint": cfg["checkpoint"],
            "dataset": cfg["dataset"],
            "compression": cfg["compression"],
            "auc": m["auc"],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "ap": m["ap"],
            "eer": m["eer"],
            "tpr_at_fpr_1": m["tpr_at_fpr_1"],
            "tpr_at_fpr_01": m["tpr_at_fpr_01"],
            "threshold": m["threshold"],
            "num_samples": m["num_samples"],
            "checkpoint_epoch": m.get("checkpoint_epoch"),
        })

    # 保存汇总 JSON
    summary_path = os.path.join(args.output_dir, "main_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"table": "Table 1: Main Results", "results": summary}, f, indent=2, ensure_ascii=False)
    log(f"汇总结果已保存到 {summary_path}")

    # 打印表格预览
    print("\n" + "=" * 95)
    print("Table 1: Main Results")
    print("=" * 95)
    header = f"{'Dataset':<22} {'AUC':>8} {'Acc':>8} {'F1':>8} {'AP':>8} {'EER':>8} {'TPR@1%':>8} {'TPR@0.1%':>10}"
    print(header)
    print("-" * 95)
    for s in summary:
        print(f"{s['config']:<22} {s['auc']:>8.4f} {s['accuracy']:>8.4f} {s['f1']:>8.4f} "
              f"{s['ap']:>8.4f} {s['eer']:>8.4f} {s['tpr_at_fpr_1']:>8.4f} {s['tpr_at_fpr_01']:>10.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
