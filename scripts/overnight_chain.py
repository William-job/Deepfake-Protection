#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""夜间实验链（无人值守自动编排）

背景：单块 8GB 笔记本 GPU 无法并行多个 Xception 级任务（显存抖动导致
7 倍减速），故全部串行。本脚本按优先级依次执行剩余实验，每步：
- 崩溃安全（上游产物为增量 JSON/checkpoint）
- 失败记录后继续下一步（不阻塞后续独立步骤）
- 全程日志写入 results/overnight/chain.log

步骤（按 Q2 录用价值排序）：
  0. 等待主鲁棒性任务完成（xception.json + mesonet.json 15 条件齐全）
  1. Xception 质量增强微调（Devil's Advocate 对照的起点）
  2. xception_qaug 鲁棒性评估（对照的 15 条件）
  3. v3 消融（复用 test 缓存）
  4. 效率表（须独占 GPU——本链即独占）
  5. 五表汇总
  6. B0 从头重训（修复本地 checkpoint 与 published 评估的分歧）
  7. B0 重训后鲁棒性重跑（增量覆盖旧 B0 结果）
  8. B0 重训后 eval_honest 复评
  9. 最终五表再汇总

用法：
    python -u scripts/overnight_chain.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "overnight")
LOG_PATH = os.path.join(LOG_DIR, "chain.log")
PY = sys.executable

CONDITIONS = ["clean", "jpeg_quality=90", "jpeg_quality=70", "jpeg_quality=50", "jpeg_quality=30",
              "noise_std=0.01", "noise_std=0.02", "noise_std=0.05",
              "blur_kernel=3", "blur_kernel=5", "blur_kernel=7",
              "brightness_factor=1.2", "brightness_factor=0.8"]


def log(msg):
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name, cmd, timeout_hours=12):
    log(f"===== STEP {name}: {' '.join(cmd)}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=timeout_hours * 3600)
        dt = (time.time() - t0) / 60
        if r.returncode == 0:
            log(f"STEP {name} 完成（{dt:.0f} 分钟）")
            return True
        log(f"STEP {name} 失败（returncode={r.returncode}，{dt:.0f} 分钟）——继续后续步骤")
        return False
    except subprocess.TimeoutExpired:
        log(f"STEP {name} 超时（>{timeout_hours}h）——终止该步，继续")
        return False
    except Exception as e:
        log(f"STEP {name} 异常: {e}——继续后续步骤")
        return False


def robustness_complete(model):
    p = os.path.join(PROJECT_ROOT, "results", "robustness_honest", f"{model}.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return all(c in d for c in CONDITIONS)
    except Exception:
        return False


def wait_for(predicate, desc, poll_s=120, max_h=8):
    log(f"等待条件: {desc}（每 {poll_s}s 轮询，最长 {max_h}h）")
    t0 = time.time()
    while time.time() - t0 < max_h * 3600:
        try:
            if predicate():
                log(f"条件满足: {desc}（等待 {(time.time() - t0) / 60:.0f} 分钟）")
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    log(f"等待超时: {desc}——放弃等待，继续链")
    return False


def main():
    log("夜间实验链启动")
    log(f"Python: {PY}")

    # ---- 0. 等主鲁棒性任务（外部进程）完成 ----
    wait_for(lambda: robustness_complete("xception") and robustness_complete("mesonet"),
             "主鲁棒性（xception+mesonet 15 条件）")

    # ---- 1. Xception qaug 微调 ----
    run_step("finetune_xception_qaug",
             [PY, "-u", "scripts/finetune_baseline_qaug.py",
              "--model", "xception",
              "--checkpoint", "results/baseline_honest/xception/seed_42/best_model.pt"])

    # ---- 2. xception_qaug 鲁棒性 ----
    run_step("robustness_xception_qaug",
             [PY, "-u", "scripts/robustness_honest.py",
              "--models", "xception_qaug", "--batch_size", "32"])

    # ---- 3. v3 消融 ----
    run_step("ablation_v3", [PY, "-u", "scripts/ablation_v3.py"])

    # ---- 4. 效率表 ----
    run_step("efficiency", [PY, "-u", "scripts/efficiency_honest.py"])

    # ---- 5. 五表汇总 ----
    run_step("aggregate", [PY, "-u", "scripts/aggregate_paper_tables.py"])

    # ---- 6. B0 从头重训（干净训练，无 resume；输出到新目录避免污染旧产物） ----
    b0_out = os.path.join(PROJECT_ROOT, "results", "baseline_retrain", "efficientnet_b0", "seed_42")
    run_step("retrain_b0",
             [PY, "-u", "baseline_full.py", "--model", "efficientnet_b0", "--seed", "42",
              "--output_dir", b0_out, "--epochs", "15", "--batch_size", "8",
              "--num_workers", "4"],
             timeout_hours=8)

    # ---- 7. B0 重训后：诚实协议复评（val→test，写入 retrain 目录） ----
    b0_ckpt = os.path.join(b0_out, "best_model.pt")
    if os.path.exists(b0_ckpt):
        run_step("eval_b0_retrain",
                 [PY, "-u", "scripts/eval_honest.py", "--model", "efficientnet_b0",
                  "--seed", "42", "--checkpoint", b0_ckpt, "--output_dir", b0_out])
        # 旧 B0 鲁棒性结果已过期 → 归档并重跑
        old = os.path.join(PROJECT_ROOT, "results", "robustness_honest", "efficientnet_b0.json")
        if os.path.exists(old):
            os.rename(old, old + ".stale_local_ckpt.bak")
            log("旧 efficientnet_b0.json 已归档（stale checkpoint 产物）")
        # 让 robustness_honest 使用重训权重：把重训 checkpoint 复制到
        # baseline_honest 标准路径（先备份旧权重与旧阈值文件）。
        backup = os.path.join(PROJECT_ROOT, "results", "baseline_honest", "efficientnet_b0",
                              "seed_42", "best_model.pt.stale.bak")
        orig = os.path.join(PROJECT_ROOT, "results", "baseline_honest", "efficientnet_b0",
                            "seed_42", "best_model.pt")
        if os.path.exists(orig) and not os.path.exists(backup):
            import shutil
            shutil.copy2(orig, backup)
            log(f"旧 B0 checkpoint 已备份: {backup}")
            shutil.copy2(b0_ckpt, orig)
            log("重训 B0 checkpoint 已就位（baseline_honest 路径）")
            # 同步新冻结阈值（eval_metrics.json 含 val Youden 阈值），
            # 否则 robustness_honest 会继续读 Linux 旧权重分布的阈值
            em_orig = os.path.join(PROJECT_ROOT, "results", "baseline_honest",
                                   "efficientnet_b0", "seed_42", "eval_metrics.json")
            em_new = os.path.join(b0_out, "eval_metrics.json")
            if os.path.exists(em_new):
                shutil.copy2(em_orig, em_orig + ".linux_eval.bak")
                shutil.copy2(em_new, em_orig)
                log("重训 B0 eval_metrics.json（新阈值）已同步")
        run_step("robustness_b0_retrain",
                 [PY, "-u", "scripts/robustness_honest.py",
                  "--models", "efficientnet_b0", "--batch_size", "16"])

    # ---- 8. 最终汇总 ----
    run_step("aggregate_final", [PY, "-u", "scripts/aggregate_paper_tables.py"])
    log("夜间实验链结束")


if __name__ == "__main__":
    main()
