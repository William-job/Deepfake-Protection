#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本机队列：夜间链收尾后自动重训 Xception + MesoNet（血统统一）

背景：published 诚实基线表（3-seed）权重在 Linux 环境（/root/trae），
本地 checkpoint 与之不一致（诊断见 results/robustness_honest/baseline_env_diagnosis.json）。
B0 已由 overnight_chain.py 重训；本脚本补齐 Xception 与 MesoNet，
使论文主表全部来自本地单一环境、可复现。

流程（对每个模型，复刻链中 B0 的处理模式）：
  1. baseline_full.py 从头重训 → results/baseline_retrain/<m>/seed_42
  2. eval_honest.py（val 冻结阈值 → test）
  3. 备份旧 checkpoint/eval_metrics → 复制重训产物到 baseline_honest 标准路径
  4. 归档旧鲁棒性 JSON → robustness_honest.py 重跑 15 条件
  5. 最后跑一次五表汇总

启动：
    python -u scripts/local_queue_after_chain.py
（等待 overnight_chain 的日志出现"夜间实验链结束"后自动接管）
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "overnight")
LOG_PATH = os.path.join(LOG_DIR, "local_queue.log")
CHAIN_LOG = os.path.join(LOG_DIR, "chain.log")

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


def run_step(name, cmd, timeout_hours=10):
    log(f"===== STEP {name}: {' '.join(cmd)}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=timeout_hours * 3600)
        dt = (time.time() - t0) / 60
        if r.returncode == 0:
            log(f"STEP {name} 完成（{dt:.0f} 分钟）")
            return True
        log(f"STEP {name} 失败（returncode={r.returncode}，{dt:.0f} 分钟）——继续")
        return False
    except subprocess.TimeoutExpired:
        log(f"STEP {name} 超时（>{timeout_hours}h）——继续")
        return False
    except Exception as e:
        log(f"STEP {name} 异常: {e}——继续")
        return False


def chain_done():
    if not os.path.exists(CHAIN_LOG):
        return False
    try:
        with open(CHAIN_LOG, "r", encoding="utf-8") as f:
            return "夜间实验链结束" in f.read()
    except Exception:
        return False


def wait_for_chain(max_h=10):
    log("等待夜间链结束（每 120s 轮询）...")
    t0 = time.time()
    while time.time() - t0 < max_h * 3600:
        if chain_done():
            log(f"夜间链已结束（等待 {(time.time() - t0) / 60:.0f} 分钟），接管本机队列")
            return True
        time.sleep(120)
    log("等待超时——强制接管（注意可能与链上 B0 鲁棒性步骤争抢 GPU）")
    return False


def robustness_complete(model):
    p = os.path.join(PROJECT_ROOT, "results", "robustness_honest", f"{model}.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            return all(c in json.load(f) for c in CONDITIONS)
    except Exception:
        return False


def promote_retrain(model):
    """重训产物提升为 baseline_honest 标准路径（旧文件先备份）。"""
    bh = os.path.join(PROJECT_ROOT, "results", "baseline_honest", model, "seed_42")
    rt = os.path.join(PROJECT_ROOT, "results", "baseline_retrain", model, "seed_42")
    ck_old = os.path.join(bh, "best_model.pt")
    ck_new = os.path.join(rt, "best_model.pt")
    em_old = os.path.join(bh, "eval_metrics.json")
    em_new = os.path.join(rt, "eval_metrics.json")
    if not (os.path.exists(ck_new) and os.path.exists(em_new)):
        log(f"[{model}] 重训产物不齐，跳过提升")
        return False
    for old, new in [(ck_old, ck_new), (em_old, em_new)]:
        bak = old + ".linux_lineage.bak"
        if os.path.exists(old) and not os.path.exists(bak):
            shutil.copy2(old, bak)
        shutil.copy2(new, old)
    log(f"[{model}] 重训 checkpoint + eval_metrics（新冻结阈值）已提升到 baseline_honest")
    # 归档旧鲁棒性结果（stale 权重产物）
    rob = os.path.join(PROJECT_ROOT, "results", "robustness_honest", f"{model}.json")
    if os.path.exists(rob):
        os.rename(rob, rob + ".stale.bak")
        log(f"[{model}] 旧鲁棒性 JSON 已归档")
    return True


def main():
    log("本机队列启动（Xception+MesoNet 重训）")
    wait_for_chain()

    for model in ["mesonet", "xception"]:  # 先小后大
        if robustness_complete(model) and not os.path.exists(
                os.path.join(PROJECT_ROOT, "results", "robustness_honest", f"{model}.json.stale.bak")):
            log(f"[{model}] 已完成且非 stale，跳过")
            continue
        rt = os.path.join(PROJECT_ROOT, "results", "baseline_retrain", model, "seed_42")
        if not os.path.exists(os.path.join(rt, "best_model.pt")):
            run_step(f"retrain_{model}",
                     [PY, "-u", "baseline_full.py", "--model", model, "--seed", "42",
                      "--output_dir", rt, "--epochs", "15", "--batch_size", "8",
                      "--num_workers", "4"], timeout_hours=8)
        run_step(f"eval_{model}",
                 [PY, "-u", "scripts/eval_honest.py", "--model", model, "--seed", "42",
                  "--checkpoint", os.path.join(rt, "best_model.pt"), "--output_dir", rt])
        if promote_retrain(model):
            run_step(f"robustness_{model}",
                     [PY, "-u", "scripts/robustness_honest.py",
                      "--models", model, "--batch_size", "32"], timeout_hours=6)

    run_step("aggregate", [PY, "-u", "scripts/aggregate_paper_tables.py"])
    log("本机队列结束")


if __name__ == "__main__":
    main()
