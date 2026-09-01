#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诚实基线训练+评估编排脚本（阶段 0.3）

串行执行每个 model × seed 的「训练 → 评估」，支持：
  - 断点续跑（跳过已存在 eval_metrics.json 的项）
  - 训练崩溃自动 resume 重试（Windows 上偶发 native crash，见 protocol.md）

用法：
    python scripts/run_honest_queue.py
    python scripts/run_honest_queue.py --dry-run
    python scripts/run_honest_queue.py --models efficientnet_b0 --seeds 42
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SEEDS = [42, 43, 44]
MODELS = ["car", "efficientnet_b0", "xception", "mesonet"]
MAX_RETRIES = 3  # 训练崩溃后的最大 resume 重试次数

BASELINE_ARGS = ["--epochs", "15", "--batch_size", "8", "--lr", "0.0001"]  # num_workers 由 --num-workers 动态传入


def latest_ckpt(model, out_dir):
    """可 resume 的训练断点路径（每 epoch 保存的 checkpoint_latest.pt）。"""
    if model == "car":
        return os.path.join(out_dir, "checkpoints", "checkpoint_latest.pt")
    return os.path.join(out_dir, "checkpoint_latest.pt")


def best_ckpt(model, out_dir):
    """最终产物（best model）路径，训练成功的判据。"""
    if model == "car":
        return os.path.join(out_dir, "checkpoints", "best_model.pt")
    return os.path.join(out_dir, "best_model.pt")


def train_cmd(model, seed, out_dir, num_workers=4):
    if model == "car":
        cmd = ["python", "train.py", "--seed", str(seed), "--output_dir", out_dir]
    else:
        cmd = ["python", "baseline_full.py", "--model", model,
               "--seed", str(seed), "--output_dir", out_dir] + BASELINE_ARGS + \
              ["--num_workers", str(num_workers)]
    # 存在断点则自动 resume（崩溃/中断后续训）
    ckpt = latest_ckpt(model, out_dir)
    if os.path.exists(ckpt):
        cmd += ["--resume", ckpt]
    return cmd


def eval_cmd(model, seed, out_dir):
    return ["python", os.path.join("scripts", "eval_honest.py"),
            "--model", model, "--checkpoint", best_ckpt(model, out_dir),
            "--seed", str(seed), "--output_dir", out_dir]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", type=str, default="results/baseline_honest")
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker 数（本机默认 4；云端多核建议 6-10）")
    args = ap.parse_args()

    root = os.path.join(PROJECT_ROOT, args.root) if not os.path.isabs(args.root) else args.root
    jobs = [(m, s, os.path.join(root, m, f"seed_{s}"))
            for m in args.models for s in args.seeds]

    print("=" * 70)
    print(f"诚实基线任务编排：{len(jobs)} 个任务（model×seed）")
    print(f"  root={root}   dry_run={args.dry_run}   max_retries={args.max_retries}")
    print("=" * 70)

    fail = False
    for m, s, out_dir in jobs:
        eval_metrics = os.path.join(out_dir, "eval_metrics.json")
        if os.path.exists(eval_metrics):
            print(f"[SKIP] {m}/seed_{s}: eval_metrics.json 已存在")
            continue

        print(f"\n{'='*70}\n[{m}/seed_{s}] -> {out_dir}\n{'='*70}")

        # ---- 训练（returncode==0 视为完成；崩溃则自动 resume 重试）----
        train_done = False
        attempt = 0
        while attempt < args.max_retries and not train_done:
            t_cmd = train_cmd(m, s, out_dir, args.num_workers)  # 每次重新组装，已存在断点则自动加 --resume
            if args.dry_run:
                print("  [训练] " + " ".join(t_cmd))
                break
            attempt += 1
            print(f"  [训练 attempt {attempt}/{args.max_retries}] " + " ".join(t_cmd))
            t0 = time.time()
            r = subprocess.run(t_cmd, cwd=PROJECT_ROOT)
            elapsed = (time.time() - t0) / 3600
            if r.returncode == 0:
                train_done = True
                print(f"  [训练] 正常结束，耗时 {elapsed:.1f}h")
            else:
                print(f"  [训练] 异常退出 code={r.returncode}（耗时 {elapsed:.1f}h），"
                      f"{'将 resume 重试' if attempt < args.max_retries else '已达重试上限'}")
                time.sleep(3)

        if not args.dry_run and not train_done:
            print(f"  [FAIL] {m}/seed_{s} 训练未正常完成（重试 {args.max_retries} 次后放弃）")
            fail = True
            break

        # ---- 评估 ----
        e_cmd = eval_cmd(m, s, out_dir)
        if args.dry_run:
            print("  [评估] " + " ".join(e_cmd))
        else:
            print("  [评估] " + " ".join(e_cmd))
            r = subprocess.run(e_cmd, cwd=PROJECT_ROOT)
            if r.returncode != 0:
                print(f"  [FAIL] 评估失败 code={r.returncode}，中断")
                fail = True
                break

    if fail:
        print("\n队列因错误中断。修复后重跑（已完成的 seed 会自动跳过）。")
        sys.exit(1)

    if not args.dry_run:
        print("\n所有任务完成。运行汇总：")
        print("  python scripts/aggregate_honest.py")
    else:
        print("\n[dry-run] 未实际执行任何命令。")


if __name__ == "__main__":
    main()