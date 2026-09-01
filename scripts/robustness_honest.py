#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诚实鲁棒性对比（论文 Robustness Table 专用）

协议（与 results/baseline_honest/protocol.md 完全一致）：
1. 全量 test 集（5418 样本）——不用 500 样本子集；
2. 阈值冻结：val 集 Youden 定阈值后冻结到所有退化条件
   （baselines 复用 seed_42 已冻结阈值；CAR v3 现场重算一次并缓存）；
3. 退化实现与 robustness_experiment.py 逐字节一致（uint8 空间），
   保证与历史结果可对照：JPEG 90/70/50/30、noise 0.01/0.02/0.05、
   blur 3/5/7、brightness 1.2/0.8；
4. 每个模型×条件前 np.random.seed(2024)：所有模型看到完全相同的
   噪声实现（唯一随机退化），消除运气差异；
5. 帧缓存：test 集只解码一次存 uint8 npy（mmap 读取），4 模型共享。

用法：
    python -u scripts/robustness_honest.py                  # 全部 4 模型 × 15 条件
    python -u scripts/robustness_honest.py --models car     # 只跑指定模型
输出：
    results/robustness_honest/<model>.json     每模型全条件指标（增量写入）
    results/robustness_honest/summary.json     汇总对比表
    results/robustness_honest/<model>_clean_preds.npz  clean 条件原始分数
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import DeepfakeDataset
from src.models.car import CAR
from src.utils.metrics import (
    find_optimal_threshold,
    compute_auc,
    compute_accuracy,
    compute_f1,
    compute_ap,
    compute_eer,
    compute_tpr_at_fpr,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "robustness_honest")
NOISE_SEED = 2024

MODEL_SPECS = {
    "car": {
        "type": "car",
        "ckpt": os.path.join(PROJECT_ROOT, "results", "final_car_v3", "checkpoints", "best_model.pt"),
        "threshold": None,  # 现场从 val 重算并缓存
    },
    "car_s43": {
        "type": "car",
        "ckpt": os.path.join(PROJECT_ROOT, "results", "cloud_recovery", "final_car_v3_s43_best.pt"),
        "threshold": None,
    },
    "car_s44": {
        "type": "car",
        "ckpt": os.path.join(PROJECT_ROOT, "results", "cloud_recovery", "final_car_v3_s44_best.pt"),
        "threshold": None,
    },
    "efficientnet_b0": {
        "type": "baseline",
        "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "efficientnet_b0", "seed_42", "best_model.pt"),
        "threshold": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "efficientnet_b0", "seed_42", "eval_metrics.json"),
    },
    "xception": {
        "type": "baseline",
        "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "xception", "seed_42", "best_model.pt"),
        "threshold": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "xception", "seed_42", "eval_metrics.json"),
    },
    "mesonet": {
        "type": "baseline",
        "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "mesonet", "seed_42", "best_model.pt"),
        "threshold": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "mesonet", "seed_42", "eval_metrics.json"),
    },
    # ---- Devil's Advocate 对照：基线 + 同款质量增强微调 ----
    "xception_qaug": {
        "type": "baseline",
        "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_qaug", "xception", "best_model.pt"),
        "threshold": None,  # 微调后需重新冻结 val 阈值
    },
    "efficientnet_b0_qaug": {
        "type": "baseline",
        # 协议注记：B0 在强/温和两档 qaug 配方下 clean val 均下降（0.9343→0.92x），
        # 无超过起点的 epoch（Xception 同配方反而 +0.36pt）。此处采用温和配方
        # (lr=5e-6, noise_focus=0.3) 固定 2 epochs 的终点权重（last_model.pt，
        # clean val 0.9224），协议与 CAR-v3 STEP4 采用噪声微调后模型一致。
        "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_qaug", "efficientnet_b0", "last_model.pt"),
        "threshold": None,
    },
    "mesonet_qaug": {
        "type": "baseline",
        "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_qaug", "mesonet", "best_model.pt"),
        "threshold": None,
    },
}

DEGRADATIONS = {
    "clean": None,
    "jpeg_quality=90": ("jpeg", 90),
    "jpeg_quality=70": ("jpeg", 70),
    "jpeg_quality=50": ("jpeg", 50),
    "jpeg_quality=30": ("jpeg", 30),
    "noise_std=0.01": ("noise", 0.01),
    "noise_std=0.02": ("noise", 0.02),
    "noise_std=0.05": ("noise", 0.05),
    "blur_kernel=3": ("blur", 3),
    "blur_kernel=5": ("blur", 5),
    "blur_kernel=7": ("blur", 7),
    "brightness_factor=1.2": ("brightness", 1.2),
    "brightness_factor=0.8": ("brightness", 0.8),
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# test 帧缓存（uint8，只解码一次）
# ---------------------------------------------------------------------------
def build_test_cache(config):
    """将全量 test 帧解码为 uint8 缓存。返回 (frames_mmap, labels, video_ids, rel_paths)。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames_path = os.path.join(CACHE_DIR, "test_frames_u8.npy")
    meta_path = os.path.join(CACHE_DIR, "test_meta.json")

    if os.path.exists(frames_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        frames = np.load(frames_path, mmap_mode="r")
        log(f"test 缓存命中: {frames.shape[0]} 样本")
        return frames, np.array(meta["labels"]), meta["video_paths"], meta["rel_paths"]

    log("首次构建 test 帧缓存（全量解码，约 10-15 分钟）...")
    ds = DeepfakeDataset(
        config.data.data_root, split="test",
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
    )
    n = len(ds)
    first = ds[0]["frames"]                      # (T, C, H, W) float [-1,1]
    t, c, h, w = first.shape
    frames = np.zeros((n, t, h, w, c), dtype=np.uint8)
    labels, video_paths, rel_paths = [], [], []

    for i in tqdm(range(n), desc="decode test", ncols=80):
        item = ds[i]
        f = item["frames"].numpy()               # (T, C, H, W) [-1,1]
        f = ((f * 0.5 + 0.5) * 255.0).clip(0, 255)  # -> [0,255] uint8 域
        frames[i] = f.transpose(0, 2, 3, 1).astype(np.uint8)  # (T,H,W,C)
        labels.append(int(item["label"].item()))
        video_paths.append(str(item["video_path"]))
        rel_paths.append(str(ds.samples[i].get("rel_path", item["video_path"])))

    np.save(frames_path, frames)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "video_paths": video_paths,
                   "rel_paths": rel_paths, "num_frames": t, "image_size": h}, f)
    log(f"test 缓存已保存: {frames_path} ({n} 样本, {frames.nbytes / 1e9:.2f} GB)")
    return np.load(frames_path, mmap_mode="r"), np.array(labels), video_paths, rel_paths


# ---------------------------------------------------------------------------
# 退化（uint8 空间，与 robustness_experiment.py 逐字节一致）
# ---------------------------------------------------------------------------
def apply_degradation_batch(frames_u8, deg_type, severity):
    """frames_u8: (B, T, H, W, C) uint8 → 退化后同形状 uint8。"""
    if deg_type is None:
        return frames_u8
    out = np.empty_like(frames_u8)
    B, T = frames_u8.shape[0], frames_u8.shape[1]
    for b in range(B):
        for t in range(T):
            img = frames_u8[b, t]
            if deg_type == "jpeg":
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(severity)]
                _, enc = cv2.imencode(".jpg", img, encode_param)
                img = cv2.imdecode(enc, 1)
            elif deg_type == "noise":
                noise = np.random.normal(0, severity * 255, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            elif deg_type == "blur":
                k = int(severity)
                if k % 2 == 0:
                    k += 1
                img = cv2.GaussianBlur(img, (k, k), 0)
            elif deg_type == "brightness":
                img = np.clip(img.astype(np.float32) * severity, 0, 255).astype(np.uint8)
            out[b, t] = img
    return out


# ---------------------------------------------------------------------------
# 模型加载与前向
# ---------------------------------------------------------------------------
def load_car(config, checkpoint_path, device):
    model = CAR(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    used_ema = False
    if isinstance(ckpt, dict) and ckpt.get("ema_shadow") is not None:
        try:
            model.load_state_dict(ckpt["ema_shadow"], strict=True)
            used_ema = True
        except RuntimeError:
            model.load_state_dict(ckpt["ema_shadow"], strict=False)
            used_ema = True
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, used_ema


def load_baseline(model_name, num_frames, checkpoint_path, device):
    from baseline_full import build_model
    model = build_model(model_name, num_frames).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def forward_probs(model, frames_t, is_car):
    if is_car:
        logits = model(frames_t)["logits"]
        return torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))
    logits = model(frames_t)
    return torch.softmax(logits, dim=1)[:, 1]


# ---------------------------------------------------------------------------
# CAR val 阈值（流式解码，一次性，按模型名缓存）
# ---------------------------------------------------------------------------
def compute_car_val_threshold(config, device, ckpt_path, cache_name="car_v3", batch_size=16):
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUT_DIR, f"{cache_name}_val_threshold.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        log(f"{cache_name} val 阈值缓存命中: {d['threshold']:.4f} (val_auc={d['val_auc']:.4f})")
        return d["threshold"], d["val_auc"]

    log(f"计算 {cache_name} val 阈值（Youden，流式解码约 15 分钟）...")
    # 优先走 val 帧缓存（与 test 缓存同构，快 ~20 倍）
    val_frames, val_labels = build_val_cache(config)
    model, _ = load_car(config, ckpt_path, device)
    preds_all = np.zeros(val_frames.shape[0], dtype=np.float64)
    with torch.no_grad():
        for i in tqdm(range(0, val_frames.shape[0], batch_size), desc=f"{cache_name}:val(thr)", ncols=80):
            batch_u8 = np.asarray(val_frames[i:i + batch_size])
            x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
            x = (x / 255.0 - 0.5) / 0.5
            p = forward_probs(model, x.to(device), is_car=True)
            preds_all[i:i + batch_u8.shape[0]] = p.cpu().numpy()
    threshold = find_optimal_threshold(preds_all, val_labels)
    val_auc = float(compute_auc(preds_all, val_labels))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": float(threshold), "val_auc": val_auc,
                   "val_num_samples": int(len(val_labels))}, f, indent=2)
    log(f"{cache_name} val 阈值: {threshold:.4f} (val_auc={val_auc:.4f})")
    return float(threshold), val_auc


# ---------------------------------------------------------------------------
# val 帧缓存 + 任意模型的 val 阈值（qaug 基线微调后需重新冻结阈值）
# ---------------------------------------------------------------------------
def build_val_cache(config):
    """全量 val 帧缓存（uint8，与 test 缓存同构）。返回 (frames_mmap, labels)。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames_path = os.path.join(CACHE_DIR, "val_frames_u8.npy")
    meta_path = os.path.join(CACHE_DIR, "val_meta.json")
    if os.path.exists(frames_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        frames = np.load(frames_path, mmap_mode="r")
        log(f"val 缓存命中: {frames.shape[0]} 样本")
        return frames, np.array(meta["labels"])

    log("首次构建 val 帧缓存（全量解码，约 15-20 分钟）...")
    ds = DeepfakeDataset(
        config.data.data_root, split="val",
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
    )
    n = len(ds)
    first = ds[0]["frames"]
    t, c, h, w = first.shape
    frames = np.zeros((n, t, h, w, c), dtype=np.uint8)
    labels = []
    for i in tqdm(range(n), desc="decode val", ncols=80):
        item = ds[i]
        f = item["frames"].numpy()
        f = ((f * 0.5 + 0.5) * 255.0).clip(0, 255)
        frames[i] = f.transpose(0, 2, 3, 1).astype(np.uint8)
        labels.append(int(item["label"].item()))
    np.save(frames_path, frames)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "num_frames": t, "image_size": h}, f)
    log(f"val 缓存已保存: {frames_path} ({n} 样本, {frames.nbytes / 1e9:.2f} GB)")
    return np.load(frames_path, mmap_mode="r"), np.array(labels)


@torch.no_grad()
def compute_baseline_val_threshold(name, model, val_frames, val_labels, device, batch_size=16):
    """基于 val 缓存冻结 baseline 阈值（Youden），结果缓存到 JSON。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUT_DIR, f"{name}_val_threshold.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        log(f"{name} val 阈值缓存命中: {d['threshold']:.4f} (val_auc={d['val_auc']:.4f})")
        return d["threshold"], d["val_auc"]

    n = val_frames.shape[0]
    preds_all = np.zeros(n, dtype=np.float64)
    for i in tqdm(range(0, n, batch_size), desc=f"{name}:val(thr)", ncols=80):
        batch_u8 = np.asarray(val_frames[i:i + batch_size])
        x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
        x = (x / 255.0 - 0.5) / 0.5
        p = forward_probs(model, x.to(device), is_car=False)
        preds_all[i:i + batch_u8.shape[0]] = p.cpu().numpy()
    threshold = find_optimal_threshold(preds_all, val_labels)
    val_auc = float(compute_auc(preds_all, val_labels))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": float(threshold), "val_auc": val_auc,
                   "val_num_samples": int(n)}, f, indent=2)
    log(f"{name} val 阈值: {threshold:.4f} (val_auc={val_auc:.4f})")
    return float(threshold), val_auc


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def compute_metrics(preds, labels, threshold):
    return {
        "auc": float(compute_auc(preds, labels)),
        "accuracy": float(compute_accuracy(preds, labels, threshold)),
        "f1": float(compute_f1(preds, labels, threshold)),
        "ap": float(compute_ap(preds, labels)),
        "eer": float(compute_eer(preds, labels)),
        "tpr_at_fpr_1": float(compute_tpr_at_fpr(preds, labels, 0.01)),
        "tpr_at_fpr_01": float(compute_tpr_at_fpr(preds, labels, 0.001)),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# 主评估循环
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_model_on_all_conditions(name, spec, config, frames_mmap, labels,
                                 video_ids, device, batch_size=16):
    out_path = os.path.join(OUT_DIR, f"{name}.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    missing = [k for k in DEGRADATIONS if k not in results]
    if not missing:
        log(f"[{name}] 全部条件已完成，跳过")
        return results

    is_car = spec["type"] == "car"
    if is_car:
        model, used_ema = load_car(config, spec["ckpt"], device)
        threshold, _ = compute_car_val_threshold(config, device, spec["ckpt"], cache_name=name)
    else:
        # qaug 变体：threshold 为 None，需经 val 缓存重新冻结
        base_name = name.replace("_qaug", "")
        model = load_baseline(base_name, config.data.num_frames, spec["ckpt"], device)
        used_ema = False
        if spec["threshold"] is not None:
            with open(spec["threshold"], "r", encoding="utf-8") as f:
                threshold = float(json.load(f)["threshold"])
        else:
            val_frames, val_labels = build_val_cache(config)
            threshold, _ = compute_baseline_val_threshold(
                name, model, val_frames, val_labels, device)
            del val_frames
    model.eval()

    n = frames_mmap.shape[0]
    for cond in missing:
        deg = DEGRADATIONS[cond]
        deg_type, severity = (deg if deg else (None, None))
        np.random.seed(NOISE_SEED)  # 各模型看到相同噪声实现
        preds_all = np.zeros(n, dtype=np.float64)
        for i in tqdm(range(0, n, batch_size), desc=f"{name}:{cond}", ncols=80):
            idx = range(i, min(i + batch_size, n))
            batch_u8 = np.asarray(frames_mmap[i:i + batch_size])       # (b,T,H,W,C)
            batch_u8 = apply_degradation_batch(batch_u8, deg_type, severity)
            x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()  # (b,T,C,H,W)
            x = (x / 255.0 - 0.5) / 0.5
            p = forward_probs(model, x.to(device), is_car)
            preds_all[i:i + batch_u8.shape[0]] = p.cpu().numpy()

        metrics = compute_metrics(preds_all, labels, threshold)
        results[cond] = metrics
        # 增量写入（崩溃安全）
        record = {
            "model": name,
            "checkpoint": spec["ckpt"],
            "threshold": threshold,
            "used_ema": used_ema,
            "num_samples": int(n),
            "noise_seed": NOISE_SEED,
            "protocol": "full-test, frozen val threshold, uint8 degradation identical to robustness_experiment.py",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        log(f"[{name}] {cond:<20} AUC={metrics['auc']:.4f} "
            f"(Δ={metrics['auc'] - results['clean']['auc']:+.4f})" if "clean" in results else
            f"[{name}] {cond:<20} AUC={metrics['auc']:.4f}")

        # clean 条件额外保存原始分数（供 per-method 分解）
        if cond == "clean":
            np.savez(os.path.join(OUT_DIR, f"{name}_clean_preds.npz"),
                     preds=preds_all, labels=labels, video_ids=np.asarray(video_ids, dtype=object))

    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--models", type=str, default="car,efficientnet_b0,xception,mesonet")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    log(f"Device: {device}")

    config = load_config(args.config)
    set_seed(42)

    os.makedirs(OUT_DIR, exist_ok=True)
    frames, labels, video_ids, rel_paths = build_test_cache(config)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in model_names:
        spec = MODEL_SPECS[name]
        if not os.path.exists(spec["ckpt"]):
            log(f"[SKIP] checkpoint 不存在: {spec['ckpt']}")
            continue
        log(f"===== 评估 {name} =====")
        eval_model_on_all_conditions(name, spec, config, frames, labels,
                                     video_ids, device, args.batch_size)

    # 汇总
    summary = {}
    for name in MODEL_SPECS:
        p = os.path.join(OUT_DIR, f"{name}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            summary[name] = {k: d.get(k) for k in DEGRADATIONS if k in d}
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"汇总已保存: {os.path.join(OUT_DIR, 'summary.json')}")
    log("完成！")


if __name__ == "__main__":
    main()
