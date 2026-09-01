#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""有损视频转码退化评估（论文 Transcoding Robustness 表专用）

动机：真实社交平台分发链路对检测器的最大威胁是"低码率视频转码"
（降分辨率 + 有损重编码），而 JPEG/blur/noise 退化族不涵盖运动补偿
残差伪影。本实验补上这一退化轴。

退化实现（诚实协议，可复现）：
    空间下采样(scale) → MPEG-4 ASP 编码-解码 round-trip（OpenCV FFmpeg
    后端 mp4v 编码器，25fps）→ 上采样回 224×224。
    等级：scale = 1.0 / 0.75 / 0.5 / 0.25（探测 PSNR 梯度
    37.4 / 29.0 / 26.9 / 23.9 dB，严格单调）。
    注：H.264/openh264 与 ffmpeg 二进制在本环境不可用（网络受限），
    MPEG-4 ASP 与 H.264 同族（运动补偿 + DCT + 量化），论文如实报告。

协议（与 robustness_honest.py 完全一致）：
- 全量 test 集 5418（uint8 帧缓存 mmap）；
- 阈值冻结：复用 robustness_honest 已冻结的 val 阈值；
- 每模型×条件增量写入（崩溃安全）。

用法（GPU 空闲时）：
    python -u scripts/robustness_transcode.py                # 全部模型
    python -u scripts/robustness_transcode.py --models car  # 指定模型

输出：
    results/robustness_transcode/<model>.json
    results/robustness_transcode/summary.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

from src.config import load_config
from robustness_honest import (
    build_test_cache, load_car, load_baseline, forward_probs,
    compute_metrics, log, OUT_DIR as RH_DIR,
)
from src.utils.metrics import compute_auc

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "robustness_transcode")
TMP_DIR = os.path.join(OUT_DIR, "_tmp")
FPS = 25

MODEL_SPECS = {
    "car": ("car", os.path.join(PROJECT_ROOT, "results", "final_car_v3", "checkpoints", "best_model.pt"),
            os.path.join(RH_DIR, "car_val_threshold.json")),
    "car_s43": ("car", os.path.join(PROJECT_ROOT, "results", "cloud_recovery", "final_car_v3_s43_best.pt"),
                os.path.join(RH_DIR, "car_s43_val_threshold.json")),
    "car_s44": ("car", os.path.join(PROJECT_ROOT, "results", "cloud_recovery", "final_car_v3_s44_best.pt"),
                os.path.join(RH_DIR, "car_s44_val_threshold.json")),
    "efficientnet_b0": ("baseline", os.path.join(PROJECT_ROOT, "results", "baseline_honest", "efficientnet_b0", "seed_42", "best_model.pt"),
                        os.path.join(RH_DIR, "efficientnet_b0.json")),
    "efficientnet_b0_qaug": ("baseline", os.path.join(PROJECT_ROOT, "results", "baseline_qaug", "efficientnet_b0", "last_model.pt"),
                             os.path.join(RH_DIR, "efficientnet_b0_qaug_val_threshold.json")),
    "xception": ("baseline", os.path.join(PROJECT_ROOT, "results", "baseline_honest", "xception", "seed_42", "best_model.pt"),
                 os.path.join(RH_DIR, "xception.json")),
    "xception_qaug": ("baseline", os.path.join(PROJECT_ROOT, "results", "baseline_qaug", "xception", "best_model.pt"),
                      os.path.join(RH_DIR, "xception_qaug_val_threshold.json")),
    "mesonet": ("baseline", os.path.join(PROJECT_ROOT, "results", "baseline_honest", "mesonet", "seed_42", "best_model.pt"),
                os.path.join(RH_DIR, "mesonet.json")),
}

# 转码退化等级：scale ∈ (0,1]，1.0 表示纯编码退化（不下采样）
LEVELS = {
    "transcode_scale=1.0": 1.0,
    "transcode_scale=0.75": 0.75,
    "transcode_scale=0.5": 0.5,
    "transcode_scale=0.25": 0.25,
}


def transcode_roundtrip(frames_u8, scale, tmp_path):
    """单样本 8 帧转码 roundtrip。

    frames_u8: (T,H,W,C) uint8 → 退化后同形状。
    下采样用 INTER_AREA（抗混叠，模拟平台转码），上采样用 INTER_LINEAR。
    """
    T, H, W, C = frames_u8.shape
    if scale < 1.0:
        hw = (max(2, int(W * scale)), max(2, int(H * scale)))
        frames_in = np.stack([
            cv2.resize(f, hw, interpolation=cv2.INTER_AREA) for f in frames_u8
        ])
    else:
        frames_in = frames_u8
    # 编码-解码 round-trip（在降分辨率空间）
    w = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                        (frames_in.shape[2], frames_in.shape[1]))
    for t in range(T):
        w.write(frames_in[t])
    w.release()
    cap = cv2.VideoCapture(tmp_path)
    back = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        back.append(f)
    cap.release()
    if len(back) != T:  # 容错：帧数不齐时取前 T 帧（探测显示恒为 T）
        back = (back + [back[-1]] * T)[:T]
    back = np.stack(back)
    if back.shape[1:3] != (H, W):
        back = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
                          for f in back])
    return back


def load_threshold(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["threshold"])


@torch.no_grad()
def eval_model(name, mtype, ckpt, threshold, config, frames, labels, device,
               batch_size=16):
    out_path = os.path.join(OUT_DIR, f"{name}.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    missing = [k for k in LEVELS if k not in results]
    if not missing:
        log(f"[{name}] 全部条件已完成，跳过")
        return results

    if mtype == "car":
        model, used_ema = load_car(config, ckpt, device)
    else:
        model = load_baseline(name.replace("_qaug", ""), config.data.num_frames, ckpt, device)
        used_ema = False
    model.eval()

    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_path = os.path.join(TMP_DIR, f"rt_{os.getpid()}.mp4")
    n = frames.shape[0]

    for cond in missing:
        scale = LEVELS[cond]
        preds_all = np.zeros(n, dtype=np.float64)
        for i in range(0, n, batch_size):
            batch_u8 = np.asarray(frames[i:i + batch_size])  # (b,T,H,W,C)
            b = batch_u8.shape[0]
            deg = np.empty_like(batch_u8)
            for j in range(b):
                deg[j] = transcode_roundtrip(batch_u8[j], scale, tmp_path)
            x = torch.from_numpy(deg).permute(0, 1, 4, 2, 3).float()
            x = ((x / 255.0 - 0.5) / 0.5).to(device)
            p = forward_probs(model, x, mtype == "car")
            preds_all[i:i + b] = p.cpu().numpy()
        metrics = compute_metrics(preds_all, labels, threshold)
        results[cond] = metrics
        clean_auc = None
        # 对账：从 robustness_honest 读 clean AUC
        rh_path = os.path.join(RH_DIR, f"{name}.json")
        if os.path.exists(rh_path):
            with open(rh_path, "r", encoding="utf-8") as f:
                clean_auc = json.load(f).get("clean", {}).get("auc")
        record = {
            "model": name,
            "checkpoint": ckpt,
            "threshold": threshold,
            "used_ema": used_ema,
            "num_samples": int(n),
            "degradation": "downscale(INTER_AREA) + MPEG-4 ASP(mp4v) roundtrip + upscale(INTER_LINEAR)",
            "clean_auc_ref": clean_auc,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        log(f"[{name}] {cond:<22} AUC={metrics['auc']:.4f}"
            + (f" (Δ vs clean={metrics['auc'] - clean_auc:+.4f})" if clean_auc else ""))

    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--models", type=str,
                    default="car,car_s43,car_s44,efficientnet_b0,efficientnet_b0_qaug,"
                            "xception,xception_qaug,mesonet")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    config = load_config(args.config)
    os.makedirs(OUT_DIR, exist_ok=True)

    frames, labels, _, _ = build_test_cache(config)

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name not in MODEL_SPECS:
            log(f"[SKIP] 未知模型: {name}")
            continue
        mtype, ckpt, thr_path = MODEL_SPECS[name]
        if not os.path.exists(ckpt):
            log(f"[SKIP] checkpoint 不存在: {ckpt}")
            continue
        if not os.path.exists(thr_path):
            log(f"[SKIP] 阈值文件不存在: {thr_path}")
            continue
        threshold = load_threshold(thr_path)
        log(f"===== {name} (threshold={threshold:.4f}) =====")
        eval_model(name, mtype, ckpt, threshold, config, frames, labels, device,
                   batch_size=args.batch_size)

    # 汇总
    summary = {}
    for name in MODEL_SPECS:
        p = os.path.join(OUT_DIR, f"{name}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            summary[name] = {k: d.get(k, {}).get("auc") if isinstance(d.get(k), dict) else d.get(k)
                             for k in list(LEVELS.keys()) + ["clean_auc_ref"]}
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log("汇总已保存: results/robustness_transcode/summary.json")
    log("完成！")


if __name__ == "__main__":
    main()
