#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""联合训练模型的跨数据集最终评估（论文 Cross-Dataset Table）

对 joint 训练产出的 checkpoint（best_ff.pt / best_joint.pt）执行：
1. FF++ test（零样本口径：模型见过的 FF++ 仅为 train/val，test 全新）：
   总体 AUC/AP + per-method（Deepfakes/Face2Face/FaceSwap/NeuralTextures）
2. Celeb-DF test：保真检查（联合训练是否牺牲域内性能，对照 3-seed 基线 0.8648）
3. 对照行：CAR-v3（未联合）的 FF++ 零样本 0.5100（train_joint.py 起点实测）

用法：
    python -u scripts/eval_joint.py --checkpoint results/joint_ff_celebdf_e12/best_ff.pt
    python -u scripts/eval_joint.py --checkpoint a.pt --checkpoint b.pt   # 多模型对比
输出：
    results/joint_eval/<ckpt名>.json + 控制台对比表
"""
import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import DeepfakeDataset
from src.data.ff_frame_dataset import FFFrameDataset
from src.models.car import CAR
from src.utils.metrics import compute_auc, compute_ap

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "joint_eval")
FF_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def car_probs(model, frames):
    logits = model(frames)["logits"]
    return torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))


@torch.no_grad()
def eval_ff_test(model, ff_root, config, device, batch_size=16):
    ds = FFFrameDataset(ff_root, num_frames=config.data.num_frames,
                        frame_stride=config.data.frame_stride,
                        image_size=config.data.image_size,
                        split="test", compression="c23")
    if len(ds) == 0:
        log("[WARN] FF++ test 为空")
        return None
    preds, labels, methods = [], [], []
    for i in tqdm(range(0, len(ds), batch_size), desc="FF++ test", ncols=80):
        batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        frames = torch.stack([b["frames"] for b in batch]).to(device)
        p = car_probs(model, frames).cpu().numpy()
        preds.append(p)
        labels.extend([int(b["label"].item()) for b in batch])
        methods.extend([b["method"] for b in batch])
    preds = np.concatenate(preds)
    labels = np.array(labels)
    methods = np.array(methods)

    out = {
        "num_samples": int(len(labels)),
        "overall_auc": float(compute_auc(preds, labels)),
        "overall_ap": float(compute_ap(preds, labels)),
        "per_method": {},
    }
    real_mask = methods == "original"
    for m in FF_METHODS:
        mm = methods == m
        if mm.sum() == 0:
            continue
        sub_p = np.concatenate([preds[mm], preds[real_mask]])
        sub_l = np.concatenate([labels[mm], labels[real_mask]])
        out["per_method"][m] = {
            "count": int(mm.sum()),
            "auc_vs_real": float(compute_auc(sub_p, sub_l)),
        }
    return out


@torch.no_grad()
def eval_celeb_test(model, config, device, batch_size=16):
    """Celeb-DF test：优先复用 uint8 帧缓存（与 robustness_honest 同源）。"""
    cache = os.path.join(PROJECT_ROOT, "results", "cache", "test_frames_u8.npy")
    meta_p = os.path.join(PROJECT_ROOT, "results", "cache", "test_meta.json")
    if not (os.path.exists(cache) and os.path.exists(meta_p)):
        log("[WARN] Celeb test 帧缓存不存在（先跑 robustness_honest.py）")
        return None
    frames = np.load(cache, mmap_mode="r")
    with open(meta_p, "r", encoding="utf-8") as f:
        labels = np.array(json.load(f)["labels"])
    preds = np.zeros(frames.shape[0], dtype=np.float64)
    for i in tqdm(range(0, frames.shape[0], batch_size), desc="Celeb test", ncols=80):
        batch_u8 = np.asarray(frames[i:i + batch_size])
        x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
        x = (x / 255.0 - 0.5) / 0.5
        p = car_probs(model, x.to(device)).cpu().numpy()
        preds[i:i + batch_u8.shape[0]] = p
    return {"num_samples": int(len(labels)),
            "overall_auc": float(compute_auc(preds, labels)),
            "overall_ap": float(compute_ap(preds, labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    ff_root = config.data.ff_root

    results = {}
    for ckpt_path in args.checkpoint:
        name = os.path.splitext(os.path.basename(os.path.dirname(ckpt_path)))[0] + "_" + \
            os.path.splitext(os.path.basename(ckpt_path))[0]
        if not os.path.exists(ckpt_path):
            log(f"[SKIP] {ckpt_path} 不存在")
            continue
        model = CAR(config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        model.eval()

        ff = eval_ff_test(model, ff_root, config, device, args.batch_size)
        celeb = eval_celeb_test(model, config, device, args.batch_size)
        rec = {"checkpoint": ckpt_path, "ff_test": ff, "celeb_test": celeb,
               "reference": {"car_v3_ff_zeroshot": 0.5100,
                             "car_3seed_celeb_clean": 0.8648},
               "timestamp": datetime.now().isoformat(timespec="seconds")}
        results[name] = rec

        out_path = os.path.join(OUT_DIR, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

        log(f"===== {name} =====")
        if ff:
            log(f"FF++ test: AUC={ff['overall_auc']:.4f} AP={ff['overall_ap']:.4f} "
                f"(ref: CAR-v3 零样本 0.5100)")
            for m in FF_METHODS:
                if m in ff["per_method"]:
                    log(f"  {m:<15} AUC={ff['per_method'][m]['auc_vs_real']:.4f}")
        if celeb:
            log(f"Celeb test: AUC={celeb['overall_auc']:.4f} (ref: 3-seed 基线 0.8648)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
