#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新架构跨数据集联合训练（P1：FF++ 零样本 0.52 → 目标 0.7+）

设计（审稿人视角）：
- 起点：CAR-v3（seed 42 最终权重）——检验"artifact 特征能否跨域迁移"
- 数据：FF++ train（4 种 manipulation）+ Celeb-DF train 混合，避免灾难遗忘
- 配方：与 phase4 相同的质量感知增强（jpeg/blur/resize/h264/noise/color + noise_focus），
  跨域伪造外观差异大，增强可抑制对特定数据集纹理的过拟合
- 监控：双 val（FF++ val + Celeb-DF val），保存"FF++ val 最优"与"联合最优"两个 checkpoint
- 输出：仅 checkpoint + 日志；最终数字在本机协议下评估

用法（云端）：
    python -u scripts/train_joint.py --config configs/cloud.yaml \
        --pretrained results/final_car_v3/checkpoints/best_model.pt \
        --output_dir results/joint_ff_celebdf
本地（Windows，num_workers=0）同理，config 用 default.yaml。
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from src.config import load_config
from src.utils.logger import setup_logger, close_logger
from src.data.dataset import DeepfakeDataset
from src.data.ff_frame_dataset import FFFrameDataset
from src.data.quality_aug import apply_quality_augmentation, _noise
from src.models.car import CAR
from src.utils.metrics import compute_auc

# ---- FF++ 帧随机起点包装（训练期数据多样性，评估仍用原版固定起点） ----
class FFTrainView(torch.utils.data.Dataset):
    """FFFrameDataset 的训练视图：随机帧起点。"""
    def __init__(self, base):
        self.base = base
        self.num_frames = base.num_frames
        self.frame_stride = base.frame_stride
        self.image_size = base.image_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base.samples[idx]
        frame_dir = sample["frame_dir"]
        pngs = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
        total = len(pngs)
        max_start = max(0, total - self.num_frames * self.frame_stride)
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        indices = [min(start + i * self.frame_stride, total - 1)
                   for i in range(self.num_frames)]
        import cv2
        frames = []
        for i in indices:
            img = cv2.imread(os.path.join(frame_dir, pngs[i]))
            if img is None:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            frames.append(img)
        arr = np.stack(frames).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return {"frames": torch.from_numpy(arr).permute(0, 3, 1, 2).float(),
                "label": torch.tensor(sample["label"], dtype=torch.float32)}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return {"frames": torch.stack([b["frames"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch])}


def car_probs(model, frames):
    out = model(frames)
    logits = out["logits"]
    return torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))


@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        p = car_probs(model, batch["frames"].to(device))
        preds.append(p.cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    model.train()
    return float(compute_auc(np.concatenate(preds), np.concatenate(labels)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloud.yaml")
    ap.add_argument("--pretrained", required=True,
                    help="起点 checkpoint（CAR-v3 最终权重）")
    ap.add_argument("--output_dir", default="results/joint_ff_celebdf")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ff_batches_per_epoch", type=int, default=300,
                    help="每 epoch FF++ 采样的 batch 数（与 Celeb-DF 混合比例控制）")
    ap.add_argument("--celeb_batches_per_epoch", type=int, default=150)
    ap.add_argument("--celeb_val_subset", type=int, default=1500,
                    help="Celeb val 监控子集大小（0=全量 9734；本机视频解码慢建议子集）")
    ap.add_argument("--noise_focus", type=float, default=0.5)
    ap.add_argument("--qaug_p", type=float, default=0.4)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    logger, writer = setup_logger(args.output_dir)

    logger.info("=" * 60)
    logger.info("跨数据集联合训练（FF++ + Celeb-DF）")
    logger.info(f"pretrained={args.pretrained}, epochs={args.epochs}, lr={args.lr}")
    logger.info(f"per-epoch batches: ff={args.ff_batches_per_epoch}, celeb={args.celeb_batches_per_epoch}")
    logger.info(f"qaug_p={args.qaug_p}, noise_focus={args.noise_focus}")

    # ---- 数据 ----
    ff_root = config.data.ff_root
    ff_train = FFTrainView(FFFrameDataset(
        ff_root, num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride, image_size=config.data.image_size,
        split="train", compression="c23"))
    ff_val = FFFrameDataset(
        ff_root, num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride, image_size=config.data.image_size,
        split="val", compression="c23")
    celeb_train = DeepfakeDataset(
        config.data.data_root, split="train",
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size)
    celeb_val = DeepfakeDataset(
        config.data.data_root, split="val",
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size)
    # Celeb val 监控子集（全 real + fake 补齐）：视频解码慢，子集与全量 AUC 高度相关
    if args.celeb_val_subset > 0 and len(celeb_val) > args.celeb_val_subset:
        _rv = [i for i, s in enumerate(celeb_val.samples) if s["label"] == 0]
        _fv = [i for i, s in enumerate(celeb_val.samples) if s["label"] == 1]
        _rng = random.Random(args.seed)
        _n_fake = min(args.celeb_val_subset - len(_rv), len(_fv))
        celeb_val = torch.utils.data.Subset(
            celeb_val, sorted(_rv + _rng.sample(_fv, _n_fake)))
        logger.info(f"Celeb val 监控子集: {len(_rv)} real + {_n_fake} fake")
    assert len(ff_train) > 0, f"FF++ train 为空：检查 {ff_root}/train.json 与 frames 目录"
    logger.info(f"FF++ train: {len(ff_train)}, FF++ val: {len(ff_val)}, "
                f"Celeb train: {len(celeb_train)}, Celeb val: {len(celeb_val)}")

    ff_loader = DataLoader(ff_train, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, collate_fn=collate_fn,
                           drop_last=True, persistent_workers=args.num_workers > 0,
                           prefetch_factor=4 if args.num_workers > 0 else None)
    celeb_loader = DataLoader(celeb_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              drop_last=True, persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    ff_val_loader = DataLoader(ff_val, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate_fn,
                               persistent_workers=args.num_workers > 0)
    celeb_val_loader = DataLoader(celeb_val, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=collate_fn,
                                  persistent_workers=args.num_workers > 0)

    # ---- 模型 ----
    model = CAR(config).to(device)
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    logger.info(f"已加载起点权重: {args.pretrained}")

    base_ff = eval_auc(model, ff_val_loader, device)
    base_celeb = eval_auc(model, celeb_val_loader, device)
    logger.info(f"起点 val AUC: FF++={base_ff:.4f}, Celeb-DF={base_celeb:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_ff, best_joint, best_epoch = base_ff, base_ff + base_celeb, -1
    for epoch in range(args.epochs):
        model.train()
        ff_iter = iter(ff_loader)
        celeb_iter = iter(celeb_loader)
        losses = []
        steps = args.ff_batches_per_epoch + args.celeb_batches_per_epoch
        for step in range(steps):
            try:
                batch = next(ff_iter if step % 3 != 2 else celeb_iter)  # 2:1 FF 侧重
            except StopIteration:
                ff_iter, celeb_iter = iter(ff_loader), iter(celeb_loader)
                batch = next(ff_iter if step % 3 != 2 else celeb_iter)
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            frames, _ = apply_quality_augmentation(frames, p=args.qaug_p)
            if args.noise_focus > 0:
                mask = torch.rand(frames.shape[0], device=frames.device) < args.noise_focus
                if mask.any():
                    frames = frames.clone()
                    frames[mask] = _noise(frames[mask])

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(frames)["logits"]
                loss = F.binary_cross_entropy_with_logits(
                    (logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1)).float(),
                    labels.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        ff_auc = eval_auc(model, ff_val_loader, device)
        celeb_auc = eval_auc(model, celeb_val_loader, device)
        joint = ff_auc + celeb_auc
        marker = ""
        if ff_auc > best_ff:
            best_ff = ff_auc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_auc_ff": ff_auc, "val_auc_celeb": celeb_auc,
                        "base_checkpoint": args.pretrained},
                       os.path.join(args.output_dir, "best_ff.pt"))
            marker += "  -> best_ff"
        if joint > best_joint:
            best_joint, best_epoch = joint, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_auc_ff": ff_auc, "val_auc_celeb": celeb_auc,
                        "base_checkpoint": args.pretrained},
                       os.path.join(args.output_dir, "best_joint.pt"))
            marker += "  -> best_joint"
        logger.info(f"Epoch {epoch} loss={np.mean(losses):.4f} "
                    f"FFval={ff_auc:.4f} CelebVal={celeb_auc:.4f}{marker}")

        with open(os.path.join(args.output_dir, "joint_result.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"pretrained": args.pretrained, "base_ff": base_ff,
                       "base_celeb": base_celeb, "best_ff": best_ff,
                       "best_epoch": best_epoch, "last_epoch": epoch,
                       "timestamp": datetime.now().isoformat(timespec="seconds")},
                      f, indent=2, ensure_ascii=False)

    logger.info(f"完成：起点 FF++ {base_ff:.4f} → best {best_ff:.4f} @ epoch {best_epoch}")
    logger.info("回传物: best_ff.pt / best_joint.pt / joint_result.json（本机协议评估）")
    close_logger(writer)


if __name__ == "__main__":
    main()
