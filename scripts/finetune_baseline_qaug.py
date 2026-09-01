#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基线 + 质量增强微调对照（Devil's Advocate 防御实验）

审稿人必然质疑："鲁棒性增益来自质量增强训练，而非架构——把同样的增强
给 baseline，它们也会鲁棒。" 本实验正面回应：
- 从 seed_42 诚实基线 checkpoint 出发，用与 CAR-v3 同族的质量增强配方
  （quality_aug p 阶梯 + 高概率真实高斯噪声，σ∈[0,0.06]）微调 baseline；
- 早停标准与 CAR 一致：clean val AUC（不能牺牲域内性能换鲁棒性）；
- 之后用 robustness_honest.py 在同一协议下评估，形成三方对比：
  CAR-v3 vs baseline(无增强) vs baseline+增强。

预期结论二选一，均可写进论文：
  a) baseline+增强也鲁棒 → 鲁棒性归因于训练配方（属于本框架的一部分），
     CAR 的优势落在"同样鲁棒但 5.19M vs 21M"的效率轴上；
  b) baseline+增强仍弱 → 架构本身对鲁棒有贡献，更强的主张。

用法：
    python -u scripts/finetune_baseline_qaug.py --model xception \
        --checkpoint results/baseline_honest/xception/seed_42/best_model.pt
输出：
    results/baseline_qaug/<model>/best_model.pt + training.log
"""
import argparse
import os
import random
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.config import load_config
from src.utils.logger import setup_logger, close_logger
from src.data.dataset import DeepfakeDataset
from src.data.transforms import VideoTransform
from src.data.quality_aug import apply_quality_augmentation, _noise
from src.utils.metrics import compute_auc

# 与 CAR-v3 课程一致的质量增强概率阶梯（阶段递进）
QAUG_SCHEDULE = [0.2, 0.4, 0.6, 0.6, 0.6, 0.6]


def log(msg):
    print(f"[finetune-qaug] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    frames = torch.stack([item["frames"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {"frames": frames, "label": labels}


@torch.no_grad()
def evaluate_clean(model, loader, device, use_amp):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        y = batch["label"].cpu().numpy()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(frames)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds.append(p)
        labels.append(y)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    model.train()
    return float(compute_auc(preds, labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    choices=["xception", "efficientnet_b0", "mesonet"])
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="起点 checkpoint（seed_42 诚实基线）")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="微调学习率")
    ap.add_argument("--noise_focus", type=float, default=0.5,
                    help="每 batch 额外以该概率对样本施加真实高斯噪声（同 CAR-v3）")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    out_dir = args.output_dir or os.path.join(PROJECT_ROOT, "results", "baseline_qaug", args.model)
    os.makedirs(out_dir, exist_ok=True)
    logger, writer = setup_logger(out_dir)

    logger.info(f"baseline 质量增强微调: {args.model} (seed={args.seed})")
    logger.info(f"起点: {args.checkpoint}")
    logger.info(f"配方: qaug_schedule={QAUG_SCHEDULE}, noise_focus={args.noise_focus}, lr={args.lr}")

    # ---- 数据（与 baseline_full.py 完全一致的采样协议） ----
    train_ds = DeepfakeDataset(config.data.data_root, split="train",
                               num_frames=config.data.num_frames,
                               frame_stride=config.data.frame_stride,
                               image_size=config.data.image_size,
                               transform=VideoTransform())
    val_ds = DeepfakeDataset(config.data.data_root, split="val",
                             num_frames=config.data.num_frames,
                             frame_stride=config.data.frame_stride,
                             image_size=config.data.image_size)
    _labels = [int(s["label"]) for s in train_ds.samples]
    _cnt = Counter(_labels)
    _weights = [1.0 / _cnt[l] for l in _labels]
    _num_samples = 2 * min(_cnt.values())
    train_sampler = WeightedRandomSampler(weights=torch.DoubleTensor(_weights),
                                          num_samples=_num_samples, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)

    # val 监控用分层子集（全 real + fake 补齐）：每 epoch 全量 9734 样本评估在
    # GPU 竞争下不可行；子集 AUC 与全量 AUC 高度相关，用于早停足够。
    # 最终发表口径的阈值冻结/全量评估由 robustness_honest.py 的 val 缓存完成。
    real_idx = [i for i, s in enumerate(val_ds.samples) if s["label"] == 0]
    fake_idx = [i for i, s in enumerate(val_ds.samples) if s["label"] == 1]
    rng = random.Random(args.seed)
    n_fake = min(1500 - len(real_idx), len(fake_idx))
    val_subset_idx = sorted(real_idx + rng.sample(fake_idx, n_fake))
    val_monitor = Subset(val_ds, val_subset_idx)
    logger.info(f"val 监控子集: {len(real_idx)} real + {n_fake} fake = {len(val_subset_idx)}")
    val_loader = DataLoader(val_monitor, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True, persistent_workers=args.num_workers > 0,
                            prefetch_factor=4 if args.num_workers > 0 else None)

    # ---- 模型：从基线 checkpoint 加载 ----
    from baseline_full import build_model
    model = build_model(args.model, config.data.num_frames).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"已加载基线权重 (epoch={ckpt.get('epoch', '?')})")
    start_val_auc = evaluate_clean(model, val_loader, device, use_amp)
    logger.info(f"起点 clean val AUC: {start_val_auc:.4f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc, best_epoch, patience = start_val_auc, -1, 0
    for epoch in range(args.epochs):
        p_qaug = QAUG_SCHEDULE[min(epoch, len(QAUG_SCHEDULE) - 1)]
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device).long()
            optimizer.zero_grad()

            # ---- 与 CAR-v3 同族的质量增强配方 ----
            frames, _ = apply_quality_augmentation(frames, p=p_qaug)
            if args.noise_focus > 0:
                B = frames.shape[0]
                mask = torch.rand(B, device=frames.device) < args.noise_focus
                if mask.any():
                    frames = frames.clone()
                    frames[mask] = _noise(frames[mask])

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(frames)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        val_auc = evaluate_clean(model, val_loader, device, use_amp)
        # 每个 epoch 都保存 last_model.pt（协议：允许 clean val 温和下降，
        # 与 CAR-v3 STEP4 噪声微调 -0.6pt 的 trade-off 叙事一致）
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "val_auc": val_auc,
                    "recipe": {"qaug_p": p_qaug, "noise_focus": args.noise_focus,
                               "base_checkpoint": args.checkpoint}},
                   os.path.join(out_dir, "last_model.pt"))
        marker = ""
        if val_auc > best_val_auc:
            best_val_auc, best_epoch, patience = val_auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "best_val_auc": best_val_auc,
                        "recipe": {"qaug_p": p_qaug, "noise_focus": args.noise_focus,
                                   "base_checkpoint": args.checkpoint}},
                       os.path.join(out_dir, "best_model.pt"))
            marker = "  -> 保存 best"
        else:
            patience += 1
        logger.info(f"Epoch {epoch} [qaug_p={p_qaug}] TrainLoss={total_loss / max(n_batches, 1):.4f} "
                    f"ValAUC={val_auc:.4f}{marker}")
        if patience >= 3:
            logger.info(f"早停 @ epoch {epoch}（起点 {start_val_auc:.4f} → best {best_val_auc:.4f} @ epoch {best_epoch}）")
            break

    logger.info(f"微调完成: best val AUC={best_val_auc:.4f} @ epoch {best_epoch} "
                f"(起点 {start_val_auc:.4f})")
    close_logger(writer)


if __name__ == "__main__":
    main()
