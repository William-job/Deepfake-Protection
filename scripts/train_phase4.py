#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 4：质量感知增强 + 五阶段课程微调。

在已训练模型（--pretrained）基础上，用五阶段渐进课程 + 质量感知增强
继续训练，重点修复 Noise σ=0.05 崩溃并提升跨数据集泛化。

用法：
    python scripts/train_phase4.py --pretrained results/final_car/checkpoints/best_model.pt \
        --output_dir results/final_car_v2 --epochs 15
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.utils.logger import setup_logger, close_logger
from src.data.dataset import DeepfakeDataset
from src.data.quality_aug import apply_quality_augmentation, _noise
from src.models.car import CAR
from src.training.losses import CARLoss
from src.training.five_stage_curriculum import FiveStageCurriculum
from src.utils.metrics import compute_all_metrics


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def build_train_loader(config, batch_size=8, num_workers=0):
    ds = DeepfakeDataset(config.data.data_root, split="train",
                         num_frames=config.data.num_frames,
                         frame_stride=config.data.frame_stride,
                         image_size=config.data.image_size)
    labels = [int(s["label"]) for s in ds.samples]
    cnt = Counter(labels)
    weights = [1.0 / cnt[l] for l in labels]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights),
                                    num_samples=2 * min(cnt.values()), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, collate_fn=collate_fn,
                      persistent_workers=num_workers > 0)


def build_val_loader(config, batch_size=8, num_workers=0):
    ds = DeepfakeDataset(config.data.data_root, split="val",
                         num_frames=config.data.num_frames,
                         frame_stride=config.data.frame_stride,
                         image_size=config.data.image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn,
                      persistent_workers=num_workers > 0)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        out = model(frames)
        logits = out["logits"]
        p = torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))
        preds.extend(p.cpu().tolist())
        labels.extend(batch["label"].tolist())
    return compute_all_metrics(np.array(preds), np.array(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--output_dir", default="results/final_car_v2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5, help="微调学习率（较小，避免破坏已有特征）")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader 工作进程数（Linux 云端建议 8；Windows 保持 0 避免死锁）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--noise_focus", type=float, default=0.0,
                    help="噪声主导微调：每 batch 额外以该概率对样本施加真实高斯噪声"
                         "（σ∈[0.03,0.06]，专攻 σ=0.05 崩溃），独立于课程质量增强")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device or "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, writer = setup_logger(log_dir)
    logger.info("=" * 60)
    logger.info("阶段 4：质量感知增强 + 五阶段课程微调")
    logger.info(f"pretrained={args.pretrained}, epochs={args.epochs}, lr={args.lr}")

    train_loader = build_train_loader(config, args.batch_size, args.num_workers)
    val_loader = build_val_loader(config, args.batch_size, args.num_workers)
    logger.info(f"Train {len(train_loader.dataset)}, Val {len(val_loader.dataset)}")

    model = CAR(config).to(device)
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    logger.info(f"加载预训练权重（epoch {ckpt.get('epoch', '?')}）")

    # 微调：较小 lr，冻结 stem 保持稳定
    for name, p in model.named_parameters():
        if "stem" in name:
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    criterion = CARLoss(
        aux_loss_weight=config.training.aux_loss_weight,
        load_balance_weight=config.training.load_balance_weight,
        difficulty_loss_weight=config.training.difficulty_loss_weight,
        min_expert_weight=config.training.min_expert_weight,
        min_expert_threshold=config.training.min_expert_threshold,
        label_smoothing=config.training.label_smoothing,
    )
    curriculum = FiveStageCurriculum(total_epochs=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_auc = 0.0
    best_epoch = -1
    patience_cnt = 0
    prev_stage = None

    for epoch in range(args.epochs):
        stage = curriculum.get_stage(epoch)
        if prev_stage is None or stage["name"] != prev_stage["name"]:
            logger.info(f"  [课程切换] epoch {epoch}: stage={stage['name']} "
                        f"(artifact_mix={stage['artifact_mix']}, quality_aug_p={stage['quality_aug_p']}, "
                        f"top_k={stage['top_k']}) — {stage['description']}")
            prev_stage = stage
        model.gating.top_k = stage["top_k"]

        # ---- train ----
        model.train()
        total_loss = 0.0
        preds_tr, labels_tr = [], []
        for batch in train_loader:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            # 质量感知增强：按当前 stage 的概率施加
            if stage["quality_aug_p"] > 0:
                frames, _ = apply_quality_augmentation(frames, p=stage["quality_aug_p"])
            # 噪声主导微调：额外高概率施加真实高斯噪声（修复 σ=0.05 崩溃）
            if args.noise_focus > 0:
                B = frames.shape[0]
                mask = torch.rand(B, device=frames.device) < args.noise_focus
                if mask.any():
                    noisy = _noise(frames[mask])
                    frames = frames.clone()
                    frames[mask] = noisy

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(frames)
                loss, _ = criterion(out, labels)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

            logits = out["logits"]
            p = torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))
            preds_tr.extend(p.detach().cpu().tolist())
            labels_tr.extend(labels.cpu().tolist())

        scheduler.step()
        m_tr = compute_all_metrics(np.array(preds_tr), np.array(labels_tr))

        # ---- val ----
        m_val = validate(model, val_loader, device)
        logger.info(f"Epoch {epoch} [{stage['name']}] "
                    f"TrainLoss={total_loss/max(1,len(train_loader)):.4f} "
                    f"TrainAUC={m_tr['auc']:.4f} ValAUC={m_val['auc']:.4f} ValF1={m_val['f1']:.4f}")

        if m_val["auc"] > best_auc:
            best_auc = m_val["auc"]
            best_epoch = epoch
            patience_cnt = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_auc": best_auc}, os.path.join(ckpt_dir, "best_model.pt"))
            logger.info(f"  -> 新最佳 val AUC={best_auc:.4f} (epoch {epoch})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"早停 @ epoch {epoch}（patience={args.patience}）")
                break

    logger.info(f"训练完成。best val AUC={best_auc:.4f} @ epoch {best_epoch}")
    with open(os.path.join(args.output_dir, "train_result.json"), "w") as f:
        import json
        json.dump({"best_val_auc": best_auc, "best_epoch": best_epoch}, f, indent=2)
    close_logger(writer)


if __name__ == "__main__":
    main()