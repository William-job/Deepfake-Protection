"""
Full-data baseline training for fair comparison with CAR.
Trains baselines on the same 38,934 training samples that CAR used.

Acceleration (2026-06-29):
  - num_workers=4 + persistent_workers + prefetch_factor (解决 Windows 视频解码瓶颈)
  - AMP 混合精度 (torch.cuda.amp.autocast + GradScaler)
  - epoch 上限 15, patience=7 (quick 训练显示 5-6 epoch 收敛)
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import random
import numpy as np
import os
import sys
import json
import timm
from collections import Counter

from src.config import load_config
from src.utils.logger import setup_logger, close_logger
from src.data.dataset import DeepfakeDataset
from src.data.transforms import VideoTransform
from src.utils.metrics import compute_all_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    frames = torch.stack([item["frames"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {"frames": frames, "label": labels}


class FrameAvgModel(nn.Module):
    def __init__(self, backbone, num_frames=8):
        super().__init__()
        self.backbone = backbone
        self.num_frames = num_frames

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        logits = self.backbone(x)
        logits = logits.view(B, T, -1).mean(dim=1)
        return logits


class MesoNet(nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(4, 4)
        feat_size = image_size // 32
        self.fc1 = nn.Linear(64 * feat_size * feat_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.pool1(torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))))
        x = self.pool2(torch.relu(self.bn4(self.conv4(torch.relu(self.bn3(self.conv3(x)))))))
        x = self.pool3(torch.relu(self.bn6(self.conv6(torch.relu(self.bn5(self.conv5(x)))))))
        x = self.pool4(torch.relu(self.bn8(self.conv8(torch.relu(self.bn7(self.conv7(x)))))))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits.view(B, T, -1).mean(dim=1)


import os as _os
import re as _re

MODEL_CONFIGS = {
    "xception": {"timm_name": "xception41", "pretrained": True, "num_classes": 2},
    "efficientnet_b0": {"timm_name": "efficientnet_b0", "pretrained": True, "num_classes": 2},
    "efficientnet_b3": {"timm_name": "efficientnet_b3", "pretrained": False, "num_classes": 2},
}

B3_CONVERTED_WEIGHTS = "C:\\Users\\86188\\.cache\\torch\\hub\\checkpoints\\efficientnet_b3_converted.pth"


def _load_efficientnet_b3_weights(backbone, weights_path):
    import torch as _torch
    ckpt = _torch.load(weights_path, map_location='cpu', weights_only=True)
    backbone.load_state_dict(ckpt, strict=True)
    return backbone


def build_model(model_name_str, num_frames):
    if model_name_str == "mesonet":
        return MesoNet()

    model_config = dict(MODEL_CONFIGS[model_name_str])
    timm_name = model_config.pop("timm_name")
    # xception 回退候选（按优先级）：xception41 (新) -> xception (旧, 本地缓存) -> tf_xception
    xception_fallbacks = ["xception", "tf_xception"]
    try:
        backbone = timm.create_model(timm_name, **model_config)
    except (RuntimeError, KeyError, Exception) as e:
        # xception41 在线下载失败时，回退到旧版 xception（本地缓存）
        if model_name_str == "xception":
            for fb_name in xception_fallbacks:
                try:
                    backbone = timm.create_model(fb_name, **model_config)
                    print(f"[INFO] xception 回退到: {fb_name}")
                    break
                except Exception as inner_e:
                    print(f"[WARN] xception 回退 {fb_name} 失败: {inner_e}")
                    continue
            else:
                raise RuntimeError(f"xception 所有变体加载失败: {e}")
        else:
            alt = {"efficientnet_b0": "tf_efficientnet_b0",
                   "efficientnet_b3": "tf_efficientnet_b3"}
            backbone = timm.create_model(alt[model_name_str], **model_config)

    if model_name_str == "efficientnet_b3" and _os.path.exists(B3_CONVERTED_WEIGHTS):
        backbone = _load_efficientnet_b3_weights(backbone, B3_CONVERTED_WEIGHTS)

    return FrameAvgModel(backbone, num_frames)


def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    """训练一个 epoch, 启用 AMP 混合精度 (若 use_amp=True)."""
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device).long()
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(frames)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n if n > 0 else 0


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    """评估, 评估阶段也可启用 autocast 以加速 forward (不影响指标)."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(frames)
        else:
            logits = model(frames)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_preds.extend(probs.tolist())
        all_labels.extend(labels.tolist())
    return compute_all_metrics(np.array(all_preds), np.array(all_labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["mesonet", "efficientnet_b0", "efficientnet_b3", "xception"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Epoch 上限 (默认 15, quick 训练显示 5-6 epoch 已收敛)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (默认 7)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 工作进程数 (默认 4, Windows 实测最优; num_workers=8 反而更慢)")
    parser.add_argument("--no_amp", action="store_true",
                        help="禁用 AMP 混合精度 (默认启用)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（默认取 config.training.seed）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认 results/baselines/{model}）")
    parser.add_argument("--resume", type=str, default=None,
                        help="续训 checkpoint 路径（加载 model_state_dict，从 checkpoint epoch+1 继续训练）")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else getattr(config.training, "seed", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else (args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    use_amp = (not args.no_amp) and (device == "cuda") and torch.cuda.is_available()
    print(f"AMP (mixed precision): {use_amp}")
    print(f"num_workers: {args.num_workers}, epochs: {args.epochs}, patience: {args.patience}")

    # AMP GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = args.output_dir if args.output_dir else f"results/baselines/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    logger, writer = setup_logger(out_dir)

    logger.info(f"Full-Data Baseline Training: {args.model} (seed={seed})")
    logger.info(f"Samples: 38,934 train / 9,734 val / 5,418 test (same as CAR)")
    logger.info(f"Config: epochs={args.epochs}, patience={args.patience}, "
                f"num_workers={args.num_workers}, amp={use_amp}, batch_size={args.batch_size}")

    logger.info("Loading datasets...")
    train_ds = DeepfakeDataset(config.data.data_root, split="train",
                               num_frames=config.data.num_frames,
                               frame_stride=config.data.frame_stride,
                               image_size=config.data.image_size,
                               transform=VideoTransform())
    val_ds = DeepfakeDataset(config.data.data_root, split="val",
                             num_frames=config.data.num_frames,
                             frame_stride=config.data.frame_stride,
                             image_size=config.data.image_size)
    test_ds = DeepfakeDataset(config.data.data_root, split="test",
                              num_frames=config.data.num_frames,
                              frame_stride=config.data.frame_stride,
                              image_size=config.data.image_size)

    # DataLoader: num_workers>0 + persistent_workers + prefetch_factor 加速视频解码
    persistent = args.num_workers > 0
    # 类别平衡采样：与 CAR 的 create_dataloader 完全一致
    # （real 890 vs fake 53196 ≈ 1:60；每 epoch 采样 2*minority，real:fake 1:1）
    _labels = [int(s["label"]) for s in train_ds.samples]
    _cnt = Counter(_labels)
    _weights = [1.0 / _cnt[l] for l in _labels]
    _minority = min(_cnt.values())
    _num_samples = 2 * _minority
    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(_weights),
        num_samples=_num_samples, replacement=True)
    logger.info(f"类别平衡采样: {dict(_cnt)} -> 权重 1/count, "
                f"num_samples={_num_samples} (=2×minority, 与 CAR 一致)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=persistent,
                              prefetch_factor=4 if persistent else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True, persistent_workers=persistent,
                            prefetch_factor=4 if persistent else None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn,
                             pin_memory=True, persistent_workers=persistent,
                             prefetch_factor=4 if persistent else None)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    model = build_model(args.model, config.data.num_frames).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # ---- Resume: 断点续训（支持完整恢复与 warm restart 两种模式）----
    start_epoch = 0
    resume_best_auc = 0.0
    resume_best_epoch = -1
    resume_patience = 0
    resume_results_log = []
    resume_is_full = False  # 是否为完整状态恢复（含 optimizer/scaler/patience/results_log）

    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"Resume checkpoint 不存在: {args.resume}")
            close_logger(writer)
            sys.exit(1)
        logger.info(f"=== Resume Session: 从 {args.resume} 恢复 ===")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        required_keys = ["model_state_dict", "epoch"]
        missing = [k for k in required_keys if k not in resume_ckpt]
        if missing:
            logger.error(f"Resume checkpoint 缺失键: {missing}（现有 keys: {list(resume_ckpt.keys())}）")
            close_logger(writer)
            sys.exit(1)
        # 加载 model_state_dict（覆盖随机初始化）
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = int(resume_ckpt["epoch"]) + 1

        # 检测恢复模式：完整恢复 vs warm restart
        resume_is_full = "optimizer_state_dict" in resume_ckpt

        # 恢复 best_val_auc（兼容新旧格式：新格式用 best_val_auc，旧格式用 val_auc）
        if "best_val_auc" in resume_ckpt:
            resume_best_auc = float(resume_ckpt["best_val_auc"])
        else:
            resume_best_auc = float(resume_ckpt.get("val_auc", 0.0))
        resume_best_epoch = int(resume_ckpt.get("epoch", -1))

        logger.info(f"  源 checkpoint epoch: {resume_ckpt['epoch']} (0-based)")
        logger.info(f"  恢复起始 epoch: {start_epoch} (1-based: Epoch {start_epoch+1})")
        logger.info(f"  恢复 best_val_auc: {resume_best_auc:.4f} (epoch {resume_best_epoch+1})")

        # 完整恢复模式：恢复 optimizer / scaler / patience / results_log
        if resume_is_full:
            logger.info("  恢复模式: 完整状态恢复 (full resume)")

            # 恢复 optimizer
            if "optimizer_state_dict" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                logger.info(f"  Optimizer state 已恢复（非 warm restart）")

            # 恢复 scaler
            if "scaler_state_dict" in resume_ckpt and use_amp:
                scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
                logger.info(f"  GradScaler state 已恢复")

            # 恢复 patience
            if "patience_counter" in resume_ckpt:
                resume_patience = int(resume_ckpt["patience_counter"])
                logger.info(f"  Patience counter 已恢复: {resume_patience}")

            # 恢复 results_log
            if "results_log" in resume_ckpt:
                resume_results_log = resume_ckpt["results_log"]
                logger.info(f"  历史训练记录已恢复: {len(resume_results_log)} 条")
        else:
            logger.info("  恢复模式: Warm restart（仅加载模型权重，optimizer 重置）")
            logger.info(f"  Optimizer state reset (warm restart), lr={args.lr}")

        # 恢复前 forward 验证
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, config.data.num_frames, 3, 224, 224).to(device)
            _ = model(dummy)
        logger.info("  Forward 验证通过 (shape 无 mismatch)")
        if start_epoch >= args.epochs:
            logger.error(f"  Resume epoch {start_epoch} >= epochs_limit {args.epochs}，无需训练")
            close_logger(writer)
            sys.exit(0)
        logger.info(f"  Resume from epoch {start_epoch-1}, training epoch {start_epoch}-{args.epochs-1}")

    best_val_auc = resume_best_auc   # 从 checkpoint 恢复
    patience = resume_patience       # 从 checkpoint 恢复
    results_log = resume_results_log # 从 checkpoint 恢复

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)

        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                    f"F1: {val_metrics['f1']:.4f}")

        results_log.append({"epoch": epoch, "loss": train_loss, "val": val_metrics})

        scheduler.step(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": best_val_auc},
                       os.path.join(out_dir, "best_model.pt"))
            logger.info(f"  -> Best! (AUC: {best_val_auc:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # 保存 latest checkpoint（原子写入，防止中断损坏）
        latest_path = os.path.join(out_dir, "checkpoint_latest.pt")
        latest_tmp = latest_path + ".tmp"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_auc": best_val_auc,
            "patience_counter": patience,
            "results_log": results_log,
        }, latest_tmp)
        os.replace(latest_tmp, latest_path)

    best_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, use_amp=use_amp)

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL TEST RESULTS ({args.model}):")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info(f"{'='*60}")

    results = {
        "model": args.model,
        "seed": seed,
        "params": total_params,
        "train_samples": len(train_ds),
        "val_auc": best_val_auc,
        "test": test_metrics,
        "epoch_log": results_log,
        "config": {
            "epochs_limit": args.epochs,
            "patience": args.patience,
            "num_workers": args.num_workers,
            "amp": use_amp,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
    }
    if args.resume:
        results["resume_info"] = {
            "resume_from": args.resume,
            "start_epoch": start_epoch,
            "resumed_best_auc": resume_best_auc,
            "resumed_best_epoch": resume_best_epoch,
            "full_resume": resume_is_full,
            "optimizer_warm_restart": not resume_is_full,
        }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    close_logger(writer)
    logger.info("Done!")


if __name__ == "__main__":
    main()
