#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 3/4 集成训练：预训练(R3) → 专业化 → 课程联合训练（含一致性 loss）。

流程（顺序固定）：
  1. Level 2/3 预训练（Artifact-Centric，可选）
  2. 加载预训练权重（--pretrained）
  3. 课程联合训练：CARLoss + counterfactual specialization + consistency loss

与 train.py 的差异：在 Trainer 基础上叠加阶段 3 的三个正则项。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.utils.logger import setup_logger, close_logger
from src.data.dataloader import create_dataloader
from src.models.car import CAR
from src.training.trainer import Trainer
from src.training.specialization import SpecializationLoss
from src.training.consistency import ConsistencyLoss
from src.training import losses as _losses_mod


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pretrained", default=None,
                    help="R3 预训练权重（results/pretrain/R3/pretrained.pt）")
    ap.add_argument("--output_dir", default="results/final_car")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--spec_weight", type=float, default=0.1)
    ap.add_argument("--cons_weight", type=float, default=0.1)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：仅 1 epoch + 每 epoch 2 个正则 batch，验证可运行")
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
    config._data.setdefault("logging", {})
    config._data["logging"]["log_dir"] = log_dir
    config._data["logging"]["checkpoint_dir"] = ckpt_dir

    logger, writer = setup_logger(log_dir)
    logger.info("=" * 60)
    logger.info("阶段 3/4 集成训练：专业化 + 一致性 + 课程联合训练")
    logger.info(f"seed={args.seed}, device={device}")
    logger.info(f"spec_weight={args.spec_weight}, cons_weight={args.cons_weight}")

    train_loader = create_dataloader(config, split="train")
    val_loader = create_dataloader(config, split="val")
    logger.info(f"Train {len(train_loader.dataset)}, Val {len(val_loader.dataset)}")

    model = CAR(config)
    if args.pretrained and os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"加载预训练权重 {args.pretrained}（missing={len(missing)}, unexpected={len(unexpected)}）")

    model = model.to(device)
    stats = model.get_parameter_stats()
    logger.info(f"参数 total={stats['total']:,}")

    # 阶段 3 正则项
    spec_loss = SpecializationLoss(model, margin=0.5, weight=args.spec_weight, device=device)
    cons_loss = ConsistencyLoss(model, weight=args.cons_weight, device=device)

    trainer = Trainer(model, config, logger, writer, device=device,
                      use_amp=not args.no_amp)

    # 在 train_epoch 中注入正则：通过包装 trainer 的 train_epoch
    orig_train_epoch = trainer.train_epoch

    def train_epoch_with_reg(train_loader_inner):
        # 先跑一次标准 epoch（内部含 CARLoss）
        train_loss, train_metrics = orig_train_epoch(train_loader_inner)
        # 叠加阶段 3 正则（每 epoch 在若干 batch 上计算，控制开销）
        model.train()
        reg_total = 0.0
        n_reg = 0
        max_reg = 2 if args.smoke else 20
        for i, batch in enumerate(train_loader_inner):
            if i >= max_reg:
                break
            frames = batch["frames"].to(device)
            sl, expert = spec_loss(frames)
            cl = cons_loss(frames)
            reg = sl + cl
            if torch.isfinite(reg):
                trainer.optimizer.zero_grad()
                reg.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                trainer.optimizer.step()
                reg_total += reg.item()
                n_reg += 1
        if n_reg > 0:
            logger.info(f"  [阶段3正则] spec+cons 平均={reg_total/n_reg:.4f} (batches={n_reg})")
        return train_loss, train_metrics

    trainer.train_epoch = train_epoch_with_reg

    logger.info("开始训练...")
    if args.smoke:
        config.training.epochs = 1
        trainer.config.training.epochs = 1
        logger.info("[冒烟模式] 仅 1 epoch")
    trainer.train(train_loader, val_loader, resume_from=None)

    close_logger(writer)
    logger.info("训练完成！")


if __name__ == "__main__":
    main()