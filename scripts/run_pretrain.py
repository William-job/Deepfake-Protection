#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 2.3：三层 Artifact-Centric 预训练管线（Level 1→2→3）。

Level 1: Shared Stem 通用视觉预训练（ImageNet，仅 R2/R3 使用）
Level 2: 四专家专属伪任务训练（ControlledSBI，专家轮流、其他冻结）
Level 3: Router 训练（专家冻结后，以 oracle routing distillation 训练门控）

输出：results/pretrain/<R>/（含 pretask 各专家 acc/auc、router loss、最终 val AUC）
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import DeepfakeDataset
from src.data.controlled_sbi import generate_batch
from src.models.car import CAR
from src.training.pretask import ExpertPretask, freeze_except, pretask_loss, pretask_metrics


def set_seed(seed):
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
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)


def build_val_loader(config, batch_size=8):
    ds = DeepfakeDataset(config.data.data_root, split="val",
                         num_frames=config.data.num_frames,
                         frame_stride=config.data.frame_stride,
                         image_size=config.data.image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_fn)


def _freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze(modules):
    for mod in modules:
        for p in mod.parameters():
            p.requires_grad = True


# ---------------- Level 2: 专家伪任务 ----------------
def pretrain_experts(model, config, device, epochs=1, max_batches=None, lr=1e-3, logger=print):
    """四专家轮流做 ControlledSBI 伪任务，返回 {expert: {acc, auc}}。"""
    train_loader = build_train_loader(config)
    results = {}
    for expert_name in ["motion", "temporal", "spectral", "boundary"]:
        logger(f"  [Level2] 开始 {expert_name} ...")
        freeze_except(model, expert_name)
        pretask = ExpertPretask(model, expert_name).to(device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        n_batch = 0
        for epoch in range(epochs):
            for batch in train_loader:
                frames = batch["frames"].to(device)
                B = frames.shape[0]
                # 一半 real（label=0）一半 artifact（label=1）
                art, _ = generate_batch(frames, pretask.artifact_type)
                x = torch.cat([frames, art], dim=0)
                y = torch.cat([torch.zeros(B, dtype=torch.long),
                               torch.ones(B, dtype=torch.long)]).to(device)
                opt.zero_grad()
                logits = pretask(x)
                loss = pretask_loss(logits, y)
                loss.backward()
                opt.step()
                n_batch += 1
                if max_batches and n_batch >= max_batches:
                    break
            if max_batches and n_batch >= max_batches:
                break
        # 评估
        pretask.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for batch in train_loader:
                frames = batch["frames"].to(device)
                B = frames.shape[0]
                art, _ = generate_batch(frames, pretask.artifact_type)
                x = torch.cat([frames, art], dim=0)
                y = torch.cat([torch.zeros(B, dtype=torch.long),
                               torch.ones(B, dtype=torch.long)]).to(device)
                all_logits.append(pretask(x).cpu())
                all_y.append(y.cpu())
                break  # 仅一批评估，控制时长
        logits = torch.cat(all_logits)
        y = torch.cat(all_y)
        acc, auc = pretask_metrics(logits, y)
        results[expert_name] = {"acc": round(acc, 4), "auc": round(auc, 4),
                                "batches": n_batch}
        logger(f"  [Level2] {expert_name:10s} pretask acc={acc:.3f} auc={auc:.3f} (batches={n_batch})")
    return results


# ---------------- Level 3: Router 训练（oracle distillation）----------------
def pretrain_router(model, config, device, epochs=1, max_batches=None, lr=1e-3, logger=print):
    """专家冻结，用 oracle routing distillation 训练门控。

    oracle q：以各专家对样本的"异常度"（fake 概率）归一化得到的路由分布，
    用 KL 蒸馏训练 gating 输出匹配 q。
    """
    _freeze(model)
    _unfreeze([model.gating, model.fusion, model.head_norms])

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    train_loader = build_train_loader(config)
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    n_batch = 0
    total_kl = 0.0
    for epoch in range(epochs):
        for batch in train_loader:
            frames = batch["frames"].to(device)
            B = frames.shape[0]
            # 随机选一类 artifact 施加到一半样本，制造多 artifact 混合分布
            art, types = generate_batch(frames)  # types ∈ {0,1,2,3}
            x = torch.cat([frames, art], dim=0)
            type_idx = torch.cat([types, types], dim=0)  # 每个样本的"主 artifact"

            # 前向（含门控）
            out = model(x)
            z, w_dense = out["z"], out["w_dense"]  # (2B, 4)

            # oracle q：主 artifact 为 one-hot（对应专家应为 1），其余均匀
            # expert_names = ["motion","temporal","spectral","boundary"]
            # artifact types: 0=temporal,1=motion,2=spectral,3=boundary
            art2expert = torch.tensor([1, 0, 2, 3], device=device)  # type->expert idx
            q = torch.zeros_like(w_dense)
            q.scatter_(1, art2expert[type_idx].unsqueeze(1), 1.0)
            q = 0.8 * q + 0.2 * (1.0 / 4.0)  # 平滑
            q = q / q.sum(dim=1, keepdim=True)

            # KL 蒸馏：log w_dense 对 q
            loss = kl(torch.log(w_dense.clamp(min=1e-8)), q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_kl += loss.item()
            n_batch += 1
            if max_batches and n_batch >= max_batches:
                break
        if max_batches and n_batch >= max_batches:
            break
    avg_kl = total_kl / max(1, n_batch)
    logger(f"  [Level3] router KL={avg_kl:.4f} (batches={n_batch})")
    return {"router_kl": round(avg_kl, 4), "batches": n_batch}


# ---------------- 最终 val AUC ----------------
@torch.no_grad()
def eval_val_auc(model, config, device):
    model.eval()
    loader = build_val_loader(config)
    preds, labels = [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        out = model(frames)
        logits = out["logits"]
        p = torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))
        preds.extend(p.cpu().tolist())
        labels.extend(batch["label"].tolist())
        if len(preds) >= 2000:  # 控制时长
            break
    from src.utils.metrics import compute_auc
    return compute_auc(np.array(preds), np.array(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", choices=["R1", "R2", "R3"], required=True,
                    help="R1=随机初始化(无 stem ImageNet)，R2=ImageNet 无 artifact 预训练，R3=完整三层预训练")
    ap.add_argument("--epochs_expert", type=int, default=1)
    ap.add_argument("--epochs_router", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=50, help="调试时限 batch 数（None=全量）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval_val", action="store_true",
                    help="是否在 Level2/3 后评估 val AUC（默认关闭，节省时间）")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device or "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[Pretrain] mode={args.mode} device={device}")

    out_dir = os.path.join("results", "pretrain", args.mode)
    os.makedirs(out_dir, exist_ok=True)

    # Level 1: stem 初始化
    model = CAR(config).to(device)
    if args.mode == "R1":
        # 随机初始化（不加载 ImageNet）：重建一个随机 stem
        from src.models.stem import SharedStem
        model.stem = SharedStem(stem_type=config.model.stem, pretrained=False, freeze=False).to(device)
        print("[Pretrain] R1: 随机初始化 stem（无 ImageNet）")
    elif args.mode == "R2":
        print("[Pretrain] R2: ImageNet stem（跳过 Level 2/3，直接评估）")
    else:
        print("[Pretrain] R3: ImageNet stem + 完整 Level 2/3 预训练")

    record = {"mode": args.mode, "seed": args.seed}

    if args.mode in ("R1", "R3"):
        # Level 2: 专家伪任务
        print(f"[Pretrain] Level 2: 专家伪任务（epochs={args.epochs_expert}, max_batches={args.max_batches}）")
        l2 = pretrain_experts(model, config, device, epochs=args.epochs_expert,
                              max_batches=args.max_batches)
        record["level2"] = l2
        # Level 3: router 训练
        print(f"[Pretrain] Level 3: router 训练（epochs={args.epochs_router}, max_batches={args.max_batches}）")
        l3 = pretrain_router(model, config, device, epochs=args.epochs_router,
                             max_batches=args.max_batches)
        record["level3"] = l3
        # 保存预训练权重
        torch.save({"model_state_dict": model.state_dict(), "mode": args.mode},
                   os.path.join(out_dir, "pretrained.pt"))

    # 最终 val AUC（可选，冒烟测试默认跳过以节省时间）
    if args.eval_val:
        auc = eval_val_auc(model, config, device)
        record["val_auc"] = round(float(auc), 4)
        print(f"[Pretrain] {args.mode} 最终 val AUC = {auc:.4f}")

    with open(os.path.join(out_dir, "pretrain_result.json"), "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[Pretrain] 已写入 {os.path.join(out_dir, 'pretrain_result.json')}", flush=True)


if __name__ == "__main__":
    main()