#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v3 最终模型消融实验（论文 Ablation Table 专用）

与旧 results/ablation_study_v2.json（旧 7.78M 架构）不同，本脚本针对
**最终论文模型 CAR-v3**（5.19M，results/final_car_v3/checkpoints/best_model.pt）。

协议：
- 全量 test 集 5418（复用 robustness_honest 的 uint8 帧缓存，免视频解码）
- 每个变体报告 AUC / AP（阈值无关指标）；主指标 AUC
- 变体：
    full          完整 CAR-v3（与 robustness_honest clean 对账）
    -motion       零化 motion 门控权重后重归一（边缘贡献）
    -temporal     同上
    -spectral     同上
    -boundary     同上
    uniform       门控权重固定 1/4（检验学习路由 vs 均匀 ensemble）
    only_spectral 仅保留最强单专家（单专家上限参照）

用法：
    python -u scripts/ablation_v3.py
输出：
    results/ablation_v3/ablation.json
"""
import json
import os
import random
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config
from src.models.car import CAR
from src.utils.metrics import compute_auc, compute_ap

CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "ablation_v3")

EXPERT_NAMES = ["motion", "temporal", "spectral", "boundary"]  # 与 car.expert_names 一致


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class GatedVariant(nn.Module):
    """在 forward 后修改门控权重并重算组合 logits。

    mode:
      "full"            不修改
      "drop:<idx>"      零化第 idx 个专家权重后重归一
      "uniform"         权重固定 1/4
      "only:<idx>"      仅保留第 idx 个专家（权重 1）
    """
    def __init__(self, base, mode):
        super().__init__()
        self.base = base
        self.mode = mode

    def forward(self, x):
        out = self.base.forward(x)
        head_outputs = out["head_outputs"]
        z = out["z"]
        logit_list = []
        for name in self.base.expert_names:
            expert_out = self.base.experts[name](head_outputs[name])
            if expert_out.dim() == 2 and expert_out.size(0) != z.size(0):
                if expert_out.size(0) % z.size(0) == 0:
                    T = expert_out.size(0) // z.size(0)
                    expert_out = expert_out.view(z.size(0), T, -1).mean(dim=1)
                else:
                    repeat_factor = z.size(0) // expert_out.size(0)
                    if repeat_factor > 0:
                        expert_out = expert_out.repeat(repeat_factor, 1)
            logit_list.append(expert_out)
        stacked = torch.stack(logit_list, dim=1).clamp(-100.0, 100.0)

        w = out["w"]
        if self.mode == "full":
            pass
        elif self.mode == "uniform":
            w = torch.full_like(w, 0.25)
        elif self.mode.startswith("drop:"):
            idx = int(self.mode.split(":")[1])
            w = w.clone()
            w[:, idx] = 0
            w_sum = w.sum(dim=1, keepdim=True)
            w_sum = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
            w = w / w_sum
        elif self.mode.startswith("only:"):
            idx = int(self.mode.split(":")[1])
            w = torch.zeros_like(w)
            w[:, idx] = 1.0
        else:
            raise ValueError(self.mode)

        y_combined = (w.unsqueeze(-1) * stacked).sum(dim=1)
        return {"logits": y_combined}


@torch.no_grad()
def eval_variant(model, frames_mmap, labels, device, batch_size=16):
    n = frames_mmap.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    for i in range(0, n, batch_size):
        batch_u8 = np.asarray(frames_mmap[i:i + batch_size])
        x = torch.from_numpy(batch_u8).permute(0, 1, 4, 2, 3).float()
        x = (x / 255.0 - 0.5) / 0.5
        logits = model(x.to(device))["logits"]
        p = torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))
        preds[i:i + batch_u8.shape[0]] = p.cpu().numpy()
    return {
        "auc": float(compute_auc(preds, labels)),
        "ap": float(compute_ap(preds, labels)),
    }


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    log(f"Device: {device}")

    frames_path = os.path.join(CACHE_DIR, "test_frames_u8.npy")
    meta_path = os.path.join(CACHE_DIR, "test_meta.json")
    assert os.path.exists(frames_path), "先运行 robustness_honest.py 生成 test 帧缓存"
    frames = np.load(frames_path, mmap_mode="r")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    labels = np.array(meta["labels"])
    log(f"test 缓存: {frames.shape[0]} 样本")

    ckpt_path = os.path.join(PROJECT_ROOT, "results", "final_car_v3", "checkpoints", "best_model.pt")
    model = CAR(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("ema_shadow") is not None:
        try:
            model.load_state_dict(ckpt["ema_shadow"], strict=True)
        except RuntimeError:
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    variants = [("full", "Full CAR-v3")]
    for i, name in enumerate(EXPERT_NAMES):
        variants.append((f"drop:{i}", f"-{name}"))
    variants.append(("uniform", "Uniform gating (1/4)"))
    variants.append(("only:2", "Only spectral"))

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for mode, display in variants:
        wrapper = GatedVariant(model, mode).to(device)
        m = eval_variant(wrapper, frames, labels, device)
        results[display] = {"mode": mode, **m}
        log(f"{display:<28} AUC={m['auc']:.4f}  AP={m['ap']:.4f}")
        del wrapper

    baseline = results["Full CAR-v3"]["auc"]
    out = {
        "checkpoint": ckpt_path,
        "protocol": "full test set 5418, uint8 cache, threshold-free metrics (AUC/AP)",
        "baseline_auc": baseline,
        "results": results,
        "delta": {k: v["auc"] - baseline for k, v in results.items()},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT_DIR, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log("已保存: results/ablation_v3/ablation.json")
    log("---- 摘要（ΔAUC vs Full） ----")
    for k, d in out["delta"].items():
        log(f"  {k:<28} Δ={d:+.4f}")


if __name__ == "__main__":
    main()
