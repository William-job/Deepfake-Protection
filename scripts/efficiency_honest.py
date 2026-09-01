#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诚实效率对比（论文 Efficiency Table 专用）

⚠️ FPS 计时必须独占 GPU——在其他 GPU 任务全部结束后运行。

对比对象（同一输入协议 (1,8,3,224,224)，与效率叙事一致）：
- CAR-v3（5.19M 新架构，最终论文模型）
- EfficientNet-B0 / Xception / MesoNet（seed_42 诚实基线同款架构）

指标：参数量（分组分解）、FLOPs（thop）、GPU/CPU 推理速度、模型体积。

用法：
    python -u scripts/efficiency_honest.py
输出：
    results/efficiency_honest/efficiency.json + 控制台对比表
"""
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from src.config import load_config
from src.models.car import CAR
from src.utils.metrics import compute_auc  # noqa: F401（保持 import 结构一致）

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "efficiency_honest")

MODEL_SPECS = {
    "car": {"type": "car", "display": "CAR (ours)",
            "ckpt": os.path.join(PROJECT_ROOT, "results", "final_car_v3", "checkpoints", "best_model.pt")},
    "efficientnet_b0": {"type": "baseline", "display": "EfficientNet-B0",
                        "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "efficientnet_b0", "seed_42", "best_model.pt")},
    "xception": {"type": "baseline", "display": "Xception",
                 "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "xception", "seed_42", "best_model.pt")},
    "mesonet": {"type": "baseline", "display": "MesoNet",
                "ckpt": os.path.join(PROJECT_ROOT, "results", "baseline_honest", "mesonet", "seed_42", "best_model.pt")},
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(name, spec, config, device):
    if spec["type"] == "car":
        model = CAR(config).to(device)
        ckpt = torch.load(spec["ckpt"], map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("ema_shadow") is not None:
            try:
                model.load_state_dict(ckpt["ema_shadow"], strict=True)
            except RuntimeError:
                model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        from baseline_full import build_model
        model = build_model(name, config.data.num_frames).to(device)
        ckpt = torch.load(spec["ckpt"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def param_breakdown(model):
    total = sum(p.numel() for p in model.parameters())
    groups = {}
    for n, p in model.named_parameters():
        key = None
        for g in ["stem", "motion_head", "temporal_head", "spectral_head", "boundary_head",
                  "experts", "fusion", "gating", "head_norms", "preprocessor"]:
            if g in n:
                key = g
                break
        if key is None:
            if "backbone" in n:
                key = "backbone"
            else:
                key = "other"
        groups[key] = groups.get(key, 0) + p.numel()
    return total, groups


def measure_flops(model, device):
    from thop import profile
    dummy = torch.randn(1, 8, 3, 224, 224).to(device)
    try:
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return float(flops), float(params), None
    except Exception as e:
        return None, None, str(e)


def measure_fps(model, device, warmup=10, runs=100):
    dummy = torch.randn(1, 8, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
    avg = (time.time() - t0) / runs
    return 1.0 / avg, avg * 1000.0


def measure_cpu_fps(model, warmup=3, runs=10):
    m = model.to("cpu")
    dummy = torch.randn(1, 8, 3, 224, 224)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        t0 = time.time()
        for _ in range(runs):
            _ = model(dummy)
    avg = (time.time() - t0) / runs
    model.to("cuda")
    return 1.0 / avg, avg * 1000.0


def main():
    config = load_config("configs/default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    assert device == "cuda", "FPS 计时需要 GPU；且需独占（其他 GPU 任务结束后再跑）"
    os.makedirs(OUT_DIR, exist_ok=True)

    dummy = torch.randn(1, 8, 3, 224, 224).to(device)
    results = {}

    for name, spec in MODEL_SPECS.items():
        if not os.path.exists(spec["ckpt"]):
            log(f"[SKIP] {name}: checkpoint 不存在")
            continue
        log(f"===== {spec['display']} =====")
        model = load_model(name, spec, config, device)

        total, groups = param_breakdown(model)
        flops, params_thop, err = measure_flops(model, device)
        gpu_fps, gpu_ms = measure_fps(model, device)
        cpu_fps, cpu_ms = measure_cpu_fps(model)

        rec = {
            "display": spec["display"],
            "checkpoint": spec["ckpt"],
            "parameters": {
                "total": total,
                "total_M": round(total / 1e6, 3),
                "groups": {k: v for k, v in sorted(groups.items(), key=lambda kv: -kv[1])},
            },
            "flops": {
                "flops_G": round(flops / 1e9, 3) if flops else None,
                "params_from_thop_M": round(params_thop / 1e6, 3) if params_thop else None,
                "error": err,
                "input_shape": [1, 8, 3, 224, 224],
            },
            "speed": {
                "gpu_fps": round(gpu_fps, 1),
                "gpu_latency_ms": round(gpu_ms, 2),
                "cpu_fps": round(cpu_fps, 2),
                "cpu_latency_ms": round(cpu_ms, 1),
            },
            "model_size": {
                "fp32_MB": round(total * 4 / 1024 / 1024, 2),
                "int8_MB_est": round(total * 4 / 1024 / 1024 / 4, 2),
            },
        }
        results[name] = rec
        log(f"  Params: {total / 1e6:.2f} M | FLOPs: {flops / 1e9:.2f} G | "
            f"GPU: {gpu_fps:.1f} FPS ({gpu_ms:.1f} ms) | CPU: {cpu_fps:.2f} FPS")
        del model
        torch.cuda.empty_cache()

    results["hardware"] = {
        "platform": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": "FPS 为 batch=1、8 帧输入、独占 GPU 下测量",
    }

    out_path = os.path.join(OUT_DIR, "efficiency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 控制台对比表
    log("\n" + "=" * 78)
    log(f"{'Model':<20}{'Params(M)':>10}{'FLOPs(G)':>10}{'GPU FPS':>10}{'CPU FPS':>10}{'FP32(MB)':>10}")
    log("-" * 78)
    for name, r in results.items():
        if name == "hardware":
            continue
        log(f"{r['display']:<20}{r['parameters']['total_M']:>10.2f}"
            f"{(r['flops']['flops_G'] or 0):>10.2f}{r['speed']['gpu_fps']:>10.1f}"
            f"{r['speed']['cpu_fps']:>10.2f}{r['model_size']['fp32_MB']:>10.1f}")
    log("=" * 78)
    log(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
