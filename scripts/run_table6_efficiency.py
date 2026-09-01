"""生成论文 Table 6 效率分析

输出内容：
  1. 参数量分解：按前缀分组（stem/各 head/experts/fusion/gating/difficulty_estimator 等）
  2. FLOPs：用 thop.profile 计算，输入 dummy [1,8,3,224,224]，记录 thop 版本
  3. FPS 测量：GPU 预热 100 次后取 1000 次平均；CPU 取 100 次（避免过慢）
  4. 模型大小：FP32 state_dict 落盘测大小；INT8 用 quantize_dynamic 量化后测大小
  5. 硬件规格：torch.cuda.get_device_name(0) + platform.processor()

输出 results/table6/efficiency_results.json
"""
import argparse
import json
import os
import platform
import sys
import tempfile
import time

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.config import load_config
from src.models.car import CAR


def log(msg):
    print(f"[INFO] {msg}")


def count_parameters(model):
    """返回 (total, trainable) 参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parameter_breakdown(model):
    """按模块前缀分组统计参数量

    分组顺序对应论文 Table 6 的组件分解。
    """
    groups = {
        "stem": 0,
        "temporal_head": 0,
        "flow_head": 0,
        "frequency_head": 0,
        "blending_head": 0,
        "experts": 0,
        "fusion": 0,
        "gating": 0,
        "difficulty_estimator": 0,
        "head_norms": 0,
        "preprocessor": 0,
        "other": 0,
    }
    detail = {}
    for name, p in model.named_parameters():
        numel = p.numel()
        matched = None
        for key in groups:
            if key == "other":
                continue
            if name.startswith(key + ".") or name.startswith(key):
                matched = key
                break
        if matched is None:
            matched = "other"
        groups[matched] += numel
        detail[name] = numel
    return groups, detail


def measure_flops(model, device):
    """用 thop.profile 计算 FLOPs 与参数量，返回 (flops, params, version)"""
    try:
        from thop import profile
        try:
            import thop
            version = getattr(thop, "__version__", "unknown")
        except Exception:
            version = "unknown"
        dummy = torch.randn(1, 8, 3, 224, 224).to(device)
        model_eval = model.to(device).eval()
        with torch.no_grad():
            flops, params = profile(model_eval, inputs=(dummy,), verbose=False)
        return float(flops), float(params), version, None
    except Exception as e:
        return None, None, None, str(e)


def measure_fps(model, device, warmup, runs):
    """测量单 batch（batch_size=1）推理 FPS"""
    model_eval = model.to(device).eval()
    dummy = torch.randn(1, 8, 3, 224, 224).to(device)
    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            _ = model_eval(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = model_eval(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
        t1 = time.time()
    elapsed = t1 - t0
    if elapsed <= 0:
        return None, None
    avg_ms = elapsed / runs * 1000.0
    fps = runs / elapsed
    return fps, avg_ms


def measure_model_size(model):
    """测量 FP32 与 INT8 模型落盘大小（字节）

    FP32：保存完整 state_dict 到临时文件。
    INT8：用 torch.quantization.quantize_dynamic 对 {nn.Linear} 量化后保存。
    """
    # FP32
    fp32_size = None
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        fp32_path = tmp.name
    try:
        torch.save(model.state_dict(), fp32_path)
        fp32_size = os.path.getsize(fp32_path)
    finally:
        if os.path.exists(fp32_path):
            os.remove(fp32_path)

    # INT8 动态量化（仅 CPU，且只对 Linear 生效）
    int8_size = None
    int8_note = None
    try:
        model_cpu = model.to("cpu").eval()
        qmodel = torch.quantization.quantize_dynamic(
            model_cpu, {nn.Linear}, dtype=torch.qint8
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            int8_path = tmp.name
        try:
            torch.save(qmodel.state_dict(), int8_path)
            int8_size = os.path.getsize(int8_path)
        finally:
            if os.path.exists(int8_path):
                os.remove(int8_path)
    except Exception as e:
        int8_note = str(e)

    return fp32_size, int8_size, int8_note


def get_hardware_info():
    """获取硬件规格信息"""
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        except Exception:
            pass
    return info


def main():
    parser = argparse.ArgumentParser(description="生成论文 Table 6 效率分析")
    parser.add_argument("--config", type=str, default="configs/joint_train.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v6/best_model_joint_celebdf.pt")
    parser.add_argument("--output_dir", type=str, default="results/table6")
    parser.add_argument("--fps_runs_gpu", type=int, default=1000)
    parser.add_argument("--fps_runs_cpu", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log("Loading config and building CAR model...")
    config = load_config(args.config)
    model = CAR(config)
    total, trainable = count_parameters(model)
    log(f"Total params: {total:,}  Trainable: {trainable:,}")

    # 1. 参数量分解
    log("Computing parameter breakdown...")
    groups, detail = parameter_breakdown(model)
    log("Parameter groups:")
    for k, v in groups.items():
        if v > 0:
            log(f"  {k:<22} {v:,}")

    # 2. FLOPs
    log("Measuring FLOPs via thop...")
    device_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    flops, thop_params, thop_version, flops_err = measure_flops(model, device_cuda)
    if flops is not None:
        log(f"FLOPs: {flops / 1e9:.2f} G  (thop {thop_version})")
    else:
        log(f"FLOPs measurement failed: {flops_err}")

    # 3. FPS
    fps_results = {}
    if torch.cuda.is_available():
        log(f"Measuring FPS on GPU (warmup=100, runs={args.fps_runs_gpu})...")
        fps_gpu, ms_gpu = measure_fps(model, "cuda", warmup=100, runs=args.fps_runs_gpu)
        fps_results["gpu"] = {"fps": fps_gpu, "latency_ms": ms_gpu, "runs": args.fps_runs_gpu}
        log(f"  GPU FPS: {fps_gpu:.2f}  ({ms_gpu:.2f} ms)")
    else:
        log("CUDA 不可用，跳过 GPU FPS 测量")
        fps_results["gpu"] = None

    log(f"Measuring FPS on CPU (warmup=10, runs={args.fps_runs_cpu})...")
    fps_cpu, ms_cpu = measure_fps(model, "cpu", warmup=10, runs=args.fps_runs_cpu)
    fps_results["cpu"] = {"fps": fps_cpu, "latency_ms": ms_cpu, "runs": args.fps_runs_cpu}
    log(f"  CPU FPS: {fps_cpu:.2f}  ({ms_cpu:.2f} ms)")

    # 4. 模型大小
    log("Measuring model size (FP32 / INT8)...")
    fp32_size, int8_size, int8_note = measure_model_size(model)
    log(f"  FP32 size: {fp32_size / (1024 * 1024):.2f} MB")
    if int8_size is not None:
        log(f"  INT8 size: {int8_size / (1024 * 1024):.2f} MB")
    else:
        log(f"  INT8 quantization failed: {int8_note}")

    # 5. 硬件规格
    hw = get_hardware_info()
    log(f"Hardware: {hw}")

    # 汇总
    def mb(b):
        return round(b / (1024 * 1024), 4) if b is not None else None

    efficiency = {
        "table": "Table 6: Efficiency Analysis",
        "checkpoint": args.checkpoint,
        "parameters": {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "total_M": round(total / 1e6, 4),
            "groups": groups,
        },
        "flops": {
            "flops": flops,
            "flops_G": round(flops / 1e9, 4) if flops is not None else None,
            "params_from_thop": thop_params,
            "thop_version": thop_version,
            "error": flops_err,
            "input_shape": [1, 8, 3, 224, 224],
        },
        "fps": fps_results,
        "model_size": {
            "fp32_bytes": fp32_size,
            "fp32_MB": mb(fp32_size),
            "int8_bytes": int8_size,
            "int8_MB": mb(int8_size),
            "int8_error": int8_note,
            "compression_ratio": round(fp32_size / int8_size, 4) if (fp32_size and int8_size) else None,
        },
        "hardware": hw,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = os.path.join(args.output_dir, "efficiency_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(efficiency, f, indent=2, ensure_ascii=False)
    log(f"效率分析结果已保存到 {out_path}")

    # 打印摘要表格
    print("\n" + "=" * 70)
    print("Table 6: Efficiency Analysis")
    print("=" * 70)
    print(f"  Total params:    {total / 1e6:.2f} M")
    print(f"  FLOPs:           {flops / 1e9:.2f} G" if flops else "  FLOPs:           N/A")
    if fps_results.get("gpu"):
        print(f"  GPU FPS:         {fps_results['gpu']['fps']:.2f}")
    print(f"  CPU FPS:         {fps_cpu:.2f}")
    print(f"  FP32 size:       {mb(fp32_size)} MB")
    if int8_size:
        print(f"  INT8 size:       {mb(int8_size)} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
