"""生成论文 Table 5 鲁棒性评估

使用 best_model_joint_celebdf.pt，在 Celeb-DF++ test 评估。

扰动类型与级别：
  - JPEG：Q30, Q50, Q70, Q90
  - 高斯噪声：σ=0.01, 0.02, 0.05
  - 高斯模糊：k=3, 5, 7
  - 亮度：factor=0.8, 1.0, 1.2
  - H.264：CRF 23, 28, 35（视频级扰动，需重新编码视频，标注 N/A 并记录原因）

实现：在推理循环中对 batch frames 应用扰动（帧在 [-1,1] 归一化空间，
扰动前反归一化到 [0,1]/uint8，扰动后重新归一化）。
每种扰动级别独立计算 AUC + ΔAUC = AUC_clean - AUC_perturbed。

阈值策略：clean val 集确定阈值后应用到所有扰动 test 集（鲁棒性评估中不针对扰动重调阈值）。
扰动实现代码同时保存到 results/table5/perturbation_code.py。
输出 results/table5/robustness_results.json
"""
import argparse
import json
import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import load_config
from src.data.dataloader import create_dataloader
from src.models.car import CAR
from src.utils.metrics import (
    compute_accuracy,
    compute_auc,
    compute_ap,
    compute_eer,
    compute_f1,
    compute_tpr_at_fpr,
    find_optimal_threshold,
)


def log(msg):
    print(f"[INFO] {msg}")


# ============================================================================
# 扰动函数（这些函数的源码会被复制到 results/table5/perturbation_code.py）
# 输入 frames: (B, T, C, H, W) 在 [-1,1] 归一化空间
# ============================================================================

def _denorm(t):
    """[-1,1] -> [0,1]"""
    return t * 0.5 + 0.5


def _renorm(t):
    """[0,1] -> [-1,1]"""
    return (t - 0.5) / 0.5


def perturb_jpeg(frames, quality):
    """JPEG 压缩：quality 越低压缩越强。需要 uint8 + cv2 逐帧编码/解码"""
    device = frames.device
    x = frames.detach().cpu()
    x = _denorm(x)
    x = (x * 255.0).clamp(0, 255).to(torch.uint8)  # (B,T,C,H,W)
    B, T, C, H, W = x.shape
    out = torch.empty_like(x)
    for b in range(B):
        for t in range(T):
            img = x[b, t].permute(1, 2, 0).numpy()  # HWC, uint8
            enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1]
            dec = cv2.imdecode(enc, 1)  # HWC
            out[b, t] = torch.from_numpy(dec).permute(2, 0, 1)
    out = out.float() / 255.0
    return _renorm(out).to(device)


def perturb_gaussian_noise(frames, sigma):
    """高斯噪声：sigma 为 [0,1] 像素空间标准差"""
    x = _denorm(frames)
    x = (x + torch.randn_like(x) * float(sigma)).clamp(0, 1)
    return _renorm(x)


def _gaussian_kernel2d(k, sigma=None):
    if sigma is None:
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k2d = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    k2d = k2d / k2d.sum()
    return k2d


def perturb_gaussian_blur(frames, k):
    """高斯模糊：kernel 大小 k（奇数）。深度可分离卷积实现"""
    device = frames.device
    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = _gaussian_kernel2d(k).to(device)  # (k,k)
    x = frames  # (B,T,C,H,W)
    B, T, C, H, W = x.shape
    x = x.view(B * T, C, H, W)
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    # depthwise conv: 每通道一个 kernel
    out = F.conv2d(x, kernel.expand(C, 1, k, k), groups=C)
    out = out.view(B, T, C, H, W)
    return out


def perturb_brightness(frames, factor):
    """亮度调整：factor>1 变亮，factor<1 变暗"""
    x = _denorm(frames)
    x = (x * float(factor)).clamp(0, 1)
    return _renorm(x)


# 扰动配置表：(display_name, type, level)
PERTURBATIONS = [
    ("JPEG Q90", "jpeg", 90),
    ("JPEG Q70", "jpeg", 70),
    ("JPEG Q50", "jpeg", 50),
    ("JPEG Q30", "jpeg", 30),
    ("Noise sigma=0.01", "noise", 0.01),
    ("Noise sigma=0.02", "noise", 0.02),
    ("Noise sigma=0.05", "noise", 0.05),
    ("Blur k=3", "blur", 3),
    ("Blur k=5", "blur", 5),
    ("Blur k=7", "blur", 7),
    ("Brightness 0.8", "brightness", 0.8),
    ("Brightness 1.0", "brightness", 1.0),
    ("Brightness 1.2", "brightness", 1.2),
    # H.264 视频级扰动：需重新编码视频，标注 N/A
    ("H.264 CRF=23", "h264", 23),
    ("H.264 CRF=28", "h264", 28),
    ("H.264 CRF=35", "h264", 35),
]

PERTURB_FUNCS = {
    "jpeg": perturb_jpeg,
    "noise": perturb_gaussian_noise,
    "blur": perturb_gaussian_blur,
    "brightness": perturb_brightness,
}


def apply_perturbation(frames, ptype, level):
    """根据类型应用扰动；H.264 不支持帧级实现，返回 None"""
    if ptype == "h264":
        return None  # requires video re-encoding, skipped
    return PERTURB_FUNCS[ptype](frames, level)


def disable_train_augmentation(config):
    if "training" in config._data and isinstance(config._data["training"], dict):
        config._data["training"]["cutmix_p"] = 0.0


def build_celebdf_loader(config, split):
    config._data["data"]["num_workers"] = 0
    return create_dataloader(config, split=split)


def load_model(config, checkpoint_path, device):
    model = CAR(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, epoch


@torch.no_grad()
def infer_with_perturbation(model, loader, device, ptype=None, level=None, desc="Eval"):
    """对 loader 推理；若指定扰动则对每个 batch 的 frames 应用扰动"""
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        if ptype is not None:
            frames = apply_perturbation(frames, ptype, level)
            if frames is None:
                return None, None  # 扰动不支持
        outputs = model(frames)
        logits = outputs["logits"]
        if logits.size(1) > 1:
            preds = torch.sigmoid(logits[:, 1])
        else:
            preds = torch.sigmoid(logits.squeeze(-1))
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds).astype(np.float64)
    labels = np.concatenate(all_labels).astype(int)
    return preds, labels


def metrics_with_threshold(preds, labels, threshold):
    return {
        "accuracy": float(compute_accuracy(preds, labels, threshold)),
        "auc": float(compute_auc(preds, labels)),
        "ap": float(compute_ap(preds, labels)),
        "f1": float(compute_f1(preds, labels, threshold)),
        "threshold": float(threshold),
        "eer": float(compute_eer(preds, labels)),
        "tpr_at_fpr_1": float(compute_tpr_at_fpr(preds, labels, 0.01)),
        "tpr_at_fpr_01": float(compute_tpr_at_fpr(preds, labels, 0.001)),
    }


def save_perturbation_code(path):
    """将扰动函数源码复制到 results/table5/perturbation_code.py"""
    code = '''"""Table 5 鲁棒性评估使用的帧级扰动函数（从 run_table5_robustness.py 复制）

输入 frames: (B, T, C, H, W) 在 [-1,1] 归一化空间
依赖: cv2, numpy, torch, torch.nn.functional
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _denorm(t):
    """[-1,1] -> [0,1]"""
    return t * 0.5 + 0.5


def _renorm(t):
    """[0,1] -> [-1,1]"""
    return (t - 0.5) / 0.5


def perturb_jpeg(frames, quality):
    """JPEG 压缩：quality 越低压缩越强。需要 uint8 + cv2 逐帧编码/解码"""
    device = frames.device
    x = frames.detach().cpu()
    x = _denorm(x)
    x = (x * 255.0).clamp(0, 255).to(torch.uint8)
    B, T, C, H, W = x.shape
    out = torch.empty_like(x)
    for b in range(B):
        for t in range(T):
            img = x[b, t].permute(1, 2, 0).numpy()
            enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1]
            dec = cv2.imdecode(enc, 1)
            out[b, t] = torch.from_numpy(dec).permute(2, 0, 1)
    out = out.float() / 255.0
    return _renorm(out).to(device)


def perturb_gaussian_noise(frames, sigma):
    """高斯噪声：sigma 为 [0,1] 像素空间标准差"""
    x = _denorm(frames)
    x = (x + torch.randn_like(x) * float(sigma)).clamp(0, 1)
    return _renorm(x)


def _gaussian_kernel2d(k, sigma=None):
    if sigma is None:
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k2d = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    k2d = k2d / k2d.sum()
    return k2d


def perturb_gaussian_blur(frames, k):
    """高斯模糊：kernel 大小 k（奇数）。深度可分离卷积实现"""
    device = frames.device
    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = _gaussian_kernel2d(k).to(device)
    x = frames
    B, T, C, H, W = x.shape
    x = x.view(B * T, C, H, W)
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(x, kernel.expand(C, 1, k, k), groups=C)
    out = out.view(B, T, C, H, W)
    return out


def perturb_brightness(frames, factor):
    """亮度调整：factor>1 变亮，factor<1 变暗"""
    x = _denorm(frames)
    x = (x * float(factor)).clamp(0, 1)
    return _renorm(x)
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)


def main():
    parser = argparse.ArgumentParser(description="生成论文 Table 5 鲁棒性评估")
    parser.add_argument("--config", type=str, default="configs/joint_train.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_v6/best_model_joint_celebdf.pt")
    parser.add_argument("--output_dir", type=str, default="results/table5")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    log(f"Device: {device}")

    config = load_config(args.config)
    disable_train_augmentation(config)
    np.random.seed(getattr(config.training, "seed", 42))
    torch.manual_seed(getattr(config.training, "seed", 42))

    log(f"Loading model from {args.checkpoint}")
    model, ckpt_epoch = load_model(config, args.checkpoint, device)
    log(f"Checkpoint epoch: {ckpt_epoch}")

    # —— clean val 定阈值 ——
    log("Building Celeb-DF++ val loader (for threshold)...")
    val_loader = build_celebdf_loader(config, "val")
    log("Inferring on clean val split...")
    val_preds, val_labels = infer_with_perturbation(model, val_loader, device, desc="Val(clean)")
    threshold = find_optimal_threshold(val_preds, val_labels) if len(np.unique(val_labels)) >= 2 else 0.5
    log(f"Clean val threshold (Youden) = {threshold:.4f}, val AUC = {compute_auc(val_preds, val_labels):.4f}")

    # —— clean baseline ——
    log("Building Celeb-DF++ test loader...")
    test_loader = build_celebdf_loader(config, "test")
    log("Inferring on clean test split (baseline)...")
    clean_preds, clean_labels = infer_with_perturbation(model, test_loader, device, desc="Test(clean)")
    baseline_metrics = metrics_with_threshold(clean_preds, clean_labels, threshold)
    baseline_auc = baseline_metrics["auc"]
    baseline_metrics["num_samples"] = int(len(clean_labels))
    log(f"Clean baseline AUC = {baseline_auc:.4f}")

    # 保存 baseline raw predictions
    np.savez(os.path.join(args.output_dir, "baseline_raw_predictions.npz"),
             preds=clean_preds, labels=clean_labels)

    # —— 各扰动 ——
    perturbation_results = []
    for display, ptype, level in PERTURBATIONS:
        log(f"\nPerturbation: {display} (type={ptype}, level={level})")
        if ptype == "h264":
            # 视频级扰动，需重新编码视频，跳过
            rec = {
                "name": display,
                "type": ptype,
                "level": level,
                "status": "skipped",
                "note": "requires video re-encoding, skipped",
                "auc": None,
                "delta_auc": None,
            }
            perturbation_results.append(rec)
            log(f"  [SKIP] {rec['note']}")
            continue

        preds, labels = infer_with_perturbation(
            model, test_loader, device, ptype=ptype, level=level, desc=f"Test({display})"
        )
        if preds is None:
            perturbation_results.append({
                "name": display, "type": ptype, "level": level,
                "status": "failed", "auc": None, "delta_auc": None,
            })
            continue

        m = metrics_with_threshold(preds, labels, threshold)
        m["num_samples"] = int(len(labels))
        m["name"] = display
        m["type"] = ptype
        m["level"] = level
        m["status"] = "ok"
        m["delta_auc"] = float(baseline_auc - m["auc"])  # ΔAUC = AUC_clean - AUC_perturbed
        perturbation_results.append(m)
        log(f"  AUC={m['auc']:.4f}  ΔAUC={m['delta_auc']:+.4f}")

    # —— 保存扰动函数源码 ——
    pert_code_path = os.path.join(args.output_dir, "perturbation_code.py")
    save_perturbation_code(pert_code_path)
    log(f"扰动函数源码已保存到 {pert_code_path}")

    out = {
        "table": "Table 5: Robustness Evaluation",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt_epoch,
        "dataset": "celebdf",
        "split": "test",
        "threshold": threshold,
        "threshold_source": "clean_val(youden)",
        "baseline": baseline_metrics,
        "perturbations": perturbation_results,
    }
    out_path = os.path.join(args.output_dir, "robustness_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"鲁棒性结果已保存到 {out_path}")

    # —— 打印表格 ——
    print("\n" + "=" * 90)
    print("Table 5: Robustness Evaluation (Celeb-DF++ test)")
    print("=" * 90)
    print(f"{'Perturbation':<25} {'AUC':>8} {'ΔAUC':>8} {'Acc':>8} {'F1':>8} {'EER':>8} {'Status':>10}")
    print("-" * 90)
    print(f"{'Clean (baseline)':<25} {baseline_auc:>8.4f} {0.0:>+8.4f} "
          f"{baseline_metrics['accuracy']:>8.4f} {baseline_metrics['f1']:>8.4f} "
          f"{baseline_metrics['eer']:>8.4f} {'ok':>10}")
    for r in perturbation_results:
        auc = f"{r['auc']:.4f}" if r["auc"] is not None else "N/A"
        delta = f"{r['delta_auc']:+.4f}" if r["delta_auc"] is not None else "N/A"
        acc = f"{r.get('accuracy', 0):.4f}" if r.get("accuracy") is not None else "N/A"
        f1 = f"{r.get('f1', 0):.4f}" if r.get("f1") is not None else "N/A"
        eer = f"{r.get('eer', 0):.4f}" if r.get("eer") is not None else "N/A"
        print(f"{r['name']:<25} {auc:>8} {delta:>8} {acc:>8} {f1:>8} {eer:>8} {r['status']:>10}")
    print("=" * 90)


if __name__ == "__main__":
    main()
