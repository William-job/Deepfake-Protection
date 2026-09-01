"""生成论文 Table 3 消融实验

基于 best_model_joint_celebdf.pt，在 Celeb-DF++ test 评估 6 项配置：
  1. Full CAR：直接评估（baseline）
  2. -Temporal：推理时将 w[:,0] 置 0 后重新归一化
  3. -Flow：推理时将 w[:,1] 置 0 后重新归一化
  4. -Frequency：推理时将 w[:,2] 置 0 后重新归一化
  5. -Blending：推理时将 w[:,3] 置 0 后重新归一化
  6. Uniform Gating：推理时将 w 强制为 [0.25,0.25,0.25,0.25]

关键：消融在推理时通过修改门控权重实现，不重新训练。
实现：单次前向获取 head_outputs + w，复用专家输出，对每个配置应用 w 修改后重算 logits，
从而每个 batch 只需一次 forward。

阈值策略：对每个消融配置，在 val 集上用 Youden 确定阈值，再在 test 集计算最终指标。
输出 results/table3/ablation_results.json + 各配置 raw_predictions.npz，并计算 ΔAUC。
"""
import argparse
import json
import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
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

# 4 个专家顺序（与 CAR.expert_names 一致）
EXPERT_NAMES = ["temporal", "flow", "frequency", "blending"]

# 6 项消融配置：(配置名, 修改规则)
#   None            -> 不修改
#   "uniform"       -> 强制 [0.25,0.25,0.25,0.25]
#   整数 i          -> 将 w[:,i] 置 0 后重新归一化
ABLATION_CONFIGS = [
    ("full", None),
    ("minus_temporal", 0),
    ("minus_flow", 1),
    ("minus_frequency", 2),
    ("minus_blending", 3),
    ("uniform_gating", "uniform"),
]


def log(msg):
    print(f"[INFO] {msg}")


def disable_train_augmentation(config):
    """评估时禁用 CutMix 等随机增强（运行时修改内存对象，不改动文件）"""
    if "training" in config._data and isinstance(config._data["training"], dict):
        config._data["training"]["cutmix_p"] = 0.0


def build_celebdf_loader(config, split):
    """构建 Celeb-DF++ 评估 loader（num_workers=0，避免 Windows 序列化问题）"""
    config._data["data"]["num_workers"] = 0
    loader = create_dataloader(config, split=split)
    return loader


def load_model(config, checkpoint_path, device):
    """加载 CAR 模型（用 model_state_dict，避免 EMA shadow 缺 stem 键的问题）"""
    model = CAR(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, epoch


def compute_expert_logits(model, head_outputs, batch_size):
    """从 head_outputs 计算各专家 logits，stack 为 (B, 4, 2)

    复刻 CAR.forward 中专家分支逻辑（含 batch 维度对齐与 clamp 防 inf）。
    """
    expert_logits = []
    for name in model.expert_names:
        expert_out = model.experts[name](head_outputs[name])
        if expert_out.dim() == 2 and expert_out.size(0) != batch_size:
            if expert_out.size(0) % batch_size == 0:
                T = expert_out.size(0) // batch_size
                expert_out = expert_out.view(batch_size, T, -1).mean(dim=1)
            else:
                repeat_factor = batch_size // expert_out.size(0)
                if repeat_factor > 0:
                    expert_out = expert_out.repeat(repeat_factor, 1)
        expert_logits.append(expert_out)
    expert_logits = torch.stack(expert_logits, dim=1)  # (B, 4, 2)
    expert_logits = expert_logits.clamp(-100.0, 100.0)
    return expert_logits


def modify_w(w, spec):
    """根据消融规则修改门控权重 w (B, 4)"""
    w = w.clone()
    if spec is None:
        return w
    if spec == "uniform":
        return torch.full_like(w, 0.25)
    # 置 0 指定专家权重后重新归一化
    w[:, spec] = 0.0
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
    return w


@torch.no_grad()
def run_ablation_inference(model, loader, device, desc="Ablation"):
    """单次前向，收集所有消融配置的 preds / labels

    每个 batch 只调用一次 model.forward，复用 head_outputs 与专家输出，
    对 6 个配置分别应用 w 修改并重算 logits，显著降低推理开销。

    返回: dict[config_name] = (preds np.ndarray, labels np.ndarray)
    """
    model.eval()
    collected = {name: {"preds": [], "labels": []} for name, _ in ABLATION_CONFIGS}

    for batch in tqdm(loader, desc=desc, leave=False):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        outputs = model(frames)
        head_outputs = outputs["head_outputs"]
        w = outputs["w"]
        B = w.size(0)

        # 复用专家输出（与 forward 内部一致）
        expert_logits = compute_expert_logits(model, head_outputs, B)  # (B, 4, 2)

        for name, spec in ABLATION_CONFIGS:
            w_mod = modify_w(w, spec)
            y_combined = (w_mod.unsqueeze(-1) * expert_logits).sum(dim=1)  # (B, 2)
            if y_combined.size(1) > 1:
                preds = torch.sigmoid(y_combined[:, 1])
            else:
                preds = torch.sigmoid(y_combined.squeeze(-1))
            collected[name]["preds"].append(preds.cpu().numpy())
            collected[name]["labels"].append(labels.cpu().numpy())

    out = {}
    for name, _ in ABLATION_CONFIGS:
        p = np.concatenate(collected[name]["preds"]).astype(np.float64)
        l = np.concatenate(collected[name]["labels"]).astype(int)
        out[name] = (p, l)
    return out


def metrics_with_threshold(preds, labels, threshold):
    """用指定阈值计算 Acc/F1，其余阈值无关指标直接计算"""
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


def save_npz(path, preds, labels):
    np.savez(path, preds=preds, labels=labels)


def main():
    parser = argparse.ArgumentParser(description="生成论文 Table 3 消融实验")
    parser.add_argument("--config", type=str, default="configs/joint_train.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_v6/best_model_joint_celebdf.pt")
    parser.add_argument("--output_dir", type=str, default="results/table3")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    log(f"Device: {device}")

    config = load_config(args.config)
    disable_train_augmentation(config)
    seed = getattr(config.training, "seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log(f"Loading model from {args.checkpoint}")
    model, ckpt_epoch = load_model(config, args.checkpoint, device)
    log(f"Checkpoint epoch: {ckpt_epoch}")

    # —— val 集确定各消融配置阈值 ——
    log("Building Celeb-DF++ val loader (for threshold)...")
    val_loader = build_celebdf_loader(config, "val")
    log("Running ablation inference on val split...")
    val_results = run_ablation_inference(model, val_loader, device, desc="Val(ablation)")
    thresholds = {}
    for name, (p, l) in val_results.items():
        thresholds[name] = find_optimal_threshold(p, l)
        log(f"  [{name}] val threshold (Youden) = {thresholds[name]:.4f}, "
            f"val AUC = {compute_auc(p, l):.4f}")

    # —— test 集计算最终指标 ——
    log("Building Celeb-DF++ test loader...")
    test_loader = build_celebdf_loader(config, "test")
    log("Running ablation inference on test split...")
    test_results = run_ablation_inference(model, test_loader, device, desc="Test(ablation)")

    full_auc = float(compute_auc(*test_results["full"]))
    ablation_summary = []
    for name, (p, l) in test_results.items():
        thr = thresholds[name]
        m = metrics_with_threshold(p, l, thr)
        m["config"] = name
        m["num_samples"] = int(len(l))
        m["val_threshold"] = float(thr)
        m["delta_auc"] = float(full_auc - m["auc"])  # ΔAUC = AUC_full - AUC_ablation
        # 保存各配置 raw_predictions.npz
        cfg_dir = os.path.join(args.output_dir, name)
        os.makedirs(cfg_dir, exist_ok=True)
        save_npz(os.path.join(cfg_dir, "raw_predictions.npz"), p, l)
        ablation_summary.append(m)
        log(f"  [{name}] test AUC={m['auc']:.4f}  ΔAUC={m['delta_auc']:+.4f}")

    out = {
        "table": "Table 3: Ablation Study",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt_epoch,
        "dataset": "celebdf",
        "split": "test",
        "expert_names": EXPERT_NAMES,
        "ablation_configs": [
            {"name": n, "rule": ("none" if s is None else ("uniform" if s == "uniform" else f"zero_w[{s}]"))}
            for n, s in ABLATION_CONFIGS
        ],
        "full_auc": full_auc,
        "results": ablation_summary,
    }
    out_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"消融结果已保存到 {out_path}")

    # —— 打印表格 ——
    print("\n" + "=" * 90)
    print("Table 3: Ablation Study (Celeb-DF++ test)")
    print("=" * 90)
    header = f"{'Configuration':<20} {'AUC':>8} {'Acc':>8} {'F1':>8} {'AP':>8} {'EER':>8} {'ΔAUC':>8}"
    print(header)
    print("-" * 90)
    for m in ablation_summary:
        print(f"{m['config']:<20} {m['auc']:>8.4f} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
              f"{m['ap']:>8.4f} {m['eer']:>8.4f} {m['delta_auc']:>+8.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
