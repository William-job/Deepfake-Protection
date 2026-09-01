"""生成论文 Table 4 跨数据集泛化

4 个评估配置（阈值在源数据集 val 上确定，泛化到目标 test 集）：
  1. Celeb-DF++ 模型 → FF++ c23 test（跨数据集）
  2. Celeb-DF++ 模型 → FF++ c40 test（跨数据集 + 跨压缩）
  3. FF++ 模型 → Celeb-DF++ test（跨数据集）
  4. FF++ 微调模型（best_model_ffpp.pt, 灾难性遗忘版）→ Celeb-DF++ test

Per-Method breakdown：FF++ test 时按 method（Deepfakes/Face2Face/FaceSwap/NeuralTextures）
分组评估，每组用 real 样本 + 该 method 的 fake 样本计算 AUC。

灾难性遗忘：对比 FF++ 微调前（best_model_joint_celebdf.pt）后（best_model_ffpp.pt）
在 Celeb-DF++ test 的 AUC 下降。

输出 results/table4/cross_dataset_results.json
"""
import argparse
import json
import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.dataloader import create_dataloader
from src.data.ff_frame_dataset import FFFrameDataset
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

# FF++ 的 4 种篡改方法
FF_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(f"[INFO] {msg}")


def disable_train_augmentation(config):
    if "training" in config._data and isinstance(config._data["training"], dict):
        config._data["training"]["cutmix_p"] = 0.0


class FFEvalCollate:
    """FF++ 评估 collate：堆叠帧/标签，保留 frame_dir 与 method"""

    def __call__(self, batch):
        return {
            "frames": torch.stack([b["frames"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "frame_dir": [b["frame_dir"] for b in batch],
            "method": [b.get("method") for b in batch],
        }


def load_model(config, checkpoint_path, device):
    """加载 CAR 模型（用 model_state_dict，更可靠）"""
    model = CAR(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, epoch


def build_celebdf_loader(config, split):
    config._data["data"]["num_workers"] = 0
    return create_dataloader(config, split=split)


def build_ffpp_loader(config, split, compression):
    ds = FFFrameDataset(
        data_root=config.data.ff_root,
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
        split=split,
        compression=compression,
    )
    return DataLoader(
        ds, batch_size=config.data.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        collate_fn=FFEvalCollate(),
    )


@torch.no_grad()
def infer_simple(model, loader, device, desc="Eval"):
    """通用推理：返回 (preds, labels, methods)"""
    model.eval()
    preds, labels, methods = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        frames = batch["frames"].to(device)
        labels.append(batch["label"].cpu().numpy())
        if "method" in batch:
            methods.extend(batch["method"])
        else:
            methods.extend(["unknown"] * len(batch["label"]))
        outputs = model(frames)
        logits = outputs["logits"]
        if logits.size(1) > 1:
            p = torch.sigmoid(logits[:, 1])
        else:
            p = torch.sigmoid(logits.squeeze(-1))
        preds.append(p.cpu().numpy())
    preds = np.concatenate(preds).astype(np.float64)
    labels = np.concatenate(labels).astype(int)
    return preds, labels, methods


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


def per_method_metrics(preds, labels, methods, threshold):
    """按篡改方法分组：每组 = 所有 real + 该 method 的 fake，计算 AUC

    返回 dict[method] = metrics + count
    """
    out = {}
    methods_arr = np.asarray(methods, dtype=object)
    for method in FF_METHODS:
        # real 样本（method == "original"） + 该 method 的 fake 样本
        mask = np.array(
            [(m == "original" or m == method) for m in methods_arr], dtype=bool
        )
        if mask.sum() < 10:
            continue
        m_preds = preds[mask]
        m_labels = labels[mask]
        if len(np.unique(m_labels)) < 2:
            continue
        m = metrics_with_threshold(m_preds, m_labels, threshold)
        m["num_samples"] = int(mask.sum())
        m["num_fake"] = int(np.sum(m_labels == 1))
        m["num_real"] = int(np.sum(m_labels == 0))
        out[method] = m
    return out


def safe_dir_name(name):
    # Windows 非法字符 + 常见分隔符
    for ch in [" ", "(", ")", "/", "+", "-", ">", "<", ":", "\"", "\\", "|", "?", "*"]:
        name = name.replace(ch, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def evaluate_config(cfg, config_obj, device, output_dir):
    """评估单个跨数据集配置：源 val 定阈值 → 目标 test 算指标"""
    log(f"\n=== {cfg['name']} ===")
    log(f"Loading model: {cfg['checkpoint']}")
    model, ckpt_epoch = load_model(config_obj, os.path.join(PROJECT_ROOT, cfg["checkpoint"]), device)

    # 源数据集 val 定阈值
    src_ds, src_comp = cfg["source"]
    threshold = 0.5
    threshold_source = "fallback_0.5"
    try:
        if src_ds == "celebdf":
            src_loader = build_celebdf_loader(config_obj, "val")
        else:
            src_loader = build_ffpp_loader(config_obj, "val", src_comp)
        log(f"Inferring on source val ({src_ds}/{src_comp}) for threshold...")
        s_preds, s_labels, _ = infer_simple(model, src_loader, device, desc=f"{cfg['name']}-srcval")
        if len(np.unique(s_labels)) >= 2:
            threshold = find_optimal_threshold(s_preds, s_labels)
            threshold_source = f"source_val({src_ds})"
    except Exception as e:
        log(f"  Source val threshold failed: {e}，回退到 0.5")
    log(f"  Threshold = {threshold:.4f} (source: {threshold_source})")

    # 目标数据集 test
    tgt_ds, tgt_comp = cfg["target"]
    if tgt_ds == "celebdf":
        tgt_loader = build_celebdf_loader(config_obj, "test")
    else:
        tgt_loader = build_ffpp_loader(config_obj, "test", tgt_comp)
    log(f"Inferring on target test ({tgt_ds}/{tgt_comp})...")
    preds, labels, methods = infer_simple(model, tgt_loader, device, desc=f"{cfg['name']}-tgttest")

    overall = metrics_with_threshold(preds, labels, threshold)
    overall["num_samples"] = int(len(labels))

    record = {
        "name": cfg["name"],
        "checkpoint": cfg["checkpoint"],
        "checkpoint_epoch": ckpt_epoch,
        "source_dataset": src_ds,
        "source_compression": src_comp,
        "target_dataset": tgt_ds,
        "target_compression": tgt_comp,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "overall": overall,
    }

    # FF++ 目标 → per-method breakdown
    if tgt_ds == "ffpp":
        record["per_method"] = per_method_metrics(preds, labels, methods, threshold)
        for mname, mm in record["per_method"].items():
            log(f"  [per-method {mname}] AUC={mm['auc']:.4f}  N={mm['num_samples']}")

    # 保存 raw_predictions
    cfg_dir = os.path.join(output_dir, safe_dir_name(cfg["name"]))
    os.makedirs(cfg_dir, exist_ok=True)
    np.savez(os.path.join(cfg_dir, "raw_predictions.npz"),
             preds=preds, labels=labels,
             methods=np.asarray(methods, dtype=object))

    log(f"  Overall: AUC={overall['auc']:.4f}  Acc={overall['accuracy']:.4f}  "
        f"F1={overall['f1']:.4f}  AP={overall['ap']:.4f}")
    # 释放模型显存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description="生成论文 Table 4 跨数据集泛化")
    parser.add_argument("--config", type=str, default="configs/joint_train.yaml")
    parser.add_argument("--output_dir", type=str, default="results/table4")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    log(f"Device: {device}")

    config = load_config(args.config)
    disable_train_augmentation(config)
    np.random.seed(getattr(config.training, "seed", 42))
    torch.manual_seed(getattr(config.training, "seed", 42))

    # 4 个跨数据集配置
    configs = [
        {"name": "Celeb-DF -> FF++ c23",
         "checkpoint": "checkpoints_v6/best_model_joint_celebdf.pt",
         "source": ("celebdf", "c23"), "target": ("ffpp", "c23")},
        {"name": "Celeb-DF -> FF++ c40",
         "checkpoint": "checkpoints_v6/best_model_joint_celebdf.pt",
         "source": ("celebdf", "c23"), "target": ("ffpp", "c40")},
        {"name": "FF++ -> Celeb-DF",
         "checkpoint": "checkpoints_v6/best_model_joint_ffpp.pt",
         "source": ("ffpp", "c23"), "target": ("celebdf", "c23")},
        {"name": "FF++-finetune -> Celeb-DF",
         "checkpoint": "checkpoints_v6/best_model_ffpp.pt",
         "source": ("ffpp", "c23"), "target": ("celebdf", "c23")},
    ]

    results = []
    for cfg in configs:
        try:
            results.append(evaluate_config(cfg, config, device, args.output_dir))
        except Exception as e:
            log(f"  [CONFIG ERROR] {cfg['name']} failed: {e}")
            import traceback as _tb
            log(_tb.format_exc())
            results.append({"name": cfg["name"], "error": str(e),
                            "overall": None, "per_method": {}})

    # —— 灾难性遗忘检测 ——
    # before = Celeb-DF 模型在 Celeb-DF test 的 AUC（即 configs[0] 的源模型在本域 test）
    # after  = FF++ 微调模型在 Celeb-DF test 的 AUC（configs[3]）
    log("\n=== Catastrophic Forgetting Analysis ===")
    forgetting = None
    try:
        log("Evaluating pre-finetune baseline on Celeb-DF test (best_model_joint_celebdf.pt)...")
        model_pre, ep_pre = load_model(config, os.path.join(PROJECT_ROOT, "checkpoints_v6/best_model_joint_celebdf.pt"), device)
        # 用 Celeb-DF val 定阈值（本域）
        val_loader = build_celebdf_loader(config, "val")
        v_preds, v_labels, _ = infer_simple(model_pre, val_loader, device, desc="pre-val")
        thr_pre = find_optimal_threshold(v_preds, v_labels) if len(np.unique(v_labels)) >= 2 else 0.5
        test_loader = build_celebdf_loader(config, "test")
        pre_preds, pre_labels, _ = infer_simple(model_pre, test_loader, device, desc="pre-test")
        pre_auc = float(compute_auc(pre_preds, pre_labels))
        pre_metrics = metrics_with_threshold(pre_preds, pre_labels, thr_pre)
        log(f"  Pre-finetune  Celeb-DF test AUC = {pre_auc:.4f}")
        del model_pre
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        after_auc = float(results[3]["overall"]["auc"])
        forgetting = {
            "before_checkpoint": "checkpoints_v6/best_model_joint_celebdf.pt",
            "after_checkpoint": "checkpoints_v6/best_model_ffpp.pt",
            "before_auc_celebdf_test": pre_auc,
            "after_auc_celebdf_test": after_auc,
            "auc_drop": float(pre_auc - after_auc),
            "before_metrics": pre_metrics,
            "after_metrics": results[3]["overall"],
        }
        log(f"  Post-finetune Celeb-DF test AUC = {after_auc:.4f}")
        log(f"  Catastrophic forgetting AUC drop = {forgetting['auc_drop']:+.4f}")
    except Exception as e:
        log(f"  Catastrophic forgetting analysis failed: {e}")
        forgetting = {"error": str(e)}

    out = {
        "table": "Table 4: Cross-Dataset Generalization",
        "configs_evaluated": len(results),
        "results": results,
        "catastrophic_forgetting": forgetting,
        "ff_methods": FF_METHODS,
    }
    out_path = os.path.join(args.output_dir, "cross_dataset_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"\n跨数据集结果已保存到 {out_path}")

    # —— 打印表格 ——
    print("\n" + "=" * 95)
    print("Table 4: Cross-Dataset Generalization")
    print("=" * 95)
    header = f"{'Config':<30} {'AUC':>8} {'Acc':>8} {'F1':>8} {'AP':>8} {'EER':>8} {'TPR@1%':>8}"
    print(header)
    print("-" * 95)
    for r in results:
        o = r.get("overall")
        if o is None:
            print(f"{r['name']:<30} [ERROR] {r.get('error', 'unknown')}")
            continue
        print(f"{r['name']:<30} {o['auc']:>8.4f} {o['accuracy']:>8.4f} {o['f1']:>8.4f} "
              f"{o['ap']:>8.4f} {o['eer']:>8.4f} {o['tpr_at_fpr_1']:>8.4f}")
        for mname, mm in r.get("per_method", {}).items():
            print(f"  - {mname:<26} {mm['auc']:>8.4f} {mm['accuracy']:>8.4f} {mm['f1']:>8.4f} "
                  f"{mm['ap']:>8.4f} {mm['eer']:>8.4f} {mm['tpr_at_fpr_1']:>8.4f}")
    print("-" * 95)
    if forgetting and "auc_drop" in forgetting:
        print(f"Catastrophic forgetting (Celeb-DF test): "
              f"before={forgetting['before_auc_celebdf_test']:.4f} -> "
              f"after={forgetting['after_auc_celebdf_test']:.4f}, "
              f"drop={forgetting['auc_drop']:+.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
