#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-method / unseen-generator 分解（论文 Generalization Breakdown）

Celeb-DF++ 的伪造视频分三类目录：FaceReenact / FaceSwap / TalkingFace。
本脚本回答两个审稿人必问的问题：
1. test 中的类别是否在 train 中出现过（seen/unseen generator 判定）；
2. 模型对各类伪造方法（尤其 unseen 类别）的 AUC 分解。

用法：
    python -u scripts/per_method_breakdown.py --npz results/robustness_honest/car_clean_preds.npz
    python -u scripts/per_method_breakdown.py --npz a.npz --npz b.npz   # 多模型对比

输入 npz 需含 preds/labels/video_ids（eval_honest / robustness_honest 均按此保存）。
"""
import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.config import load_config
from src.data.dataset import DeepfakeDataset
from src.utils.metrics import compute_auc, compute_ap

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "per_method")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def category_of(rel_path):
    """'Celeb-synthesis/FaceSwap/xxx.mp4' → 'FaceSwap'；real → 'real'。"""
    p = rel_path.replace("\\", "/")
    if p.startswith("Celeb-synthesis/"):
        parts = p.split("/")
        return parts[1] if len(parts) > 2 else "Celeb-synthesis(other)"
    if p.startswith("Celeb-real/"):
        return "real_Celeb"
    if p.startswith("YouTube-real/"):
        return "real_YouTube"
    return "other"


def load_split_categories(config):
    """只读元数据（不解码视频），返回各 split 的 {category: count}。"""
    counts = {}
    for split in ["train", "val", "test"]:
        ds = DeepfakeDataset(
            config.data.data_root, split=split,
            num_frames=config.data.num_frames,
            frame_stride=config.data.frame_stride,
            image_size=config.data.image_size,
        )
        c = {}
        for s in ds.samples:
            cat = category_of(s.get("rel_path", s["video_path"]))
            c[cat] = c.get(cat, 0) + 1
        counts[split] = c
        log(f"{split}: {c}")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--npz", action="append", required=True, help="可多次传入")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. split 类别构成（seen/unseen 判定） ----
    split_counts = load_split_categories(config)
    fake_cats = sorted({c for sc in split_counts.values() for c in sc
                        if c not in ("real_Celeb", "real_YouTube", "other")})
    train_cats = set(split_counts["train"].keys())
    val_cats = set(split_counts["val"].keys())
    test_cats = set(split_counts["test"].keys())

    unseen_in_test = sorted(c for c in fake_cats if c in test_cats and c not in train_cats)
    seen_in_test = sorted(c for c in fake_cats if c in test_cats and c in train_cats)
    log(f"test 中 seen 类别: {seen_in_test}")
    log(f"test 中 unseen 类别: {unseen_in_test}")

    # ---- 2. 逐 npz 分解 ----
    results = {}
    for npz_path in args.npz:
        name = os.path.splitext(os.path.basename(npz_path))[0]
        if not os.path.exists(npz_path):
            log(f"[SKIP] {npz_path} 不存在")
            continue
        d = np.load(npz_path, allow_pickle=True)
        preds, labels = d["preds"], d["labels"]
        video_ids = [str(v).replace("\\", "/") for v in d["video_ids"]]

        # rel_path → 类别（video_ids 为绝对路径，含 Celeb-synthesis 段即可定位）
        cats = []
        for v in video_ids:
            if "Celeb-synthesis/" in v:
                seg = v.split("Celeb-synthesis/", 1)[1]
                cats.append(seg.split("/")[0] if "/" in seg else "Celeb-synthesis(other)")
            elif "Celeb-real" in v:
                cats.append("real_Celeb")
            elif "YouTube-real" in v:
                cats.append("real_YouTube")
            else:
                cats.append("other")
        cats = np.array(cats)

        real_mask = np.isin(cats, ["real_Celeb", "real_YouTube"])
        real_preds, real_labels = preds[real_mask], labels[real_mask]
        if (real_labels == 0).mean() < 1.0:
            log(f"[WARN] {name}: real 池中混有 label=1 的样本，请检查")

        per_cat = {"overall_auc": float(compute_auc(preds, labels)),
                   "overall_ap": float(compute_ap(preds, labels)),
                   "num_real": int(real_mask.sum())}
        for cat in fake_cats:
            m = cats == cat
            if m.sum() == 0:
                continue
            # AUC：该类别 fake vs 全部 real（标准 per-manipulation 协议）
            sub_preds = np.concatenate([preds[m], real_preds])
            sub_labels = np.concatenate([labels[m], real_labels])
            per_cat[cat] = {
                "count": int(m.sum()),
                "auc_vs_real": float(compute_auc(sub_preds, sub_labels)),
                "ap_vs_real": float(compute_ap(sub_preds, sub_labels)),
                "seen_in_train": cat in train_cats,
            }
        results[name] = per_cat

        log(f"===== {name} =====")
        log(f"overall AUC={per_cat['overall_auc']:.4f} (real n={per_cat['num_real']})")
        for cat in fake_cats:
            if cat in per_cat:
                tag = "seen" if per_cat[cat]["seen_in_train"] else "UNSEEN"
                log(f"  {cat:<15} n={per_cat[cat]['count']:>5}  "
                    f"AUC={per_cat[cat]['auc_vs_real']:.4f}  ({tag})")

    # ---- 3. 保存 ----
    out = {
        "split_category_counts": split_counts,
        "fake_categories": fake_cats,
        "seen_in_test": seen_in_test,
        "unseen_in_test": unseen_in_test,
        "results": results,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = args.output or os.path.join(OUT_DIR, "breakdown.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
