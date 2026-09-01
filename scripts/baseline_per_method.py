#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""审稿人 P0 数据补充（零 GPU，全部来自冻结 npz/json）：

A) per-manipulation AUC：全部基线（Kimi P0 —— FaceSwap 弱点是 CAR 特有还是通病）
B) TPR@FPR 安全指标表（ChatGPT P0 —— 低误报预算下的检测率=规避攻击成功率）
   写入 results/per_method/baseline_breakdown.json + paper/tables.md 增补表

逻辑完全复用 scripts/per_method_breakdown.py 的 category_of（家族=路径中
Celeb-synthesis/ 后第一段），不依赖数据盘（纯 npz 内的 video_ids）。
"""
import json
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(PROJECT_ROOT, "results", *p)

MODELS = [
    ("car", "CAR (s42)"), ("car_s43", "CAR (s43)"), ("car_s44", "CAR (s44)"),
    ("mesonet", "MesoNet"), ("efficientnet_b0", "B0"),
    ("efficientnet_b0_qaug", "B0+QA"), ("xception", "Xception"),
    ("xception_qaug", "Xception+QA"),
]
NPZ = {m: R("robustness_honest", f"{m}_clean_preds.npz") for m, _ in MODELS}
JSON = {m: R("robustness_honest", f"{m}.json") for m, _ in MODELS}


def category_of(v):
    """复刻 per_method_breakdown.py：Celeb-synthesis/<家族>/... → 家族。"""
    v = v.replace("\\", "/")
    if "Celeb-synthesis/" in v:
        seg = v.split("Celeb-synthesis/", 1)[1]
        return seg.split("/")[0] if "/" in seg else "other"
    if "Celeb-real" in v:
        return "real_Celeb"
    if "YouTube-real" in v:
        return "real_YouTube"
    return "other"


# ---- A) per-manipulation AUC ------------------------------------------------
out = {"per_manipulation": {}, "tpr": {}, "timestamp": datetime.now().isoformat(timespec="seconds")}

print("=" * 72)
print("A) Per-manipulation AUC (vs all real, clean, n=5,418)")
print("=" * 72)
hdr = f"{'model':<16}" + "".join(f"{c:>14}" for c in ["FaceReenact", "FaceSwap", "TalkingFace"])
print(hdr)
for m, label in MODELS:
    with np.load(NPZ[m], allow_pickle=True) as d:
        preds, labels = d["preds"].astype(np.float64), d["labels"].astype(np.int64)
        cats = np.array([category_of(str(v)) for v in d["video_ids"]])
    real_mask = np.isin(cats, ["real_Celeb", "real_YouTube"])
    row = {"n_real": int(real_mask.sum())}
    rstr = f"{label:<16}"
    for fam in ["FaceReenact", "FaceSwap", "TalkingFace"]:
        m_f = cats == fam
        sub_p = np.concatenate([preds[m_f], preds[real_mask]])
        sub_l = np.concatenate([labels[m_f], labels[real_mask]])
        auc = float(roc_auc_score(sub_l, sub_p))
        row[fam] = {"n": int(m_f.sum()), "auc": round(auc, 4)}
        rstr += f"{auc:>14.3f}"
    out["per_manipulation"][m] = row
    print(rstr)

# ---- B) TPR@FPR（安全指标） --------------------------------------------------
print()
print("=" * 72)
print("B) TPR@FPR = 1% / 0.1%  (frozen thresholds; low-FPR detection rate)")
print("=" * 72)
conds = ["clean", "noise_std=0.05", "blur_kernel=7", "jpeg_quality=30",
         "brightness_factor=1.2"]
print(f"{'model':<16}" + "".join(f"{c.split('=')[0][:5]:>16}" for c in conds))
for m, label in MODELS:
    d = json.load(open(JSON[m], encoding="utf-8"))
    row = {}
    rstr = f"{label:<16}"
    for c in conds:
        e = d[c]
        row[c] = {"tpr@fpr1%": round(e["tpr_at_fpr_1"], 4),
                  "tpr@fpr0.1%": round(e["tpr_at_fpr_01"], 4)}
        rstr += f"  {e['tpr_at_fpr_1']:.3f}/{e['tpr_at_fpr_01']:.3f}"
    out["tpr"][m] = row
    print(rstr)

# CAR 3-seed 汇总
car_tpr = {c: (np.mean([out["tpr"][s][c]["tpr@fpr1%"] for s in ["car", "car_s43", "car_s44"]]),
               np.std([out["tpr"][s][c]["tpr@fpr1%"] for s in ["car", "car_s43", "car_s44"]], ddof=1))
           for c in conds}
print("\nCAR 3-seed TPR@FPR=1%: " +
      ", ".join(f"{c.split('=')[0][:5]}={m:.3f}±{s:.3f}" for c, (m, s) in car_tpr.items()))

dst = R("per_method", "baseline_breakdown.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nsaved: {dst}")
