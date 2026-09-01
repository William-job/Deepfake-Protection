#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""专业化矩阵验证（阶段 5.2 / 论文 Specialization Evidence）

问题：四专家是「专业化分工」还是「冗余 ensemble」？

证据设计：
1. 响应矩阵 M[artifact][expert]：对测试样本施加第 k 类 ControlledSBI 伪 artifact，
   测量各专家的 fake 概率响应。若专业化成立，第 k 类 artifact 应最大激活其对应专家
   （对角占优；注意 artifact→expert 映射为 0→temporal, 1→motion, 2→spectral, 3→boundary）。
2. Δ 响应矩阵：相对 clean（无 artifact）的提升量，剔除基础偏置。
3. 独立判别力：各专家在原始 test 样本上的独立 AUC（real vs fake）。
4. inter-expert 特征相似度：head_norm 特征两两余弦（4×4），off-diagonal 均值越低
   越说明专家看的是不同信号（非 ensemble 冗余）。
5. 路由分布：gating 权重 w 在 test 上的均值/熵（专家是否被差异化使用）。

样本：全部 178 real + 分层抽样 fake，共 ~1200。
输出：results/specialization/matrix.json + heatmap PNG。
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.dataset import DeepfakeDataset
from src.data.controlled_sbi import generate_batch
from src.models.car import CAR
from src.utils.metrics import compute_auc

EXPERT_ORDER = ["motion", "temporal", "spectral", "boundary"]
ARTIFACT_NAMES = ["temporal_jitter", "motion_ghost", "compression_art", "boundary_seam"]
# artifact k → 对应专家在 EXPERT_ORDER 中的索引（0→temporal=1, 1→motion=0, 2→spectral=2, 3→boundary=3）
ART2EXPERT = {0: 1, 1: 0, 2: 2, 3: 3}

# ---------------------------------------------------------------------------
# 强剂量探针（评估侧专用，不改训练代码）
#
# 动机：ControlledSBI 的 temporal_jitter 只交换 2 帧、motion_ghost 只复制
# 1-2 帧，而 TemporalHead 的 Δx 经 3D 卷积平滑 + 全局池化、MotionHead 对
# 7 个相邻帧残差对取均值——单点扰动会被时间维聚合稀释（spectral/boundary
# 探针作用于每一帧故无此问题）。强剂量探针把扰动铺满时间维，检验专家
# 对该 artifact 家族的真实响应能力（剂量响应设计）。
# ---------------------------------------------------------------------------
STRONG_PROBES = {
    "temporal_shuffle": {  # 全帧随机重排：几乎所有 Δx 被破坏
        "artifact_family": "temporal",
        "expert": "temporal",
    },
    "motion_freeze": {  # 后半段冻结为单帧复制：多数相邻对运动残差归零
        "artifact_family": "motion",
        "expert": "motion",
    },
}


def apply_strong_probe(frames, probe_name):
    """frames: (B,T,C,H,W)。返回同形状扰动张量。"""
    B, T = frames.shape[0], frames.shape[1]
    if probe_name == "temporal_shuffle":
        out = frames.clone()
        for b in range(B):
            perm = torch.randperm(T, device=frames.device)
            while torch.equal(perm, torch.arange(T, device=frames.device)):
                perm = torch.randperm(T, device=frames.device)
            out[b] = frames[b, perm]
        return out
    if probe_name == "motion_freeze":
        out = frames.clone()
        freeze_from = T // 2 - 1  # 从中间帧开始冻结后半段
        out[:, freeze_from:] = frames[:, freeze_from:freeze_from + 1]
        return out
    raise ValueError(f"unknown probe: {probe_name}")

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "specialization")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def expert_probs(car, frames):
    """frames: (B,T,C,H,W) → 各专家 fake 概率 (B,4) + head_norm 特征 (B,4,D) + 完整 forward。"""
    out = car(frames)
    head_outs = out["head_outputs"]
    logits_list = [car.experts[n](head_outs[n]) for n in EXPERT_ORDER]
    probs = torch.stack([torch.softmax(l, dim=1)[:, 1] for l in logits_list], dim=1)  # (B,4)
    feats = torch.stack([car.head_norms[n](head_outs[n]) for n in EXPERT_ORDER], dim=1)  # (B,4,D)
    return probs, feats, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--checkpoint", type=str,
                    default=os.path.join(PROJECT_ROOT, "results", "final_car_v3", "checkpoints", "best_model.pt"))
    ap.add_argument("--num_fake", type=int, default=1022)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    set_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    log(f"Device: {device}")
    model = CAR(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("ema_shadow") is not None:
        try:
            model.load_state_dict(ckpt["ema_shadow"], strict=True)
            log("已加载 EMA 权重")
        except RuntimeError:
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- 抽样：全部 real + 分层 fake ----
    ds = DeepfakeDataset(
        config.data.data_root, split="test",
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
    )
    real_idx = [i for i, s in enumerate(ds.samples) if s["label"] == 0]
    fake_idx = [i for i, s in enumerate(ds.samples) if s["label"] == 1]
    rng = random.Random(args.seed)
    fake_sel = rng.sample(fake_idx, min(args.num_fake, len(fake_idx)))
    sel = sorted(real_idx + fake_sel)
    log(f"样本: {len(real_idx)} real + {len(fake_sel)} fake = {len(sel)}")

    # ---- 收集响应 ----
    resp = {k: [] for k in range(4)}        # artifact k → (B,4) 专家 fake 概率
    resp_clean = []                          # clean → (B,4)
    resp_strong = {p: [] for p in STRONG_PROBES}  # 强剂量探针 → (B,4)
    labels_all, feats_all, w_all, diff_all = [], [], [], []

    for i in range(0, len(sel), args.batch_size):
        idxs = sel[i:i + args.batch_size]
        batch = [ds[j] for j in idxs]
        frames = torch.stack([b["frames"] for b in batch]).to(device)
        labels_all.extend([int(b["label"].item()) for b in batch])

        probs, feats, out = expert_probs(model, frames)
        resp_clean.append(probs.cpu())
        feats_all.append(feats.cpu())
        w_all.append(out["w"].cpu())
        diff_all.append(out["difficulty"].cpu())

        for k in range(4):
            aug, _ = generate_batch(frames, k)
            p_k, _, _ = expert_probs(model, aug)
            resp[k].append(p_k.cpu())

        for probe_name in STRONG_PROBES:
            aug = apply_strong_probe(frames, probe_name)
            p_s, _, _ = expert_probs(model, aug)
            resp_strong[probe_name].append(p_s.cpu())

    labels = np.array(labels_all)
    resp_clean = torch.cat(resp_clean).numpy()                       # (N,4)
    resp = {k: torch.cat(v).numpy() for k, v in resp.items()}        # each (N,4)
    feats = torch.cat(feats_all)                                     # (N,4,D)
    w = torch.cat(w_all).numpy()                                     # (N,4)
    diff = torch.cat(diff_all).numpy().flatten()

    # ---- 1. 响应矩阵（整体 + 按类别） ----
    def mean_matrix(r_dict):
        return np.stack([r_dict[k].mean(axis=0) for k in range(4)])  # (4 artifact, 4 expert)

    M_all = mean_matrix(resp)
    M_real = mean_matrix({k: resp[k][labels == 0] for k in range(4)})
    M_fake = mean_matrix({k: resp[k][labels == 1] for k in range(4)})
    M_clean_row = resp_clean.mean(axis=0)                            # (4,)
    M_delta = M_all - M_clean_row[None, :]                           # Δ 响应

    # 对角占优判定（artifact k 的最大响应专家 == 对应专家？）
    diag_hits, margins = [], []
    for k in range(4):
        target = ART2EXPERT[k]
        row = M_all[k]
        argmax_e = int(np.argmax(row))
        diag_hits.append(argmax_e == target)
        margins.append(float(row[target] - np.max(np.delete(row, target))))
    diag_rate = float(np.mean(diag_hits))

    # ---- 1b. 强剂量探针矩阵（剂量响应） ----
    resp_strong_np = {p: torch.cat(v).numpy() for p, v in resp_strong.items()}
    strong_rows, strong_report = {}, {}
    for probe_name, meta in STRONG_PROBES.items():
        row = resp_strong_np[probe_name].mean(axis=0)              # (4,)
        delta = row - M_clean_row                                  # 相对 clean 的 Δ
        target_e = EXPERT_ORDER.index(meta["expert"])
        strong_rows[probe_name] = row.tolist()
        strong_report[probe_name] = {
            "expert": meta["expert"],
            "row": row.tolist(),
            "delta_vs_clean": delta.tolist(),
            "target_expert_response": float(row[target_e]),
            "target_expert_delta": float(delta[target_e]),
            "argmax_expert": EXPERT_ORDER[int(np.argmax(row))],
            "argmax_is_target": int(np.argmax(row)) == target_e,
        }

    # ---- 2. 各专家独立判别力（clean test 样本上 real vs fake AUC） ----
    expert_aucs = {}
    for e, name in enumerate(EXPERT_ORDER):
        expert_aucs[name] = float(compute_auc(resp_clean[:, e], labels))

    # ---- 3. inter-expert 特征相似度 ----
    fn = F.normalize(feats, p=2, dim=-1)                             # (N,4,D)
    # 每样本 4×4 余弦矩阵，取均值
    cos = torch.einsum("ned,nfd->nef", fn, fn)                       # (N,4,4)
    cos_mean = cos.mean(dim=0).numpy()                               # (4,4)
    off_diag = cos_mean[~np.eye(4, dtype=bool)]
    off_diag_mean = float(off_diag.mean())

    # ---- 4. 路由统计 ----
    w_mean = w.mean(axis=0)
    w_entropy = float(-(w * np.log(np.clip(w, 1e-9, 1))).sum(axis=1).mean())

    # ---- 输出 ----
    result = {
        "checkpoint": args.checkpoint,
        "num_samples": int(len(labels)),
        "num_real": int((labels == 0).sum()),
        "num_fake": int((labels == 1).sum()),
        "expert_order": EXPERT_ORDER,
        "artifact_order": ARTIFACT_NAMES,
        "artifact_to_expert": {ARTIFACT_NAMES[k]: EXPERT_ORDER[ART2EXPERT[k]] for k in range(4)},
        "response_matrix_all": M_all.tolist(),
        "response_matrix_real": M_real.tolist(),
        "response_matrix_fake": M_fake.tolist(),
        "response_matrix_delta": M_delta.tolist(),
        "clean_response": M_clean_row.tolist(),
        "strong_probes": {
            "note": "强剂量探针：把同族扰动铺满时间维（标准 ControlledSBI 探针仅扰动 1-2 帧，"
                    "会被时间维均值池化稀释）。用于剂量响应验证，检验专家对 artifact 家族的真实响应能力。",
            "rows": strong_rows,
            "report": strong_report,
        },
        "diagonal_dominance": {
            "per_artifact_hit": {ARTIFACT_NAMES[k]: bool(diag_hits[k]) for k in range(4)},
            "hit_rate": diag_rate,
            "margin_to_best_other": {ARTIFACT_NAMES[k]: margins[k] for k in range(4)},
        },
        "expert_standalone_auc": expert_aucs,
        "inter_expert_cosine": cos_mean.tolist(),
        "inter_expert_offdiag_mean": off_diag_mean,
        "gating": {
            "mean_weights": {EXPERT_ORDER[e]: float(w_mean[e]) for e in range(4)},
            "mean_entropy": w_entropy,
            "difficulty_mean": float(diff.mean()),
            "difficulty_std": float(diff.std()),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = os.path.join(OUT_DIR, "matrix.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---- 控制台摘要 ----
    log("=" * 70)
    log("响应矩阵 M[artifact][expert]（专家 fake 概率均值）")
    header = f"{'artifact\\expert':<20}" + "".join(f"{n:>12}" for n in EXPERT_ORDER)
    log(header)
    for k in range(4):
        row = f"{ARTIFACT_NAMES[k]:<20}" + "".join(f"{M_all[k, e]:>12.4f}" for e in range(4))
        mark = " ←对角" if diag_hits[k] else f" ←错配(应激活 {EXPERT_ORDER[ART2EXPERT[k]]})"
        log(row + mark)
    log(f"对角占优命中率: {diag_rate:.2f} (4 类 artifact)")
    log("---- 强剂量探针（剂量响应） ----")
    for probe_name, rep in strong_report.items():
        row_str = ", ".join(f"{n}={rep['row'][e]:.4f}" for e, n in enumerate(EXPERT_ORDER))
        log(f"{probe_name:<18} {row_str}")
        log(f"{'':<18} 目标专家 {rep['expert']} Δ={rep['target_expert_delta']:+.4f} "
            f"(argmax={'命中' if rep['argmax_is_target'] else rep['argmax_expert']})")
    log(f"clean 响应基线: " + ", ".join(f"{n}={M_clean_row[e]:.4f}" for e, n in enumerate(EXPERT_ORDER)))
    log("各专家独立 AUC: " + ", ".join(f"{n}={v:.4f}" for n, v in expert_aucs.items()))
    log(f"inter-expert 余弦 off-diagonal 均值: {off_diag_mean:.4f}（越低越非冗余）")
    log(f"门控权重均值: " + ", ".join(f"{EXPERT_ORDER[e]}={w_mean[e]:.3f}" for e in range(4))
        + f" | 平均熵: {w_entropy:.3f}")
    log(f"已保存: {out_path}")

    # ---- 热力图（论文图） ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, mat, title in [(axes[0], M_all, "Expert response matrix"),
                               (axes[1], M_delta, "Δ response vs clean")]:
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(4)); ax.set_xticklabels(EXPERT_ORDER)
            ax.set_yticks(range(4)); ax.set_yticklabels(ARTIFACT_NAMES)
            ax.set_xlabel("Expert"); ax.set_ylabel("Artifact")
            ax.set_title(title)
            for k in range(4):
                for e in range(4):
                    ax.text(e, k, f"{mat[k, e]:.3f}", ha="center", va="center", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Counterfactual specialization (CAR-v3, n={len(labels)})")
        fig.tight_layout()
        png = os.path.join(OUT_DIR, "matrix_heatmap.png")
        fig.savefig(png, dpi=200)
        log(f"热力图已保存: {png}")
    except Exception as e:
        log(f"[WARN] 绘图失败（不影响 JSON）: {e}")


if __name__ == "__main__":
    main()
