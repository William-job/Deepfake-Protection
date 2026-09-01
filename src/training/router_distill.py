"""阶段 3.2：Oracle Routing Distillation。

门控以"各专家在当前样本上的异常度"为 teacher 进行蒸馏，
而非从 BCE 联合训练直接习得。

流程（顺序固定，不可颠倒）：
  1. 专家先冻结（已完成 Level 2 预训练）
  2. 由各专家对样本的 fake 概率（异常度）归一化构造 oracle distribution q
  3. 用 KL 蒸馏训练门控输出 w_dense 匹配 q
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.controlled_sbi import generate_batch

EXPERT_ORDER = ["motion", "temporal", "spectral", "boundary"]
# artifact_type -> EXPERT_ORDER 顺序索引
ART2ORDER = {0: 1, 1: 0, 2: 2, 3: 3}  # temporal->1, motion->0, spectral->2, boundary->3


@torch.no_grad()
def build_oracle_distribution(car_model, frames):
    """由各专家对样本的 fake 概率构造 oracle 路由分布 q。

    对每个样本施加随机一类 artifact，然后以"该 artifact 对应专家的 fake 概率"
    作为该专家应得权重的证据，归一化后得到 q。
    返回 (q, artifact_types)。
    """
    B = frames.shape[0]
    art, types = generate_batch(frames)  # types: (B,) in {0,1,2,3}
    art = art.to(next(car_model.parameters()).device)

    preprocessed = car_model.preprocessor(art)
    features = car_model.stem(preprocessed["rgb"])
    head_outs = {
        "motion": car_model.motion_head(features),
        "temporal": car_model.temporal_head(features),
        "spectral": car_model.spectral_head(features),
        "boundary": car_model.boundary_head(features),
    }
    probs = []
    for n in EXPERT_ORDER:
        p = F.softmax(car_model.experts[n](head_outs[n]), dim=1)[:, 1]  # (B,)
        probs.append(p)
    probs = torch.stack(probs, dim=1)  # (B, 4)

    # oracle：异常度越高的专家权重越大（softmax 归一化）
    q = F.softmax(probs * 4.0, dim=1)  # 温度 4 锐化
    return q, types


def oracle_distillation_loss(w_dense, q):
    """KL 蒸馏：训练门控 w_dense 匹配 oracle q。"""
    return F.kl_div(torch.log(w_dense.clamp(min=1e-8)), q, reduction="batchmean")


def train_router_step(car_model, frames, optimizer, device):
    """单步 router 蒸馏训练。返回 (kl_loss, q, w_dense)。"""
    frames = frames.to(device)
    q, _ = build_oracle_distribution(car_model, frames)

    # 前向（仅 gating/fusion/head_norms 可训练）
    out = car_model(frames)
    w_dense = out["w_dense"]

    loss = oracle_distillation_loss(w_dense, q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), q.detach(), w_dense.detach()