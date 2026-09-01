"""阶段 3.1：Counterfactual Expert Specialization。

通过 targeted augmentation 使每个专家对对应 artifact 的响应更强，
而非让所有专家共同学习同一 shortcut。

对应关系（与 ControlledSBI 一致）：
  motion   ← motion_ghost（运动断裂增强）
  temporal ← temporal_jitter（帧序抖动增强）
  spectral ← compression_art（高频截断增强）
  boundary ← boundary_seam（边界接缝增强）

Specialization loss：施加对应 artifact 增强时，要求对应专家的 fake 概率
显著高于其他专家（margin loss），促使专家专业化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.controlled_sbi import generate_batch

EXPERT_TO_ARTIFACT = {
    "motion": 1, "temporal": 0, "spectral": 2, "boundary": 3,
}
EXPERT_ORDER = ["motion", "temporal", "spectral", "boundary"]


def specialization_loss(expert_logits_list, artifact_type, margin=0.5):
    """对一批施加了某类 artifact 的样本，鼓励对应专家的 fake 概率高于其他专家。

    参数:
        expert_logits_list: list of (B, 2)，长度 4，顺序与 EXPERT_ORDER 一致
        artifact_type: int（0-3），当前 batch 施加的 artifact 类型
        margin: 对应专家的 fake 概率应高于其他专家至少 margin
    返回:
        loss: 标量
    """
    # 对应专家索引 = artifact_type 映射到 EXPERT_ORDER 顺序
    art2order = {0: 1, 1: 0, 2: 2, 3: 3}  # artifact_type -> EXPERT_ORDER idx
    target_idx = art2order[artifact_type]

    probs = [F.softmax(l, dim=1)[:, 1] for l in expert_logits_list]  # 各专家 fake 概率
    p_target = probs[target_idx]
    losses = []
    for i, p in enumerate(probs):
        if i == target_idx:
            continue
        losses.append(F.relu(margin - (p_target - p)))
    return torch.stack(losses).mean()


def apply_targeted_augmentation(frames, expert_name):
    """对 batch 施加对应专家的 targeted augmentation，返回 (augmented, artifact_type)。"""
    art_type = EXPERT_TO_ARTIFACT[expert_name]
    aug, types = generate_batch(frames, art_type)
    return aug, art_type


class SpecializationLoss(nn.Module):
    """可嵌入训练循环的 counterfactual specialization loss。

    用法:
        spec_loss_fn = SpecializationLoss(car_model, margin=0.5, weight=0.1)
        loss = spec_loss_fn(frames)   # frames: (B, T, C, H, W)
    """

    def __init__(self, car_model, margin=0.5, weight=0.1, device="cuda"):
        super().__init__()
        self.car = car_model
        self.margin = margin
        self.weight = weight
        self.device = device

    def forward(self, frames):
        # 随机选一个专家施加其 targeted augmentation
        expert_name = EXPERT_ORDER[torch.randint(0, 4, (1,)).item()]
        aug, art_type = apply_targeted_augmentation(frames, expert_name)
        aug = aug.to(self.device)

        # 前向各专家（不更新门控/fusion，仅取 expert logits）
        preprocessed = self.car.preprocessor(aug)
        features = self.car.stem(preprocessed["rgb"])
        head_outs = {
            "motion": self.car.motion_head(features),
            "temporal": self.car.temporal_head(features),
            "spectral": self.car.spectral_head(features),
            "boundary": self.car.boundary_head(features),
        }
        logits_list = [self.car.experts[n](head_outs[n]) for n in EXPERT_ORDER]
        loss = specialization_loss(logits_list, art_type, margin=self.margin)
        return self.weight * loss, expert_name