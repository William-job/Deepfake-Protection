"""阶段 2.2：四专家专属伪任务（Artifact-Centric Pretraining, Level 2）。

每个专家对一类 ControlledSBI 伪 artifact 做专属判别预训练：
  - motion   ← motion_ghost（运动断裂）     判别运动是否平滑可解释
  - temporal ← temporal_jitter（帧序抖动）  判别时序是否连续
  - spectral ← compression_art（高频截断）  判别频谱是否异常
  - boundary ← boundary_seam（边界接缝）    判别边界/身份是否一致

伪任务为二分类（real=0 / artifact=1），用各专家的 head+expert 在
"其他专家与门控冻结"的条件下独立训练，产出 expert 的预训练权重。
"""
import torch
import torch.nn as nn

EXPERT_TO_ARTIFACT = {
    "motion": 1,     # motion_ghost
    "temporal": 0,   # temporal_jitter
    "spectral": 2,   # compression_art
    "boundary": 3,   # boundary_seam
}


class ExpertPretask(nn.Module):
    """单个专家的伪任务判别头：复用 CAR 的 head+expert 做 real/artifact 二分类。

    forward(x) -> logits (B, 2)；x 为原始帧 (B, T, C, H, W)。
    内部走 stem→head→expert 的对应分支，仅该分支参数可训练。
    """

    def __init__(self, car_model, expert_name):
        super().__init__()
        assert expert_name in EXPERT_TO_ARTIFACT
        self.expert_name = expert_name
        self.car = car_model
        self.artifact_type = EXPERT_TO_ARTIFACT[expert_name]

    def forward(self, x):
        preprocessed = self.car.preprocessor(x)
        features = self.car.stem(preprocessed["rgb"])
        if self.expert_name == "motion":
            h = self.car.motion_head(features)
        elif self.expert_name == "temporal":
            h = self.car.temporal_head(features)
        elif self.expert_name == "spectral":
            h = self.car.spectral_head(features)
        elif self.expert_name == "boundary":
            h = self.car.boundary_head(features)
        return self.car.experts[self.expert_name](h)


def freeze_except(car_model, expert_name):
    """冻结除指定 head+expert 外的所有参数（含 stem）。"""
    for p in car_model.parameters():
        p.requires_grad = False
    for mod in [getattr(car_model, f"{expert_name}_head"),
                car_model.experts[expert_name]]:
        for p in mod.parameters():
            p.requires_grad = True


def pretask_loss(logits, labels):
    """real/artifact 二分类交叉熵。labels: (B,) long，0=real, 1=artifact。"""
    return nn.functional.cross_entropy(logits, labels)


def pretask_metrics(logits, labels):
    """返回 (acc, auc)。AUC 用 fake 概率 = softmax[:,1]。"""
    probs = torch.softmax(logits, dim=1)[:, 1]
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    try:
        from src.utils.metrics import compute_auc
        auc = compute_auc(probs.detach().cpu().numpy(), labels.detach().cpu().numpy())
    except Exception:
        auc = 0.5
    return acc, auc