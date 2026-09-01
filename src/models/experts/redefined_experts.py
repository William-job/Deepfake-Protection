import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_channels, hidden_dim, num_classes):
    """紧凑两层 MLP 分类头（Linear(in->hidden/2) -> Linear(hidden/2->2)）。"""
    return nn.Sequential(
        nn.Linear(in_channels, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim // 2, num_classes),
    )


class MotionExpert(nn.Module):
    """MWCE 专家：运动扭曲一致性分类器。

    接收 MotionHead 的 latent 特征 (B, C)，判别运动是否平滑可解释。
    真实人脸运动连续 → 残差小；Deepfake 逐帧生成 → 残差大且不连贯。
    """

    def __init__(self, in_channels=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.classifier = _mlp(in_channels, hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.classifier(x)


class TemporalExpert(nn.Module):
    """MTDE 专家：多尺度时序动态分类器。

    接收 TemporalHead 的 latent 特征 (B, C)，基于 Δx/Δ²x 时序残差判别
    时序不连续与抖动（紧凑 MLP，参数预算友好）。
    """

    def __init__(self, in_channels=128, hidden_dim=256, num_classes=2, num_frames=8):
        super().__init__()
        self.classifier = _mlp(in_channels, hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.classifier(x)


class SpectralExpert(nn.Module):
    """SSCE 专家：频谱-结构一致性分类器。

    接收 SpectralHead 的 latent 特征 (B, C)，基于 band-ratio 与局部-全局
    一致性判别频谱异常（压缩/上采样/混叠痕迹），不含绝对能量捷径。
    """

    def __init__(self, in_channels=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.classifier = _mlp(in_channels, hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.classifier(x)


class BoundaryExpert(nn.Module):
    """BICE 专家：边界-身份一致性分类器。

    接收 BoundaryHead 的 latent 特征 (B, C)，判别边界接缝不连续与
    身份细节不一致（面部融合痕迹）。
    """

    def __init__(self, in_channels=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.classifier = _mlp(in_channels, hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)