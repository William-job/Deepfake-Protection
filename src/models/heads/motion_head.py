import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack_video(features, num_frames):
    """多尺度特征插值到 f_4 分辨率后通道拼接，返回 (B, T, C, H, W)。"""
    target_size = features["f_4"].shape[-2:]
    feat_list = []
    for key in ["f_1", "f_2", "f_3", "f_4"]:
        f = features[key]
        if f.dim() == 5:
            B, T, C, H, W = f.shape
            f = f.view(B * T, C, H, W)
        f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
        feat_list.append(f)
    x = torch.cat(feat_list, dim=1)
    B_T, C, H, W = x.shape
    B = B_T // num_frames
    return x.view(B, num_frames, C, H, W)


class MotionHead(nn.Module):
    """运动扭曲一致性 Head（MWCE 的 feature extractor）。

    学习目标：度量"相邻帧特征能否被一段平滑的运动场相互解释"。
    真实人脸视频的运动应平滑连续；Deepfake 因逐帧独立生成，运动
    解释残差大且不连贯。输出运动一致性残差的 latent 表示 (B, latent_dim)。
    """

    def __init__(self, in_channels_list, latent_dim=128, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.latent_dim = latent_dim

        total_channels = sum(in_channels_list)
        self.input_proj = nn.Conv2d(total_channels, latent_dim, kernel_size=1)

        # 运动残差编码：输入 [f_t, f_{t+1}, f_{t+1}-f_t]（单层以控制参数预算）
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(latent_dim * 3, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

        # 残差幅度 -> latent
        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, features):
        x = _stack_video(features, self.num_frames)  # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = self.input_proj(x.view(B * T, C, H, W))
        C = x.shape[1]
        x = x.view(B, T, C, H, W)

        residuals = []
        for t in range(T - 1):
            f_t = x[:, t]          # (B, C, H, W)
            f_next = x[:, t + 1]   # (B, C, H, W)
            diff = f_next - f_t    # 未解释的运动残差
            pair = torch.cat([f_t, f_next, diff], dim=1)  # (B, 3C, H, W)
            residuals.append(self.motion_encoder(pair))    # (B, C, H, W)

        # 聚合所有相邻帧的运动残差（均值）
        motion = torch.stack(residuals, dim=1).mean(dim=1)  # (B, C, H, W)
        return self.output_proj(motion)                       # (B, latent_dim)