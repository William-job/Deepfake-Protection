import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack_video(features, num_frames):
    """与旧 head 一致：多尺度特征插值到 f_4 空间分辨率后通道拼接。

    返回 (B, T, C, H, W)，供差分/时序处理。
    """
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


class TemporalHead(nn.Module):
    """多尺度时序动态 Head（MTDE 的 feature extractor）。

    输入多尺度 stem 特征，输出 (B, latent_dim)：
      1. 多尺度拼接 + 1x1 投影到 latent_dim
      2. 一阶差分 Δx 与二阶差分 Δ²x（捕捉伪造视频的时序抖动与不连续性）
      3. 轻量 3D 卷积做时序平滑后全局池化

    相比旧实现提升了时序建模容量（由 ~45K 参数扩至数百 K 量级），
    为阶段 1 的 MTDE 专家提供更丰富的时序残差表示。
    """

    def __init__(self, in_channels_list, latent_dim=128, num_frames=8):
        super().__init__()
        self.num_frames = num_frames

        total_channels = sum(in_channels_list)
        self.input_proj = nn.Conv2d(total_channels, latent_dim, kernel_size=1)

        # 差分特征编码（对 Δx / Δ²x 的融合）
        self.diff_proj = nn.Conv2d(latent_dim * 3, latent_dim, kernel_size=1)

        # 轻量 3D 时序卷积（在时间维做局部平滑）
        self.temporal_conv = nn.Conv3d(
            latent_dim, latent_dim,
            kernel_size=(3, 3, 3), padding=(1, 1, 1),
            groups=latent_dim, bias=False,
        )
        self.bn = nn.BatchNorm3d(latent_dim)

        # 容量提升层
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
        )

        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

    def forward(self, features):
        x = _stack_video(features, self.num_frames)  # (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # 多尺度投影
        x = self.input_proj(x.view(B * T, C, H, W))  # (B*T, latent_dim, H, W)
        C = x.shape[1]

        # 一阶/二阶差分（在时间维）
        x_t = x.view(B, T, C, H, W)
        dx = x_t[:, 1:] - x_t[:, :-1]                       # (B, T-1, C, H, W)
        dx = torch.cat([dx[:, :1], dx], dim=1)              # pad 到 T
        d2x = dx[:, 1:] - dx[:, :-1]                        # (B, T-1, C, H, W)
        d2x = torch.cat([d2x[:, :1], d2x], dim=1)           # pad 到 T

        # 融合 x / Δx / Δ²x
        fused = torch.cat([x_t, dx, d2x], dim=2)           # (B, T, 3C, H, W)
        fused = self.diff_proj(fused.view(B * T, 3 * C, H, W))  # (B*T, C, H, W)

        # 3D 时序卷积
        fused = fused.view(B, T, C, H, W)
        fused = fused.permute(0, 2, 1, 3, 4)               # (B, C, T, H, W)
        fused = F.relu(self.bn(self.temporal_conv(fused)))

        # 池化 + MLP
        pooled = self.output_proj(fused)                    # (B, C)
        return self.mlp(pooled)                              # (B, latent_dim)