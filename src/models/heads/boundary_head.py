import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack_video(features, num_frames):
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


class BoundaryHead(nn.Module):
    """边界-身份一致性 Head（BICE 的 feature extractor）。

    关注伪造视频常见的边界接缝与身份不一致：
      1. 边界接缝：Sobel 梯度幅度刻画面部融合边界的锐利度/不连续性
      2. 局部纹理：浅层卷积提取的局部纹理统计
      3. 身份一致性：中心区域与全局特征表示的余弦偏差（伪造常改变身份细节）

    输出 (B, latent_dim) 的边界-身份一致性 latent 表示。
    """

    def __init__(self, in_channels_list, latent_dim=128, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.latent_dim = latent_dim

        total_channels = sum(in_channels_list)
        self.input_proj = nn.Conv2d(total_channels, latent_dim, kernel_size=1)

        # 局部纹理分支
        self.texture_conv = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

        # 边界+纹理+身份 -> latent
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
        )

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _grad_mag(self, x):
        """x: (B, C, H, W) -> 梯度幅度 (B, C, H, W)，fp32 防溢出。"""
        B, C, H, W = x.shape
        sx = self.sobel_x.repeat(C, 1, 1, 1)
        sy = self.sobel_y.repeat(C, 1, 1, 1)
        gx = F.conv2d(x, sx, padding=1, groups=C)
        gy = F.conv2d(x, sy, padding=1, groups=C)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            mag = torch.sqrt(gx.float() ** 2 + gy.float() ** 2 + 1e-6)
        return mag.to(x.dtype)

    def forward(self, features):
        x = _stack_video(features, self.num_frames)      # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = self.input_proj(x.view(B * T, C, H, W))      # (B*T, latent_dim, H, W)
        C = x.shape[1]
        x = x.view(B, T, C, H, W).mean(dim=1)            # 时序平均 (B, C, H, W)

        # 1) 边界接缝：梯度幅度统计
        grad = self._grad_mag(x)                          # (B, C, H, W)
        boundary_feat = grad.mean(dim=(2, 3))             # (B, C)

        # 2) 局部纹理
        texture = self.texture_conv(x)                    # (B, C, H, W)
        texture_feat = texture.mean(dim=(2, 3))           # (B, C)

        # 3) 身份一致性：中心区域 vs 全局的余弦偏差
        Hc, Wc = H // 4, W // 4
        center = x[:, :, Hc:H - Hc, Wc:W - Wc]           # (B, C, H/2, W/2)
        center_vec = F.normalize(center.mean(dim=(2, 3)), p=2, dim=-1)   # (B, C)
        global_vec = F.normalize(x.mean(dim=(2, 3)), p=2, dim=-1)        # (B, C)
        identity_dev = (1.0 - (center_vec * global_vec).sum(dim=-1, keepdim=True))  # (B, 1)
        identity_feat = identity_dev.expand(-1, C)       # 广播到 C 维 (B, C)

        feat = torch.cat([boundary_feat, texture_feat, identity_feat], dim=1)  # (B, 3C)
        return self.output_proj(feat)                     # (B, latent_dim)