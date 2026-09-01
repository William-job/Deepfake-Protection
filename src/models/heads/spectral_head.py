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


def _dct_basis(size, device, dtype):
    """构造 DCT-II 正交基 (size, size)。"""
    n = torch.arange(size, device=device, dtype=dtype)
    k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    basis = torch.cos(torch.pi * (n + 0.5) * k / size)
    basis[0] = basis[0] * (1.0 / (size ** 0.5))
    basis[1:] = basis[1:] * ((2.0 / size) ** 0.5)
    return basis  # (size, size)


class SpectralHead(nn.Module):
    """频谱-结构一致性 Head（SSCE 的 feature extractor）。

    避免旧 FFT 的"绝对能量尺度分类捷径"：
      1. 用 DCT 把每帧特征分解到 num_bands 个频段（低频→高频）
      2. 计算 band-ratio（对数比，消除绝对能量、编码压缩无关的不变量）
      3. 引入局部-全局一致性（局部谱统计与全局谱统计的差）

    输出 (B, latent_dim) 的频谱一致性 latent 表示。
    """

    def __init__(self, in_channels_list, latent_dim=128, num_frames=8, num_bands=8):
        super().__init__()
        self.num_frames = num_frames
        self.num_bands = num_bands
        self.latent_dim = latent_dim

        total_channels = sum(in_channels_list)
        self.input_proj = nn.Conv2d(total_channels, latent_dim, kernel_size=1)

        # band-ratio + 局部-全局一致性 -> latent
        feat_dim = num_bands + num_bands  # band-ratio(8) + local-global(8)
        self.output_proj = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def _dct2(self, x):
        """对 (B, C, H, W) 做二维 DCT，返回同形状频谱（对数幅度）。"""
        B, C, H, W = x.shape
        db_h = _dct_basis(H, x.device, x.dtype)   # (H, H)
        db_w = _dct_basis(W, x.device, x.dtype)   # (W, W)
        # X_dct = D_h @ X @ D_w^T
        x = torch.matmul(db_h, x)                  # (B, C, H, W)
        x = torch.matmul(x, db_w.t())              # (B, C, H, W)
        return torch.log1p(x.abs())                # log-magnitude

    def _band_profile(self, spec):
        """按频率半径把谱分成 num_bands 个环带，返回每带能量 (B, C, num_bands)。"""
        B, C, H, W = spec.shape
        yy = torch.linspace(-1, 1, H, device=spec.device).view(H, 1).expand(H, W)
        xx = torch.linspace(-1, 1, W, device=spec.device).view(1, W).expand(H, W)
        radius = torch.sqrt(xx ** 2 + yy ** 2)  # (H, W) in [0, ~1.41]
        radius = radius / radius.max()          # 归一化到 [0, 1]
        band_edges = torch.linspace(0, 1, self.num_bands + 1, device=spec.device)
        profiles = []
        for i in range(self.num_bands):
            mask = ((radius >= band_edges[i]) & (radius < band_edges[i + 1])).float()
            mask = mask.view(1, 1, H, W)
            energy = (spec * mask).sum(dim=(2, 3)) / (mask.sum() + 1e-6)  # (B, C)
            profiles.append(energy)
        return torch.stack(profiles, dim=-1)  # (B, C, num_bands)

    def forward(self, features):
        x = _stack_video(features, self.num_frames)      # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = self.input_proj(x.view(B * T, C, H, W))      # (B*T, latent_dim, H, W)
        C = x.shape[1]

        # 一次性对所有帧做 DCT 与 band profile（避免逐帧 Python 循环）
        spec = self._dct2(x)                              # (B*T, C, H, W)
        bands = self._band_profile(spec)                  # (B*T, C, num_bands)
        bands = bands.view(B, T, C, self.num_bands).mean(dim=1)   # (B, C, num_bands)
        bands = bands.mean(dim=1)                          # (B, num_bands)

        # 局部-全局一致性：高频带能量 vs 全谱平均能量的差
        global_mean = spec.mean(dim=(2, 3)).view(B, T, C)   # (B, T, C)
        # 高频带 = 最后一带能量
        high_band = self._band_profile_last(spec).view(B, T, C).mean(dim=(1, 2))  # (B,)
        local_global = high_band - global_mean.mean(dim=(1, 2))    # (B,)

        # band-ratio：相邻频段对数比（对绝对能量/压缩不变）
        eps = 1e-6
        log_bands = torch.log(bands + eps)                     # (B, num_bands)
        band_ratio = log_bands[:, 1:] - log_bands[:, :-1]       # (B, num_bands-1)
        band_ratio = torch.cat([band_ratio, band_ratio[:, -1:]], dim=1)  # (B, num_bands)

        feat = torch.cat([band_ratio, local_global.unsqueeze(-1).expand(-1, self.num_bands)], dim=1)
        return self.output_proj(feat)                             # (B, latent_dim)

    def _band_profile_last(self, spec):
        """仅取最高频带能量 (B*T, C)。"""
        B, C, H, W = spec.shape
        yy = torch.linspace(-1, 1, H, device=spec.device).view(H, 1).expand(H, W)
        xx = torch.linspace(-1, 1, W, device=spec.device).view(1, W).expand(H, W)
        radius = torch.sqrt(xx ** 2 + yy ** 2)
        radius = radius / radius.max()
        mask = (radius >= (self.num_bands - 1) / self.num_bands).float().view(1, 1, H, W)
        return (spec * mask).sum(dim=(2, 3)) / (mask.sum() + 1e-6)