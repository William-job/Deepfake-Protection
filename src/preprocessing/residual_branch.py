import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPyramid(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels

    def gaussian_kernel(self, size=5, sigma=1.0):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        g_2d = g.unsqueeze(0) * g.unsqueeze(1)
        return g_2d.view(1, 1, size, size)

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            is_video = True
        else:
            is_video = False
            B = x.size(0)
            C = x.size(1)

        kernel = self.gaussian_kernel().to(x.device)
        pyramid = []
        current = x

        for _ in range(self.levels):
            blurred = F.conv2d(current, kernel.repeat(C, 1, 1, 1), padding=2, groups=C)
            residual = current - blurred
            pyramid.append(residual)
            current = F.avg_pool2d(blurred, 2)

        high_freq = pyramid[0]
        if len(pyramid) > 1:
            upsampled = F.interpolate(
                pyramid[1],
                size=high_freq.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            high_freq = high_freq + upsampled

        if is_video:
            high_freq = high_freq.view(B, T, C, H, W)

        return high_freq


class ResidualBranch(nn.Module):
    def __init__(self, denoise_sigma=1.0):
        super().__init__()
        self.laplacian = LaplacianPyramid(levels=3)
        self.denoise_sigma = denoise_sigma

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
        else:
            x_flat = x
            B = x.size(0)
            C = x.size(1)

        laplacian = self.laplacian(x)

        noise = torch.randn_like(x_flat) * self.denoise_sigma / 255.0
        denoised = x_flat + noise

        residual = laplacian

        if x.dim() == 5:
            residual = residual.view(B, T, C, H, W)

        return residual
