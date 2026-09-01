import torch
import torch.nn as nn
import torch.nn.functional as F


class BlendingHead(nn.Module):
    def __init__(self, in_channels_list, latent_dim=128, num_frames=8):
        super().__init__()
        self.num_frames = num_frames

        total_channels = sum(in_channels_list)
        self.input_proj = nn.Conv2d(total_channels, latent_dim, kernel_size=1)

        self.grad_conv = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, features):
        feat_list = []
        target_size = features["f_4"].shape[-2:]

        for key in ["f_1", "f_2", "f_3", "f_4"]:
            f = features[key]
            if f.dim() == 5:
                B, T, C, H, W = f.shape
                f = f.view(B * T, C, H, W)
            f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            feat_list.append(f)

        x = torch.cat(feat_list, dim=1)
        x = self.input_proj(x)

        T = self.num_frames
        B_T, C, H, W = x.shape
        B = B_T // T
        x = x.view(B, T, C, H, W).mean(dim=1)

        sobel_x = self.sobel_x.repeat(C, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        x = torch.cat([x, grad_mag], dim=1)
        x = self.grad_conv(x)
        x = self.output_proj(x)

        return x