import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyHead(nn.Module):
    def __init__(self, in_channels_list, latent_dim=128, num_frames=8, num_bands=16):
        super().__init__()
        self.num_frames = num_frames
        self.num_bands = num_bands
        self.latent_dim = latent_dim

        total_channels = sum(in_channels_list)
        self.input_proj = nn.Conv2d(total_channels, latent_dim, kernel_size=1)

        self.band_filters = nn.Parameter(torch.randn(num_bands, latent_dim) * 0.02)

        self.output_proj = nn.Sequential(
            nn.Linear(num_bands, latent_dim),
            nn.ReLU(),
        )

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
        x = x.view(B, T, C, H, W)

        x_freq = torch.fft.rfft(x.float(), dim=1)
        x_freq_real = x_freq.real

        # band_filters: (num_bands, latent_dim)
        # x_freq_real: (B, T//2+1, C, H, W)
        # We want to project C dimension through band filters
        bands = torch.einsum("btchw,nc->btnhw", x_freq_real.abs(), self.band_filters)

        x = bands.mean(dim=(1, 3, 4))  # (B, num_bands)
        x = self.output_proj(x)

        return x
