import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableFourier(nn.Module):
    def __init__(self, in_channels, num_bands=16):
        super().__init__()
        self.num_bands = num_bands
        self.band_filters = nn.Parameter(torch.randn(num_bands, in_channels) * 0.02)

    def forward(self, x):
        # Disable autocast for complex FFT operations (ComplexHalf is unstable)
        with torch.amp.autocast('cuda', enabled=False):
            B, T, C = x.shape
            x_f32 = x.float()
            x_freq = torch.fft.rfft(x_f32, dim=1)
            x_freq_real = x_freq.real
            x_freq_imag = x_freq.imag

            filter_weights = (self.band_filters.float().sum(dim=0) / self.num_bands)
            bands_real = torch.einsum("btc,c->btc", x_freq_real, filter_weights)
            bands_imag = torch.einsum("btc,c->btc", x_freq_imag, filter_weights)

            filtered = torch.complex(bands_real, bands_imag)
            x_filtered = torch.fft.irfft(filtered, n=T, dim=1)

        return x_filtered.to(x.dtype)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        B, T, C = x.shape
        avg_pool = x.mean(dim=1)
        attn = self.fc2(F.relu(self.fc1(avg_pool)))
        attn = torch.sigmoid(attn).unsqueeze(1)
        return x * attn


class FrequencyExpert(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.fourier = LearnableFourier(hidden_dim, num_bands=16)
        self.channel_attn = ChannelAttention(hidden_dim, reduction=8)

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x can be (B, C) or (B, T, C) or (B, C, H, W)
        if x.dim() == 2:
            # (B, C) -> (B, 1, C)
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).transpose(1, 2)
        elif x.dim() == 3:
            pass  # Already (B, T, C)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        x = self.input_proj(x)
        x = self.fourier(x)
        x = self.channel_attn(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
