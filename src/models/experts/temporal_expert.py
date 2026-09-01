import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=8):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class TemporalExpert(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=256, num_classes=2, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_dim)

        self.tsm_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_dim)

        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(hidden_dim // 2)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        # x comes from head output: (B, latent_dim)
        # We need to reshape it for 3D conv
        if x.dim() == 2:
            B, C = x.shape
            # Reshape to (B, C, 1, 1, 1) for 3D conv
            x = x.view(B, C, 1, 1, 1)
        elif x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B, C, T, 1, 1)
        elif x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, 1, H, W)
        elif x.dim() == 5:
            pass  # Already 5D
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.tsm_conv(x)))
        x = F.relu(self.bn3(self.conv2(x)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
