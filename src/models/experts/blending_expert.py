import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, edge_index=None):
        B, C, H, W = x.shape

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_x = sobel_x.repeat(C, 1, 1, 1).to(x.device)
        sobel_y = sobel_y.repeat(C, 1, 1, 1).to(x.device)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        # fp16 下 grad_x**2 易溢出（1000^2=1e6 > 65504），转 fp32 计算梯度幅度
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            grad_mag = torch.sqrt(grad_x.float() ** 2 + grad_y.float() ** 2 + 1e-6)
        grad_mag = grad_mag.to(x.dtype)

        edge_feat = torch.cat([x, grad_mag], dim=1)
        out = self.edge_conv(edge_feat)
        return out


class BlendingExpert(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=1)

        self.edge_gcn = EdgeConv(hidden_dim // 2, hidden_dim // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x can be (B, C) or (B, T, C) or (B, C, H, W) or (B, T, C, H, W)
        if x.dim() == 2:
            # (B, C) -> (B, C, 1, 1)
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            # (B, T, C) -> (B, C, T, 1)
            x = x.transpose(1, 2).unsqueeze(-1)
        elif x.dim() == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        elif x.dim() == 4:
            pass  # Already (B, C, H, W)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        x = self.input_proj(x)
        x = self.edge_gcn(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if x.size(0) % 8 == 0 and x.dim() == 2:
            T = 8
            B = x.size(0) // T
            if B > 0:
                x = x.view(B, T, -1).mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        return x
