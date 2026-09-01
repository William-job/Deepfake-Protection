import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamProcessor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, feat_t, feat_t1):
        diff = feat_t1 - feat_t
        x = torch.cat([feat_t, diff], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class FlowHead(nn.Module):
    def __init__(self, in_channels_list, latent_dim=128, num_frames=8):
        super().__init__()
        self.num_frames = num_frames

        total_channels = sum(in_channels_list)
        self.input_proj = nn.Conv2d(total_channels, latent_dim, kernel_size=1)

        self.stream = StreamProcessor(latent_dim)

        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_dim, latent_dim),
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
        B_T = x.shape[0]
        B = B_T // T
        C, H, W = x.shape[1:]
        x = x.view(B, T, C, H, W)

        flow_feats = []
        for t in range(T - 1):
            flow_feat = self.stream(x[:, t], x[:, t + 1])
            flow_feats.append(flow_feat)

        x = torch.stack(flow_feats, dim=1).mean(dim=1)
        x = self.output_proj(x)

        return x