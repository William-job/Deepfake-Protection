"""阶段 3.3：一致性 Loss（鲁棒性正则）。

要求模型在常见质量退化（JPEG / resize / blur / brightness）下预测保持一致，
直接针对阶段 4 的鲁棒性目标（缩小 clean 与 corruption 的 AUC 退化差）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def jpeg_compress(frames, quality=60):
    """模拟 JPEG 压缩（高频截断 + 量化噪声），可微近似。"""
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W)
    small = F.interpolate(x, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
    x = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    noise = (torch.rand_like(x) - 0.5) * (100 - quality) / 100.0
    return (x + noise).view(B, T, C, H, W)


def resize_degrade(frames, scale=0.5):
    """先降采样再升采样（resize 退化）。"""
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W)
    small = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    x = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    return x.view(B, T, C, H, W)


def gaussian_blur(frames, kernel=5):
    """均值滤波近似高斯模糊。"""
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W)
    x = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return x.view(B, T, C, H, W)


def brightness_shift(frames, delta=0.15):
    """亮度偏移。"""
    return (frames + delta).clamp(-1.0, 1.0)


_DEGRADES = {
    "jpeg": jpeg_compress,
    "resize": resize_degrade,
    "blur": gaussian_blur,
    "brightness": brightness_shift,
}


class ConsistencyLoss(nn.Module):
    """clean 与多种 corruption 下的预测一致性 loss。

    用法:
        cons = ConsistencyLoss(car_model, weight=0.1, device="cuda")
        loss = cons(frames)   # 返回标量
    """

    def __init__(self, car_model, weight=0.1, device="cuda"):
        super().__init__()
        self.car = car_model
        self.weight = weight
        self.device = device

    def _prob(self, frames):
        out = self.car(frames)
        logits = out["logits"]
        return torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))

    def forward(self, frames):
        frames = frames.to(self.device)
        p_clean = self._prob(frames).detach()  # teacher 用 clean，不回传
        losses = []
        for name, fn in _DEGRADES.items():
            deg = fn(frames)
            p_deg = self._prob(deg)
            losses.append(F.smooth_l1_loss(p_deg, p_clean))
        return self.weight * torch.stack(losses).mean()