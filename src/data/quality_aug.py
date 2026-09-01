"""阶段 4.2：质量感知增强（Quality-Aware Augmentation）。

训练期对 batch 随机施加常见质量退化，使模型对 codec/blur/noise 等扰动不敏感，
直接针对鲁棒性目标（修复 Noise σ=0.05 崩溃、缩小 clean 与 corruption 退化差）。

六类退化（每 batch 随机选一类施加到部分样本，概率 p）：
  jpeg / blur / resize / h264 / noise / color

输入 frames: (B, T, C, H, W)，值域 [-1, 1]（与 DeepfakeDataset 归一化一致）。
输出同形状退化后的帧。
"""
import torch
import torch.nn.functional as F


def _jpeg(x, quality=None):
    B, T, C, H, W = x.shape
    q = torch.randint(30, 95, (1,)).item() if quality is None else quality
    v = x.view(B * T, C, H, W)
    small = F.interpolate(v, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
    v = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    noise = (torch.rand_like(v) - 0.5) * (100 - q) / 100.0 * 0.5
    return (v + noise).view(B, T, C, H, W)


def _blur(x, k=None):
    B, T, C, H, W = x.shape
    k = int(torch.randint(3, 9, (1,)).item()) if k is None else k
    if k % 2 == 0:
        k += 1
    v = x.view(B * T, C, H, W)
    v = F.avg_pool2d(v, kernel_size=k, stride=1, padding=k // 2)
    return v.view(B, T, C, H, W)


def _resize(x, scale=None):
    B, T, C, H, W = x.shape
    s = float(torch.rand(1).item() * 0.4 + 0.4) if scale is None else scale  # [0.4, 0.8]
    v = x.view(B * T, C, H, W)
    small = F.interpolate(v, scale_factor=s, mode="bilinear", align_corners=False)
    v = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    return v.view(B, T, C, H, W)


def _h264(x, crf=None):
    """H.264 近似：块状量化 + 高频截断。"""
    B, T, C, H, W = x.shape
    v = x.view(B * T, C, H, W)
    small = F.interpolate(v, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
    v = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    block = 8
    bh, bw = H // block, W // block
    noise = (torch.rand(B * T, C, bh, bw, device=x.device) - 0.5) * 0.25
    noise = F.interpolate(noise, size=(H, W), mode="nearest")
    return (v + noise).view(B, T, C, H, W)


def _noise(x, std=None):
    """与评估协议完全一致的真实高斯噪声（robustness_experiment.py 的噪声生成方式）。

    评估：img[0,255] + N(0, σ*255)，clip 到 [0,255]，再归一化回 [-1,1]。
    训练输入 x ∈ [-1,1]，等价地转 [0,1] → 加 N(0,σ) → clip [0,1] → 回 [-1,1]。
    σ 覆盖到 0.05 崩溃区间（重点修复）。
    """
    if std is None:
        # 采样 σ ∈ [0, 0.06]，与评估 0.01/0.02/0.05 同分布族
        std = float(torch.rand(1).item() * 0.06)
    x01 = (x + 1.0) / 2.0                      # [-1,1] -> [0,1]
    noisy = (x01 + torch.randn_like(x01) * std).clamp(0.0, 1.0)
    return noisy * 2.0 - 1.0                    # [0,1] -> [-1,1]


def _color(x):
    """亮度/对比度/饱和度扰动。"""
    B, T, C, H, W = x.shape
    v = x.view(B * T, C, H, W)
    delta = (torch.rand(1).item() - 0.5) * 0.3
    factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
    mean = v.mean(dim=(2, 3), keepdim=True)
    v = (v - mean) * factor + mean + delta
    return v.clamp(-1.0, 1.0).view(B, T, C, H, W)


_DEGRADES = {
    "jpeg": _jpeg,
    "blur": _blur,
    "resize": _resize,
    "h264": _h264,
    "noise": _noise,
    "color": _color,
}
DEGRADE_NAMES = list(_DEGRADES.keys())


def apply_quality_augmentation(frames, p=0.5):
    """对 batch 以概率 p 随机施加一类质量退化。

    参数:
        frames: (B, T, C, H, W)
        p: 每个样本被退化的概率
    返回:
        (augmented_frames, degrade_names)  # degrade_names: list[str]，每样本的退化类型
    """
    B = frames.shape[0]
    out = frames.clone()
    names = []
    for i in range(B):
        if torch.rand(1).item() < p:
            name = DEGRADE_NAMES[torch.randint(0, len(DEGRADE_NAMES), (1,)).item()]
            out[i] = _DEGRADES[name](frames[i].unsqueeze(0)).squeeze(0)
            names.append(name)
        else:
            names.append("clean")
    return out, names