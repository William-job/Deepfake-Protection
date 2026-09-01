"""ControlledSBI：可控伪 artifact 生成器（阶段 2.1）。

对真实视频帧序列施加四类可控变换，产出带类型标签的伪 artifact 样本，
用于 Artifact-Centric 预训练（Level 2 伪任务）与专业化矩阵探针。

四类变换与新四专家一一对应：
  - temporal_jitter  → temporal（MTDE）：帧序抖动/打乱，破坏时序连续性
  - motion_ghost     → motion（MWCE）：帧复制导致运动断裂/重影
  - compression_art  → spectral（SSCE）：高频截断/块状压缩伪影
  - boundary_seam    → boundary（BICE）：面部掩码边界混合接缝

每类输出 (frames, type_index)，type_index ∈ {0:temporal, 1:motion, 2:spectral, 3:boundary}。
输入 frames: torch.Tensor (T, C, H, W)，值域任意（对增强鲁棒）。
"""
import torch
import torch.nn.functional as F


ARTIFACT_TYPES = ["temporal", "motion", "spectral", "boundary"]


def temporal_jitter(frames):
    """帧序抖动：随机交换中间若干帧的顺序，保证破坏时序连续性（type 0）。

    兼容 (B, T, ...) 与 (T, ...) 两种输入。
    """
    squeeze = frames.dim() == 4
    x = frames.unsqueeze(0) if squeeze else frames
    B, T = x.shape[0], x.shape[1]
    artifact = x.clone()
    for b in range(B):
        if T >= 3:
            i = torch.randint(1, T - 2, (1,)).item()
            j = torch.randint(1, T - 1, (1,)).item()
            while j == i:
                j = torch.randint(1, T - 1, (1,)).item()
            tmp = artifact[b, i].clone()
            artifact[b, i] = x[b, j]
            artifact[b, j] = tmp
    return (artifact.squeeze(0) if squeeze else artifact), 0


def motion_ghost(frames):
    """运动断裂/重影：把某一帧复制给相邻帧，使运动解释失败（type 1）。

    兼容 (B, T, ...) 与 (T, ...) 两种输入。
    """
    squeeze = frames.dim() == 4
    x = frames.unsqueeze(0) if squeeze else frames
    B, T = x.shape[0], x.shape[1]
    artifact = x.clone()
    for b in range(B):
        if T >= 2:
            src = torch.randint(0, T - 1, (1,)).item()
            dst = min(src + 1, T - 1)
            artifact[b, dst] = x[b, src]
            if T >= 4:
                src2 = torch.randint(0, T - 2, (1,)).item()
                artifact[b, src2 + 2] = x[b, src2]
    return (artifact.squeeze(0) if squeeze else artifact), 1


def compression_art(frames):
    """高频截断/块状压缩伪影（type 2）。

    对每帧：下采样→上采样（丢高频），叠加 8x8 块状量化噪声，模拟 JPEG/H.264。
    输入 (B, T, C, H, W) 或 (T, C, H, W)。
    """
    squeeze = frames.dim() == 4
    x = frames.unsqueeze(0) if squeeze else frames
    B, T, C, H, W = x.shape
    # 高频截断：先降采样再升采样
    small = F.interpolate(x.view(B * T, C, H, W), size=(H // 8, W // 8), mode="bilinear", align_corners=False)
    artifact = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    # 块状噪声（8x8 块常数偏移）
    block = 8
    bh, bw = H // block, W // block
    noise = (torch.rand(B * T, C, bh, bw, device=frames.device) - 0.5) * 0.3
    noise = F.interpolate(noise, size=(H, W), mode="nearest")
    artifact = (artifact + noise).view(B, T, C, H, W)
    return (artifact.squeeze(0) if squeeze else artifact), 2


def _rand_mask(T, H, W, device):
    """生成随机椭圆掩码（模拟面部区域），返回 (T,1,H,W) 软边掩码。"""
    mask = torch.zeros(1, 1, H, W, device=device)
    cy, cx = H // 2 + torch.randint(-H // 8, H // 8, (1,)).item(), W // 2 + torch.randint(-W // 8, W // 8, (1,)).item()
    ry, rx = H // 3, W // 3
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    mask[0, 0][ellipse] = 1.0
    # 软边（均值滤波模拟羽化）
    mask = F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7)
    return mask.expand(T, 1, H, W)


def boundary_seam(frames):
    """面部掩码边界混合接缝（type 3）。

    把掩码内的内容替换为"模糊/加噪版本"，沿掩码边界形成可见接缝，
    模拟 Deepfake 的面部融合边界。输入 (B, T, C, H, W) 或 (T, C, H, W)。
    """
    squeeze = frames.dim() == 4
    x = frames.unsqueeze(0) if squeeze else frames
    B, T, C, H, W = x.shape
    mask = _rand_mask(T, H, W, frames.device)          # (T,1,H,W)
    blurred = F.avg_pool2d(x.view(B * T, C, H, W), kernel_size=5, stride=1, padding=2)
    noise = torch.randn_like(blurred) * 0.05
    inner = (blurred + noise).view(B, T, C, H, W)
    artifact = x * (1 - mask.unsqueeze(0)) + inner * mask.unsqueeze(0)
    return (artifact.squeeze(0) if squeeze else artifact), 3


_GENERATORS = {
    0: temporal_jitter,
    1: motion_ghost,
    2: compression_art,
    3: boundary_seam,
}


def generate(frames, artifact_type=None):
    """生成单个伪 artifact 样本。

    参数:
        frames: (T, C, H, W) tensor
        artifact_type: None | int(0-3) | str("temporal"/"motion"/"spectral"/"boundary")
    返回:
        (artifact_frames, type_index)
    """
    if artifact_type is None:
        artifact_type = torch.randint(0, 4, (1,)).item()
    elif isinstance(artifact_type, str):
        artifact_type = ARTIFACT_TYPES.index(artifact_type)
    return _GENERATORS[artifact_type](frames)


def generate_batch(frames, artifact_type=None):
    """对 batch (B, T, C, H, W) 逐样本生成，返回 (artifact_batch, type_indices tensor)。"""
    B = frames.shape[0]
    out, types = [], []
    for i in range(B):
        t = torch.randint(0, 4, (1,)).item() if artifact_type is None else artifact_type
        a, ti = generate(frames[i], t)
        out.append(a)
        types.append(ti)
    return torch.stack(out, dim=0), torch.tensor(types, dtype=torch.long)