import torch
import numpy as np


class SyntheticArtifactGenerator:
    def __init__(self, num_frames=8, image_size=224):
        self.num_frames = num_frames
        self.image_size = image_size

    def generate_temporal_artifact(self, frames):
        T, C, H, W = frames.shape
        artifact = frames.clone()
        shift = torch.randint(1, 4, (1,)).item()
        for t in range(shift, T):
            artifact[t] = frames[t - shift]
        gt = torch.tensor([1.0, 0.0, 0.0, 0.0])
        return artifact, gt

    def generate_flow_artifact(self, frames):
        T, C, H, W = frames.shape
        artifact = frames.clone()
        noise = torch.randn_like(frames) * 0.05
        artifact = artifact + noise
        gt = torch.tensor([0.0, 1.0, 0.0, 0.0])
        return artifact, gt

    def generate_frequency_artifact(self, frames):
        T, C, H, W = frames.shape
        artifact = frames.clone()
        freq_noise = torch.randn(T, C, H // 4, W // 4)
        freq_noise = torch.nn.functional.interpolate(
            freq_noise, size=(H, W), mode="bilinear", align_corners=False
        )
        artifact = artifact + freq_noise * 0.1
        gt = torch.tensor([0.0, 0.0, 1.0, 0.0])
        return artifact, gt

    def generate_blending_artifact(self, frames):
        T, C, H, W = frames.shape
        artifact = frames.clone()
        mask = torch.zeros(1, 1, H, W)
        h_start, w_start = H // 4, W // 4
        mask[:, :, h_start:h_start + H // 2, w_start:w_start + W // 2] = 1.0
        kernel_size = 7
        mask = torch.nn.functional.avg_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)
        noise = torch.randn_like(frames) * 0.1
        artifact = frames * (1 - mask) + (frames + noise) * mask
        gt = torch.tensor([0.0, 0.0, 0.0, 1.0])
        return artifact, gt

    def generate(self, frames, artifact_type=None):
        if artifact_type is None:
            artifact_type = np.random.choice(["temporal", "flow", "frequency", "blending"])

        if artifact_type == "temporal":
            return self.generate_temporal_artifact(frames)
        elif artifact_type == "flow":
            return self.generate_flow_artifact(frames)
        elif artifact_type == "frequency":
            return self.generate_frequency_artifact(frames)
        elif artifact_type == "blending":
            return self.generate_blending_artifact(frames)
        else:
            gt = torch.tensor([0.0, 0.0, 0.0, 0.0])
            return frames, gt