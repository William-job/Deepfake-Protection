import numpy as np
import torch


class CutMixCollate:
    def __init__(self, p=0.5, alpha=1.0, spatial_consistency=True):
        self.p = p
        self.alpha = alpha
        self.spatial_consistency = spatial_consistency

    def __call__(self, batch):
        frames = torch.stack([item["frames"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])

        B, T, C, H, W = frames.shape

        if np.random.random() < self.p and B >= 2:
            lam = np.random.beta(self.alpha, self.alpha)

            idx = torch.randperm(B)

            cut_w = int(W * np.sqrt(1.0 - lam))
            cut_h = int(H * np.sqrt(1.0 - lam))

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            x1 = max(0, cx - cut_w // 2)
            y1 = max(0, cy - cut_h // 2)
            x2 = min(W, x1 + cut_w)
            y2 = min(H, y1 + cut_h)

            if x2 > x1 and y2 > y1:
                if self.spatial_consistency:
                    patch_src = frames[idx, :, :, y1:y2, x1:x2]
                    frames[:, :, :, y1:y2, x1:x2] = patch_src
                else:
                    for t in range(T):
                        patch_src = frames[idx, t, :, y1:y2, x1:x2]
                        frames[:, t, :, y1:y2, x1:x2] = patch_src

                lam_real = 1.0 - ((x2 - x1) * (y2 - y1)) / float(W * H)
                mixed_labels = lam_real * labels + (1.0 - lam_real) * labels[idx]
                labels = mixed_labels

        return {"frames": frames, "label": labels}
