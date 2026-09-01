import torch
import numpy as np


class VideoTransform:
    def __init__(self, horizontal_flip=0.5, rotation=5, brightness=0.1, contrast=0.1):
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, frames):
        if np.random.random() < self.horizontal_flip:
            frames = np.flip(frames, axis=2).copy()

        if self.rotation > 0 and np.random.random() < 0.3:
            angle = np.random.uniform(-self.rotation, self.rotation)
            frames = self._rotate_frames(frames, angle)

        if self.brightness > 0:
            delta = np.random.uniform(-self.brightness, self.brightness)
            frames = np.clip(frames + delta, -1.0, 1.0)

        if self.contrast > 0:
            factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = frames.mean(axis=(1, 2, 3), keepdims=True)
            frames = (frames - mean) * factor + mean
            frames = np.clip(frames, -1.0, 1.0)

        return frames

    def _rotate_frames(self, frames, angle):
        import cv2
        T, H, W, C = frames.shape
        center = (W // 2, H // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = np.zeros_like(frames)
        for t in range(T):
            rotated[t] = cv2.warpAffine(frames[t], matrix, (W, H))
        return rotated