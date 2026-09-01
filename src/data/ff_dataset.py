import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class FFDataset(Dataset):
    """FaceForensics++ dataset loader for cross-dataset generalization experiments.

    Expected directory structure (after download):
        data_root/
            original_sequences/
                youtube/
                    c23/videos/     # real videos
            manipulated_sequences/
                Deepfakes/
                    c23/videos/     # fake videos
                Face2Face/
                    c23/videos/
                FaceSwap/
                    c23/videos/
                NeuralTextures/
                    c23/videos/
                FaceShifter/
                    c23/videos/
    """

    def __init__(self, data_root, split="test", compression="c23",
                 num_frames=8, frame_stride=2, image_size=224,
                 transform=None, methods=None):
        """
        Args:
            data_root: Path to FF++ dataset root
            split: "test" for cross-dataset eval (FF++ has fixed train/val/test splits)
            compression: "c23" or "c40"
            num_frames: Number of frames to sample per video
            frame_stride: Stride between sampled frames
            image_size: Resize frames to this size
            transform: Optional additional transforms
            methods: List of manipulation methods to include.
                     None means all methods.
                     Options: ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
        """
        self.data_root = data_root
        self.split = split
        self.compression = compression
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.transform = transform

        if methods is None:
            self.methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
        else:
            self.methods = methods

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        # Real videos: original_sequences/youtube/<compression>/videos/
        real_dir = os.path.join(
            self.data_root, "original_sequences", "youtube",
            self.compression, "videos"
        )
        if os.path.exists(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if fname.endswith(".mp4"):
                    samples.append({
                        "video_path": os.path.join(real_dir, fname),
                        "label": 0,  # Real
                        "is_fake": False,
                        "method": "original"
                    })

        # Fake videos: manipulated_sequences/<method>/<compression>/videos/
        for method in self.methods:
            fake_dir = os.path.join(
                self.data_root, "manipulated_sequences", method,
                self.compression, "videos"
            )
            if not os.path.exists(fake_dir):
                continue
            for fname in sorted(os.listdir(fake_dir)):
                if fname.endswith(".mp4"):
                    samples.append({
                        "video_path": os.path.join(fake_dir, fname),
                        "label": 1,  # Fake
                        "is_fake": True,
                        "method": method
                    })

        print(f"[FF++] Total: {len(samples)} videos")
        real_count = sum(1 for s in samples if s["label"] == 0)
        fake_count = sum(1 for s in samples if s["label"] == 1)
        print(f"[FF++]   Real: {real_count}, Fake: {fake_count}")
        for method in ["original"] + self.methods:
            count = sum(1 for s in samples if s["method"] == method)
            print(f"[FF++]   {method}: {count}")

        return samples

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames(self, frames):
        total_frames = len(frames)
        if total_frames == 0:
            return np.zeros((self.num_frames, self.image_size, self.image_size, 3), dtype=np.float32)

        max_start = max(0, total_frames - self.num_frames * self.frame_stride)

        if self.split == "train":
            if max_start > 0:
                start_idx = np.random.randint(0, max_start)
            else:
                start_idx = 0
        else:
            start_idx = max_start // 2 if max_start > 0 else 0

        indices = [start_idx + i * self.frame_stride for i in range(self.num_frames)]
        indices = [min(i, total_frames - 1) for i in indices]

        sampled = []
        for idx in indices:
            frame = frames[idx]
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            sampled.append(frame)

        return np.stack(sampled, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self._load_frames(sample["video_path"])
        frames = self._sample_frames(frames)

        frames = frames / 255.0
        frames = (frames - 0.5) / 0.5

        if self.transform:
            frames = self.transform(frames)

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        return {
            "frames": frames,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "is_fake": sample["is_fake"],
            "video_path": sample["video_path"],
            "method": sample["method"],
        }
