"""
FF++ Frame Dataset — loads pre-extracted frame PNGs instead of videos.
Expected structure (after extracting D:\FaceForensics++.zip):
    ff_root/
        original_sequences/youtube/c23/frames/<video_id>/*.png
        manipulated_sequences/<method>/c23/frames/<video_id>/*.png
"""
import os, json, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from collections import Counter


class FFFrameDataset(Dataset):
    def __init__(self, data_root, num_frames=8, frame_stride=2, image_size=224,
                 split="test", compression="c23", methods=None):
        self.data_root = data_root
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.split = split
        self.compression = compression

        if methods is None:
            self.methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        else:
            self.methods = methods

        self.samples = self._load_samples()

    def _load_samples(self):
        json_path = os.path.join(self.data_root, f"{self.split}.json")
        if not os.path.exists(json_path):
            print(f"[FF++] No {self.split}.json found at {json_path}")
            return []

        with open(json_path, "r") as f:
            pairs = json.load(f)

        samples = []
        for pair in pairs:
            if isinstance(pair, list) and len(pair) == 2:
                real_id, fake_id = str(pair[0]), str(pair[1])
            else:
                continue

            # Real sample
            real_frames_dir = os.path.join(self.data_root, "original_sequences", "youtube",
                                           self.compression, "frames", real_id)
            if os.path.exists(real_frames_dir):
                pngs = sorted([f for f in os.listdir(real_frames_dir) if f.endswith('.png')])
                if len(pngs) >= self.num_frames * self.frame_stride:
                    samples.append({
                        "frame_dir": real_frames_dir,
                        "frame_count": len(pngs),
                        "label": 0, "is_fake": False,
                        "method": "original",
                    })

            # Fake sample — dir name is {real_id}_{fake_id}
            fake_dir_name = f"{real_id}_{fake_id}"
            for method in self.methods:
                fake_frames_dir = os.path.join(self.data_root, "manipulated_sequences", method,
                                                self.compression, "frames", fake_dir_name)
                if os.path.exists(fake_frames_dir):
                    pngs = sorted([f for f in os.listdir(fake_frames_dir) if f.endswith('.png')])
                    if len(pngs) >= self.num_frames * self.frame_stride:
                        samples.append({
                            "frame_dir": fake_frames_dir,
                            "frame_count": len(pngs),
                            "label": 1, "is_fake": True,
                            "method": method,
                        })

        print(f"[FF++ Frames] Split={self.split}, Pairs={len(pairs)}, Samples: {len(samples)}")
        real_count = sum(1 for s in samples if not s["is_fake"])
        fake_count = sum(1 for s in samples if s["is_fake"])
        print(f"[FF++ Frames]   Real: {real_count}, Fake: {fake_count}")

        from collections import Counter
        method_counts = Counter(s["method"] for s in samples)
        for method, count in sorted(method_counts.items()):
            print(f"[FF++ Frames]   {method}: {count}")

        return samples

    def _load_frames(self, frame_dir):
        pngs = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        total = len(pngs)
        max_start = max(0, total - self.num_frames * self.frame_stride)
        start_idx = max_start // 2 if max_start > 0 else 0
        indices = [start_idx + i * self.frame_stride for i in range(self.num_frames)]
        indices = [min(i, total - 1) for i in indices]

        frames = []
        for idx in indices:
            path = os.path.join(frame_dir, pngs[idx])
            frame = cv2.imread(path)
            if frame is None:
                frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame)

        return np.stack(frames, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self._load_frames(sample["frame_dir"])
        frames = frames / 255.0
        frames = (frames - 0.5) / 0.5
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        return {"frames": frames,
                "label": torch.tensor(sample["label"], dtype=torch.float32),
                "is_fake": sample["is_fake"],
                "method": sample["method"],
                "frame_dir": sample["frame_dir"]}
