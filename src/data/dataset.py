import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    def __init__(self, data_root, split="train",
                 num_frames=8, frame_stride=2, image_size=224,
                 transform=None, val_ratio=0.2, seed=42):
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.transform = transform
        self.val_ratio = val_ratio
        self.seed = seed

        self.samples = self._load_samples()

    def _parse_test_list(self, list_path):
        """解析 List_of_testing_videos.txt，返回 forward-slash 归一化的 test 集合与标签映射。

        Celeb-DF 官方格式：每行 "<label> <rel_path>"，label=1 表示真实视频、label=0
        表示伪造视频（与采集阶段的 label 约定相反，故在 _load_samples 中用 1 - orig_label
        换算为 0=real / 1=fake）。路径统一归一化为 forward-slash 以便与 all_videos 精确匹配。
        """
        test_set = set()
        label_map = {}
        if not os.path.exists(list_path):
            return test_set, label_map

        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                label_str, rel_path = parts
                label = int(label_str)
                rel_path = rel_path.strip().replace(os.sep, "/")
                test_set.add(rel_path)
                label_map[rel_path] = label

        return test_set, label_map

    def _collect_real_videos(self, real_dir, rel_prefix):
        videos = {}
        if not os.path.exists(real_dir):
            return videos

        for fname in os.listdir(real_dir):
            if fname.endswith((".mp4", ".avi", ".mov")):
                full_path = os.path.join(real_dir, fname)
                rel_path = os.path.join(rel_prefix, fname)
                videos[rel_path] = {"video_path": full_path, "label": 0, "is_fake": False}
        return videos

    def _collect_fake_videos(self, synthesis_dir):
        """递归采集 Celeb-synthesis 下所有伪造视频。

        改用 os.walk 而非硬编码三层目录遍历，避免任意嵌套层级下的漏采或重复计数；
        rel_path 统一为相对 data_root 的路径（与 List_of_testing_videos.txt 一致），
        以便在 _load_samples 中做精确匹配。
        """
        videos = {}
        if not os.path.exists(synthesis_dir):
            return videos

        exts = (".mp4", ".avi", ".mov")
        for dirpath, _dirs, files in os.walk(synthesis_dir):
            for fname in files:
                if fname.lower().endswith(exts):
                    full_path = os.path.join(dirpath, fname)
                    rel_path = os.path.join("Celeb-synthesis", os.path.relpath(full_path, synthesis_dir))
                    videos[rel_path] = {"video_path": full_path, "label": 1, "is_fake": True}

        return videos

    def _load_samples(self):
        list_path = os.path.join(self.data_root, "List_of_testing_videos.txt")
        test_set, test_label_map = self._parse_test_list(list_path)

        celebrity_real = self._collect_real_videos(
            os.path.join(self.data_root, "Celeb-real"), "Celeb-real"
        )
        youtube_real = self._collect_real_videos(
            os.path.join(self.data_root, "YouTube-real"), "YouTube-real"
        )
        synthesis_fake = self._collect_fake_videos(
            os.path.join(self.data_root, "Celeb-synthesis")
        )

        all_videos = {}
        all_videos.update(celebrity_real)
        all_videos.update(youtube_real)
        all_videos.update(synthesis_fake)

        test_samples = []
        train_val_samples = []
        matched_test_paths = set()

        for rel_path, sample in all_videos.items():
            rel_path_normalized = rel_path.replace(os.sep, "/")
            sample["rel_path"] = rel_path

            # 精确匹配（替代脆弱的 endswith）：test 列表路径已归一化为 forward-slash
            if rel_path_normalized in test_set:
                test_samples.append(sample)
                matched_test_paths.add(rel_path_normalized)
                orig_label = test_label_map.get(rel_path_normalized)
                if orig_label is not None:
                    sample["label"] = 1 - orig_label
            else:
                train_val_samples.append(sample)

        # 校验：test 列表的每一条都应精确命中一个已采集视频，否则说明路径格式不匹配
        unmatched = test_set - matched_test_paths
        if unmatched:
            print(f"[Celeb-DF++] WARNING: {len(unmatched)} 个 test 列表条目未匹配到任何视频（可能为路径格式不匹配）")
            for p in sorted(unmatched)[:10]:
                print(f"    - {p}")

        rng = random.Random(self.seed)
        rng.shuffle(train_val_samples)

        split_idx = int(len(train_val_samples) * (1 - self.val_ratio))
        train_samples = train_val_samples[:split_idx]
        val_samples = train_val_samples[split_idx:]

        print(f"[Celeb-DF++] Total: {len(all_videos)} videos")
        print(f"[Celeb-DF++]   Train: {len(train_samples)}")
        print(f"[Celeb-DF++]   Val:   {len(val_samples)}")
        print(f"[Celeb-DF++]   Test:  {len(test_samples)}")

        if self.split == "train":
            return train_samples
        elif self.split == "val":
            return val_samples
        elif self.split == "test":
            return test_samples
        else:
            return train_samples

    def _load_frames(self, video_path):
        """Load only sampled frames (not all frames) for efficiency."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []

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

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame)

        cap.release()

        if len(frames) < self.num_frames:
            # Pad with last frame if video too short
            while len(frames) < self.num_frames and len(frames) > 0:
                frames.append(frames[-1])

        if len(frames) == 0:
            return np.zeros((self.num_frames, self.image_size, self.image_size, 3), dtype=np.float32)

        return np.stack(frames, axis=0).astype(np.float32)

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
        }