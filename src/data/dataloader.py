import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataset import DeepfakeDataset
from src.data.transforms import VideoTransform
from src.data.cutmix import CutMixCollate


def _worker_init_fn(worker_id, seed):
    """Worker initialization function for DataLoader determinism."""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloader(config, split="train", shuffle=None):
    if shuffle is None:
        shuffle = (split == "train")

    transform = VideoTransform() if split == "train" else None

    dataset = DeepfakeDataset(
        data_root=config.data.data_root,
        split=split,
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
        transform=transform,
        val_ratio=config.data.val_ratio,
        seed=config.training.seed,
    )

    sampler = None
    if split == "train":
        labels = [int(s["label"]) for s in dataset.samples]
        class_counts = {}
        for lbl in labels:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        class_weights = {}
        for lbl, cnt in class_counts.items():
            class_weights[lbl] = 1.0 / cnt

        sample_weights = [class_weights[lbl] for lbl in labels]

        minority_count = min(class_counts.values())
        num_samples = 2 * minority_count

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=num_samples,
            replacement=True,
        )

        real_count = class_counts.get(0, 0)
        fake_count = class_counts.get(1, 0)
        print(f"[WeightedSampler] Real: {real_count}, Fake: {fake_count}")
        print(f"[WeightedSampler] Real weight: {class_weights.get(0, 0):.6f}, "
              f"Fake weight: {class_weights.get(1, 0):.6f}")
        print(f"[WeightedSampler] Samples per epoch: {num_samples} "
              f"(batch_size={config.data.batch_size}, ~{num_samples // config.data.batch_size} batches)")

    cutmix_p = getattr(config.training, "cutmix_p", 0.0)
    collate_fn = CutMixCollate(p=cutmix_p, alpha=getattr(config.training, "cutmix_alpha", 1.0)) if cutmix_p > 0 else None

    # Task 14: worker_init_fn for DataLoader determinism
    seed = getattr(config.training, "seed", 42)

    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        shuffle=False if sampler else shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=collate_fn,
        worker_init_fn=lambda worker_id: _worker_init_fn(worker_id, seed),
    )

    return dataloader