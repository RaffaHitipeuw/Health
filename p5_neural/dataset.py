"""
P5 Dataset Utilities

Dataset loading and batch preparation for neural rPPG models.

Supports:
    Synthetic video generation for testing
    Video file loading
    Signal-based loading
    Feature-based loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, Callable
import dataclasses


@dataclasses.dataclass
class BatchTemplate:
    """Template for batch structure."""
    video: torch.Tensor  # (B, T, H, W, C) or (B*T, H, W, C)
    bpm: torch.Tensor  # (B,)
    confidence: Optional[torch.Tensor] = None  # (B,) or None
    quality: Optional[torch.Tensor] = None  # (B,) or None
    classical_features: Optional[torch.Tensor] = None  # (B, F) or None


class SyntheticVideoDataset(Dataset):
    """Synthetic dataset for testing and development.

    Generates video with cardiac signal for quick iteration.
    """

    def __init__(
        self,
        num_samples: int = 100,
        sequence_length: int = 128,
        height: int = 64,
        width: int = 64,
        channels: int = 3,
        fps: float = 30.0,
        bpm_range: Tuple[float, float] = (60.0, 100.0),
        noise_std: float = 0.1,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.height = height
        self.width = width
        self.channels = channels
        self.fps = fps
        self.bpm_range = bpm_range
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

        # Pre-generate all targets
        self.targets = self.rng.uniform(bpm_range[0], bpm_range[1], num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate video with cardiac signal
        bpm = self.targets[idx]
        bpm_hz = bpm / 60.0
        t = np.arange(self.sequence_length) / self.fps

        # Cardiac signal
        cardiac = np.sin(2 * np.pi * bpm_hz * t) * 0.5 + 0.5

        # Video with cardiac modulation
        noise = self.rng.randn(self.sequence_length, self.height, self.width, self.channels).astype(np.float32) * self.noise_std
        video = cardiac[:, None, None, None] + noise
        video = np.clip(video, 0, 1).astype(np.float32)

        return {
            "video": torch.from_numpy(video),  # (T, H, W, C)
            "bpm": torch.tensor([bpm], dtype=torch.float32),
            "confidence": torch.tensor([1.0]),  # Placeholder
            "quality": torch.tensor([1.0]),  # Placeholder
        }

    def to_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Create DataLoader from this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )


class SignalDataset(Dataset):
    """Dataset from pre-computed temporal signals.

    For use when video processing is expensive.
    """

    def __init__(
        self,
        signals: np.ndarray,  # (N, T, F)
        targets: np.ndarray,  # (N,) BPM values
        qualities: Optional[np.ndarray] = None  # (N,) quality values
    ):
        self.signals = torch.from_numpy(signals.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.qualities = torch.from_numpy(qualities.astype(np.float32)) if qualities is not None else None

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "video": self.signals[idx],  # (T, F)
            "bpm": self.targets[idx:idx+1],
            "quality": self.qualities[idx:idx+1] if self.qualities is not None else torch.tensor([1.0])
        }


def collate_video_batch(batch):
    """Collate video batch handling different formats."""
    # Ensure consistent tensor type
    videos = [b["video"] for b in batch]
    bpms = torch.stack([b["bpm"] for b in batch])
    confs = torch.stack([b.get("confidence", torch.tensor([1.0])) for b in batch])
    quals = torch.stack([b.get("quality", torch.tensor([1.0])) for b in batch])

    # Handle video shapes
    if videos[0].dim() == 5:
        # (T, H, W, C) -> (B, T, H, W, C)
        max_T = max(v.shape[0] for v in videos)
        B = len(videos)
        H, W, C = videos[0].shape[1:]
        padded = torch.zeros(B, max_T, H, W, C)
        for i, v in enumerate(videos):
            padded[i, :len(v)] = v
        videos = padded
    else:
        videos = torch.stack(videos)

    return {
        "video": videos,
        "bpm": bpms.squeeze(-1),
        "confidence": confs.squeeze(-1),
        "quality": quals.squeeze(-1)
    }
