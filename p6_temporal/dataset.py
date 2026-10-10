"""
P6 Temporal Dataset Utilities

Dataset utilities for temporal physiological models.

Supports:
    - Synthetic temporal signal generation
    - Sliding window extraction
    - Variable-length sequence handling
    - Proper batching with padding and masking
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import dataclasses


@dataclasses.dataclass
class TemporalBatch:
    """Batch structure for temporal models."""
    features: torch.Tensor  # (B, T, F) or (B, T, H, W, C)
    bpm: torch.Tensor  # (B,) ground truth BPM
    mask: Optional[torch.Tensor] = None  # (B, T) True for padded positions
    quality: Optional[torch.Tensor] = None  # (B,) signal quality
    confidence: Optional[torch.Tensor] = None  # (B,) confidence
    bvp: Optional[torch.Tensor] = None  # (B, T) ground truth BVP waveform
    subject_id: Optional[torch.Tensor] = None  # (B,) for subject-aware batching


class TemporalSignalDataset(Dataset):
    """Dataset for temporal physiological signal sequences.

    Handles variable-length sequences with proper masking.
    """

    def __init__(
        self,
        signals: np.ndarray,  # (N, T, F) or list of variable-length arrays
        targets: np.ndarray,  # (N,) BPM values
        bvp: Optional[np.ndarray] = None,  # (N, T) BVP waveforms
        quality: Optional[np.ndarray] = None,  # (N,) quality scores
        subject_ids: Optional[np.ndarray] = None,  # (N,) subject identifiers
        fps: float = 30.0,
        pad_value: float = 0.0
    ):
        self.signals = signals
        self.targets = targets
        self.bvp = bvp
        self.quality = quality
        self.subject_ids = subject_ids
        self.fps = fps
        self.pad_value = pad_value

        # Validate shapes
        assert len(signals) == len(targets), "Signals and targets must have same length"
        if bvp is not None:
            assert len(bvp) == len(targets), "BVP and targets must have same length"
        if quality is not None:
            assert len(quality) == len(targets), "Quality and targets must have same length"

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        signal = self.signals[idx]
        target = self.targets[idx]

        item = {
            "features": torch.from_numpy(signal.astype(np.float32)),
            "bpm": torch.tensor(target, dtype=torch.float32),
        }

        if self.bvp is not None:
            item["bvp"] = torch.from_numpy(self.bvp[idx].astype(np.float32))

        if self.quality is not None:
            item["quality"] = torch.tensor(self.quality[idx], dtype=torch.float32)

        if self.subject_ids is not None:
            item["subject_id"] = torch.tensor(self.subject_ids[idx], dtype=torch.long)

        return item


class SlidingWindowDataset(Dataset):
    """Dataset with sliding window extraction for long sequences.

    Extracts fixed-length windows from longer sequences.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],  # List of (T_i, F) variable-length sequences
        targets: List[float],  # List of target BPM values per sequence
        window_length: int = 128,
        stride: int = 64,
        min_length: int = 32,
        pad: bool = False
    ):
        self.window_length = window_length
        self.stride = stride
        self.min_length = min_length
        self.pad = pad

        # Extract windows from each sequence
        self.windows = []
        self.window_targets = []
        self.original_indices = []

        for seq_idx, (seq, target) in enumerate(zip(sequences, targets)):
            T, F = seq.shape if seq.ndim > 1 else (len(seq), 1)

            if T < min_length:
                if pad:
                    # Pad short sequences
                    padded = np.pad(seq, ((0, window_length - T), (0, 0)) if seq.ndim > 1 else ((0, window_length - T),))
                    self.windows.append(padded)
                    self.window_targets.append(target)
                    self.original_indices.append(seq_idx)
                continue

            # Sliding window extraction
            for start in range(0, T - window_length + 1, stride):
                end = start + window_length
                self.windows.append(seq[start:end])
                self.window_targets.append(target)
                self.original_indices.append(seq_idx)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        target = self.window_targets[idx]

        # Ensure 2D array
        if window.ndim == 1:
            window = window.reshape(-1, 1)

        return {
            "features": torch.from_numpy(window.astype(np.float32)),
            "bpm": torch.tensor(target, dtype=torch.float32),
            "original_index": self.original_indices[idx]
        }


class SyntheticTemporalDataset(Dataset):
    """Synthetic dataset for testing temporal models.

    Generates temporal signals with cardiac modulation.
    """

    def __init__(
        self,
        num_samples: int = 100,
        sequence_length: int = 128,
        feature_dim: int = 3,
        fps: float = 30.0,
        bpm_range: Tuple[float, float] = (60.0, 100.0),
        noise_std: float = 0.1,
        seed: int = 42,
        include_bvp: bool = True
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.fps = fps
        self.bpm_range = bpm_range
        self.noise_std = noise_std
        self.include_bvp = include_bvp
        self.rng = np.random.RandomState(seed)

        # Pre-generate all targets and waveforms
        self.targets = self.rng.uniform(bpm_range[0], bpm_range[1], num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bpm = self.targets[idx]
        bpm_hz = bpm / 60.0
        t = np.arange(self.sequence_length) / self.fps

        # Generate cardiac BVP waveform
        bvp = np.sin(2 * np.pi * bpm_hz * t) + 0.5 * np.sin(4 * np.pi * bpm_hz * t)
        bvp = (bvp - bvp.min()) / (bvp.max() - bvp.min() + 1e-9)

        # Generate features with cardiac modulation
        noise = self.rng.randn(self.sequence_length, self.feature_dim).astype(np.float32) * self.noise_std
        features = bvp[:, np.newaxis] + noise
        features = np.clip(features, 0, 1).astype(np.float32)

        item = {
            "features": torch.from_numpy(features),
            "bpm": torch.tensor(bpm, dtype=torch.float32),
        }

        if self.include_bvp:
            item["bvp"] = torch.from_numpy(bvp.astype(np.float32))

        return item


def collate_temporal_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for temporal batches.

    Handles variable-length sequences with padding and masking.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch with padding and mask
    """
    if len(batch) == 0:
        return {}

    # Determine feature dimensions
    sample_features = batch[0]["features"]
    is_video = sample_features.dim() == 5

    # Extract components
    features_list = [b["features"] for b in batch]
    bpm_list = torch.stack([b["bpm"] for b in batch])

    # Pad features to same length
    lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)
    max_len = lengths.max().item()

    if is_video:
        # Video format: (T, H, W, C)
        B = len(features_list)
        T, H, W, C = features_list[0].shape
        padded_features = torch.zeros(B, max_len, H, W, C, dtype=features_list[0].dtype)
    else:
        # Feature format: (T, F)
        feat_dim = features_list[0].shape[-1]
        padded_features = torch.zeros(len(features_list), max_len, feat_dim, dtype=features_list[0].dtype)

    # Fill padded features
    for i, feat in enumerate(features_list):
        length = feat.shape[0]
        padded_features[i, :length] = feat

    # Create mask (True for padded positions)
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)

    collated = {
        "features": padded_features,
        "bpm": bpm_list,
        "mask": mask,
        "lengths": lengths
    }

    # Optional components
    if "bvp" in batch[0]:
        bvp_list = [b["bvp"] for b in batch]
        padded_bvp = torch.zeros(len(batch), max_len, dtype=bvp_list[0].dtype)
        for i, bvp in enumerate(bvp_list):
            padded_bvp[i, :len(bvp)] = bvp
        collated["bvp"] = padded_bvp

    if "quality" in batch[0]:
        collated["quality"] = torch.stack([b["quality"] for b in batch])

    if "confidence" in batch[0]:
        collated["confidence"] = torch.stack([b["confidence"] for b in batch])

    if "subject_id" in batch[0]:
        collated["subject_id"] = torch.stack([b["subject_id"] for b in batch])

    if "original_index" in batch[0]:
        collated["original_index"] = torch.tensor([b["original_index"] for b in batch])

    return collated


def create_temporal_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False
) -> DataLoader:
    """Create a DataLoader for temporal datasets.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        drop_last: Whether to drop incomplete last batch

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_temporal_batch,
        drop_last=drop_last
    )
