"""
P7 Self-Supervised Dataset Utilities

Dataset utilities for self-supervised learning of physiological signals.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


class UnlabeledTemporalDataset(Dataset):
    """
    Dataset for unlabeled temporal sequences.

    Suitable for self-supervised learning where ground truth labels are not required.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],  # List of (T, F) or (T,) sequences
        subject_ids: Optional[List[int]] = None,  # Subject identifiers
        fps: float = 30.0,
        min_length: int = 64
    ):
        """
        Args:
            sequences: List of temporal sequences
            subject_ids: Optional subject IDs for avoiding same-subject negatives
            fps: Frames per second
            min_length: Minimum sequence length
        """
        self.sequences = []
        self.subject_ids = []

        for i, seq in enumerate(sequences):
            if len(seq) >= min_length:
                self.sequences.append(seq)
                self.subject_ids.append(subject_ids[i] if subject_ids else i)

        self.fps = fps

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        subject_id = self.subject_ids[idx]

        # Convert to tensor
        if isinstance(seq, list):
            seq = np.array(seq)

        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)

        return {
            "features": torch.from_numpy(seq.astype(np.float32)),
            "subject_id": torch.tensor(subject_id, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long)
        }


class SyntheticSelfSupervisedDataset(Dataset):
    """
    Synthetic dataset for testing self-supervised learning.

    Generates temporal signals with cardiac modulation for testing.
    """

    def __init__(
        self,
        num_samples: int = 100,
        sequence_length: int = 128,
        feature_dim: int = 3,
        fps: float = 30.0,
        bpm_range: Tuple[float, float] = (60.0, 100.0),
        noise_std: float = 0.1,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.fps = fps
        self.bpm_range = bpm_range
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

        # Pre-generate all signals
        self.signals = []
        self.bpms = []

        for _ in range(num_samples):
            bpm = self.rng.uniform(*bpm_range)
            bpm_hz = bpm / 60.0
            t = np.arange(sequence_length) / fps

            # Cardiac signal
            cardiac = np.sin(2 * np.pi * bpm_hz * t) + 0.5 * np.sin(4 * np.pi * bpm_hz * t)
            cardiac = (cardiac - cardiac.min()) / (cardiac.max() - cardiac.min() + 1e-9)

            # Add noise
            noise = self.rng.randn(sequence_length, feature_dim).astype(np.float32) * noise_std
            signal = cardiac[:, np.newaxis] + noise
            signal = np.clip(signal, 0, 1).astype(np.float32)

            self.signals.append(signal)
            self.bpms.append(bpm)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.signals[idx]),
            "bpm": torch.tensor(self.bpms[idx], dtype=torch.float32),
            "subject_id": torch.tensor(idx // 10, dtype=torch.long),  # Group by 10
            "index": torch.tensor(idx, dtype=torch.long)
        }


class MultiViewDataset(Dataset):
    """
    Dataset that generates multiple views per sample.

    Used for contrastive learning.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        num_views: int = 2,
        crop_length: Optional[int] = None,
        augment_fn: Optional[Callable] = None
    ):
        """
        Args:
            base_dataset: Base dataset to create views from
            num_views: Number of views to generate per sample
            crop_length: Length to crop sequences to
            augment_fn: Optional augmentation function
        """
        self.base_dataset = base_dataset
        self.num_views = num_views
        self.crop_length = crop_length
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with multiple views."""
        sample = self.base_dataset[idx]
        features = sample["features"]

        # Apply optional cropping
        if self.crop_length is not None and features.shape[0] > self.crop_length:
            start = np.random.randint(0, features.shape[0] - self.crop_length + 1)
            features = features[start:start+self.crop_length]

        # Create multiple views
        views = []
        for _ in range(self.num_views):
            view = features.clone()

            # Apply augmentations
            if self.augment_fn is not None:
                result = self.augment_fn(view)
                # Handle both list and tensor results
                if isinstance(result, list):
                    view = result[0]  # Take first if list
                else:
                    view = result

            views.append(view)

        return {
            "features": features,  # Original
            "views": torch.stack(views),  # (num_views, T, F)
            "subject_id": sample.get("subject_id", torch.tensor(0)),
            "index": sample.get("index", torch.tensor(idx))
        }


def collate_ssl_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for self-supervised batches.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch with views
    """
    if len(batch) == 0:
        return {}

    # Stack original features
    features = torch.stack([b["features"] for b in batch])

    # Stack views (each is num_views x T x F)
    views = torch.stack([b["views"] for b in batch])  # (B, num_views, T, F)

    # Subject IDs
    subject_ids = torch.stack([b["subject_id"] for b in batch])

    # Indices
    indices = torch.stack([b["index"] for b in batch])

    return {
        "features": features,
        "views": views,
        "subject_ids": subject_ids,
        "indices": indices
    }


def create_ssl_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False
) -> DataLoader:
    """
    Create DataLoader for self-supervised learning.

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
        collate_fn=collate_ssl_batch,
        drop_last=drop_last
    )


def prepare_ssl_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple:
    """
    Prepare batch for SSL training.

    Args:
        batch: Batch from DataLoader
        device: Target device

    Returns:
        Tuple of (features, views, indices, subject_ids)
    """
    features = batch["features"].to(device)
    views = batch["views"].to(device)  # (B, num_views, T, F)
    indices = batch["indices"].to(device)
    subject_ids = batch["subject_ids"].to(device)

    return features, views, indices, subject_ids


def flatten_views(views: torch.Tensor) -> torch.Tensor:
    """
    Flatten batch of views for contrastive loss.

    Args:
        views: (B, num_views, T, F) tensor

    Returns:
        (B*num_views, T, F) tensor
    """
    B, V, T, F = views.shape
    return views.reshape(B * V, T, F)
