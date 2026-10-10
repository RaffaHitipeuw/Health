"""
P8 Multi-Task Dataset Utilities

Dataset utilities for multi-task physiological modeling.
Supports variable-length sequences and missing labels.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass


@dataclass
class MultiTaskSample:
    """Single sample for multi-task learning."""
    features: np.ndarray                    # (T, F) input features
    hr: Optional[float] = None             # Heart rate in BPM
    bvp: Optional[np.ndarray] = None        # (T,) BVP waveform
    sqi: Optional[float] = None            # Signal quality [0, 1]
    confidence: Optional[float] = None     # Confidence [0, 1]
    subject_id: Optional[int] = None        # Subject ID
    index: int = 0                          # Sample index


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task physiological learning.

    Supports missing labels - tasks without targets will have
    availability masks set accordingly.
    """

    def __init__(
        self,
        features: List[np.ndarray],
        hr: Optional[List[float]] = None,
        bvp: Optional[List[np.ndarray]] = None,
        sqi: Optional[List[float]] = None,
        confidence: Optional[List[float]] = None,
        subject_ids: Optional[List[int]] = None,
        fps: float = 30.0,
        min_length: int = 32
    ):
        """
        Args:
            features: List of (T, F) feature arrays
            hr: List of HR values (None for missing)
            bvp: List of BVP waveforms (None for missing)
            sqi: List of SQI values (None for missing)
            confidence: List of confidence values (None for missing)
            subject_ids: List of subject IDs
            fps: Frames per second
            min_length: Minimum sequence length
        """
        self.features = []
        self.hr = []
        self.bvp = []
        self.sqi = []
        self.confidence = []
        self.subject_ids = []
        self.fps = fps

        # Validate and filter
        for i, feat in enumerate(features):
            if len(feat) >= min_length:
                self.features.append(feat.astype(np.float32))
                self.hr.append(hr[i] if hr is not None else None)
                self.bvp.append(bvp[i] if bvp is not None else None)
                self.sqi.append(sqi[i] if sqi is not None else None)
                self.confidence.append(confidence[i] if confidence is not None else None)
                self.subject_ids.append(subject_ids[i] if subject_ids else i)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with all available targets."""
        feat = self.features[idx]

        item = {
            "features": torch.from_numpy(feat),
            "hr_available": torch.tensor(self.hr[idx] is not None, dtype=torch.bool),
            "bvp_available": torch.tensor(self.bvp[idx] is not None, dtype=torch.bool),
            "sqi_available": torch.tensor(self.sqi[idx] is not None, dtype=torch.bool),
            "confidence_available": torch.tensor(self.confidence[idx] is not None, dtype=torch.bool),
            "subject_id": torch.tensor(self.subject_ids[idx], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

        # Add available targets
        if self.hr[idx] is not None:
            item["hr"] = torch.tensor(self.hr[idx], dtype=torch.float32)

        if self.bvp[idx] is not None:
            bvp = self.bvp[idx]
            if isinstance(bvp, np.ndarray):
                item["bvp"] = torch.from_numpy(bvp.astype(np.float32))
            else:
                item["bvp"] = torch.tensor(bvp, dtype=torch.float32)

        if self.sqi[idx] is not None:
            item["sqi"] = torch.tensor(self.sqi[idx], dtype=torch.float32)

        if self.confidence[idx] is not None:
            item["confidence"] = torch.tensor(self.confidence[idx], dtype=torch.float32)

        return item


class SyntheticMultiTaskDataset(Dataset):
    """
    Synthetic dataset for testing multi-task learning.

    Generates temporal signals with cardiac modulation and
    ground truth for all tasks.
    """

    def __init__(
        self,
        num_samples: int = 100,
        sequence_length: int = 128,
        feature_dim: int = 3,
        fps: float = 30.0,
        hr_range: Tuple[float, float] = (55.0, 100.0),
        noise_std: float = 0.1,
        missing_label_prob: float = 0.0,
        seed: int = 42,
        include_all_tasks: bool = True
    ):
        """
        Args:
            num_samples: Number of samples
            sequence_length: Sequence length
            feature_dim: Feature dimension
            fps: Frames per second
            hr_range: HR range in BPM
            noise_std: Noise standard deviation
            missing_label_prob: Probability of missing labels (for testing)
            seed: Random seed
            include_all_tasks: Whether to include all tasks
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.fps = fps
        self.hr_range = hr_range
        self.noise_std = noise_std
        self.missing_label_prob = missing_label_prob
        self.include_all_tasks = include_all_tasks
        self.rng = np.random.RandomState(seed)

        # Generate all samples
        self.samples = []
        for i in range(num_samples):
            sample = self._generate_sample(i)
            self.samples.append(sample)

    def _generate_sample(self, idx: int) -> Dict[str, Any]:
        """Generate a single synthetic sample."""
        # Random HR
        hr = self.rng.uniform(*self.hr_range)
        hr_hz = hr / 60.0
        t = np.arange(self.sequence_length) / self.fps

        # BVP waveform (ground truth)
        bvp = np.sin(2 * np.pi * hr_hz * t) + 0.5 * np.sin(4 * np.pi * hr_hz * t)
        bvp = (bvp - bvp.min()) / (bvp.max() - bvp.min() + 1e-9)

        # Features with cardiac modulation
        noise = self.rng.randn(self.sequence_length, self.feature_dim).astype(np.float32) * self.noise_std
        features = bvp[:, np.newaxis] + noise
        features = np.clip(features, 0, 1).astype(np.float32)

        # SQI (correlated with noise level)
        sqi = 1.0 / (1.0 + self.noise_std * 10)
        sqi = np.clip(sqi + self.rng.uniform(-0.1, 0.1), 0, 1)

        # Confidence (based on signal quality)
        confidence = sqi + self.rng.uniform(-0.1, 0.1)
        confidence = np.clip(confidence, 0, 1)

        sample = {
            "features": features,
            "hr": hr,
            "bvp": bvp.astype(np.float32),
            "sqi": sqi,
            "confidence": confidence,
            "subject_id": idx // 10  # Group by 10
        }

        # Apply missing label probability
        if not self.include_all_tasks and self.missing_label_prob > 0:
            if self.rng.random() < self.missing_label_prob:
                task = self.rng.choice(["hr", "bvp", "sqi", "confidence"])
                sample[task] = None

        return sample

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        sample = self.samples[idx]
        feat = sample["features"]

        item = {
            "features": torch.from_numpy(feat),
            "hr": torch.tensor(sample["hr"], dtype=torch.float32),
            "hr_available": torch.tensor(sample["hr"] is not None, dtype=torch.bool),
            "bvp": torch.from_numpy(sample["bvp"]),
            "bvp_available": torch.tensor(sample["bvp"] is not None, dtype=torch.bool),
            "sqi": torch.tensor(sample["sqi"], dtype=torch.float32),
            "sqi_available": torch.tensor(sample["sqi"] is not None, dtype=torch.bool),
            "confidence": torch.tensor(sample["confidence"], dtype=torch.float32),
            "confidence_available": torch.tensor(sample["confidence"] is not None, dtype=torch.bool),
            "subject_id": torch.tensor(sample["subject_id"], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

        return item


def collate_multitask_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for multi-task batches.

    Handles variable-length sequences and missing labels.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch with padding and masks
    """
    if len(batch) == 0:
        return {}

    # Determine dimensions
    sample_features = batch[0]["features"]
    is_video = sample_features.dim() == 5

    # Extract components
    features_list = [b["features"] for b in batch]

    # Pad features
    lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)
    max_len = lengths.max().item()

    if is_video:
        B = len(features_list)
        T, H, W, C = features_list[0].shape
        padded_features = torch.zeros(B, max_len, H, W, C, dtype=features_list[0].dtype)
    else:
        feat_dim = features_list[0].shape[-1]
        padded_features = torch.zeros(len(features_list), max_len, feat_dim, dtype=features_list[0].dtype)

    # Fill padded features
    for i, feat in enumerate(features_list):
        length = feat.shape[0]
        padded_features[i, :length] = feat

    # Create mask
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)

    collated = {
        "features": padded_features,
        "mask": mask,
        "lengths": lengths
    }

    # HR targets (may be missing)
    hr_tensors = []
    hr_available = torch.zeros(len(batch), dtype=torch.bool)
    for i, b in enumerate(batch):
        if "hr" in b and "hr_available" in b and b["hr_available"].item():
            hr_tensors.append(b["hr"])
            hr_available[i] = True

    if hr_tensors:
        collated["hr"] = torch.stack(hr_tensors)
    else:
        collated["hr"] = torch.zeros(len(batch))
    collated["hr_available"] = hr_available

    # BVP targets (may be missing)
    bvp_tensors = []
    bvp_available = torch.zeros(len(batch), dtype=torch.bool)
    for i, b in enumerate(batch):
        if "bvp" in b and "bvp_available" in b and b["bvp_available"].item():
            bvp = b["bvp"]
            if bvp.shape[0] < max_len:
                bvp_padded = torch.zeros(max_len)
                bvp_padded[:bvp.shape[0]] = bvp
                bvp_tensors.append(bvp_padded)
            else:
                bvp_tensors.append(bvp[:max_len])
            bvp_available[i] = True
        else:
            bvp_tensors.append(torch.zeros(max_len))
            bvp_available[i] = False

    collated["bvp"] = torch.stack(bvp_tensors)
    collated["bvp_available"] = bvp_available

    # SQI targets
    sqi_tensors = []
    sqi_available = torch.zeros(len(batch), dtype=torch.bool)
    for i, b in enumerate(batch):
        if "sqi" in b and "sqi_available" in b and b["sqi_available"].item():
            sqi_tensors.append(b["sqi"])
            sqi_available[i] = True
        else:
            sqi_tensors.append(torch.tensor(0.5))
            sqi_available[i] = False

    collated["sqi"] = torch.stack(sqi_tensors)
    collated["sqi_available"] = sqi_available

    # Confidence targets
    conf_tensors = []
    conf_available = torch.zeros(len(batch), dtype=torch.bool)
    for i, b in enumerate(batch):
        if "confidence" in b and "confidence_available" in b and b["confidence_available"].item():
            conf_tensors.append(b["confidence"])
            conf_available[i] = True
        else:
            conf_tensors.append(torch.tensor(0.5))
            conf_available[i] = False

    collated["confidence"] = torch.stack(conf_tensors)
    collated["confidence_available"] = conf_available

    # Metadata
    if "subject_id" in batch[0]:
        collated["subject_id"] = torch.stack([b["subject_id"] for b in batch])
    if "index" in batch[0]:
        collated["indices"] = torch.stack([b["index"] for b in batch])

    return collated


def create_multitask_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create DataLoader for multi-task datasets.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        drop_last: Whether to drop incomplete last batch
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_multitask_batch,
        drop_last=drop_last,
        pin_memory=pin_memory
    )


def create_partial_labels_batch(
    num_samples: int,
    tasks_available: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Create a batch with partial labels for testing missing label handling.

    Args:
        num_samples: Number of samples
        tasks_available: List of tasks with available labels

    Returns:
        Dictionary with batch data
    """
    if tasks_available is None:
        tasks_available = ["hr", "bvp"]

    batch = {
        "features": torch.randn(num_samples, 128, 3),
        "mask": torch.zeros(num_samples, 128, dtype=torch.bool),
        "hr_available": torch.tensor("hr" in tasks_available),
        "bvp_available": torch.tensor("bvp" in tasks_available),
        "sqi_available": torch.tensor("sqi" in tasks_available),
        "confidence_available": torch.tensor("confidence" in tasks_available),
    }

    # Add available targets
    if "hr" in tasks_available:
        batch["hr"] = torch.rand(num_samples) * 60 + 40  # HR in [40, 100]
    else:
        batch["hr"] = torch.zeros(num_samples)

    if "bvp" in tasks_available:
        batch["bvp"] = torch.randn(num_samples, 128)
    else:
        batch["bvp"] = torch.zeros(num_samples, 128)

    if "sqi" in tasks_available:
        batch["sqi"] = torch.rand(num_samples)
    else:
        batch["sqi"] = torch.zeros(num_samples)

    if "confidence" in tasks_available:
        batch["confidence"] = torch.rand(num_samples)
    else:
        batch["confidence"] = torch.zeros(num_samples)

    # Expand availability to batch size
    if batch["hr_available"].item():
        batch["hr_available"] = batch["hr_available"].expand(num_samples)
    if batch["bvp_available"].item():
        batch["bvp_available"] = batch["bvp_available"].expand(num_samples)
    if batch["sqi_available"].item():
        batch["sqi_available"] = batch["sqi_available"].expand(num_samples)
    if batch["confidence_available"].item():
        batch["confidence_available"] = batch["confidence_available"].expand(num_samples)

    return batch
