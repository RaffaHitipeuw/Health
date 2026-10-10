"""
P7 Self-Supervised Augmentations

Physiologically-appropriate augmentation strategies for temporal signals.

Augmentations are designed to preserve physiological information while
creating diverse views of the same underlying signal.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for augmentations."""
    noise_std: float = 0.05
    dropout_prob: float = 0.1
    scale_range: Tuple[float, float] = (0.9, 1.1)
    time_shift_max: float = 0.1
    crop_ratio: Tuple[float, float] = (0.5, 1.0)
    frequency_preserve: bool = True


class TemporalCrop:
    """
    Random temporal cropping for creating views.

    Crops a random portion of the temporal sequence.
    """

    def __init__(self, min_ratio: float = 0.5, max_ratio: float = 1.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Apply temporal crop.

        Args:
            x: (T, F) or (B, T, F) tensor

        Returns:
            Tuple of (cropped_tensor, start_idx, end_idx)
        """
        if x.dim() == 2:
            T, F = x.shape
            crop_ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            crop_len = int(T * crop_ratio)
            start = np.random.randint(0, T - crop_len + 1)
            end = start + crop_len
            return x[start:end], start, end
        else:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")


class GaussianNoise:
    """
    Add Gaussian noise to signal.

    Noise should be small enough to preserve physiological patterns.
    """

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise."""
        noise = torch.randn_like(x) * self.std
        return x + noise


class SignalDropout:
    """
    Randomly dropout signal channels or timesteps.

    Can simulate missing data or sensor dropout.
    """

    def __init__(self, prob: float = 0.1, dim: int = -1):
        """
        Args:
            prob: Dropout probability
            dim: Dimension to apply dropout (0=time, 1=feature)
        """
        self.prob = prob
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply signal dropout."""
        if self.dim == -1 or self.dim == 1:
            # Feature dropout (channel-wise)
            mask = torch.rand(x.shape[-1], device=x.device) > self.prob
            return x * mask
        else:
            # Temporal dropout
            mask = torch.rand(x.shape[0], device=x.device) > self.prob
            return x * mask.unsqueeze(-1)


class TimeShift:
    """
    Random temporal shift.

    Shifts the signal in time, simulating acquisition timing variations.
    """

    def __init__(self, max_shift: float = 0.1):
        """
        Args:
            max_shift: Maximum shift as fraction of sequence length
        """
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal shift."""
        T = x.shape[0]
        max_shift_samples = int(T * self.max_shift)

        if max_shift_samples > 0:
            shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
            if shift > 0:
                return torch.cat([x[shift:], x[-1:].expand(shift, -1)], dim=0)
            elif shift < 0:
                return torch.cat([x[0:1].expand(-shift, -1), x[:shift]], dim=0)
        return x


class AmplitudeScale:
    """
    Random amplitude scaling.

    Preserves frequency content while varying amplitude.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply amplitude scaling."""
        scale = np.random.uniform(*self.scale_range)
        return x * scale


class FrequencyMask:
    """
    Mask frequency components (similar to SpecAugment for audio).

    Randomly masks bands in the frequency representation.
    """

    def __init__(self, num_masks: int = 1, mask_width: float = 0.1):
        self.num_masks = num_masks
        self.mask_width = mask_width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking."""
        T, F = x.shape
        mask_width_samples = max(1, int(F * self.mask_width))

        for _ in range(self.num_masks):
            f0 = np.random.randint(0, F - mask_width_samples + 1)
            x[:, f0:f0+mask_width_samples] = 0

        return x


class TemporalMask:
    """
    Mask temporal segments.

    Randomly masks contiguous segments in time.
    """

    def __init__(self, num_masks: int = 1, mask_width: float = 0.1):
        self.num_masks = num_masks
        self.mask_width = mask_width

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal masking.

        Returns:
            Tuple of (masked_tensor, mask)
        """
        T = x.shape[0]
        mask_width_samples = max(1, int(T * self.mask_width))

        mask = torch.ones(T, dtype=torch.bool, device=x.device)
        for _ in range(self.num_masks):
            t0 = np.random.randint(0, T - mask_width_samples + 1)
            mask[t0:t0+mask_width_samples] = False

        return x * mask.unsqueeze(-1), mask


class ComposeAugmentations:
    """
    Compose multiple augmentations.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations in sequence."""
        for t in self.transforms:
            x = t(x)
        return x


class PhysiologicalAugmenter:
    """
    Physiological-specific augmenter.

    Combines augmentations that preserve physiological information.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        if config is None:
            config = AugmentationConfig()
        self.config = config

        # Build augmentation pipeline
        self.augmentations = ComposeAugmentations([
            AmplitudeScale(config.scale_range),
            GaussianNoise(config.noise_std),
            SignalDropout(config.dropout_prob),
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply physiological augmentations."""
        return self.augmentations(x)


class MultiViewAugmenter:
    """
    Creates multiple views from the same input.

    Used for contrastive learning to create positive pairs.
    """

    def __init__(
        self,
        base_augmenter: PhysiologicalAugmenter,
        num_views: int = 2,
        crop_length: Optional[int] = None
    ):
        self.base_augmenter = base_augmenter
        self.num_views = num_views
        self.crop_length = crop_length

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Create multiple augmented views.

        Args:
            x: (T, F) input sequence

        Returns:
            List of augmented views
        """
        views = []

        for _ in range(self.num_views):
            view = x.clone()

            # Optional temporal crop
            if self.crop_length is not None and view.shape[0] > self.crop_length:
                start = np.random.randint(0, view.shape[0] - self.crop_length + 1)
                view = view[start:start+self.crop_length]

            # Apply base augmentations
            view = self.base_augmenter(view)
            views.append(view)

        return views


def create_default_augmenter(
    noise_std: float = 0.05,
    dropout_prob: float = 0.1,
    scale_range: Tuple[float, float] = (0.9, 1.1)
) -> PhysiologicalAugmenter:
    """Create default physiological augmenter."""
    config = AugmentationConfig(
        noise_std=noise_std,
        dropout_prob=dropout_prob,
        scale_range=scale_range
    )
    return PhysiologicalAugmenter(config)


def create_multiview_augmenter(
    num_views: int = 2,
    crop_length: Optional[int] = None,
    **augmenter_kwargs
) -> MultiViewAugmenter:
    """Create multi-view augmenter for contrastive learning."""
    base = create_default_augmenter(**augmenter_kwargs)
    return MultiViewAugmenter(base, num_views, crop_length)
