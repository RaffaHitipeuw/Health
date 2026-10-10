"""
P5 Neural Model Base Classes

Provides shared interfaces, configurations, and utilities for neural rPPG models.

Architecture:
    All models inherit from BaseModel
    Consistent input/output tensor shapes
    Unified inference interface
    Checkpoint saving/loading
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Base configuration for all neural models."""
    name: str = "base_model"
    input_channels: int = 3  # RGB
    sequence_length: int = 128  # frames
    fps: float = 30.0
    cardiac_band_hz: Tuple[float, float] = (0.833, 3.0)
    device: str = "cpu"
    seed: int = 42


@dataclass
class ModelOutput:
    """Standard output from all neural models."""
    bpm: torch.Tensor  # Heart rate estimate
    bpm_confidence: torch.Tensor  # Model confidence [0, 1]
    signal_quality: torch.Tensor  # Signal quality estimate [0, 1]
    bvp: Optional[torch.Tensor] = None  # BVP waveform if available
    features: Optional[Dict[str, torch.Tensor]] = None  # Intermediate features


class BaseModel(nn.Module):
    """Base class for all P5 neural models.

    All models should inherit from this and implement forward().
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass - must be implemented by subclasses.

        Args:
            x: Input tensor of shape (B, T, C) or (B, T, H, W, C)

        Returns:
            ModelOutput with estimated outputs
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor to standard format.

        Args:
            x: Raw input

        Returns:
            Preprocessed tensor ready for forward()
        """
        # Default: just move to device
        return x.to(self.device)

    def postprocess(self, raw_output: torch.Tensor) -> torch.Tensor:
        """Postprocess raw output if needed.

        Args:
            raw_output: Raw model output

        Returns:
            Processed output
        """
        return raw_output

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state to save
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> Tuple['BaseModel', Optional[torch.optim.Optimizer]]:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load to

        Returns:
            Tuple of (model, optimizer) - optimizer may be None
        """
        checkpoint = torch.load(path, map_location=device)

        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = None
        if "optimizer_state_dict" in checkpoint:
            # Optimizer would need to be recreated with model parameters
            pass

        return model, optimizer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get torch device (CUDA if available."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
