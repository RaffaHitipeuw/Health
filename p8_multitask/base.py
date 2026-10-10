"""
P8 Multi-Task Base Configuration

Base configurations, data classes, and utilities for multi-task
physiological modeling.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """Available task types for multi-task learning."""
    BVP = "bvp"           # BVP waveform prediction
    HR = "hr"             # Heart rate estimation
    SQI = "sqi"           # Signal quality index
    CONFIDENCE = "confidence"  # Prediction confidence/reliability


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task physiological model.

    Attributes:
        name: Model name
        input_dim: Feature dimension per timestep
        hidden_dim: Shared encoder hidden dimension
        num_encoder_layers: Number of encoder layers
        encoder_type: Type of encoder ("lstm", "gru", "transformer", "attention")
        bidirectional: Whether to use bidirectional encoder
        dropout: Dropout probability
        sequence_length: Default sequence length

        # Task enable flags
        enable_bvp: Enable BVP waveform prediction task
        enable_hr: Enable heart rate estimation task
        enable_sqi: Enable signal quality index task
        enable_confidence: Enable confidence estimation task

        # Loss weights
        bvp_weight: Weight for BVP loss
        hr_weight: Weight for HR loss
        sqi_weight: Weight for signal quality loss
        confidence_weight: Weight for confidence loss

        # Loss aggregation
        loss_aggregation: How to combine losses ("sum", "mean", "weighted_sum")

        # Encoder initialization
        encoder_from_p7: Whether to initialize encoder from P7 pretrained
        p7_checkpoint_path: Path to P7 checkpoint (if encoder_from_p7 is True)

        # Data settings
        fps: Frames per second
        cardiac_band_hz: Cardiac frequency band
        output_sequence_length: Output sequence length for BVP

        device: Device string
        seed: Random seed
    """
    name: str = "multitask_physiological"

    # Encoder settings
    input_dim: int = 3
    hidden_dim: int = 128
    num_encoder_layers: int = 2
    encoder_type: str = "lstm"  # "lstm", "gru", "transformer", "attention"
    bidirectional: bool = True

    # Model settings
    dropout: float = 0.2
    sequence_length: int = 128

    # Task enable flags
    enable_bvp: bool = True
    enable_hr: bool = True
    enable_sqi: bool = True
    enable_confidence: bool = True

    # Loss weights
    bvp_weight: float = 1.0
    hr_weight: float = 1.0
    sqi_weight: float = 0.5
    confidence_weight: float = 0.5

    # Loss aggregation
    loss_aggregation: str = "weighted_sum"  # "sum", "mean", "weighted_sum"

    # P7 encoder initialization
    encoder_from_p7: bool = False
    p7_checkpoint_path: Optional[str] = None

    # Data settings
    fps: float = 30.0
    cardiac_band_hz: Tuple[float, float] = (0.833, 3.0)
    output_sequence_length: int = 128

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16

    device: str = "cpu"
    seed: int = 42

    def get_enabled_tasks(self) -> List[TaskType]:
        """Get list of enabled task types."""
        tasks = []
        if self.enable_bvp:
            tasks.append(TaskType.BVP)
        if self.enable_hr:
            tasks.append(TaskType.HR)
        if self.enable_sqi:
            tasks.append(TaskType.SQI)
        if self.enable_confidence:
            tasks.append(TaskType.CONFIDENCE)
        return tasks

    def get_task_weights(self) -> Dict[TaskType, float]:
        """Get weights for enabled tasks."""
        return {
            TaskType.BVP: self.bvp_weight,
            TaskType.HR: self.hr_weight,
            TaskType.SQI: self.sqi_weight,
            TaskType.CONFIDENCE: self.confidence_weight,
        }

    def get_encoder_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_encoder_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
            "fps": self.fps,
            "cardiac_band_hz": self.cardiac_band_hz,
        }


@dataclass
class MultiTaskOutput:
    """Output from multi-task physiological model.

    Contains predictions for all enabled tasks. Unavailable tasks
    have None values.
    """
    # Heart rate
    hr: Optional[torch.Tensor] = None          # (B,) HR estimates in BPM
    hr_logits: Optional[torch.Tensor] = None   # (B,) raw HR logits before activation

    # BVP waveform
    bvp: Optional[torch.Tensor] = None          # (B, T) BVP waveform predictions

    # Signal quality
    sqi: Optional[torch.Tensor] = None         # (B,) SQI estimates [0, 1]

    # Confidence
    confidence: Optional[torch.Tensor] = None  # (B,) confidence estimates [0, 1]

    # Shared representation
    representation: Optional[torch.Tensor] = None  # (B, hidden_dim) shared features

    # Attention weights if applicable
    attention_weights: Optional[torch.Tensor] = None  # (B, T)

    # Hidden state if applicable
    hidden_state: Optional[Tuple] = None

    def get_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.hr is not None:
            result["hr"] = self.hr
        if self.hr_logits is not None:
            result["hr_logits"] = self.hr_logits
        if self.bvp is not None:
            result["bvp"] = self.bvp
        if self.sqi is not None:
            result["sqi"] = self.sqi
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result

    def has_task(self, task: TaskType) -> bool:
        """Check if a task output is available."""
        if task == TaskType.BVP:
            return self.bvp is not None
        elif task == TaskType.HR:
            return self.hr is not None
        elif task == TaskType.SQI:
            return self.sqi is not None
        elif task == TaskType.CONFIDENCE:
            return self.confidence is not None
        return False


@dataclass
class MultiTaskBatch:
    """Batch structure for multi-task learning."""
    features: torch.Tensor                    # (B, T, F) input features
    mask: Optional[torch.Tensor] = None        # (B, T) True for padded positions

    # Task-specific targets
    hr: Optional[torch.Tensor] = None          # (B,) heart rate targets
    hr_available: Optional[torch.Tensor] = None  # (B,) True where HR target is valid

    bvp: Optional[torch.Tensor] = None         # (B, T) BVP waveform targets
    bvp_available: Optional[torch.Tensor] = None  # (B,) True where BVP target is valid

    sqi: Optional[torch.Tensor] = None         # (B,) SQI targets
    sqi_available: Optional[torch.Tensor] = None  # (B,) True where SQI target is valid

    confidence: Optional[torch.Tensor] = None  # (B,) confidence targets
    confidence_available: Optional[torch.Tensor] = None  # (B,) True where confidence is valid

    # Metadata
    subject_ids: Optional[torch.Tensor] = None  # (B,) subject IDs
    indices: Optional[torch.Tensor] = None      # (B,) sample indices

    def get_available_mask(self, task: TaskType) -> Optional[torch.Tensor]:
        """Get availability mask for a task."""
        if task == TaskType.BVP:
            return self.bvp_available
        elif task == TaskType.HR:
            return self.hr_available
        elif task == TaskType.SQI:
            return self.sqi_available
        elif task == TaskType.CONFIDENCE:
            return self.confidence_available
        return None

    def get_target(self, task: TaskType) -> Optional[torch.Tensor]:
        """Get target for a task."""
        if task == TaskType.BVP:
            return self.bvp
        elif task == TaskType.HR:
            return self.hr
        elif task == TaskType.SQI:
            return self.sqi
        elif task == TaskType.CONFIDENCE:
            return self.confidence
        return None

    def has_task(self, task: TaskType) -> bool:
        """Check if a task has targets in this batch."""
        target = self.get_target(task)
        available = self.get_available_mask(task)
        if target is None:
            return False
        if available is not None:
            return available.any()
        return True


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get torch device (CUDA if available)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
