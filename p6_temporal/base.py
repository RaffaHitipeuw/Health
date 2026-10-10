"""
P6 Temporal Model Base Classes

Base interfaces and configurations for temporal physiological models.
Extends P5 base models with sequence-aware interfaces.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class TemporalModelConfig:
    """Configuration for temporal physiological models."""
    name: str = "temporal_model"
    input_dim: int = 3  # Feature dimension per timestep
    hidden_dim: int = 128  # Hidden state dimension
    num_layers: int = 2  # Number of temporal layers
    dropout: float = 0.2  # Dropout probability
    sequence_length: int = 128  # Default sequence length
    bidirectional: bool = False  # Bidirectional processing
    output_type: str = "bpm"  # Output type: "bpm", "bvp", "both"
    fps: float = 30.0  # Frames per second
    cardiac_band_hz: Tuple[float, float] = (0.833, 3.0)  # Cardiac frequency band
    device: str = "cpu"
    seed: int = 42


@dataclass
class TemporalOutput:
    """Output from temporal physiological models."""
    bpm: torch.Tensor  # Heart rate estimate (B,)
    bpm_confidence: torch.Tensor  # Model confidence (B,)
    signal_quality: torch.Tensor  # Signal quality (B,)
    bvp: Optional[torch.Tensor] = None  # BVP waveform (B, T) if available
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # RNN hidden state
    attention_weights: Optional[torch.Tensor] = None  # Temporal attention (B, T)


class BaseTemporalModel(nn.Module):
    """Base class for temporal physiological models.

    All P6 temporal models inherit from this class.
    """

    def __init__(self, config: TemporalModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> TemporalOutput:
        """Forward pass - must be implemented by subclasses.

        Args:
            x: Input tensor of shape (B, T, F) or (B, T, H, W, C)
            mask: Optional mask tensor (B, T) for variable-length sequences

        Returns:
            TemporalOutput with estimated physiological outputs
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def forward_sequence(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Per-step forward for autoregressive inference.

        Args:
            x: Single timestep input (B, F) or (B, H, W, C)
            hidden: Previous hidden state

        Returns:
            Tuple of (output, hidden_state)
        """
        raise NotImplementedError("Subclasses must implement forward_sequence()")

    def get_output_dim(self) -> int:
        """Get the output dimension for the given configuration."""
        if self.config.bidirectional:
            return self.config.hidden_dim * 2
        return self.config.hidden_dim


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


def pad_sequence(sequences: List[torch.Tensor], batch_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences.

    Args:
        sequences: List of tensors with shape (T, F) or (T,)
        batch_first: If True, output is (B, T, *); else (T, B, *)

    Returns:
        Tuple of (padded_tensor, lengths)
    """
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    max_len = lengths.max().item()

    # Determine output shape
    if sequences[0].dim() == 1:
        # 1D sequences (e.g., raw signals)
        padded = torch.zeros(len(sequences), max_len, dtype=sequences[0].dtype)
    else:
        # Multi-dimensional sequences
        padded = torch.zeros(len(sequences), max_len, *sequences[0].shape[1:], dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded[i, :length] = seq

    if not batch_first:
        padded = padded.transpose(0, 1)

    return padded, lengths


def create_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create attention mask for packed sequences.

    Args:
        lengths: Sequence lengths (B,)
        max_len: Maximum length (defaults to lengths.max())

    Returns:
        Boolean mask tensor (B, max_len) where True indicates padded position
    """
    if max_len is None:
        max_len = lengths.max().item()
    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
    return mask
