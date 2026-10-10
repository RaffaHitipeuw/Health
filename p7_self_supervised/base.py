"""
P7 Self-Supervised Learning Base Configuration

Base configuration and utilities for self-supervised physiological representation learning.
"""

import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class SelfSupervisedConfig:
    """Configuration for self-supervised learning."""
    name: str = "ssl_temporal"
    # Encoder settings (will be connected to P6 models)
    encoder_type: str = "lstm"  # "lstm", "gru", "transformer", "attention"
    encoder_hidden_dim: int = 128
    encoder_num_layers: int = 2
    encoder_bidirectional: bool = True

    # Projection head settings
    projection_dim: int = 128
    representation_dim: int = 64
    projection_layers: int = 2

    # Training settings
    temperature: float = 0.1  # Temperature for contrastive loss
    contrastive_weight: float = 1.0
    consistency_weight: float = 0.5
    predictive_weight: float = 0.3
    byol_weight: float = 0.0

    # Data settings
    input_dim: int = 3  # Feature dimension per timestep
    sequence_length: int = 128  # Default sequence length
    crop_length: int = 64  # Length for temporal crops
    fps: float = 30.0
    cardiac_band_hz: Tuple[float, float] = (0.833, 3.0)

    # Augmentation settings
    noise_std: float = 0.05
    dropout_prob: float = 0.1
    scale_range: Tuple[float, float] = (0.9, 1.1)
    time_shift_max: float = 0.1  # Max time shift as fraction

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32

    device: str = "cpu"
    seed: int = 42


@dataclass
class SSLBatch:
    """Batch structure for self-supervised learning."""
    # Original sequence (B, T, F)
    features: torch.Tensor
    # Augmented views (B, V, T', F) where V is number of views
    views: torch.Tensor
    # Original index for tracking (B,)
    indices: Optional[torch.Tensor] = None
    # Subject ID for avoiding same-subject negatives (B,)
    subject_ids: Optional[torch.Tensor] = None
    # Sequence mask (B, T)
    mask: Optional[torch.Tensor] = None


@dataclass
class SSLOutput:
    """Output from self-supervised model."""
    representations: torch.Tensor  # (B, D) learned representations
    projections: torch.Tensor  # (B, projection_dim) projection for contrastive
    loss: Optional[torch.Tensor] = None  # Total loss
    loss_dict: Optional[dict] = None  # Individual loss components


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


def get_encoder_from_p6(encoder_type: str, config: SelfSupervisedConfig):
    """Get encoder from P6 models.

    Args:
        encoder_type: Type of encoder ("lstm", "gru", "transformer", "attention")
        config: Self-supervised configuration

    Returns:
        Initialized encoder model
    """
    if encoder_type == "lstm":
        from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig
        cfg = LSTMConfig(
            input_dim=config.input_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            bidirectional=config.encoder_bidirectional,
            sequence_length=config.sequence_length
        )
        return LSTMTemporal(cfg)

    elif encoder_type == "gru":
        from p6_temporal.model_gru import GRUTemporal, GRUConfig
        cfg = GRUConfig(
            input_dim=config.input_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            bidirectional=config.encoder_bidirectional,
            sequence_length=config.sequence_length
        )
        return GRUTemporal(cfg)

    elif encoder_type == "transformer":
        from p6_temporal.model_transformer import TransformerTemporal, TransformerConfig
        cfg = TransformerConfig(
            input_dim=config.input_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            num_heads=4,
            sequence_length=config.sequence_length
        )
        return TransformerTemporal(cfg)

    elif encoder_type == "attention":
        from p6_temporal.model_attention_temporal import AttentionTemporal, TemporalAttentionConfig
        cfg = TemporalAttentionConfig(
            input_dim=config.input_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_heads=4,
            sequence_length=config.sequence_length
        )
        return AttentionTemporal(cfg)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
