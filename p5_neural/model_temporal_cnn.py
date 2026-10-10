"""
P5 Model 1: Temporal CNN

A CNN-based temporal architecture for learning physiological patterns
from temporal signal or feature sequences.

Architecture:
    Input: (B, T, F) temporal signal features
           where T=sequence_length, F=feature_dim
        or (B, T, 1) raw 1D signal

        1D temporal convolutions
        BatchNorm + ReLU after each conv
        Global temporal pooling
        FC layers for regression
        Output: BPM, confidence, quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from .base import BaseModel, ModelOutput


@dataclass
class TemporalCNNConfig:
    """Configuration for TemporalCNN - standalone config class."""
    name: str = "temporal_cnn"
    feature_dim: int = 3  # RGB or feature channels
    hidden_dims: Tuple[int, ...] = (64, 128, 256)
    kernel_sizes: Tuple[int, ...] = (7, 5, 3)
    dropout: float = 0.2
    input_channels: int = 3  # RGB
    sequence_length: int = 128  # frames
    fps: float = 30.0
    device: str = "cpu"
    seed: int = 42


class TemporalBlock(nn.Module):
    """Single temporal convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalCNN(BaseModel):
    """
    Temporal CNN for rPPG estimation.

    Processes temporal signal sequences using 1D convolutions to capture
    cardiac rhythms and estimate heart rate.

    Input shape: (B, T, F) where T=sequence_length, F=features
    Output: ModelOutput with BPM, confidence, signal quality
    """

    def __init__(self, config: TemporalCNNConfig):
        super().__init__(config)
        cfg = config

        # Build temporal CNN layers
        self.blocks = nn.ModuleList()
        in_ch = cfg.feature_dim

        for out_ch, ksz in zip(cfg.hidden_dims, cfg.kernel_sizes):
            self.blocks.append(
                TemporalBlock(in_ch, out_ch, ksz, cfg.dropout)
            )
            in_ch = out_ch

        # Global temporal pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output heads
        final_dim = cfg.hidden_dims[-1]
        self.head_bpm = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1)  # BPM regression
        )

        self.head_confidence = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] confidence
        )

        self.head_quality = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] quality
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, F) where F=feature_dim

        Returns:
            ModelOutput with BPM estimate and confidence
        """
        B, T, F = x.shape

        # (B, T, F) -> (B, F, T) for Conv1d
        x = x.transpose(1, 2)

        # Temporal convolutions
        for block in self.blocks:
            x = block(x)

        # Global temporal pooling: (B, C, T) -> (B, C, 1) -> (B, C)
        x = self.global_pool(x).squeeze(-1)

        # Output heads
        bpm = self.head_bpm(x).squeeze(-1)  # (B,)
        confidence = self.head_confidence(x).squeeze(-1)  # (B,)
        quality = self.head_quality(x).squeeze(-1)  # (B,)

        return ModelOutput(
            bpm=bpm,
            bpm_confidence=confidence,
            signal_quality=quality
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess to (B, T, F) format."""
        if x.dim() == 2:
            # (B*T, F) -> (B, T, F)
            x = x.unsqueeze(0) if x.dim() == 1 else x
        if x.dim() == 1:
            # Single signal: (T,) -> (1, T, 1)
            x = x.unsqueeze(0).unsqueeze(-1)
        return x.to(self.device)


class SignalTemporalCNN(TemporalCNN):
    """TemporalCNN variant for 1D signal input (e.g., raw RGB mean)."""

    def __init__(self, sequence_length: int = 128, hidden_dims: Tuple[int, ...] = (64, 128, 256)):
        config = TemporalCNNConfig(
            name="signal_temporal_cnn",
            feature_dim=1,  # Single signal channel
            sequence_length=sequence_length,
            hidden_dims=hidden_dims
        )
        super().__init__(config)


class FeatureTemporalCNN(TemporalCNN):
    """TemporalCNN variant for multi-feature input (e.g., RGB + motion + quality)."""

    def __init__(self, feature_dim: int, sequence_length: int = 128,
                 hidden_dims: Tuple[int, ...] = (64, 128, 256)):
        config = TemporalCNNConfig(
            name="feature_temporal_cnn",
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            hidden_dims=hidden_dims
        )
        super().__init__(config)
