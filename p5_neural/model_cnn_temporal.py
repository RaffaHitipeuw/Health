"""
P5 Model 2: CNN + Temporal Convolution

A model combining convolutional feature extraction with temporal
convolution layers to capture local and longer-range temporal patterns.

Architecture:
    2D CNN for spatial feature extraction from frames
    1D Temporal convolutions for time-series patterns
    Multi-scale temporal aggregation
    Output: BPM, confidence, quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import BaseModel, ModelConfig, ModelOutput


class CNN2DBlock(nn.Module):
    """2D convolutional block for spatial feature extraction."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if stride >= 2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class TemporalConv1DBlock(nn.Module):
    """1D temporal convolution block with dilated convolutions."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MultiScaleTemporalAggregator(nn.Module):
    """Aggregates features at multiple temporal scales."""

    def __init__(self, channels: int, scales: Tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, k, padding=k//2),
                nn.BatchNorm1d(channels),
                nn.ReLU()
            )
            for k in scales
        ])
        self.fusion = nn.Conv1d(channels * len(scales), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        branches = [branch(x) for branch in self.branches]
        x = torch.cat(branches, dim=1)  # (B, C*len(scales), T)
        x = self.fusion(x)  # (B, C, T)
        return x


class CNNTemporalModel(BaseModel):
    """
    CNN + Temporal Convolution model for rPPG.

    Architecture:
        2D CNN: Extracts spatial features per frame
        Temporal Conv: Captures time-series patterns
        Multi-scale aggregation: Different temporal receptive fields
        FC heads: Regression to BPM/confidence/quality

    Input: (B, T, H, W, C) video frames
    Output: ModelOutput with BPM, confidence, quality
    """

    def __init__(
        self,
        config: ModelConfig,
        spatial_channels: Tuple[int, ...] = (16, 32, 64),
        temporal_channels: int = 128,
        temporal_scales: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.3
    ):
        super().__init__(config)

        # Spatial 2D CNN for frame features
        self.spatial_cnn = nn.ModuleList()
        in_ch = config.input_channels
        for out_ch in spatial_channels:
            self.spatial_cnn.append(CNN2DBlock(in_ch, out_ch))
            in_ch = out_ch

        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.spatial_flat = nn.Flatten(2)  # (B, T, C, H', W') -> (B, T, C*H'*W')

        # Project to temporal channels
        spatial_out = spatial_channels[-1] * 16  # 4*4 spatial
        self.spatial_proj = nn.Linear(spatial_out, temporal_channels)

        # Temporal 1D convolutions with dilation
        self.temporal_conv1 = TemporalConv1DBlock(temporal_channels, temporal_channels, 3, dilation=1)
        self.temporal_conv2 = TemporalConv1DBlock(temporal_channels, temporal_channels, 3, dilation=2)
        self.temporal_conv3 = TemporalConv1DBlock(temporal_channels, temporal_channels, 3, dilation=4)

        # Multi-scale aggregation
        self.multi_scale = MultiScaleTemporalAggregator(temporal_channels, temporal_scales)

        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Output heads
        hidden_dim = temporal_channels
        self.head_bpm = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.head_confidence = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.head_quality = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass.

        Args:
            x: Video tensor (B, T, H, W, C) frames

        Returns:
            ModelOutput with BPM estimate
        """
        B, T, H, W, C = x.shape

        # Process each frame through spatial CNN
        # x: (B, T, H, W, C) -> transpose -> (B*T, C, H, W)
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Spatial CNN
        for block in self.spatial_cnn:
            x = block(x)

        # Pool and flatten spatial
        x = self.spatial_pool(x)  # (B*T, C, 4, 4)
        x = x.view(B * T, -1)  # (B*T, C*16)

        # Reshape for temporal processing
        # (B*T, spatial_dim) -> (B, T, spatial_dim) -> (B, spatial_dim, T)
        x = x.view(B, T, -1).permute(0, 2, 1).contiguous()

        # Project to temporal channels
        x = self.spatial_proj(x.transpose(1, 2)).transpose(1, 2)  # (B, temporal_ch, T)

        # Temporal convolutions with residual
        x = x + self.temporal_conv1(x)
        x = x + self.temporal_conv2(x)
        x = x + self.temporal_conv3(x)

        # Multi-scale aggregation
        x = self.multi_scale(x)

        # Temporal pooling
        x = self.temporal_pool(x).squeeze(-1)  # (B, temporal_ch)

        # Output heads
        bpm = self.head_bpm(x).squeeze(-1)
        confidence = self.head_confidence(x).squeeze(-1)
        quality = self.head_quality(x).squeeze(-1)

        return ModelOutput(
            bpm=bpm,
            bpm_confidence=confidence,
            signal_quality=quality
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess to (B, T, H, W, C) format."""
        if x.dim() == 4:
            # (B, H, W, C) -> (1, T, H, W, C) if single batch
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            # (T, H, W, C) -> (1, T, H, W, C)
            x = x.unsqueeze(0)
        return x.to(self.device)


class LightweightCNNTemporal(nn.Module):
    """Lightweight CNN-Temporal for resource-constrained inference."""

    def __init__(self, input_channels: int = 3, sequence_length: int = 128):
        super().__init__()

        # Lightweight spatial features
        self.spatial = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.spatial_dim = 32 * 4  # 128

        # Lightweight temporal
        self.temporal = nn.Sequential(
            nn.Conv1d(self.spatial_dim, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        # Heads
        self.head_bpm = nn.Linear(64, 1)
        self.head_conf = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.head_quality = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, H, W, C) or (T, H, W, C)
        if x.dim() == 5:
            B, T, H, W, C = x.shape
            x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected 5D input, got {x.dim()}D")

        x = self.spatial(x)  # (B*T, 32, 2, 2)
        x = x.view(B, T, -1).permute(0, 2, 1)  # (B, 128, T)
        x = self.temporal(x).squeeze(-1)  # (B, 64)

        return (
            self.head_bpm(x).squeeze(-1),
            self.head_conf(x).squeeze(-1),
            self.head_quality(x).squeeze(-1)
        )
