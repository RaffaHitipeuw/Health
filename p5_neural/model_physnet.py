"""
P5 Model 3: PhysNet-Style Architecture

A spatiotemporal neural network inspired by PhysNet for video-based
physiological signal estimation, adapted to Sanubari input pipeline.

Architecture (PhysNet-style):
    3D Convolutions for spatiotemporal feature extraction
    Temporal modeling with 3D convolutions
    Frame-level predictions aggregated over time
    Output: BPM, confidence, quality

Reference inspiration: PhysNet design principles:
    - Joint spatiotemporal feature learning
    - Dilated temporal convolutions for long-range dependencies
    - Frame-level to video-level aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import BaseModel, ModelConfig, ModelOutput


class SpatioTemporalBlock3D(nn.Module):
    """3D convolution block for spatiotemporal features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1)
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalAggregationBlock1D(nn.Module):
    """1D temporal aggregation with different dilation rates."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2, 4)
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=d * (kernel_size - 1) // 2, dilation=d),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True)
            )
            for d in dilations
        ])
        self.fusion = nn.Conv1d(channels * len(dilations), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [branch(x) for branch in self.branches]
        x = torch.cat(branches, dim=1)
        return self.fusion(x)


class PhysNetStyleModel(BaseModel):
    """
    PhysNet-inspired spatiotemporal model for rPPG.

    Architecture:
        - 3D convolutions for joint spatial-temporal feature learning
        - Frame-level feature extraction with temporal modeling
        - Aggregation to video-level predictions
        - Multi-task heads for BPM, confidence, quality

    Input: (B, T, H, W, C) video tensor
    Output: ModelOutput with BPM estimate
    """

    def __init__(
        self,
        config: ModelConfig,
        temporal_channels: int = 32,
        spatial_channels: Tuple[int, int, int] = (32, 64, 64),
        dropout: float = 0.3
    ):
        super().__init__(config)

        # Spatiotemporal feature extraction
        self.spatiotemporal1 = SpatioTemporalBlock3D(
            config.input_channels, spatial_channels[0], (3, 5, 5)
        )
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.spatiotemporal2 = SpatioTemporalBlock3D(
            spatial_channels[0], spatial_channels[1], (3, 3, 3)
        )
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.spatiotemporal3 = SpatioTemporalBlock3D(
            spatial_channels[1], spatial_channels[2], (3, 3, 3)
        )

        # Temporal aggregation
        self.temporal_agg = TemporalAggregationBlock1D(
            spatial_channels[2], kernel_size=3, dilations=(1, 2, 4)
        )

        # Temporal pooling to fixed length
        self.temporal_pool = nn.AdaptiveAvgPool3d((8, 1, 1))

        # FC layers
        fc_input = spatial_channels[2] * 8

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Output heads
        self.head_bpm = nn.Linear(128, 1)
        self.head_confidence = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.head_quality = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass with 3D spatiotemporal processing.

        Args:
            x: Video tensor (B, T, H, W, C)

        Returns:
            ModelOutput with BPM estimate
        """
        # x: (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Spatiotemporal feature extraction
        x = self.spatiotemporal1(x)
        x = self.pool1(x)

        x = self.spatiotemporal2(x)
        x = self.pool2(x)

        x = self.spatiotemporal3(x)

        # Pool spatial to fixed size
        x = self.temporal_pool(x)  # (B, C, 8, 1, 1)

        # Reshape for temporal aggregation
        B, C, T_f, H1, W1 = x.shape
        x = x.view(B, C * T_f * H1 * W1)

        # FC
        x = self.fc(x)

        # Outputs
        bpm = self.head_bpm(x).squeeze(-1)
        confidence = self.head_confidence(x).squeeze(-1)
        quality = self.head_quality(x).squeeze(-1)

        return ModelOutput(
            bpm=bpm,
            bpm_confidence=confidence,
            signal_quality=quality
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is (B, T, H, W, C)."""
        if x.dim() == 4:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(-1)
        return x.to(self.device)


class PhysNetLite(nn.Module):
    """
    Lightweight PhysNet variant for faster inference.

    Simplified architecture with fewer parameters.
    """

    def __init__(self, input_channels: int = 3, sequence_length: int = 128):
        super().__init__()
        # Lightweight 3D convolutions
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 16, (3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        # Temporal pooling to fixed length
        self.temporal_pool = nn.AdaptiveAvgPool3d((8, 1, 1))

        # FC
        fc_dim = 64 * 8

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.head_bpm = nn.Linear(128, 1)
        self.head_conf = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.head_quality = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.temporal_pool(x)
        x = self.fc(x.squeeze(-1).squeeze(-1))

        return (
            self.head_bpm(x).squeeze(-1),
            self.head_conf(x).squeeze(-1),
            self.head_quality(x).squeeze(-1)
        )
