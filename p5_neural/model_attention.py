"""
P5 Model 4: TS-CAN Style / Attention-Based rPPG Model

An attention-based model for video-based physiological estimation.

Architecture:
    Spatial attention for focusing on informative regions
    Temporal attention for selecting reliable frames
    Self-attention for global context
    Multi-head design for different aspects of the signal
    Output: BPM, confidence, quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import BaseModel, ModelConfig, ModelOutput


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on informative regions."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)  # (B, C)
        y = self.fc(y).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


class TemporalAttention(nn.Module):
    """Temporal attention for selecting reliable frames."""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        attn = self.attention(x)  # (B, T, 1)
        attn = F.softmax(attn, dim=1)  # Normalize over time
        return x * attn  # Weighted sum


class SelfAttention(nn.Module):
    """Self-attention for global context."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return self.norm(x + x)  # Residual


class TSCANStyleModel(BaseModel):
    """
    TS-CAN-inspired model with spatial and temporal attention.

    Architecture:
        Frame-level spatial attention
        Temporal attention for frame weighting
        Self-attention for global context
        Multi-task heads

    Input: (B, T, H, W, C) video
    Output: ModelOutput with BPM, confidence, quality
    """

    def __init__(
        self,
        config: ModelConfig,
        base_channels: int = 32,
        attention_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__(config)

        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(config.input_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            SpatialAttention(base_channels),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            SpatialAttention(base_channels * 2),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        enc_channels = base_channels * 2 * 64  # After spatial pooling

        # Temporal encoder with attention
        self.temporal_encoder = nn.Sequential(
            nn.Linear(enc_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            TemporalAttention(256)
        )

        # Self-attention for context
        self.self_attention = nn.Sequential(
            SelfAttention(256, attention_heads),
            nn.Dropout(dropout)
        )

        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # FC
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Heads
        self.head_bpm = nn.Linear(128, 1)
        self.head_confidence = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.head_quality = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass with attention mechanisms.

        Args:
            x: Video tensor (B, T, H, W, C)

        Returns:
            ModelOutput with attention-weighted predictions
        """
        B, T, H, W, C = x.shape

        # Encode each frame
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T, C, H, W)
        x = self.frame_encoder(x)  # (B*T, channels, H', W')
        x = x.reshape(B, T, -1)  # (B, T, features)

        # Temporal processing with attention
        x = self.temporal_encoder(x)  # (B, T, 256)
        x = self.self_attention(x)  # (B, T, 256)

        # Temporal pooling
        x = x.transpose(1, 2)  # (B, 256, T)
        x = self.temporal_pool(x).squeeze(-1)  # (B, 256)

        # FC
        x = self.fc(x)  # (B, 128)

        return ModelOutput(
            bpm=self.head_bpm(x).squeeze(-1),
            bpm_confidence=self.head_confidence(x).squeeze(-1),
            signal_quality=self.head_quality(x).squeeze(-1)
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is (B, T, H, W, C)."""
        if x.dim() == 4:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(-1)
        return x.to(self.device)


class SimpleAttentionModel(nn.Module):
    """
    Simplified attention model for faster inference.

    Lightweight version with essential attention mechanisms.
    """

    def __init__(self, input_channels: int = 3, hidden_dim: int = 64):
        super().__init__()

        # Frame encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.feat_dim = hidden_dim * 16  # After spatial pooling

        # Project features to attention dimension
        self.feat_proj = nn.Linear(self.feat_dim, hidden_dim)

        # Simple temporal attention - compute attention weights over time
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # FC
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.head_bpm = nn.Linear(64, 1)
        self.head_conf = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.head_quality = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape

        # Encode frames
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        x = self.encoder(x)  # (B*T, hidden_dim, 4, 4)
        x = x.reshape(B, T, -1)  # (B, T, hidden_dim * 16)

        # Project features
        x = self.feat_proj(x)  # (B, T, hidden_dim)

        # Temporal attention weights
        attn_weights = self.temporal_attn(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize over time

        # Apply attention and aggregate
        x = (x * attn_weights).sum(dim=1)  # (B, hidden_dim)

        # FC
        x = self.fc(x)  # (B, 64)

        return (
            self.head_bpm(x).squeeze(-1),
            self.head_conf(x).squeeze(-1),
            self.head_quality(x).squeeze(-1)
        )
