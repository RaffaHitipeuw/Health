"""
P6 Model: ConvLSTM Temporal Physiological Model

Convolutional LSTM combining spatial CNN with temporal LSTM for
spatiotemporal physiological pattern learning.

Architecture:
    Input: (B, T, H, W, C) or (B, T, F)
    Conv2D for spatial feature extraction per timestep
    LSTM for temporal modeling across spatial features
    Output heads for BPM, confidence, quality, and BVP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from .base import BaseTemporalModel, TemporalModelConfig, TemporalOutput


@dataclass
class ConvLSTMConfig(TemporalModelConfig):
    """Configuration for ConvLSTM temporal model."""
    name: str = "convlstm_temporal"
    input_height: int = 32
    input_width: int = 32
    spatial_channels: Tuple[int, ...] = (16, 32)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g gates
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single ConvLSTM step.

        Args:
            x: Input (B, input_dim, H, W)
            state: Tuple of (h, c) hidden states (B, hidden_dim, H, W)

        Returns:
            Tuple of (output, (new_h, new_c))
        """
        h, c = state

        # Concatenate along channel dimension
        combined = torch.cat([x, h], dim=1)

        # Compute gates
        gates = self.conv(combined)

        # Split into gates
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Update cell state
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class ConvLSTMTemporal(BaseTemporalModel):
    """
    ConvLSTM temporal model for spatiotemporal physiological estimation.

    Combines spatial CNN feature extraction with LSTM temporal modeling.
    Suitable for processing video frames directly.

    Input shape: (B, T, H, W, C) for video or (B, T, F) for features
    Output: TemporalOutput with BPM, confidence, signal quality
    """

    def __init__(self, config: ConvLSTMConfig):
        super().__init__(config)
        cfg = config

        self.is_video_input = True  # Can detect based on input

        # Spatial feature extractor (Conv2D per frame)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(cfg.input_dim, cfg.spatial_channels[0], 3, padding=1),
            nn.BatchNorm2d(cfg.spatial_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(cfg.spatial_channels[0], cfg.spatial_channels[1], 3, padding=1),
            nn.BatchNorm2d(cfg.spatial_channels[1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Calculate spatial feature dimension
        spatial_feat_dim = cfg.spatial_channels[-1] * 16  # After 4x4 pooling

        # Project spatial features to temporal dimension
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_feat_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU()
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            bidirectional=cfg.bidirectional
        )

        # Output dimension
        lstm_output_dim = cfg.hidden_dim * 2 if cfg.bidirectional else cfg.hidden_dim

        # Output heads
        self.head_bpm = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1)
        )

        self.head_confidence = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.head_quality = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # BVP prediction head
        if cfg.output_type in ["bvp", "both"]:
            self.head_bvp = nn.Sequential(
                nn.Linear(lstm_output_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> TemporalOutput:
        """
        Forward pass through ConvLSTM temporal model.

        Args:
            x: Input tensor (B, T, H, W, C) for video or (B, T, F) for features
            mask: Optional mask tensor (B, T)

        Returns:
            TemporalOutput with physiological estimates
        """
        # Detect input type
        if x.dim() == 5:
            # Video input: (B, T, H, W, C)
            return self._forward_video(x, mask)
        elif x.dim() == 3:
            # Feature input: (B, T, F)
            return self._forward_features(x, mask)
        else:
            raise ValueError(f"Expected 3D (B,T,F) or 5D (B,T,H,W,C) input, got {x.dim()}D")

    def _forward_video(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> TemporalOutput:
        """Process video input with spatial and temporal modeling."""
        B, T, H, W, C = x.shape

        # Reshape for spatial processing: (B*T, H, W, C) -> (B*T, C, H, W)
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Spatial feature extraction - process all frames at once through CNN
        x = self.spatial_encoder(x)  # (B*T, spatial_ch, H', W')

        # Flatten spatial dimensions for projection
        # Get shape after CNN
        B_total, C_out, H_out, W_out = x.shape
        x = x.reshape(B_total, C_out * H_out * W_out)  # (B*T, flattened_features)

        # Reshape to (B, T, features)
        x = x.reshape(B, T, -1)  # (B, T, flattened_features)

        # Project spatial features to hidden dimension
        x = self.spatial_proj(x)  # (B, T, hidden_dim)

        # Temporal LSTM
        return self._temporal_forward(x, mask)

    def _forward_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> TemporalOutput:
        """Process feature input directly."""
        B, T, F = x.shape

        # Project features to hidden dimension
        if F != self.config.hidden_dim:
            x = nn.functional.relu(nn.Linear(F, self.config.hidden_dim)(x))

        return self._temporal_forward(x, mask)

    def _temporal_forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> TemporalOutput:
        """Common temporal processing."""
        # LSTM forward
        lstm_out, hidden = self.lstm(x)

        # Apply mask if provided
        if mask is not None:
            lstm_out = lstm_out * (~mask).unsqueeze(-1).float().clamp(min=0.1)

        # Global aggregation
        if mask is not None:
            lengths = (~mask).sum(dim=1).long()
            last_indices = (lengths - 1).clamp(min=0)
            last_hidden = lstm_out.gather(1, last_indices.unsqueeze(-1).expand(-1, -1, lstm_out.shape[-1]))
            last_hidden = last_hidden.squeeze(1)
        else:
            last_hidden = lstm_out[:, -1, :]

        # Output predictions
        bpm = self.head_bpm(last_hidden).squeeze(-1)
        confidence = self.head_confidence(last_hidden).squeeze(-1)
        quality = self.head_quality(last_hidden).squeeze(-1)

        # BVP prediction
        bvp = None
        if hasattr(self, 'head_bvp'):
            bvp = self.head_bvp(lstm_out)

        return TemporalOutput(
            bpm=bpm,
            bpm_confidence=confidence,
            signal_quality=quality,
            bvp=bvp,
            hidden_state=hidden
        )

    def forward_sequence(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """Per-timestep forward."""
        if x.dim() == 4:
            # Video frame
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.spatial_encoder(x)
            x = x.reshape(B, -1)
            x = self.spatial_proj(x)
        elif x.dim() == 2:
            # Features
            x = self.spatial_proj(x) if x.shape[-1] != self.config.hidden_dim else x

        # This is a simplified per-step forward
        # Full implementation would maintain hidden states
        raise NotImplementedError("Per-step forward not fully implemented for ConvLSTM")
