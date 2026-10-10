"""
P6 Model: LSTM-based Temporal Physiological Model

Long Short-Term Memory network for learning physiological patterns
across sequential video frames and signal windows.

Architecture:
    Input: (B, T, F) - Batch of temporal feature sequences
    LSTM layers with configurable hidden dimensions
    Optional bidirectional processing
    Output heads for BPM, confidence, quality, and BVP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from .base import BaseTemporalModel, TemporalModelConfig, TemporalOutput


@dataclass
class LSTMConfig(TemporalModelConfig):
    """Configuration for LSTM temporal model."""
    name: str = "lstm_temporal"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True


class TemporalLSTMLayer(nn.Module):
    """Single LSTM layer with projection."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single timestep forward.

        Args:
            x: Input tensor (B, input_dim)
            state: Tuple of (h, c) hidden states

        Returns:
            Tuple of (output, new_state)
        """
        h, c = self.lstm(x, state)
        h = self.dropout(h)
        return h, (h, c)


class LSTMTemporal(BaseTemporalModel):
    """
    LSTM-based temporal model for physiological signal estimation.

    Processes temporal sequences using LSTM layers to capture
    cardiac rhythms and estimate heart rate over time.

    Input shape: (B, T, F) where T=sequence_length, F=input_dim
    Output: TemporalOutput with BPM, confidence, signal quality, optional BVP
    """

    def __init__(self, config: LSTMConfig):
        super().__init__(config)
        cfg = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            bidirectional=cfg.bidirectional
        )

        # Output dimension after LSTM
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

        # BVP prediction head (waveform output)
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
        Forward pass through LSTM temporal model.

        Args:
            x: Input tensor (B, T, F)
            mask: Optional mask tensor (B, T)

        Returns:
            TemporalOutput with physiological estimates
        """
        B, T, F = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # LSTM forward
        lstm_out, hidden = self.lstm(x)  # (B, T, hidden_dim * dirs)

        # Apply mask if provided (mask out padded positions)
        if mask is not None:
            mask_float = (~mask).unsqueeze(-1).float()  # (B, T, 1)
            lstm_out = lstm_out * mask_float

        # Global temporal aggregation
        if mask is not None and mask.any():
            # Use last valid (non-padded) timestep
            lengths = (~mask).sum(dim=1).long()  # (B,)
            max_len = lstm_out.shape[1]

            # Clamp lengths to valid range
            lengths = lengths.clamp(min=1, max=max_len)

            # Get last hidden for each sequence
            batch_indices = torch.arange(B, device=x.device)
            last_indices = (lengths - 1).clamp(min=0)  # (B,)
            last_hidden = lstm_out[batch_indices, last_indices]  # (B, hidden_dim * dirs)
        else:
            # No mask or all valid - use last timestep
            last_hidden = lstm_out[:, -1, :]  # (B, hidden_dim * dirs)

        # Output predictions
        bpm = self.head_bpm(last_hidden).squeeze(-1)  # (B,)
        confidence = self.head_confidence(last_hidden).squeeze(-1)  # (B,)
        quality = self.head_quality(last_hidden).squeeze(-1)  # (B,)

        # BVP prediction - predict from pooled representation
        bvp = None
        if hasattr(self, 'head_bvp'):
            # Use pooled representation for BVP prediction
            pooled = last_hidden.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)
            # Apply BVP head to each timestep
            B_temp, T_temp, _ = pooled.shape
            bvp_flat = self.head_bvp(pooled.reshape(B_temp * T_temp, -1))
            bvp = bvp_flat.reshape(B_temp, T_temp)

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
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Per-timestep forward for autoregressive inference.

        Args:
            x: Single timestep input (B, F)
            hidden: Previous hidden state (h, c)

        Returns:
            Tuple of (output, new_hidden_state)
        """
        B, F = x.shape

        # Project input
        x = self.input_proj(x)  # (B, hidden_dim)

        # Initialize hidden state if not provided
        if hidden is None:
            num_directions = 2 if self.config.bidirectional else 1
            h = torch.zeros(B, self.config.hidden_dim, device=x.device)
            c = torch.zeros(B, self.config.hidden_dim, device=x.device)
            hidden = (h, c)

        # LSTM cell forward
        h, c = self.lstm(x, hidden)
        h = self.dropout(h) if hasattr(self, 'dropout') else h

        # Output from this timestep
        bpm = self.head_bpm(h).squeeze(-1)

        return bpm, (h, c)


class BiLSTMTemporal(LSTMTemporal):
    """Bidirectional LSTM temporal model.

    Processes sequences in both forward and backward directions.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, sequence_length: int = 128):
        config = LSTMConfig(
            name="bilstm_temporal",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            bidirectional=True
        )
        super().__init__(config)
