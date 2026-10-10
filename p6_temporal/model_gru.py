"""
P6 Model: GRU-based Temporal Physiological Model

Gated Recurrent Unit network for learning physiological patterns.
A lighter-weight alternative to LSTM with fewer parameters.

Architecture:
    Input: (B, T, F) - Batch of temporal feature sequences
    GRU layers with configurable hidden dimensions
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
class GRUConfig(TemporalModelConfig):
    """Configuration for GRU temporal model."""
    name: str = "gru_temporal"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True


class GRUTemporal(BaseTemporalModel):
    """
    GRU-based temporal model for physiological signal estimation.

    Lighter-weight alternative to LSTM with comparable performance
    on shorter sequences.

    Input shape: (B, T, F)
    Output: TemporalOutput with BPM, confidence, signal quality
    """

    def __init__(self, config: GRUConfig):
        super().__init__(config)
        cfg = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout)
        )

        # GRU layers
        self.gru = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            bidirectional=cfg.bidirectional
        )

        # Output dimension
        gru_output_dim = cfg.hidden_dim * 2 if cfg.bidirectional else cfg.hidden_dim

        # Output heads
        self.head_bpm = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1)
        )

        self.head_confidence = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.head_quality = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # BVP prediction head
        if cfg.output_type in ["bvp", "both"]:
            self.head_bvp = nn.Sequential(
                nn.Linear(gru_output_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> TemporalOutput:
        """
        Forward pass through GRU temporal model.

        Args:
            x: Input tensor (B, T, F)
            mask: Optional mask tensor (B, T)

        Returns:
            TemporalOutput with physiological estimates
        """
        B, T, F = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # GRU forward
        gru_out, hidden = self.gru(x)  # (B, T, hidden_dim * dirs)

        # Apply mask if provided
        if mask is not None:
            mask_float = (~mask).unsqueeze(-1).float()
            gru_out = gru_out * mask_float

        # Global aggregation
        if mask is not None and mask.any():
            lengths = (~mask).sum(dim=1).long()
            max_len = gru_out.shape[1]
            lengths = lengths.clamp(min=1, max=max_len)
            batch_indices = torch.arange(B, device=x.device)
            last_indices = (lengths - 1).clamp(min=0)
            last_hidden = gru_out[batch_indices, last_indices]
        else:
            last_hidden = gru_out[:, -1, :]

        # Output predictions
        bpm = self.head_bpm(last_hidden).squeeze(-1)
        confidence = self.head_confidence(last_hidden).squeeze(-1)
        quality = self.head_quality(last_hidden).squeeze(-1)

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
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-timestep forward for autoregressive inference.

        Args:
            x: Single timestep input (B, F)
            hidden: Previous hidden state

        Returns:
            Tuple of (output, new_hidden_state)
        """
        B, F = x.shape

        # Project input
        x = self.input_proj(x)

        # Initialize hidden state if not provided
        if hidden is None:
            num_directions = 2 if self.config.bidirectional else 1
            h = torch.zeros(num_directions, B, self.config.hidden_dim, device=x.device)
            hidden = h

        # GRU forward
        out, hidden = self.gru(x.unsqueeze(1), hidden)
        out = out.squeeze(1)

        # Output from this timestep
        bpm = self.head_bpm(out).squeeze(-1)

        return bpm, hidden


class BiGRUTemporal(GRUTemporal):
    """Bidirectional GRU temporal model."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, sequence_length: int = 128):
        config = GRUConfig(
            name="bilstm_temporal",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            bidirectional=True
        )
        super().__init__(config)
