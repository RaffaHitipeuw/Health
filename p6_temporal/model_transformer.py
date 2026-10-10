"""
P6 Model: Transformer-based Temporal Physiological Model

Transformer architecture with self-attention for learning long-range
temporal dependencies in physiological signals.

Architecture:
    Input: (B, T, F) - Batch of temporal feature sequences
    Positional encoding for temporal information
    Multi-head self-attention layers
    Output heads for BPM, confidence, quality, and BVP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from .base import BaseTemporalModel, TemporalModelConfig, TemporalOutput


@dataclass
class TransformerConfig(TemporalModelConfig):
    """Configuration for Transformer temporal model."""
    name: str = "transformer_temporal"
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    bidirectional: bool = False  # Not used for Transformer, but kept for interface


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal information."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor (B, T, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTemporal(BaseTemporalModel):
    """
    Transformer-based temporal model for physiological signal estimation.

    Uses self-attention to capture long-range temporal dependencies
    and global context in physiological signals.

    Input shape: (B, T, F)
    Output: TemporalOutput with BPM, confidence, signal quality
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        cfg = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU()
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            cfg.hidden_dim,
            max_len=cfg.sequence_length + 100,
            dropout=cfg.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers
        )

        # Temporal attention pooling for weighted aggregation
        self.temporal_attention = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim // 4, 1)
        )

        # Output heads
        self.head_bpm = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1)
        )

        self.head_confidence = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.head_quality = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # BVP prediction head
        if cfg.output_type in ["bvp", "both"]:
            self.head_bvp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> TemporalOutput:
        """
        Forward pass through Transformer temporal model.

        Args:
            x: Input tensor (B, T, F)
            mask: Optional mask tensor (B, T) - True for padded positions

        Returns:
            TemporalOutput with physiological estimates
        """
        B, T, F = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask if mask is provided
        # Transformer expects True for masked (padded) positions
        src_key_padding_mask = mask if mask is not None else None

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, hidden_dim)

        # Temporal attention pooling
        attn_weights = self.temporal_attention(x)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Store attention weights
        attn_out = (x * attn_weights).sum(dim=1)  # (B, hidden_dim)

        # Output predictions
        bpm = self.head_bpm(attn_out).squeeze(-1)
        confidence = self.head_confidence(attn_out).squeeze(-1)
        quality = self.head_quality(attn_out).squeeze(-1)

        # BVP prediction - predict from pooled representation
        bvp = None
        if hasattr(self, 'head_bvp'):
            # Use pooled representation for BVP prediction
            pooled = pooled.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)
            # Apply BVP head to each timestep
            B_temp, T_temp, _ = pooled.shape
            bvp_flat = self.head_bvp(pooled.reshape(B_temp * T_temp, -1))
            bvp = bvp_flat.reshape(B_temp, T_temp)

        return TemporalOutput(
            bpm=bpm,
            bpm_confidence=confidence,
            signal_quality=quality,
            bvp=bvp,
            attention_weights=attn_weights.squeeze(-1)
        )

    def forward_sequence(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Autoregressive forward for online inference.

        Args:
            x: Single timestep input (B, F)
            cache: Cached key/value states for efficiency

        Returns:
            Tuple of (output, updated_cache)
        """
        B, F = x.shape
        device = x.device

        # Initialize cache if not provided
        if cache is None:
            cache = {
                'prev_x': None,
                'hidden': torch.zeros(1, B, self.config.hidden_dim, device=device),
                'step': 0
            }

        # Project input
        x = self.input_proj(x.unsqueeze(1))  # (B, 1, hidden_dim)

        # Add positional encoding for this step
        pos = cache['step'] % (self.config.sequence_length + 100)
        x = x + self.pos_encoder.pe[:, pos:pos+1, :]

        # For simplicity in this implementation, we do a full forward pass
        # In a production system, this would use cached attention
        x = self.transformer(x)  # (B, 1, hidden_dim)

        cache['step'] += 1

        # Output
        bpm = self.head_bpm(x.squeeze(1)).squeeze(-1)

        return bpm, cache
