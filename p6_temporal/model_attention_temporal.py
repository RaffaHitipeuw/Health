"""
P6 Model: Attention-based Temporal Physiological Model

Temporal attention mechanisms for learning physiological dynamics
across sequential video frames and signal windows.

Architecture:
    Input: (B, T, F) - Batch of temporal feature sequences
    Multi-head temporal attention
    Temporal convolution for local patterns
    Output heads for BPM, confidence, quality, and BVP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from .base import BaseTemporalModel, TemporalModelConfig, TemporalOutput


@dataclass
class TemporalAttentionConfig(TemporalModelConfig):
    """Configuration for attention-based temporal model."""
    name: str = "attention_temporal"
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention for temporal sequences."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention forward.

        Args:
            x: Input tensor (B, T, hidden_dim)
            mask: Optional mask (B, T) - True for padded positions

        Returns:
            Tuple of (output, attention_weights)
        """
        B, T, _ = x.shape

        # Linear projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Apply mask if provided
        if mask is not None:
            # Expand mask: (B, T) -> (B, 1, 1, T) -> (B, H, T, T)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask_expanded, float('-inf'))

        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # (B, H, T, d)

        # Reshape output
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        out = self.out(out)

        return out, attn_weights.mean(dim=1)  # Average attention across heads


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with dilated convolutions."""

    def __init__(self, channels: int, kernel_sizes: Tuple[int, ...] = (3, 5, 7), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, k, padding=k // 2),
                nn.BatchNorm1d(channels),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        self.fusion = nn.Conv1d(channels * len(kernel_sizes), channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        branches = [branch(x) for branch in self.branches]
        x = torch.cat(branches, dim=1)
        x = self.fusion(x)
        return self.dropout(x)


class AttentionTemporal(BaseTemporalModel):
    """
    Attention-based temporal model for physiological signal estimation.

    Combines multi-head self-attention with temporal convolutions
    to capture both long-range dependencies and local patterns.

    Input shape: (B, T, F)
    Output: TemporalOutput with BPM, confidence, signal quality
    """

    def __init__(self, config: TemporalAttentionConfig):
        super().__init__(config)
        cfg = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout)
        )

        # Positional encoding for attention
        self.pos_encoding = nn.Parameter(torch.randn(1, cfg.sequence_length, cfg.hidden_dim) * 0.02)

        # Multi-head attention layers
        self.attention1 = MultiHeadTemporalAttention(
            cfg.hidden_dim, cfg.num_heads, cfg.dropout
        )
        self.attention_norm1 = nn.LayerNorm(cfg.hidden_dim)

        self.attention2 = MultiHeadTemporalAttention(
            cfg.hidden_dim, cfg.num_heads, cfg.dropout
        )
        self.attention_norm2 = nn.LayerNorm(cfg.hidden_dim)

        # Temporal convolution for local patterns
        self.temporal_conv = TemporalConvBlock(cfg.hidden_dim, (3, 5, 7), cfg.dropout)
        self.conv_norm = nn.LayerNorm(cfg.hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
            nn.Dropout(cfg.dropout)
        )
        self.ffn_norm = nn.LayerNorm(cfg.hidden_dim)

        # Temporal pooling for aggregation
        self.temporal_pool = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim // 2, 1)
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
        Forward pass through attention-based temporal model.

        Args:
            x: Input tensor (B, T, F)
            mask: Optional mask tensor (B, T)

        Returns:
            TemporalOutput with physiological estimates
        """
        B, T, F = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Add positional encoding
        if T <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :T, :]
        else:
            # Extend positional encoding if needed
            extra = T - self.pos_encoding.shape[1]
            pos_extended = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_extended

        # Attention block 1 with residual
        attn_out, attn_weights1 = self.attention1(x, mask)
        x = self.attention_norm1(x + attn_out)

        # Temporal convolution with residual
        x_conv = x.transpose(1, 2)  # (B, hidden_dim, T)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, T, hidden_dim)
        x = self.conv_norm(x + x_conv)

        # Attention block 2 with residual
        attn_out, attn_weights2 = self.attention2(x, mask)
        x = self.attention_norm2(x + attn_out)

        # Feed-forward network with residual
        x = self.ffn_norm(x + self.ffn(x))

        # Temporal attention pooling for aggregation
        pool_weights = self.temporal_pool(x)  # (B, T, 1)
        pool_weights = torch.softmax(pool_weights, dim=1)
        pooled = (x * pool_weights).sum(dim=1)  # (B, hidden_dim)

        # Output predictions
        bpm = self.head_bpm(pooled).squeeze(-1)
        confidence = self.head_confidence(pooled).squeeze(-1)
        quality = self.head_quality(pooled).squeeze(-1)

        # BVP prediction - predict from pooled representation
        bvp = None
        if hasattr(self, 'head_bvp'):
            # Use pooled representation for BVP prediction
            pooled_bvp = pooled.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)
            # Apply BVP head to each timestep
            B_temp, T_temp, _ = pooled_bvp.shape
            bvp_flat = self.head_bvp(pooled_bvp.reshape(B_temp * T_temp, -1))
            bvp = bvp_flat.reshape(B_temp, T_temp)

        # Average attention weights from both layers
        avg_attn = (attn_weights1 + attn_weights2) / 2

        return TemporalOutput(
            bpm=bpm,
            bpm_confidence=confidence,
            signal_quality=quality,
            bvp=bvp,
            attention_weights=avg_attn
        )

    def forward_sequence(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Incremental forward for online inference.

        Args:
            x: Single timestep input (B, F)
            cache: Cache of previous states

        Returns:
            Tuple of (output, updated_cache)
        """
        B, F = x.shape
        device = x.device

        if cache is None:
            cache = {
                'key': [],
                'value': [],
                't': 0
            }

        # Project input
        x = self.input_proj(x.unsqueeze(1))  # (B, 1, hidden_dim)

        # Add positional encoding for current step
        if cache['t'] < self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, cache['t']:cache['t']+1, :]

        # Compute attention with cached keys/values
        q = self.attention1.query(x)
        k = self.attention1.key(x)
        v = self.attention1.value(x)

        cache['key'].append(k)
        cache['value'].append(v)

        if len(cache['key']) > 1:
            k_full = torch.cat(cache['key'], dim=1)
            v_full = torch.cat(cache['value'], dim=1)

            # Compute attention
            attn = (q @ k_full.transpose(-2, -1)) * self.attention1.scale
            attn = torch.softmax(attn, dim=-1)
            out = attn @ v_full
        else:
            out = v

        out = self.attention1.out(out)
        cache['t'] += 1

        # Output BPM (simplified - full forward would continue through all layers)
        bpm = self.head_bpm(out.squeeze(1)).squeeze(-1)

        return bpm, cache
