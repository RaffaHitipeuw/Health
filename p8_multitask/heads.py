"""
P8 Task-Specific Heads

Modular task-specific output heads for multi-task physiological modeling.
Each head can be independently enabled/disabled and trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class BaseTaskHead(nn.Module):
    """Base class for task-specific heads."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout

    def _get_dropout(self) -> nn.Dropout:
        return nn.Dropout(self.dropout_prob)


class HRHead(BaseTaskHead):
    """Heart Rate (HR) estimation head.

    Estimates heart rate in BPM from shared representation.
    Uses configurable output range.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        hr_min: float = 40.0,
        hr_max: float = 200.0
    ):
        super().__init__(input_dim, output_dim=1, dropout=dropout)
        self.hr_min = hr_min
        self.hr_max = hr_max

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        representation: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            representation: (B, input_dim) shared representation
            return_logits: If True, return raw logits without scaling

        Returns:
            (B,) HR estimates in BPM
        """
        logits = self.head(representation).squeeze(-1)  # (B,)

        if return_logits:
            return logits

        # Scale to HR range [hr_min, hr_max]
        hr = torch.sigmoid(logits) * (self.hr_max - self.hr_min) + self.hr_min
        return hr


class BVPHead(BaseTaskHead):
    """BVP waveform prediction head.

    Predicts BVP waveform from shared representation.
    Supports both sequence-level and per-timestep prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_length: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_temporal_conv: bool = True
    ):
        super().__init__(input_dim, output_dim=output_length, dropout=dropout)
        self.output_length = output_length
        self.use_temporal_conv = use_temporal_conv

        if use_temporal_conv:
            # Temporal convolutional approach
            self.temporal_conv = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.to_waveform = nn.Linear(hidden_dim, output_length)
        else:
            # Simple linear projection
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_length)
            )

    def forward(
        self,
        representation: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            representation: (B, input_dim) shared representation
            temporal_features: (B, T, input_dim) optional temporal features

        Returns:
            (B, output_length) BVP waveform predictions
        """
        if self.use_temporal_conv and temporal_features is not None:
            # Use temporal features with shared representation
            # Broadcast representation to match temporal length
            B, T, _ = temporal_features.shape
            rep_expanded = representation.unsqueeze(1).expand(-1, T, -1)  # (B, T, input_dim)

            # Combine
            combined = torch.cat([temporal_features, rep_expanded], dim=-1)  # (B, T, input_dim*2)

            # Process
            x = self.temporal_conv(combined)  # (B, T, hidden_dim)
            bvp = self.to_waveform(x)  # (B, T, output_length)
            return bvp.squeeze(-1) if bvp.shape[-1] == 1 else bvp
        else:
            # Use representation only - need self.head
            if hasattr(self, 'head'):
                x = self.head(representation)  # (B, output_length)
            else:
                # Fallback: project representation to output
                if not hasattr(self, '_bvp_proj'):
                    # Create projection layer on demand
                    self._bvp_proj = nn.Linear(representation.shape[-1], self.output_length)
                x = self._bvp_proj(representation)
            return x


class SQIHead(BaseTaskHead):
    """Signal Quality Index (SQI) prediction head.

    Predicts signal quality score from shared representation.
    Output is in [0, 1] range.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, output_dim=1, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            representation: (B, input_dim) shared representation

        Returns:
            (B,) SQI estimates in [0, 1]
        """
        return self.head(representation).squeeze(-1)


class ConfidenceHead(BaseTaskHead):
    """Prediction confidence / reliability head.

    Estimates how confident the model is about its predictions.
    Output is in [0, 1] range.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        num_predictions: int = 1
    ):
        super().__init__(input_dim, output_dim=num_predictions, dropout=dropout)
        self.num_predictions = num_predictions

        self.head = nn.Sequential(
            nn.Linear(input_dim * num_predictions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_predictions),
            nn.Sigmoid()
        )

    def forward(
        self,
        representation: torch.Tensor,
        predictions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            representation: (B, input_dim) shared representation
            predictions: Optional additional predictions to condition on

        Returns:
            (B,) or (B, num_predictions) confidence estimates in [0, 1]
        """
        if predictions is not None and self.num_predictions > 1:
            # Concatenate representation with predictions
            x = torch.cat([representation, predictions], dim=-1)
            return self.head(x).squeeze(-1)
        else:
            return self.head(representation).squeeze(-1)


class TemporalPoolingHead(nn.Module):
    """Head that pools temporal features before prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "attention"  # "attention", "mean", "max", "last"
    ):
        super().__init__()
        self.pooling = pooling
        self.input_dim = input_dim
        self.output_dim = output_dim

        if pooling == "attention":
            self.attention_weights = nn.Linear(input_dim, 1)
            self.attention_softmax = nn.Softmax(dim=1)

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if i > 0 else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.predictor = nn.Linear(in_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (B, T, input_dim) temporal features
            mask: (B, T) optional mask for padded positions

        Returns:
            Tuple of (pooled_output, attention_weights or None)
        """
        B, T, _ = x.shape

        if self.pooling == "attention":
            # Compute attention weights
            attn = self.attention_weights(x).squeeze(-1)  # (B, T)

            # Apply mask
            if mask is not None:
                attn = attn.masked_fill(mask, float('-inf'))

            attn_weights = self.attention_softmax(attn)  # (B, T)

            # Weighted sum
            pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, input_dim)
            return pooled, attn_weights

        elif self.pooling == "mean":
            if mask is not None:
                # Mean over valid positions
                valid_mask = ~mask
                pooled = (x * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True)
            else:
                pooled = x.mean(dim=1)
            return pooled, None

        elif self.pooling == "max":
            if mask is not None:
                # Max over valid positions
                x_masked = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
                pooled = x_masked.max(dim=1)[0]
            else:
                pooled = x.max(dim=1)[0]
            return pooled, None

        elif self.pooling == "last":
            if mask is not None:
                # Get last valid position for each sequence
                lengths = (~mask).sum(dim=1) - 1
                lengths = lengths.clamp(min=0)
                batch_indices = torch.arange(B, device=x.device)
                pooled = x[batch_indices, lengths]
            else:
                pooled = x[:, -1]
            return pooled, None

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")


class MultiScaleHead(nn.Module):
    """Head that uses multiple temporal scales for prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        scales: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.2
    ):
        super().__init__()
        self.scales = scales
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Per-scale heads
        self.scale_heads = nn.ModuleDict()
        for scale in scales:
            self.scale_heads[str(scale)] = nn.Sequential(
                nn.Linear(input_dim * scale, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(len(scales) * (hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with multi-scale pooling.

        Args:
            x: (B, T, input_dim) temporal features
            mask: (B, T) optional mask

        Returns:
            (B, output_dim) predictions
        """
        B, T, _ = x.shape
        scale_outputs = []

        for scale in self.scales:
            if scale == 1:
                # No pooling
                pooled = x.mean(dim=1) if mask is None else (x * (~mask).unsqueeze(-1)).sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
            elif scale == 2:
                # Pairwise
                if T % 2 == 0:
                    x_reshaped = x.view(B, T // 2, 2, -1)
                    pooled = x_reshaped.mean(dim=2).mean(dim=1)
                else:
                    pooled = x.mean(dim=1)
            else:
                # General downsampling
                new_len = T // scale
                if new_len > 0:
                    x_cropped = x[:, :new_len * scale]
                    x_reshaped = x_cropped.view(B, new_len, scale, -1)
                    pooled = x_reshaped.mean(dim=2).mean(dim=1)
                else:
                    pooled = x.mean(dim=1)

            # Process with scale head
            out = self.scale_heads[str(scale)](pooled)
            scale_outputs.append(out)

        # Concatenate and aggregate
        combined = torch.cat(scale_outputs, dim=-1)
        return self.aggregator(combined)


def create_head(
    task: str,
    input_dim: int,
    config: Optional[Dict] = None
) -> BaseTaskHead:
    """Factory function to create task heads.

    Args:
        task: Task type ("hr", "bvp", "sqi", "confidence")
        input_dim: Input dimension
        config: Optional configuration dict

    Returns:
        Appropriate head for the task
    """
    config = config or {}
    hidden_dim = config.get("hidden_dim", 64)
    dropout = config.get("dropout", 0.2)

    if task == "hr":
        return HRHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            hr_min=config.get("hr_min", 40.0),
            hr_max=config.get("hr_max", 200.0)
        )
    elif task == "bvp":
        return BVPHead(
            input_dim=input_dim,
            output_length=config.get("output_length", 128),
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    elif task == "sqi":
        return SQIHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    elif task == "confidence":
        return ConfidenceHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown task type: {task}")
