"""
P7 Self-Supervised Encoder

Encoder wrapper for self-supervised learning with projection heads.
Integrates with P6 temporal models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .base import SelfSupervisedConfig, SSLOutput


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps encoder representations to a space suitable for contrastive loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Final projection layer
        layers.extend([
            nn.Linear(in_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        ])

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project representations."""
        return self.projection(x)


class RepresentationHead(nn.Module):
    """
    Head for extracting fixed-size representations.

    Used for downstream tasks after SSL pre-training.
    """

    def __init__(
        self,
        input_dim: int,
        representation_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, representation_dim),
                nn.ReLU() if _ < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if _ < num_layers - 1 else nn.Identity()
            ])
            in_dim = representation_dim

        self.representation = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract representations."""
        return self.representation(x)


class SelfSupervisedEncoder(nn.Module):
    """
    Self-supervised encoder with projection and representation heads.

    Integrates:
    - P6 temporal encoder
    - Projection head for contrastive learning
    - Representation head for downstream tasks

    Usage:
    1. Initialize from P6 model
    2. Pre-train with SSL objectives
    3. Use representation head for downstream tasks
    """

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config

        # Get encoder from P6
        self.encoder_type = config.encoder_type

        # Import and create encoder based on type
        if config.encoder_type == "lstm":
            from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig
            enc_cfg = LSTMConfig(
                input_dim=config.input_dim,
                hidden_dim=config.encoder_hidden_dim,
                num_layers=config.encoder_num_layers,
                bidirectional=config.encoder_bidirectional,
                sequence_length=config.sequence_length
            )
            self.encoder = LSTMTemporal(enc_cfg)
            encoder_output_dim = config.encoder_hidden_dim * (2 if config.encoder_bidirectional else 1)

        elif config.encoder_type == "gru":
            from p6_temporal.model_gru import GRUTemporal, GRUConfig
            enc_cfg = GRUConfig(
                input_dim=config.input_dim,
                hidden_dim=config.encoder_hidden_dim,
                num_layers=config.encoder_num_layers,
                bidirectional=config.encoder_bidirectional,
                sequence_length=config.sequence_length
            )
            self.encoder = GRUTemporal(enc_cfg)
            encoder_output_dim = config.encoder_hidden_dim * (2 if config.encoder_bidirectional else 1)

        elif config.encoder_type == "transformer":
            from p6_temporal.model_transformer import TransformerTemporal, TransformerConfig
            enc_cfg = TransformerConfig(
                input_dim=config.input_dim,
                hidden_dim=config.encoder_hidden_dim,
                num_layers=config.encoder_num_layers,
                num_heads=4,
                sequence_length=config.sequence_length
            )
            self.encoder = TransformerTemporal(enc_cfg)
            encoder_output_dim = config.encoder_hidden_dim

        elif config.encoder_type == "attention":
            from p6_temporal.model_attention_temporal import AttentionTemporal, TemporalAttentionConfig
            enc_cfg = TemporalAttentionConfig(
                input_dim=config.input_dim,
                hidden_dim=config.encoder_hidden_dim,
                num_heads=4,
                sequence_length=config.sequence_length
            )
            self.encoder = AttentionTemporal(enc_cfg)
            encoder_output_dim = config.encoder_hidden_dim

        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")

        self.encoder_output_dim = encoder_output_dim

        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dim=config.projection_dim,
            output_dim=config.projection_dim,
            num_layers=config.projection_layers
        )

        # Representation head for downstream tasks
        self.representation_head = RepresentationHead(
            input_dim=encoder_output_dim,
            representation_dim=config.representation_dim,
            num_layers=1
        )

    def forward(
        self,
        x: torch.Tensor,
        return_projection: bool = True,
        return_representation: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> SSLOutput:
        """
        Forward pass through encoder and heads.

        Args:
            x: (B, T, F) input tensor
            return_projection: Whether to compute projection
            return_representation: Whether to compute representation
            mask: Optional sequence mask

        Returns:
            SSLOutput with representations and projections
        """
        # Encoder forward
        encoder_output = self.encoder(x, mask=mask)

        # Get pooled representation from encoder output
        # Use the hidden state or pooled output
        if hasattr(encoder_output, 'hidden_state') and encoder_output.hidden_state is not None:
            # For LSTM/GRU: use last hidden state
            if isinstance(encoder_output.hidden_state, tuple):
                hidden = encoder_output.hidden_state[0]  # (num_layers*directions, B, hidden_dim)
                # Average over layers and take last direction
                if hidden.dim() == 3:
                    hidden = hidden.mean(dim=0)  # (B, hidden_dim)
                pooled = hidden
            else:
                pooled = encoder_output.hidden_state
        else:
            # For Transformer/Attention: use BPM output as proxy for representation
            # (In a full implementation, we'd extract intermediate features)
            pooled = encoder_output.bpm.unsqueeze(-1)  # (B, 1)
            pooled = pooled.expand(-1, self.encoder_output_dim)  # (B, encoder_output_dim)

        # Normalize pooled representation
        pooled = F.normalize(pooled, dim=1)

        outputs = {}

        # Projection for contrastive learning
        if return_projection:
            projection = self.projection_head(pooled)
            projection = F.normalize(projection, dim=1)
            outputs['projection'] = projection

        # Representation for downstream tasks
        if return_representation:
            representation = self.representation_head(pooled)
            representation = F.normalize(representation, dim=1)
            outputs['representation'] = representation

        return SSLOutput(
            representations=outputs.get('representation', pooled),
            projections=outputs.get('projection', pooled)
        )

    def get_encoder(self) -> nn.Module:
        """Get the underlying encoder for fine-tuning."""
        return self.encoder

    def freeze_encoder(self):
        """Freeze encoder parameters for training projection head only."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def extract_representation(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract representation for downstream task.

        Args:
            x: (B, T, F) input tensor
            mask: Optional sequence mask

        Returns:
            (B, representation_dim) representations
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, return_projection=False, return_representation=True, mask=mask)
        return output.representations

    @classmethod
    def from_p6_checkpoint(
        cls,
        checkpoint_path: str,
        config: SelfSupervisedConfig,
        device: str = "cpu"
    ) -> 'SelfSupervisedEncoder':
        """
        Initialize encoder from a P6 checkpoint.

        Args:
            checkpoint_path: Path to P6 model checkpoint
            config: Self-supervised configuration
            device: Device to load to

        Returns:
            Initialized SelfSupervisedEncoder
        """
        ssl_encoder = cls(config)

        # Load encoder weights
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            ssl_encoder.encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            ssl_encoder.encoder.load_state_dict(checkpoint)

        return ssl_encoder


class SimpleTemporalEncoder(nn.Module):
    """
    Simple temporal encoder for when P6 models are not available.

    A basic temporal encoder that can be used standalone.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Simple LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output dimension
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Temporal pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, F) input
            mask: (B, T) optional mask

        Returns:
            (B, output_dim) pooled representation
        """
        B, T, F = x.shape

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # If bidirectional, concatenate forward and backward final hidden
        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            pooled = torch.cat([h_forward, h_backward], dim=-1)
        else:
            pooled = h_n[-1]

        return pooled


def create_ssl_encoder(
    config: SelfSupervisedConfig,
    use_p6: bool = True
) -> SelfSupervisedEncoder:
    """
    Factory function to create SSL encoder.

    Args:
        config: SSL configuration
        use_p6: Whether to use P6 models (True) or simple encoder (False)

    Returns:
        SelfSupervisedEncoder instance
    """
    if use_p6:
        return SelfSupervisedEncoder(config)
    else:
        # Create simple encoder config
        from .base import SelfSupervisedConfig as SimpleConfig
        simple_config = SimpleConfig(
            encoder_type="simple",
            encoder_hidden_dim=config.encoder_hidden_dim,
            encoder_num_layers=config.encoder_num_layers,
            encoder_bidirectional=config.encoder_bidirectional,
            input_dim=config.input_dim,
            projection_dim=config.projection_dim,
            representation_dim=config.representation_dim
        )
        return SimpleSelfSupervisedEncoder(simple_config)


class SimpleSelfSupervisedEncoder(nn.Module):
    """
    Simplified SSL encoder using basic temporal encoder.

    Used when P6 models are not available.
    """

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config

        # Simple temporal encoder
        self.encoder = SimpleTemporalEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            bidirectional=config.encoder_bidirectional
        )

        encoder_output_dim = config.encoder_hidden_dim * (2 if config.encoder_bidirectional else 1)

        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dim=config.projection_dim,
            output_dim=config.projection_dim,
            num_layers=config.projection_layers
        )

        # Representation head
        self.representation_head = RepresentationHead(
            input_dim=encoder_output_dim,
            representation_dim=config.representation_dim,
            num_layers=1
        )

    def forward(
        self,
        x: torch.Tensor,
        return_projection: bool = True,
        return_representation: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> SSLOutput:
        """Forward pass."""
        pooled = self.encoder(x, mask=mask)
        pooled = F.normalize(pooled, dim=1)

        outputs = {}

        if return_projection:
            projection = self.projection_head(pooled)
            projection = F.normalize(projection, dim=1)
            outputs['projection'] = projection

        if return_representation:
            representation = self.representation_head(pooled)
            representation = F.normalize(representation, dim=1)
            outputs['representation'] = representation

        return SSLOutput(
            representations=outputs.get('representation', pooled),
            projections=outputs.get('projection', pooled)
        )
