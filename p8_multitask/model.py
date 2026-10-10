"""
P8 Multi-Task Physiological Model

Unified multi-task learning model that supports multiple physiological
prediction tasks with a shared encoder and task-specific heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union

from .base import (
    MultiTaskConfig,
    MultiTaskOutput,
    TaskType,
)
from .heads import (
    HRHead,
    BVPHead,
    SQIHead,
    ConfidenceHead,
    TemporalPoolingHead,
)


class SharedEncoder(nn.Module):
    """Shared encoder for multi-task learning.

    Integrates with P5/P6 temporal models as the backbone.
    Can optionally initialize from P7 pretrained representations.
    """

    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config

        # Create encoder based on type
        if config.encoder_type == "lstm":
            self._create_lstm_encoder()
        elif config.encoder_type == "gru":
            self._create_gru_encoder()
        elif config.encoder_type == "transformer":
            self._create_transformer_encoder()
        elif config.encoder_type == "attention":
            self._create_attention_encoder()
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")

        # Determine output dimension
        self.output_dim = config.hidden_dim
        if config.bidirectional:
            self.output_dim *= 2

        # Representation normalization
        self.representation_norm = nn.LayerNorm(self.output_dim)

    def _create_lstm_encoder(self):
        """Create LSTM-based encoder."""
        from p6_temporal.model_lstm import LSTMTemporal, LSTMConfig

        enc_cfg = LSTMConfig(
            name="multitask_lstm",
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_encoder_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout,
            sequence_length=self.config.sequence_length
        )
        self.encoder = LSTMTemporal(enc_cfg)

    def _create_gru_encoder(self):
        """Create GRU-based encoder."""
        try:
            from p6_temporal.model_gru import GRUTemporal, GRUConfig

            enc_cfg = GRUConfig(
                name="multitask_gru",
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_encoder_layers,
                bidirectional=self.config.bidirectional,
                dropout=self.config.dropout,
                sequence_length=self.config.sequence_length
            )
            self.encoder = GRUTemporal(enc_cfg)
        except ImportError:
            # Fallback to simple GRU
            self._create_simple_encoder("gru")

    def _create_transformer_encoder(self):
        """Create Transformer-based encoder."""
        try:
            from p6_temporal.model_transformer import TransformerTemporal, TransformerConfig

            enc_cfg = TransformerConfig(
                name="multitask_transformer",
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_encoder_layers,
                num_heads=4,
                dropout=self.config.dropout,
                sequence_length=self.config.sequence_length
            )
            self.encoder = TransformerTemporal(enc_cfg)
        except ImportError:
            self._create_simple_encoder("transformer")

    def _create_attention_encoder(self):
        """Create attention-based encoder."""
        try:
            from p6_temporal.model_attention_temporal import AttentionTemporal, TemporalAttentionConfig

            enc_cfg = TemporalAttentionConfig(
                name="multitask_attention",
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=4,
                dropout=self.config.dropout,
                sequence_length=self.config.sequence_length
            )
            self.encoder = AttentionTemporal(enc_cfg)
        except ImportError:
            self._create_simple_encoder("attention")

    def _create_simple_encoder(self, kind: str):
        """Create simple encoder as fallback."""
        if kind == "gru":
            self.encoder = nn.GRU(
                input_size=self.config.input_dim,
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.num_encoder_layers,
                batch_first=True,
                dropout=self.config.dropout if self.config.num_encoder_layers > 1 else 0,
                bidirectional=self.config.bidirectional
            )
        elif kind == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.input_dim,
                nhead=4,
                dim_feedforward=self.config.hidden_dim,
                dropout=self.config.dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.config.num_encoder_layers
            )
        elif kind == "attention":
            # Simple self-attention
            self.encoder = nn.MultiheadAttention(
                embed_dim=self.config.input_dim,
                num_heads=4,
                dropout=self.config.dropout,
                batch_first=True
            )
        else:
            self.encoder = nn.LSTM(
                input_size=self.config.input_dim,
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.num_encoder_layers,
                batch_first=True,
                dropout=self.config.dropout if self.config.num_encoder_layers > 1 else 0,
                bidirectional=self.config.bidirectional
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """
        Forward pass through encoder.

        Args:
            x: (B, T, F) input features
            mask: (B, T) optional mask

        Returns:
            Tuple of (representation, attention_weights, hidden_state)
        """
        B, T, F = x.shape

        # Handle different encoder types
        if hasattr(self.encoder, 'forward'):
            # Use P6 model interface
            try:
                output = self.encoder(x, mask=mask)

                # Extract representation from TemporalOutput
                # The encoder returns TemporalOutput with bpm (B,) and hidden_state
                if hasattr(output, 'hidden_state') and output.hidden_state is not None:
                    # Use hidden state for representation
                    if isinstance(output.hidden_state, tuple):
                        hidden = output.hidden_state[0]  # h_n: (num_layers*directions, B, hidden_dim)
                        # Take last layer
                        if hidden.dim() == 3:
                            if self.config.bidirectional:
                                # For bidirectional: h_forward and h_backward
                                h = hidden[-2:]  # Last two (forward and backward)
                                representation = torch.cat([h[0], h[1]], dim=1)  # (B, hidden_dim*2)
                            else:
                                representation = hidden[-1]  # (B, hidden_dim)
                        else:
                            representation = hidden
                    else:
                        representation = output.hidden_state
                elif hasattr(output, 'bpm'):
                    # Use BPM output and expand to representation
                    # BPM is (B,), need to project to hidden_dim
                    bpm_expanded = output.bpm.unsqueeze(-1)  # (B, 1)
                    representation = bpm_expanded.expand(-1, self.output_dim)  # (B, hidden_dim)
                else:
                    representation = x.mean(dim=1)  # Fallback: mean pooling

                # Ensure correct dimension
                if representation.shape[-1] != self.output_dim:
                    # Project to correct dimension
                    if not hasattr(self, 'output_proj'):
                        self.output_proj = nn.Linear(representation.shape[-1], self.output_dim).to(x.device)
                    representation = self.output_proj(representation)

                attention = output.attention_weights if hasattr(output, 'attention_weights') else None
                hidden = output.hidden_state if hasattr(output, 'hidden_state') else None

                return representation, attention, hidden
            except Exception as e:
                pass

        # Fallback: use raw encoder
        if isinstance(self.encoder, nn.LSTM) or isinstance(self.encoder, nn.GRU):
            encoder_out, hidden = self.encoder(x)
            # Use last timestep
            if mask is not None:
                lengths = (~mask).sum(dim=1) - 1
                lengths = lengths.clamp(min=0)
                batch_idx = torch.arange(B, device=x.device)
                representation = encoder_out[batch_idx, lengths]
            else:
                representation = encoder_out[:, -1]
            return representation, None, hidden

        elif isinstance(self.encoder, nn.MultiheadAttention):
            # Self-attention
            attn_out, attn_weights = self.encoder(x, x, x, key_padding_mask=mask)
            representation = attn_out.mean(dim=1)
            return representation, attn_weights, None

        elif isinstance(self.encoder, nn.TransformerEncoder):
            # Transformer
            encoder_out = self.encoder(x, src_key_padding_mask=mask)
            representation = encoder_out.mean(dim=1)
            return representation, None, None

        else:
            # Generic forward
            encoder_out = self.encoder(x)
            representation = encoder_out.mean(dim=1)
            return representation, None, None

    def load_from_p7(self, checkpoint_path: str, device: str = "cpu"):
        """Load encoder weights from P7 checkpoint."""
        try:
            from p7_self_supervised.encoder import SelfSupervisedEncoder
            from p7_self_supervised.base import SelfSupervisedConfig

            # Load P7 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Try to extract encoder state
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Load into encoder
            self.encoder.load_state_dict(state_dict)
            print(f"Loaded encoder from P7 checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load from P7 checkpoint: {e}")


class MultiTaskPhysiologicalModel(nn.Module):
    """
    Unified multi-task model for physiological signal estimation.

    Architecture:
        Input (B, T, F) → Shared Encoder → Shared Representation
                                              ↓
                        ┌─────────────────────┼─────────────────────┐
                        ↓                     ↓                     ↓
                    HR Head              BVP Head              SQI/Conf Head

    Features:
        - Configurable shared encoder (P6 integration)
        - Optional P7 pretrained initialization
        - Task-specific heads with independent enabling
        - Missing label handling
        - Variable-length sequence support
    """

    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        self.enabled_tasks = config.get_enabled_tasks()

        # Shared encoder
        self.encoder = SharedEncoder(config)

        # Get encoder output dimension
        encoder_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        # No need for separate temporal pool - we use encoder output directly

        # Task-specific heads
        self.heads = nn.ModuleDict()

        if config.enable_hr:
            self.heads["hr"] = HRHead(
                input_dim=encoder_output_dim,
                hidden_dim=config.hidden_dim // 2,
                dropout=config.dropout
            )

        if config.enable_bvp:
            self.heads["bvp"] = BVPHead(
                input_dim=encoder_output_dim * 2,  # Include temporal features
                output_length=config.output_sequence_length,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                use_temporal_conv=True
            )

        if config.enable_sqi:
            self.heads["sqi"] = SQIHead(
                input_dim=encoder_output_dim,
                hidden_dim=config.hidden_dim // 2,
                dropout=config.dropout
            )

        if config.enable_confidence:
            self.heads["confidence"] = ConfidenceHead(
                input_dim=encoder_output_dim,
                hidden_dim=config.hidden_dim // 2,
                dropout=config.dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_representation: bool = True,
        enabled_tasks: Optional[List[TaskType]] = None
    ) -> MultiTaskOutput:
        """
        Forward pass through multi-task model.

        Args:
            x: (B, T, F) input features
            mask: (B, T) optional mask for variable-length sequences
            return_representation: Whether to include representation in output
            enabled_tasks: Override enabled tasks (default: use config)

        Returns:
            MultiTaskOutput with predictions for all enabled tasks
        """
        B, T, F = x.shape

        # Determine which tasks to run
        if enabled_tasks is None:
            enabled_tasks = self.enabled_tasks

        # Encode
        encoder_features, attention_weights, hidden_state = self.encoder(x, mask=mask)

        # Process encoder features for different uses
        B, T, F = x.shape

        # Get pooled representation for task heads
        if encoder_features.dim() == 2:
            representation = self.encoder.representation_norm(encoder_features)
        elif encoder_features.dim() == 3:
            representation = encoder_features.mean(dim=1)  # Pool temporal dimension
            representation = self.encoder.representation_norm(representation)
        else:
            representation = encoder_features

        # Initialize outputs
        output = MultiTaskOutput(
            representation=representation if return_representation else None,
            attention_weights=attention_weights,
            hidden_state=hidden_state
        )

        # For BVP task: need temporal features
        # Use the encoder output directly (temporal features) or pass None
        temporal_features = None
        if encoder_features.dim() == 3:
            # Encoder output has temporal dimension
            temporal_features = encoder_features

        # Run task heads
        for task in enabled_tasks:
            if task == TaskType.HR and "hr" in self.heads:
                output.hr = self.heads["hr"](representation)

            elif task == TaskType.BVP and "bvp" in self.heads:
                output.bvp = self.heads["bvp"](representation, temporal_features)

            elif task == TaskType.SQI and "sqi" in self.heads:
                output.sqi = self.heads["sqi"](representation)

            elif task == TaskType.CONFIDENCE and "confidence" in self.heads:
                output.confidence = self.heads["confidence"](representation)

        return output

    def get_encoder(self) -> SharedEncoder:
        """Get the shared encoder."""
        return self.encoder

    def get_head(self, task: TaskType) -> Optional[nn.Module]:
        """Get a specific task head."""
        task_name = task.value
        return self.heads.get(task_name)

    def freeze_encoder(self):
        """Freeze encoder parameters for task-specific training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_head(self, task: TaskType):
        """Freeze a specific task head."""
        head = self.get_head(task)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = False

    def unfreeze_head(self, task: TaskType):
        """Unfreeze a specific task head."""
        head = self.get_head(task)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters grouped by component."""
        encoder_params = list(self.encoder.parameters())
        head_params = list(self.heads.parameters())

        return encoder_params + head_params

    @classmethod
    def from_config(cls, config: MultiTaskConfig) -> 'MultiTaskPhysiologicalModel':
        """Create model from configuration."""
        return cls(config)

    @classmethod
    def create_with_pretrained_encoder(
        cls,
        config: MultiTaskConfig,
        encoder_state_dict: Dict[str, torch.Tensor],
        load_encoder: bool = True
    ) -> 'MultiTaskPhysiologicalModel':
        """Create model with pretrained encoder weights.

        Args:
            config: Model configuration
            encoder_state_dict: State dict for encoder
            load_encoder: Whether to load the weights

        Returns:
            Initialized model
        """
        model = cls(config)

        if load_encoder and encoder_state_dict:
            try:
                model.encoder.load_state_dict(encoder_state_dict, strict=False)
                print("Loaded pretrained encoder weights")
            except Exception as e:
                print(f"Warning: Could not load encoder weights: {e}")

        return model


def create_multitask_model(
    encoder_type: str = "lstm",
    input_dim: int = 3,
    hidden_dim: int = 128,
    tasks: Optional[List[str]] = None,
    **kwargs
) -> MultiTaskPhysiologicalModel:
    """
    Factory function to create multi-task model.

    Args:
        encoder_type: Type of encoder ("lstm", "gru", "transformer", "attention")
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        tasks: List of tasks to enable ["bvp", "hr", "sqi", "confidence"]
        **kwargs: Additional configuration options

    Returns:
        MultiTaskPhysiologicalModel instance
    """
    if tasks is None:
        tasks = ["bvp", "hr", "sqi", "confidence"]

    config = MultiTaskConfig(
        encoder_type=encoder_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        enable_bvp="bvp" in tasks,
        enable_hr="hr" in tasks,
        enable_sqi="sqi" in tasks,
        enable_confidence="confidence" in tasks,
        **kwargs
    )

    return cls(config)
