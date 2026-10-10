"""
P6 Temporal Physiological Model Package

Temporal modeling for physiological signal estimation from sequential inputs.

Models:
    - LSTM-based temporal model
    - GRU-based temporal model
    - Transformer-based temporal model
    - ConvLSTM for spatiotemporal patterns
    - Bidirectional temporal models
"""

__version__ = "1.0.0"

# Base and configuration
from p6_temporal.base import TemporalModelConfig, TemporalOutput, set_seed, get_device

# Temporal models
from p6_temporal.model_lstm import LSTMTemporal, BiLSTMTemporal, LSTMConfig
from p6_temporal.model_gru import GRUTemporal, BiGRUTemporal, GRUConfig
from p6_temporal.model_transformer import TransformerTemporal, TransformerConfig
from p6_temporal.model_convlstm import ConvLSTMTemporal, ConvLSTMConfig
from p6_temporal.model_attention_temporal import AttentionTemporal, TemporalAttentionConfig

# Dataset
from p6_temporal.dataset import TemporalSignalDataset, SlidingWindowDataset, collate_temporal_batch

# Training
from p6_temporal.training import (
    TemporalLoss, BVPLoss, FrequencyLoss,
    train_temporal_epoch, validate_temporal,
    save_checkpoint, load_checkpoint
)

__all__ = [
    "__version__",
    "TemporalModelConfig", "TemporalOutput", "set_seed", "get_device",
    "LSTMTemporal", "BiLSTMTemporal", "LSTMConfig",
    "GRUTemporal", "BiGRUTemporal", "GRUConfig",
    "TransformerTemporal", "TransformerConfig",
    "ConvLSTMTemporal", "ConvLSTMConfig",
    "AttentionTemporal", "TemporalAttentionConfig",
    "TemporalSignalDataset", "SlidingWindowDataset", "collate_temporal_batch",
    "TemporalLoss", "BVPLoss", "FrequencyLoss",
    "train_temporal_epoch", "validate_temporal",
    "save_checkpoint", "load_checkpoint",
]
