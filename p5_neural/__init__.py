"""
P5 Neural Baseline Package
Five model families for Sanubari rPPG estimation.
"""

__version__ = "1.0.0"

# Base classes
from p5_neural.base import BaseModel, ModelConfig, ModelOutput, set_seed, get_device

# Models
from p5_neural.model_temporal_cnn import TemporalCNN, TemporalCNNConfig
from p5_neural.model_cnn_temporal import CNNTemporalModel
from p5_neural.model_physnet import PhysNetStyleModel, PhysNetLite
from p5_neural.model_attention import TSCANStyleModel, SimpleAttentionModel
from p5_neural.model_hybrid import HybridModel, LightweightHybrid

# Training
from p5_neural.training import (
    RMSELoss, MAPELoss, CombinedLoss,
    get_optimizer, get_scheduler,
    compute_metrics, train_epoch, validate,
    save_checkpoint, load_checkpoint
)

# Dataset
from p5_neural.dataset import SyntheticVideoDataset, SignalDataset

__all__ = [
    "__version__",
    "BaseModel", "ModelConfig", "ModelOutput", "set_seed", "get_device",
    "TemporalCNN", "CNNTemporalModel",
    "PhysNetStyleModel", "PhysNetLite",
    "TSCANStyleModel", "SimpleAttentionModel",
    "HybridModel", "LightweightHybrid",
    "RMSELoss", "MAPELoss", "CombinedLoss",
    "get_optimizer", "get_scheduler",
    "compute_metrics", "train_epoch", "validate",
    "save_checkpoint", "load_checkpoint",
    "SyntheticVideoDataset", "SignalDataset",
]
