"""
P7 Self-Supervised Physiological Representation Learning Package

Self-supervised learning methods for learning physiological representations
from unlabeled or weakly labeled video and signal sequences.

Methods:
    - Temporal Contrastive (SimCLR-style)
    - Physiological Temporal Consistency
    - Predictive Coding / Masked Reconstruction
    - Multi-crop Temporal Augmentation
"""

__version__ = "1.0.0"

# Base configuration
from p7_self_supervised.base import SelfSupervisedConfig, set_seed, get_device

# Self-supervised objectives
from p7_self_supervised.objectives import (
    ContrastiveLoss,
    TemporalConsistencyLoss,
    PredictiveLoss,
    BYOLLoss,
    CombinedSSLLoss
)

# Augmentations
from p7_self_supervised.augmentations import (
    TemporalCrop,
    GaussianNoise,
    SignalDropout,
    TimeShift,
    AmplitudeScale,
    ComposeAugmentations,
    PhysiologicalAugmenter,
    create_multiview_augmenter
)

# Encoder wrapper
from p7_self_supervised.encoder import (
    SelfSupervisedEncoder,
    SimpleSelfSupervisedEncoder,
    ProjectionHead,
    RepresentationHead
)

# Dataset
from p7_self_supervised.dataset import (
    UnlabeledTemporalDataset,
    SyntheticSelfSupervisedDataset,
    create_ssl_dataloader
)

# Training
from p7_self_supervised.training import (
    train_ssl_epoch,
    validate_ssl,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    "__version__",
    "SelfSupervisedConfig", "set_seed", "get_device",
    "ContrastiveLoss", "TemporalConsistencyLoss", "PredictiveLoss",
    "BYOLLoss", "CombinedSSLLoss",
    "TemporalCrop", "GaussianNoise", "SignalDropout",
    "TimeShift", "AmplitudeScale", "ComposeAugmentations",
    "PhysiologicalAugmenter", "create_multiview_augmenter",
    "SelfSupervisedEncoder", "SimpleSelfSupervisedEncoder", "ProjectionHead", "RepresentationHead",
    "UnlabeledTemporalDataset", "SyntheticSelfSupervisedDataset",
    "create_ssl_dataloader",
    "train_ssl_epoch", "validate_ssl", "save_checkpoint", "load_checkpoint",
]
