"""
P8 Multi-Task Physiological Engine

A unified multi-task learning system that learns shared physiological
representations and supports multiple related prediction tasks.

Tasks Supported:
    - BVP waveform prediction
    - Heart rate (HR) estimation
    - Signal quality index (SQI) prediction
    - Prediction confidence / reliability estimation

Architecture:
    Shared Encoder (P5/P6/P7) → Task-Specific Heads
                                ├── BVP Head
                                ├── HR Head
                                ├── SQI Head
                                └── Confidence Head
"""

__version__ = "1.0.0"

from .base import (
    MultiTaskConfig,
    MultiTaskOutput,
    TaskType,
    set_seed,
    get_device,
)

from .model import MultiTaskPhysiologicalModel

from .heads import (
    BVPHead,
    HRHead,
    SQIHead,
    ConfidenceHead,
)

from .losses import (
    MultiTaskLoss,
    BVPLossFn,
    HRLossFn,
    SQILossFn,
    ConfidenceLossFn,
)

from .dataset import (
    MultiTaskDataset,
    SyntheticMultiTaskDataset,
    collate_multitask_batch,
    create_multitask_dataloader,
)

from .training import (
    train_multitask_epoch,
    validate_multitask,
    save_checkpoint,
    load_checkpoint,
    get_optimizer,
    get_scheduler,
)

__all__ = [
    "__version__",
    # Base
    "MultiTaskConfig",
    "MultiTaskOutput",
    "TaskType",
    "set_seed",
    "get_device",
    # Model
    "MultiTaskPhysiologicalModel",
    # Heads
    "BVPHead",
    "HRHead",
    "SQIHead",
    "ConfidenceHead",
    # Losses
    "MultiTaskLoss",
    "BVPLossFn",
    "HRLossFn",
    "SQILossFn",
    "ConfidenceLossFn",
    # Dataset
    "MultiTaskDataset",
    "SyntheticMultiTaskDataset",
    "collate_multitask_batch",
    "create_multitask_dataloader",
    # Training
    "train_multitask_epoch",
    "validate_multitask",
    "save_checkpoint",
    "load_checkpoint",
    "get_optimizer",
    "get_scheduler",
]
