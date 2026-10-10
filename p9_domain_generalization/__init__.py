"""
P9 Domain Generalization Module

Domain-aware training and evaluation for multi-task physiological modeling.

This module provides:
    - Domain metadata structures and utilities
    - Domain-aware data splitting
    - Domain generalization training methods (ERM, GroupDRO)
    - Cross-domain evaluation infrastructure
"""

__version__ = "1.0.0"

from .base import (
    DomainMetadata,
    DomainSplit,
    DomainAwareConfig,
    EvaluationProtocol,
    SplitType,
    DomainAwareDataset,
    set_seed,
)

from .splits import (
    create_domain_split,
    create_leave_one_domain_out_splits,
    validate_split,
    DomainSplitResult,
)

from .sampling import (
    DomainBalancedSampler,
    DomainWeightedSampler,
    GroupDROSampler,
    create_domain_sampler,
)

from .training import (
    DomainGeneralizationTrainer,
    GroupDROLoss,
    train_domain_generalization,
    evaluate_per_domain,
)

from .evaluation import (
    DomainMetrics,
    compute_domain_metrics,
    aggregate_across_domains,
    check_leakage,
    summarize_domain_shift,
)

__all__ = [
    "__version__",
    # Base
    "DomainMetadata",
    "DomainSplit",
    "DomainAwareConfig",
    "EvaluationProtocol",
    "SplitType",
    "DomainAwareDataset",
    "set_seed",
    # Splits
    "create_domain_split",
    "create_leave_one_domain_out_splits",
    "validate_split",
    "DomainSplitResult",
    # Sampling
    "DomainBalancedSampler",
    "DomainWeightedSampler",
    "GroupDROSampler",
    "create_domain_sampler",
    # Training
    "DomainGeneralizationTrainer",
    "GroupDROLoss",
    "train_domain_generalization",
    "evaluate_per_domain",
    # Evaluation
    "DomainMetrics",
    "compute_domain_metrics",
    "aggregate_across_domains",
    "check_leakage",
    "summarize_domain_shift",
]
