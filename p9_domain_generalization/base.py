"""
P9 Domain Generalization Base

Domain metadata structures and utilities for domain-aware training.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib


class SplitType(Enum):
    """Types of data splits for domain generalization."""
    RANDOM = "random"                    # Random train/val/test split
    SUBJECT_HELD_OUT = "subject_held_out"  # Hold out subjects
    DOMAIN_HELD_OUT = "domain_held_out"   # Hold out entire domains
    LEAVE_ONE_DOMAIN_OUT = "leave_one_domain_out"  # Leave-one-domain-out cross-validation


class EvaluationProtocol(Enum):
    """Evaluation protocols for domain generalization."""
    IN_DISTRIBUTION = "in_distribution"    # Standard train/val/test within same distribution
    DOMAIN_GENERALIZATION = "domain_generalization"  # Evaluate on held-out domains
    LEAVE_ONE_DOMAIN_OUT = "leave_one_domain_out"  # Cross-validation over domains


@dataclass
class DomainMetadata:
    """
    Domain metadata for a single sample.

    Available domain identifiers:
        - dataset: Source dataset name
        - device: Recording device
        - environment: Recording environment (lab, outdoor, etc.)
        - lighting: Lighting condition
        - subject_id: Subject identifier
        - session_id: Recording session
        - protocol: Experimental protocol

    If metadata is not available, fields will be None.
    """
    sample_id: str                    # Unique sample identifier
    subject_id: Optional[int] = None  # Subject identifier
    dataset: Optional[str] = None     # Source dataset name
    device: Optional[str] = None       # Recording device
    environment: Optional[str] = None   # Recording environment
    lighting: Optional[str] = None     # Lighting condition
    session_id: Optional[str] = None  # Recording session
    protocol: Optional[str] = None    # Experimental protocol

    # Additional metadata (flexible storage)
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_domain_key(self, domain_type: str) -> str:
        """
        Get the domain key for a given domain type.

        Args:
            domain_type: One of "dataset", "device", "environment", "lighting", "subject", "session"

        Returns:
            Domain key as string, or "unknown" if not available
        """
        domain_map = {
            "dataset": self.dataset,
            "device": self.device,
            "environment": self.environment,
            "lighting": self.lighting,
            "subject": str(self.subject_id) if self.subject_id is not None else None,
            "session": self.session_id,
            "protocol": self.protocol,
        }
        value = domain_map.get(domain_type)
        return value if value is not None else "unknown"

    def get_all_domains(self) -> Dict[str, str]:
        """Get all available domain identifiers."""
        domains = {}
        for key in ["dataset", "device", "environment", "lighting", "session", "protocol"]:
            value = getattr(self, key, None)
            if value is not None:
                domains[key] = value
        if self.subject_id is not None:
            domains["subject"] = str(self.subject_id)
        return domains

    @classmethod
    def create_unknown(cls, sample_id: str, subject_id: Optional[int] = None) -> 'DomainMetadata':
        """Create metadata with all unknown fields."""
        return cls(sample_id=sample_id, subject_id=subject_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "dataset": self.dataset,
            "device": self.device,
            "environment": self.environment,
            "lighting": self.lighting,
            "session_id": self.session_id,
            "protocol": self.protocol,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DomainMetadata':
        """Create from dictionary."""
        known_keys = {"sample_id", "subject_id", "dataset", "device", "environment",
                      "lighting", "session_id", "protocol"}
        known = {k: d.get(k) for k in known_keys}
        extra = {k: v for k, v in d.items() if k not in known_keys}
        return cls(**known, extra=extra)


@dataclass
class DomainSplit:
    """
    Result of a domain-aware data split.

    Contains indices for train, validation, and test sets,
    along with metadata about the split.
    """
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray

    # Domain information
    train_domains: Dict[str, List[int]] = field(default_factory=dict)
    val_domains: Dict[str, List[int]] = field(default_factory=dict)
    test_domains: Dict[str, List[int]] = field(default_factory=dict)

    # Held-out domains (for held-out domain splits)
    held_out_domains: Optional[List[str]] = None

    # Metadata
    split_type: SplitType = SplitType.RANDOM
    domain_key: str = "dataset"
    seed: int = 42

    # Statistics
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0

    def __post_init__(self):
        """Compute counts if not provided."""
        if self.train_count == 0:
            self.train_count = len(self.train_indices)
        if self.val_count == 0:
            self.val_count = len(self.val_indices)
        if self.test_count == 0:
            self.test_count = len(self.test_indices)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the split."""
        return {
            "split_type": self.split_type.value,
            "domain_key": self.domain_key,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "total_count": self.train_count + self.val_count + self.test_count,
            "train_domain_count": len(self.train_domains),
            "held_out_domains": self.held_out_domains,
        }

    def get_domain_distribution(self, indices: np.ndarray, metadata_list: List[DomainMetadata]) -> Dict[str, int]:
        """Get domain distribution for given indices."""
        dist = defaultdict(int)
        for idx in indices:
            meta = metadata_list[idx]
            key = meta.get_domain_key(self.domain_key)
            dist[key] += 1
        return dict(dist)


@dataclass
class DomainAwareConfig:
    """Configuration for domain-aware training."""
    # Split configuration
    split_type: SplitType = SplitType.RANDOM
    domain_key: str = "dataset"  # Key for domain grouping
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Subject/group separation
    subject_held_out: bool = False
    min_samples_per_domain: int = 5

    # Domain generalization method
    method: str = "erm"  # "erm", "groupdro", "domain_balanced"
    groupdro_eta: float = 0.01  # Step size for GroupDRO
    domain_weighting: str = "uniform"  # "uniform", "inverse", "sqrt_inverse"

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    epochs: int = 50

    # Evaluation
    evaluation_protocol: EvaluationProtocol = EvaluationProtocol.IN_DISTRIBUTION

    # Reproducibility
    seed: int = 42
    device: str = "cpu"


class DomainAwareDataset:
    """
    Dataset wrapper that includes domain metadata.

    Wraps an existing dataset and adds domain-aware functionality.
    """

    def __init__(
        self,
        base_dataset,  # Any dataset with features
        metadata: Optional[List[DomainMetadata]] = None,
        domain_key: str = "dataset"
    ):
        """
        Args:
            base_dataset: Underlying dataset
            metadata: List of domain metadata for each sample
            domain_key: Primary key for domain grouping
        """
        self.base_dataset = base_dataset
        self.metadata = metadata
        self.domain_key = domain_key

        # Generate metadata if not provided
        if metadata is None:
            self.metadata = []
            for i in range(len(base_dataset)):
                self.metadata.append(DomainMetadata.create_unknown(
                    sample_id=f"sample_{i}",
                    subject_id=i // 10  # Default grouping
                ))

        # Precompute domain indices
        self._domain_indices: Dict[str, List[int]] = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            domain = meta.get_domain_key(domain_key)
            self._domain_indices[domain].append(i)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """Get sample from base dataset."""
        return self.base_dataset[idx]

    def get_domain_of_sample(self, idx: int) -> str:
        """Get the domain of a sample."""
        if idx < len(self.metadata):
            return self.metadata[idx].get_domain_key(self.domain_key)
        return "unknown"

    def get_indices_by_domain(self, domain: str) -> List[int]:
        """Get all sample indices for a domain."""
        return self._domain_indices.get(domain, [])

    def get_all_domains(self) -> List[str]:
        """Get all unique domains."""
        return list(self._domain_indices.keys())

    def get_domain_counts(self) -> Dict[str, int]:
        """Get count of samples per domain."""
        return {d: len(indices) for d, indices in self._domain_indices.items()}

    def filter_by_domain(self, domains: List[str]) -> List[int]:
        """Get indices of samples in specified domains."""
        result = []
        for domain in domains:
            result.extend(self._domain_indices.get(domain, []))
        return result

    def filter_excluding_domain(self, domain: str) -> List[int]:
        """Get indices of samples NOT in specified domain."""
        excluded = set(self._domain_indices.get(domain, []))
        return [i for i in range(len(self)) if i not in excluded]

    def get_subject_ids(self) -> List[Optional[int]]:
        """Get all subject IDs."""
        return [meta.subject_id for meta in self.metadata]

    def create_split(
        self,
        train_domains: List[str],
        val_domains: List[str],
        test_domains: List[str],
        seed: int = 42
    ) -> DomainSplit:
        """
        Create a domain split.

        Args:
            train_domains: Domains to use for training
            val_domains: Domains to use for validation
            test_domains: Domains to use for testing

        Returns:
            DomainSplit with indices
        """
        np.random.seed(seed)

        train_indices = []
        val_indices = []
        test_indices = []

        for i, meta in enumerate(self.metadata):
            domain = meta.get_domain_key(self.domain_key)

            if domain in train_domains:
                train_indices.append(i)
            elif domain in val_domains:
                val_indices.append(i)
            elif domain in test_domains:
                test_indices.append(i)

        return DomainSplit(
            train_indices=np.array(train_indices),
            val_indices=np.array(val_indices),
            test_indices=np.array(test_indices),
            train_domains={d: len(self._domain_indices.get(d, [])) for d in train_domains},
            val_domains={d: len(self._domain_indices.get(d, [])) for d in val_domains},
            test_domains={d: len(self._domain_indices.get(d, [])) for d in test_domains},
            held_out_domains=[d for d in self.get_all_domains() if d not in train_domains + val_domains + test_domains],
            split_type=SplitType.DOMAIN_HELD_OUT,
            domain_key=self.domain_key,
            seed=seed
        )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
