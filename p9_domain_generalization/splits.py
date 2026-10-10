"""
P9 Domain-Aware Data Splitting

Utilities for creating domain-aware train/validation/test splits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

from .base import (
    DomainMetadata,
    DomainSplit,
    DomainAwareConfig,
    SplitType,
    set_seed,
)


@dataclass
class DomainSplitResult:
    """Result of a domain split operation."""
    split: DomainSplit
    domain_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)  # split_name -> domain -> count
    warnings: List[str] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the split."""
        return {
            "split_type": self.split.split_type.value,
            "domain_key": self.split.domain_key,
            "train_count": self.split.train_count,
            "val_count": self.split.val_count,
            "test_count": self.split.test_count,
            "domain_counts": self.domain_counts,
            "held_out_domains": self.split.held_out_domains,
            "warnings": self.warnings,
        }


def create_domain_split(
    metadata: List[DomainMetadata],
    domain_key: str = "dataset",
    split_type: SplitType = SplitType.RANDOM,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_samples_per_domain: int = 5,
    subject_key: str = "subject"
) -> DomainSplitResult:
    """
    Create a domain-aware data split.

    Args:
        metadata: List of domain metadata for each sample
        domain_key: Key for domain grouping (dataset, device, environment, etc.)
        split_type: Type of split to create
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
        min_samples_per_domain: Minimum samples required per domain
        subject_key: Key for subject grouping (for subject-held-out splits)

    Returns:
        DomainSplitResult with split and statistics

    Raises:
        ValueError: If split configuration is invalid or impossible
    """
    set_seed(seed)
    np.random.seed(seed)

    warnings_list = []
    n_samples = len(metadata)

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Group samples by domain
    domain_samples: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        domain = meta.get_domain_key(domain_key)
        domain_samples[domain].append(i)

    # Group samples by subject
    subject_samples: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        subject = str(meta.subject_id) if meta.subject_id is not None else "unknown"
        subject_samples[subject].append(i)

    available_domains = list(domain_samples.keys())
    n_domains = len(available_domains)

    if n_domains < 2:
        warnings_list.append(
            f"Only {n_domains} domain(s) available. "
            f"Domain-held-out splits may not be meaningful."
        )

    # Check for domains with too few samples
    for domain, indices in domain_samples.items():
        if len(indices) < min_samples_per_domain:
            warnings_list.append(
                f"Domain '{domain}' has only {len(indices)} samples "
                f"(minimum: {min_samples_per_domain})."
            )

    if split_type == SplitType.RANDOM:
        split, domain_counts = _create_random_split(
            domain_samples, domain_key, train_ratio, val_ratio, seed
        )

    elif split_type == SplitType.SUBJECT_HELD_OUT:
        split, domain_counts = _create_subject_held_out_split(
            domain_samples, subject_samples, domain_key, train_ratio, val_ratio,
            test_ratio, seed, min_samples_per_domain
        )
        warnings_list.append("Subject-held-out split: samples from same subject will not appear in multiple splits.")

    elif split_type == SplitType.DOMAIN_HELD_OUT:
        split, domain_counts = _create_domain_held_out_split(
            domain_samples, domain_key, train_ratio, val_ratio, test_ratio,
            seed, min_samples_per_domain
        )

    elif split_type == SplitType.LEAVE_ONE_DOMAIN_OUT:
        # For leave-one-domain-out, we create a single split with one domain held out
        split, domain_counts = _create_leave_one_domain_out_split(
            domain_samples, domain_key, train_ratio, val_ratio, seed, min_samples_per_domain
        )

    else:
        raise ValueError(f"Unknown split type: {split_type}")

    result = DomainSplitResult(
        split=split,
        domain_counts=domain_counts,
        warnings=warnings_list
    )

    return result


def _create_random_split(
    domain_samples: Dict[str, List[int]],
    domain_key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[DomainSplit, Dict[str, Dict[str, int]]]:
    """Create a random split with domain-level stratification."""
    set_seed(seed)

    # Collect actual sample indices from domain_samples, preserving their values
    all_indices = []
    for indices in domain_samples.values():
        all_indices.extend(indices)
    all_indices = np.array(all_indices)
    np.random.shuffle(all_indices)

    n = len(all_indices)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = np.array(sorted(all_indices[:train_end]))
    val_indices = np.array(sorted(all_indices[train_end:val_end]))
    test_indices = np.array(sorted(all_indices[val_end:]))

    # Compute domain counts for each split
    domain_counts = _compute_domain_counts(
        train_indices, val_indices, test_indices, domain_samples, domain_key
    )

    split = DomainSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split_type=SplitType.RANDOM,
        domain_key=domain_key,
        seed=seed
    )

    return split, domain_counts


def _create_subject_held_out_split(
    domain_samples: Dict[str, List[int]],
    subject_samples: Dict[str, List[int]],
    domain_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_samples_per_domain: int
) -> Tuple[DomainSplit, Dict[str, Dict[str, int]]]:
    """Create a subject-held-out split."""
    set_seed(seed)

    subjects = list(subject_samples.keys())
    np.random.shuffle(subjects)

    n_subjects = len(subjects)
    train_end = int(n_subjects * train_ratio)
    val_end = train_end + int(n_subjects * val_ratio)

    train_subjects = set(subjects[:train_end])
    val_subjects = set(subjects[train_end:val_end])
    test_subjects = set(subjects[val_end:])

    train_indices = []
    val_indices = []
    test_indices = []

    for i, (subj, indices) in enumerate(subject_samples.items()):
        if subj in train_subjects:
            train_indices.extend(indices)
        elif subj in val_subjects:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    train_indices = np.array(sorted(train_indices))
    val_indices = np.array(sorted(val_indices))
    test_indices = np.array(sorted(test_indices))

    # Compute domain counts
    domain_counts = _compute_domain_counts(
        train_indices, val_indices, test_indices, domain_samples, domain_key
    )

    split = DomainSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split_type=SplitType.SUBJECT_HELD_OUT,
        domain_key=domain_key,
        seed=seed
    )

    return split, domain_counts


def _create_domain_held_out_split(
    domain_samples: Dict[str, List[int]],
    domain_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_samples_per_domain: int
) -> Tuple[DomainSplit, Dict[str, Dict[str, int]]]:
    """Create a domain-held-out split."""
    set_seed(seed)

    domains = list(domain_samples.keys())
    np.random.shuffle(domains)

    n_domains = len(domains)
    train_end = int(n_domains * train_ratio)
    val_end = train_end + int(n_domains * val_ratio)

    train_domains = domains[:train_end]
    val_domains = domains[train_end:val_end]
    test_domains = domains[val_end:]

    train_indices = []
    val_indices = []
    test_indices = []

    for domain, indices in domain_samples.items():
        if domain in train_domains:
            train_indices.extend(indices)
        elif domain in val_domains:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    train_indices = np.array(sorted(train_indices))
    val_indices = np.array(sorted(val_indices))
    test_indices = np.array(sorted(test_indices))

    domain_counts = {
        "train": {d: len(domain_samples[d]) for d in train_domains},
        "val": {d: len(domain_samples[d]) for d in val_domains},
        "test": {d: len(domain_samples[d]) for d in test_domains},
    }

    split = DomainSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split_type=SplitType.DOMAIN_HELD_OUT,
        domain_key=domain_key,
        seed=seed,
        train_domains={d: len(domain_samples[d]) for d in train_domains},
        val_domains={d: len(domain_samples[d]) for d in val_domains},
        test_domains={d: len(domain_samples[d]) for d in test_domains},
        held_out_domains=test_domains
    )

    return split, domain_counts


def _create_leave_one_domain_out_split(
    domain_samples: Dict[str, List[int]],
    domain_key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_samples_per_domain: int
) -> Tuple[DomainSplit, Dict[str, Dict[str, int]]]:
    """Create a leave-one-domain-out split (single held-out domain)."""
    set_seed(seed)

    domains = list(domain_samples.keys())
    np.random.shuffle(domains)

    if len(domains) < 2:
        raise ValueError(
            f"Leave-one-domain-out requires at least 2 domains, got {len(domains)}"
        )

    # Hold out the first domain
    held_out_domain = domains[0]
    remaining_domains = domains[1:]

    train_indices = []
    val_indices = []
    test_indices = domain_samples[held_out_domain].copy()

    # Split remaining domains into train/val
    np.random.shuffle(remaining_domains)
    n_train = max(1, int(len(remaining_domains) * train_ratio))

    train_domains = remaining_domains[:n_train]
    val_domains = remaining_domains[n_train:]

    for domain in train_domains:
        train_indices.extend(domain_samples[domain])
    for domain in val_domains:
        val_indices.extend(domain_samples[domain])

    train_indices = np.array(sorted(train_indices))
    val_indices = np.array(sorted(val_indices))
    test_indices = np.array(sorted(test_indices))

    all_train_domains = train_domains + val_domains

    domain_counts = {
        "train": {d: len(domain_samples[d]) for d in train_domains},
        "val": {d: len(domain_samples[d]) for d in val_domains},
        "test": {d: len(domain_samples[d]) for d in [held_out_domain]},
    }

    split = DomainSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split_type=SplitType.LEAVE_ONE_DOMAIN_OUT,
        domain_key=domain_key,
        seed=seed,
        train_domains={d: len(domain_samples[d]) for d in train_domains},
        val_domains={d: len(domain_samples[d]) for d in val_domains},
        test_domains={d: len(domain_samples[d]) for d in [held_out_domain]},
        held_out_domains=[held_out_domain]
    )

    return split, domain_counts


def _compute_domain_counts(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    domain_samples: Dict[str, List[int]],
    domain_key: str
) -> Dict[str, Dict[str, int]]:
    """Compute domain distribution for each split."""
    # Create reverse mapping
    sample_to_domain = {}
    for domain, indices in domain_samples.items():
        for idx in indices:
            sample_to_domain[idx] = domain

    train_domains = defaultdict(int)
    val_domains = defaultdict(int)
    test_domains = defaultdict(int)

    for idx in train_indices:
        train_domains[sample_to_domain.get(idx, "unknown")] += 1
    for idx in val_indices:
        val_domains[sample_to_domain.get(idx, "unknown")] += 1
    for idx in test_indices:
        test_domains[sample_to_domain.get(idx, "unknown")] += 1

    return {
        "train": dict(train_domains),
        "val": dict(val_domains),
        "test": dict(test_domains),
    }


def create_leave_one_domain_out_splits(
    metadata: List[DomainMetadata],
    domain_key: str = "dataset",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_samples_per_domain: int = 5
) -> List[DomainSplitResult]:
    """
    Create leave-one-domain-out cross-validation splits.

    Args:
        metadata: List of domain metadata
        domain_key: Key for domain grouping
        train_ratio: Fraction for training within remaining domains
        val_ratio: Fraction for validation within remaining domains
        seed: Base random seed
        min_samples_per_domain: Minimum samples per domain

    Returns:
        List of DomainSplitResult, one for each held-out domain
    """
    # Group by domain
    domain_samples: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        domain = meta.get_domain_key(domain_key)
        domain_samples[domain].append(i)

    domains = list(domain_samples.keys())

    if len(domains) < 2:
        raise ValueError(
            f"Leave-one-domain-out requires at least 2 domains, got {len(domains)}"
        )

    results = []

    for held_out_idx, held_out_domain in enumerate(domains):
        # Create split with this domain held out
        remaining_domains = [d for d in domains if d != held_out_domain]

        train_indices = []
        val_indices = []
        test_indices = domain_samples[held_out_domain].copy()

        # Split remaining domains
        np.random.seed(seed + held_out_idx)
        np.random.shuffle(remaining_domains)
        n_train = max(1, int(len(remaining_domains) * train_ratio))

        train_domains = remaining_domains[:n_train]
        val_domains = remaining_domains[n_train:]

        for domain in train_domains:
            train_indices.extend(domain_samples[domain])
        for domain in val_domains:
            val_indices.extend(domain_samples[domain])

        train_indices = np.array(sorted(train_indices))
        val_indices = np.array(sorted(val_indices))
        test_indices = np.array(sorted(test_indices))

        split = DomainSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            split_type=SplitType.LEAVE_ONE_DOMAIN_OUT,
            domain_key=domain_key,
            seed=seed,
            train_domains={d: len(domain_samples[d]) for d in train_domains},
            val_domains={d: len(domain_samples[d]) for d in val_domains},
            test_domains={d: len(domain_samples[d]) for d in [held_out_domain]},
            held_out_domains=[held_out_domain]
        )

        domain_counts = {
            "train": {d: len(domain_samples[d]) for d in train_domains},
            "val": {d: len(domain_samples[d]) for d in val_domains},
            "test": {d: len(domain_samples[d]) for d in [held_out_domain]},
        }

        results.append(DomainSplitResult(
            split=split,
            domain_counts=domain_counts,
            warnings=[f"Held-out domain: {held_out_domain}"]
        ))

    return results


def validate_split(
    split: DomainSplit,
    metadata: List[DomainMetadata],
    check_leakage: bool = True,
    check_subject_overlap: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate a domain split.

    Args:
        split: DomainSplit to validate
        metadata: Original metadata list
        check_leakage: Whether to check for data leakage
        check_subject_overlap: Whether to check for subject overlap

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for overlapping indices
    train_set = set(split.train_indices)
    val_set = set(split.val_indices)
    test_set = set(split.test_indices)

    if train_set & val_set:
        issues.append("Train and validation sets overlap")
    if train_set & test_set:
        issues.append("Train and test sets overlap")
    if val_set & test_set:
        issues.append("Validation and test sets overlap")

    # Check for missing indices
    all_indices = set(range(len(metadata)))
    split_indices = train_set | val_set | test_set
    if split_indices != all_indices:
        missing = all_indices - split_indices
        issues.append(f"Missing indices: {len(missing)} samples not assigned")

    # Check for subject overlap if requested
    if check_subject_overlap:
        train_subjects = set()
        val_subjects = set()
        test_subjects = set()

        for idx in split.train_indices:
            subj = metadata[idx].subject_id
            if subj is not None:
                train_subjects.add(subj)
        for idx in split.val_indices:
            subj = metadata[idx].subject_id
            if subj is not None:
                val_subjects.add(subj)
        for idx in split.test_indices:
            subj = metadata[idx].subject_id
            if subj is not None:
                test_subjects.add(subj)

        subject_overlap_train_val = train_subjects & val_subjects
        subject_overlap_train_test = train_subjects & test_subjects

        if subject_overlap_train_val:
            issues.append(
                f"Subject overlap between train and val: {len(subject_overlap_train_val)} subjects"
            )
        if subject_overlap_train_test:
            issues.append(
                f"Subject overlap between train and test: {len(subject_overlap_train_test)} subjects"
            )

    # Check for held-out domain leakage
    if check_leakage and split.held_out_domains:
        held_out_set = set(split.held_out_domains)

        # Check if held-out domains appear in train
        train_domains_in_split = split.train_domains.keys() if hasattr(split, 'train_domains') else set()
        overlap = held_out_set & set(train_domains_in_split)
        if overlap:
            issues.append(f"Held-out domains appear in train: {overlap}")

    is_valid = len(issues) == 0
    return is_valid, issues


def compute_split_hash(split: DomainSplit) -> str:
    """
    Compute a hash of the split for reproducibility.

    Args:
        split: DomainSplit to hash

    Returns:
        Hash string
    """
    # Create a deterministic string representation
    content = (
        f"{split.split_type.value}|"
        f"{split.domain_key}|"
        f"{split.seed}|"
        f"{len(split.train_indices)}|"
        f"{len(split.val_indices)}|"
        f"{len(split.test_indices)}|"
        f"{sorted(split.train_indices)[:10]}|"
        f"{sorted(split.test_indices)[:10]}"
    )
    return hashlib.md5(content.encode()).hexdigest()[:16]
