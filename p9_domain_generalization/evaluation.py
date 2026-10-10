"""
P9 Domain Evaluation

Evaluation utilities for cross-domain and domain generalization assessment.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

from .base import DomainSplit


@dataclass
class DomainMetrics:
    """Metrics for a single domain."""
    domain: str
    sample_count: int

    # Task-specific metrics
    hr_mae: Optional[float] = None
    hr_rmse: Optional[float] = None
    hr_bias: Optional[float] = None

    bvp_mae: Optional[float] = None
    bvp_corr: Optional[float] = None

    sqi_mae: Optional[float] = None
    sqi_accuracy: Optional[float] = None

    # Overall
    loss: Optional[float] = None
    sample_count: int = 0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across domains."""
    # Mean metrics across domains
    mean_hr_mae: Optional[float] = None
    mean_hr_rmse: Optional[float] = None
    std_hr_mae: Optional[float] = None

    mean_bvp_mae: Optional[float] = None
    std_bvp_mae: Optional[float] = None

    mean_sqi_mae: Optional[float] = None
    std_sqi_mae: Optional[float] = None

    # Per-domain breakdown
    per_domain: Dict[str, DomainMetrics] = field(default_factory=dict)

    # Statistics
    num_domains: int = 0
    total_samples: int = 0
    domain_sample_counts: Dict[str, int] = field(default_factory=dict)


def compute_domain_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    domain: str = "unknown"
) -> DomainMetrics:
    """
    Compute metrics for a single domain.

    Args:
        predictions: Dict of predictions per task
        targets: Dict of targets per task
        domain: Domain name

    Returns:
        DomainMetrics with computed values
    """
    metrics = DomainMetrics(domain=domain, sample_count=0)

    # Count valid samples
    if "hr" in predictions and "hr" in targets:
        hr_pred = predictions["hr"]
        hr_target = targets["hr"]

        if hr_pred.numel() > 0 and hr_target.numel() > 0:
            hr_mae = float(torch.mean(torch.abs(hr_pred - hr_target)))
            hr_rmse = float(torch.sqrt(torch.mean((hr_pred - hr_target) ** 2)))
            hr_bias = float(torch.mean(hr_pred - hr_target))

            metrics.hr_mae = hr_mae
            metrics.hr_rmse = hr_rmse
            metrics.hr_bias = hr_bias
            metrics.sample_count = hr_pred.numel()

    # BVP metrics
    if "bvp" in predictions and "bvp" in targets:
        bvp_pred = predictions["bvp"]
        bvp_target = targets["bvp"]

        if bvp_pred.numel() > 0 and bvp_target.numel() > 0:
            bvp_mae = float(torch.mean(torch.abs(bvp_pred - bvp_target)))
            metrics.bvp_mae = bvp_mae

            # Correlation
            if bvp_pred.dim() == 1:
                corr = torch.corrcoef(torch.stack([bvp_pred, bvp_target]))[0, 1]
                metrics.bvp_corr = float(corr)

    # SQI metrics
    if "sqi" in predictions and "sqi" in targets:
        sqi_pred = predictions["sqi"]
        sqi_target = targets["sqi"]

        if sqi_pred.numel() > 0 and sqi_target.numel() > 0:
            sqi_mae = float(torch.mean(torch.abs(sqi_pred - sqi_target)))
            metrics.sqi_mae = sqi_mae

    return metrics


def aggregate_across_domains(
    per_domain_metrics: Dict[str, DomainMetrics]
) -> AggregatedMetrics:
    """
    Aggregate metrics across domains.

    Args:
        per_domain_metrics: Dict mapping domain to DomainMetrics

    Returns:
        AggregatedMetrics with means, stds, and breakdowns
    """
    if not per_domain_metrics:
        return AggregatedMetrics()

    # Collect values for each metric
    hr_mae_values = []
    hr_rmse_values = []
    bvp_mae_values = []
    sqi_mae_values = []

    domain_sample_counts = {}

    for domain, metrics in per_domain_metrics.items():
        domain_sample_counts[domain] = metrics.sample_count

        if metrics.hr_mae is not None:
            hr_mae_values.append(metrics.hr_mae)
        if metrics.hr_rmse is not None:
            hr_rmse_values.append(metrics.hr_rmse)
        if metrics.bvp_mae is not None:
            bvp_mae_values.append(metrics.bvp_mae)
        if metrics.sqi_mae is not None:
            sqi_mae_values.append(metrics.sqi_mae)

    # Compute means and stds
    agg = AggregatedMetrics(
        per_domain=per_domain_metrics,
        num_domains=len(per_domain_metrics),
        total_samples=sum(domain_sample_counts.values()),
        domain_sample_counts=domain_sample_counts
    )

    if hr_mae_values:
        agg.mean_hr_mae = float(np.mean(hr_mae_values))
        agg.std_hr_mae = float(np.std(hr_mae_values))

    if hr_rmse_values:
        agg.mean_hr_rmse = float(np.mean(hr_rmse_values))

    if bvp_mae_values:
        agg.mean_bvp_mae = float(np.mean(bvp_mae_values))
        agg.std_bvp_mae = float(np.std(bvp_mae_values))

    if sqi_mae_values:
        agg.mean_sqi_mae = float(np.mean(sqi_mae_values))
        agg.std_sqi_mae = float(np.std(sqi_mae_values))

    return agg


def check_leakage(
    split: DomainSplit,
    metadata: List,
    domain_key: str = "dataset"
) -> Tuple[bool, List[str]]:
    """
    Check for potential data leakage in a split.

    Args:
        split: DomainSplit to check
        metadata: Original metadata list
        domain_key: Key for domain grouping

    Returns:
        Tuple of (has_leakage, list_of_issues)
    """
    issues = []

    # Check index overlap
    train_set = set(split.train_indices)
    val_set = set(split.val_indices)
    test_set = set(split.test_indices)

    if train_set & val_set:
        issues.append(f"Train-val overlap: {len(train_set & val_set)} samples")

    if train_set & test_set:
        issues.append(f"Train-test overlap: {len(train_set & test_set)} samples")

    if val_set & test_set:
        issues.append(f"Val-test overlap: {len(val_set & test_set)} samples")

    # Check subject overlap (if subject_id available)
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

    if train_subjects & val_subjects:
        issues.append(f"Subject overlap train-val: {len(train_subjects & val_subjects)} subjects")

    if train_subjects & test_subjects:
        issues.append(f"Subject overlap train-test: {len(train_subjects & test_subjects)} subjects")

    # Check domain held-out consistency
    if split.held_out_domains:
        train_domains = set(split.train_domains.keys()) if split.train_domains else set()
        for held_out in split.held_out_domains:
            if held_out in train_domains:
                issues.append(f"Held-out domain '{held_out}' appears in train domains")

    has_leakage = len(issues) > 0
    return has_leakage, issues


def format_domain_results(
    per_domain: Dict[str, DomainMetrics],
    aggregate: AggregatedMetrics,
    include_unsupported: bool = False
) -> str:
    """
    Format domain results for display.

    Args:
        per_domain: Per-domain metrics
        aggregate: Aggregated metrics
        include_unsupported: Whether to include unsupported metrics

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Domain Generalization Results")
    lines.append("=" * 60)

    # Aggregate summary
    lines.append("\nAggregate Performance:")
    lines.append("-" * 40)

    if aggregate.mean_hr_mae is not None:
        lines.append(f"  HR MAE: {aggregate.mean_hr_mae:.2f} ± {aggregate.std_hr_mae:.2f} BPM")

    if aggregate.mean_bvp_mae is not None:
        lines.append(f"  BVP MAE: {aggregate.mean_bvp_mae:.4f} ± {aggregate.std_bvp_mae:.4f}")

    if aggregate.mean_sqi_mae is not None:
        lines.append(f"  SQI MAE: {aggregate.mean_sqi_mae:.4f} ± {aggregate.std_sqi_mae:.4f}")

    lines.append(f"\n  Domains evaluated: {aggregate.num_domains}")
    lines.append(f"  Total samples: {aggregate.total_samples}")

    # Per-domain breakdown
    lines.append("\nPer-Domain Performance:")
    lines.append("-" * 40)

    for domain, metrics in per_domain.items():
        lines.append(f"\n  Domain: {domain} (n={metrics.sample_count})")

        if metrics.hr_mae is not None:
            lines.append(f"    HR MAE: {metrics.hr_mae:.2f} BPM")
        else:
            lines.append("    HR MAE: N/A")

        if metrics.bvp_mae is not None:
            lines.append(f"    BVP MAE: {metrics.bvp_mae:.4f}")
        else:
            lines.append("    BVP MAE: N/A")

        if metrics.sqi_mae is not None:
            lines.append(f"    SQI MAE: {metrics.sqi_mae:.4f}")
        else:
            lines.append("    SQI MAE: N/A")

    return "\n".join(lines)


def compute_worst_case_domain(
    per_domain: Dict[str, DomainMetrics],
    metric: str = "hr_mae"
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Identify worst-performing domain.

    Args:
        per_domain: Per-domain metrics
        metric: Which metric to use

    Returns:
        Tuple of (worst_domain, worst_value, best_value)
    """
    values = {}

    for domain, metrics in per_domain.items():
        if metric == "hr_mae" and metrics.hr_mae is not None:
            values[domain] = metrics.hr_mae
        elif metric == "hr_rmse" and metrics.hr_rmse is not None:
            values[domain] = metrics.hr_rmse
        elif metric == "bvp_mae" and metrics.bvp_mae is not None:
            values[domain] = metrics.bvp_mae
        elif metric == "sqi_mae" and metrics.sqi_mae is not None:
            values[domain] = metrics.sqi_mae

    if not values:
        return None, None, None

    worst_domain = max(values, key=values.get)
    best_domain = min(values, key=values.get)

    return worst_domain, values[worst_domain], values[best_domain]


def summarize_domain_shift(
    train_domains: Dict[str, int],
    test_domains: Dict[str, int]
) -> Dict[str, Any]:
    """
    Summarize domain shift between train and test.

    Args:
        train_domains: Domain distribution in training set
        test_domains: Domain distribution in test set

    Returns:
        Dictionary with shift summary
    """
    all_domains = set(train_domains.keys()) | set(test_domains.keys())

    summary = {
        "train_domains": list(train_domains.keys()),
        "test_domains": list(test_domains.keys()),
        "shared_domains": list(set(train_domains.keys()) & set(test_domains.keys())),
        "unseen_domains": list(set(test_domains.keys()) - set(train_domains.keys())),
        "train_sample_counts": train_domains,
        "test_sample_counts": test_domains,
    }

    # Compute shift indicators
    train_total = sum(train_domains.values())
    test_total = sum(test_domains.values())

    # KL divergence approximation (simplified)
    kl_approx = 0.0
    for domain in all_domains:
        p_train = train_domains.get(domain, 0) / max(train_total, 1)
        p_test = test_domains.get(domain, 0) / max(test_total, 1)
        if p_train > 0 and p_test > 0:
            kl_approx += p_test * np.log(p_test / p_train)

    summary["kl_divergence_approx"] = float(kl_approx)
    summary["domain_shift_detected"] = len(summary["unseen_domains"]) > 0 or kl_approx > 0.1

    return summary
