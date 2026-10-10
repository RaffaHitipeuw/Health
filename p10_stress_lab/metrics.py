"""
P10 Real-World Stress Lab - Metrics

Metrics for evaluating physiological sensing under stress conditions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .base import (
    WindowRecord,
    SessionRecord,
    ConditionType,
    QualityLevel,
)


# =============================================================================
# Metric Structures
# =============================================================================

@dataclass
class WindowMetrics:
    """Metrics for a single window."""
    window_id: str
    session_id: str
    is_valid: bool

    # HR metrics (BPM)
    hr_error: Optional[float] = None  # |estimate - reference|
    hr_absolute_error: Optional[float] = None
    hr_bias: Optional[float] = None   # estimate - reference

    # BVP metrics
    bvp_corr: Optional[float] = None  # Pearson correlation
    bvp_mae: Optional[float] = None

    # Quality indicators
    sqi: Optional[float] = None
    confidence: Optional[float] = None
    hr_confidence: Optional[float] = None

    # Condition flags
    has_lighting_issue: bool = False
    has_motion_issue: bool = False
    has_roi_issue: bool = False
    has_quality_issue: bool = False

    def is_reliable(self, max_hr_error: float = 10.0) -> bool:
        """Check if window produces reliable estimates."""
        return (
            self.is_valid and
            self.hr_error is not None and
            self.hr_error <= max_hr_error
        )


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""
    session_id: str
    experiment_id: str

    # Counts
    total_windows: int = 0
    valid_windows: int = 0
    failed_windows: int = 0

    # Valid window rate
    valid_rate: float = 0.0

    # HR aggregate metrics (over valid windows with reference)
    hr_mae: Optional[float] = None
    hr_rmse: Optional[float] = None
    hr_bias: Optional[float] = None
    hr_std: Optional[float] = None

    # Quality aggregates
    mean_sqi: Optional[float] = None
    mean_confidence: Optional[float] = None
    mean_hr_confidence: Optional[float] = None

    # Condition breakdown
    condition_counts: Dict[str, int] = field(default_factory=dict)
    quality_counts: Dict[str, int] = field(default_factory=dict)

    # Window-level metrics
    window_metrics: List[WindowMetrics] = field(default_factory=list)


@dataclass
class ConditionMetrics:
    """Metrics aggregated by condition type."""
    condition_type: str

    # Counts
    total_windows: int = 0
    valid_windows: int = 0

    # HR metrics
    hr_mae: Optional[float] = None
    hr_rmse: Optional[float] = None
    hr_bias: Optional[float] = None
    hr_std: Optional[float] = None

    # Quality aggregates
    mean_sqi: Optional[float] = None
    mean_confidence: Optional[float] = None

    # Reliability
    reliable_rate: float = 0.0  # Fraction with HR error < threshold
    valid_rate: float = 0.0   # Fraction of valid windows


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for complete experiment."""
    experiment_id: str
    name: str

    # Overall counts
    total_sessions: int = 0
    total_windows: int = 0
    valid_windows: int = 0
    failed_windows: int = 0

    # Overall valid rate
    overall_valid_rate: float = 0.0

    # HR aggregate metrics
    hr_mae: Optional[float] = None
    hr_rmse: Optional[float] = None
    hr_bias: Optional[float] = None
    hr_std: Optional[float] = None

    # Per-session metrics
    session_metrics: Dict[str, SessionMetrics] = field(default_factory=dict)

    # Per-condition metrics
    condition_metrics: Dict[str, ConditionMetrics] = field(default_factory=dict)

    # Per-quality-level metrics
    quality_metrics: Dict[str, ConditionMetrics] = field(default_factory=dict)

    # Subject-level metrics (if subject_ids available)
    subject_metrics: Dict[str, SessionMetrics] = field(default_factory=dict)


# =============================================================================
# Window-Level Metric Computation
# =============================================================================

def compute_window_metrics(
    window: WindowRecord,
    hr_error_threshold: float = 10.0,
) -> WindowMetrics:
    """
    Compute metrics for a single window.

    Args:
        window: WindowRecord with estimates and references
        hr_error_threshold: Threshold for reliable HR (BPM)

    Returns:
        WindowMetrics with computed values
    """
    metrics = WindowMetrics(
        window_id=window.window_id,
        session_id=window.session_id,
        is_valid=window.is_valid,
    )

    if not window.is_valid:
        return metrics

    # HR metrics
    if window.hr_estimate is not None and window.hr_reference is not None:
        error = window.hr_estimate - window.hr_reference
        metrics.hr_error = abs(error)
        metrics.hr_absolute_error = abs(error)
        metrics.hr_bias = error

    # BVP correlation
    if window.bvp_signal is not None and window.bvp_reference is not None:
        try:
            corr = np.corrcoef(window.bvp_signal, window.bvp_reference)[0, 1]
            metrics.bvp_corr = float(corr)
        except Exception:
            pass

        metrics.bvp_mae = float(np.mean(np.abs(window.bvp_signal - window.bvp_reference)))

    # Quality indicators
    metrics.sqi = window.signal_quality
    metrics.confidence = window.confidence_estimate
    metrics.hr_confidence = window.hr_confidence

    # Condition flags
    metrics.has_lighting_issue = any(
        c.condition_type in {
            ConditionType.LIGHTING_LOW,
            ConditionType.LIGHTING_VARIABLE,
        }
        for c in window.conditions
    )
    metrics.has_motion_issue = any(
        c.condition_type in {
            ConditionType.MOTION_MODERATE,
            ConditionType.MOTION_SEVERE,
        }
        for c in window.conditions
    )
    metrics.has_roi_issue = any(
        c.condition_type in {
            ConditionType.ROI_PARTIAL_LOSS,
            ConditionType.ROI_SEVERE_LOSS,
        }
        for c in window.conditions
    )
    metrics.has_quality_issue = any(
        c.condition_type in {
            ConditionType.SQI_LOW,
            ConditionType.SQI_FAILED,
            ConditionType.SIGNAL_DEGRADED,
        }
        for c in window.conditions
    )

    return metrics


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_session_metrics(
    session: SessionRecord,
    hr_error_threshold: float = 10.0,
) -> SessionMetrics:
    """
    Aggregate metrics for a session.

    Args:
        session: SessionRecord with windows
        hr_error_threshold: Threshold for reliable HR

    Returns:
        SessionMetrics with aggregated values
    """
    metrics = SessionMetrics(
        session_id=session.session_id,
        experiment_id=session.experiment_id,
        total_windows=session.num_windows,
    )

    # Count valid/failed
    valid_windows = [w for w in session.windows if w.is_valid]
    metrics.valid_windows = len(valid_windows)
    metrics.failed_windows = session.num_windows - metrics.valid_windows
    metrics.valid_rate = metrics.valid_windows / max(1, session.num_windows)

    # Compute window-level metrics
    window_metrics = []
    hr_errors = []
    hr_biases = []
    sqis = []
    confidences = []
    hr_confidences = []

    condition_counts: Dict[str, int] = {}
    quality_counts: Dict[str, int] = {}

    for window in session.windows:
        wm = compute_window_metrics(window, hr_error_threshold)
        window_metrics.append(wm)

        if wm.is_valid:
            if wm.hr_error is not None:
                hr_errors.append(wm.hr_error)
            if wm.hr_bias is not None:
                hr_biases.append(wm.hr_bias)
            if wm.sqi is not None:
                sqis.append(wm.sqi)
            if wm.confidence is not None:
                confidences.append(wm.confidence)
            if wm.hr_confidence is not None:
                hr_confidences.append(wm.hr_confidence)

        # Count conditions
        for cond in window.conditions:
            cond_name = cond.condition_type.value if isinstance(cond.condition_type, ConditionType) else str(cond.condition_type)
            condition_counts[cond_name] = condition_counts.get(cond_name, 0) + 1

        # Count quality levels
        ql_name = window.quality_level.value if isinstance(window.quality_level, QualityLevel) else str(window.quality_level)
        quality_counts[ql_name] = quality_counts.get(ql_name, 0) + 1

    metrics.window_metrics = window_metrics
    metrics.condition_counts = condition_counts
    metrics.quality_counts = quality_counts

    # Aggregate HR metrics
    if hr_errors:
        metrics.hr_mae = float(np.mean(hr_errors))
        metrics.hr_rmse = float(np.sqrt(np.mean(np.array(hr_errors) ** 2)))
    if hr_biases:
        metrics.hr_bias = float(np.mean(hr_biases))
        metrics.hr_std = float(np.std(hr_biases))

    # Aggregate quality
    if sqis:
        metrics.mean_sqi = float(np.mean(sqis))
    if confidences:
        metrics.mean_confidence = float(np.mean(confidences))
    if hr_confidences:
        metrics.mean_hr_confidence = float(np.mean(hr_confidences))

    return metrics


def aggregate_condition_metrics(
    windows: List[WindowRecord],
    condition_filter: Optional[ConditionType] = None,
    hr_error_threshold: float = 10.0,
) -> ConditionMetrics:
    """
    Aggregate metrics for windows with specific condition.

    Args:
        windows: List of WindowRecords
        condition_filter: Condition type to filter by
        hr_error_threshold: Threshold for reliable HR

    Returns:
        ConditionMetrics aggregated for condition
    """
    condition_name = condition_filter.value if condition_filter else "all"
    metrics = ConditionMetrics(condition_type=condition_name)

    # Filter windows
    if condition_filter:
        filtered = [w for w in windows if w.has_condition(condition_filter)]
    else:
        filtered = windows

    metrics.total_windows = len(filtered)
    valid = [w for w in filtered if w.is_valid]
    metrics.valid_windows = len(valid)
    metrics.valid_rate = metrics.valid_windows / max(1, metrics.total_windows)

    if not valid:
        return metrics

    # Compute HR metrics
    hr_errors = []
    hr_biases = []
    sqis = []
    confidences = []
    reliable_count = 0

    for window in valid:
        wm = compute_window_metrics(window, hr_error_threshold)

        if wm.hr_error is not None:
            hr_errors.append(wm.hr_error)
        if wm.hr_bias is not None:
            hr_biases.append(wm.hr_bias)
        if wm.sqi is not None:
            sqis.append(wm.sqi)
        if wm.confidence is not None:
            confidences.append(wm.confidence)
        if wm.is_reliable(hr_error_threshold):
            reliable_count += 1

    metrics.reliable_rate = reliable_count / max(1, len(valid))

    if hr_errors:
        metrics.hr_mae = float(np.mean(hr_errors))
        metrics.hr_rmse = float(np.sqrt(np.mean(np.array(hr_errors) ** 2)))
    if hr_biases:
        metrics.hr_bias = float(np.mean(hr_biases))
        metrics.hr_std = float(np.std(hr_biases))
    if sqis:
        metrics.mean_sqi = float(np.mean(sqis))
    if confidences:
        metrics.mean_confidence = float(np.mean(confidences))

    return metrics


def aggregate_experiment_metrics(
    experiment,
    hr_error_threshold: float = 10.0,
) -> ExperimentMetrics:
    """
    Aggregate metrics for complete experiment.

    Args:
        experiment: ExperimentRecord with sessions
        hr_error_threshold: Threshold for reliable HR

    Returns:
        ExperimentMetrics with all aggregations
    """
    metrics = ExperimentMetrics(
        experiment_id=experiment.experiment_id,
        name=experiment.name,
        total_sessions=experiment.num_sessions,
    )

    all_windows = experiment.get_all_windows()
    metrics.total_windows = len(all_windows)

    valid_windows = [w for w in all_windows if w.is_valid]
    metrics.valid_windows = len(valid_windows)
    metrics.failed_windows = metrics.total_windows - metrics.valid_windows
    metrics.overall_valid_rate = metrics.valid_windows / max(1, metrics.total_windows)

    # Per-session metrics
    for session in experiment.sessions:
        sm = aggregate_session_metrics(session, hr_error_threshold)
        metrics.session_metrics[session.session_id] = sm

        # Subject aggregation
        if session.subject_id:
            if session.subject_id not in metrics.subject_metrics:
                metrics.subject_metrics[session.subject_id] = SessionMetrics(
                    session_id=f"subject_{session.subject_id}",
                    experiment_id=experiment.experiment_id,
                )
            # Merge into subject metrics (simplified)
            subj = metrics.subject_metrics[session.subject_id]
            subj.total_windows += sm.total_windows
            subj.valid_windows += sm.valid_windows
            subj.failed_windows += sm.failed_windows

    # Normalize subject metrics
    for subj in metrics.subject_metrics.values():
        subj.valid_rate = subj.valid_windows / max(1, subj.total_windows)

    # Per-condition metrics
    for cond_type in ConditionType:
        cm = aggregate_condition_metrics(all_windows, cond_type, hr_error_threshold)
        if cm.total_windows > 0:
            metrics.condition_metrics[cond_type.value] = cm

    # Overall condition metrics
    metrics.condition_metrics["all"] = aggregate_condition_metrics(
        all_windows, None, hr_error_threshold
    )

    # Per-quality metrics
    for quality in QualityLevel:
        qm = aggregate_condition_metrics(all_windows, None, hr_error_threshold)
        qm.condition_type = quality.value
        filtered = [w for w in all_windows if w.quality_level == quality]
        qm.total_windows = len(filtered)
        qm.valid_windows = sum(1 for w in filtered if w.is_valid)
        qm.valid_rate = qm.valid_windows / max(1, qm.total_windows)

        if qm.valid_windows > 0:
            metrics.quality_metrics[quality.value] = qm

    # Overall HR metrics - collect actual per-window errors
    overall_hr_errors = []
    overall_hr_biases = []

    for session in experiment.sessions:
        for window in session.windows:
            if window.is_valid and window.hr_estimate is not None and window.hr_reference is not None:
                error = abs(window.hr_estimate - window.hr_reference)
                bias = window.hr_estimate - window.hr_reference
                overall_hr_errors.append(error)
                overall_hr_biases.append(bias)

    if overall_hr_errors:
        metrics.hr_mae = float(np.mean(overall_hr_errors))
        metrics.hr_rmse = float(np.sqrt(np.mean(np.array(overall_hr_errors) ** 2)))
    if overall_hr_biases:
        metrics.hr_bias = float(np.mean(overall_hr_biases))
        metrics.hr_std = float(np.std(overall_hr_biases))

    return metrics
