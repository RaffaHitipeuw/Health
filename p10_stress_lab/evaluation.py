"""
P10 Real-World Stress Lab - Evaluation Pipeline

Evaluation of physiological sensing under stress conditions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import json

from .base import (
    ExperimentRecord,
    SessionRecord,
    WindowRecord,
    ExperimentConfig,
    ConditionType,
    QualityLevel,
    set_seed,
)
from .conditions import classify_window_conditions, classify_window_quality
from .metrics import (
    ExperimentMetrics,
    SessionMetrics,
    aggregate_session_metrics,
    aggregate_experiment_metrics,
)


class StressLabEvaluator:
    """
    Evaluator for stress condition experiments.

    Provides evaluation of physiological sensing system behavior
    under various stress conditions.
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        hr_error_threshold: float = 10.0,
    ):
        """
        Args:
            config: Experiment configuration
            hr_error_threshold: Threshold for reliable HR estimate (BPM)
        """
        self.config = config or ExperimentConfig()
        self.hr_error_threshold = hr_error_threshold

    def evaluate_session(
        self,
        session: SessionRecord,
        apply_conditions: bool = True,
    ) -> Tuple[SessionRecord, SessionMetrics]:
        """
        Evaluate a single session.

        Args:
            session: SessionRecord to evaluate
            apply_conditions: Whether to detect conditions from data

        Returns:
            Tuple of (updated_session, session_metrics)
        """
        # Classify windows
        if apply_conditions:
            for window in session.windows:
                if window.is_valid:
                    # Classify conditions from window data
                    conditions = classify_window_conditions(window, self.config)
                    window.conditions = conditions

                    # Set quality level
                    sqi_type, sqi_score = classify_window_quality(
                        window.signal_quality,
                        window.confidence_estimate,
                        window.hr_confidence,
                        self.config,
                    )
                    window.quality_level = self._sqi_to_quality(sqi_type)

        # Compute metrics
        metrics = aggregate_session_metrics(session, self.hr_error_threshold)

        return session, metrics

    def evaluate_experiment(
        self,
        experiment: ExperimentRecord,
        apply_conditions: bool = True,
    ) -> Tuple[ExperimentRecord, ExperimentMetrics]:
        """
        Evaluate complete experiment.

        Args:
            experiment: ExperimentRecord with sessions
            apply_conditions: Whether to detect conditions from data

        Returns:
            Tuple of (updated_experiment, experiment_metrics)
        """
        set_seed(self.config.seed)

        # Evaluate each session
        for session in experiment.sessions:
            session, _ = self.evaluate_session(session, apply_conditions)

        # Compute experiment-level metrics
        metrics = aggregate_experiment_metrics(experiment, self.hr_error_threshold)

        return experiment, metrics

    def _sqi_to_quality(self, sqi_type: ConditionType) -> QualityLevel:
        """Map SQI condition to quality level."""
        if sqi_type == ConditionType.SQI_HIGH:
            return QualityLevel.EXCELLENT
        elif sqi_type == ConditionType.SQI_MEDIUM:
            return QualityLevel.GOOD
        elif sqi_type == ConditionType.SQI_LOW:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR

    def compare_conditions(
        self,
        experiment: ExperimentRecord,
        condition_a: ConditionType,
        condition_b: ConditionType,
    ) -> Dict[str, Any]:
        """
        Compare metrics between two conditions.

        Args:
            experiment: Experiment with evaluated sessions
            condition_a: First condition type
            condition_b: Second condition type

        Returns:
            Dict with comparison results
        """
        windows = experiment.get_all_windows()

        # Get windows for each condition
        windows_a = [w for w in windows if w.has_condition(condition_a)]
        windows_b = [w for w in windows if w.has_condition(condition_b)]

        if not windows_a or not windows_b:
            return {"error": "Insufficient windows for comparison"}

        # Compute metrics for each
        from .metrics import aggregate_condition_metrics

        metrics_a = aggregate_condition_metrics(windows_a, condition_a, self.hr_error_threshold)
        metrics_b = aggregate_condition_metrics(windows_b, condition_b, self.hr_error_threshold)

        # Compare
        comparison = {
            "condition_a": condition_a.value,
            "condition_b": condition_b.value,
            "count_a": metrics_a.total_windows,
            "count_b": metrics_b.total_windows,
            "hr_mae_a": metrics_a.hr_mae,
            "hr_mae_b": metrics_b.hr_mae,
            "hr_mae_diff": (
                (metrics_a.hr_mae or 0) - (metrics_b.hr_mae or 0)
                if metrics_a.hr_mae is not None and metrics_b.hr_mae is not None
                else None
            ),
            "reliable_rate_a": metrics_a.reliable_rate,
            "reliable_rate_b": metrics_b.reliable_rate,
            "reliable_rate_diff": metrics_a.reliable_rate - metrics_b.reliable_rate,
        }

        return comparison


def evaluate_session(
    session: SessionRecord,
    config: Optional[ExperimentConfig] = None,
    hr_error_threshold: float = 10.0,
) -> Tuple[SessionRecord, SessionMetrics]:
    """
    Convenience function to evaluate a session.

    Args:
        session: SessionRecord to evaluate
        config: Experiment configuration
        hr_error_threshold: Threshold for reliable HR

    Returns:
        Tuple of (updated_session, session_metrics)
    """
    evaluator = StressLabEvaluator(config, hr_error_threshold)
    return evaluator.evaluate_session(session)


def evaluate_experiment(
    experiment: ExperimentRecord,
    config: Optional[ExperimentConfig] = None,
    hr_error_threshold: float = 10.0,
) -> Tuple[ExperimentRecord, ExperimentMetrics]:
    """
    Convenience function to evaluate an experiment.

    Args:
        experiment: ExperimentRecord to evaluate
        config: Experiment configuration
        hr_error_threshold: Threshold for reliable HR

    Returns:
        Tuple of (updated_experiment, experiment_metrics)
    """
    evaluator = StressLabEvaluator(config, hr_error_threshold)
    return evaluator.evaluate_experiment(experiment)


def load_experiment_from_json(filepath: str) -> ExperimentRecord:
    """
    Load experiment from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        ExperimentRecord
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return _dict_to_experiment(data)


def save_experiment_to_json(experiment: ExperimentRecord, filepath: str) -> None:
    """
    Save experiment to JSON file.

    Args:
        experiment: ExperimentRecord to save
        filepath: Output path
    """
    data = _experiment_to_dict(experiment)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def _experiment_to_dict(exp: ExperimentRecord) -> Dict:
    """Convert ExperimentRecord to dict."""
    return {
        "experiment_id": exp.experiment_id,
        "name": exp.name,
        "description": exp.description,
        "created_at": exp.created_at.isoformat() if hasattr(exp.created_at, 'isoformat') else str(exp.created_at),
        "source_pipeline": exp.source_pipeline,
        "p9_split_type": exp.p9_split_type,
        "domain_key": exp.domain_key,
        "sessions": [_session_to_dict(s) for s in exp.sessions],
    }


def _session_to_dict(session: SessionRecord) -> Dict:
    """Convert SessionRecord to dict."""
    return {
        "session_id": session.session_id,
        "experiment_id": session.experiment_id,
        "subject_id": session.subject_id,
        "dataset": session.dataset,
        "device": session.device,
        "environment": session.environment,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "duration": session.duration,
        "windows": [_window_to_dict(w) for w in session.windows],
        "metadata": session.metadata,
    }


def _window_to_dict(window: WindowRecord) -> Dict:
    """Convert WindowRecord to dict."""
    return window.to_dict()


def _dict_to_experiment(data: Dict) -> ExperimentRecord:
    """Convert dict to ExperimentRecord."""
    from datetime import datetime

    exp = ExperimentRecord(
        experiment_id=data["experiment_id"],
        name=data["name"],
        description=data.get("description", ""),
        source_pipeline=data.get("source_pipeline"),
        p9_split_type=data.get("p9_split_type"),
        domain_key=data.get("domain_key"),
    )

    if "created_at" in data:
        exp.created_at = datetime.fromisoformat(data["created_at"])

    if "sessions" in data:
        exp.sessions = [_dict_to_session(s) for s in data["sessions"]]

    return exp


def _dict_to_session(data: Dict) -> SessionRecord:
    """Convert dict to SessionRecord."""
    session = SessionRecord(
        session_id=data["session_id"],
        experiment_id=data["experiment_id"],
        subject_id=data.get("subject_id"),
        dataset=data.get("dataset"),
        device=data.get("device"),
        environment=data.get("environment"),
        start_time=data.get("start_time"),
        end_time=data.get("end_time"),
        duration=data.get("duration"),
        metadata=data.get("metadata", {}),
    )

    if "windows" in data:
        session.windows = [_dict_to_window(w) for w in data["windows"]]

    return session


def _dict_to_window(data: Dict) -> WindowRecord:
    """Convert dict to WindowRecord."""
    window = WindowRecord(
        window_id=data["window_id"],
        session_id=data["session_id"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        duration=data["duration"],
        rgb_mean=np.array(data["rgb_mean"]) if data.get("rgb_mean") else None,
        signal_quality=data.get("signal_quality"),
        hr_estimate=data.get("hr_estimate"),
        hr_confidence=data.get("hr_confidence"),
        confidence_estimate=data.get("confidence_estimate"),
        hr_reference=data.get("hr_reference"),
        is_valid=data.get("is_valid", True),
        failure_reason=data.get("failure_reason"),
    )

    if "quality_level" in data:
        window.quality_level = QualityLevel(data["quality_level"])

    return window
