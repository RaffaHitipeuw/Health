"""
P10 Real-World Stress Lab - Experiment Runner

Reproducible experiment runner for stress condition evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import hashlib
from datetime import datetime

from .base import (
    ExperimentRecord,
    SessionRecord,
    WindowRecord,
    ExperimentConfig,
    ConditionType,
    QualityLevel,
    StressCondition,
    generate_experiment_id,
    set_seed,
)
from .evaluation import StressLabEvaluator
from .metrics import ExperimentMetrics


class StressLabRunner:
    """
    Runner for reproducible stress condition experiments.

    Handles:
        - Experiment configuration
        - Data preparation
        - Model execution
        - Evaluation
        - Report generation
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        hr_error_threshold: float = 10.0,
    ):
        """
        Args:
            config: Experiment configuration
            hr_error_threshold: Threshold for reliable HR (BPM)
        """
        self.config = config or ExperimentConfig()
        self.hr_error_threshold = hr_error_threshold
        self.evaluator = StressLabEvaluator(self.config, hr_error_threshold)

        # State
        self.experiment: Optional[ExperimentRecord] = None
        self.metrics: Optional[ExperimentMetrics] = None

    def create_experiment(
        self,
        name: str,
        description: str = "",
        experiment_id: Optional[str] = None,
    ) -> ExperimentRecord:
        """
        Create a new experiment record.

        Args:
            name: Experiment name
            description: Experiment description
            experiment_id: Optional explicit ID (generated if not provided)

        Returns:
            ExperimentRecord
        """
        if experiment_id is None:
            experiment_id = generate_experiment_id(name, self.config.seed)

        self.experiment = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            description=description,
            config=self.config,
        )

        return self.experiment

    def add_session(
        self,
        session_id: str,
        subject_id: Optional[str] = None,
        dataset: Optional[str] = None,
        device: Optional[str] = None,
        environment: Optional[str] = None,
        **metadata,
    ) -> SessionRecord:
        """
        Add a session to the current experiment.

        Args:
            session_id: Unique session identifier
            subject_id: Subject identifier
            dataset: Source dataset name
            device: Recording device
            environment: Recording environment
            **metadata: Additional metadata

        Returns:
            SessionRecord
        """
        if self.experiment is None:
            raise ValueError("No experiment created. Call create_experiment() first.")

        session = SessionRecord(
            session_id=session_id,
            experiment_id=self.experiment.experiment_id,
            subject_id=subject_id,
            dataset=dataset,
            device=device,
            environment=environment,
            metadata=metadata,
        )

        self.experiment.sessions.append(session)
        return session

    def add_window(
        self,
        session_id: str,
        window_id: str,
        start_time: float,
        end_time: float,
        features: Optional[torch.Tensor] = None,
        rgb_mean: Optional[np.ndarray] = None,
        signal_quality: Optional[float] = None,
        hr_estimate: Optional[float] = None,
        hr_confidence: Optional[float] = None,
        hr_reference: Optional[float] = None,
        bvp_signal: Optional[np.ndarray] = None,
        bvp_reference: Optional[np.ndarray] = None,
        confidence_estimate: Optional[float] = None,
        is_valid: bool = True,
        failure_reason: Optional[str] = None,
    ) -> WindowRecord:
        """
        Add a window to a session.

        Args:
            session_id: Parent session ID
            window_id: Unique window ID
            start_time: Window start time (seconds)
            end_time: Window end time (seconds)
            features: Input features tensor (T, F)
            rgb_mean: Mean RGB values
            signal_quality: SQI estimate (0-1)
            hr_estimate: Heart rate estimate (BPM)
            hr_confidence: HR confidence (0-1)
            hr_reference: Reference HR (BPM)
            bvp_signal: BVP waveform estimate
            bvp_reference: Reference BVP
            confidence_estimate: Overall confidence (0-1)
            is_valid: Whether window is valid
            failure_reason: Reason for invalidity

        Returns:
            WindowRecord
        """
        if self.experiment is None:
            raise ValueError("No experiment created. Call create_experiment() first.")

        # Find session
        session = None
        for s in self.experiment.sessions:
            if s.session_id == session_id:
                session = s
                break

        if session is None:
            raise ValueError(f"Session {session_id} not found. Add session first.")

        window = WindowRecord(
            window_id=window_id,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            features=features,
            rgb_mean=rgb_mean,
            signal_quality=signal_quality,
            hr_estimate=hr_estimate,
            hr_confidence=hr_confidence,
            hr_reference=hr_reference,
            bvp_signal=bvp_signal,
            bvp_reference=bvp_reference,
            confidence_estimate=confidence_estimate,
            is_valid=is_valid,
            failure_reason=failure_reason,
        )

        session.windows.append(window)
        return window

    def add_synthetic_windows(
        self,
        session_id: str,
        num_windows: int,
        duration: float = 10.0,
        seed: int = 42,
        hr_range: Tuple[float, float] = (60.0, 100.0),
        sqi_range: Tuple[float, float] = (0.5, 1.0),
        hr_error: float = 5.0,
    ) -> List[WindowRecord]:
        """
        Add synthetic windows for testing.

        Args:
            session_id: Parent session ID
            num_windows: Number of windows to generate
            duration: Window duration (seconds)
            seed: Random seed
            hr_range: HR range (min, max) BPM
            sqi_range: SQI range (min, max)
            hr_error: HR estimation error (BPM)

        Returns:
            List of WindowRecords
        """
        np.random.seed(seed)
        windows = []

        for i in range(num_windows):
            hr_true = np.random.uniform(*hr_range)
            sqi = np.random.uniform(*sqi_range)
            hr_est = hr_true + np.random.normal(0, hr_error)

            rgb = np.random.uniform(80, 180, size=3)

            window = self.add_window(
                session_id=session_id,
                window_id=f"window_{i}",
                start_time=i * duration,
                end_time=(i + 1) * duration,
                rgb_mean=rgb,
                signal_quality=sqi,
                hr_estimate=hr_est,
                hr_confidence=sqi,
                hr_reference=hr_true,
                confidence_estimate=sqi,
            )
            windows.append(window)

        return windows

    def run(
        self,
        apply_conditions: bool = True,
    ) -> Tuple[ExperimentRecord, ExperimentMetrics]:
        """
        Run evaluation on current experiment.

        Args:
            apply_conditions: Whether to detect conditions from data

        Returns:
            Tuple of (experiment, metrics)
        """
        if self.experiment is None:
            raise ValueError("No experiment created. Call create_experiment() first.")

        set_seed(self.config.seed)

        self.experiment, self.metrics = self.evaluator.evaluate_experiment(
            self.experiment,
            apply_conditions=apply_conditions,
        )

        return self.experiment, self.metrics

    def get_results(self) -> Tuple[Optional[ExperimentRecord], Optional[ExperimentMetrics]]:
        """Get current results."""
        return self.experiment, self.metrics


def run_stress_experiment(
    name: str,
    sessions_data: List[Dict],
    config: Optional[ExperimentConfig] = None,
    description: str = "",
    hr_error_threshold: float = 10.0,
    apply_conditions: bool = True,
) -> Tuple[ExperimentRecord, ExperimentMetrics]:
    """
    Convenience function to run a complete stress experiment.

    Args:
        name: Experiment name
        sessions_data: List of session data dicts with windows
        config: Experiment configuration
        description: Experiment description
        hr_error_threshold: Threshold for reliable HR (BPM)
        apply_conditions: Whether to detect conditions

    Returns:
        Tuple of (experiment, metrics)

    Example sessions_data:
        [
            {
                "session_id": "s1",
                "subject_id": "subj1",
                "dataset": "dataset_a",
                "windows": [
                    {"window_id": "w1", "start_time": 0, "end_time": 10, "hr_estimate": 72, "hr_reference": 70, ...},
                    ...
                ]
            },
            ...
        ]
    """
    runner = StressLabRunner(config, hr_error_threshold)

    # Create experiment
    exp = runner.create_experiment(name, description)

    # Add sessions and windows
    for session_data in sessions_data:
        session = runner.add_session(
            session_id=session_data["session_id"],
            subject_id=session_data.get("subject_id"),
            dataset=session_data.get("dataset"),
            device=session_data.get("device"),
            environment=session_data.get("environment"),
        )

        for window_data in session_data.get("windows", []):
            runner.add_window(
                session_id=session_data["session_id"],
                **window_data,
            )

    # Run evaluation
    exp, metrics = runner.run(apply_conditions=apply_conditions)

    return exp, metrics
