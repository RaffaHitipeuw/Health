"""
P10 Real-World Stress Lab

Experimental evaluation layer for assessing physiological sensing system behavior
under realistic recording conditions.

This module provides:
    - Experiment and session metadata structures
    - Stress condition definitions and detection
    - Condition-specific evaluation metrics
    - Reproducible experiment runner
    - Report generation

Note: This module provides evaluation infrastructure. Real-world validation
requires actual experiments with diverse recording conditions.
"""

__version__ = "1.0.0"

from .base import (
    ExperimentRecord,
    SessionRecord,
    WindowRecord,
    ConditionType,
    StressCondition,
    ExperimentConfig,
    QualityLevel,
)

from .conditions import (
    detect_lighting_condition,
    detect_motion_level,
    detect_roi_stability,
    classify_window_quality,
    get_supported_conditions,
)

from .metrics import (
    WindowMetrics,
    SessionMetrics,
    ConditionMetrics,
    ExperimentMetrics,
    compute_window_metrics,
    aggregate_session_metrics,
    aggregate_condition_metrics,
    aggregate_experiment_metrics,
)

from .evaluation import (
    StressLabEvaluator,
    evaluate_session,
    evaluate_experiment,
)

from .runner import (
    StressLabRunner,
    run_stress_experiment,
)

from .report import (
    ExperimentReport,
    generate_report,
    save_report,
    load_report,
)

__all__ = [
    "__version__",
    # Base
    "ExperimentRecord",
    "SessionRecord",
    "WindowRecord",
    "ConditionType",
    "StressCondition",
    "ExperimentConfig",
    "QualityLevel",
    # Conditions
    "detect_lighting_condition",
    "detect_motion_level",
    "detect_roi_stability",
    "classify_window_quality",
    "get_supported_conditions",
    # Metrics
    "WindowMetrics",
    "SessionMetrics",
    "ConditionMetrics",
    "ExperimentMetrics",
    "compute_window_metrics",
    "aggregate_session_metrics",
    "aggregate_condition_metrics",
    "aggregate_experiment_metrics",
    # Evaluation
    "StressLabEvaluator",
    "evaluate_session",
    "evaluate_experiment",
    # Runner
    "StressLabRunner",
    "run_stress_experiment",
    # Report
    "ExperimentReport",
    "generate_report",
    "save_report",
    "load_report",
]
