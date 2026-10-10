"""
P10 Real-World Stress Lab - Base Structures

Metadata structures for experiment records, session data, and condition classification.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib


class ConditionType(Enum):
    """Types of stress conditions that can affect physiological sensing."""
    LIGHTING_NORMAL = "lighting_normal"
    LIGHTING_LOW = "lighting_low"
    LIGHTING_VARIABLE = "lighting_variable"
    MOTION_NONE = "motion_none"
    MOTION_MILD = "motion_mild"
    MOTION_MODERATE = "motion_moderate"
    MOTION_SEVERE = "motion_severe"
    ROI_STABLE = "roi_stable"
    ROI_PARTIAL_LOSS = "roi_partial_loss"
    ROI_SEVERE_LOSS = "roi_severe_loss"
    SQI_HIGH = "sqi_high"
    SQI_MEDIUM = "sqi_medium"
    SQI_LOW = "sqi_low"
    SQI_FAILED = "sqi_failed"
    SIGNAL_MISSING = "signal_missing"
    SIGNAL_DEGRADED = "signal_degraded"
    UNKNOWN = "unknown"


class QualityLevel(Enum):
    """Overall quality assessment levels."""
    UNKNOWN = "unknown"       # Quality not assessed
    EXCELLENT = "excellent"  # All signals optimal
    GOOD = "good"           # Minor degradations, reliable estimates
    FAIR = "fair"            # Some degradation, estimates may vary
    POOR = "poor"           # Significant degradation, estimates unreliable
    FAILED = "failed"       # No valid estimates


@dataclass
class StressCondition:
    """
    Represents a detected stress condition in a recording window.

    Attributes:
        condition_type: Type of condition detected
        severity: Severity level 0.0-1.0
        evidence: Evidence values supporting the detection
        is_actionable: Whether this condition can be compensated for
    """
    condition_type: ConditionType
    severity: float  # 0.0 to 1.0
    evidence: Dict[str, float] = field(default_factory=dict)
    is_actionable: bool = True
    description: str = ""

    def __post_init__(self):
        if self.severity < 0:
            self.severity = 0.0
        elif self.severity > 1:
            self.severity = 1.0

    def is_detected(self, threshold: float = 0.5) -> bool:
        """Check if condition is detected above threshold."""
        return self.severity >= threshold


@dataclass
class WindowRecord:
    """
    Record for a single analysis window.

    Attributes:
        window_id: Unique window identifier
        session_id: Parent session identifier
        start_time: Window start time (seconds)
        end_time: Window end time (seconds)
        duration: Window duration (seconds)

        # Signal data
        features: Input features tensor (T, F)
        rgb_mean: Mean RGB values over window
        signal_quality: SQI estimate

        # Physiological estimates
        hr_estimate: Heart rate estimate (BPM)
        hr_confidence: HR estimate confidence
        bvp_signal: BVP waveform estimate
        confidence_estimate: Overall confidence

        # Reference (when available)
        hr_reference: Reference HR (BPM)
        bvp_reference: Reference BVP

        # Conditions detected
        conditions: List of detected stress conditions
        quality_level: Overall quality assessment

        # Validity flags
        is_valid: Whether window has valid estimates
        failure_reason: Reason for invalidity if any
    """
    window_id: str
    session_id: str
    start_time: float
    end_time: float
    duration: float

    # Signal data
    features: Optional[torch.Tensor] = None
    rgb_mean: Optional[np.ndarray] = None
    signal_quality: Optional[float] = None

    # Physiological estimates
    hr_estimate: Optional[float] = None
    hr_confidence: Optional[float] = None
    bvp_signal: Optional[np.ndarray] = None
    confidence_estimate: Optional[float] = None

    # Reference (when available)
    hr_reference: Optional[float] = None
    bvp_reference: Optional[np.ndarray] = None

    # Conditions
    conditions: List[StressCondition] = field(default_factory=list)
    quality_level: QualityLevel = QualityLevel.UNKNOWN

    # Validity
    is_valid: bool = True
    failure_reason: Optional[str] = None

    def get_condition_by_type(self, condition_type: ConditionType) -> Optional[StressCondition]:
        """Get condition of specific type if present."""
        for cond in self.conditions:
            if cond.condition_type == condition_type:
                return cond
        return None

    def has_condition(self, condition_type: ConditionType, threshold: float = 0.5) -> bool:
        """Check if specific condition is detected."""
        cond = self.get_condition_by_type(condition_type)
        return cond is not None and cond.is_detected(threshold)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "window_id": self.window_id,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "rgb_mean": self.rgb_mean.tolist() if self.rgb_mean is not None else None,
            "signal_quality": self.signal_quality,
            "hr_estimate": self.hr_estimate,
            "hr_confidence": self.hr_confidence,
            "confidence_estimate": self.confidence_estimate,
            "hr_reference": self.hr_reference,
            "is_valid": self.is_valid,
            "failure_reason": self.failure_reason,
            "quality_level": self.quality_level.value if isinstance(self.quality_level, QualityLevel) else self.quality_level,
            "conditions": [
                {
                    "type": c.condition_type.value,
                    "severity": c.severity,
                    "evidence": c.evidence,
                    "is_actionable": c.is_actionable,
                }
                for c in self.conditions
            ],
        }


@dataclass
class SessionRecord:
    """
    Record for a recording session.

    Attributes:
        session_id: Unique session identifier
        experiment_id: Parent experiment identifier
        subject_id: Subject identifier (if known)
        dataset: Source dataset name (if known)
        device: Recording device (if known)
        environment: Recording environment (if known)
        start_time: Session start time
        end_time: Session end time
        duration: Total recording duration

        # Windows
        windows: List of window records

        # Metadata
        metadata: Additional session metadata
    """
    session_id: str
    experiment_id: str
    subject_id: Optional[str] = None
    dataset: Optional[str] = None
    device: Optional[str] = None
    environment: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None

    # Windows
    windows: List[WindowRecord] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.duration is None and self.start_time is not None and self.end_time is not None:
            self.duration = self.end_time - self.start_time

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    @property
    def num_valid_windows(self) -> int:
        return sum(1 for w in self.windows if w.is_valid)

    def get_windows_by_condition(self, condition_type: ConditionType, threshold: float = 0.5) -> List[WindowRecord]:
        """Get all windows with specific condition detected."""
        return [w for w in self.windows if w.has_condition(condition_type, threshold)]

    def get_windows_by_quality(self, quality_level: QualityLevel) -> List[WindowRecord]:
        """Get all windows with specific quality level."""
        return [w for w in self.windows if w.quality_level == quality_level]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "experiment_id": self.experiment_id,
            "subject_id": self.subject_id,
            "dataset": self.dataset,
            "device": self.device,
            "environment": self.environment,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "num_windows": self.num_windows,
            "num_valid_windows": self.num_valid_windows,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentRecord:
    """
    Record for a complete stress experiment.

    Attributes:
        experiment_id: Unique experiment identifier
        name: Experiment name
        description: Experiment description
        created_at: Experiment creation timestamp
        config: Experiment configuration

        # Sessions
        sessions: List of session records

        # Source
        source_pipeline: Source pipeline version
        p9_split_type: P9 split type if applicable
        domain_key: Domain key for P9 if applicable
    """
    experiment_id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    config: Optional["ExperimentConfig"] = None

    # Sessions
    sessions: List[SessionRecord] = field(default_factory=list)

    # Source tracking
    source_pipeline: Optional[str] = None
    p9_split_type: Optional[str] = None
    domain_key: Optional[str] = None

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def num_windows(self) -> int:
        return sum(s.num_windows for s in self.sessions)

    @property
    def num_valid_windows(self) -> int:
        return sum(s.num_valid_windows for s in self.sessions)

    def get_all_windows(self) -> List[WindowRecord]:
        """Get all windows from all sessions."""
        windows = []
        for session in self.sessions:
            windows.extend(session.windows)
        return windows

    def get_sessions_by_subject(self, subject_id: str) -> List[SessionRecord]:
        """Get all sessions for a subject."""
        return [s for s in self.sessions if s.subject_id == subject_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "num_sessions": self.num_sessions,
            "num_windows": self.num_windows,
            "num_valid_windows": self.num_valid_windows,
            "source_pipeline": self.source_pipeline,
            "p9_split_type": self.p9_split_type,
            "domain_key": self.domain_key,
        }


@dataclass
class ExperimentConfig:
    """Configuration for stress lab experiment."""
    # Window settings
    window_duration: float = 10.0  # seconds
    window_overlap: float = 5.0    # seconds
    min_window_duration: float = 5.0  # minimum valid window

    # Quality thresholds
    sqi_threshold_high: float = 0.8
    sqi_threshold_medium: float = 0.5
    sqi_threshold_low: float = 0.3

    confidence_threshold: float = 0.5
    hr_confidence_threshold: float = 0.4

    # Condition detection thresholds
    lighting_low_threshold: float = 50.0   # RGB mean
    lighting_var_threshold: float = 30.0  # RGB variance
    motion_threshold_mild: float = 0.01    # motion magnitude
    motion_threshold_moderate: float = 0.05
    motion_threshold_severe: float = 0.1
    roi_loss_threshold_mild: float = 0.1
    roi_loss_threshold_severe: float = 0.3

    # Evaluation settings
    compute_per_condition: bool = True
    compute_per_session: bool = True
    compute_per_subject: bool = True

    # Output
    save_individual_windows: bool = False
    output_format: str = "json"  # "json" or "markdown"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.min_window_duration > self.window_duration:
            self.min_window_duration = self.window_duration / 2


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


def generate_experiment_id(name: str, seed: int = 42) -> str:
    """Generate deterministic experiment ID from name."""
    content = f"{name}|{seed}"
    return hashlib.md5(content.encode()).hexdigest()[:12]
