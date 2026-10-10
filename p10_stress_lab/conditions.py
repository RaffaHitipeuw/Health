"""
P10 Real-World Stress Lab - Condition Detection

Detection and classification of stress conditions that affect physiological sensing.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

from .base import (
    ConditionType,
    StressCondition,
    WindowRecord,
    ExperimentConfig,
)


# =============================================================================
# Supported Conditions
# =============================================================================

SUPPORTED_CONDITIONS = {
    ConditionType.LIGHTING_NORMAL,
    ConditionType.LIGHTING_LOW,
    ConditionType.LIGHTING_VARIABLE,
    ConditionType.MOTION_NONE,
    ConditionType.MOTION_MILD,
    ConditionType.MOTION_MODERATE,
    ConditionType.MOTION_SEVERE,
    ConditionType.ROI_STABLE,
    ConditionType.ROI_PARTIAL_LOSS,
    ConditionType.ROI_SEVERE_LOSS,
    ConditionType.SQI_HIGH,
    ConditionType.SQI_MEDIUM,
    ConditionType.SQI_LOW,
    ConditionType.SQI_FAILED,
    ConditionType.SIGNAL_MISSING,
    ConditionType.SIGNAL_DEGRADED,
}


def get_supported_conditions() -> List[ConditionType]:
    """Get list of supported condition types."""
    return list(SUPPORTED_CONDITIONS)


# =============================================================================
# Lighting Condition Detection
# =============================================================================

def detect_lighting_condition(
    rgb_mean: np.ndarray,
    rgb_variance: Optional[np.ndarray] = None,
    config: Optional[ExperimentConfig] = None,
) -> StressCondition:
    """
    Detect lighting condition from RGB statistics.

    Args:
        rgb_mean: Mean RGB values, shape (3,) or (T, 3)
        rgb_variance: RGB variance, shape (3,) or (T, 3)
        config: Experiment configuration with thresholds

    Returns:
        StressCondition with lighting classification

    Note:
        Requires rgb_mean from actual recording. Returns UNKNOWN if insufficient data.
    """
    if rgb_mean is None or len(rgb_mean) == 0:
        return StressCondition(
            condition_type=ConditionType.UNKNOWN,
            severity=1.0,
            description="No RGB data available",
            is_actionable=False,
        )

    # Compute mean brightness
    if rgb_mean.ndim > 1:
        brightness = np.mean(rgb_mean, axis=0)
    else:
        brightness = rgb_mean

    mean_brightness = np.mean(brightness)

    # Default thresholds
    if config is not None:
        low_thresh = config.lighting_low_threshold
        var_thresh = config.lighting_var_threshold
    else:
        low_thresh = 50.0
        var_thresh = 30.0

    # Check for low lighting
    if mean_brightness < low_thresh:
        severity = 1.0 - (mean_brightness / low_thresh)
        return StressCondition(
            condition_type=ConditionType.LIGHTING_LOW,
            severity=max(0.5, severity),
            evidence={"mean_brightness": float(mean_brightness), "threshold": low_thresh},
            description=f"Low lighting detected: brightness={mean_brightness:.1f}",
        )

    # Check for variable lighting
    if rgb_variance is not None:
        if rgb_variance.ndim > 1:
            variance = np.std(rgb_variance, axis=0)
        else:
            variance = rgb_variance

        max_variance = np.max(variance)

        if max_variance > var_thresh:
            severity = min(1.0, max_variance / (var_thresh * 2))
            return StressCondition(
                condition_type=ConditionType.LIGHTING_VARIABLE,
                severity=severity,
                evidence={"variance": float(max_variance), "threshold": var_thresh},
                description=f"Variable lighting detected: variance={max_variance:.1f}",
            )

    # Normal lighting
    return StressCondition(
        condition_type=ConditionType.LIGHTING_NORMAL,
        severity=0.0,
        evidence={"mean_brightness": float(mean_brightness)},
        is_actionable=False,
        description="Normal lighting conditions",
    )


# =============================================================================
# Motion Condition Detection
# =============================================================================

def detect_motion_level(
    motion_vectors: Optional[np.ndarray] = None,
    roi_displacement: Optional[np.ndarray] = None,
    face_tracking_loss: float = 0.0,
    config: Optional[ExperimentConfig] = None,
) -> StressCondition:
    """
    Detect motion level from motion estimates.

    Args:
        motion_vectors: Estimated motion magnitude per frame
        roi_displacement: ROI center displacement over time
        face_tracking_loss: Fraction of frames with tracking loss
        config: Experiment configuration with thresholds

    Returns:
        StressCondition with motion classification

    Note:
        Requires motion estimates from tracking system.
        Returns UNKNOWN if no motion data available.
    """
    if motion_vectors is None and roi_displacement is None:
        # No motion data - use tracking loss as proxy
        if face_tracking_loss > 0:
            severity = face_tracking_loss
            return StressCondition(
                condition_type=ConditionType.MOTION_SEVERE if severity > 0.5 else ConditionType.MOTION_MODERATE,
                severity=severity,
                evidence={"tracking_loss": float(face_tracking_loss)},
                description=f"Face tracking loss: {face_tracking_loss*100:.1f}%",
            )
        return StressCondition(
            condition_type=ConditionType.MOTION_NONE,
            severity=0.0,
            is_actionable=False,
            description="No motion detected",
        )

    # Default thresholds
    if config is not None:
        mild = config.motion_threshold_mild
        moderate = config.motion_threshold_moderate
        severe = config.motion_threshold_severe
    else:
        mild, moderate, severe = 0.01, 0.05, 0.1

    # Compute motion magnitude
    motion_magnitude = 0.0

    if motion_vectors is not None:
        motion_magnitude = np.mean(np.abs(motion_vectors))

    if roi_displacement is not None:
        disp_magnitude = np.mean(np.abs(roi_displacement))
        motion_magnitude = max(motion_magnitude, disp_magnitude)

    # Classify motion level
    if motion_magnitude >= severe:
        severity = min(1.0, motion_magnitude / (severe * 2))
        return StressCondition(
            condition_type=ConditionType.MOTION_SEVERE,
            severity=max(0.7, severity),
            evidence={"motion_magnitude": float(motion_magnitude), "threshold": severe},
            description=f"Severe motion: magnitude={motion_magnitude:.3f}",
        )
    elif motion_magnitude >= moderate:
        severity = motion_magnitude / moderate
        return StressCondition(
            condition_type=ConditionType.MOTION_MODERATE,
            severity=max(0.4, severity),
            evidence={"motion_magnitude": float(motion_magnitude), "threshold": moderate},
            description=f"Moderate motion: magnitude={motion_magnitude:.3f}",
        )
    elif motion_magnitude >= mild:
        severity = motion_magnitude / mild
        return StressCondition(
            condition_type=ConditionType.MOTION_MILD,
            severity=max(0.1, severity),
            evidence={"motion_magnitude": float(motion_magnitude), "threshold": mild},
            description=f"Mild motion: magnitude={motion_magnitude:.3f}",
        )
    else:
        return StressCondition(
            condition_type=ConditionType.MOTION_NONE,
            severity=0.0,
            evidence={"motion_magnitude": float(motion_magnitude)},
            is_actionable=False,
            description="No significant motion",
        )


# =============================================================================
# ROI Stability Detection
# =============================================================================

def detect_roi_stability(
    roi_coverage: float = 1.0,
    roi_center_stability: float = 0.0,
    face_detection_confidence: float = 1.0,
    config: Optional[ExperimentConfig] = None,
) -> StressCondition:
    """
    Detect ROI (Region of Interest) stability issues.

    Args:
        roi_coverage: Fraction of expected face area detected (0-1)
        roi_center_stability: Std of ROI center displacement
        face_detection_confidence: Face detector confidence (0-1)
        config: Experiment configuration with thresholds

    Returns:
        StressCondition with ROI stability classification

    Note:
        Requires face detection/tracking system output.
        Returns STABLE with low severity if no data available.
    """
    # Default thresholds
    if config is not None:
        mild = config.roi_loss_threshold_mild
        severe = config.roi_loss_threshold_severe
    else:
        mild, severe = 0.1, 0.3

    # Check coverage
    coverage_loss = 1.0 - roi_coverage

    if coverage_loss >= severe or roi_coverage < (1.0 - severe):
        severity = max(0.7, coverage_loss)
        return StressCondition(
            condition_type=ConditionType.ROI_SEVERE_LOSS,
            severity=severity,
            evidence={
                "roi_coverage": float(roi_coverage),
                "coverage_loss": float(coverage_loss),
            },
            description=f"Severe ROI loss: coverage={roi_coverage*100:.1f}%",
        )
    elif coverage_loss >= mild or roi_coverage < (1.0 - mild):
        severity = max(0.3, coverage_loss * 2)
        return StressCondition(
            condition_type=ConditionType.ROI_PARTIAL_LOSS,
            severity=severity,
            evidence={
                "roi_coverage": float(roi_coverage),
                "coverage_loss": float(coverage_loss),
            },
            description=f"Partial ROI loss: coverage={roi_coverage*100:.1f}%",
        )
    else:
        return StressCondition(
            condition_type=ConditionType.ROI_STABLE,
            severity=0.0,
            evidence={"roi_coverage": float(roi_coverage)},
            is_actionable=False,
            description="ROI stable",
        )


# =============================================================================
# Signal Quality Classification
# =============================================================================

def classify_window_quality(
    sqi: Optional[float] = None,
    confidence: Optional[float] = None,
    hr_confidence: Optional[float] = None,
    config: Optional[ExperimentConfig] = None,
) -> Tuple[ConditionType, float]:
    """
    Classify window based on signal quality indicators.

    Args:
        sqi: Signal Quality Index (0-1)
        confidence: Overall confidence estimate (0-1)
        hr_confidence: HR estimate confidence (0-1)
        config: Experiment configuration with thresholds

    Returns:
        Tuple of (ConditionType, quality_score)
    """
    if config is not None:
        high = config.sqi_threshold_high
        medium = config.sqi_threshold_medium
        low = config.sqi_threshold_low
        conf_thresh = config.confidence_threshold
        hr_conf_thresh = config.hr_confidence_threshold
    else:
        high, medium, low = 0.8, 0.5, 0.3
        conf_thresh, hr_conf_thresh = 0.5, 0.4

    # Use best available quality indicator
    quality = sqi if sqi is not None else confidence
    if quality is None and hr_confidence is not None:
        quality = hr_confidence

    if quality is None:
        return ConditionType.SQI_FAILED, 0.0

    # Check for low confidence despite SQI
    if sqi is not None and confidence is not None:
        if sqi > low and confidence < conf_thresh:
            # SQI ok but system unsure - degrade quality
            quality = min(quality, confidence)

    # Classify
    if quality >= high:
        return ConditionType.SQI_HIGH, quality
    elif quality >= medium:
        return ConditionType.SQI_MEDIUM, quality
    elif quality >= low:
        return ConditionType.SQI_LOW, quality
    else:
        return ConditionType.SQI_FAILED, quality


# =============================================================================
# Combined Window Classification
# =============================================================================

def classify_window_conditions(
    window: WindowRecord,
    config: Optional[ExperimentConfig] = None,
) -> List[StressCondition]:
    """
    Classify all conditions for a window.

    Args:
        window: WindowRecord with available data
        config: Experiment configuration

    Returns:
        List of detected StressConditions
    """
    conditions = []

    # Lighting condition
    if window.rgb_mean is not None:
        lighting = detect_lighting_condition(window.rgb_mean, config=config)
        conditions.append(lighting)

    # Signal quality condition
    sqi_type, sqi_score = classify_window_quality(
        window.signal_quality,
        window.confidence_estimate,
        window.hr_confidence,
        config=config,
    )
    sqi_cond = StressCondition(
        condition_type=sqi_type,
        severity=1.0 - sqi_score,
        evidence={"sqi": sqi_score} if sqi_score > 0 else {},
        description=f"SQI classification: {sqi_type.value}",
    )
    conditions.append(sqi_cond)

    # Signal missing/degraded
    if window.hr_estimate is None:
        conditions.append(StressCondition(
            condition_type=ConditionType.SIGNAL_MISSING,
            severity=1.0,
            description="No physiological estimates available",
        ))
    elif sqi_score < 0.3:
        conditions.append(StressCondition(
            condition_type=ConditionType.SIGNAL_DEGRADED,
            severity=1.0 - sqi_score,
            evidence={"sqi": sqi_score},
            description="Signal degraded below reliable threshold",
        ))

    return conditions


def detect_conditions_from_window(
    rgb_mean: Optional[np.ndarray] = None,
    sqi: Optional[float] = None,
    confidence: Optional[float] = None,
    hr_confidence: Optional[float] = None,
    motion_vectors: Optional[np.ndarray] = None,
    roi_coverage: float = 1.0,
    config: Optional[ExperimentConfig] = None,
) -> List[StressCondition]:
    """
    Detect conditions from raw window data.

    Args:
        rgb_mean: Mean RGB values
        sqi: Signal quality index
        confidence: Confidence estimate
        hr_confidence: HR confidence
        motion_vectors: Motion magnitude per frame
        roi_coverage: Fraction of face visible
        config: Experiment configuration

    Returns:
        List of StressConditions
    """
    conditions = []

    # Lighting
    if rgb_mean is not None:
        conditions.append(detect_lighting_condition(rgb_mean, config=config))

    # Signal quality
    sqi_type, sqi_score = classify_window_quality(
        sqi, confidence, hr_confidence, config=config
    )
    conditions.append(StressCondition(
        condition_type=sqi_type,
        severity=1.0 - sqi_score,
        evidence={"sqi": sqi_score},
    ))

    # Motion (if data available)
    motion = detect_motion_level(motion_vectors=motion_vectors, config=config)
    if motion.condition_type != ConditionType.MOTION_NONE:
        conditions.append(motion)

    # ROI stability
    if roi_coverage < 1.0:
        conditions.append(detect_roi_stability(roi_coverage=roi_coverage, config=config))

    return conditions
