"""
Failure analysis for benchmark experiments.

Tracks and reports failure modes in a structured way.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict


@dataclass
class FailureEvent:
    """Single failure event."""
    frame_index: int
    timestamp: float
    failure_type: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureAnalysis:
    """
    Comprehensive failure analysis for a benchmark experiment.
    """
    # Frame-level failures
    total_frames: int = 0
    frames_with_face: int = 0
    frames_without_face: int = 0
    frames_invalid: int = 0
    frames_exception: int = 0

    # Prediction-level failures
    total_predictions: int = 0
    valid_predictions: int = 0
    zero_bpm_predictions: int = 0
    sqi_gated_predictions: int = 0

    # Alignment failures
    total_gt_samples: int = 0
    aligned_samples: int = 0
    unmatched_predictions: int = 0
    unmatched_gt: int = 0

    # Metric failures
    mae_available: bool = True
    rmse_available: bool = True
    pearson_available: bool = True
    insufficient_aligned_samples: bool = False

    # Failure events
    failure_events: List[FailureEvent] = field(default_factory=list)

    @property
    def frame_failure_rate(self) -> float:
        """Fraction of frames that failed."""
        if self.total_frames == 0:
            return 0.0
        return self.frames_without_face / self.total_frames

    @property
    def prediction_failure_rate(self) -> float:
        """Fraction of predictions that are invalid."""
        if self.total_predictions == 0:
            return 0.0
        return self.zero_bpm_predictions / self.total_predictions

    @property
    def alignment_rate(self) -> float:
        """Fraction of predictions that aligned with GT."""
        if self.total_predictions == 0:
            return 0.0
        return self.aligned_samples / self.total_predictions

    @property
    def summary(self) -> Dict[str, Any]:
        """Get failure analysis summary."""
        return {
            "frames": {
                "total": self.total_frames,
                "with_face": self.frames_with_face,
                "without_face": self.frames_without_face,
                "invalid": self.frames_invalid,
                "exception": self.frames_exception,
                "failure_rate": round(self.frame_failure_rate, 4),
            },
            "predictions": {
                "total": self.total_predictions,
                "valid": self.valid_predictions,
                "zero_bpm": self.zero_bpm_predictions,
                "sqi_gated": self.sqi_gated_predictions,
                "failure_rate": round(self.prediction_failure_rate, 4),
            },
            "alignment": {
                "gt_samples": self.total_gt_samples,
                "aligned": self.aligned_samples,
                "unmatched_predictions": self.unmatched_predictions,
                "unmatched_gt": self.unmatched_gt,
                "alignment_rate": round(self.alignment_rate, 4),
            },
            "metrics": {
                "mae_available": self.mae_available,
                "rmse_available": self.rmse_available,
                "pearson_available": self.pearson_available,
                "insufficient_samples": self.insufficient_aligned_samples,
            },
            "valid_samples_for_metrics": (
                self.aligned_samples >= 10 and
                self.mae_available and
                self.rmse_available and
                self.pearson_available
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = self.summary.copy()
        d["failure_events"] = [
            {
                "frame_index": e.frame_index,
                "timestamp": e.timestamp,
                "type": e.failure_type,
                "reason": e.reason,
                "details": e.details,
            }
            for e in self.failure_events[:100]  # Limit events to first 100
        ]
        if len(self.failure_events) > 100:
            d["failure_events_truncated"] = True
            d["total_failure_events"] = len(self.failure_events)
        return d

    def log_failure(
        self,
        frame_index: int,
        timestamp: float,
        failure_type: str,
        reason: str,
        **details
    ):
        """Log a failure event."""
        self.failure_events.append(FailureEvent(
            frame_index=frame_index,
            timestamp=timestamp,
            failure_type=failure_type,
            reason=reason,
            details=details,
        ))

    def update_from_result(self, result: Dict[str, Any]):
        """Update analysis from benchmark result."""
        self.total_frames = result.get("n_frames_processed", 0)
        self.frames_without_face = result.get("n_frames_no_face", 0)
        self.frames_invalid = result.get("n_frames_invalid", 0)
        self.total_predictions = result.get("n_predictions_total", 0)
        self.zero_bpm_predictions = result.get("n_predictions_zero_bpm", 0)
        self.total_gt_samples = result.get("n_gt_total", 0)
        self.aligned_samples = result.get("n_aligned", 0)
        self.unmatched_predictions = result.get("n_unmatched_predictions", 0)
        self.unmatched_gt = result.get("n_unmatched_gt", 0)
        self.frames_with_face = self.total_frames - self.frames_without_face

        # Metric availability
        self.mae_available = result.get("mae") is not None
        self.rmse_available = result.get("rmse") is not None
        self.pearson_available = result.get("pearson_r") is not None
        self.insufficient_aligned_samples = self.aligned_samples < 10

        self.valid_predictions = self.total_predictions - self.zero_bpm_predictions


# Common failure types
FAILURE_TYPES = {
    "FACE_NOT_DETECTED": "MediaPipe did not detect a face in the frame",
    "ROI_INVALID": "ROI extraction returned invalid region",
    "ROI_BRIGHTNESS_LOW": "ROI brightness below minimum threshold",
    "ROI_BRIGHTNESS_HIGH": "ROI brightness above maximum threshold",
    "SQI_GATED": "Signal quality below SQI gate threshold",
    "MOTION_REJECTED": "Motion detector rejected frame",
    "EXPOSURE_DRIFT": "Exposure drift detected and frozen output",
    "POSE_INVALID": "Head pose outside acceptable range",
    "ZERO_BPM": "BPM estimate is zero or invalid",
    "INVALID_BPM": "BPM estimate outside physiological range",
    "ALIGNMENT_FAILED": "No GT sample within tolerance",
    "METRIC_UNAVAILABLE": "Insufficient aligned samples for metric",
}


def format_failure_report(analysis: FailureAnalysis) -> str:
    """
    Format failure analysis as human-readable report.

    Parameters
    ----------
    analysis : FailureAnalysis
        Failure analysis to format.

    Returns
    -------
    str
        Formatted failure report.
    """
    summary = analysis.summary

    lines = [
        "=" * 60,
        "BENCHMARK FAILURE ANALYSIS",
        "=" * 60,
        "",
        "FRAMES:",
        f"  Total:         {summary['frames']['total']}",
        f"  With face:     {summary['frames']['with_face']}",
        f"  Without face:  {summary['frames']['without_face']}",
        f"  Invalid:      {summary['frames']['invalid']}",
        f"  Exception:     {summary['frames']['exception']}",
        f"  Failure rate: {summary['frames']['failure_rate']:.1%}",
        "",
        "PREDICTIONS:",
        f"  Total:        {summary['predictions']['total']}",
        f"  Valid:        {summary['predictions']['valid']}",
        f"  Zero BPM:     {summary['predictions']['zero_bpm']}",
        f"  SQI gated:    {summary['predictions']['sqi_gated']}",
        f"  Failure rate: {summary['predictions']['failure_rate']:.1%}",
        "",
        "ALIGNMENT:",
        f"  GT samples:           {summary['alignment']['gt_samples']}",
        f"  Aligned:              {summary['alignment']['aligned']}",
        f"  Unmatched predictions: {summary['alignment']['unmatched_predictions']}",
        f"  Unmatched GT:         {summary['alignment']['unmatched_gt']}",
        f"  Alignment rate:        {summary['alignment']['alignment_rate']:.1%}",
        "",
        "METRICS:",
        f"  MAE available:     {summary['metrics']['mae_available']}",
        f"  RMSE available:    {summary['metrics']['rmse_available']}",
        f"  Pearson available: {summary['metrics']['pearson_available']}",
        f"  Insufficient:      {summary['metrics']['insufficient_samples']}",
        f"  Valid for reporting: {summary['valid_samples_for_metrics']}",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)
