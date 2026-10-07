"""
Timestamp alignment module for benchmark experiments.

Implements robust timestamp-based alignment between predictions and ground truth
with explicit tolerance enforcement.

ALIGNMENT RULES (per Phase 1 requirements):
1. Use timestamps, not indices for alignment
2. Match each prediction to nearest GT within tolerance
3. Reject pairs where |pred_ts - gt_ts| > tolerance
4. Each GT observation can only match one prediction
5. Output explicit statistics about alignment quality
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict


@dataclass
class AlignmentResult:
    """
    Result of timestamp-based alignment.

    Attributes
    ----------
    aligned_predictions : np.ndarray
        Predicted BPM values that aligned with GT.
    aligned_ground_truth : np.ndarray
        GT BPM values matched to predictions.
    alignment_deltas : np.ndarray
        Time difference for each aligned pair (|pred_ts - gt_ts|).
    diagnostics : Dict
        Alignment statistics including rejection counts.
    """
    aligned_predictions: np.ndarray
    aligned_ground_truth: np.ndarray
    alignment_deltas: np.ndarray
    diagnostics: Dict = field(default_factory=dict)

    @property
    def n_aligned(self) -> int:
        return len(self.aligned_predictions)

    @property
    def mean_delta(self) -> float:
        if len(self.alignment_deltas) == 0:
            return float('nan')
        return float(np.mean(self.alignment_deltas))

    @property
    def max_delta(self) -> float:
        if len(self.alignment_deltas) == 0:
            return float('nan')
        return float(np.max(self.alignment_deltas))


def align_timestamps(
    pred_timestamps: np.ndarray,
    pred_bpms: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_bpms: np.ndarray,
    tolerance: float,
) -> AlignmentResult:
    """
    Align predictions to ground truth using timestamp-based nearest-neighbor matching.

    This implements the canonical alignment algorithm:
    1. For each prediction, find the nearest unused GT timestamp
    2. Accept only if |pred_ts - nearest_gt_ts| <= tolerance
    3. Each GT observation matches at most one prediction

    Parameters
    ----------
    pred_timestamps : np.ndarray
        Prediction timestamps (seconds).
    pred_bpms : np.ndarray
        Predicted BPM values.
    gt_timestamps : np.ndarray
        Ground truth timestamps (seconds).
    gt_bpms : np.ndarray
        Ground truth BPM values.
    tolerance : float
        Maximum time difference (seconds) for valid match.

    Returns
    -------
    AlignmentResult
        Aligned pairs with diagnostics.

    Notes
    -----
    - Uses greedy matching: predictions processed in timestamp order
    - Each GT observation can only match one prediction
    - No GT observation can be matched to multiple predictions
    """
    # Convert to numpy arrays
    pred_ts = np.asarray(pred_timestamps, dtype=float)
    pred_bpm = np.asarray(pred_bpms, dtype=float)
    gt_ts = np.asarray(gt_timestamps, dtype=float)
    gt_bpm = np.asarray(gt_bpms, dtype=float)

    n_pred = len(pred_ts)
    n_gt = len(gt_ts)

    # Sort GT by timestamp for efficient matching
    gt_order = np.argsort(gt_ts)
    gt_ts_sorted = gt_ts[gt_order]
    gt_bpm_sorted = gt_bpm[gt_order]

    # Track which GT observations have been used
    gt_used = np.zeros(n_gt, dtype=bool)

    # Output arrays
    aligned_pred = []
    aligned_gt = []
    aligned_deltas = []
    n_rejected_tolerance = 0
    n_rejected_no_gt = 0

    for i in range(n_pred):
        p_ts = pred_ts[i]
        p_bpm = pred_bpm[i]

        # Skip invalid predictions
        if not np.isfinite(p_ts) or not np.isfinite(p_bpm):
            continue

        # Find nearest unused GT
        best_dt = tolerance + 1.0  # Initialize above tolerance
        best_gt_idx = None
        best_gt_ts = None
        best_gt_val = None

        for j in range(n_gt):
            if gt_used[gt_order[j]]:
                continue

            dt = abs(p_ts - gt_ts_sorted[j])
            if dt < best_dt:
                best_dt = dt
                best_gt_idx = gt_order[j]  # Original index for tracking
                best_gt_ts = gt_ts_sorted[j]
                best_gt_val = gt_bpm_sorted[j]

        # Accept only if within tolerance
        if best_gt_idx is not None and best_dt <= tolerance:
            aligned_pred.append(p_bpm)
            aligned_gt.append(best_gt_val)
            aligned_deltas.append(best_dt)
            gt_used[best_gt_idx] = True
        elif best_gt_idx is None:
            n_rejected_no_gt += 1
        else:
            n_rejected_tolerance += 1

    # Build result
    result = AlignmentResult(
        aligned_predictions=np.array(aligned_pred),
        aligned_ground_truth=np.array(aligned_gt),
        alignment_deltas=np.array(aligned_deltas),
        diagnostics={
            "n_predictions": n_pred,
            "n_gt": n_gt,
            "n_aligned": len(aligned_pred),
            "n_rejected_tolerance": n_rejected_tolerance,
            "n_rejected_no_gt": n_rejected_no_gt,
            "n_unmatched_predictions": n_rejected_tolerance + n_rejected_no_gt,
            "n_unmatched_gt": int(np.sum(~gt_used)),
            "alignment_rate": len(aligned_pred) / max(n_pred, 1),
            "tolerance_seconds": tolerance,
            "mean_delta_seconds": float(np.mean(aligned_deltas)) if len(aligned_deltas) > 0 else float('nan'),
            "max_delta_seconds": float(np.max(aligned_deltas)) if len(aligned_deltas) > 0 else float('nan'),
        }
    )

    return result


def align_with_predictions_dict(
    predictions: List[Dict],
    gt_timestamps: np.ndarray,
    gt_bpms: np.ndarray,
    tolerance: float,
    ts_key: str = "timestamp",
    bpm_key: str = "bpm",
) -> AlignmentResult:
    """
    Align predictions (as dicts) to ground truth.

    Parameters
    ----------
    predictions : List[Dict]
        List of prediction dicts with timestamp and BPM.
    gt_timestamps : np.ndarray
        Ground truth timestamps.
    gt_bpms : np.ndarray
        Ground truth BPM values.
    tolerance : float
        Maximum time difference for valid match.
    ts_key : str
        Key for timestamp in prediction dict.
    bpm_key : str
        Key for BPM in prediction dict.

    Returns
    -------
    AlignmentResult
        Aligned pairs with diagnostics.
    """
    pred_ts = np.array([p[ts_key] for p in predictions])
    pred_bpm = np.array([p[bpm_key] for p in predictions])

    return align_timestamps(pred_ts, pred_bpm, gt_timestamps, gt_bpms, tolerance)


def validate_alignment_input(
    pred_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    tolerance: float,
) -> List[str]:
    """
    Validate alignment inputs and return list of warnings/errors.

    Parameters
    ----------
    pred_timestamps : np.ndarray
        Prediction timestamps.
    gt_timestamps : np.ndarray
        Ground truth timestamps.
    tolerance : float
        Alignment tolerance.

    Returns
    -------
    List[str]
        List of validation messages (empty if valid).
    """
    messages = []

    if tolerance <= 0:
        messages.append(f"WARNING: Tolerance must be positive, got {tolerance}")

    if len(pred_timestamps) == 0:
        messages.append("WARNING: No predictions to align")

    if len(gt_timestamps) == 0:
        messages.append("WARNING: No ground truth to align against")

    # Check for potential issues
    pred_range = np.ptp(pred_timestamps) if len(pred_timestamps) > 0 else 0
    gt_range = np.ptp(gt_timestamps) if len(gt_timestamps) > 0 else 0

    if pred_range > 0 and gt_range > 0:
        if pred_range < gt_range * 0.5:
            messages.append(
                f"WARNING: Prediction time range ({pred_range:.1f}s) much shorter "
                f"than GT range ({gt_range:.1f}s)"
            )

        if not np.any(np.abs(pred_timestamps[:, None] - gt_timestamps) <= tolerance):
            messages.append(
                f"WARNING: No predictions within {tolerance}s of any GT timestamp"
            )

    return messages
