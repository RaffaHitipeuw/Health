"""
Classical Method Experiments for Phase 2.

Runs controlled experiments comparing classical methods under the same benchmark protocol.

Uses Phase 1 infrastructure for evaluation.
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from rppg_phase2_signal_lab import (
    ClassicalSignalLab,
    ClassicalMethodResult,
    BPMExtractor,
    ClassicalMethod,
)
from rppg_benchmark import compute_metrics, BenchmarkResult
from rppg_benchmark_failures import FailureAnalysis


@dataclass
class ClassicalExperimentResult:
    """Result of a classical method experiment."""
    experiment_id: str
    method: str
    dataset: str
    video_id: str

    # Video info
    video_path: str
    video_fps: float
    duration_seconds: float

    # Signal characteristics
    n_samples: int
    signal_valid: bool
    snr_db: float

    # BPM estimates
    bpm_estimated: float
    bpm_windows: List[Dict] = field(default_factory=list)

    # Aligned with GT
    aligned_bpms: List[float] = field(default_factory=list)
    aligned_gt: List[float] = field(default_factory=list)
    n_aligned: int = 0

    # Metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    pearson_r: Optional[float] = None

    # Failure
    failure_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class ClassicalExperimentRunner:
    """
    Runner for classical method experiments.

    Uses a common preprocessing contract and benchmark protocol
    to ensure fair method comparison.
    """

    def __init__(
        self,
        fps: float = 30.0,
        methods: Optional[List[str]] = None,
        alignment_tolerance: float = 2.0,
    ):
        """
        Initialize experiment runner.

        Parameters
        ----------
        fps : float
            Frame rate for processing.
        methods : List[str]
            Methods to benchmark.
        alignment_tolerance : float
            GT alignment tolerance in seconds.
        """
        self.fps = fps
        self.methods = methods or ["GREEN", "CHROM", "POS", "ICA"]
        self.alignment_tolerance = alignment_tolerance
        self._lab = ClassicalSignalLab(fps=fps, methods=self.methods)

    def ingest_rgb(
        self,
        r: float,
        g: float,
        b: float,
        timestamp: float = 0.0,
    ):
        """Ingest RGB sample into all method processors."""
        self._lab.ingest_frame(r, g, b, timestamp)

    def process(self) -> Dict[str, ClassicalMethodResult]:
        """Process all accumulated data."""
        return self._lab.process_all()

    def reset(self):
        """Reset lab state."""
        self._lab.reset()


@dataclass
class MethodComparisonResult:
    """Comparison of multiple classical methods."""
    experiment_id: str
    dataset: str
    video_id: str

    # Per-method results
    method_results: Dict[str, ClassicalExperimentResult] = field(default_factory=dict)

    # Comparison matrix
    mae_ranking: List[Tuple[str, float]] = field(default_factory=list)
    rmse_ranking: List[Tuple[str, float]] = field(default_factory=list)
    pearson_ranking: List[Tuple[str, float]] = field(default_factory=list)

    # Summary
    best_mae_method: Optional[str] = None
    best_rmse_method: Optional[str] = None
    best_pearson_method: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "video_id": self.video_id,
            "methods": list(self.method_results.keys()),
            "mae_ranking": self.mae_ranking,
            "rmse_ranking": self.rmse_ranking,
            "pearson_ranking": self.pearson_ranking,
            "best_method": {
                "mae": self.best_mae_method,
                "rmse": self.best_rmse_method,
                "pearson": self.best_pearson_method,
            },
            "details": {
                name: result.to_dict()
                for name, result in self.method_results.items()
            },
        }


def compare_methods(
    method_results: Dict[str, ClassicalMethodResult],
    gt_timestamps: np.ndarray,
    gt_bpms: np.ndarray,
    experiment_id: str,
    dataset: str = "benchmark",
    video_id: str = "unknown",
) -> MethodComparisonResult:
    """
    Compare multiple methods under the same GT.

    Parameters
    ----------
    method_results : Dict[str, ClassicalMethodResult]
        Results from all methods.
    gt_timestamps : np.ndarray
        GT timestamps.
    gt_bpms : np.ndarray
        GT BPM values.
    experiment_id : str
        Experiment identifier.
    dataset : str
        Dataset name.
    video_id : str
        Video identifier.

    Returns
    -------
    MethodComparisonResult
        Comparison of all methods.
    """
    comparison = MethodComparisonResult(
        experiment_id=experiment_id,
        dataset=dataset,
        video_id=video_id,
    )

    mae_scores = []
    rmse_scores = []
    pearson_scores = []

    for method_name, result in method_results.items():
        if not result.signal_valid:
            continue

        # Build experiment result
        exp_result = ClassicalExperimentResult(
            experiment_id=f"{experiment_id}_{method_name}",
            method=method_name,
            dataset=dataset,
            video_id=video_id,
            video_path="",
            video_fps=result.fps,
            duration_seconds=len(result.timestamps) / result.fps if len(result.timestamps) > 0 else 0,
            n_samples=len(result.bvp_signal),
            signal_valid=result.signal_valid,
            snr_db=result.snr_db,
            bpm_estimated=result.bpm,
            bpm_windows=[{"bpm": bpm, "ts": ts, **diag}
                          for bpm, ts, diag in result.bpm_windows],
        )

        # Align with GT using timestamp
        aligned_preds, aligned_gt = align_predictions_with_gt(
            result.bpm_windows,
            gt_timestamps,
            gt_bpms,
            tolerance=2.0,
        )

        exp_result.aligned_bpms = aligned_preds
        exp_result.aligned_gt = aligned_gt
        exp_result.n_aligned = len(aligned_preds)

        # Compute metrics if we have enough aligned samples
        if len(aligned_preds) >= 10:
            pred_arr = np.array(aligned_preds)
            gt_arr = np.array(aligned_gt)

            benchmark_result = compute_metrics(pred_arr, gt_arr, config_name=method_name)
            exp_result.mae = benchmark_result.mae.value
            exp_result.rmse = benchmark_result.rmse.value
            exp_result.pearson_r = benchmark_result.pearson_r.value

            mae_scores.append((method_name, exp_result.mae))
            rmse_scores.append((method_name, exp_result.rmse))
            pearson_scores.append((method_name, exp_result.pearson_r.value))

        comparison.method_results[method_name] = exp_result

    # Rank methods
    comparison.mae_ranking = sorted(mae_scores, key=lambda x: x[1])
    comparison.rmse_ranking = sorted(rmse_scores, key=lambda x: x[1])
    comparison.pearson_ranking = sorted(pearson_scores, key=lambda x: -x[1])  # Higher is better

    if comparison.mae_ranking:
        comparison.best_mae_method = comparison.mae_ranking[0][0]
    if comparison.rmse_ranking:
        comparison.best_rmse_method = comparison.rmse_ranking[0][0]
    if comparison.pearson_ranking:
        comparison.best_pearson_method = comparison.pearson_ranking[0][0]

    return comparison


def align_predictions_with_gt(
    bpm_windows: List[Tuple[float, float, Dict]],
    gt_timestamps: np.ndarray,
    gt_bpms: np.ndarray,
    tolerance: float = 2.0,
) -> Tuple[List[float], List[float]]:
    """
    Align BPM windows with ground truth.

    Parameters
    ----------
    bpm_windows : List[Tuple[bpm, timestamp, diag]
        BPM estimates with timestamps.
    gt_timestamps : np.ndarray
        GT timestamps.
    gt_bpms : np.ndarray
        GT BPM values.
    tolerance : float
        Alignment tolerance in seconds.

    Returns
    -------
    Tuple[List[float], List[float]]
        (aligned_predictions, aligned_gt) arrays.
    """
    if len(bpm_windows) == 0 or len(gt_timestamps) == 0:
        return [], []

    aligned_preds = []
    aligned_gt = []

    for bpm, ts, _ in bpm_windows:
        if bpm <= 0:
            continue

        # Find nearest GT within tolerance
        dt = np.abs(gt_timestamps - ts)
        min_idx = np.argmin(dt)
        if dt[min_idx] <= tolerance:
            aligned_preds.append(bpm)
            aligned_gt.append(gt_bpms[min_idx])

    return aligned_preds, aligned_gt


# =============================================================================
# METHOD MATRIX
# =============================================================================

def generate_method_matrix(
    comparison: MethodComparisonResult,
) -> Dict[str, Any]:
    """
    Generate method comparison matrix.

    Returns a structured matrix comparing methods on key characteristics.
    """
    matrix = {
        "header": [
            "Method",
            "BPM Est",
            "SNR (dB)",
            "N Aligned",
            "MAE",
            "RMSE",
            "Pearson r",
        ],
        "rows": [],
    }

    for method_name, result in comparison.method_results.items():
        row = [
            method_name,
            f"{result.bpm_estimated:.1f}" if result.bpm_estimated > 0 else "N/A",
            f"{result.snr_db:.2f}" if result.snr_db > -999 else "N/A",
            str(result.n_aligned),
            f"{result.mae:.2f}" if result.mae is not None else "N/A",
            f"{result.rmse:.2f}" if result.rmse is not None else "N/A",
            f"{result.pearson_r:.3f}" if result.pearson_r is not None else "N/A",
        ]
        matrix["rows"].append(row)

    return matrix


def format_comparison_report(comparison: MethodComparisonResult) -> str:
    """Format method comparison as human-readable report."""
    lines = [
        "=" * 70,
        f"CLASSICAL METHOD COMPARISON: {comparison.experiment_id}",
        "=" * 70,
        f"Dataset: {comparison.dataset}",
        f"Video:   {comparison.video_id}",
        "",
        "METHOD RANKINGS:",
        f"  Best MAE:     {comparison.best_mae_method or 'N/A'}",
        f"  Best RMSE:    {comparison.best_rmse_method or 'N/A'}",
        f"  Best Pearson:  {comparison.best_pearson_method or 'N/A'}",
        "",
        "-" * 70,
        "METHOD RESULTS:",
    ]

    for method_name, result in comparison.method_results.items():
        lines.append(f"\n{method_name}:")
        lines.append(f"  BPM Estimate: {result.bpm_estimated:.1f}")
        lines.append(f"  SNR:          {result.snr_db:.2f} dB")
        lines.append(f"  Aligned:      {result.n_aligned}")
        if result.mae is not None:
            lines.append(f"  MAE:          {result.mae:.2f} BPM")
        if result.rmse is not None:
            lines.append(f"  RMSE:         {result.rmse:.2f} BPM")
        if result.pearson_r is not None:
            lines.append(f"  Pearson r:    {result.pearson_r:.4f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Classical Method Experiments Ready")
    print("\nUsage:")
    print("  from rppg_phase2_experiments import ClassicalExperimentRunner")
    print("  runner = ClassicalExperimentRunner(fps=30.0, methods=['CHROM', 'POS']")
    print("  runner.ingest_rgb(r, g, b, timestamp)")
    print("  results = runner.process()")
