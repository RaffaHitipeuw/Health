"""
Minimal benchmark runner for prerecorded video + ground truth experiments.

Provides the minimum executable benchmark path for frozen FusionEngineV2.

Execution path:
    video → VideoIterator → MediaPipe landmarks → FusionEngineV2.update()
        → timestamped predictions → GroundTruthInterface alignment → metrics → JSON result
"""

import cv2
import json
import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# MediaPipe import with version compatibility
try:
    import mediapipe as mp
    # Try new API (MediaPipe 0.10+)
    try:
        from mediapipe.tasks import python as mp_tasks
        _MEDIAPIPE_VERSION = "new"
    except ImportError:
        _MEDIAPIPE_VERSION = "old"
except ImportError:
    mp = None
    _MEDIAPIPE_VERSION = None

from rppg_config import cfg
from rppg_core import MultiROIFusionEngineV2
from rppg_vitals import GroundTruthInterface
from rppg_benchmark_video import VideoIterator
from rppg_benchmark_gt import (
    load_ground_truth,
    load_ground_truth_for_interface,
    GroundTruthData,
    GroundTruthLoadError,
)
from rppg_benchmark import (
    compute_metrics,
    MetricResult,
    BlandAltmanResult,
    BenchmarkResult,
    ReproducibilityProtocol,
)


# =============================================================================
# Result Schema
# =============================================================================

@dataclass
class BenchmarkExperimentResult:
    """
    Experiment result schema for benchmark output.

    Contains video metadata, prediction/GT alignment, metrics, and runtime info.
    """
    experiment_id: str
    dataset: str
    subject_id: Optional[str]
    video_id: Optional[str]

    # Video metadata
    video_path: str
    video_fps: float
    video_frame_count: int
    video_duration_seconds: float
    video_width: int
    video_height: int

    # Benchmark configuration
    alignment_tolerance_seconds: float
    timestamp_source: str = "video_frame_timeline"

    # Predictions collected during run
    predictions: List[Dict] = field(default_factory=list)
    ground_truth: List[Dict] = field(default_factory=list)

    # Aligned prediction/GT pairs
    aligned_predictions: List[Dict] = field(default_factory=list)

    # Metrics
    mae: Optional[float] = None
    mae_ci: Optional[Tuple[float, float]] = None
    rmse: Optional[float] = None
    rmse_ci: Optional[Tuple[float, float]] = None
    pearson_r: Optional[float] = None
    pearson_ci: Optional[Tuple[float, float]] = None

    # Alignment diagnostics
    n_predictions_total: int = 0
    n_gt_total: int = 0
    n_aligned: int = 0
    n_unmatched_predictions: int = 0
    n_unmatched_gt: int = 0

    # Failure tracking
    n_frames_processed: int = 0
    n_frames_no_face: int = 0
    n_frames_invalid: int = 0
    n_predictions_zero_bpm: int = 0

    # Runtime
    elapsed_seconds: float = 0.0
    fps_processing: float = 0.0

    # Reproducibility
    config: Dict[str, Any] = field(default_factory=dict)
    software_version: str = "1.0.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert tuple CI to list for JSON
        if self.mae_ci is not None:
            d["mae_ci"] = list(self.mae_ci)
        if self.rmse_ci is not None:
            d["rmse_ci"] = list(self.rmse_ci)
        if self.pearson_ci is not None:
            d["pearson_ci"] = list(self.pearson_ci)
        # Remove Nones for cleaner JSON
        d = {k: v for k, v in d.items() if v is not None}
        return d

    def save_json(self, output_path: str) -> str:
        """Save result to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return output_path


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Minimal benchmark runner for prerecorded video experiments.

    Reuses:
    - VideoIterator for deterministic frame iteration
    - MediaPipe FaceMesh for face landmark detection
    - FusionEngineV2.update() for frozen rPPG processing
    - GroundTruthInterface for timestamp-based alignment
    - compute_metrics() for metric calculation
    """

    # Default alignment tolerance in seconds
    DEFAULT_ALIGN_TOLERANCE = 2.0

    def __init__(
        self,
        video_path: str,
        gt_path: str,
        experiment_id: str,
        alignment_tolerance: float = DEFAULT_ALIGN_TOLERANCE,
        dataset: str = "unknown",
        subject_id: Optional[str] = None,
        video_id: Optional[str] = None,
        enable_logging: bool = False,
    ):
        """
        Initialize benchmark runner.

        Parameters
        ----------
        video_path : str
            Path to prerecorded video file.
        gt_path : str
            Path to ground truth CSV file.
        experiment_id : str
            Unique identifier for this experiment.
        alignment_tolerance : float
            Maximum time difference (seconds) for GT matching.
        dataset : str
            Dataset name identifier.
        subject_id : Optional[str]
            Subject identifier if known.
        video_id : Optional[str]
            Video identifier if known.
        enable_logging : bool
            Enable reproducibility logging in fusion engine.
        """
        self.video_path = video_path
        self.gt_path = gt_path
        self.experiment_id = experiment_id
        self.alignment_tolerance = alignment_tolerance
        self.dataset = dataset
        self.subject_id = subject_id
        self.video_id = video_id
        self.enable_logging = enable_logging

        # State initialized in run()
        self._video: Optional[VideoIterator] = None
        self._face_mesh: Optional[Any] = None
        self._fusion_engine: Optional[MultiROIFusionEngineV2] = None
        self._gt_interface: Optional[GroundTruthInterface] = None
        self._gt_data: Optional[GroundTruthData] = None
        self._repro_protocol: Optional[ReproducibilityProtocol] = None

        # Results
        self._predictions: List[Dict] = []
        self._result: Optional[BenchmarkExperimentResult] = None
        self._start_time: float = 0.0

    def _initialize_video(self) -> VideoIterator:
        """Initialize video iterator."""
        try:
            return VideoIterator(self.video_path)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Cannot initialize video: {e}")

    def _initialize_face_mesh(self):
        """Initialize MediaPipe FaceMesh."""
        if mp is None:
            raise RuntimeError(
                "MediaPipe is not installed. "
                "Install with: pip install mediapipe"
            )

        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            # MediaPipe 0.9.x and earlier - use legacy API
            return mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            # MediaPipe 0.10+ detected - needs legacy solutions API
            raise RuntimeError(
                "MediaPipe 0.10+ detected, which uses a different API. "
                "This benchmark requires MediaPipe with face_mesh solutions API. "
                "Install an older version: pip install mediapipe==0.9.0"
            )

    def _initialize_fusion_engine(self) -> MultiROIFusionEngineV2:
        """Initialize FusionEngineV2."""
        return MultiROIFusionEngineV2(enable_logging=self.enable_logging)

    def _load_ground_truth(self) -> GroundTruthData:
        """Load and validate ground truth file."""
        try:
            gt_data = load_ground_truth(self.gt_path)
            return gt_data
        except GroundTruthLoadError as e:
            raise ValueError(f"Cannot load ground truth: {e}")

    def _setup_gt_interface(self, gt_data: GroundTruthData) -> GroundTruthInterface:
        """
        Set up GroundTruthInterface with loaded GT data.

        Aligns observations using timestamps from the GT file.
        """
        gti = GroundTruthInterface(align_tolerance=self.alignment_tolerance)

        # Load GT into interface
        gt_series = load_ground_truth_for_interface(self.gt_path)
        gti.gt_bpm_series = gt_series

        return gti

    def _process_video_frame(
        self,
        frame,
        face_landmarks,
        h: int,
        w: int,
        timestamp: float,
    ) -> Tuple[float, float, bool]:
        """
        Process single video frame through FusionEngineV2.

        Returns (fused_bpm, fused_sqi, success).
        """
        result = self._fusion_engine.update(frame, face_landmarks, h, w)
        return result.fused_bpm, result.fused_sqi, True

    def run(
        self,
        max_frames: Optional[int] = None,
        verbose: bool = False,
    ) -> BenchmarkExperimentResult:
        """
        Execute benchmark on prerecorded video.

        Parameters
        ----------
        max_frames : Optional[int]
            Maximum frames to process (None for all frames).
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        BenchmarkExperimentResult
            Complete experiment results.

        Raises
        ------
        ValueError
            If video or GT file cannot be loaded.
        """
        self._start_time = time.time()

        if verbose:
            print(f"[BENCHMARK] Starting experiment: {self.experiment_id}")
            print(f"  Video: {self.video_path}")
            print(f"  GT: {self.gt_path}")
            print(f"  Alignment tolerance: {self.alignment_tolerance}s")

        # Initialize components
        self._video = self._initialize_video()
        self._face_mesh = self._initialize_face_mesh()
        self._fusion_engine = self._initialize_fusion_engine()
        self._gt_data = self._load_ground_truth()
        self._gt_interface = self._setup_gt_interface(self._gt_data)
        self._repro_protocol = ReproducibilityProtocol()

        # Initialize result
        self._result = BenchmarkExperimentResult(
            experiment_id=self.experiment_id,
            dataset=self.dataset,
            subject_id=self.subject_id,
            video_id=self.video_id,
            video_path=self.video_path,
            video_fps=self._video.fps,
            video_frame_count=self._video.frame_count,
            video_duration_seconds=self._video.duration_seconds,
            video_width=self._video.width,
            video_height=self._video.height,
            alignment_tolerance_seconds=self.alignment_tolerance,
            timestamp_source="video_frame_timeline",
            ground_truth=[
                {"timestamp": e.timestamp, "bpm": e.bpm}
                for e in self._gt_data.entries
            ],
        )

        # Statistics
        n_frames = 0
        n_no_face = 0
        n_invalid = 0
        n_zero_bpm = 0
        predictions = []

        if verbose:
            print(f"  Video: {self._video.fps:.1f}fps, "
                  f"{self._video.frame_count} frames, "
                  f"{self._video.duration_seconds:.1f}s")

        # Process video frames
        for frame_data in self._video:
            if max_frames is not None and n_frames >= max_frames:
                break

            n_frames += 1

            # MediaPipe face detection
            rgb = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
            mp_result = self._face_mesh.process(rgb)
            face_landmarks = (
                mp_result.multi_face_landmarks[0]
                if mp_result.multi_face_landmarks else None
            )

            if face_landmarks is None:
                n_no_face += 1
                if verbose and n_frames % 30 == 0:
                    print(f"  [frame {n_frames}] No face detected")
                continue

            # Process through FusionEngineV2
            h, w = frame_data.height, frame_data.width
            fused_bpm, fused_sqi, success = self._process_video_frame(
                frame_data.frame,
                face_landmarks,
                h, w,
                frame_data.timestamp_seconds,
            )

            if not success:
                n_invalid += 1
                continue

            # Record prediction with VIDEO TIMESTAMP (not wall-clock)
            # This is critical: timestamp must come from video timing for GT alignment
            self._gt_interface.record_estimate(fused_bpm)

            prediction = {
                "frame_index": frame_data.frame_index,
                "timestamp": frame_data.timestamp_seconds,
                "bpm": fused_bpm,
                "sqi": fused_sqi,
            }
            predictions.append(prediction)

            if fused_bpm <= 0:
                n_zero_bpm += 1

            if verbose and n_frames % 100 == 0:
                print(f"  [frame {n_frames}/{self._video.frame_count}] "
                      f"BPM={fused_bpm:.1f}, SQI={fused_sqi:.1f}")

        # Close video
        self._video.close()

        # Store predictions
        self._predictions = predictions
        self._result.predictions = predictions
        self._result.n_frames_processed = n_frames
        self._result.n_frames_no_face = n_no_face
        self._result.n_frames_invalid = n_invalid
        self._result.n_predictions_zero_bpm = n_zero_bpm
        self._result.n_predictions_total = len(predictions)
        self._result.n_gt_total = self._gt_data.n_entries

        # Align predictions with ground truth
        # Note: GroundTruthInterface.align_observations() may have a tolerance enforcement bug.
        # We apply post-hoc filtering to ensure tolerance is respected.
        aligned_est, aligned_gt, align_diag = self._gt_interface.align_observations(
            tolerance=self.alignment_tolerance
        )

        # Post-hoc tolerance filter: remove pairs where dt > tolerance
        # This ensures alignment actually respects the tolerance parameter
        tolerance = self.alignment_tolerance
        filtered_pairs = []
        gt_sorted = sorted(self._gt_data.entries, key=lambda e: e.timestamp)
        est_timestamps = [p["timestamp"] for p in predictions]
        used_gt = set()

        for i, est_ts in enumerate(est_timestamps):
            best_dt = tolerance + 1.0
            best_gt_idx = None
            for j, gt_entry in enumerate(gt_sorted):
                if j in used_gt:
                    continue
                dt = abs(est_ts - gt_entry.timestamp)
                if dt < best_dt:
                    best_dt = dt
                    best_gt_idx = j
            if best_gt_idx is not None and best_dt <= tolerance:
                filtered_pairs.append((aligned_est[i], aligned_gt[i], best_dt))
                used_gt.add(best_gt_idx)

        # Update aligned arrays with filtered pairs
        aligned_est = np.array([p[0] for p in filtered_pairs])
        aligned_gt = np.array([p[1] for p in filtered_pairs])

        # Recalculate diagnostics with proper tolerance enforcement
        n_matched = len(aligned_est)
        n_unmatched_est = len(predictions) - n_matched
        n_unmatched_gt = self._gt_data.n_entries - n_matched

        self._result.n_aligned = n_matched
        self._result.n_unmatched_predictions = n_unmatched_est
        self._result.n_unmatched_gt = n_unmatched_gt

        # Build aligned predictions for output
        for i in range(len(aligned_est)):
            self._result.aligned_predictions.append({
                "timestamp_est": float(predictions[i]["timestamp"]) if i < len(predictions) else 0.0,
                "bpm_predicted": float(aligned_est[i]),
                "bpm_ground_truth": float(aligned_gt[i]),
            })

        # Compute metrics if enough aligned samples
        if self._result.n_aligned >= 10:
            measured = aligned_est
            reference = aligned_gt

            # MAE, RMSE using bootstrap
            mae_result, rmse_result, pearson_result = self._compute_bootstrap_metrics(
                measured, reference
            )

            self._result.mae = mae_result.value
            self._result.mae_ci = (mae_result.ci_lower, mae_result.ci_upper)
            self._result.rmse = rmse_result.value
            self._result.rmse_ci = (rmse_result.rmse_ci if hasattr(rmse_result, 'rmse_ci') else None)
            self._result.pearson_r = pearson_result.value
            self._result.pearson_ci = (pearson_result.ci_lower, pearson_result.ci_upper)

        elif self._result.n_aligned >= 3:
            # Limited samples: compute basic metrics only
            mae_val = float(np.mean(np.abs(aligned_est - aligned_gt)))
            rmse_val = float(np.sqrt(np.mean((aligned_est - aligned_gt)**2)))
            self._result.mae = mae_val
            self._result.rmse = rmse_val
            self._result.pearson_r = None  # Insufficient for Pearson

        # Runtime info
        elapsed = time.time() - self._start_time
        self._result.elapsed_seconds = elapsed
        self._result.fps_processing = n_frames / elapsed if elapsed > 0 else 0.0

        # Config snapshot
        self._result.config = {
            "alignment_tolerance": self.alignment_tolerance,
            "min_aligned_for_metrics": 10,
            "enable_logging": self.enable_logging,
        }

        if verbose:
            print(f"\n[BENCHMARK] Experiment complete: {self.experiment_id}")
            print(f"  Frames processed: {n_frames}")
            print(f"  Predictions: {len(predictions)}")
            print(f"  GT entries: {self._gt_data.n_entries}")
            print(f"  Aligned: {self._result.n_aligned}")
            print(f"  Elapsed: {elapsed:.1f}s ({self._result.fps_processing:.1f} fps)")
            if self._result.mae is not None:
                print(f"  MAE: {self._result.mae:.2f} BPM")
                print(f"  RMSE: {self._result.rmse:.2f} BPM")
            if self._result.pearson_r is not None:
                print(f"  Pearson r: {self._result.pearson_r:.4f}")

        return self._result

    def _compute_bootstrap_metrics(
        self,
        measured: np.ndarray,
        reference: np.ndarray,
    ) -> Tuple[MetricResult, MetricResult, MetricResult]:
        """
        Compute bootstrap confidence intervals for metrics.

        Reuses the bootstrap_metric function from rppg_benchmark.
        """
        # Use bootstrap_metric from rppg_benchmark
        mae_result = self._bootstrap_metric(
            lambda m, r: float(np.mean(np.abs(m - r))),
            measured,
            reference,
        )
        rmse_result = self._bootstrap_metric(
            lambda m, r: float(np.sqrt(np.mean((m - r)**2))),
            measured,
            reference,
        )

        # Pearson
        try:
            from scipy.stats import pearsonr as sp_pearsonr
            pearson_fn = lambda m, r: float(sp_pearsonr(m, r)[0])
        except Exception:
            def pearson_fn(m, r):
                c = np.corrcoef(m, r)
                return float(c[0, 1]) if c.shape == (2, 2) else 0.0

        pearson_result = self._bootstrap_metric(pearson_fn, measured, reference)

        return mae_result, rmse_result, pearson_result

    def _bootstrap_metric(
        self,
        fn,
        measured: np.ndarray,
        reference: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> MetricResult:
        """Compute bootstrap metric with confidence intervals."""
        from rppg_benchmark import bootstrap_metric
        return bootstrap_metric(fn, measured, reference, n_bootstrap, confidence)


def benchmark_video(
    video_path: str,
    gt_path: str,
    output_path: str,
    experiment_id: str,
    subject_id: Optional[str] = None,
    video_id: Optional[str] = None,
    dataset: str = "benchmark",
    alignment_tolerance: float = 2.0,
    max_frames: Optional[int] = None,
    verbose: bool = False,
) -> BenchmarkExperimentResult:
    """
    Execute benchmark on a prerecorded video.

    This is the primary public API for benchmark execution.

    Parameters
    ----------
    video_path : str
        Path to prerecorded video file.
    gt_path : str
        Path to ground truth CSV file with columns: timestamp, bpm.
    output_path : str
        Path for output JSON result file.
    experiment_id : str
        Unique identifier for this experiment.
    subject_id : Optional[str]
        Subject identifier for metadata.
    video_id : Optional[str]
        Video identifier for metadata.
    dataset : str
        Dataset name for result metadata.
    alignment_tolerance : float
        Maximum time difference (seconds) for GT matching. Default: 2.0.
    max_frames : Optional[int]
        Limit frames to process (for testing). Default: None (all frames).
    verbose : bool
        Print progress to stdout. Default: False.

    Returns
    -------
    BenchmarkExperimentResult
        Complete experiment results including metrics and alignment info.

    Raises
    ------
    ValueError
        If video or GT file cannot be loaded.
    """
    runner = BenchmarkRunner(
        video_path=video_path,
        gt_path=gt_path,
        experiment_id=experiment_id,
        alignment_tolerance=alignment_tolerance,
        dataset=dataset,
        subject_id=subject_id,
        video_id=video_id,
    )

    result = runner.run(max_frames=max_frames, verbose=verbose)

    # Save result
    result.save_json(output_path)

    return result
