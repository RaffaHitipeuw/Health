# -*- coding: utf-8 -*-
"""
Phase 2 Real-Data Benchmark - Complete Implementation

This script runs the full Phase 2 real-data benchmark when a real face video
and ground truth file are provided.

USAGE:
    python rppg_phase2_real_benchmark.py --video path/to/video.mp4 --gt path/to/gt.csv

REQUIREMENTS:
    1. Real face video (MP4, AVI, etc.) with actual human face
    2. Ground truth BPM file with timestamps
    3. miniconda3 Python with MediaPipe 0.10.35

INPUT FORMAT:
    Ground truth CSV:
        timestamp,bpm
        0.000,72.0
        0.500,71.5
        ...

PYTHON ENVIRONMENT:
    Must use miniconda3 Python for MediaPipe:
        /c/miniconda3/python.exe rppg_phase2_real_benchmark.py ...
"""

import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Change to repo directory
os.chdir('D:/main/Projects/Health')
sys.path.insert(0, os.getcwd())

# =============================================================================
# DEPENDENCY IMPORTS
# =============================================================================

# MediaPipe imports (from miniconda3)
try:
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision.core import vision_task_running_mode
    from mediapipe import Image, ImageFormat
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARN] MediaPipe not available. Install with: pip install mediapipe==0.10.35")

# V2 imports
try:
    from rppg_core import MultiROIFusionEngineV2
    from rppg_config import cfg
    HAS_RPPG = True
except ImportError:
    HAS_RPPG = False
    print("[WARN] Sanubari V2 not available")

# Classical methods
try:
    from rppg_algorithms import extract_green, extract_chrom, extract_pos, extract_ica
    HAS_CLASSICAL = True
except ImportError:
    HAS_CLASSICAL = False
    print("[WARN] Classical methods not available")

# =============================================================================
# V2 LANDMARK INDICES
# =============================================================================

FOREHEAD_LANDMARKS = [109, 67, 108, 151, 337, 297, 338]
CHEEK_LEFT_LANDMARKS = [50, 101, 118, 117, 116, 123, 147, 187, 207, 206, 205, 36, 142, 126, 100, 203]
CHEEK_RIGHT_LANDMARKS = [280, 330, 347, 346, 345, 352, 376, 411, 427, 426, 425, 266, 371, 355, 329, 423]
ALL_V2_LANDMARKS = FOREHEAD_LANDMARKS + CHEEK_LEFT_LANDMARKS + CHEEK_RIGHT_LANDMARKS

# =============================================================================
# LANDMARK ADAPTER CLASSES
# =============================================================================

class LandmarkPoint:
    """Single landmark point compatible with V2 format."""
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class LandmarkList:
    """Container compatible with V2 iteration."""
    __slots__ = ('_points',)
    def __init__(self, points):
        self._points = points
    @property
    def landmark(self):
        return self
    def __iter__(self):
        return iter(self._points)
    def __len__(self):
        return len(self._points)
    def __getitem__(self, key):
        return self._points[key]

# =============================================================================
# FACE LANDMARKER
# =============================================================================

class FaceLandmarkerWrapper:
    """MediaPipe FaceLandmarker wrapper for Sanubari V2."""

    def __init__(self, model_path: str = 'models/face_landmarker.task'):
        if not HAS_MEDIAPIPE:
            raise RuntimeError("MediaPipe not available")

        self.model_path = model_path
        options = FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._detections = 0
        self._misses = 0

    def process(self, rgb_frame: np.ndarray) -> Optional[LandmarkList]:
        """Detect face landmarks from RGB frame."""
        if rgb_frame is None or rgb_frame.size == 0:
            self._misses += 1
            return None

        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            self._misses += 1
            return None

        self._detections += 1
        landmarks = result.face_landmarks[0]

        # Convert to V2-compatible format
        return LandmarkList([
            LandmarkPoint(lm.x, lm.y, lm.z) for lm in landmarks
        ])

    def get_stats(self) -> Dict:
        """Get detection statistics."""
        total = self._detections + self._misses
        rate = self._detections / total if total > 0 else 0
        return {
            'detections': self._detections,
            'misses': self._misses,
            'total': total,
            'detection_rate': rate
        }

    def close(self):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

# =============================================================================
# GROUND TRUTH LOADER
# =============================================================================

def load_ground_truth(gt_path: str) -> Tuple[List[float], List[float]]:
    """
    Load ground truth from CSV file.

    Parameters
    ----------
    gt_path : str
        Path to ground truth CSV file with columns: timestamp, bpm

    Returns
    -------
    Tuple[List[float], List[float]]
        (timestamps, bpm_values)
    """
    timestamps = []
    bpm_values = []

    with open(gt_path, 'r') as f:
        # Skip header
        header = f.readline()
        if 'timestamp' not in header.lower() or 'bpm' not in header.lower():
            raise ValueError("GT file must have 'timestamp,bpm' header")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    bpm = float(parts[1])
                    timestamps.append(t)
                    bpm_values.append(bpm)
                except ValueError:
                    continue

    return timestamps, bpm_values

def validate_ground_truth(timestamps: List[float], bpm_values: List[float]) -> Dict:
    """Validate ground truth for physiological plausibility."""
    n = len(timestamps)

    # Check monotonic timestamps
    is_monotonic = all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)) if n > 1 else True

    # Check BPM range
    bpm_min = min(bpm_values) if bpm_values else 0
    bpm_max = max(bpm_values) if bpm_values else 0

    # Physiological plausible range (resting: 40-180 BPM)
    is_plausible = all(40 <= b <= 200 for b in bpm_values)

    # Duration
    duration = max(timestamps) - min(timestamps) if timestamps else 0

    return {
        'n_entries': n,
        'is_monotonic': is_monotonic,
        'bpm_min': bpm_min,
        'bpm_max': bpm_max,
        'bpm_mean': np.mean(bpm_values) if bpm_values else 0,
        'is_plausible': is_plausible,
        'duration': duration,
        'timestamp_min': min(timestamps) if timestamps else 0,
        'timestamp_max': max(timestamps) if timestamps else 0
    }

# =============================================================================
# ALIGNMENT
# =============================================================================

def align_predictions_to_gt(
    predictions: List[Dict],
    gt_timestamps: List[float],
    gt_bpm: List[float],
    tolerance: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Align predictions to ground truth using nearest-neighbor matching.

    Parameters
    ----------
    predictions : List[Dict]
        List of predictions with 'timestamp' and 'bpm' keys
    gt_timestamps : List[float]
        Ground truth timestamps
    gt_bpm : List[float]
        Ground truth BPM values
    tolerance : float
        Maximum time difference for alignment (seconds)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[Dict]]
        (aligned_pred_bpm, aligned_gt_bpm, alignment_info)
    """
    aligned_pred = []
    aligned_gt = []
    alignment_info = []

    gt_sorted = sorted(zip(gt_timestamps, gt_bpm), key=lambda x: x[0])

    for pred in predictions:
        pred_ts = pred['timestamp']
        pred_bpm = pred['bpm']

        if pred_bpm <= 0:
            continue

        # Find nearest GT
        best_dt = tolerance + 1
        best_gt_ts = None
        best_gt_bpm = None

        for gt_ts, gt_val in gt_sorted:
            dt = abs(pred_ts - gt_ts)
            if dt < best_dt:
                best_dt = dt
                best_gt_ts = gt_ts
                best_gt_bpm = gt_val

        if best_dt <= tolerance:
            aligned_pred.append(pred_bpm)
            aligned_gt.append(best_gt_bpm)
            alignment_info.append({
                'pred_ts': pred_ts,
                'gt_ts': best_gt_ts,
                'pred_bpm': pred_bpm,
                'gt_bpm': best_gt_bpm,
                'dt': best_dt,
                'error': abs(pred_bpm - best_gt_bpm)
            })

    return np.array(aligned_pred), np.array(aligned_gt), alignment_info

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict:
    """Compute benchmark metrics."""
    if len(pred) < 3:
        return {'error': 'Insufficient aligned samples'}

    # MAE
    mae = np.mean(np.abs(pred - gt))

    # RMSE
    rmse = np.sqrt(np.mean((pred - gt)**2))

    # Bias
    bias = np.mean(pred - gt)

    # Pearson correlation
    if len(pred) > 2:
        corr = np.corrcoef(pred, gt)[0, 1]
    else:
        corr = None

    # Standard deviation
    pred_std = np.std(pred)
    gt_std = np.std(gt)

    return {
        'n_aligned': len(pred),
        'mae': float(mae),
        'rmse': float(rmse),
        'bias': float(bias),
        'pearson_r': float(corr) if corr is not None else None,
        'pred_std': float(pred_std),
        'gt_std': float(gt_std)
    }

def compute_bootstrap_ci(
    metric_fn,
    pred: np.ndarray,
    gt: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric."""
    n = len(pred)
    if n < 10:
        return None, None, None

    values = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        values.append(metric_fn(pred[idx], gt[idx]))

    values = np.array(values)
    alpha = 1 - confidence
    lower = np.percentile(values, alpha/2 * 100)
    upper = np.percentile(values, (1 - alpha/2) * 100)
    estimate = np.mean(values)

    return estimate, lower, upper

# =============================================================================
# VIDEO PROCESSOR
# =============================================================================

class VideoProcessor:
    """Process video and extract rPPG signals."""

    def __init__(
        self,
        landmarker: FaceLandmarkerWrapper,
        fusion_engine: MultiROIFusionEngineV2,
        methods: List[str] = None
    ):
        self.landmarker = landmarker
        self.fusion_engine = fusion_engine
        self.methods = methods or ['V2']

        # Buffers for classical methods
        self.r_buffer = []
        self.g_buffer = []
        self.b_buffer = []
        self.timestamps = []

        # ROI parameters
        self.roi_size = 50  # pixels

    def process_frame(
        self,
        frame: np.ndarray,
        landmarks: LandmarkList,
        timestamp: float,
        h: int, w: int
    ) -> Dict:
        """Process single frame."""
        result = self.fusion_engine.update(frame, landmarks, h, w)

        # Extract RGB from ROIs
        rgb = self._extract_roi_rgb(frame, landmarks, h, w)
        if rgb is not None:
            self.r_buffer.append(rgb[0])
            self.g_buffer.append(rgb[1])
            self.b_buffer.append(rgb[2])
            self.timestamps.append(timestamp)

        return {
            'timestamp': timestamp,
            'bpm': result.fused_bpm,
            'sqi': result.fused_sqi,
            'confidence': result.session_confidence
        }

    def _extract_roi_rgb(
        self,
        frame: np.ndarray,
        landmarks: LandmarkList,
        h: int, w: int
    ) -> Optional[np.ndarray]:
        """Extract mean RGB from forehead ROI."""
        forehead_pts = []
        for idx in FOREHEAD_LANDMARKS:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            forehead_pts.append((x, y))

        if not forehead_pts:
            return None

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(forehead_pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # Extract mean RGB
        mean_b = np.mean(frame[:, :, 0][mask > 0])
        mean_g = np.mean(frame[:, :, 1][mask > 0])
        mean_r = np.mean(frame[:, :, 2][mask > 0])

        return np.array([mean_r, mean_g, mean_b])

    def get_v2_predictions(self) -> List[Dict]:
        """Get V2 predictions from fusion engine."""
        # Re-run to collect predictions
        return []

    def get_classical_predictions(self) -> Dict[str, List[float]]:
        """Extract classical method signals from buffers."""
        if len(self.r_buffer) < 128:
            return {}

        r = np.array(self.r_buffer)
        g = np.array(self.g_buffer)
        b = np.array(self.b_buffer)

        results = {}
        for method in self.methods:
            if method == 'GREEN':
                signal = extract_green(r, g, b)
                results['GREEN'] = signal
            elif method == 'CHROM':
                signal = extract_chrom(r, g, b)
                results['CHROM'] = signal
            elif method == 'POS':
                signal = extract_pos(r, g, b)
                results['POS'] = signal
            elif method == 'ICA':
                signal = extract_ica(r, g, b)
                results['ICA'] = signal

        return results

# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(
    video_path: str,
    gt_path: str,
    output_dir: str = 'benchmark_results',
    methods: List[str] = None,
    alignment_tolerance: float = 2.0,
    max_frames: int = None
) -> Dict:
    """
    Run complete Phase 2 real-data benchmark.

    Parameters
    ----------
    video_path : str
        Path to input video file
    gt_path : str
        Path to ground truth CSV file
    output_dir : str
        Output directory for results
    methods : List[str]
        Methods to evaluate (default: V2)
    alignment_tolerance : float
        GT alignment tolerance in seconds
    max_frames : int, optional
        Maximum frames to process

    Returns
    -------
    Dict
        Complete benchmark results
    """
    print("=" * 70)
    print("PHASE 2 REAL-DATA BENCHMARK")
    print("=" * 70)
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth
    print("[1/6] Loading ground truth...")
    gt_timestamps, gt_bpm = load_ground_truth(gt_path)
    gt_validation = validate_ground_truth(gt_timestamps, gt_bpm)
    print("  GT entries: {}".format(gt_validation['n_entries']))
    print("  Duration: {:.1f}s".format(gt_validation['duration']))
    print("  BPM range: {:.1f} - {:.1f}".format(gt_validation['bpm_min'], gt_validation['bpm_max']))
    print("  Plausible: {}".format("YES" if gt_validation['is_plausible'] else "NO"))
    print()

    # Initialize components
    print("[2/6] Initializing components...")

    landmarker = FaceLandmarkerWrapper()
    print("  FaceLandmarker: OK")

    fusion_engine = MultiROIFusionEngineV2(enable_logging=False)
    print("  FusionEngineV2: OK")

    # Open video
    print("[3/6] Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: {}".format(video_path))

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_frame_count / video_fps if video_fps > 0 else 0

    print("  Video: {}x{} @ {:.1f} fps".format(video_width, video_height, video_fps))
    print("  Frames: {}, Duration: {:.1f}s".format(video_frame_count, video_duration))
    print()

    # Process video
    print("[4/6] Processing video...")
    predictions = []
    frame_idx = 0
    face_detected = 0
    face_missed = 0

    start_time = time.time()

    while True:
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        timestamp = frame_idx / video_fps

        # Convert to RGB and detect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = landmarker.process(frame_rgb)

        if landmarks is None:
            face_missed += 1
            continue

        face_detected += 1

        # Process through V2
        result = fusion_engine.update(frame, landmarks, h, w)

        predictions.append({
            'frame': frame_idx,
            'timestamp': timestamp,
            'bpm': result.fused_bpm,
            'sqi': result.fused_sqi,
            'valid': result.fused_bpm > 0
        })

        frame_idx += 1

        if frame_idx % 100 == 0:
            print("  Processed {}/{} frames".format(frame_idx, video_frame_count))

    cap.release()
    process_time = time.time() - start_time

    print("  Total frames: {}".format(frame_idx))
    print("  Face detected: {} ({:.1f}%)".format(face_detected, 100*face_detected/(face_detected+face_missed)))
    print("  Processing time: {:.1f}s".format(process_time))
    print()

    # Align predictions to GT
    print("[5/6] Aligning predictions to ground truth...")
    valid_predictions = [p for p in predictions if p['valid']]

    aligned_pred, aligned_gt, alignment_info = align_predictions_to_gt(
        valid_predictions, gt_timestamps, gt_bpm, alignment_tolerance
    )

    print("  Valid predictions: {}".format(len(valid_predictions)))
    print("  Aligned pairs: {}".format(len(aligned_pred)))
    print()

    # Compute metrics
    print("[6/6] Computing metrics...")
    metrics = compute_metrics(aligned_pred, aligned_gt)

    print("  MAE: {:.2f} BPM".format(metrics['mae']))
    print("  RMSE: {:.2f} BPM".format(metrics['rmse']))
    print("  Bias: {:.2f} BPM".format(metrics['bias']))
    if metrics['pearson_r'] is not None:
        print("  Pearson r: {:.4f}".format(metrics['pearson_r']))
    print()

    # Compile results
    results = {
        'experiment_id': 'phase2_realdatabenchmark',
        'timestamp': datetime.now().isoformat(),
        'video': {
            'path': video_path,
            'fps': video_fps,
            'width': video_width,
            'height': video_height,
            'frame_count': frame_idx,
            'duration': video_duration
        },
        'ground_truth': {
            'path': gt_path,
            'validation': gt_validation
        },
        'processing': {
            'frames_processed': frame_idx,
            'face_detected': face_detected,
            'face_missed': face_missed,
            'face_detection_rate': face_detected / (face_detected + face_missed) if (face_detected + face_missed) > 0 else 0,
            'valid_predictions': len(valid_predictions),
            'aligned_pairs': len(aligned_pred),
            'alignment_tolerance': alignment_tolerance,
            'processing_time': process_time,
            'processing_fps': frame_idx / process_time if process_time > 0 else 0
        },
        'metrics': metrics,
        'methods': {
            'V2': {
                'name': 'Sanubari V2',
                'metrics': metrics
            }
        }
    }

    # Save results
    output_file = os.path.join(output_dir, 'phase2_realdata_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to: {}".format(output_file))

    landmarker.close()

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2 Real-Data Benchmark')
    parser.add_argument('--video', '-v', required=True, help='Input video path')
    parser.add_argument('--gt', '-g', required=True, help='Ground truth CSV path')
    parser.add_argument('--output', '-o', default='benchmark_results', help='Output directory')
    parser.add_argument('--methods', '-m', nargs='+', default=['V2'], help='Methods to evaluate')
    parser.add_argument('--tolerance', '-t', type=float, default=2.0, help='Alignment tolerance (s)')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')

    args = parser.parse_args()

    results = run_benchmark(
        video_path=args.video,
        gt_path=args.gt,
        output_dir=args.output,
        methods=args.methods,
        alignment_tolerance=args.tolerance,
        max_frames=args.max_frames
    )
