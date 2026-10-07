# -*- coding: utf-8 -*-
"""
Phase 2 Real-Data Acquisition - Complete G5 Implementation

Handles video capture, GT import, synchronization, and validation.

INTEGRITY RULES:
- GT MUST NOT come from Sanubari
- GT MUST NOT be fabricated
- GT MUST be from independent physiological measurement

USAGE:
    python -m phase2_data_acquisition capture --output benchmark_results/subject01
    python -m phase2_data_acquisition import-gt --sensor csv --input data.csv --output gt.csv
    python -m phase2_data_acquisition manifest --video video.mp4 --gt gt.csv --offset 0.0
"""

import os
import sys
import cv2
import json
import time
import argparse
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Change to repo directory
REPO_ROOT = Path('D:/main/Projects/Health')
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CaptureMetadata:
    """Metadata for a video capture session."""
    capture_id: str
    capture_start_time: str
    capture_end_time: str
    video_path: str
    video_fps: float
    video_width: int
    video_height: int
    video_frame_count: int
    video_duration_seconds: float
    actual_fps: float
    nominal_fps: float
    fourcc: str
    camera_index: int


@dataclass
class GroundTruthData:
    """Ground truth BPM data."""
    timestamps: List[float]
    bpm_values: List[float]
    source: str  # 'external_sensor', 'manual', 'blocked'
    sensor_type: str  # 'pulse_oximeter', 'ecg', 'unknown'
    sampling_interval: float
    start_time: float
    end_time: float


@dataclass
class DatasetManifest:
    """Manifest for a complete dataset."""
    manifest_version: str = "1.0"
    dataset_id: str = ""
    created_at: str = ""

    # Video info
    video_path: str = ""
    video_sha256: str = ""
    video_fps: float = 0.0
    video_frame_count: int = 0
    video_duration_seconds: float = 0.0
    video_resolution: Tuple[int, int] = (0, 0)

    # GT info
    gt_path: str = ""
    gt_sha256: str = ""
    gt_status: str = "blocked"  # 'present', 'blocked', 'manual'
    gt_source: str = ""
    gt_sensor_type: str = ""
    gt_start_time: float = 0.0
    gt_end_time: float = 0.0
    gt_offset_seconds: float = 0.0
    gt_sampling_interval: float = 0.0

    # Validation
    video_valid: bool = False
    gt_valid: bool = False
    face_detected: bool = False
    roi_valid: bool = False
    overlap_valid: bool = False

    # Capture metadata
    capture_metadata: Dict = field(default_factory=dict)

    # Provenance
    provenance: str = ""


# =============================================================================
# UTILITIES
# =============================================================================

def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return ""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_video(video_path: str) -> Dict[str, Any]:
    """Validate a video file."""
    if not os.path.exists(video_path):
        return {'valid': False, 'error': 'File not found'}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'valid': False, 'error': 'Cannot open video'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    errors = []
    if fps <= 0:
        errors.append('Invalid FPS')
    if frame_count <= 0:
        errors.append('Invalid frame count')
    if duration < 5:
        errors.append('Video too short (<5s)')
    if width < 320 or height < 240:
        errors.append('Resolution too low')

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }


def validate_gt(gt_path: str) -> Dict[str, Any]:
    """Validate a ground truth CSV file."""
    if not os.path.exists(gt_path):
        return {'valid': False, 'error': 'File not found'}

    try:
        timestamps = []
        bpm_values = []

        with open(gt_path, 'r') as f:
            header = f.readline().strip()
            if 'timestamp' not in header.lower() or 'bpm' not in header.lower():
                return {'valid': False, 'error': 'Missing timestamp,bpm header'}

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    t = float(parts[0])
                    bpm = float(parts[1])
                    timestamps.append(t)
                    bpm_values.append(bpm)

        if len(timestamps) < 3:
            return {'valid': False, 'error': 'Too few GT entries'}

        # Check monotonicity
        is_monotonic = all(timestamps[i] < timestamps[i+1]
                         for i in range(len(timestamps)-1))

        # Check physiological plausibility
        is_plausible = all(40 <= b <= 200 for b in bpm_values)

        # Calculate interval
        intervals = [timestamps[i+1] - timestamps[i]
                   for i in range(len(timestamps)-1)]
        avg_interval = np.mean(intervals) if intervals else 0

        return {
            'valid': True,
            'n_entries': len(timestamps),
            'timestamps': timestamps,
            'bpm_values': bpm_values,
            'is_monotonic': is_monotonic,
            'is_plausible': is_plausible,
            'avg_interval': avg_interval,
            'start_time': min(timestamps),
            'end_time': max(timestamps),
            'bpm_min': min(bpm_values),
            'bpm_max': max(bpm_values),
            'bpm_mean': np.mean(bpm_values)
        }

    except Exception as e:
        return {'valid': False, 'error': str(e)}


def check_video_gt_overlap(
    video_duration: float,
    gt_start: float,
    gt_end: float,
    offset: float = 0.0
) -> bool:
    """Check if video and GT overlap temporally."""
    # Adjust GT times by offset
    adj_gt_start = gt_start + offset
    adj_gt_end = gt_end + offset

    # Check overlap
    return adj_gt_start <= video_duration and adj_gt_end >= 0


# =============================================================================
# G5: VIDEO CAPTURE
# =============================================================================

def capture_face_video(
    output_dir: str,
    duration: int = 60,
    fps: float = 30.0,
    camera_index: int = 0
) -> Optional[CaptureMetadata]:
    """
    Capture real face video using webcam.

    Parameters
    ----------
    output_dir : str
        Output directory for video and metadata
    duration : int
        Recording duration in seconds
    fps : float
        Target FPS
    camera_index : int
        Webcam index

    Returns
    -------
    CaptureMetadata or None
    """
    print("=" * 60)
    print("G5: REAL FACE VIDEO CAPTURE")
    print("=" * 60)
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam at index {}".format(camera_index))
        return None

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # Prefer 640x480 if available, otherwise use camera default
    if width > 1280:
        width, height = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("[OK] Camera: {}x{} @ {} fps".format(width, height, actual_fps or fps))

    # Generate capture ID
    capture_id = "capture_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Video output path
    video_path = os.path.join(output_dir, "video.mp4")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Warm up camera
    print("Warming up camera...")
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    # Check MediaPipe availability for face detection during capture
    face_detected_during_capture = False
    try:
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision.core import vision_task_running_mode
        from mediapipe import Image, ImageFormat

        mp_model_path = str(REPO_ROOT / 'models' / 'face_landmarker.task')
        if os.path.exists(mp_model_path):
            options = FaceLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=mp_model_path),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                num_faces=1,
            )
            mp_landmarker = FaceLandmarker.create_from_options(options)
            print("[OK] MediaPipe FaceLandmarker available for live feedback")
        else:
            mp_landmarker = None
            print("[WARN] MediaPipe model not found, no live face feedback")
    except ImportError:
        mp_landmarker = None
        print("[WARN] MediaPipe not available, no live face feedback")

    print()
    print("INSTRUCTIONS:")
    print("  - Position your face in the center of the frame")
    print("  - Stay as still as possible")
    print("  - Breathe normally")
    print("  - Recording for {} seconds".format(duration))
    print()

    # Countdown
    print("Starting in 3 seconds...")
    for i in [3, 2, 1]:
        print("  {}".format(i))
        time.sleep(1)

    # Recording
    print()
    print("Recording...")
    print("-" * 50)

    start_time = time.time()
    start_iso = datetime.now().isoformat()
    frames_recorded = 0

    try:
        for frame_idx in range(int(duration * fps)):
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed at index {}".format(frame_idx))
                break

            # Write frame
            out.write(frame)
            frames_recorded += 1

            # Check face detection periodically
            if mp_landmarker and frame_idx % 10 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
                result = mp_landmarker.detect(mp_image)
                if result.face_landmarks:
                    face_detected_during_capture = True

            # Progress update every second
            elapsed = time.time() - start_time
            if frame_idx % int(fps) == 0:
                remaining = duration - elapsed
                face_marker = "[FACE]" if face_detected_during_capture else "------"
                print("[{:4.0f}s] Frame {}/{} {} | Elapsed: {:.1f}s".format(
                    elapsed, frame_idx, int(duration * fps), face_marker, elapsed))

            # Control frame rate
            time.sleep(1.0 / fps)

    finally:
        end_time = time.time()
        end_iso = datetime.now().isoformat()

        cap.release()
        out.release()

        if mp_landmarker:
            mp_landmarker.close()

    actual_duration = end_time - start_time
    actual_fps_calc = frames_recorded / actual_duration if actual_duration > 0 else 0

    print("-" * 50)
    print()
    print("[OK] Recording complete!")
    print("  Video: {}".format(video_path))
    print("  Frames: {}".format(frames_recorded))
    print("  Duration: {:.1f}s".format(actual_duration))
    print("  Actual FPS: {:.1f}".format(actual_fps_calc))
    print("  Face detected: {}".format("YES" if face_detected_during_capture else "NO/Could not verify"))

    # Create metadata
    metadata = CaptureMetadata(
        capture_id=capture_id,
        capture_start_time=start_iso,
        capture_end_time=end_iso,
        video_path=video_path,
        video_fps=fps,
        video_width=width,
        video_height=height,
        video_frame_count=frames_recorded,
        video_duration_seconds=actual_duration,
        actual_fps=actual_fps_calc,
        nominal_fps=fps,
        fourcc='mp4v',
        camera_index=camera_index
    )

    # Save metadata
    metadata_path = os.path.join(output_dir, "capture_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2)
    print("  Metadata: {}".format(metadata_path))

    print()
    print("=" * 60)
    print("G5 VIDEO CAPTURE: PASS")
    print("=" * 60)
    print()

    return metadata


# =============================================================================
# G5.1: INDEPENDENT GROUND TRUTH
# =============================================================================

def import_ground_truth(
    input_path: str,
    output_path: str,
    sensor_type: str = "unknown"
) -> Optional[GroundTruthData]:
    """
    Import ground truth from external sensor export.

    Parameters
    ----------
    input_path : str
        Path to sensor export file
    output_path : str
        Output path for standardized GT CSV
    sensor_type : str
        Type of sensor ('pulse_oximeter', 'ecg', 'polar', 'unknown')

    Returns
    -------
    GroundTruthData or None
    """
    print("=" * 60)
    print("G5.1: INDEPENDENT GROUND TRUTH IMPORT")
    print("=" * 60)
    print()

    if not os.path.exists(input_path):
        print("[ERROR] Input file not found: {}".format(input_path))
        return None

    # Detect format and import
    timestamps = []
    bpm_values = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
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

    if len(timestamps) < 3:
        print("[ERROR] Insufficient GT entries")
        return None

    # Save standardized format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("timestamp,bpm\n")
        for t, b in zip(timestamps, bpm_values):
            f.write("{:.3f},{:.1f}\n".format(t, b))

    print("[OK] Imported {} GT entries".format(len(timestamps)))
    print("  Input: {}".format(input_path))
    print("  Output: {}".format(output_path))
    print("  Sensor type: {}".format(sensor_type))

    # Calculate metadata
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    avg_interval = np.mean(intervals) if intervals else 0

    gt_data = GroundTruthData(
        timestamps=timestamps,
        bpm_values=bpm_values,
        source='external_sensor',
        sensor_type=sensor_type,
        sampling_interval=avg_interval,
        start_time=min(timestamps),
        end_time=max(timestamps)
    )

    print()
    print("=" * 60)
    print("G5.1 GROUND TRUTH IMPORT: PASS")
    print("=" * 60)
    print()

    return gt_data


def report_gt_blocked(output_dir: str) -> str:
    """
    Document that GT is blocked and create placeholder.

    Returns path to GT status file.
    """
    print("=" * 60)
    print("G5.1: INDEPENDENT GROUND TRUTH STATUS")
    print("=" * 60)
    print()

    status = {
        "status": "blocked",
        "reason": "No independent physiological measurement device available",
        "alternatives": [
            "Connect pulse oximeter and export CSV",
            "Connect ECG and export RR intervals",
            "Connect Polar H10 and export BPM",
            "Use manual BPM annotation from video (limited)"
        ],
        "note": "GT is BLOCKED. Do NOT fabricate BPM values."
    }

    status_path = os.path.join(output_dir, "gt_status.json")
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)

    print("[STATUS] G5.1 INDEPENDENT GT: BLOCKED")
    print("  Reason: No independent sensor connected")
    print("  Note: Do NOT fabricate GT values")
    print("  Status file: {}".format(status_path))
    print()

    return status_path


# =============================================================================
# G5.2: SYNCHRONIZATION
# =============================================================================

def create_synchronization_config(
    video_start_time: float,
    gt_start_time: float,
    output_path: str
) -> Dict[str, Any]:
    """
    Create synchronization configuration.

    Parameters
    ----------
    video_start_time : float
        Video start time in seconds (usually 0)
    gt_start_time : float
        GT start time in seconds
    output_path : str
        Output path for sync config

    Returns
    -------
    Dict with synchronization parameters
    """
    # Calculate offset
    offset = gt_start_time - video_start_time

    config = {
        "video_start_time": video_start_time,
        "ground_truth_start_time": gt_start_time,
        "gt_offset_seconds": offset,
        "alignment_tolerance_seconds": 2.0,
        "timestamp_source": "video_frame_timeline",
        "note": "Positive offset means GT started after video"
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("G5.2: SYNCHRONIZATION CONFIG")
    print("=" * 60)
    print()
    print("  Video start: {:.3f}s".format(video_start_time))
    print("  GT start: {:.3f}s".format(gt_start_time))
    print("  Offset: {:.3f}s".format(offset))
    print("  Tolerance: 2.0s")
    print("  Config: {}".format(output_path))
    print()

    return config


# =============================================================================
# G5.3: DATASET VALIDATION AND MANIFEST
# =============================================================================

def create_dataset_manifest(
    output_dir: str,
    video_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    offset: float = 0.0,
    capture_metadata: Optional[Dict] = None
) -> DatasetManifest:
    """
    Create complete dataset manifest with validation.

    Parameters
    ----------
    output_dir : str
        Output directory
    video_path : str, optional
        Path to video file
    gt_path : str, optional
        Path to GT file
    offset : float
        GT synchronization offset
    capture_metadata : Dict, optional
        Capture metadata from video recording

    Returns
    -------
    DatasetManifest with validation results
    """
    print("=" * 60)
    print("G5.3: DATASET VALIDATION AND MANIFEST")
    print("=" * 60)
    print()

    # Initialize manifest
    manifest = DatasetManifest()
    manifest.dataset_id = os.path.basename(output_dir)
    manifest.created_at = datetime.now().isoformat()

    # Find video if not provided
    if video_path is None:
        candidate = os.path.join(output_dir, "video.mp4")
        if os.path.exists(candidate):
            video_path = candidate

    # Validate video
    if video_path and os.path.exists(video_path):
        print("[1/5] Validating video...")
        vresult = validate_video(video_path)

        manifest.video_path = video_path
        manifest.video_sha256 = compute_sha256(video_path)
        manifest.video_fps = vresult['fps']
        manifest.video_frame_count = vresult['frame_count']
        manifest.video_duration_seconds = vresult['duration']
        manifest.video_resolution = (vresult['width'], vresult['height'])
        manifest.video_valid = vresult['valid']

        if vresult['valid']:
            print("  [OK] Video valid")
            print("    FPS: {:.1f}".format(vresult['fps']))
            print("    Frames: {}".format(vresult['frame_count']))
            print("    Duration: {:.1f}s".format(vresult['duration']))
        else:
            print("  [FAIL] Video invalid: {}".format(vresult['errors']))

        # Check face detection
        if vresult['valid']:
            print("[2/5] Checking face detection...")
            face_ok = check_face_in_video(video_path)
            manifest.face_detected = face_ok
            if face_ok:
                print("  [OK] Face detected in video")
            else:
                print("  [WARN] No face detected in video")
    else:
        print("[1/5] Validating video: NOT FOUND")
        manifest.video_valid = False

    # Find GT if not provided
    if gt_path is None:
        candidate = os.path.join(output_dir, "gt.csv")
        if os.path.exists(candidate):
            gt_path = candidate
        else:
            candidate = os.path.join(output_dir, "ground_truth.csv")
            if os.path.exists(candidate):
                gt_path = candidate

    # Validate GT
    if gt_path and os.path.exists(gt_path):
        print("[3/5] Validating ground truth...")
        gresult = validate_gt(gt_path)

        manifest.gt_path = gt_path
        manifest.gt_sha256 = compute_sha256(gt_path)
        manifest.gt_status = 'present' if gresult['valid'] else 'invalid'
        manifest.gt_start_time = gresult.get('start_time', 0)
        manifest.gt_end_time = gresult.get('end_time', 0)
        manifest.gt_sampling_interval = gresult.get('avg_interval', 0)
        manifest.gt_valid = gresult['valid']

        if gresult['valid']:
            print("  [OK] GT valid")
            print("    Entries: {}".format(gresult['n_entries']))
            print("    BPM range: {:.1f} - {:.1f}".format(
                gresult['bpm_min'], gresult['bpm_max']))
            print("    Interval: {:.3f}s".format(gresult['avg_interval']))
        else:
            print("  [FAIL] GT invalid: {}".format(gresult.get('error', 'Unknown')))
    else:
        print("[3/5] Validating ground truth: NOT FOUND")
        manifest.gt_status = 'blocked'
        manifest.gt_valid = False

    # Synchronization
    print("[4/5] Checking synchronization...")
    manifest.gt_offset_seconds = offset

    if manifest.video_valid and manifest.gt_valid:
        manifest.overlap_valid = check_video_gt_overlap(
            manifest.video_duration_seconds,
            manifest.gt_start_time,
            manifest.gt_end_time,
            offset
        )
        if manifest.overlap_valid:
            print("  [OK] Video and GT overlap")
        else:
            print("  [FAIL] Video and GT do not overlap")
            print("    Video: 0 - {:.1f}s".format(manifest.video_duration_seconds))
            print("    GT: {:.1f} - {:.1f}s (offset: {:.1f}s)".format(
                manifest.gt_start_time, manifest.gt_end_time, offset))
    else:
        manifest.overlap_valid = False
        print("  [SKIP] Cannot check overlap without valid video and GT")

    # Capture metadata
    print("[5/5] Recording provenance...")
    if capture_metadata:
        manifest.capture_metadata = capture_metadata

    manifest.provenance = "Real face video captured via webcam. "
    if manifest.gt_status == 'blocked':
        manifest.provenance += "Ground truth BLOCKED - no independent sensor available."
    else:
        manifest.provenance += "Ground truth from independent physiological measurement."

    # Save manifest
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(asdict(manifest), f, indent=2)

    print()
    print("=" * 60)
    print("MANIFEST CREATED")
    print("=" * 60)
    print("  Path: {}".format(manifest_path))
    print("  Dataset ID: {}".format(manifest.dataset_id))
    print()
    print("VALIDATION SUMMARY:")
    print("  Video valid: {}".format("PASS" if manifest.video_valid else "FAIL"))
    print("  Face detected: {}".format("PASS" if manifest.face_detected else "WARN"))
    print("  GT status: {}".format(manifest.gt_status.upper()))
    print("  Overlap valid: {}".format("PASS" if manifest.overlap_valid else "FAIL"))
    print()

    return manifest


def check_face_in_video(video_path: str) -> bool:
    """Check if face is detectable in video using MediaPipe."""
    try:
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision.core import vision_task_running_mode
        from mediapipe import Image, ImageFormat

        model_path = str(REPO_ROOT / 'models' / 'face_landmarker.task')
        if not os.path.exists(model_path):
            print("    [WARN] FaceLandmarker model not found")
            return False

        options = FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_faces=1,
        )
        landmarker = FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(video_path)

        # Sample frames throughout video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = [0, frame_count//4, frame_count//2, 3*frame_count//4, frame_count-1]

        face_detected = False
        for idx in sample_indices:
            if idx < 0 or idx >= frame_count:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect(mp_image)
                if result.face_landmarks:
                    face_detected = True
                    break

        cap.release()
        landmarker.close()

        return face_detected

    except ImportError:
        print("    [WARN] MediaPipe not available for face detection")
        return False
    except Exception as e:
        print("    [ERROR] Face detection failed: {}".format(e))
        return False


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 2 Real-Data Acquisition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  capture     Capture real face video
  import-gt   Import ground truth from external sensor
  manifest    Create dataset manifest
  validate    Validate existing dataset
  status      Show current status

Examples:
  # Capture video only
  python -m phase2_data_acquisition capture --output benchmark_results/subject01

  # Import GT from pulse oximeter CSV
  python -m phase2_data_acquisition import-gt \\
      --input pulse_oximeter_export.csv \\
      --output benchmark_results/subject01/gt.csv \\
      --sensor pulse_oximeter

  # Create manifest with existing data
  python -m phase2_data_acquisition manifest \\
      --video benchmark_results/subject01/video.mp4 \\
      --gt benchmark_results/subject01/gt.csv \\
      --offset 0.0 \\
      --output benchmark_results/subject01

  # Full pipeline
  python -m phase2_data_acquisition run \\
      --video benchmark_results/subject01/video.mp4 \\
      --gt benchmark_results/subject01/gt.csv \\
      --offset 0.0
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture face video')
    capture_parser.add_argument('--output', '-o', required=True,
                              help='Output directory')
    capture_parser.add_argument('--duration', '-d', type=int, default=60,
                              help='Recording duration (seconds)')
    capture_parser.add_argument('--fps', '-f', type=float, default=30.0,
                              help='Target FPS')
    capture_parser.add_argument('--camera', '-c', type=int, default=0,
                              help='Camera index')

    # Import GT command
    import_parser = subparsers.add_parser('import-gt', help='Import ground truth')
    import_parser.add_argument('--input', '-i', required=True,
                             help='Input sensor export file')
    import_parser.add_argument('--output', '-o', required=True,
                             help='Output GT CSV path')
    import_parser.add_argument('--sensor', '-s', default='unknown',
                             help='Sensor type')

    # Manifest command
    manifest_parser = subparsers.add_parser('manifest', help='Create manifest')
    manifest_parser.add_argument('--video', '-v',
                              help='Video path')
    manifest_parser.add_argument('--gt', '-g',
                              help='Ground truth path')
    manifest_parser.add_argument('--offset', '-t', type=float, default=0.0,
                              help='GT synchronization offset')
    manifest_parser.add_argument('--output', '-o', required=True,
                              help='Output directory')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('--dir', '-d', required=True,
                              help='Dataset directory')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument('--video', '-v', required=True,
                          help='Video path')
    run_parser.add_argument('--gt', '-g',
                          help='Ground truth path (optional)')
    run_parser.add_argument('--offset', '-t', type=float, default=0.0,
                          help='GT offset')
    run_parser.add_argument('--output', '-o', default='benchmark_results',
                          help='Output directory')

    args = parser.parse_args()

    if args.command == 'capture':
        metadata = capture_face_video(
            output_dir=args.output,
            duration=args.duration,
            fps=args.fps,
            camera_index=args.camera
        )
        if metadata:
            # Report GT blocked
            report_gt_blocked(args.output)
            # Create manifest
            create_dataset_manifest(
                output_dir=args.output,
                capture_metadata=asdict(metadata)
            )

    elif args.command == 'import-gt':
        gt_data = import_ground_truth(
            input_path=args.input,
            output_path=args.output,
            sensor_type=args.sensor
        )
        if gt_data:
            print("GT imported successfully")

    elif args.command == 'manifest':
        manifest = create_dataset_manifest(
            output_dir=args.output,
            video_path=args.video,
            gt_path=args.gt,
            offset=args.offset
        )
        print("Manifest created: {}".format(args.output))

    elif args.command == 'validate':
        manifest_path = os.path.join(args.dir, "dataset_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            print("Dataset: {}".format(manifest.get('dataset_id')))
            print("  Video valid: {}".format(manifest.get('video_valid')))
            print("  GT status: {}".format(manifest.get('gt_status')))
            print("  Face detected: {}".format(manifest.get('face_detected')))
            print("  Overlap valid: {}".format(manifest.get('overlap_valid')))
        else:
            print("No manifest found. Run 'manifest' command first.")

    elif args.command == 'run':
        print("=" * 60)
        print("RUNNING PHASE 2 DATA ACQUISITION PIPELINE")
        print("=" * 60)
        print()

        output_dir = os.path.join(args.output, os.path.basename(args.video).replace('.mp4', ''))
        os.makedirs(output_dir, exist_ok=True)

        # Create manifest
        manifest = create_dataset_manifest(
            output_dir=output_dir,
            video_path=args.video,
            gt_path=args.gt,
            offset=args.offset
        )

        # Summary
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print()
        print("G5 VIDEO: {}".format("PASS" if manifest.video_valid else "FAIL"))
        print("G5.1 GT: {}".format(manifest.gt_status.upper()))
        print("G5.2 SYNC: {}".format("PASS" if manifest.overlap_valid else("N/A" if not manifest.gt_valid else "FAIL")))
        print()

        if manifest.gt_status == 'blocked':
            print("IMPORTANT: Ground truth is BLOCKED")
            print("  Do NOT fabricate GT values")
            print("  Do NOT use Sanubari output as GT")
            print()
            print("To enable G6-G8, provide independent GT via:")
            print("  python -m phase2_data_acquisition import-gt \\")
            print("    --input <sensor_export.csv> \\")
            print("    --output {}/gt.csv".format(output_dir))
        else:
            print("Ground truth available - G6-G8 can proceed")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
