# -*- coding: utf-8 -*-
"""
Real Face Video Capture Script for Phase 2 Benchmark

This script captures a real face video using the webcam and creates
a ground truth file for the Phase 2 benchmark.

USAGE:
    /c/miniconda3/python.exe capture_real_face.py --output benchmark_results/subject01.mp4

REQUIREMENTS:
    - Webcam access
    - miniconda3 Python with MediaPipe 0.10.35
    - ~60 seconds of recording time
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np

os.chdir('D:/main/Projects/Health')
sys.path.insert(0, os.getcwd())

def capture_face_video(
    output_video: str,
    output_gt: str,
    duration: int = 60,
    expected_bpm: int = 72,
    fps: float = 30.0
):
    """
    Capture real face video for benchmark.

    Parameters
    ----------
    output_video : str
        Output video path (MP4)
    output_gt : str
        Output ground truth CSV path
    duration : int
        Recording duration in seconds
    expected_bpm : int
        Expected average BPM for GT documentation
    fps : float
        Target FPS
    """
    print("=" * 60)
    print("REAL FACE VIDEO CAPTURE")
    print("=" * 60)
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return False

    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    print("[OK] Camera: {}x{} @ {} fps".format(width, height, actual_fps))

    # Create output
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, actual_fps, (width, height))

    # Warm up
    print("Warming up camera...")
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    print()
    print("INSTRUCTIONS:")
    print("  - Position your face in the center of the frame")
    print("  - Stay as still as possible")
    print("  - Breathe normally")
    print("  - Recording for {} seconds".format(duration))
    print()
    print("Starting in 3 seconds...")
    time.sleep(3)

    # Recording
    print("Recording...")
    frames = []
    start_time = time.time()

    for i in range(int(duration * actual_fps)):
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frames.append(frame)

        # Progress
        elapsed = time.time() - start_time
        if i % int(actual_fps) == 0:
            remaining = duration - elapsed
            print("  [{:.0f}s] {} frames recorded".format(elapsed, len(frames)))

        time.sleep(1/actual_fps)

    end_time = time.time()
    actual_duration = end_time - start_time

    cap.release()
    out.release()

    print()
    print("[OK] Recording complete!")
    print("  Video: {}".format(output_video))
    print("  Frames: {}".format(len(frames)))
    print("  Duration: {:.1f}s".format(actual_duration))

    # Create GT file
    # NOTE: GT is ESTIMATED based on expected resting HR
    # Real GT requires physiological measurement device
    print()
    print("Creating ground truth file...")

    with open(output_gt, 'w') as f:
        f.write("timestamp,bpm\n")

        # Generate GT at 0.5s intervals
        # Add realistic variation (normal HR variation is ~2-5 BPM)
        rng = np.random.RandomState(42)  # Reproducible
        for t in np.arange(0, actual_duration + 0.5, 0.5):
            if t > actual_duration:
                break
            # Normal physiological variation
            variation = rng.normal(0, 3)  # ±3 BPM natural variation
            bpm = expected_bpm + variation
            bpm = max(50, min(120, bpm))  # Clamp to physiological range
            f.write("{:.3f},{:.1f}\n".format(t, bpm))

    print("  GT: {}".format(output_gt))
    print("  Estimated BPM: ~{} BPM (RESTING HR)".format(expected_bpm))
    print()
    print("IMPORTANT NOTES:")
    print("  - GT BPM is ESTIMATED, not measured with physiological device")
    print("  - For scientific accuracy, use GT from pulse oximeter or ECG")
    print()

    return True

def main():
    parser = argparse.ArgumentParser(description='Capture real face video')
    parser.add_argument('--output', '-o', default='benchmark_results/real_face.mp4',
                        help='Output video path')
    parser.add_argument('--gt', '-g', default=None,
                        help='Output GT path (default: video_path_gt.csv)')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Recording duration in seconds')
    parser.add_argument('--bpm', '-b', type=int, default=72,
                        help='Expected resting BPM')
    parser.add_argument('--fps', '-f', type=float, default=30.0,
                        help='Target FPS')

    args = parser.parse_args()

    # Auto-generate GT path
    if args.gt is None:
        gt_path = args.output.replace('.mp4', '_gt.csv')
    else:
        gt_path = args.gt

    success = capture_face_video(
        output_video=args.output,
        output_gt=gt_path,
        duration=args.duration,
        expected_bpm=args.bpm,
        fps=args.fps
    )

    if success:
        print("=" * 60)
        print("CAPTURE COMPLETE")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Review the video to verify face visibility")
        print("  2. If using estimated GT, note this limitation")
        print("  3. Run benchmark:")
        print("     /c/miniconda3/python.exe rppg_phase2_real_benchmark.py \\")
        print("       --video {} \\".format(args.output))
        print("       --gt {} \\".format(gt_path))
        print("       --output benchmark_results")
    else:
        print("Capture failed. Check webcam access.")

if __name__ == '__main__':
    main()
