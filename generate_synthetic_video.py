#!/usr/bin/env python3
"""
Generate synthetic video fixture for benchmark infrastructure testing.

This creates a minimal video with simulated face landmarks
that can be used to test the benchmark pipeline without real data.

Usage:
    python generate_synthetic_video.py --output synthetic_video.mp4 --gt synthetic_gt.csv --frames 90
"""

import cv2
import numpy as np
import argparse
import csv
import os


def generate_synthetic_face_frame(width=640, height=480, frame_idx=0):
    """
    Generate a synthetic frame with simulated face region.

    Returns a BGR frame with a reddish face region and skin-colored background.
    """
    # Background: skin-like tone
    frame = np.full((height, width, 3), (180, 155, 135), dtype=np.uint8)

    # Face region (centered, with subtle pulse simulation)
    center_x, center_y = width // 2, height // 2
    face_w, face_h = 200, 260

    # Simulate subtle color variation (heartbeat)
    pulse = int(5 * np.sin(2 * np.pi * frame_idx / 30))  # ~1Hz at 30fps
    r_base = 180 + pulse

    # Draw ellipse for face
    cv2.ellipse(
        frame,
        (center_x, center_y),
        (face_w // 2, face_h // 2),
        0, 0, 360,
        (r_base, 130, 120),
        -1,
    )

    # Add some texture
    noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Apply Gaussian blur for more realistic appearance
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame


def generate_synthetic_video(
    output_path: str,
    fps: float = 30.0,
    n_frames: int = 90,
    width: int = 640,
    height: int = 480,
):
    """
    Generate synthetic video file.

    Parameters
    ----------
    output_path : str
        Output video path (.mp4)
    fps : float
        Frames per second
    n_frames : int
        Number of frames to generate
    width : int
        Frame width
    height : int
        Frame height
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    for i in range(n_frames):
        frame = generate_synthetic_face_frame(width, height, i)
        writer.write(frame)

    writer.release()
    print(f"[INFO] Generated synthetic video: {output_path}")
    print(f"       {n_frames} frames @ {fps}fps = {n_frames/fps:.1f}s")


def generate_synthetic_gt(
    output_path: str,
    fps: float = 30.0,
    n_frames: int = 90,
    base_bpm: float = 72.0,
):
    """
    Generate synthetic ground truth CSV.

    Parameters
    ----------
    output_path : str
        Output CSV path
    fps : float
        Frames per second (for timestamp calculation)
    n_frames : int
        Number of frames
    base_bpm : float
        Base heart rate in BPM
    """
    # GT at every 0.5 seconds
    gt_interval = 0.5
    gt_times = np.arange(0, n_frames / fps, gt_interval)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'bpm'])

        for ts in gt_times:
            # Add realistic variability
            bpm = base_bpm + np.random.normal(0, 1.5)
            writer.writerow([f"{ts:.3f}", f"{bpm:.1f}"])

    print(f"[INFO] Generated synthetic GT: {output_path}")
    print(f"       {len(gt_times)} GT entries @ {gt_interval}s interval")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic video + GT for benchmark testing"
    )
    parser.add_argument(
        "--output",
        default="synthetic_fixture",
        help="Base output name (without extension)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=90,
        help="Number of frames (default: 90 = 3s @ 30fps)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=72.0,
        help="Base BPM for GT (default: 72)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )

    args = parser.parse_args()

    video_path = f"{args.output}.mp4"
    gt_path = f"{args.output}_gt.csv"

    print("=" * 60)
    print("SYNTHETIC VIDEO GENERATOR")
    print("=" * 60)

    generate_synthetic_video(
        output_path=video_path,
        fps=args.fps,
        n_frames=args.frames,
        width=args.width,
        height=args.height,
    )

    generate_synthetic_gt(
        output_path=gt_path,
        fps=args.fps,
        n_frames=args.frames,
        base_bpm=args.bpm,
    )

    print("\n" + "=" * 60)
    print("Use with benchmark runner:")
    print(f"  python benchmark_video.py \\")
    print(f"      --video {video_path} \\")
    print(f"      --gt {gt_path} \\")
    print(f"      --output results/ \\")
    print(f"      --experiment-id SYNTHETIC-TEST \\")
    print(f"      --verbose")
    print("=" * 60)


if __name__ == "__main__":
    main()
