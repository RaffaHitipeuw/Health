#!/usr/bin/env python3
"""
CLI entry point for benchmark execution.

Usage:
    python benchmark_video.py --video video.mp4 --gt ground_truth.csv --output results/ --experiment-id EXP-001

Optional arguments:
    --subject-id SUB001
    --video-id VID001
    --dataset COHFACE
    --alignment-tolerance 2.0
    --max-frames 300
    --verbose
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rppg_benchmark_run import benchmark_video, BenchmarkExperimentResult


def main():
    parser = argparse.ArgumentParser(
        description="rPPG Benchmark Runner — Execute prerecorded video + ground truth experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python benchmark_video.py --video video.mp4 --gt gt.csv --output results/

  # With metadata
  python benchmark_video.py \\
      --video data/subject01/video.mp4 \\
      --gt data/subject01/ground_truth.csv \\
      --output results/ \\
      --experiment-id SUB01_SESSION01 \\
      --subject-id SUB01 \\
      --video-id VID01 \\
      --dataset COHFACE

  # Test run (limit frames)
  python benchmark_video.py --video video.mp4 --gt gt.csv --output results/ --max-frames 300 --verbose
        """
    )

    # Required arguments
    parser.add_argument(
        "--video",
        required=True,
        help="Path to prerecorded video file (mp4, avi, etc.)",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to ground truth CSV file with columns: timestamp, bpm",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory or file path for results (JSON)",
    )

    # Experiment identifier
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Unique experiment identifier (e.g., SUB01_SESSION01)",
    )

    # Optional metadata
    parser.add_argument(
        "--subject-id",
        help="Subject identifier for metadata",
    )
    parser.add_argument(
        "--video-id",
        help="Video identifier for metadata",
    )
    parser.add_argument(
        "--dataset",
        default="benchmark",
        help="Dataset name (default: benchmark)",
    )

    # Benchmark configuration
    parser.add_argument(
        "--alignment-tolerance",
        type=float,
        default=2.0,
        help="Maximum time difference (seconds) for GT matching (default: 2.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process (for testing; default: all frames)",
    )

    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stdout",
    )

    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.gt):
        print(f"[ERROR] Ground truth file not found: {args.gt}")
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if os.path.isdir(output_path):
        # Output is directory: create filename from experiment_id
        output_path = os.path.join(output_path, f"{args.experiment_id}.json")
    else:
        # Output is file path: use as-is
        pass

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Print configuration
    print("=" * 60)
    print("rPPG BENCHMARK RUNNER")
    print("=" * 60)
    print(f"  Experiment ID:    {args.experiment_id}")
    print(f"  Dataset:          {args.dataset}")
    print(f"  Video:            {args.video}")
    print(f"  Ground Truth:     {args.gt}")
    print(f"  Output:           {output_path}")
    print(f"  Alignment tol:    {args.alignment_tolerance}s")
    if args.subject_id:
        print(f"  Subject ID:       {args.subject_id}")
    if args.video_id:
        print(f"  Video ID:         {args.video_id}")
    if args.max_frames:
        print(f"  Max frames:       {args.max_frames}")
    print("=" * 60)

    # Execute benchmark
    try:
        result = benchmark_video(
            video_path=args.video,
            gt_path=args.gt,
            output_path=output_path,
            experiment_id=args.experiment_id,
            subject_id=args.subject_id,
            video_id=args.video_id,
            dataset=args.dataset,
            alignment_tolerance=args.alignment_tolerance,
            max_frames=args.max_frames,
            verbose=args.verbose,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"  Frames processed:  {result.n_frames_processed}")
        print(f"  No face:         {result.n_frames_no_face}")
        print(f"  Predictions:     {result.n_predictions_total}")
        print(f"  Zero BPM:        {result.n_predictions_zero_bpm}")
        print(f"  GT entries:       {result.n_gt_total}")
        print(f"  Aligned:         {result.n_aligned}")
        print(f"  Elapsed:         {result.elapsed_seconds:.1f}s ({result.fps_processing:.1f} fps)")

        if result.mae is not None:
            print(f"\n  METRICS:")
            print(f"    MAE:           {result.mae:.2f} BPM")
            print(f"    RMSE:          {result.rmse:.2f} BPM")
            if result.pearson_r is not None:
                print(f"    Pearson r:     {result.pearson_r:.4f}")

        print(f"\n  Results saved: {output_path}")
        print("=" * 60)

        sys.exit(0)

    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Benchmark cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
