"""
Synthetic benchmark example for Phase 1.

This demonstrates that the benchmark infrastructure is mathematically wired correctly.

IMPORTANT: This validates INFRASTRUCTURE, not physiological accuracy.
Synthetic results are NOT evidence that Sanubari measures human physiology.

Usage:
    python benchmark_synthetic_example.py
"""

import os
import sys
import json
import numpy as np
import tempfile
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rppg_benchmark_alignment import align_timestamps
from rppg_benchmark import compute_metrics, MetricResult, BlandAltmanResult
from rppg_benchmark_manifest import create_manifest


def create_synthetic_gt(
    path: str,
    duration: float = 30.0,
    base_bpm: float = 72.0,
    interval: float = 0.5,
    noise_std: float = 1.0,
) -> dict:
    """
    Create synthetic ground truth with known properties.

    Parameters
    ----------
    path : str
        Output CSV path.
    duration : float
        Recording duration in seconds.
    base_bpm : float
        Base heart rate.
    interval : float
        GT sample interval in seconds.
    noise_std : float
        BPM noise standard deviation.

    Returns
    -------
    dict
        GT metadata including timestamps and BPM values.
    """
    timestamps = np.arange(0, duration, interval)
    bpms = base_bpm + np.random.randn(len(timestamps)) * noise_std

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'bpm'])
        for ts, bpm in zip(timestamps, bpms):
            writer.writerow([f"{ts:.3f}", f"{bpm:.1f}"])

    return {
        "path": path,
        "timestamps": timestamps,
        "bpms": bpms,
        "duration": duration,
        "base_bpm": base_bpm,
        "interval": interval,
        "noise_std": noise_std,
        "n_samples": len(timestamps),
    }


def create_synthetic_predictions(
    gt: dict,
    error_bpm: float = 0.0,
    noise_std: float = 0.5,
    missing_rate: float = 0.0,
) -> dict:
    """
    Create synthetic predictions that can be perfectly validated.

    Parameters
    ----------
    gt : dict
        Ground truth metadata.
    error_bpm : float
        Systematic error to add (bias).
    noise_std : float
        Random noise standard deviation.
    missing_rate : float
        Fraction of GT samples with no prediction.

    Returns
    -------
    dict
        Prediction metadata.
    """
    np.random.seed(42)  # Deterministic

    predictions = []
    gt_timestamps = gt["timestamps"]
    gt_bpms = gt["bpms"]

    for i, (ts, bpm) in enumerate(zip(gt_timestamps, gt_bpms)):
        if np.random.rand() < missing_rate:
            continue  # Skip this sample

        # Create prediction with known error
        pred_bpm = bpm + error_bpm + np.random.randn() * noise_std

        predictions.append({
            "timestamp": ts,
            "bpm": pred_bpm,
            "sqi": 80.0,
            "valid": True,
        })

    return {
        "predictions": predictions,
        "error_bpm": error_bpm,
        "noise_std": noise_std,
        "missing_rate": missing_rate,
        "n_predictions": len(predictions),
    }


def run_synthetic_benchmark(
    base_bpm: float = 72.0,
    error_bpm: float = 0.0,
    noise_std: float = 1.0,
    alignment_tolerance: float = 0.5,
) -> dict:
    """
    Run synthetic benchmark with known ground truth.

    Returns result dict with alignment and metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = os.path.join(tmpdir, "gt.csv")
        output_path = os.path.join(tmpdir, "result.json")

        # Create synthetic data
        gt = create_synthetic_gt(gt_path, duration=30.0, base_bpm=base_bpm)
        preds = create_synthetic_predictions(gt, error_bpm=error_bpm, noise_std=noise_std)

        # Create manifest
        manifest = create_manifest(
            experiment_id="SYNTHETIC-VALIDATION",
            video_path="synthetic://generated",
            gt_path=gt_path,
            dataset="synthetic_fixture",
            alignment_tolerance=alignment_tolerance,
            gt_sample_count=gt["n_samples"],
            gt_timestamp_range=[0.0, gt["duration"]],
        )

        # Align
        pred_ts = np.array([p["timestamp"] for p in preds["predictions"]])
        pred_bpm = np.array([p["bpm"] for p in preds["predictions"]])
        gt_ts = gt["timestamps"]
        gt_bpm = gt["bpms"]

        alignment = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, alignment_tolerance)

        # Compute metrics
        if alignment.n_aligned >= 10:
            metrics = compute_metrics(
                alignment.aligned_predictions,
                alignment.aligned_ground_truth,
                config_name="synthetic_validation",
            )
        else:
            # Manual calculation for small samples
            mae = float(np.mean(np.abs(alignment.aligned_predictions - alignment.aligned_ground_truth)))
            rmse = float(np.sqrt(np.mean((alignment.aligned_predictions - alignment.aligned_ground_truth)**2)))
            metrics = None

        # Compile result
        result = {
            "experiment_id": manifest.experiment_id,
            "timestamp": manifest.timestamp,
            "data_type": "SYNTHETIC",
            "scientific_warning": "Synthetic data validates infrastructure, NOT physiological accuracy",
            "gt": {
                "n_samples": gt["n_samples"],
                "duration": gt["duration"],
                "base_bpm": base_bpm,
                "noise_std": gt["noise_std"],
            },
            "predictions": {
                "n_total": preds["n_predictions"],
                "error_bpm": error_bpm,
                "noise_std": noise_std,
            },
            "alignment": {
                "n_aligned": alignment.n_aligned,
                "alignment_rate": alignment.diagnostics["alignment_rate"],
                "mean_delta": alignment.mean_delta,
                "max_delta": alignment.max_delta,
            },
            "metrics": {},
        }

        if metrics is not None:
            result["metrics"] = {
                "mae": metrics.mae.value,
                "mae_ci": (metrics.mae.ci_lower, metrics.mae.ci_upper),
                "rmse": metrics.rmse.value,
                "rmse_ci": (metrics.rmse.ci_lower, metrics.rmse.ci_upper),
                "pearson_r": metrics.pearson_r.value,
                "pearson_ci": (metrics.pearson_r.ci_lower, metrics.pearson_r.ci_upper),
                "bias": metrics.bland_altman.bias,
            }
        else:
            result["metrics"] = {
                "mae": mae,
                "rmse": rmse,
                "note": "Insufficient samples for bootstrap CI",
            }

        return result


def test_perfect_alignment():
    """
    Test 1: Perfect alignment (error_bpm=0, noise_std=0).

    Expected: MAE ≈ 0, RMSE ≈ 0, Pearson undefined or 1.
    """
    print("\n" + "=" * 70)
    print("TEST 1: PERFECT ALIGNMENT")
    print("=" * 70)
    print("Conditions: error_bpm=0, noise_std=0")
    print("Expected: MAE ≈ 0, RMSE ≈ 0")

    result = run_synthetic_benchmark(error_bpm=0.0, noise_std=0.0)

    mae = result["metrics"]["mae"]
    rmse = result["metrics"]["rmse"]

    print(f"\nResults:")
    print(f"  MAE:  {mae:.6f} BPM")
    print(f"  RMSE: {rmse:.6f} BPM")
    print(f"  Aligned: {result['alignment']['n_aligned']} pairs")

    # Verify
    if mae < 0.01 and rmse < 0.01:
        print(f"\n  ✓ PASS: Perfect alignment gives near-zero error")
        return True
    else:
        print(f"\n  ✗ FAIL: Expected MAE≈0, RMSE≈0")
        return False


def test_known_bias():
    """
    Test 2: Known systematic bias.

    Expected: MAE ≈ |error_bpm|, RMSE ≈ |error_bpm|.
    """
    print("\n" + "=" * 70)
    print("TEST 2: KNOWN BIAS")
    print("=" * 70)
    print("Conditions: error_bpm=5.0, noise_std=0")
    print("Expected: MAE ≈ 5.0, RMSE ≈ 5.0")

    result = run_synthetic_benchmark(error_bpm=5.0, noise_std=0.0)

    mae = result["metrics"]["mae"]
    rmse = result["metrics"]["rmse"]
    bias = result["metrics"]["bias"]

    print(f"\nResults:")
    print(f"  MAE:   {mae:.2f} BPM")
    print(f"  RMSE:  {rmse:.2f} BPM")
    print(f"  Bias:  {bias:+.2f} BPM")

    # Verify
    if abs(mae - 5.0) < 0.1 and abs(bias - 5.0) < 0.1:
        print(f"\n  ✓ PASS: Known bias correctly detected")
        return True
    else:
        print(f"\n  ✗ FAIL: Expected MAE≈5.0, bias≈5.0")
        return False


def test_random_noise():
    """
    Test 3: Random noise only.

    Expected: MAE ≈ 0.8 * noise_std (for Gaussian), bias ≈ 0.
    """
    print("\n" + "=" * 70)
    print("TEST 3: RANDOM NOISE")
    print("=" * 70)
    print("Conditions: error_bpm=0, noise_std=2.0")
    print("Expected: MAE ≈ 1.6 (sqrt(2/pi)*std for half-normal)")

    result = run_synthetic_benchmark(error_bpm=0.0, noise_std=2.0)

    mae = result["metrics"]["mae"]
    bias = result["metrics"]["bias"]

    print(f"\nResults:")
    print(f"  MAE:   {mae:.2f} BPM")
    print(f"  Bias:  {bias:+.2f} BPM")

    # Verify: MAE should be around 1.6 for noise_std=2.0
    # (sqrt(2/pi) * 2.0 ≈ 1.6)
    expected_mae = np.sqrt(2 / np.pi) * 2.0
    if abs(mae - expected_mae) < 0.3 and abs(bias) < 0.3:
        print(f"\n  ✓ PASS: Random noise produces expected MAE")
        return True
    else:
        print(f"\n  ✗ FAIL: Expected MAE≈{expected_mae:.2f}, bias≈0")
        return False


def test_combined_bias_noise():
    """
    Test 4: Combined bias and noise.

    Expected: bias ≈ error_bpm, MAE ≈ |error| + random_component.
    """
    print("\n" + "=" * 70)
    print("TEST 4: COMBINED BIAS AND NOISE")
    print("=" * 70)
    print("Conditions: error_bpm=3.0, noise_std=1.5")
    print("Expected: bias ≈ 3.0, MAE ≈ 3.0 + 1.2 ≈ 4.2")

    result = run_synthetic_benchmark(error_bpm=3.0, noise_std=1.5)

    mae = result["metrics"]["mae"]
    rmse = result["metrics"]["rmse"]
    bias = result["metrics"]["bias"]

    print(f"\nResults:")
    print(f"  MAE:   {mae:.2f} BPM")
    print(f"  RMSE:  {rmse:.2f} BPM")
    print(f"  Bias:  {bias:+.2f} BPM")

    # Verify: bias should be close to 3.0
    if abs(bias - 3.0) < 0.3:
        print(f"\n  ✓ PASS: Bias correctly measured")
        return True
    else:
        print(f"\n  ✗ FAIL: Expected bias≈3.0")
        return False


def test_alignment_diagnostics():
    """
    Test 5: Verify alignment diagnostics are populated.
    """
    print("\n" + "=" * 70)
    print("TEST 5: ALIGNMENT DIAGNOSTICS")
    print("=" * 70)

    result = run_synthetic_benchmark(error_bpm=0.0, noise_std=0.5)

    alignment = result["alignment"]

    print(f"\nDiagnostics:")
    print(f"  n_aligned: {alignment['n_aligned']}")
    print(f"  alignment_rate: {alignment['alignment_rate']:.2%}")
    print(f"  mean_delta: {alignment['mean_delta']:.4f}s")
    print(f"  max_delta: {alignment['max_delta']:.4f}s")

    if alignment['n_aligned'] > 0 and alignment['mean_delta'] < 0.3:
        print(f"\n  ✓ PASS: Diagnostics populated correctly")
        return True
    else:
        print(f"\n  ✗ FAIL: Unexpected diagnostic values")
        return False


def main():
    """Run all synthetic benchmark tests."""
    print("\n" + "=" * 70)
    print("SYNTHETIC BENCHMARK VALIDATION")
    print("=" * 70)
    print("\n⚠️  WARNING: Synthetic data validates INFRASTRUCTURE only.")
    print("⚠️  Synthetic results are NOT evidence of physiological accuracy.")
    print("\nThis test proves:")
    print("  1. Timestamp alignment works correctly")
    print("  2. Metrics are mathematically wired")
    print("  3. Error propagation is correct")
    print("  4. Diagnostics are populated")

    tests = [
        ("Perfect Alignment", test_perfect_alignment),
        ("Known Bias", test_known_bias),
        ("Random Noise", test_random_noise),
        ("Combined", test_combined_bias_noise),
        ("Diagnostics", test_alignment_diagnostics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            print(f"\n  ✗ ERROR: {type(e).__name__}: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("SYNTHETIC BENCHMARK SUMMARY")
    print("=" * 70)

    for name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")

    all_passed = all(passed for _, passed, _ in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe benchmark infrastructure is mathematically verified.")
        print("Synthetic results are NOT evidence of physiological accuracy.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Review infrastructure before proceeding.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
