"""
Tests for timestamp alignment module.

Tests cover:
1. Exact timestamp matching
2. Tolerance boundary cases
3. Outside tolerance rejection
4. Duplicate timestamp handling
5. Missing GT handling
6. Missing predictions handling
7. Unsorted timestamps
8. Perfect prediction metrics
"""

import sys
import os
import numpy as np
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rppg_benchmark_alignment import (
    align_timestamps,
    align_with_predictions_dict,
    validate_alignment_input,
    AlignmentResult,
)


class TestAlignmentBasics:
    """Basic alignment tests."""

    def test_exact_match(self):
        """Test exact timestamp matching."""
        pred_ts = np.array([1.0, 2.0, 3.0])
        pred_bpm = np.array([72.0, 75.0, 78.0])
        gt_ts = np.array([1.0, 2.0, 3.0])
        gt_bpm = np.array([72.0, 75.0, 78.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.1)

        assert result.n_aligned == 3, f"Expected 3 aligned, got {result.n_aligned}"
        np.testing.assert_array_almost_equal(
            result.aligned_predictions, [72.0, 75.0, 78.0]
        )
        np.testing.assert_array_almost_equal(result.alignment_deltas, [0.0, 0.0, 0.0])
        print(f"  ✓ Exact match: {result.n_aligned} pairs, deltas={result.alignment_deltas}")

    def test_nearest_neighbor(self):
        """Test nearest neighbor matching."""
        pred_ts = np.array([1.0, 2.0])
        pred_bpm = np.array([72.0, 75.0])
        gt_ts = np.array([1.1, 2.1])  # Offset by 0.1s
        gt_bpm = np.array([72.0, 75.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.5)

        assert result.n_aligned == 2
        assert np.all(result.alignment_deltas <= 0.11)
        print(f"  ✓ Nearest neighbor: deltas={result.alignment_deltas}")

    def test_alignment_with_dicts(self):
        """Test alignment using dict-based predictions."""
        predictions = [
            {"timestamp": 1.0, "bpm": 72.0},
            {"timestamp": 2.0, "bpm": 75.0},
        ]
        gt_ts = np.array([1.05, 2.05])
        gt_bpm = np.array([72.0, 75.0])

        result = align_with_predictions_dict(
            predictions, gt_ts, gt_bpm, tolerance=0.5
        )

        assert result.n_aligned == 2
        print(f"  ✓ Dict alignment: {result.n_aligned} pairs")


class TestToleranceEnforcement:
    """Test tolerance boundary enforcement."""

    def test_within_tolerance(self):
        """Test that predictions within tolerance are accepted."""
        pred_ts = np.array([1.0])
        pred_bpm = np.array([72.0])
        gt_ts = np.array([1.0, 2.0, 3.0])
        gt_bpm = np.array([72.0, 75.0, 78.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.5)

        assert result.n_aligned == 1
        assert result.diagnostics["n_rejected_tolerance"] == 0
        print(f"  ✓ Within tolerance: aligned={result.n_aligned}")

    def test_outside_tolerance(self):
        """Test that predictions outside tolerance are rejected."""
        pred_ts = np.array([1.0])
        pred_bpm = np.array([72.0])
        gt_ts = np.array([5.0, 6.0, 7.0])  # All far from pred
        gt_bpm = np.array([72.0, 75.0, 78.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.5)

        assert result.n_aligned == 0
        assert result.diagnostics["n_unmatched_predictions"] == 1
        print(f"  ✓ Outside tolerance: rejected={result.diagnostics['n_rejected_tolerance']}")

    def test_tolerance_boundary_exact(self):
        """Test tolerance boundary at exact limit."""
        pred_ts = np.array([1.0])
        pred_bpm = np.array([72.0])
        gt_ts = np.array([1.5])  # Exactly at tolerance
        gt_bpm = np.array([72.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.5)

        assert result.n_aligned == 1, "Should accept at exactly tolerance boundary"
        print(f"  ✓ Tolerance boundary: delta={result.alignment_deltas[0]}")

    def test_tolerance_boundary_just_over(self):
        """Test tolerance boundary just over limit."""
        pred_ts = np.array([1.0])
        pred_bpm = np.array([72.0])
        gt_ts = np.array([1.51])  # Just over tolerance
        gt_bpm = np.array([72.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.5)

        assert result.n_aligned == 0, "Should reject just over tolerance"
        print(f"  ✓ Just over tolerance: rejected (delta=0.51 > 0.5)")

    def test_tolerance_zero(self):
        """Test zero tolerance requires exact match."""
        pred_ts = np.array([1.0, 2.0])
        pred_bpm = np.array([72.0, 75.0])
        gt_ts = np.array([1.0, 2.001])  # Second GT is 1ms off
        gt_bpm = np.array([72.0, 75.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.0)

        assert result.n_aligned == 1, "Only exact match should align"
        assert result.diagnostics["n_unmatched_predictions"] == 1
        print(f"  ✓ Zero tolerance: {result.n_aligned} exact, 1 rejected")


class TestGTExhaustion:
    """Test that GT observations are not reused."""

    def test_single_gt_multiple_preds(self):
        """Test that one GT doesn't match multiple predictions."""
        pred_ts = np.array([1.0, 1.1, 1.2])
        pred_bpm = np.array([72.0, 73.0, 74.0])
        gt_ts = np.array([1.0])  # Only one GT
        gt_bpm = np.array([72.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.5)

        assert result.n_aligned == 1, "Only one prediction should match"
        assert result.diagnostics["n_unmatched_predictions"] == 2
        assert result.diagnostics["n_unmatched_gt"] == 0, "GT should be used"
        print(f"  ✓ GT exhaustion: 1 aligned, 2 rejected")

    def test_multiple_gt_single_pred(self):
        """Test that one prediction matches nearest available GT."""
        pred_ts = np.array([1.5])
        pred_bpm = np.array([72.0])
        gt_ts = np.array([1.0, 2.0, 3.0])
        gt_bpm = np.array([70.0, 80.0, 90.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=1.0)

        assert result.n_aligned == 1
        # Should match nearest GT (t=1.0, delta=0.5)
        assert abs(result.aligned_ground_truth[0] - 70.0) < 0.1
        print(f"  ✓ Single pred: matched nearest GT={result.aligned_ground_truth[0]}")


class TestUnsortedTimestamps:
    """Test handling of unsorted timestamps."""

    def test_unsorted_predictions(self):
        """Test that unsorted predictions are handled correctly."""
        pred_ts = np.array([3.0, 1.0, 2.0])  # Unsorted
        pred_bpm = np.array([78.0, 72.0, 75.0])
        gt_ts = np.array([1.0, 2.0, 3.0])
        gt_bpm = np.array([72.0, 75.0, 78.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.1)

        assert result.n_aligned == 3
        print(f"  ✓ Unsorted predictions: {result.n_aligned} aligned")

    def test_unsorted_gt(self):
        """Test that unsorted GT is handled correctly."""
        pred_ts = np.array([1.0, 2.0, 3.0])
        pred_bpm = np.array([72.0, 75.0, 78.0])
        # GT is unsorted but has 4 entries so exhaustion doesn't cause issues
        gt_ts = np.array([3.0, 1.0, 2.0, 4.0])  # Unsorted
        gt_bpm = np.array([78.0, 72.0, 75.0, 80.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.2)

        # All 3 predictions should match (sorted GT: [1.0, 2.0, 3.0, 4.0])
        assert result.n_aligned == 3, f"Expected 3 aligned, got {result.n_aligned}"
        print(f"  ✓ Unsorted GT: {result.n_aligned} aligned")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        pred_ts = np.array([])
        pred_bpm = np.array([])
        gt_ts = np.array([1.0, 2.0])
        gt_bpm = np.array([72.0, 75.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.5)

        assert result.n_aligned == 0
        assert result.diagnostics["n_unmatched_gt"] == 2
        print(f"  ✓ Empty predictions: 0 aligned, 2 unmatched GT")

    def test_empty_gt(self):
        """Test handling of empty GT."""
        pred_ts = np.array([1.0, 2.0])
        pred_bpm = np.array([72.0, 75.0])
        gt_ts = np.array([])
        gt_bpm = np.array([])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.5)

        assert result.n_aligned == 0
        assert result.diagnostics["n_unmatched_predictions"] == 2
        print(f"  ✓ Empty GT: 0 aligned, 2 unmatched predictions")

    def test_identical_timestamps(self):
        """Test handling of identical timestamps (duplicates in GT)."""
        pred_ts = np.array([1.0, 1.0])  # Two predictions at same time
        pred_bpm = np.array([72.0, 73.0])
        gt_ts = np.array([1.0])  # One GT
        gt_bpm = np.array([72.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.1)

        # First prediction matches, second is unmatched (GT exhausted)
        assert result.n_aligned == 1
        print(f"  ✓ Duplicate timestamps: 1 aligned (GT exhausted)")


class TestMetricsFromAlignment:
    """Test that alignment produces correct metrics."""

    def test_perfect_alignment_mae_zero(self):
        """Test that perfect alignment gives MAE = 0."""
        pred_ts = np.array([1.0, 2.0, 3.0])
        pred_bpm = np.array([72.0, 75.0, 78.0])
        gt_ts = np.array([1.0, 2.0, 3.0])
        gt_bpm = np.array([72.0, 75.0, 78.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.1)

        errors = np.abs(result.aligned_predictions - result.aligned_ground_truth)
        mae = np.mean(errors)

        assert mae == 0.0, f"Perfect alignment should give MAE=0, got {mae}"
        print(f"  ✓ Perfect alignment: MAE={mae:.6f}")

    def test_known_offset_mae(self):
        """Test that known offset produces expected MAE."""
        pred_ts = np.array([1.0, 2.0, 3.0])
        pred_bpm = np.array([74.0, 77.0, 80.0])  # All +2 BPM offset
        gt_ts = np.array([1.0, 2.0, 3.0])
        gt_bpm = np.array([72.0, 75.0, 78.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.1)

        errors = np.abs(result.aligned_predictions - result.aligned_ground_truth)
        mae = np.mean(errors)

        assert abs(mae - 2.0) < 0.01, f"Expected MAE=2.0, got {mae}"
        print(f"  ✓ Known offset: MAE={mae:.2f} (expected 2.0)")

    def test_diagnostics_complete(self):
        """Test that diagnostics are complete."""
        pred_ts = np.array([1.0, 2.0, 3.0])
        pred_bpm = np.array([72.0, 75.0, 78.0])
        gt_ts = np.array([1.0, 2.0, 3.0])
        gt_bpm = np.array([72.0, 75.0, 78.0])

        result = align_timestamps(pred_ts, pred_bpm, gt_ts, gt_bpm, tolerance=0.1)

        diag = result.diagnostics
        assert "n_predictions" in diag
        assert "n_gt" in diag
        assert "n_aligned" in diag
        assert "n_rejected_tolerance" in diag
        assert "n_unmatched_predictions" in diag
        assert "n_unmatched_gt" in diag
        assert "alignment_rate" in diag
        assert "tolerance_seconds" in diag
        assert "mean_delta_seconds" in diag
        assert "max_delta_seconds" in diag
        print(f"  ✓ Diagnostics complete: {list(diag.keys())}")


class TestValidation:
    """Test input validation."""

    def test_validate_empty_input(self):
        """Test validation with empty inputs."""
        messages = validate_alignment_input(
            np.array([]), np.array([1.0, 2.0]), 0.5
        )
        assert any("No predictions" in m for m in messages)
        print(f"  ✓ Empty prediction validation: {len(messages)} warnings")

    def test_validate_negative_tolerance(self):
        """Test validation with negative tolerance."""
        messages = validate_alignment_input(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), -0.5
        )
        assert any("Tolerance must be positive" in m for m in messages)
        print(f"  ✓ Negative tolerance validation: flagged")


def run_tests():
    """Run all alignment tests."""
    print("\n" + "=" * 70)
    print("TIMESTAMP ALIGNMENT TESTS")
    print("=" * 70)

    test_classes = [
        ("Basic Alignment", TestAlignmentBasics),
        ("Tolerance Enforcement", TestToleranceEnforcement),
        ("GT Exhaustion", TestGTExhaustion),
        ("Unsorted Timestamps", TestUnsortedTimestamps),
        ("Edge Cases", TestEdgeCases),
        ("Metrics", TestMetricsFromAlignment),
        ("Validation", TestValidation),
    ]

    total_passed = 0
    total_failed = 0

    for class_name, test_class in test_classes:
        print(f"\n--- {class_name} ---")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                test_fn = getattr(instance, method_name)
                print(f"  {method_name}...", end=" ")
                try:
                    test_fn()
                    total_passed += 1
                except AssertionError as e:
                    print(f"FAIL: {e}")
                    total_failed += 1
                except Exception as e:
                    print(f"ERROR: {type(e).__name__}: {e}")
                    total_failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
