"""
Unit tests for ground-truth timestamp alignment (Issue 7) and repeated GT fix (Issue 8).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time


class TestGroundTruthAlignment:
    """Test timestamp-based ground truth alignment."""

    def test_align_matching_timestamps(self):
        """Test that matching timestamps are correctly paired."""
        from rppg_vitals import GroundTruthInterface

        gti = GroundTruthInterface(align_tolerance=1.0)

        # Manually set ground truth at known timestamps
        gti.gt_bpm_series = [
            (0.0, 72.0, 'manual'),
            (1.0, 75.0, 'manual'),
            (5.0, 78.0, 'manual'),
        ]

        # Estimate at overlapping timestamps
        gti.est_bpm_series = [
            (0.1, 73.0),
            (1.1, 74.0),
            (2.0, 76.0),  # Should not match (no GT near 2.0 within tolerance)
        ]

        est_arr, gt_arr, diag = gti.align_observations(tolerance=1.0)

        assert len(est_arr) == 2, f"Expected 2 matches, got {len(est_arr)}"
        assert diag['n_matched'] == 2
        assert diag['n_unmatched_est'] == 1  # t=2.0 has no GT match
        assert diag['n_unmatched_gt'] == 1  # t=5.0 has no estimate match
        assert diag['tolerance'] == 1.0

        print(f"  ✓ 2 estimates matched, 1 unmatched (t=2.0, no nearby GT)")
        print(f"  ✓ 1 GT unmatched (t=5.0, no estimate nearby)")

    def test_align_no_false_pairing(self):
        """
        Test that t=2.0 estimate is NOT incorrectly paired with t=5.0 GT.

        This is the core issue: index-based pairing would pair 3rd estimate (t=2.0)
        with 3rd GT (t=5.0), producing a false 4 BPM error.
        Timestamp-based alignment should NOT do this.
        """
        from rppg_vitals import GroundTruthInterface

        gti = GroundTruthInterface(align_tolerance=1.0)

        # GT: t=0.0, 1.0, 5.0
        gti.gt_bpm_series = [
            (0.0, 72.0, 'manual'),
            (1.0, 75.0, 'manual'),
            (5.0, 80.0, 'manual'),
        ]

        # Estimates at t=0.1, 1.1, 2.0
        # t=2.0 should NOT be matched to t=5.0 GT (diff=3.0s > tolerance=1.0s)
        gti.est_bpm_series = [
            (0.1, 73.0),
            (1.1, 74.0),
            (2.0, 76.0),  # No GT within 1s tolerance
        ]

        est_arr, gt_arr, diag = gti.align_observations(tolerance=1.0)

        # Only 2 estimates should be matched
        assert len(est_arr) == 2, \
            f"Should only match 2 estimates, not pair t=2.0 with t=5.0: got {len(est_arr)}"

        # The unmatched estimate should NOT have been paired with t=5.0
        # If it were, we'd see gt_arr[-1] = 80.0, but it should be 75.0 (2nd GT)
        assert gt_arr[-1] == 75.0, \
            f"Last matched GT should be 75.0 (t=1.0), not 80.0 (t=5.0): got {gt_arr[-1]}"

        print("  ✓ t=2.0 estimate NOT incorrectly paired with t=5.0 GT")
        print(f"  ✓ Correctly unmatched: {diag['n_unmatched_est']} est, {diag['n_unmatched_gt']} GT")

    def test_align_bland_altman_with_alignment(self):
        """Test that compute_bland_altman uses alignment by default."""
        from rppg_vitals import GroundTruthInterface

        gti = GroundTruthInterface(align_tolerance=2.0)

        # GT at t=0, 1, 2, 5, 6, 7 (6 entries)
        gti.gt_bpm_series = [
            (0.0, 72.0, 'manual'),
            (1.0, 75.0, 'manual'),
            (2.0, 76.0, 'manual'),
            (5.0, 78.0, 'manual'),
            (6.0, 79.0, 'manual'),
            (7.0, 80.0, 'manual'),
        ]

        # Estimates at t=0.1, 1.1, 2.1, 5.2, 6.1, 10.0
        # t=10.0 should NOT be matched (no GT within 2s tolerance)
        gti.est_bpm_series = [
            (0.1, 73.0),
            (1.1, 76.0),
            (2.1, 77.0),
            (5.2, 79.0),
            (6.1, 78.0),
            (10.0, 81.0),  # Not matched
        ]

        result = gti.compute_bland_altman(use_alignment=True)

        # Should have 5 matched pairs
        assert result.get('n_matched') == 5, f"Expected 5 matched: {result.get('n_matched')}"
        assert result.get('n_unmatched_est') == 1  # t=10.0 not matched
        assert result.get('n_unmatched_gt') == 1  # t=7.0 not matched
        assert 'alignment_tolerance' in result
        assert result.get('mean_diff_bpm') != 0.0  # Should have real diff
        assert 'error' not in result, f"Should not error: {result}"

        print(f"  ✓ Bland-Altman with alignment: n_matched={result['n_matched']}")
        print(f"  ✓ mean_diff={result['mean_diff_bpm']:.1f} BPM, unmatched: {result['n_unmatched_est']} est + {result['n_unmatched_gt']} GT")

    def test_align_pearson_with_alignment(self):
        """Test that compute_pearson uses alignment by default."""
        from rppg_vitals import GroundTruthInterface

        gti = GroundTruthInterface(align_tolerance=2.0)

        gti.gt_bpm_series = [
            (0.0, 70.0, 'manual'),
            (2.0, 75.0, 'manual'),
            (4.0, 80.0, 'manual'),
        ]

        gti.est_bpm_series = [
            (0.2, 71.0),
            (2.2, 76.0),
            (4.2, 79.0),
        ]

        result = gti.compute_pearson(use_alignment=True)

        assert 'error' not in result
        assert result['n_matched'] == 3
        assert 'alignment_tolerance' in result

        print(f"  ✓ Pearson with alignment: r={result['pearson_r']:.4f}, n_matched={result['n_matched']}")

    def test_get_alignment_diagnostics(self):
        """Test that alignment diagnostics are accessible."""
        from rppg_vitals import GroundTruthInterface

        gti = GroundTruthInterface(align_tolerance=1.0)
        gti.gt_bpm_series = [(0.0, 72.0, 'manual')]
        gti.est_bpm_series = [(0.5, 73.0)]

        gti.align_observations()
        diag = gti.get_alignment_diagnostics()

        assert 'n_matched' in diag
        assert 'tolerance' in diag
        assert diag['n_matched'] == 1

        print(f"  ✓ Diagnostics accessible: {diag}")


class TestRepeatedGTFix:
    """Test that ground truth is not repeatedly appended."""

    def test_gt_interface_does_not_repeat(self):
        """Test that GroundTruthInterface only records each GT entry once."""
        from rppg_vitals import GroundTruthInterface

        gti = GroundTruthInterface()

        # Set GT once
        gti.set_ground_truth(75.0, 'manual')

        # Record multiple estimates
        for _ in range(10):
            gti.record_estimate(73.0)

        # GT series should have only 1 entry
        assert len(gti.gt_bpm_series) == 1, \
            f"GT series should have 1 entry, got {len(gti.gt_bpm_series)}"

        # Estimate series should have 10 entries
        assert len(gti.est_bpm_series) == 10

        print(f"  ✓ GT series: {len(gti.gt_bpm_series)} entries (not repeated)")
        print(f"  ✓ Est series: {len(gti.est_bpm_series)} entries")


def run_tests():
    """Run all ground truth tests."""
    print("\n" + "=" * 60)
    print("TESTING: Ground Truth Alignment & Repeated GT Fix")
    print("=" * 60)

    suite = TestGroundTruthAlignment()
    gt_fix_suite = TestRepeatedGTFix()

    tests = [
        ("Timestamp matching", suite.test_align_matching_timestamps),
        ("No false pairing (t=2.0 vs t=5.0)",
         suite.test_align_no_false_pairing),
        ("Bland-Altman with alignment",
         suite.test_align_bland_altman_with_alignment),
        ("Pearson with alignment",
         suite.test_align_pearson_with_alignment),
        ("Alignment diagnostics accessible",
         suite.test_get_alignment_diagnostics),
        ("GT interface does not repeat entries",
         gt_fix_suite.test_gt_interface_does_not_repeat),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
