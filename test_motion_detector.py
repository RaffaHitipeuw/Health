"""
Unit tests for V1 MotionDetector bug fix.

Issue: MotionArtifactDetector references self.enter_threshold which doesn't exist.
Fix: Added enter_threshold and exit_threshold properties.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rppg_core import MotionArtifactDetector
from rppg_config import cfg


class TestMotionDetector:
    """Test suite for MotionArtifactDetector."""

    def test_enter_threshold_property_exists(self):
        """Test that enter_threshold property is accessible."""
        detector = MotionArtifactDetector()

        # Should be accessible as property
        assert hasattr(detector, 'enter_threshold'), \
            "MotionArtifactDetector should have enter_threshold property"

        # Should return the configured threshold
        assert detector.enter_threshold == cfg.MOTION_ENTER_THRESHOLD, \
            f"enter_threshold should equal cfg.MOTION_ENTER_THRESHOLD ({cfg.MOTION_ENTER_THRESHOLD})"

        print(f"  ✓ enter_threshold = {detector.enter_threshold}")

    def test_exit_threshold_property_exists(self):
        """Test that exit_threshold property is accessible."""
        detector = MotionArtifactDetector()

        assert hasattr(detector, 'exit_threshold'), \
            "MotionArtifactDetector should have exit_threshold property"

        assert detector.exit_threshold == cfg.MOTION_EXIT_THRESHOLD, \
            f"exit_threshold should equal cfg.MOTION_EXIT_THRESHOLD ({cfg.MOTION_EXIT_THRESHOLD})"

        print(f"  ✓ exit_threshold = {detector.exit_threshold}")

    def test_motion_detector_integration(self):
        """
        Test that the motion detector can be instantiated and used
        without AttributeError on enter_threshold.

        This was the bug: V1 code calls motion_detector.enter_threshold
        but the attribute didn't exist.
        """
        detector = MotionArtifactDetector()

        # Simulate the V1 access pattern
        motion_score = 0.5
        enter_thresh = detector.enter_threshold
        exit_thresh = detector.exit_threshold

        # Should not raise AttributeError
        assert motion_score <= enter_thresh, \
            f"motion_score ({motion_score}) should be <= enter_threshold ({enter_thresh})"

        # The bug was accessing enter_threshold - verify it works
        # This would have raised AttributeError before the fix
        threshold_used = detector.enter_threshold
        assert threshold_used > 0, "Threshold should be positive"

        print(f"  ✓ No AttributeError when accessing enter_threshold ({threshold_used})")

    def test_reset_preserves_threshold_properties(self):
        """Test that reset() doesn't clear threshold properties."""
        detector = MotionArtifactDetector()
        detector.reset()

        # Properties should still be accessible after reset
        assert detector.enter_threshold == cfg.MOTION_ENTER_THRESHOLD
        assert detector.exit_threshold == cfg.MOTION_EXIT_THRESHOLD

        print("  ✓ Threshold properties preserved after reset")


def run_tests():
    """Run all motion detector tests."""
    print("\n" + "=" * 60)
    print("TESTING: V1 Motion Detector Bug Fix")
    print("=" * 60)

    test_suite = TestMotionDetector()

    tests = [
        ("enter_threshold property exists",
         test_suite.test_enter_threshold_property_exists),
        ("exit_threshold property exists",
         test_suite.test_exit_threshold_property_exists),
        ("Motion detector integration (no AttributeError)",
         test_suite.test_motion_detector_integration),
        ("Reset preserves threshold properties",
         test_suite.test_reset_preserves_threshold_properties),
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
        except AttributeError as e:
            print(f"  ATTRIBUTE ERROR: {e}")
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
