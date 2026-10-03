"""
Unit tests for rppg_uncertainty module.
Tests the BayesianHREstimator and UncertaintyAwareConfidence classes.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rppg_uncertainty import BayesianHREstimator, UncertaintyAwareConfidence


class TestBayesianHREstimator:
    """Test suite for Bayesian heart rate estimator."""

    def test_covariance_update_mathematical_correctness(self):
        """
        Test that the covariance update is mathematically correct.
        The Kalman covariance update should be: P_new = (I - KH)P_pred
        not simply P_new = (I - KH)
        """
        estimator = BayesianHREstimator(q_bpm=1.0, q_vel=0.01, r_base=25.0)

        # First update: initialize
        est1 = estimator.update(75.0, 100.0, roi_agreement=1.0)
        P1 = estimator.P.copy()

        # Second update: should evolve properly
        est2 = estimator.update(76.0, 95.0, roi_agreement=1.0)
        P2 = estimator.P.copy()

        # Covariance should remain finite
        assert np.isfinite(P2).all(), f"Covariance is not finite: {P2}"
        assert np.isfinite(P1).all(), f"Covariance is not finite: {P1}"

        # Covariance diagonal should remain non-negative
        assert P2[0, 0] >= 0, f"Covariance diagonal P[0,0] is negative: {P2[0,0]}"
        assert P2[1, 1] >= 0, f"Covariance diagonal P[1,1] is negative: {P2[1,1]}"

        # Covariance should be symmetric
        assert np.allclose(P2, P2.T), f"Covariance is not symmetric: {P2}"

        # Posterior variance should be less than prior + process noise
        # (uncertainty should not grow unboundedly with good measurements)
        assert P2[0, 0] <= P1[0, 0] + estimator.q_bpm * 2, \
            f"Variance grew unexpectedly: P1[0,0]={P1[0,0]}, P2[0,0]={P2[0,0]}"

        print("  ✓ Covariance remains finite")
        print("  ✓ Covariance remains PSD (non-negative diagonal)")
        print("  ✓ Covariance is symmetric")
        print("  ✓ Variance behaves reasonably with measurements")

    def test_covariance_positive_semidefinite(self):
        """Test that covariance matrix remains positive semidefinite."""
        estimator = BayesianHREstimator(q_bpm=0.5, q_vel=0.01, r_base=20.0)

        # Run multiple updates
        for i in range(20):
            bpm = 70.0 + np.random.randn() * 2.0
            sqi = 70.0 + np.random.randn() * 10.0
            sqi = np.clip(sqi, 0.0, 100.0)
            estimator.update(bpm, sqi, roi_agreement=0.8)

        P = estimator.P

        # Check eigenvalues for positive semidefiniteness
        eigenvalues = np.linalg.eigvalsh(P)

        # All eigenvalues should be non-negative (within numerical tolerance)
        min_eigenvalue = eigenvalues[0]
        assert min_eigenvalue >= -1e-8, \
            f"Covariance has negative eigenvalue: {min_eigenvalue}"

        print(f"  ✓ Covariance eigenvalues all non-negative: {eigenvalues}")

    def test_repeated_updates_stability(self):
        """Test that repeated updates do not produce pathological covariance behavior."""
        estimator = BayesianHREstimator(q_bpm=1.0, q_vel=0.01, r_base=25.0)

        # Store variance history
        var_history = []

        # Run many updates with consistent measurements
        for i in range(100):
            # Consistent BPM around 75 with good SQI
            bpm = 75.0 + np.random.randn() * 1.0
            sqi = 90.0 + np.random.randn() * 5.0
            sqi = np.clip(sqi, 0.0, 100.0)
            estimator.update(bpm, sqi, roi_agreement=0.9)
            var_history.append(estimator.P[0, 0])

        # Variance should converge to a reasonable value
        final_var = var_history[-1]
        assert 0.1 < final_var < 50.0, \
            f"Variance did not converge to reasonable value: {final_var}"

        # Variance should not oscillate wildly in the last 20 updates
        last_vars = var_history[-20:]
        var_diff = np.max(last_vars) - np.min(last_vars)
        assert var_diff < final_var * 0.5, \
            f"Variance oscillating too much: {var_diff}"

        print(f"  ✓ Variance converged to {final_var:.4f}")
        print(f"  ✓ Variance oscillation in last 20 updates: {var_diff:.4f}")

    def test_low_noise_reduces_uncertainty(self):
        """Test that lower measurement noise leads to reduced posterior uncertainty."""
        estimator_high_noise = BayesianHREstimator(r_base=50.0)
        estimator_low_noise = BayesianHREstimator(r_base=5.0)

        # Run same updates with different noise levels
        np.random.seed(42)
        for _ in range(20):
            bpm = 75.0 + np.random.randn() * 1.0
            sqi = 95.0

            estimator_high_noise.update(bpm, sqi, roi_agreement=1.0)
            estimator_low_noise.update(bpm, sqi, roi_agreement=1.0)

        # Low noise estimator should have lower posterior variance
        var_high = estimator_high_noise.P[0, 0]
        var_low = estimator_low_noise.P[0, 0]

        assert var_low < var_high, \
            f"Low noise should reduce uncertainty: var_low={var_low}, var_high={var_high}"

        print(f"  ✓ Low noise variance ({var_low:.4f}) < High noise variance ({var_high:.4f})")

    def test_uncertainty_aware_confidence_integration(self):
        """Test integration between uncertainty engine and confidence scoring."""
        engine = UncertaintyAwareConfidence(window_size=60)

        # Update with measurements
        for i in range(30):
            bpm = 75.0 + np.random.randn() * 2.0
            sqi = 80.0 + np.random.randn() * 10.0
            sqi = np.clip(sqi, 0.0, 100.0)
            engine.update(bpm, sqi, agreement=0.8)

        confidence, metrics = engine.get_confidence_metrics()

        # Should have non-zero confidence after warmup
        assert confidence > 0, f"Confidence should be > 0 after warmup: {confidence}"

        # Should have uncertainty metrics
        assert 'total_uncertainty' in metrics
        assert 'aleatoric_uncertainty' in metrics
        assert 'epistemic_uncertainty' in metrics

        # Total uncertainty should be finite
        assert np.isfinite(metrics['total_uncertainty'])

        print(f"  ✓ Confidence after warmup: {confidence:.1f}%")
        print(f"  ✓ Total uncertainty: {metrics['total_uncertainty']:.2f}")


def run_tests():
    """Run all uncertainty tests."""
    print("\n" + "=" * 60)
    print("TESTING: rppg_uncertainty module")
    print("=" * 60)

    test_suite = TestBayesianHREstimator()

    # Run tests
    tests = [
        ("Covariance Update Mathematical Correctness",
         test_suite.test_covariance_update_mathematical_correctness),
        ("Covariance Positive Semidefinite",
         test_suite.test_covariance_positive_semidefinite),
        ("Repeated Updates Stability",
         test_suite.test_repeated_updates_stability),
        ("Low Noise Reduces Uncertainty",
         test_suite.test_low_noise_reduces_uncertainty),
        ("Uncertainty Confidence Integration",
         test_suite.test_uncertainty_aware_confidence_integration),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
            print(f"  PASS")
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
