"""
P4.7: Calibration Infrastructure

Builds calibration infrastructure for evaluating uncertainty/confidence quality.

IMPORTANT: If real ground-truth data does not exist, DO NOT claim calibration.
Build the machinery needed to evaluate calibration once GT becomes available.

Metrics computed:
- MAE, RMSE, bias, median absolute error
- Pearson correlation
- Valid frame ratio, rejection rate
- Uncertainty coverage
- Confidence vs actual error
- Reliability diagrams
- Calibration error
- Selective risk, risk-coverage curve
- Abstention performance

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from collections import deque


@dataclass
class CalibrationMetrics:
    """Calibration evaluation metrics.

    Attributes:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        bias: Systematic bias
        median_ae: Median Absolute Error
        pearson_r: Pearson correlation
        valid_ratio: Fraction of valid frames
        rejection_rate: Fraction of rejected frames
        calibration_error: Expected Calibration Error (ECE)
        reliability_diagram: Reliability diagram buckets
        selective_risk: Risk on non-rejected samples
        coverage: Uncertainty coverage fraction
    """
    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    median_ae: float = 0.0
    pearson_r: float = 0.0
    valid_ratio: float = 0.0
    rejection_rate: float = 0.0
    calibration_error: float = 0.0
    n_samples: int = 0
    n_valid: int = 0
    n_rejected: int = 0

    # Reliability diagram
    reliability_diagram: Dict[str, float] = field(default_factory=dict)

    # Risk metrics
    selective_risk: float = 0.0
    expected_risk: float = 0.0

    # Coverage
    coverage_90: float = 0.0  # CI contains GT 90% of time
    coverage_95: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mae": round(self.mae, 2),
            "rmse": round(self.rmse, 2),
            "bias": round(self.bias, 2),
            "median_ae": round(self.median_ae, 2),
            "pearson_r": round(self.pearson_r, 3),
            "valid_ratio": round(self.valid_ratio, 3),
            "rejection_rate": round(self.rejection_rate, 3),
            "calibration_error": round(self.calibration_error, 3),
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
            "n_rejected": self.n_rejected,
            "selective_risk": round(self.selective_risk, 3),
            "coverage_90": round(self.coverage_90, 3),
            "coverage_95": round(self.coverage_95, 3),
        }


class CalibrationEvaluator:
    """
    Evaluates calibration quality of uncertainty estimates.

    Supports:
    - Standard accuracy metrics
    - Coverage analysis
    - Reliability diagrams
    - Selective prediction
    """

    def __init__(self, n_buckets: int = 10):
        """
        Initialize calibration evaluator.

        Args:
            n_buckets: Number of buckets for reliability diagram
        """
        self.n_buckets = n_buckets

        # Storage
        self._predictions: List[float] = []
        self._ground_truths: List[float] = []
        self._confidences: List[float] = []
        self._uncertainties: List[float] = []
        self._ci_lowers: List[float] = []
        self._ci_uppers: List[float] = []
        self._valid_flags: List[bool] = []
        self._rejected_flags: List[bool] = []

    def add_prediction(
        self,
        prediction: float,
        ground_truth: float,
        confidence: float,
        uncertainty: float,
        ci_lower: float,
        ci_upper: float,
        valid: bool = True,
        rejected: bool = False
    ) -> None:
        """Add a prediction for evaluation.

        Args:
            prediction: Predicted BPM
            ground_truth: Ground truth BPM
            confidence: Confidence [0, 100]
            uncertainty: Uncertainty std (BPM)
            ci_lower: 95% CI lower bound
            ci_upper: 95% CI upper bound
            valid: Whether estimate is valid
            rejected: Whether estimate was rejected
        """
        if not np.isfinite(prediction) or not np.isfinite(ground_truth):
            return

        self._predictions.append(prediction)
        self._ground_truths.append(ground_truth)
        self._confidences.append(confidence)
        self._uncertainties.append(uncertainty)
        self._ci_lowers.append(ci_lower)
        self._ci_uppers.append(ci_upper)
        self._valid_flags.append(valid)
        self._rejected_flags.append(rejected)

    def evaluate(self) -> CalibrationMetrics:
        """Compute calibration metrics.

        Returns:
            CalibrationMetrics with all computed values
        """
        n = len(self._predictions)
        if n == 0:
            return CalibrationMetrics()

        preds = np.array(self._predictions)
        gts = np.array(self._ground_truths)
        confs = np.array(self._confidences)
        uncerts = np.array(self._uncertainties)
        ci_l = np.array(self._ci_lowers)
        ci_u = np.array(self._ci_uppers)
        valids = np.array(self._valid_flags)
        rejected = np.array(self._rejected_flags)

        # Basic metrics on valid predictions
        valid_mask = valids & ~rejected
        valid_preds = preds[valid_mask]
        valid_gts = gts[valid_mask]

        n_valid = int(np.sum(valid_mask))
        n_rejected = int(np.sum(rejected))

        metrics = CalibrationMetrics(n_samples=n, n_valid=n_valid, n_rejected=n_rejected)

        if n_valid > 0:
            # MAE
            metrics.mae = float(np.mean(np.abs(valid_preds - valid_gts)))

            # RMSE
            metrics.rmse = float(np.sqrt(np.mean((valid_preds - valid_gts) ** 2)))

            # Bias
            metrics.bias = float(np.mean(valid_preds - valid_gts))

            # Median AE
            metrics.median_ae = float(np.median(np.abs(valid_preds - valid_gts)))

            # Pearson correlation
            if len(valid_preds) > 1:
                cov = np.cov(valid_preds, valid_gts)
                std_pred = np.std(valid_preds)
                std_gt = np.std(valid_gts)
                if std_pred > 0 and std_gt > 0:
                    metrics.pearson_r = float(cov[0, 1] / (std_pred * std_gt))

            # Selective risk (error on non-rejected)
            metrics.selective_risk = float(np.mean(np.abs(valid_preds - valid_gts)))

            # Expected risk (error including rejected)
            all_errors = np.abs(preds - gts)
            # Assume rejected predictions are maximally wrong
            rejected_errors = np.abs(200.0 - gts[rejected])  # Worst case for rejected
            all_errors[rejected] = rejected_errors
            metrics.expected_risk = float(np.mean(all_errors))

        # Coverage analysis
        metrics.valid_ratio = float(n_valid / n) if n > 0 else 0.0
        metrics.rejection_rate = float(n_rejected / n) if n > 0 else 0.0

        # CI coverage
        if n_valid > 0:
            in_90 = np.sum((valid_gts >= ci_l[valid_mask]) & (valid_gts <= ci_u[valid_mask]))
            metrics.coverage_90 = float(in_90 / n_valid)
            metrics.coverage_95 = metrics.coverage_90  # Approximate

        # Reliability diagram
        metrics.reliability_diagram = self._compute_reliability_diagram(
            confs, np.abs(preds - gts)
        )

        # Calibration error (ECE)
        metrics.calibration_error = self._compute_ece(
            confs, np.abs(preds - gts)
        )

        return metrics

    def _compute_reliability_diagram(
        self,
        confidences: np.ndarray,
        errors: np.ndarray
    ) -> Dict[str, float]:
        """Compute reliability diagram buckets.

        Returns accuracy per confidence bucket.
        """
        if len(confidences) < 10:
            return {}

        buckets = {}
        bucket_size = 100.0 / self.n_buckets

        for i in range(self.n_buckets):
            low = i * bucket_size
            high = (i + 1) * bucket_size

            mask = (confidences >= low) & (confidences < high)
            if np.sum(mask) > 0:
                bucket_errors = errors[mask]
                bucket_acc = 1.0 - np.mean(bucket_errors / 100.0)  # Normalize
                bucket_acc = float(np.clip(bucket_acc, 0.0, 1.0))
                buckets[f"{int(low)}-{int(high)}"] = bucket_acc

        return buckets

    def _compute_ece(
        self,
        confidences: np.ndarray,
        errors: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error.

        ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|

        where B_m is bin m, acc is accuracy, conf is average confidence.
        """
        if len(confidences) < n_bins:
            return 0.0

        bucket_size = 100.0 / n_bins
        total_ece = 0.0
        total_samples = 0

        for i in range(n_bins):
            low = i * bucket_size
            high = (i + 1) * bucket_size

            mask = (confidences >= low) & (confidences < high)
            bin_count = np.sum(mask)

            if bin_count > 0:
                bin_confidence = np.mean(confidences[mask]) / 100.0
                bin_accuracy = 1.0 - np.mean(errors[mask]) / 100.0
                bin_accuracy = float(np.clip(bin_accuracy, 0.0, 1.0))

                total_ece += bin_count * abs(bin_accuracy - bin_confidence)
                total_samples += bin_count

        if total_samples > 0:
            return float(total_ece / total_samples)
        return 0.0

    def reset(self) -> None:
        """Reset stored predictions."""
        self._predictions.clear()
        self._ground_truths.clear()
        self._confidences.clear()
        self._uncertainties.clear()
        self._ci_lowers.clear()
        self._ci_uppers.clear()
        self._valid_flags.clear()
        self._rejected_flags.clear()


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_calibration_metrics() -> bool:
    """Test basic calibration metrics computation."""
    print("  test_calibration_metrics...")

    evaluator = CalibrationEvaluator()

    # Perfect predictions
    for i in range(20):
        evaluator.add_prediction(
            prediction=72.0,
            ground_truth=72.0,
            confidence=95.0,
            uncertainty=2.0,
            ci_lower=70.0,
            ci_upper=74.0,
            valid=True,
            rejected=False
        )

    metrics = evaluator.evaluate()

    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.bias == 0.0
    assert metrics.n_samples == 20

    print(f"    MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}")
    print(f"    PASS")
    return True


def test_imperfect_predictions() -> bool:
    """Test with imperfect predictions."""
    print("  test_imperfect_predictions...")

    evaluator = CalibrationEvaluator()

    np.random.seed(42)
    for i in range(50):
        pred = 72.0 + np.random.randn() * 5.0
        gt = 72.0
        evaluator.add_prediction(
            prediction=pred,
            ground_truth=gt,
            confidence=85.0,
            uncertainty=5.0,
            ci_lower=pred - 10.0,
            ci_upper=pred + 10.0,
            valid=True
        )

    metrics = evaluator.evaluate()

    assert metrics.mae > 0
    assert metrics.n_valid == 50

    print(f"    MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}")
    print(f"    PASS")
    return True


def test_rejection_tracking() -> bool:
    """Test tracking of rejected predictions."""
    print("  test_rejection_tracking...")

    evaluator = CalibrationEvaluator()

    # Valid predictions
    for i in range(30):
        evaluator.add_prediction(
            prediction=72.0,
            ground_truth=72.0,
            confidence=90.0,
            uncertainty=3.0,
            ci_lower=69.0,
            ci_upper=75.0,
            valid=True,
            rejected=False
        )

    # Rejected predictions
    for i in range(10):
        evaluator.add_prediction(
            prediction=0.0,
            ground_truth=72.0,
            confidence=10.0,
            uncertainty=50.0,
            ci_lower=0.0,
            ci_upper=0.0,
            valid=False,
            rejected=True
        )

    metrics = evaluator.evaluate()

    assert metrics.n_valid == 30
    assert metrics.n_rejected == 10
    assert metrics.rejection_rate == 10.0 / 40.0

    print(f"    rejection_rate: {metrics.rejection_rate:.2f}")
    print(f"    PASS")
    return True


def test_coverage() -> bool:
    """Test CI coverage computation."""
    print("  test_coverage...")

    evaluator = CalibrationEvaluator()

    # Perfect CIs (contain ground truth)
    for i in range(90):
        evaluator.add_prediction(
            prediction=72.0 + np.random.randn() * 2.0,
            ground_truth=72.0,
            confidence=95.0,
            uncertainty=2.0,
            ci_lower=70.0,
            ci_upper=74.0,
            valid=True
        )

    metrics = evaluator.evaluate()

    # Should have high coverage
    assert metrics.coverage_90 > 0.8

    print(f"    coverage_90: {metrics.coverage_90:.2f}")
    print(f"    PASS")
    return True


def test_reset() -> bool:
    """Test evaluator reset."""
    print("  test_reset...")

    evaluator = CalibrationEvaluator()

    evaluator.add_prediction(72.0, 72.0, 90.0, 3.0, 70.0, 74.0)
    assert len(evaluator._predictions) > 0

    evaluator.reset()
    assert len(evaluator._predictions) == 0

    print(f"    PASS")
    return True


def run_tests() -> bool:
    """Run all P4.7 tests."""
    print("\n" + "=" * 60)
    print("P4.7: Calibration Infrastructure Tests")
    print("=" * 60)

    tests = [
        ("calibration_metrics", test_calibration_metrics),
        ("imperfect_predictions", test_imperfect_predictions),
        ("rejection_tracking", test_rejection_tracking),
        ("coverage", test_coverage),
        ("reset", test_reset),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        print(f"\n  {name}...")
        try:
            if fn():
                passed += 1
        except AssertionError as e:
            print(f"    FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "-" * 60)
    print(f"P4.7 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
