"""
P4.9: Reliability Failure Analysis

Expands failure taxonomy for P4 reliability/uncertainty system.

Failure categories:
- HIGH_CONFIDENCE_WRONG_ESTIMATE
- LOW_CONFIDENCE_CORRECT_ESTIMATE
- CONFIDENCE_COLLAPSE
- CONFIDENCE_STAGNATION
- UNCERTAINTY_UNDERESTIMATION
- UNCERTAINTY_OVERESTIMATION
- CIRCULAR_CONFIDENCE
- CORRELATED_EVIDENCE_DOUBLE_COUNT
- TEMPORAL_CONFIDENCE_LAG
- FALSE_RECOVERY
- FALSE_REJECTION
- PLAUSIBILITY_GATE_FALSE_POSITIVE
- PLAUSIBILITY_GATE_FALSE_NEGATIVE
- CANDIDATE_RELIABILITY_MISMATCH
- SPECTRAL_CONFIDENCE_FAILURE
- MOTION_CONFIDENCE_FAILURE

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class P4FailureType(Enum):
    """P4-specific reliability failure types."""
    # Confidence errors
    HIGH_CONF_WRONG = "high_confidence_wrong"
    LOW_CONF_CORRECT = "low_confidence_correct"
    CONFIDENCE_COLLAPSE = "confidence_collapse"
    CONFIDENCE_STAGNATION = "confidence_stagnation"

    # Uncertainty errors
    UNCERTAINTY_UNDERESTIMATION = "uncertainty_underestimation"
    UNCERTAINTY_OVERESTIMATION = "uncertainty_overestimation"
    CIRCULAR_CONFIDENCE = "circular_confidence"
    DOUBLE_COUNT = "correlated_evidence_double_count"

    # Temporal errors
    TEMPORAL_LAG = "temporal_confidence_lag"
    FALSE_RECOVERY = "false_recovery"
    FALSE_REJECTION = "false_rejection"

    # Plausibility gate errors
    PLAUSIBILITY_FP = "plausibility_gate_false_positive"
    PLAUSIBILITY_FN = "plausibility_gate_false_negative"

    # Component failures
    CANDIDATE_MISMATCH = "candidate_reliability_mismatch"
    SPECTRAL_CONF_FAILURE = "spectral_confidence_failure"
    MOTION_CONF_FAILURE = "motion_confidence_failure"


@dataclass
class P4Failure:
    """P4 reliability failure instance."""
    failure_type: P4FailureType
    severity: str = "warning"  # info, warning, error
    confidence: float = 0.0  # Detection confidence
    details: Dict[str, Any] = field(default_factory=dict)
    frame: int = 0

    def __str__(self) -> str:
        return f"P4Failure({self.failure_type.value}, conf={self.confidence:.2f})"


@dataclass
class P4FailureAnalysisResult:
    """Complete P4 failure analysis result."""
    failures: List[P4Failure] = field(default_factory=list)
    n_failures: int = 0
    severity_counts: Dict[str, int] = field(default_factory=dict)
    type_counts: Dict[str, int] = field(default_factory=dict)
    valid: bool = True


class P4FailureAnalyzer:
    """
    Analyzes P4-specific reliability failures.
    """

    def __init__(
        self,
        confidence_high_threshold: float = 80.0,
        confidence_low_threshold: float = 30.0,
        lag_threshold: int = 10,
        stagnation_threshold: int = 30
    ):
        """
        Initialize failure analyzer.

        Args:
            confidence_high_threshold: High confidence threshold
            confidence_low_threshold: Low confidence threshold
            lag_threshold: Frames for temporal lag detection
            stagnation_threshold: Frames for stagnation detection
        """
        self.high_threshold = confidence_high_threshold
        self.low_threshold = confidence_low_threshold
        self.lag_threshold = lag_threshold
        self.stagnation_threshold = stagnation_threshold

        # History tracking
        self._confidence_history: List[float] = []
        self._evidence_history: List[float] = []
        self._frame_count = 0

    def analyze(
        self,
        bpm_error: float,
        confidence: float,
        uncertainty: float,
        predicted_uncertainty: float,
        evidence_trend: float,
        plausibility_state: Optional[str] = None,
        frame: int = 0
    ) -> P4FailureAnalysisResult:
        """Analyze for P4-specific failures.

        Args:
            bpm_error: Prediction error (|pred - gt|)
            confidence: Current confidence
            uncertainty: Actual uncertainty
            predicted_uncertainty: Predicted/estimated uncertainty
            evidence_trend: Evidence trend direction
            plausibility_state: Plausibility state
            frame: Frame number

        Returns:
            P4FailureAnalysisResult with detected failures
        """
        self._frame_count += 1
        failures = []

        # Track history
        self._confidence_history.append(confidence)
        if len(self._confidence_history) > 50:
            self._confidence_history.pop(0)
        self._evidence_history.append(evidence_trend)
        if len(self._evidence_history) > 50:
            self._evidence_history.pop(0)

        # HIGH_CONFIDENCE_WRONG: High confidence but wrong estimate
        if confidence > self.high_threshold and bpm_error > 10.0:
            failures.append(P4Failure(
                failure_type=P4FailureType.HIGH_CONF_WRONG,
                severity="error",
                confidence=min(1.0, confidence / 100.0),
                details={"error": bpm_error, "confidence": confidence},
                frame=frame
            ))

        # LOW_CONFIDENCE_CORRECT: Low confidence but correct estimate
        if confidence < self.low_threshold and bpm_error < 5.0:
            failures.append(P4Failure(
                failure_type=P4FailureType.LOW_CONF_CORRECT,
                severity="warning",
                confidence=1.0 - confidence / 100.0,
                details={"error": bpm_error, "confidence": confidence},
                frame=frame
            ))

        # UNCERTAINTY_UNDERESTIMATION: Predicted uncertainty too low
        if predicted_uncertainty > 0 and uncertainty > predicted_uncertainty * 2.0:
            failures.append(P4Failure(
                failure_type=P4FailureType.UNCERTAINTY_UNDERESTIMATION,
                severity="warning",
                confidence=min(1.0, uncertainty / (predicted_uncertainty + 1e-6)),
                details={
                    "actual": uncertainty,
                    "predicted": predicted_uncertainty
                },
                frame=frame
            ))

        # UNCERTAINTY_OVERESTIMATION: Predicted uncertainty too high
        if predicted_uncertainty > 0 and predicted_uncertainty > uncertainty * 3.0:
            failures.append(P4Failure(
                failure_type=P4FailureType.UNCERTAINTY_OVERESTIMATION,
                severity="info",
                confidence=min(1.0, predicted_uncertainty / (uncertainty + 1e-6)),
                details={
                    "actual": uncertainty,
                    "predicted": predicted_uncertainty
                },
                frame=frame
            ))

        # CONFIDENCE_STAGNATION: Confidence doesn't change for too long
        if len(self._confidence_history) >= self.stagnation_threshold:
            recent = self._confidence_history[-self.stagnation_threshold:]
            if np.std(recent) < 1.0:  # Nearly constant
                failures.append(P4Failure(
                    failure_type=P4FailureType.CONFIDENCE_STAGNATION,
                    severity="warning",
                    confidence=0.8,
                    details={"recent_std": float(np.std(recent))},
                    frame=frame
                ))

        # TEMPORAL_LAG: Evidence trend but confidence doesn't follow
        if len(self._evidence_history) >= self.lag_threshold:
            recent_trend = np.mean(self._evidence_history[-self.lag_threshold:])
            recent_conf_change = self._confidence_history[-1] - np.mean(
                self._confidence_history[-self.lag_threshold:]
            )

            if abs(recent_trend) > 0.1 and abs(recent_conf_change) < 2.0:
                failures.append(P4Failure(
                    failure_type=P4FailureType.TEMPORAL_LAG,
                    severity="warning",
                    confidence=0.7,
                    details={
                        "evidence_trend": recent_trend,
                        "conf_change": recent_conf_change
                    },
                    frame=frame
                ))

        # Aggregate results
        severity_counts = {}
        type_counts = {}
        for f in failures:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
            type_counts[f.failure_type.value] = type_counts.get(f.failure_type.value, 0) + 1

        return P4FailureAnalysisResult(
            failures=failures,
            n_failures=len(failures),
            severity_counts=severity_counts,
            type_counts=type_counts
        )

    def reset(self) -> None:
        """Reset analyzer state."""
        self._confidence_history.clear()
        self._evidence_history.clear()
        self._frame_count = 0


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_high_confidence_wrong() -> bool:
    """Test detection of high confidence wrong estimate."""
    print("  test_high_confidence_wrong...")

    analyzer = P4FailureAnalyzer()

    result = analyzer.analyze(
        bpm_error=15.0,  # Large error
        confidence=90.0,  # High confidence
        uncertainty=5.0,
        predicted_uncertainty=3.0,
        evidence_trend=0.0,
        frame=1
    )

    high_conf_failures = [
        f for f in result.failures
        if f.failure_type == P4FailureType.HIGH_CONF_WRONG
    ]

    assert len(high_conf_failures) > 0

    print(f"    detected: {len(high_conf_failures)} failures")
    print(f"    PASS")
    return True


def test_uncertainty_underestimation() -> bool:
    """Test detection of underestimated uncertainty."""
    print("  test_uncertainty_underestimation...")

    analyzer = P4FailureAnalyzer()

    result = analyzer.analyze(
        bpm_error=5.0,
        confidence=70.0,
        uncertainty=15.0,  # Actual uncertainty high
        predicted_uncertainty=5.0,  # But we predicted low
        evidence_trend=0.0,
        frame=1
    )

    under_failures = [
        f for f in result.failures
        if f.failure_type == P4FailureType.UNCERTAINTY_UNDERESTIMATION
    ]

    assert len(under_failures) > 0

    print(f"    detected: {len(under_failures)} failures")
    print(f"    PASS")
    return True


def test_confidence_stagnation() -> bool:
    """Test detection of confidence stagnation."""
    print("  test_confidence_stagnation...")

    analyzer = P4FailureAnalyzer(stagnation_threshold=10)

    # Add history with stagnant confidence
    for i in range(10):
        analyzer.analyze(
            bpm_error=5.0,
            confidence=50.0,  # Same confidence
            uncertainty=5.0,
            predicted_uncertainty=5.0,
            evidence_trend=0.0,
            frame=i
        )

    result = analyzer.analyze(
        bpm_error=5.0,
        confidence=50.5,  # Almost same
        uncertainty=5.0,
        predicted_uncertainty=5.0,
        evidence_trend=0.0,
        frame=10
    )

    stagnation_failures = [
        f for f in result.failures
        if f.failure_type == P4FailureType.CONFIDENCE_STAGNATION
    ]

    assert len(stagnation_failures) > 0

    print(f"    detected: {len(stagnation_failures)} failures")
    print(f"    PASS")
    return True


def test_no_failure_on_good_estimate() -> bool:
    """Test that good estimates don't trigger failures."""
    print("  test_no_failure_on_good_estimate...")

    analyzer = P4FailureAnalyzer()

    result = analyzer.analyze(
        bpm_error=2.0,  # Small error
        confidence=85.0,  # Appropriate confidence
        uncertainty=3.0,
        predicted_uncertainty=3.0,
        evidence_trend=0.0,
        frame=1
    )

    # Should have no failures
    assert result.n_failures == 0

    print(f"    no failures (as expected)")
    print(f"    PASS")
    return True


def run_tests() -> bool:
    """Run all P4.9 tests."""
    print("\n" + "=" * 60)
    print("P4.9: P4 Failure Analysis Tests")
    print("=" * 60)

    tests = [
        ("high_confidence_wrong", test_high_confidence_wrong),
        ("uncertainty_underestimation", test_uncertainty_underestimation),
        ("confidence_stagnation", test_confidence_stagnation),
        ("no_failure_on_good_estimate", test_no_failure_on_good_estimate),
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
    print(f"P4.9 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
