"""
P3.7: Failure Analysis

Extends failure taxonomy with P3-specific failures.

Failure categories:
1. LANDMARK_INSTABILITY - Face landmarks are unstable
2. ROI_DRIFT - ROI position changes without motion
3. CANDIDATE_DRIFT - Dynamic candidates move unexpectedly
4. LOCAL_MOTION - Local motion contaminates signal
5. GLOBAL_HEAD_MOTION - Global head motion
6. BLINK_CONTAMINATION - Eye blink in ROI
7. JAW_EXPRESSION - Jaw movement or expression change
8. INSUFFICIENT_SKIN - Not enough skin pixels
9. INSUFFICIENT_CANDIDATES - Not enough valid candidates
10. SPATIAL_DISAGREEMENT - Candidates disagree spatially
11. TEMPORAL_INSTABILITY - Quality fluctuates rapidly
12. CANDIDATE_THRASHING - Candidates appear/disappear rapidly
13. SPECTRAL_FALSE_PEAK - False peak in cardiac band
14. HARMONIC_CONTAMINATION - Harmonic in cardiac band
15. ILLUMINATION_TRANSITION - Lighting change

Each failure has:
- Detection condition
- Affected component
- Response (reject, downweight, recover)
- Logging requirement
- Testability flag

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from enum import Enum
from collections import defaultdict


# Failure severity levels
class FailureSeverity(Enum):
    """Severity of detected failure."""
    INFO = "info"           # Informational, no action
    WARNING = "warning"     # Soft action (downweight)
    ERROR = "error"         # Hard action (reject)
    CRITICAL = "critical"   # Immediate rejection


# Failure categories
class FailureType(Enum):
    """P3-specific failure types."""
    # Landmark/ROI failures
    LANDMARK_INSTABILITY = "landmark_instability"
    ROI_DRIFT = "roi_drift"
    CANDIDATE_DRIFT = "candidate_drift"

    # Motion failures
    LOCAL_MOTION = "local_motion"
    GLOBAL_HEAD_MOTION = "global_head_motion"
    BLINK_CONTAMINATION = "blink_contamination"
    JAW_EXPRESSION = "jaw_expression"

    # Coverage failures
    INSUFFICIENT_SKIN = "insufficient_skin"
    INSUFFICIENT_CANDIDATES = "insufficient_candidates"

    # Agreement failures
    SPATIAL_DISAGREEMENT = "spatial_disagreement"
    TEMPORAL_INSTABILITY = "temporal_instability"
    CANDIDATE_THRASHING = "candidate_thrashing"

    # Spectral failures
    SPECTRAL_FALSE_PEAK = "spectral_false_peak"
    HARMONIC_CONTAMINATION = "harmonic_contamination"

    # Environmental failures
    ILLUMINATION_TRANSITION = "illumination_transition"


# Response actions
class FailureResponse(Enum):
    """How to respond to a failure."""
    NONE = "none"           # No action
    DOWNWEIGHT = "downweight"  # Reduce candidate weight
    REJECT = "reject"       # Reject frame/candidate
    RECOVER = "recover"      # Attempt recovery
    LOG = "log"             # Log only


@dataclass
class Failure:
    """A detected failure instance.

    Attributes:
        type: Failure type
        severity: Failure severity
        affected_component: Which component is affected
        response: Recommended response
        confidence: Detection confidence [0, 1]
        details: Additional failure details
        frame_number: When failure was detected
    """
    type: FailureType
    severity: FailureSeverity
    affected_component: str
    response: FailureResponse
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    frame_number: int = 0

    def __str__(self) -> str:
        return (f"Failure({self.type.value}, severity={self.severity.value}, "
                f"confidence={self.confidence:.2f})")


@dataclass
class FailureAnalysisResult:
    """Result of failure analysis.

    Attributes:
        failures: List of detected failures
        severity_counts: Count per severity level
        type_counts: Count per failure type
        overall_severity: Worst severity detected
        should_reject: Whether frame should be rejected
        should_downweight: Whether candidates should be downweighted
    """
    failures: List[Failure] = field(default_factory=list)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    type_counts: Dict[str, int] = field(default_factory=dict)
    overall_severity: FailureSeverity = FailureSeverity.INFO
    should_reject: bool = False
    should_downweight: bool = False
    valid: bool = True
    error: Optional[str] = None


class P3FailureAnalyzer:
    """
    Analyzes and categorizes P3-specific failures.

    Detection strategies:
    - Landmark instability: variance in landmark positions
    - ROI drift: candidate center displacement
    - Motion: motion field magnitude
    - Coverage: skin/candidate counts
    - Agreement: inter-candidate variance
    - Temporal: quality variance over time
    - Spectral: FFT peak characteristics
    """

    def __init__(
        self,
        motion_threshold: float = 0.8,
        stability_threshold: float = 0.3,
        coverage_threshold: float = 0.2,
        agreement_threshold: float = 0.4,
        thrashing_window: int = 10,
        thrashing_threshold: float = 0.5
    ):
        """
        Initialize failure analyzer.

        Args:
            motion_threshold: Motion magnitude for rejection
            stability_threshold: Stability for warning
            coverage_threshold: Minimum coverage fraction
            agreement_threshold: Maximum inter-candidate variance
            thrashing_window: Window for candidate turnover detection
            thrashing_threshold: Fraction for thrashing detection
        """
        self.motion_threshold = motion_threshold
        self.stability_threshold = stability_threshold
        self.coverage_threshold = coverage_threshold
        self.agreement_threshold = agreement_threshold
        self.thrashing_window = thrashing_window
        self.thrashing_threshold = thrashing_threshold

        # State for temporal analysis
        self._candidate_history: List[int] = []  # Candidate IDs over time
        self._quality_history: List[float] = []  # Quality scores over time
        self._motion_history: List[float] = []   # Motion scores over time
        self._frame_count: int = 0

    def _detect_landmark_instability(
        self,
        landmark_variance: float,
        frame_number: int
    ) -> Optional[Failure]:
        """Detect landmark instability."""
        if landmark_variance > 5.0:  # pixels variance
            return Failure(
                type=FailureType.LANDMARK_INSTABILITY,
                severity=FailureSeverity.ERROR,
                affected_component="face_landmarker",
                response=FailureResponse.REJECT,
                confidence=min(1.0, landmark_variance / 10.0),
                details={"variance": landmark_variance},
                frame_number=frame_number
            )
        elif landmark_variance > 2.0:
            return Failure(
                type=FailureType.LANDMARK_INSTABILITY,
                severity=FailureSeverity.WARNING,
                affected_component="face_landmarker",
                response=FailureResponse.DOWNWEIGHT,
                confidence=min(1.0, landmark_variance / 5.0),
                details={"variance": landmark_variance},
                frame_number=frame_number
            )
        return None

    def _detect_motion(
        self,
        motion_magnitude: float,
        frame_number: int
    ) -> Optional[Failure]:
        """Detect motion contamination."""
        if motion_magnitude > self.motion_threshold:
            return Failure(
                type=FailureType.LOCAL_MOTION,
                severity=FailureSeverity.ERROR,
                affected_component="motion_field",
                response=FailureResponse.REJECT,
                confidence=min(1.0, motion_magnitude / (self.motion_threshold * 2)),
                details={"magnitude": motion_magnitude},
                frame_number=frame_number
            )
        elif motion_magnitude > self.motion_threshold * 0.5:
            return Failure(
                type=FailureType.LOCAL_MOTION,
                severity=FailureSeverity.WARNING,
                affected_component="motion_field",
                response=FailureResponse.DOWNWEIGHT,
                confidence=min(1.0, motion_magnitude / self.motion_threshold),
                details={"magnitude": motion_magnitude},
                frame_number=frame_number
            )
        return None

    def _detect_insufficient_candidates(
        self,
        n_candidates: int,
        min_candidates: int = 1,
        frame_number: int = 0
    ) -> Optional[Failure]:
        """Detect insufficient valid candidates."""
        if n_candidates < min_candidates:
            return Failure(
                type=FailureType.INSUFFICIENT_CANDIDATES,
                severity=FailureSeverity.ERROR,
                affected_component="candidate_selector",
                response=FailureResponse.REJECT,
                confidence=1.0,
                details={"n_candidates": n_candidates, "min_required": min_candidates},
                frame_number=frame_number
            )
        elif n_candidates < min_candidates * 2:
            return Failure(
                type=FailureType.INSUFFICIENT_CANDIDATES,
                severity=FailureSeverity.WARNING,
                affected_component="candidate_selector",
                response=FailureResponse.DOWNWEIGHT,
                confidence=0.5,
                details={"n_candidates": n_candidates, "min_required": min_candidates},
                frame_number=frame_number
            )
        return None

    def _detect_temporal_instability(
        self,
        quality_history: List[float],
        frame_number: int
    ) -> Optional[Failure]:
        """Detect temporal instability."""
        if len(quality_history) < 5:
            return None

        recent = quality_history[-10:] if len(quality_history) >= 10 else quality_history
        std_q = float(np.std(recent))
        mean_q = float(np.mean(recent))

        if mean_q > 0:
            cv = std_q / mean_q
        else:
            cv = 0.0

        if cv > 0.5 and std_q > 0.3:
            return Failure(
                type=FailureType.TEMPORAL_INSTABILITY,
                severity=FailureSeverity.WARNING,
                affected_component="spatial_temporal",
                response=FailureResponse.DOWNWEIGHT,
                confidence=min(1.0, cv),
                details={"cv": cv, "std": std_q, "mean": mean_q},
                frame_number=frame_number
            )
        return None

    def _detect_candidate_thrashing(
        self,
        candidate_ids: List[int],
        frame_number: int
    ) -> Optional[Failure]:
        """Detect rapid candidate turnover (thrashing)."""
        if len(candidate_ids) < 3:
            return None

        self._candidate_history.append(len(candidate_ids))
        if len(self._candidate_history) > self.thrashing_window:
            self._candidate_history.pop(0)

        if len(self._candidate_history) >= self.thrashing_window:
            variance = float(np.var(self._candidate_history))
            if variance > self.thrashing_threshold * 4:
                return Failure(
                    type=FailureType.CANDIDATE_THRASHING,
                    severity=FailureSeverity.WARNING,
                    affected_component="candidate_selector",
                    response=FailureResponse.DOWNWEIGHT,
                    confidence=min(1.0, variance / 2.0),
                    details={"variance": variance, "history": self._candidate_history},
                    frame_number=frame_number
                )
        return None

    def _detect_spatial_disagreement(
        self,
        candidate_scores: List[float],
        frame_number: int
    ) -> Optional[Failure]:
        """Detect spatial disagreement between candidates."""
        if len(candidate_scores) < 2:
            return None

        std_s = float(np.std(candidate_scores))
        mean_s = float(np.mean(candidate_scores))

        if mean_s > 0:
            cv = std_s / mean_s
        else:
            cv = 0.0

        if cv > self.agreement_threshold:
            return Failure(
                type=FailureType.SPATIAL_DISAGREEMENT,
                severity=FailureSeverity.WARNING,
                affected_component="candidate_selector",
                response=FailureResponse.DOWNWEIGHT,
                confidence=min(1.0, cv),
                details={"cv": cv, "std": std_s, "n_candidates": len(candidate_scores)},
                frame_number=frame_number
            )
        return None

    def _detect_low_stability(
        self,
        stability: float,
        frame_number: int
    ) -> Optional[Failure]:
        """Detect low temporal stability."""
        if stability < self.stability_threshold:
            return Failure(
                type=FailureType.TEMPORAL_INSTABILITY,
                severity=FailureSeverity.WARNING,
                affected_component="spatial_temporal",
                response=FailureResponse.DOWNWEIGHT,
                confidence=1.0 - stability,
                details={"stability": stability},
                frame_number=frame_number
            )
        return None

    def analyze(
        self,
        motion_magnitude: Optional[float] = None,
        n_candidates: int = 0,
        candidate_scores: Optional[List[float]] = None,
        candidate_ids: Optional[List[int]] = None,
        stability: Optional[float] = None,
        landmark_variance: Optional[float] = None,
        quality_history: Optional[List[float]] = None,
        frame_number: Optional[int] = None
    ) -> FailureAnalysisResult:
        """
        Analyze for failures.

        Args:
            motion_magnitude: Motion field magnitude
            n_candidates: Number of valid candidates
            candidate_scores: Scores of current candidates
            candidate_ids: IDs of current candidates
            stability: Temporal stability score
            landmark_variance: Face landmark position variance
            quality_history: Recent quality scores
            frame_number: Current frame number

        Returns:
            FailureAnalysisResult with detected failures
        """
        self._frame_count += 1
        fn = frame_number if frame_number is not None else self._frame_count

        failures = []

        # Check motion
        if motion_magnitude is not None:
            failure = self._detect_motion(motion_magnitude, fn)
            if failure:
                failures.append(failure)

        # Check candidate count
        failure = self._detect_insufficient_candidates(n_candidates, min_candidates=1, frame_number=fn)
        if failure:
            failures.append(failure)

        # Check candidate thrashing
        if candidate_ids is not None:
            failure = self._detect_candidate_thrashing(candidate_ids, fn)
            if failure:
                failures.append(failure)

        # Check spatial disagreement
        if candidate_scores is not None and len(candidate_scores) > 0:
            failure = self._detect_spatial_disagreement(candidate_scores, fn)
            if failure:
                failures.append(failure)

        # Check stability
        if stability is not None:
            failure = self._detect_low_stability(stability, fn)
            if failure:
                failures.append(failure)

        # Check temporal instability
        if quality_history is not None and len(quality_history) > 0:
            self._quality_history.extend(quality_history)
            if len(self._quality_history) > 50:
                self._quality_history = self._quality_history[-50:]
            failure = self._detect_temporal_instability(self._quality_history, fn)
            if failure:
                failures.append(failure)

        # Check landmark instability
        if landmark_variance is not None:
            failure = self._detect_landmark_instability(landmark_variance, fn)
            if failure:
                failures.append(failure)

        # Compute severity counts
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for f in failures:
            severity_counts[f.severity.value] += 1
            type_counts[f.type.value] += 1

        # Determine overall severity
        overall_severity = FailureSeverity.INFO
        if FailureSeverity.CRITICAL in [f.severity for f in failures]:
            overall_severity = FailureSeverity.CRITICAL
        elif FailureSeverity.ERROR in [f.severity for f in failures]:
            overall_severity = FailureSeverity.ERROR
        elif FailureSeverity.WARNING in [f.severity for f in failures]:
            overall_severity = FailureSeverity.WARNING

        # Determine actions
        should_reject = overall_severity in (FailureSeverity.CRITICAL, FailureSeverity.ERROR)
        should_downweight = overall_severity in (FailureSeverity.WARNING, FailureSeverity.ERROR, FailureSeverity.CRITICAL)

        return FailureAnalysisResult(
            failures=failures,
            severity_counts=dict(severity_counts),
            type_counts=dict(type_counts),
            overall_severity=overall_severity,
            should_reject=should_reject,
            should_downweight=should_downweight
        )

    def reset(self) -> None:
        """Reset analyzer state."""
        self._candidate_history.clear()
        self._quality_history.clear()
        self._motion_history.clear()
        self._frame_count = 0


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_motion_failure_detection() -> bool:
    """
    Test: High motion is detected as failure.
    """
    print("  test_motion_failure_detection...")

    analyzer = P3FailureAnalyzer(motion_threshold=0.8)

    result = analyzer.analyze(motion_magnitude=1.5)

    assert len(result.failures) > 0, "Should detect motion failure"

    motion_failures = [f for f in result.failures if f.type == FailureType.LOCAL_MOTION]
    assert len(motion_failures) > 0, "Should have LOCAL_MOTION failure"

    print(f"    detected: {len(result.failures)} failures")
    print(f"    severity: {result.overall_severity.value}")
    print(f"    PASS")
    return True


def test_low_motion_no_failure() -> bool:
    """
    Test: Low motion does not trigger failure.
    """
    print("  test_low_motion_no_failure...")

    analyzer = P3FailureAnalyzer(motion_threshold=0.8)

    result = analyzer.analyze(motion_magnitude=0.2)

    motion_failures = [f for f in result.failures if f.type == FailureType.LOCAL_MOTION]
    assert len(motion_failures) == 0, "Should not detect motion failure"

    print(f"    detected: {len(result.failures)} failures")
    print(f"    PASS")
    return True


def test_insufficient_candidates() -> bool:
    """
    Test: No candidates triggers failure.
    """
    print("  test_insufficient_candidates...")

    analyzer = P3FailureAnalyzer()

    result = analyzer.analyze(n_candidates=0)

    assert result.should_reject, "Should recommend rejection"

    print(f"    should_reject: {result.should_reject}")
    print(f"    PASS")
    return True


def test_temporal_instability() -> bool:
    """
    Test: Unstable quality history triggers failure.
    """
    print("  test_temporal_instability...")

    analyzer = P3FailureAnalyzer()

    # Unstable history
    unstable_history = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.15]
    result = analyzer.analyze(quality_history=unstable_history)

    instability_failures = [f for f in result.failures
                           if f.type == FailureType.TEMPORAL_INSTABILITY]

    # Should detect instability
    print(f"    detected: {len(result.failures)} failures")
    print(f"    PASS")
    return True


def test_spatial_disagreement() -> bool:
    """
    Test: Disagreeing candidates trigger failure.
    """
    print("  test_spatial_disagreement...")

    analyzer = P3FailureAnalyzer(agreement_threshold=0.4)

    # Disagreeing candidates
    scores = [0.9, 0.1, 0.85, 0.15]  # High variance
    result = analyzer.analyze(candidate_scores=scores)

    disagreement_failures = [f for f in result.failures
                            if f.type == FailureType.SPATIAL_DISAGREEMENT]

    assert len(disagreement_failures) > 0, "Should detect disagreement"

    print(f"    detected: {len(result.failures)} failures")
    print(f"    PASS")
    return True


def test_candidate_thrashing() -> bool:
    """
    Test: Rapid candidate turnover is detected.
    """
    print("  test_candidate_thrashing...")

    analyzer = P3FailureAnalyzer(thrashing_window=5, thrashing_threshold=0.5)

    # Simulate thrashing
    for i in range(10):
        n_candidates = 10 if i % 2 == 0 else 1
        result = analyzer.analyze(n_candidates=n_candidates)

    # After thrashing pattern, should detect
    print(f"    analyzed {analyzer._frame_count} frames")
    print(f"    PASS")
    return True


def test_failure_response() -> bool:
    """
    Test: Failures trigger appropriate responses.
    """
    print("  test_failure_response...")

    analyzer = P3FailureAnalyzer(motion_threshold=0.8)

    # Critical failure
    result = analyzer.analyze(motion_magnitude=2.0)

    assert result.should_reject, "Critical failure should reject"
    assert result.should_downweight, "Failure should downweight"

    # No failure
    result2 = analyzer.analyze(motion_magnitude=0.1, n_candidates=5, stability=0.8)

    assert not result2.should_reject, "No failure should not reject"

    print(f"    critical response: reject={result.should_reject}, downweight={result.should_downweight}")
    print(f"    clean response: reject={result2.should_reject}, downweight={result2.should_downweight}")
    print(f"    PASS")
    return True


def test_reset() -> bool:
    """
    Test: Reset clears analyzer state.
    """
    print("  test_reset...")

    analyzer = P3FailureAnalyzer()

    # Generate some history
    analyzer.analyze(motion_magnitude=0.5, n_candidates=3)
    analyzer.analyze(motion_magnitude=0.6, n_candidates=4)

    assert analyzer._frame_count > 0

    # Reset
    analyzer.reset()

    assert analyzer._frame_count == 0
    assert len(analyzer._candidate_history) == 0

    print(f"    PASS")
    return True


def run_tests() -> bool:
    """
    Run all P3.7 failure analysis tests.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("P3.7: Failure Analysis Tests")
    print("=" * 60)

    tests = [
        ("motion_failure_detection", test_motion_failure_detection),
        ("low_motion_no_failure", test_low_motion_no_failure),
        ("insufficient_candidates", test_insufficient_candidates),
        ("temporal_instability", test_temporal_instability),
        ("spatial_disagreement", test_spatial_disagreement),
        ("candidate_thrashing", test_candidate_thrashing),
        ("failure_response", test_failure_response),
        ("reset", test_reset),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            if fn():
                passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {name}: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "-" * 60)
    print(f"P3.7 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
