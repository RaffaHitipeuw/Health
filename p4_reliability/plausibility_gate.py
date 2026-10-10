"""
P4.5: Physiological Plausibility Gate

Implements a conservative physiological plausibility layer.

Purpose:
- Detect clearly implausible or internally inconsistent estimates
- Distinguish: PLAUSIBLE, UNCERTAIN, INVALID

IMPORTANT: This is NOT a medical diagnosis system.
This layer only evaluates whether the measured signal behaves
plausibly within the configured measurement context.

Do NOT label users with:
- arrhythmia
- disease
- stress disorder
- cardiovascular disease
- hypertension
- etc.

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum


class PlausibilityState(Enum):
    """Physiological plausibility states."""
    PLAUSIBLE = "plausible"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"


@dataclass
class PlausibilityCheck:
    """Single plausibility check result.

    Attributes:
        check_name: Name of the check
        passed: Whether check passed
        severity: How severe the failure is
        value: Measured value
        threshold: Threshold used
        details: Additional information
    """
    check_name: str
    passed: bool
    severity: str = "info"  # "info", "warning", "error"
    value: float = 0.0
    threshold: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlausibilityResult:
    """Complete plausibility gate result.

    Attributes:
        state: Final plausibility state
        checks: List of individual checks
        passed: Whether all checks passed
        n_errors: Number of error-level failures
        n_warnings: Number of warning-level failures
        bpm: Tested BPM value
        details: Additional information
    """
    state: PlausibilityState = PlausibilityState.PLAUSIBLE
    checks: List[PlausibilityCheck] = field(default_factory=list)
    passed: bool = True
    n_errors: int = 0
    n_warnings: int = 0
    bpm: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def get_check(self, name: str) -> Optional[PlausibilityCheck]:
        """Get check by name."""
        for check in self.checks:
            if check.check_name == name:
                return check
        return None


class PhysiologicalPlausibilityGate:
    """
    Evaluates physiological plausibility of estimates.

    Checks performed:
    1. BPM range check
    2. Frame-to-frame jump check
    3. Temporal trajectory check
    4. Spectral instability check
    5. Harmonic consistency check
    6. Candidate agreement check
    7. Measurement timeout check
    """

    def __init__(
        self,
        min_bpm: float = 40.0,
        max_bpm: float = 200.0,
        resting_min: float = 50.0,
        resting_max: float = 100.0,
        max_jump_bpm: float = 20.0,
        max_jump_pct: float = 0.25,
        spectral_instability_threshold: float = 0.5,
        harmonic_ratio_threshold: float = 0.5,
        candidate_cv_threshold: float = 0.3,
        timeout_frames: int = 300
    ):
        """
        Initialize plausibility gate.

        Args:
            min_bpm: Absolute minimum physiologically plausible BPM
            max_bpm: Absolute maximum physiologically plausible BPM
            resting_min: Lower end of typical resting heart rate
            resting_max: Upper end of typical resting heart rate
            max_jump_bpm: Maximum frame-to-frame change (BPM)
            max_jump_pct: Maximum frame-to-frame change (percentage)
            spectral_instability_threshold: Spectral instability threshold
            harmonic_ratio_threshold: Harmonic contamination threshold
            candidate_cv_threshold: Candidate agreement CV threshold
            timeout_frames: Frames without valid estimate before INVALID
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.resting_min = resting_min
        self.resting_max = resting_max
        self.max_jump_bpm = max_jump_bpm
        self.max_jump_pct = max_jump_pct
        self.spectral_instability_threshold = spectral_instability_threshold
        self.harmonic_ratio_threshold = harmonic_ratio_threshold
        self.candidate_cv_threshold = candidate_cv_threshold
        self.timeout_frames = timeout_frames

        # State tracking
        self._last_bpm: Optional[float] = None
        self._recent_bpms: List[float] = []
        self._frame_count: int = 0
        self._invalid_count: int = 0

    def evaluate(
        self,
        bpm: float,
        previous_bpm: Optional[float] = None,
        spectral_instability: float = 0.0,
        harmonic_ratio: float = 0.0,
        candidate_cv: float = 0.0,
        candidate_estimates: Optional[List[float]] = None,
        confidence: float = 0.0
    ) -> PlausibilityResult:
        """Evaluate physiological plausibility.

        Args:
            bpm: Current BPM estimate
            previous_bpm: Previous BPM estimate
            spectral_instability: Spectral instability [0, 1]
            harmonic_ratio: Harmonic contamination ratio [0, 1]
            candidate_cv: Candidate estimate CV [0, 1]
            candidate_estimates: List of candidate BPM estimates
            confidence: Current confidence [0, 100]

        Returns:
            PlausibilityResult with evaluation
        """
        self._frame_count += 1
        checks = []

        # 1. BPM Range Check
        range_check = self._check_bpm_range(bpm)
        checks.append(range_check)

        # 2. Frame-to-Frame Jump Check
        jump_check = self._check_frame_jump(bpm, previous_bpm)
        checks.append(jump_check)

        # 3. Temporal Trajectory Check
        trajectory_check = self._check_trajectory(bpm)
        checks.append(trajectory_check)

        # 4. Spectral Instability Check
        spectral_check = self._check_spectral_instability(spectral_instability)
        checks.append(spectral_check)

        # 5. Harmonic Consistency Check
        harmonic_check = self._check_harmonic_consistency(harmonic_ratio)
        checks.append(harmonic_check)

        # 6. Candidate Agreement Check
        candidate_check = self._check_candidate_agreement(candidate_cv, candidate_estimates)
        checks.append(candidate_check)

        # 7. Measurement Timeout Check
        timeout_check = self._check_timeout(confidence)
        checks.append(timeout_check)

        # Aggregate results
        n_errors = sum(1 for c in checks if c.severity == "error")
        n_warnings = sum(1 for c in checks if c.severity == "warning")

        # Determine state
        if n_errors > 0:
            state = PlausibilityState.INVALID
        elif n_warnings > 0:
            state = PlausibilityState.UNCERTAIN
        else:
            state = PlausibilityState.PLAUSIBLE

        # Update tracking
        self._last_bpm = bpm
        if state != PlausibilityState.INVALID:
            self._recent_bpms.append(bpm)
            if len(self._recent_bpms) > 30:
                self._recent_bpms.pop(0)
            self._invalid_count = 0
        else:
            self._invalid_count += 1

        return PlausibilityResult(
            state=state,
            checks=checks,
            passed=state == PlausibilityState.PLAUSIBLE,
            n_errors=n_errors,
            n_warnings=n_warnings,
            bpm=bpm,
            details={
                "frame": self._frame_count,
                "last_bpm": self._last_bpm,
                "invalid_count": self._invalid_count
            }
        )

    def _check_bpm_range(self, bpm: float) -> PlausibilityCheck:
        """Check if BPM is in physiologically plausible range."""
        if self.min_bpm <= bpm <= self.max_bpm:
            return PlausibilityCheck(
                check_name="bpm_range",
                passed=True,
                value=bpm,
                threshold=(self.min_bpm, self.max_bpm)
            )
        else:
            return PlausibilityCheck(
                check_name="bpm_range",
                passed=False,
                severity="error",
                value=bpm,
                threshold=(self.min_bpm, self.max_bpm),
                details={"reason": "outside_physiological_range"}
            )

    def _check_frame_jump(
        self,
        bpm: float,
        previous_bpm: Optional[float]
    ) -> PlausibilityCheck:
        """Check for implausible frame-to-frame jumps."""
        if previous_bpm is None or previous_bpm <= 0:
            return PlausibilityCheck(
                check_name="frame_jump",
                passed=True,
                value=0.0,
                threshold=(0.0, self.max_jump_bpm)
            )

        jump = abs(bpm - previous_bpm)
        jump_pct = jump / max(previous_bpm, 1.0)

        passed = jump <= self.max_jump_bpm and jump_pct <= self.max_jump_pct

        return PlausibilityCheck(
            check_name="frame_jump",
            passed=passed,
            severity="error" if not passed else "info",
            value=jump,
            threshold=self.max_jump_bpm,
            details={
                "jump_bpm": jump,
                "jump_pct": jump_pct,
                "previous_bpm": previous_bpm
            }
        )

    def _check_trajectory(self, bpm: float) -> PlausibilityCheck:
        """Check temporal trajectory stability."""
        if len(self._recent_bpms) < 5:
            return PlausibilityCheck(
                check_name="trajectory",
                passed=True,
                value=0.0,
                threshold=0.0
            )

        recent = self._recent_bpms[-10:]  # Last 10 estimates
        trajectory_std = float(np.std(recent))

        # Trajectory is suspicious if very high variance
        passed = trajectory_std < 15.0

        return PlausibilityCheck(
            check_name="trajectory",
            passed=passed,
            severity="warning" if not passed else "info",
            value=trajectory_std,
            threshold=15.0,
            details={"recent_mean": float(np.mean(recent))}
        )

    def _check_spectral_instability(
        self,
        instability: float
    ) -> PlausibilityCheck:
        """Check spectral stability."""
        passed = instability < self.spectral_instability_threshold

        return PlausibilityCheck(
            check_name="spectral_stability",
            passed=passed,
            severity="warning" if not passed else "info",
            value=instability,
            threshold=self.spectral_instability_threshold
        )

    def _check_harmonic_consistency(
        self,
        harmonic_ratio: float
    ) -> PlausibilityCheck:
        """Check harmonic contamination."""
        passed = harmonic_ratio < self.harmonic_ratio_threshold

        return PlausibilityCheck(
            check_name="harmonic_consistency",
            passed=passed,
            severity="warning" if not passed else "info",
            value=harmonic_ratio,
            threshold=self.harmonic_ratio_threshold
        )

    def _check_candidate_agreement(
        self,
        candidate_cv: float,
        candidate_estimates: Optional[List[float]]
    ) -> PlausibilityCheck:
        """Check inter-candidate agreement."""
        if candidate_cv <= 0 and not candidate_estimates:
            return PlausibilityCheck(
                check_name="candidate_agreement",
                passed=True,
                value=0.0,
                threshold=self.candidate_cv_threshold
            )

        # Use CV if provided, otherwise compute
        cv = candidate_cv
        if candidate_estimates and len(candidate_estimates) > 1:
            cv = float(np.std(candidate_estimates) / max(np.mean(candidate_estimates), 1.0))

        passed = cv < self.candidate_cv_threshold

        return PlausibilityCheck(
            check_name="candidate_agreement",
            passed=passed,
            severity="warning" if not passed else "info",
            value=cv,
            threshold=self.candidate_cv_threshold
        )

    def _check_timeout(self, confidence: float) -> PlausibilityCheck:
        """Check for measurement timeout."""
        passed = self._invalid_count < self.timeout_frames

        return PlausibilityCheck(
            check_name="measurement_timeout",
            passed=passed,
            severity="error" if not passed else "info",
            value=self._invalid_count,
            threshold=self.timeout_frames
        )

    def reset(self) -> None:
        """Reset gate state."""
        self._last_bpm = None
        self._recent_bpms.clear()
        self._frame_count = 0
        self._invalid_count = 0


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_plausible_bpm() -> bool:
    """Test that plausible BPM passes."""
    print("  test_plausible_bpm...")

    gate = PhysiologicalPlausibilityGate()

    result = gate.evaluate(bpm=72.0)

    assert result.state == PlausibilityState.PLAUSIBLE
    assert result.passed

    print(f"    state: {result.state.value}")
    print(f"    PASS")
    return True


def test_implausible_bpm() -> bool:
    """Test that implausible BPM fails."""
    print("  test_implausible_bpm...")

    gate = PhysiologicalPlausibilityGate(min_bpm=50.0, max_bpm=180.0)

    # Very low BPM
    result = gate.evaluate(bpm=30.0)
    assert result.state == PlausibilityState.INVALID

    # Very high BPM
    result = gate.evaluate(bpm=220.0)
    assert result.state == PlausibilityState.INVALID

    print(f"    PASS")
    return True


def test_frame_jump() -> bool:
    """Test frame-to-frame jump detection."""
    print("  test_frame_jump...")

    gate = PhysiologicalPlausibilityGate(max_jump_bpm=15.0)

    # Normal jump
    result = gate.evaluate(bpm=72.0, previous_bpm=70.0)
    assert result.state == PlausibilityState.PLAUSIBLE

    # Large jump
    result = gate.evaluate(bpm=95.0, previous_bpm=70.0)
    assert result.state == PlausibilityState.INVALID

    print(f"    PASS")
    return True


def test_spectral_instability() -> bool:
    """Test spectral instability detection."""
    print("  test_spectral_instability...")

    gate = PhysiologicalPlausibilityGate(spectral_instability_threshold=0.5)

    # Stable
    result = gate.evaluate(bpm=72.0, spectral_instability=0.2)
    assert result.state == PlausibilityState.PLAUSIBLE

    # Unstable
    result = gate.evaluate(bpm=72.0, spectral_instability=0.8)
    assert result.state == PlausibilityState.UNCERTAIN

    print(f"    PASS")
    return True


def test_harmonic_contamination() -> bool:
    """Test harmonic contamination detection."""
    print("  test_harmonic_contamination...")

    gate = PhysiologicalPlausibilityGate(harmonic_ratio_threshold=0.4)

    # Clean
    result = gate.evaluate(bpm=72.0, harmonic_ratio=0.2)
    assert result.state == PlausibilityState.PLAUSIBLE

    # Contaminated
    result = gate.evaluate(bpm=72.0, harmonic_ratio=0.6)
    assert result.state == PlausibilityState.UNCERTAIN

    print(f"    PASS")
    return True


def test_candidate_disagreement() -> bool:
    """Test candidate disagreement detection."""
    print("  test_candidate_disagreement...")

    gate = PhysiologicalPlausibilityGate(candidate_cv_threshold=0.2)

    # Agreement
    result = gate.evaluate(bpm=72.0, candidate_cv=0.1)
    assert result.state == PlausibilityState.PLAUSIBLE

    # Disagreement
    result = gate.evaluate(bpm=72.0, candidate_cv=0.4)
    assert result.state == PlausibilityState.UNCERTAIN

    print(f"    PASS")
    return True


def test_multiple_failures() -> bool:
    """Test that multiple failures aggregate correctly."""
    print("  test_multiple_failures...")

    gate = PhysiologicalPlausibilityGate(
        min_bpm=60.0,
        max_bpm=100.0,
        spectral_instability_threshold=0.3
    )

    # BPM in range but spectral issues
    result = gate.evaluate(bpm=72.0, spectral_instability=0.8)

    assert result.n_errors == 0
    assert result.n_warnings == 1

    print(f"    errors: {result.n_errors}, warnings: {result.n_warnings}")
    print(f"    PASS")
    return True


def test_check_by_name() -> bool:
    """Test retrieving checks by name."""
    print("  test_check_by_name...")

    gate = PhysiologicalPlausibilityGate()
    result = gate.evaluate(bpm=72.0)

    range_check = result.get_check("bpm_range")
    assert range_check is not None
    assert range_check.passed

    print(f"    PASS")
    return True


def run_tests() -> bool:
    """Run all P4.5 tests."""
    print("\n" + "=" * 60)
    print("P4.5: Physiological Plausibility Gate Tests")
    print("=" * 60)

    tests = [
        ("plausible_bpm", test_plausible_bpm),
        ("implausible_bpm", test_implausible_bpm),
        ("frame_jump", test_frame_jump),
        ("spectral_instability", test_spectral_instability),
        ("harmonic_contamination", test_harmonic_contamination),
        ("candidate_disagreement", test_candidate_disagreement),
        ("multiple_failures", test_multiple_failures),
        ("check_by_name", test_check_by_name),
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
    print(f"P4.5 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
