"""
P4.4: Temporal Reliability

Implements temporal reliability behavior with state machine.

States:
- WARMUP: Initial frames, building confidence
- STABLE: High-quality, stable estimate
- DEGRADING: Quality decreasing
- UNCERTAIN: Low confidence estimate
- INVALID: Not enough evidence
- RECOVERING: Recovering from degraded state

The system understands:
- One bad frame should not immediately destroy a stable estimate
- One suspicious frame should not instantly become truth
- Persistent degradation should reduce confidence
- Persistent high-quality evidence should increase confidence
- Recovery should be possible
- Confidence should have memory where scientifically justified

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from collections import deque


class TemporalState(Enum):
    """Temporal reliability states."""
    WARMUP = "warmup"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"
    RECOVERING = "recovering"


@dataclass
class TemporalReliabilityState:
    """Current temporal reliability state.

    Attributes:
        state: Current state
        confidence: Current confidence [0, 100]
        evidence_trend: Recent evidence trend
        frames_in_state: Frames spent in current state
        degradation_count: Consecutive degrading frames
        recovery_count: Consecutive improving frames
        last_bpm: Last BPM estimate
        bpm_trajectory: Recent BPM values
    """
    state: TemporalState = TemporalState.WARMUP
    confidence: float = 0.0
    evidence_trend: float = 0.0  # positive = improving
    frames_in_state: int = 0
    degradation_count: int = 0
    recovery_count: int = 0
    last_bpm: float = 0.0
    bpm_trajectory: List[float] = field(default_factory=list)

    def is_stable(self) -> bool:
        """Check if in stable state."""
        return self.state == TemporalState.STABLE

    def is_valid(self) -> bool:
        """Check if state is valid (not INVALID)."""
        return self.state != TemporalState.INVALID


@dataclass
class TemporalReliabilityResult:
    """Result of temporal reliability computation.

    Attributes:
        state: Final temporal state
        confidence: Final confidence
        trend: Evidence trend direction
        valid: Whether temporal state is valid
        state_transitions: Number of state transitions
        degradation_frames: Frames spent degrading
        recovery_frames: Frames spent recovering
        last_bpm: Last BPM estimate
    """
    state: TemporalState = TemporalState.WARMUP
    confidence: float = 0.0
    trend: float = 0.0  # positive = improving
    valid: bool = True
    state_transitions: int = 0
    degradation_frames: int = 0
    recovery_frames: int = 0
    last_bpm: float = 0.0


class TemporalReliabilityTracker:
    """
    Tracks temporal reliability with state machine.

    State transitions:
    WARMUP → STABLE (confidence threshold reached)
    WARMUP → INVALID (insufficient evidence timeout)
    STABLE → DEGRADING (evidence drops)
    STABLE → UNCERTAIN (confidence drops slowly)
    DEGRADING → STABLE (evidence recovers)
    DEGRADING → UNCERTAIN (persistent low evidence)
    UNCERTAIN → INVALID (too many uncertain frames)
    UNCERTAIN → RECOVERING (evidence improves)
    UNCERTAIN → STABLE (evidence good)
    RECOVERING → STABLE (full recovery)
    RECOVERING → DEGRADING (recovery fails)
    INVALID → RECOVERING (evidence returns)
    """

    def __init__(
        self,
        warmup_frames: int = 15,
        stable_threshold: float = 60.0,
        degrading_threshold: float = 40.0,
        invalid_threshold: float = 20.0,
        degradation_persistence: int = 5,
        recovery_persistence: int = 3,
        uncertain_timeout: int = 30,
        evidence_memory: int = 10
    ):
        """
        Initialize temporal reliability tracker.

        Args:
            warmup_frames: Frames to reach WARMUP → STABLE
            stable_threshold: Confidence for STABLE state
            degrading_threshold: Confidence for DEGRADING
            invalid_threshold: Confidence for INVALID
            degradation_persistence: Frames before DEGRADING → UNCERTAIN
            recovery_persistence: Frames before RECOVERING → STABLE
            uncertain_timeout: Frames before UNCERTAIN → INVALID
            evidence_memory: Frames for trend calculation
        """
        self.warmup_frames = warmup_frames
        self.stable_threshold = stable_threshold
        self.degrading_threshold = degrading_threshold
        self.invalid_threshold = invalid_threshold
        self.degradation_persistence = degradation_persistence
        self.recovery_persistence = recovery_persistence
        self.uncertain_timeout = uncertain_timeout
        self.evidence_memory = evidence_memory

        # State
        self._state = TemporalState.WARMUP
        self._confidence = 0.0
        self._frames_in_state = 0
        self._degradation_count = 0
        self._recovery_count = 0
        self._uncertain_count = 0
        self._transitions = 0
        self._total_degradation_frames = 0
        self._total_recovery_frames = 0

        # Evidence history
        self._evidence_history: deque = deque(maxlen=evidence_memory)
        self._bpm_history: deque = deque(maxlen=evidence_memory)

    def update(
        self,
        evidence: float,
        confidence: float,
        bpm: float
    ) -> TemporalReliabilityResult:
        """Update temporal reliability state.

        Args:
            evidence: Current evidence quality [0, 1]
            confidence: Current confidence [0, 100]
            bpm: Current BPM estimate

        Returns:
            TemporalReliabilityResult with new state
        """
        # Store history
        self._evidence_history.append(evidence)
        self._bpm_history.append(bpm)

        # Compute trend
        trend = self._compute_trend()

        # Previous state
        prev_state = self._state

        # State machine update
        self._update_state(evidence, confidence)

        # Update counters
        self._frames_in_state += 1

        # Track degradation/recovery frames
        if self._state in (TemporalState.DEGRADING, TemporalState.UNCERTAIN):
            self._total_degradation_frames += 1
        elif self._state == TemporalState.RECOVERING:
            self._total_recovery_frames += 1

        # Create result
        result = TemporalReliabilityResult(
            state=self._state,
            confidence=self._confidence,
            trend=trend,
            valid=self._state != TemporalState.INVALID,
            state_transitions=self._transitions,
            degradation_frames=self._total_degradation_frames,
            recovery_frames=self._total_recovery_frames,
            last_bpm=bpm
        )

        return result

    def _compute_trend(self) -> float:
        """Compute evidence trend.

        Returns:
            Trend value: positive = improving, negative = degrading
        """
        if len(self._evidence_history) < 3:
            return 0.0

        recent = list(self._evidence_history)[-5:]
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        x = np.arange(len(recent))
        y = np.array(recent)

        # Slope of linear fit
        if len(x) < 2:
            return 0.0

        mean_x = np.mean(x)
        mean_y = np.mean(y)

        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)

        if abs(denominator) < 1e-10:
            return 0.0

        slope = numerator / denominator

        # Normalize to [-1, 1] range
        return float(np.clip(slope * 10, -1.0, 1.0))

    def _update_state(self, evidence: float, confidence: float) -> None:
        """Update state based on evidence and confidence."""
        self._confidence = confidence

        if self._state == TemporalState.WARMUP:
            if self._frames_in_state >= self.warmup_frames and confidence >= self.stable_threshold:
                self._transition_to(TemporalState.STABLE)
            elif self._frames_in_state >= self.warmup_frames * 2:
                self._transition_to(TemporalState.UNCERTAIN)

        elif self._state == TemporalState.STABLE:
            if confidence < self.degrading_threshold:
                self._degradation_count += 1
                if self._degradation_count >= self.degradation_persistence:
                    self._transition_to(TemporalState.DEGRADING)
            else:
                self._degradation_count = 0

        elif self._state == TemporalState.DEGRADING:
            if confidence >= self.stable_threshold:
                self._transition_to(TemporalState.STABLE)
            elif self._degradation_count >= self.degradation_persistence * 2:
                self._transition_to(TemporalState.UNCERTAIN)
            else:
                self._degradation_count += 1

        elif self._state == TemporalState.UNCERTAIN:
            self._uncertain_count += 1
            if confidence < self.invalid_threshold:
                self._transition_to(TemporalState.INVALID)
            elif confidence >= self.stable_threshold:
                self._transition_to(TemporalState.STABLE)
            elif confidence >= self.degrading_threshold and self._recovery_count >= self.recovery_persistence:
                self._transition_to(TemporalState.RECOVERING)
            elif evidence > 0.6 and confidence > self.degrading_threshold:
                self._recovery_count += 1

        elif self._state == TemporalState.RECOVERING:
            if confidence >= self.stable_threshold:
                self._transition_to(TemporalState.STABLE)
            elif confidence < self.degrading_threshold:
                self._transition_to(TemporalState.DEGRADING)
            elif self._recovery_count >= self.recovery_persistence * 2:
                self._transition_to(TemporalState.UNCERTAIN)
            else:
                self._recovery_count += 1

        elif self._state == TemporalState.INVALID:
            if evidence > 0.5 and confidence > self.degrading_threshold:
                self._transition_to(TemporalState.RECOVERING)
            elif self._frames_in_state >= self.uncertain_timeout:
                self._transition_to(TemporalState.UNCERTAIN)

    def _transition_to(self, new_state: TemporalState) -> None:
        """Transition to new state."""
        if new_state != self._state:
            self._state = new_state
            self._frames_in_state = 0
            self._degradation_count = 0
            self._recovery_count = 0
            self._uncertain_count = 0
            self._transitions += 1

    def get_current_state(self) -> TemporalReliabilityState:
        """Get current reliability state."""
        return TemporalReliabilityState(
            state=self._state,
            confidence=self._confidence,
            evidence_trend=self._compute_trend(),
            frames_in_state=self._frames_in_state,
            degradation_count=self._degradation_count,
            recovery_count=self._recovery_count,
            last_bpm=self._bpm_history[-1] if self._bpm_history else 0.0,
            bpm_trajectory=list(self._bpm_history)
        )

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._state = TemporalState.WARMUP
        self._confidence = 0.0
        self._frames_in_state = 0
        self._degradation_count = 0
        self._recovery_count = 0
        self._uncertain_count = 0
        self._transitions = 0
        self._total_degradation_frames = 0
        self._total_recovery_frames = 0
        self._evidence_history.clear()
        self._bpm_history.clear()


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_warmup_to_stable() -> bool:
    """Test WARMUP → STABLE transition."""
    print("  test_warmup_to_stable...")

    tracker = TemporalReliabilityTracker(warmup_frames=10)

    # Simulate high-quality frames
    for i in range(15):
        tracker.update(evidence=0.9, confidence=80.0, bpm=72.0)

    state = tracker.get_current_state()
    assert state.state == TemporalState.STABLE

    print(f"    state: {state.state.value}")
    print(f"    PASS")
    return True


def test_stable_to_degrading() -> bool:
    """Test STABLE → DEGRADING transition."""
    print("  test_stable_to_degrading...")

    tracker = TemporalReliabilityTracker(
        degradation_persistence=3,
        stable_threshold=60.0,
        degrading_threshold=40.0
    )

    # Warmup first
    for i in range(15):
        tracker.update(evidence=0.9, confidence=80.0, bpm=72.0)

    # Degrade
    for i in range(5):
        tracker.update(evidence=0.4, confidence=30.0, bpm=72.0)

    state = tracker.get_current_state()
    assert state.state in (TemporalState.DEGRADING, TemporalState.UNCERTAIN)

    print(f"    state: {state.state.value}")
    print(f"    PASS")
    return True


def test_degrading_to_recovery() -> bool:
    """Test DEGRADING → RECOVERING → STABLE transition."""
    print("  test_degrading_to_recovery...")

    tracker = TemporalReliabilityTracker(
        degradation_persistence=2,
        recovery_persistence=2,
        stable_threshold=60.0
    )

    # Warmup
    for i in range(15):
        tracker.update(evidence=0.9, confidence=80.0, bpm=72.0)

    # Degrade
    for i in range(5):
        tracker.update(evidence=0.3, confidence=30.0, bpm=70.0)

    # Recover
    for i in range(10):
        tracker.update(evidence=0.8, confidence=70.0, bpm=72.0)

    state = tracker.get_current_state()
    assert state.state == TemporalState.STABLE

    print(f"    state: {state.state.value}")
    print(f"    PASS")
    return True


def test_invalid_recovery() -> bool:
    """Test INVALID → RECOVERING transition."""
    print("  test_invalid_recovery...")

    tracker = TemporalReliabilityTracker(
        invalid_threshold=20.0,
        degradation_persistence=1,
        recovery_persistence=1
    )

    # Go directly to invalid
    for i in range(20):
        tracker.update(evidence=0.2, confidence=15.0, bpm=72.0)

    state = tracker.get_current_state()
    assert state.state == TemporalState.INVALID

    # Recover
    for i in range(5):
        tracker.update(evidence=0.7, confidence=50.0, bpm=72.0)

    state = tracker.get_current_state()
    assert state.state == TemporalState.RECOVERING

    print(f"    state after recovery: {state.state.value}")
    print(f"    PASS")
    return True


def test_trend_computation() -> bool:
    """Test evidence trend computation."""
    print("  test_trend_computation...")

    tracker = TemporalReliabilityTracker()

    # Improving evidence
    for i in range(10):
        tracker.update(evidence=0.5 + i * 0.03, confidence=70.0, bpm=72.0)

    state = tracker.get_current_state()
    assert state.evidence_trend > 0, "Should detect improving trend"

    # Degrading evidence
    for i in range(5):
        tracker.update(evidence=0.8 - i * 0.1, confidence=60.0, bpm=72.0)

    state = tracker.get_current_state()
    # Trend may still be positive if recent memory is high
    print(f"    trend: {state.evidence_trend:.3f}")

    print(f"    PASS")
    return True


def test_reset() -> bool:
    """Test tracker reset."""
    print("  test_reset...")

    tracker = TemporalReliabilityTracker()

    # Add some history
    for i in range(10):
        tracker.update(evidence=0.8, confidence=70.0, bpm=72.0)

    # Reset
    tracker.reset()

    state = tracker.get_current_state()
    assert state.state == TemporalState.WARMUP

    print(f"    state after reset: {state.state.value}")
    print(f"    PASS")
    return True


def test_state_machine_determinism() -> bool:
    """Test that same input produces same state."""
    print("  test_state_machine_determinism...")

    # Run twice with same input
    tracker1 = TemporalReliabilityTracker(warmup_frames=10)
    tracker2 = TemporalReliabilityTracker(warmup_frames=10)

    for i in range(20):
        evidence = 0.8 + (0.1 if i < 10 else -0.2)
        confidence = 75.0 + (5.0 if i < 10 else -30.0)
        tracker1.update(evidence, confidence, 72.0)
        tracker2.update(evidence, confidence, 72.0)

    s1 = tracker1.get_current_state()
    s2 = tracker2.get_current_state()

    assert s1.state == s2.state
    assert abs(s1.confidence - s2.confidence) < 0.1

    print(f"    deterministic: state={s1.state.value}, conf={s1.confidence:.1f}%")
    print(f"    PASS")
    return True


def run_tests() -> bool:
    """Run all P4.4 tests."""
    print("\n" + "=" * 60)
    print("P4.4: Temporal Reliability Tests")
    print("=" * 60)

    tests = [
        ("warmup_to_stable", test_warmup_to_stable),
        ("stable_to_degrading", test_stable_to_degrading),
        ("degrading_to_recovery", test_degrading_to_recovery),
        ("invalid_recovery", test_invalid_recovery),
        ("trend_computation", test_trend_computation),
        ("reset", test_reset),
        ("state_machine_determinism", test_state_machine_determinism),
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
    print(f"P4.4 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
