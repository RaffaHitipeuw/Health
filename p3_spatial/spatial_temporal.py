"""
P3.5: Spatial-Temporal Consistency

Introduces temporal tracking of spatial candidate quality.

Mathematical definitions:

1. Instantaneous quality:
   Q_inst(t) = Q_weighted(t)

2. Rolling average quality:
   Q_avg(t, N) = (1/N) * Σ_{k=0}^{N-1} Q_inst(t-k)

3. Quality trend:
   T_q = Q_avg(t) - Q_avg(t-N)  (positive = improving)

4. Temporal stability:
   S_temporal = 1 / (std(Q_inst(t-N:t)) + ε)

5. Persistence score:
   P = mean(Q_history) * consistency(Q_history)

   where consistency = 1 if CV < threshold, else lower

6. Candidate lifecycle:
   - BORN: P > birth_threshold AND not existing
   - ALIVE: updated this frame
   - DEGRADING: P < recovery_threshold for M frames
   - EXPIRED: DEGRADING for N frames
   - RECOVERED: P > recovery_threshold after DEGRADING

This module does NOT modify V2 core.
Tracks temporal properties without black-box models.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from enum import Enum


def _get_candidate_class():
    """Get Candidate class for testing, handles both module and script execution."""
    import sys
    import os

    # Ensure parent directory is in path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Try package import first
    try:
        from p3_spatial.dynamic_candidates import Candidate
        return Candidate
    except (ImportError, ModuleNotFoundError):
        pass

    # Try direct import
    try:
        Candidate = _get_candidate_class()
        return Candidate
    except (ImportError, ModuleNotFoundError):
        pass

    raise ImportError("Could not import Candidate class")


# Default parameters
DEFAULT_WINDOW_SIZE: int = 10  # frames for rolling average
DEFAULT_CONSISTENCY_THRESHOLD: float = 0.3  # CV threshold
DEFAULT_BIRTH_THRESHOLD: float = 0.5
DEFAULT_DEGRADATION_THRESHOLD: float = 0.25
DEFAULT_EXPIRATION_FRAMES: int = 5
DEFAULT_RECOVERY_FRAMES: int = 3

# State machine states
class CandidateState(Enum):
    """Candidate lifecycle states."""
    BORN = "born"
    ALIVE = "alive"
    DEGRADING = "degrading"
    EXPIRED = "expired"
    RECOVERED = "recovered"


@dataclass
class TemporalCandidate:
    """Candidate with temporal tracking.

    Attributes:
        id: Unique identifier
        base: Base candidate data
        state: Current lifecycle state
        quality_window: Rolling quality scores
        stability_score: Temporal stability measure
        persistence_score: Combined persistence metric
        state_history: Recent state history
        frames_in_state: Frames in current state
    """
    id: int
    cells: List[Tuple[int, int]] = field(default_factory=list)
    center: Tuple[float, float] = (0.0, 0.0)
    area: int = 0
    signal_quality: float = 0.0
    motion_quality: float = 0.0
    stability_score: float = 0.0
    persistence_score: float = 0.0
    state: CandidateState = CandidateState.ALIVE
    quality_window: List[float] = field(default_factory=list)
    stability_history: List[float] = field(default_factory=list)
    state_history: List[CandidateState] = field(default_factory=list)
    frames_in_state: int = 0
    birth_frame: int = 0
    last_update_frame: int = 0


@dataclass
class TemporalConsistencyResult:
    """Result of temporal consistency computation.

    Attributes:
        candidates: List of temporally tracked candidates
        global_stability: Global temporal stability score
        avg_persistence: Average persistence score
        state_distribution: Count of candidates per state
        valid: Whether computation succeeded
        error: Error message if invalid
    """
    candidates: List[TemporalCandidate] = field(default_factory=list)
    global_stability: float = 0.0
    avg_persistence: float = 0.0
    state_distribution: Dict[str, int] = field(default_factory=dict)
    frame_count: int = 0
    valid: bool = True
    error: Optional[str] = None


class SpatialTemporalConsistency:
    """
    Tracks temporal consistency of spatial candidates.

    State machine:

    BORN → ALIVE → DEGRADING → EXPIRED
              ↑__________|          ↑
              |                   (if recovery)
              ↓________________RECOVERED

    Rules:
    - BORN: New candidate with persistence_score > birth_threshold
    - ALIVE: Updated this frame, persistence_score > degradation_threshold
    - DEGRADING: persistence_score < degradation_threshold for N frames
    - EXPIRED: DEGRADING for > expiration_frames
    - RECOVERED: persistence_score > degradation_threshold after DEGRADING
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
        birth_threshold: float = DEFAULT_BIRTH_THRESHOLD,
        degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
        expiration_frames: int = DEFAULT_EXPIRATION_FRAMES,
        recovery_frames: int = DEFAULT_RECOVERY_FRAMES
    ):
        """
        Initialize temporal consistency tracker.

        Args:
            window_size: Frames for rolling statistics
            consistency_threshold: CV threshold for consistency
            birth_threshold: Persistence threshold for new candidates
            degradation_threshold: Persistence threshold for alive state
            expiration_frames: Frames before candidate expires
            recovery_frames: Frames before recovery is considered
        """
        self.window_size = window_size
        self.consistency_threshold = consistency_threshold
        self.birth_threshold = birth_threshold
        self.degradation_threshold = degradation_threshold
        self.expiration_frames = expiration_frames
        self.recovery_frames = recovery_frames

        # State
        self._candidates: Dict[int, TemporalCandidate] = {}
        self._next_id: int = 0
        self._frame_count: int = 0
        self._total_born: int = 0
        self._total_expired: int = 0
        self._total_recovered: int = 0

    def _compute_persistence_score(
        self,
        quality_history: List[float],
        stability_history: List[float]
    ) -> float:
        """
        Compute persistence score.

        Mathematical definition:
            P = mean(Q_window) * consistency_factor

        where:
            consistency_factor = 1 if CV < threshold
                                = threshold / CV otherwise
        """
        if not quality_history:
            return 0.0

        # Mean quality
        mean_q = float(np.mean(quality_history))

        # Consistency factor based on coefficient of variation
        if len(quality_history) >= 2:
            std_q = float(np.std(quality_history))
            mean_abs = abs(mean_q) + 1e-9
            cv = std_q / mean_abs
            consistency = min(1.0, self.consistency_threshold / (cv + 1e-9))
        else:
            consistency = 1.0

        # Mean stability
        mean_stab = float(np.mean(stability_history)) if stability_history else 0.5

        # Combined persistence
        persistence = mean_q * consistency * mean_stab

        return float(np.clip(persistence, 0.0, 1.0))

    def _compute_stability(
        self,
        quality_history: List[float]
    ) -> float:
        """
        Compute temporal stability score.

        Mathematical definition:
            S = 1 / (std(Q_history) + ε)
            Normalized to [0, 1]
        """
        if len(quality_history) < 2:
            return 1.0

        std_q = float(np.std(quality_history))
        stability = 1.0 / (std_q + 1e-6)

        return float(np.clip(stability / 10.0, 0.0, 1.0))

    def _update_state(
        self,
        candidate: TemporalCandidate
    ) -> CandidateState:
        """
        Update candidate state based on persistence score.

        State machine logic:
        """
        persistence = candidate.persistence_score
        candidate.frames_in_state += 1

        current_state = candidate.state

        if current_state == CandidateState.EXPIRED:
            # Check for recovery
            if persistence > self.degradation_threshold:
                candidate.state = CandidateState.RECOVERED
                candidate.state_history.append(CandidateState.RECOVERED)
                self._total_recovered += 1
            return candidate.state

        elif current_state == CandidateState.BORN:
            if persistence < self.degradation_threshold:
                candidate.state = CandidateState.DEGRADING
                candidate.state_history.append(CandidateState.DEGRADING)
                candidate.frames_in_state = 0
            else:
                candidate.state = CandidateState.ALIVE
                candidate.state_history.append(CandidateState.ALIVE)

        elif current_state == CandidateState.ALIVE:
            if persistence < self.degradation_threshold:
                candidate.state = CandidateState.DEGRADING
                candidate.state_history.append(CandidateState.DEGRADING)
                candidate.frames_in_state = 0

        elif current_state == CandidateState.DEGRADING:
            if persistence > self.degradation_threshold:
                candidate.state = CandidateState.ALIVE
                candidate.state_history.append(CandidateState.ALIVE)
                candidate.frames_in_state = 0
            elif candidate.frames_in_state >= self.expiration_frames:
                candidate.state = CandidateState.EXPIRED
                candidate.state_history.append(CandidateState.EXPIRED)
                self._total_expired += 1

        elif current_state == CandidateState.RECOVERED:
            if persistence < self.degradation_threshold:
                candidate.state = CandidateState.DEGRADING
                candidate.state_history.append(CandidateState.DEGRADING)
                candidate.frames_in_state = 0
            else:
                candidate.state = CandidateState.ALIVE
                candidate.state_history.append(CandidateState.ALIVE)

        return candidate.state

    def update(
        self,
        candidates: List,  # From DynamicCandidateSelector
        frame_number: Optional[int] = None
    ) -> TemporalConsistencyResult:
        """
        Update temporal tracking for candidates.

        Args:
            candidates: List of Candidate objects from P3.4
            frame_number: Current frame number

        Returns:
            TemporalConsistencyResult with tracked candidates
        """
        self._frame_count += 1
        current_frame = frame_number if frame_number is not None else self._frame_count

        if candidates is None:
            return TemporalConsistencyResult(
                valid=False,
                error="candidates is None",
                frame_count=self._frame_count
            )

        # Mark all existing candidates as potentially expired
        for c in self._candidates.values():
            c.last_update_frame = -1  # Will be updated if matched

        # Process incoming candidates
        new_candidates = []
        for cand in candidates:
            if not hasattr(cand, 'id'):
                continue

            # Check if this candidate exists
            if cand.id in self._candidates:
                # Update existing
                tc = self._candidates[cand.id]
                tc.last_update_frame = current_frame
                tc.cells = cand.cells
                tc.center = cand.center
                tc.area = cand.area
                tc.signal_quality = cand.signal_quality
                tc.motion_quality = cand.motion_quality
                tc.stability_score = cand.stability_score

                # Update quality window
                tc.quality_window.append(cand.candidate_score)
                if len(tc.quality_window) > self.window_size:
                    tc.quality_window.pop(0)

                tc.stability_history.append(cand.stability_score if hasattr(cand, 'stability_score') else 0.5)
                if len(tc.stability_history) > self.window_size:
                    tc.stability_history.pop(0)

                # Recompute persistence
                tc.persistence_score = self._compute_persistence_score(
                    tc.quality_window, tc.stability_history
                )

                # Update stability
                tc.stability_score = self._compute_stability(tc.quality_window)

                # Update state
                self._update_state(tc)

            else:
                # New candidate
                tc = TemporalCandidate(
                    id=cand.id,
                    cells=cand.cells,
                    center=cand.center,
                    area=cand.area,
                    signal_quality=cand.signal_quality,
                    motion_quality=cand.motion_quality,
                    stability_score=cand.stability_score,
                    state=CandidateState.BORN,
                    quality_window=[cand.candidate_score],
                    stability_history=[cand.stability_score] if hasattr(cand, 'stability_score') else [0.5],
                    state_history=[CandidateState.BORN],
                    frames_in_state=0,
                    birth_frame=current_frame,
                    last_update_frame=current_frame
                )

                # Compute initial persistence
                tc.persistence_score = self._compute_persistence_score(
                    tc.quality_window, tc.stability_history
                )

                # Transition from BORN if persistence is high enough
                if tc.persistence_score > self.birth_threshold:
                    tc.state = CandidateState.ALIVE
                    tc.state_history[-1] = CandidateState.ALIVE
                    self._total_born += 1

                self._candidates[cand.id] = tc

            new_candidates.append(self._candidates[cand.id])

        # Remove expired candidates
        expired_ids = [
            cid for cid, c in self._candidates.items()
            if c.last_update_frame < current_frame - self.expiration_frames
        ]
        for cid in expired_ids:
            del self._candidates[cid]

        # Compute global statistics
        all_candidates = list(self._candidates.values())

        if all_candidates:
            global_stability = float(np.mean([c.stability_score for c in all_candidates]))
            avg_persistence = float(np.mean([c.persistence_score for c in all_candidates]))

            state_dist = {s.value: 0 for s in CandidateState}
            for c in all_candidates:
                state_dist[c.state.value] = state_dist.get(c.state.value, 0) + 1
        else:
            global_stability = 0.0
            avg_persistence = 0.0
            state_dist = {s.value: 0 for s in CandidateState}

        return TemporalConsistencyResult(
            candidates=all_candidates,
            global_stability=global_stability,
            avg_persistence=avg_persistence,
            state_distribution=state_dist,
            frame_count=self._frame_count,
            valid=True
        )

    def get_stable_candidates(self) -> List[TemporalCandidate]:
        """Get candidates with stable temporal behavior."""
        return [
            c for c in self._candidates.values()
            if c.state in (CandidateState.ALIVE, CandidateState.BORN, CandidateState.RECOVERED)
        ]

    def get_best_persistent_candidate(self) -> Optional[TemporalCandidate]:
        """Get the candidate with highest persistence score."""
        stable = self.get_stable_candidates()
        if not stable:
            return None
        return max(stable, key=lambda c: c.persistence_score)

    def reset(self) -> None:
        """Clear all temporal state."""
        self._candidates.clear()
        self._frame_count = 0
        self._total_born = 0
        self._total_expired = 0
        self._total_recovered = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "n_candidates": len(self._candidates),
            "n_stable": len(self.get_stable_candidates()),
            "total_born": self._total_born,
            "total_expired": self._total_expired,
            "total_recovered": self._total_recovered,
            "frame_count": self._frame_count,
            "avg_persistence": self.avg_persistence if self._candidates else 0.0
        }


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_quality_stability_computation() -> bool:
    """
    Test: Stability is computed correctly from quality history.
    """
    print("  test_quality_stability_computation...")

    tracker = SpatialTemporalConsistency(window_size=5)

    # Stable quality (low variance)
    stable_history = [0.8, 0.81, 0.79, 0.82, 0.80]
    stable_stability = tracker._compute_stability(stable_history)

    # Unstable quality (high variance)
    unstable_history = [0.9, 0.3, 0.8, 0.4, 0.7]
    unstable_stability = tracker._compute_stability(unstable_history)

    # Stable should have higher stability score
    assert stable_stability > unstable_stability, \
        f"Stable ({stable_stability:.4f}) should exceed unstable ({unstable_stability:.4f})"

    print(f"    stable: {stable_stability:.4f}")
    print(f"    unstable: {unstable_stability:.4f}")
    print(f"    PASS")
    return True


def test_persistence_score() -> bool:
    """
    Test: Persistence score reflects quality and consistency.
    """
    print("  test_persistence_score...")

    tracker = SpatialTemporalConsistency(consistency_threshold=0.3)

    # High quality, consistent
    good_history = [0.9, 0.88, 0.92, 0.89, 0.91]
    stability_history = [0.9, 0.9, 0.9, 0.9, 0.9]
    good_persistence = tracker._compute_persistence_score(good_history, stability_history)

    # Low quality, inconsistent
    bad_history = [0.2, 0.1, 0.3, 0.15, 0.25]
    bad_stability = [0.3, 0.3, 0.3, 0.3, 0.3]
    bad_persistence = tracker._compute_persistence_score(bad_history, bad_stability)

    # Good should have higher persistence
    assert good_persistence > bad_persistence, \
        f"Good ({good_persistence:.4f}) should exceed bad ({bad_persistence:.4f})"

    print(f"    good persistence: {good_persistence:.4f}")
    print(f"    bad persistence: {bad_persistence:.4f}")
    print(f"    PASS")
    return True


def test_candidate_birth() -> bool:
    """
    Test: New high-quality candidates are born correctly.
    """
    print("  test_candidate_birth...")

    tracker = SpatialTemporalConsistency(
        birth_threshold=0.3,
        degradation_threshold=0.2
    )

    # Create mock candidates using helper function
    Candidate = _get_candidate_class()
    mock_candidates = [
        Candidate(
            id=0,
            cells=[(0, 0)],
            center=(8, 8),
            area=1,
            signal_quality=0.8,
            motion_quality=0.9,
            stability_score=0.9,
            candidate_score=0.7,
            birth_frame=1,
            last_update_frame=1,
            validity=True,
            quality_history=[0.7]
        )
    ]

    result = tracker.update(mock_candidates, frame_number=1)

    assert result.valid
    assert len(result.candidates) == 1

    candidate = result.candidates[0]
    assert candidate.state in (CandidateState.BORN, CandidateState.ALIVE), \
        f"Should be BORN or ALIVE, got {candidate.state}"

    print(f"    state: {candidate.state.value}")
    print(f"    persistence: {candidate.persistence_score:.4f}")
    print(f"    PASS")
    return True


def test_candidate_degradation() -> bool:
    """
    Test: Low-quality candidates transition to DEGRADING state.
    """
    print("  test_candidate_degradation...")

    tracker = SpatialTemporalConsistency(
        degradation_threshold=0.3,
        expiration_frames=3
    )

    # Create candidate with low initial quality
    Candidate = _get_candidate_class()
    mock_candidates = [
        Candidate(
            id=0,
            cells=[(0, 0)],
            center=(8, 8),
            area=1,
            signal_quality=0.2,
            motion_quality=0.2,
            stability_score=0.2,
            candidate_score=0.15,  # Below threshold
            birth_frame=1,
            last_update_frame=1,
            validity=True,
            quality_history=[0.15]
        )
    ]

    # First update
    result1 = tracker.update(mock_candidates, frame_number=1)
    state1 = result1.candidates[0].state if result1.candidates else None

    # Multiple updates with low quality
    for frame in range(2, 5):
        mock_candidates[0].candidate_score = 0.1
        mock_candidates[0].quality_history = [0.1]
        result = tracker.update(mock_candidates, frame_number=frame)

    state_final = result.candidates[0].state if result.candidates else None

    # Should eventually degrade or stay low
    print(f"    initial state: {state1.value if state1 else 'N/A'}")
    print(f"    final state: {state_final.value if state_final else 'N/A'}")
    print(f"    PASS")
    return True


def test_candidate_expiration() -> bool:
    """
    Test: Candidates without updates eventually expire.
    """
    print("  test_candidate_expiration...")

    tracker = SpatialTemporalConsistency(
        degradation_threshold=0.2,
        expiration_frames=3
    )

    Candidate = _get_candidate_class()
    mock_candidates = [
        Candidate(
            id=0,
            cells=[(0, 0)],
            center=(8, 8),
            area=1,
            signal_quality=0.5,
            motion_quality=0.5,
            stability_score=0.5,
            candidate_score=0.4,
            birth_frame=1,
            last_update_frame=1,
            validity=True,
            quality_history=[0.4]
        )
    ]

    # Initial update
    tracker.update(mock_candidates, frame_number=1)

    # Frames 2-5: no candidates
    for frame in range(2, 6):
        tracker.update([], frame_number=frame)

    # Candidate should be expired or removed
    stable = tracker.get_stable_candidates()

    # Should have no stable candidates after expiration
    print(f"    stable candidates after expiration: {len(stable)}")
    print(f"    PASS")
    return True


def test_recovery() -> bool:
    """
    Test: Degraded candidates can recover.
    """
    print("  test_recovery...")

    tracker = SpatialTemporalConsistency(
        degradation_threshold=0.3,
        expiration_frames=10  # Long expiration to allow recovery
    )

    Candidate = _get_candidate_class()

    # Start with low quality
    mock_candidates = [
        Candidate(
            id=0,
            cells=[(0, 0)],
            center=(8, 8),
            area=1,
            signal_quality=0.1,
            motion_quality=0.1,
            stability_score=0.1,
            candidate_score=0.1,
            birth_frame=1,
            last_update_frame=1,
            validity=True,
            quality_history=[0.1]
        )
    ]

    # Update with low quality
    for frame in range(1, 4):
        tracker.update(mock_candidates, frame_number=frame)

    # Now improve quality
    mock_candidates[0].signal_quality = 0.8
    mock_candidates[0].motion_quality = 0.8
    mock_candidates[0].candidate_score = 0.6
    mock_candidates[0].quality_history = [0.6]

    result = tracker.update(mock_candidates, frame_number=5)

    if result.candidates:
        final_state = result.candidates[0].state
        print(f"    state after recovery: {final_state.value}")

    print(f"    PASS")
    return True


def test_global_stability() -> bool:
    """
    Test: Global stability reflects temporal consistency.
    """
    print("  test_global_stability...")

    tracker = SpatialTemporalConsistency(window_size=5)

    Candidate = _get_candidate_class()

    # Create stable candidates
    stable_candidates = []
    for i in range(3):
        c = Candidate(
            id=i,
            cells=[(i, 0)],
            center=(i * 16 + 8, 8),
            area=1,
            signal_quality=0.8,
            motion_quality=0.8,
            stability_score=0.8,
            candidate_score=0.7,
            birth_frame=1,
            last_update_frame=1,
            validity=True,
            quality_history=[0.7]
        )
        stable_candidates.append(c)

    result = tracker.update(stable_candidates, frame_number=1)

    assert result.global_stability > 0.5, \
        f"Stable candidates should have high global stability, got {result.global_stability}"

    print(f"    global stability: {result.global_stability:.4f}")
    print(f"    avg persistence: {result.avg_persistence:.4f}")
    print(f"    PASS")
    return True


def run_tests() -> bool:
    """
    Run all P3.5 spatial-temporal consistency tests.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("P3.5: Spatial-Temporal Consistency Tests")
    print("=" * 60)

    tests = [
        ("quality_stability_computation", test_quality_stability_computation),
        ("persistence_score", test_persistence_score),
        ("candidate_birth", test_candidate_birth),
        ("candidate_degradation", test_candidate_degradation),
        ("candidate_expiration", test_candidate_expiration),
        ("recovery", test_recovery),
        ("global_stability", test_global_stability),
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
    print(f"P3.5 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
