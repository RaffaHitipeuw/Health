"""
P4.6: Confidence-Aware Estimation Pipeline

Integrates reliability and uncertainty into the estimation pipeline.

Pipeline:
P3 candidates
    ↓
candidate reliability
    ↓
signal estimates
    ↓
uncertainty estimation
    ↓
confidence-aware fusion
    ↓
final estimate
    ↓
reliability report

Modes:
- BASELINE: V2 behavior unchanged
- RELIABILITY_ONLY: Add reliability evidence only
- UNCERTAINTY_AWARE: Add uncertainty decomposition only
- FULL_P4: All P4 components

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

# Import P4 components
try:
    from p4_reliability.reliability_evidence import (
        ReliabilityEvidenceCollector, ReliabilityEvidence,
        EvidenceSource
    )
    from p4_reliability.uncertainty_decomposition import (
        UncertaintyDecomposer, UncertaintyDecomposition,
        UncertaintySource
    )
    from p4_reliability.confidence_fusion import (
        ConfidenceUncertaintyFusion, FusedEstimate, FusionMethod
    )
    from p4_reliability.temporal_reliability import (
        TemporalReliabilityTracker, TemporalState, TemporalReliabilityResult
    )
    from p4_reliability.plausibility_gate import (
        PhysiologicalPlausibilityGate, PlausibilityState, PlausibilityResult
    )
except ImportError:
    # Handle module-level import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class P4Mode(Enum):
    """P4 operating modes."""
    BASELINE = "baseline"           # No P4 modifications
    RELIABILITY_ONLY = "reliability_only"  # Reliability evidence only
    UNCERTAINTY_AWARE = "uncertainty_aware"  # Uncertainty only
    FULL_P4 = "full_p4"           # All P4 components


@dataclass
class P4EstimateResult:
    """Complete P4 estimation result.

    Attributes:
        bpm: Final heart rate estimate
        uncertainty: Uncertainty decomposition
        reliability: Reliability evidence
        temporal_state: Temporal reliability state
        plausibility: Physiological plausibility
        fused_estimate: Fused estimate from candidates
        valid: Whether estimate is valid
        confidence: Final confidence percentage
        mode: Operating mode used
    """
    bpm: float = 0.0
    uncertainty: Optional[Any] = None  # UncertaintyDecomposition
    reliability: Optional[Any] = None  # ReliabilityEvidence
    temporal_state: Optional[Any] = None  # TemporalReliabilityResult
    plausibility: Optional[Any] = None  # PlausibilityResult
    fused_estimate: Optional[Any] = None  # FusedEstimate
    valid: bool = True
    confidence: float = 0.0
    mode: str = "baseline"


class P4Estimator:
    """
    Complete P4 confidence-aware estimation pipeline.

    Integrates:
    - P4.1: Reliability Evidence Model
    - P4.2: Uncertainty Decomposition
    - P4.3: Confidence/Uncertainty Fusion
    - P4.4: Temporal Reliability
    - P4.5: Physiological Plausibility Gate
    """

    def __init__(
        self,
        mode: P4Mode = P4Mode.FULL_P4,
        # Fusion settings
        fusion_method: FusionMethod = FusionMethod.VARIANCE_WEIGHTED,
        # Reliability settings
        reliability_window: int = 30,
        # Uncertainty settings
        min_coverage: float = 0.3,
        max_std: float = 15.0,
        # Temporal settings
        warmup_frames: int = 15,
        stable_threshold: float = 60.0,
        # Plausibility settings
        min_bpm: float = 40.0,
        max_bpm: float = 200.0,
        # Baseline use P3
        use_p3: bool = True
    ):
        """
        Initialize P4 estimator.

        Args:
            mode: P4 operating mode
            fusion_method: Fusion method for combining estimates
            reliability_window: Frames for reliability history
            min_coverage: Minimum evidence coverage for VALID
            max_std: Maximum BPM std for VALID
            warmup_frames: Frames for temporal warmup
            stable_threshold: Confidence for STABLE state
            min_bpm: Minimum physiologically plausible BPM
            max_bpm: Maximum physiologically plausible BPM
            use_p3: Whether to use P3 spatial components
        """
        self.mode = mode
        self.use_p3 = use_p3

        # Initialize components based on mode
        self._init_components(
            fusion_method=fusion_method,
            reliability_window=reliability_window,
            min_coverage=min_coverage,
            max_std=max_std,
            warmup_frames=warmup_frames,
            stable_threshold=stable_threshold,
            min_bpm=min_bpm,
            max_bpm=max_bpm
        )

    def _init_components(
        self,
        fusion_method,
        reliability_window: int,
        min_coverage: float,
        max_std: float,
        warmup_frames: int,
        stable_threshold: float,
        min_bpm: float,
        max_bpm: float
    ):
        """Initialize components based on mode."""
        # P4.1: Reliability Evidence
        self.reliability_collector = None
        if self.mode in (P4Mode.RELIABILITY_ONLY, P4Mode.FULL_P4):
            self.reliability_collector = ReliabilityEvidenceCollector(
                window_size=reliability_window
            )

        # P4.2: Uncertainty Decomposition
        self.uncertainty_decomposer = None
        if self.mode in (P4Mode.UNCERTAINTY_AWARE, P4Mode.FULL_P4):
            self.uncertainty_decomposer = UncertaintyDecomposer()

        # P4.3: Confidence Fusion
        self.fusion = None
        if self.mode in (P4Mode.UNCERTAINTY_AWARE, P4Mode.FULL_P4):
            self.fusion = ConfidenceUncertaintyFusion(method=fusion_method)

        # P4.4: Temporal Reliability
        self.temporal_tracker = None
        if self.mode in (P4Mode.FULL_P4,):
            self.temporal_tracker = TemporalReliabilityTracker(
                warmup_frames=warmup_frames,
                stable_threshold=stable_threshold
            )

        # P4.5: Plausibility Gate
        self.plausibility_gate = None
        if self.mode in (P4Mode.FULL_P4,):
            self.plausibility_gate = PhysiologicalPlausibilityGate(
                min_bpm=min_bpm,
                max_bpm=max_bpm
            )

    def estimate(
        self,
        bpm_estimate: float,
        sqi: float,
        candidate_estimates: Optional[List[Tuple[float, float]]] = None,
        # P3 inputs (optional)
        p3_spatial_quality: Optional[float] = None,
        p3_motion_confidence: Optional[float] = None,
        p3_temporal_stability: Optional[float] = None,
        p3_candidate_scores: Optional[List[float]] = None,
        # Additional inputs
        spectral_snr_db: float = 0.0,
        model_variance: float = 4.0
    ) -> P4EstimateResult:
        """Estimate with P4 reliability and uncertainty.

        Args:
            bpm_estimate: Primary BPM estimate
            sqi: Signal Quality Index [0, 100]
            candidate_estimates: List of (bpm, uncertainty) tuples
            p3_spatial_quality: P3 spatial quality [0, 1]
            p3_motion_confidence: P3 motion confidence [0, 1]
            p3_temporal_stability: P3 temporal stability [0, 1]
            p3_candidate_scores: P3 candidate confidence scores
            spectral_snr_db: Spectral SNR in cardiac band
            model_variance: Model/estimator variance

        Returns:
            P4EstimateResult with complete estimation
        """
        result = P4EstimateResult(mode=self.mode.value)

        # BASELINE mode: return simple estimate
        if self.mode == P4Mode.BASELINE:
            result.bpm = bpm_estimate
            result.confidence = sqi
            result.valid = bpm_estimate > 0
            return result

        # P4.1: Collect Reliability Evidence
        reliability = None
        if self.reliability_collector:
            self._collect_reliability(
                sqi=sqi,
                spectral_snr=spectral_snr_db,
                p3_spatial_quality=p3_spatial_quality,
                p3_motion_confidence=p3_motion_confidence,
                p3_temporal_stability=p3_temporal_stability
            )
            reliability = self.reliability_collector.build_current_evidence()
            result.reliability = reliability

        # P4.2: Decompose Uncertainty
        uncertainty = None
        if self.uncertainty_decomposer:
            coverage = reliability.coverage if reliability else 1.0
            spatial_q = reliability.get_effective_quality(
                EvidenceSource.SPATIAL_QUALITY
            ) if reliability else 0.8
            motion_q = reliability.get_effective_quality(
                EvidenceSource.MOTION_QUALITY
            ) if reliability else 0.8
            temporal_q = reliability.get_effective_quality(
                EvidenceSource.TEMPORAL_QUALITY
            ) if reliability else 0.8

            uncertainty = self.uncertainty_decomposer.decompose(
                measurement_noise=3.0,
                sqi=sqi,
                p3_spatial_quality=spatial_q,
                p3_motion_confidence=motion_q,
                temporal_stability=temporal_q,
                spectral_snr_db=spectral_snr_db,
                model_variance=model_variance,
                evidence_coverage=coverage,
                mean_bpm=bpm_estimate,
                n_frames=1
            )
            result.uncertainty = uncertainty

        # P4.3: Confidence Fusion from candidates
        fused = None
        if self.fusion and candidate_estimates:
            fused = self.fusion.fuse(candidate_estimates)
            result.fused_estimate = fused

        # P4.4: Temporal Reliability
        temporal_state = None
        evidence_for_temporal = reliability.total_evidence if reliability else sqi / 100.0
        confidence_for_temporal = fused.confidence if fused else sqi
        if self.temporal_tracker:
            temporal_state = self.temporal_tracker.update(
                evidence=evidence_for_temporal,
                confidence=confidence_for_temporal,
                bpm=bpm_estimate
            )
            result.temporal_state = temporal_state

        # P4.5: Physiological Plausibility
        plausibility = None
        if self.plausibility_gate:
            # Get candidate CV if available
            candidate_cv = 0.0
            if candidate_estimates and len(candidate_estimates) > 1:
                bpms = [e[0] for e in candidate_estimates]
                if np.mean(bpms) > 0:
                    candidate_cv = float(np.std(bpms) / np.mean(bpms))

            plausibility = self.plausibility_gate.evaluate(
                bpm=bpm_estimate,
                previous_bpm=temporal_state.last_bpm if temporal_state else None,
                spectral_instability=1.0 - temporal_q if reliability else 0.5,
                candidate_cv=candidate_cv
            )
            result.plausibility = plausibility

        # Compute final BPM and confidence
        final_bpm, final_confidence = self._compute_final_estimate(
            bpm_estimate=bpm_estimate,
            fused=fused,
            temporal=temporal_state,
            plausibility=plausibility,
            uncertainty=uncertainty
        )

        result.bpm = final_bpm
        result.confidence = final_confidence
        result.valid = (
            plausibility.state.value != "invalid" if plausibility else True
        )

        return result

    def _collect_reliability(
        self,
        sqi: float,
        spectral_snr: float,
        p3_spatial_quality: Optional[float],
        p3_motion_confidence: Optional[float],
        p3_temporal_stability: Optional[float]
    ) -> None:
        """Collect reliability evidence."""
        if not self.reliability_collector:
            return

        # Signal quality
        self.reliability_collector.set_signal_quality(sqi)

        # Spectral quality
        snr_normalized = np.clip((spectral_snr + 20) / 40, 0, 1)
        self.reliability_collector.set_spectral_quality(
            snr_db=spectral_snr,
            peak_prominence=snr_normalized
        )

        # P3 spatial quality
        if p3_spatial_quality is not None:
            self.reliability_collector.set_spatial_quality(p3_spatial_quality)

        # P3 motion quality
        if p3_motion_confidence is not None:
            self.reliability_collector.set_motion_quality(p3_motion_confidence)

        # Temporal quality
        if p3_temporal_stability is not None:
            self.reliability_collector.set_temporal_quality(p3_temporal_stability)

    def _compute_final_estimate(
        self,
        bpm_estimate: float,
        fused: Optional[FusedEstimate],
        temporal: Optional[TemporalReliabilityResult],
        plausibility: Optional[PlausibilityResult],
        uncertainty: Optional[UncertaintyDecomposition]
    ) -> Tuple[float, float]:
        """Compute final estimate considering all factors."""
        # Start with base estimate
        final_bpm = bpm_estimate
        confidence_components = []

        # Add fused estimate if available
        if fused and fused.valid:
            # Weight between direct estimate and fused
            fused_weight = 0.7
            final_bpm = fused_weight * fused.bpm + (1 - fused_weight) * bpm_estimate
            confidence_components.append(fused.confidence)

        # Adjust confidence from temporal state
        if temporal:
            if temporal.state == TemporalState.STABLE:
                confidence_components.append(temporal.confidence)
            elif temporal.state == TemporalState.WARMUP:
                confidence_components.append(temporal.confidence * 0.8)
            elif temporal.state in (TemporalState.DEGRADING, TemporalState.UNCERTAIN):
                confidence_components.append(temporal.confidence * 0.6)
            else:  # INVALID
                confidence_components.append(temporal.confidence * 0.2)

        # Adjust from plausibility
        if plausibility:
            if plausibility.state == PlausibilityState.PLAUSIBLE:
                confidence_components.append(95.0)
            elif plausibility.state == PlausibilityState.UNCERTAIN:
                confidence_components.append(60.0)
            else:
                confidence_components.append(20.0)

        # Compute final confidence
        if confidence_components:
            final_confidence = float(np.mean(confidence_components))
        else:
            final_confidence = 50.0

        return final_bpm, final_confidence

    def reset(self) -> None:
        """Reset all components."""
        if self.reliability_collector:
            self.reliability_collector.reset()
        if self.temporal_tracker:
            self.temporal_tracker.reset()
        if self.plausibility_gate:
            self.plausibility_gate.reset()


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_baseline_mode() -> bool:
    """Test BASELINE mode."""
    print("  test_baseline_mode...")

    estimator = P4Estimator(mode=P4Mode.BASELINE)

    result = estimator.estimate(bpm_estimate=72.0, sqi=85.0)

    assert result.mode == "baseline"
    assert result.bpm == 72.0
    assert result.confidence == 85.0

    print(f"    PASS")
    return True


def test_reliability_only_mode() -> bool:
    """Test RELIABILITY_ONLY mode."""
    print("  test_reliability_only_mode...")

    estimator = P4Estimator(mode=P4Mode.RELIABILITY_ONLY)

    result = estimator.estimate(
        bpm_estimate=72.0,
        sqi=85.0,
        p3_spatial_quality=0.8,
        p3_motion_confidence=0.9
    )

    assert result.reliability is not None
    assert result.reliability.has_source(EvidenceSource.SIGNAL_QUALITY)

    print(f"    PASS")
    return True


def test_full_p4_mode() -> bool:
    """Test FULL_P4 mode."""
    print("  test_full_p4_mode...")

    estimator = P4Estimator(mode=P4Mode.FULL_P4)

    result = estimator.estimate(
        bpm_estimate=72.0,
        sqi=85.0,
        candidate_estimates=[(72.0, 3.0), (73.0, 5.0)],
        p3_spatial_quality=0.8,
        p3_motion_confidence=0.9,
        p3_temporal_stability=0.85,
        spectral_snr_db=10.0
    )

    assert result.reliability is not None
    assert result.uncertainty is not None
    assert result.temporal_state is not None
    assert result.plausibility is not None
    assert result.fused_estimate is not None

    print(f"    BPM: {result.bpm:.1f}, Confidence: {result.confidence:.1f}%")
    print(f"    PASS")
    return True


def test_plausibility_rejection() -> bool:
    """Test that implausible BPM is flagged."""
    print("  test_plausibility_rejection...")

    estimator = P4Estimator(mode=P4Mode.FULL_P4)

    # Implausible BPM
    result = estimator.estimate(
        bpm_estimate=250.0,  # Way too high
        sqi=90.0
    )

    assert result.plausibility is not None
    assert result.plausibility.state == PlausibilityState.INVALID

    print(f"    PASS")
    return True


def test_reset() -> bool:
    """Test estimator reset."""
    print("  test_reset...")

    estimator = P4Estimator(mode=P4Mode.FULL_P4)

    # Add some state
    estimator.estimate(bpm_estimate=72.0, sqi=85.0)

    # Reset
    estimator.reset()

    print(f"    PASS")
    return True


def run_tests() -> bool:
    """Run all P4.6 tests."""
    print("\n" + "=" * 60)
    print("P4.6: Confidence-Aware Estimation Tests")
    print("=" * 60)

    tests = [
        ("baseline_mode", test_baseline_mode),
        ("reliability_only_mode", test_reliability_only_mode),
        ("full_p4_mode", test_full_p4_mode),
        ("plausibility_rejection", test_plausibility_rejection),
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
    print(f"P4.6 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
