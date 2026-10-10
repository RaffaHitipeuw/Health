"""
P4.1: Reliability Evidence Model

Creates a formal reliability evidence model that captures multiple
evidence sources for estimating physiological signal reliability.

Evidence sources:
- signal_quality: Overall signal quality indicator
- spectral_quality: Cardiac-band spectral quality
- spatial_quality: P3 spatial quality contribution
- motion_quality: P3 motion quality contribution
- temporal_quality: Temporal consistency
- physiological_plausibility: Plausibility gate result
- candidate_stability: P3 candidate persistence
- agreement_quality: Inter-candidate agreement
- harmonic_risk: Harmonic contamination risk
- illumination_quality: Illumination stability
- evidence_coverage: Fraction of evidence sources available

Each evidence component has:
- definition
- source
- units/range
- interpretation
- failure mode
- whether higher/lower is better
- whether independent or derived
- circularity risk

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from collections import deque


# Evidence value ranges
QUALITY_RANGE = (0.0, 1.0)  # 0 = worst, 1 = best
CONFIDENCE_RANGE = (0.0, 100.0)  # percentage


class EvidenceSource(Enum):
    """Sources of reliability evidence."""
    SIGNAL_QUALITY = "signal_quality"           # Raw signal quality
    SPECTRAL_QUALITY = "spectral_quality"       # Cardiac band spectral quality
    SPATIAL_QUALITY = "spatial_quality"         # P3 spatial quality
    MOTION_QUALITY = "motion_quality"           # P3 motion quality
    TEMPORAL_QUALITY = "temporal_quality"       # Temporal consistency
    PHYSIOLOGICAL_PLAUSIBILITY = "physiological_plausibility"  # Plausibility gate
    CANDIDATE_STABILITY = "candidate_stability"   # P3 candidate persistence
    AGREEMENT_QUALITY = "agreement_quality"       # Inter-candidate agreement
    HARMONIC_RISK = "harmonic_risk"             # Harmonic contamination
    ILLUMINATION_QUALITY = "illumination_quality"  # Illumination stability


@dataclass
class EvidenceComponent:
    """Single evidence component.

    Attributes:
        name: Human-readable name
        source: Where this evidence comes from
        value: Evidence value [0, 1]
        confidence: Confidence in this evidence [0, 100]
        higher_is_better: Whether higher value indicates better quality
        is_independent: Whether this is independent evidence
        circularity_risk: Whether this risks circular reasoning
        metadata: Additional context
    """
    name: str
    source: EvidenceSource
    value: float = 0.0
    confidence: float = 100.0  # confidence in the evidence itself
    higher_is_better: bool = True
    is_independent: bool = True
    circularity_risk: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_value(self) -> float:
        """Get value normalized to [0, 1] where 1 = best."""
        if self.higher_is_better:
            return np.clip(self.value, 0.0, 1.0)
        else:
            return np.clip(1.0 - self.value, 0.0, 1.0)

    def effective_quality(self) -> float:
        """Effective quality considering evidence confidence."""
        return self.normalized_value() * (self.confidence / 100.0)


@dataclass
class ReliabilityEvidence:
    """Complete reliability evidence model.

    Aggregates all evidence sources into a unified model.

    Attributes:
        timestamp: Frame/timestamp this evidence applies to
        components: Dictionary of evidence components
        coverage: Fraction of evidence sources available
        total_evidence: Aggregated evidence score
        confidence: Overall confidence in the evidence model
        valid: Whether evidence is valid
        error: Error message if invalid
    """
    timestamp: float = 0.0
    components: Dict[str, EvidenceComponent] = field(default_factory=dict)
    coverage: float = 0.0
    total_evidence: float = 0.0
    confidence: float = 0.0
    valid: bool = True
    error: Optional[str] = None

    def get_component(self, source: EvidenceSource) -> Optional[EvidenceComponent]:
        """Get evidence component by source."""
        return self.components.get(source.value)

    def get_effective_quality(self, source: EvidenceSource) -> float:
        """Get effective quality for a source."""
        comp = self.get_component(source)
        if comp is None:
            return 0.0
        return comp.effective_quality()

    def get_independent_evidence(self) -> List[float]:
        """Get only independent evidence values."""
        return [
            comp.effective_quality()
            for comp in self.components.values()
            if comp.is_independent
        ]

    def get_all_evidence(self) -> List[float]:
        """Get all evidence values."""
        return [comp.effective_quality() for comp in self.components.values()]

    def has_source(self, source: EvidenceSource) -> bool:
        """Check if evidence source is available."""
        return source.value in self.components

    def update_coverage(self) -> None:
        """Update coverage fraction."""
        if not self.components:
            self.coverage = 0.0
            return

        # Count components with actual values
        active = sum(1 for c in self.components.values() if c.value > 0)
        total = len(self.components)
        self.coverage = float(active) / float(total) if total > 0 else 0.0

    def compute_total_evidence(self, method: str = "mean") -> float:
        """Compute total evidence score.

        Args:
            method: Aggregation method ("mean", "min", "weighted_mean")
        """
        independent = self.get_independent_evidence()
        if not independent:
            return 0.0

        if method == "mean":
            self.total_evidence = float(np.mean(independent))
        elif method == "min":
            self.total_evidence = float(np.min(independent))
        elif method == "weighted_mean":
            # Weight by confidence
            weights = []
            values = []
            for comp in self.components.values():
                if comp.is_independent:
                    weights.append(comp.confidence)
                    values.append(comp.normalized_value())
            if weights:
                total_weight = sum(weights)
                self.total_evidence = sum(w * v for w, v in zip(weights, values)) / total_weight
            else:
                self.total_evidence = 0.0
        else:
            self.total_evidence = float(np.mean(independent))

        return self.total_evidence

    def compute_confidence(self) -> float:
        """Compute overall confidence in the evidence model."""
        if not self.components:
            self.confidence = 0.0
            return self.confidence

        # Consider coverage and average evidence confidence
        cover_factor = self.coverage
        avg_evidence_conf = np.mean([c.confidence for c in self.components.values()])

        self.confidence = cover_factor * (avg_evidence_conf / 100.0) * 100.0
        return self.confidence


class ReliabilityEvidenceCollector:
    """
    Collects and manages reliability evidence.

    Builds evidence from multiple sources over time.
    """

    def __init__(
        self,
        window_size: int = 30,
        evidence_sources: Optional[List[EvidenceSource]] = None
    ):
        """
        Initialize evidence collector.

        Args:
            window_size: Frames for temporal aggregation
            evidence_sources: List of sources to track
        """
        self.window_size = window_size
        self.evidence_sources = evidence_sources or list(EvidenceSource)

        # Current evidence
        self._current: Dict[str, EvidenceComponent] = {}

        # Temporal history
        self._history: deque = deque(maxlen=window_size)

        # Frame counter
        self._frame_count: int = 0

    def set_evidence(
        self,
        source: EvidenceSource,
        value: float,
        confidence: float = 100.0,
        higher_is_better: bool = True,
        is_independent: bool = True,
        circularity_risk: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set evidence for a source.

        Args:
            source: Evidence source enum
            value: Evidence value [0, 1] or [0, 100] depending on context
            confidence: Confidence in this evidence [0, 100]
            higher_is_better: Whether higher value = better quality
            is_independent: Whether independent evidence
            circularity_risk: Risk of circular reasoning
            metadata: Additional context
        """
        # Normalize value to [0, 1] for standard sources
        if source in (EvidenceSource.SIGNAL_QUALITY,
                      EvidenceSource.SPECTRAL_QUALITY,
                      EvidenceSource.SPATIAL_QUALITY,
                      EvidenceSource.MOTION_QUALITY,
                      EvidenceSource.TEMPORAL_QUALITY,
                      EvidenceSource.CANDIDATE_STABILITY,
                      EvidenceSource.AGREEMENT_QUALITY,
                      EvidenceSource.ILLUMINATION_QUALITY):
            normalized_value = np.clip(value, 0.0, 1.0)
        elif source == EvidenceSource.PHYSIOLOGICAL_PLAUSIBILITY:
            # 1 = plausible, 0 = implausible
            normalized_value = np.clip(value, 0.0, 1.0)
        elif source == EvidenceSource.HARMONIC_RISK:
            # 0 = no risk, 1 = high risk (invert for quality)
            normalized_value = np.clip(1.0 - value, 0.0, 1.0)
        else:
            normalized_value = np.clip(value, 0.0, 1.0)

        self._current[source.value] = EvidenceComponent(
            name=source.value,
            source=source,
            value=normalized_value,
            confidence=np.clip(confidence, 0.0, 100.0),
            higher_is_better=higher_is_better,
            is_independent=is_independent,
            circularity_risk=circularity_risk,
            metadata=metadata or {}
        )

    def set_signal_quality(self, sqi: float) -> None:
        """Set signal quality evidence (from SQI).

        Args:
            sqi: Signal Quality Index [0, 100]
        """
        self.set_evidence(
            EvidenceSource.SIGNAL_QUALITY,
            value=sqi / 100.0,
            confidence=90.0,  # SQI is reasonably reliable
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False,
            metadata={"raw_sqi": sqi}
        )

    def set_spectral_quality(
        self,
        snr_db: float,
        peak_prominence: float,
        confidence: float = 85.0
    ) -> None:
        """Set spectral quality evidence.

        Args:
            snr_db: SNR in cardiac band (dB)
            peak_prominence: Peak prominence [0, 1]
            confidence: Confidence in spectral analysis
        """
        # Combine SNR and prominence
        snr_normalized = np.clip((snr_db + 20.0) / 40.0, 0.0, 1.0)  # -20 to 20 dB -> 0 to 1
        combined = 0.5 * snr_normalized + 0.5 * peak_prominence

        self.set_evidence(
            EvidenceSource.SPECTRAL_QUALITY,
            value=combined,
            confidence=confidence,
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False,
            metadata={"snr_db": snr_db, "peak_prominence": peak_prominence}
        )

    def set_spatial_quality(self, p3_quality_map_mean: float) -> None:
        """Set P3 spatial quality evidence.

        Args:
            p3_quality_map_mean: Mean quality from P3 spatial quality map
        """
        # P3 quality is already normalized
        self.set_evidence(
            EvidenceSource.SPATIAL_QUALITY,
            value=p3_quality_map_mean,
            confidence=80.0,  # Depends on P3 implementation
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False
        )

    def set_motion_quality(self, motion_confidence: float) -> None:
        """Set P3 motion quality evidence.

        Args:
            motion_confidence: Motion confidence from P3 motion field [0, 1]
        """
        self.set_evidence(
            EvidenceSource.MOTION_QUALITY,
            value=motion_confidence,
            confidence=75.0,
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False
        )

    def set_temporal_quality(self, stability_score: float) -> None:
        """Set temporal quality evidence.

        Args:
            stability_score: Temporal stability [0, 1]
        """
        self.set_evidence(
            EvidenceSource.TEMPORAL_QUALITY,
            value=stability_score,
            confidence=85.0,
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False
        )

    def set_physiological_plausibility(
        self,
        is_plausible: bool,
        confidence: float = 90.0
    ) -> None:
        """Set physiological plausibility evidence.

        Args:
            is_plausible: Whether estimate is physiologically plausible
            confidence: Confidence in plausibility assessment
        """
        self.set_evidence(
            EvidenceSource.PHYSIOLOGICAL_PLAUSIBILITY,
            value=1.0 if is_plausible else 0.0,
            confidence=confidence,
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False
        )

    def set_candidate_stability(self, persistence_score: float) -> None:
        """Set P3 candidate stability evidence.

        Args:
            persistence_score: Candidate persistence [0, 1]
        """
        self.set_evidence(
            EvidenceSource.CANDIDATE_STABILITY,
            value=persistence_score,
            confidence=80.0,
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False
        )

    def set_agreement_quality(self, inter_candidate_cv: float) -> None:
        """Set inter-candidate agreement evidence.

        Args:
            inter_candidate_cv: Coefficient of variation between candidates
        """
        # Low CV = high agreement
        agreement = np.clip(1.0 - inter_candidate_cv, 0.0, 1.0)
        self.set_evidence(
            EvidenceSource.AGREEMENT_QUALITY,
            value=agreement,
            confidence=85.0,
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False,
            metadata={"cv": inter_candidate_cv}
        )

    def set_harmonic_risk(self, harmonic_ratio: float) -> None:
        """Set harmonic contamination risk.

        Args:
            harmonic_ratio: Ratio of harmonic power to fundamental [0, 1]
        """
        self.set_evidence(
            EvidenceSource.HARMONIC_RISK,
            value=harmonic_ratio,
            confidence=80.0,
            higher_is_better=False,  # Lower = less risk
            is_independent=True,
            circularity_risk=False
        )

    def set_illumination_quality(self, illumination_stability: float) -> None:
        """Set illumination stability evidence.

        Args:
            illumination_stability: Illumination stability [0, 1]
        """
        self.set_evidence(
            EvidenceSource.ILLUMINATION_QUALITY,
            value=illumination_stability,
            confidence=75.0,
            higher_is_better=True,
            is_independent=True,
            circularity_risk=False
        )

    def build_current_evidence(self) -> ReliabilityEvidence:
        """Build current reliability evidence.

        Returns:
            ReliabilityEvidence for current frame
        """
        self._frame_count += 1

        evidence = ReliabilityEvidence(
            timestamp=float(self._frame_count),
            components=self._current.copy()
        )
        evidence.update_coverage()
        evidence.compute_total_evidence(method="weighted_mean")
        evidence.compute_confidence()

        # Store in history
        self._history.append(evidence)

        return evidence

    def get_temporal_evidence(self) -> Optional[ReliabilityEvidence]:
        """Get temporally aggregated evidence.

        Returns:
            Aggregated evidence over temporal window
        """
        if not self._history:
            return None

        # Aggregate over history
        timestamps = [e.timestamp for e in self._history]

        # Mean evidence per source
        aggregated: Dict[str, List[float]] = {}
        for e in self._history:
            for source, comp in e.components.items():
                if source not in aggregated:
                    aggregated[source] = []
                aggregated[source].append(comp.value)

        result = ReliabilityEvidence(
            timestamp=float(np.mean(timestamps)),
            components={},
            valid=True
        )

        for source, values in aggregated.items():
            result.components[source] = EvidenceComponent(
                name=source,
                source=EvidenceSource(source),
                value=float(np.mean(values)),
                confidence=100.0,  # Aggregated
                is_independent=True
            )

        result.update_coverage()
        result.compute_total_evidence()
        result.compute_confidence()

        return result

    def reset(self) -> None:
        """Reset evidence collector."""
        self._current.clear()
        self._history.clear()
        self._frame_count = 0


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_evidence_collection() -> bool:
    """Test basic evidence collection."""
    print("  test_evidence_collection...")

    collector = ReliabilityEvidenceCollector()

    # Set evidence
    collector.set_signal_quality(85.0)
    collector.set_spectral_quality(snr_db=10.0, peak_prominence=0.8)
    collector.set_motion_quality(0.9)
    collector.set_temporal_quality(0.85)
    collector.set_physiological_plausibility(True)

    evidence = collector.build_current_evidence()

    assert evidence.valid
    assert evidence.has_source(EvidenceSource.SIGNAL_QUALITY)
    assert evidence.has_source(EvidenceSource.SPECTRAL_QUALITY)
    assert evidence.coverage > 0

    print(f"    coverage: {evidence.coverage:.2f}")
    print(f"    total_evidence: {evidence.total_evidence:.3f}")
    print(f"    PASS")
    return True


def test_evidence_normalization() -> bool:
    """Test that evidence values are properly normalized."""
    print("  test_evidence_normalization...")

    collector = ReliabilityEvidenceCollector()

    # Test SQI normalization (0-100 -> 0-1)
    collector.set_signal_quality(50.0)
    e1 = collector.build_current_evidence()
    assert 0.4 < e1.get_component(EvidenceSource.SIGNAL_QUALITY).value < 0.6

    # Test harmonic risk inversion (low risk = high quality)
    collector.set_harmonic_risk(0.2)  # Low risk
    e2 = collector.build_current_evidence()
    assert e2.get_component(EvidenceSource.HARMONIC_RISK).value > 0.7

    print(f"    PASS")
    return True


def test_independent_evidence() -> bool:
    """Test independent evidence filtering."""
    print("  test_independent_evidence...")

    collector = ReliabilityEvidenceCollector()
    collector.set_signal_quality(80.0)
    collector.set_spectral_quality(10.0, 0.8)

    evidence = collector.build_current_evidence()
    independent = evidence.get_independent_evidence()

    assert len(independent) >= 2

    print(f"    independent sources: {len(independent)}")
    print(f"    PASS")
    return True


def test_evidence_aggregation() -> bool:
    """Test evidence aggregation methods."""
    print("  test_evidence_aggregation...")

    collector = ReliabilityEvidenceCollector()
    collector.set_signal_quality(90.0)
    collector.set_spectral_quality(15.0, 0.9)
    collector.set_motion_quality(0.8)

    evidence = collector.build_current_evidence()

    # Test different aggregation methods
    mean_evidence = evidence.compute_total_evidence("mean")
    min_evidence = evidence.compute_total_evidence("min")
    weighted_evidence = evidence.compute_total_evidence("weighted_mean")

    assert min_evidence <= mean_evidence <= weighted_evidence + 0.01

    print(f"    mean: {mean_evidence:.3f}, min: {min_evidence:.3f}")
    print(f"    PASS")
    return True


def test_temporal_history() -> bool:
    """Test temporal evidence aggregation."""
    print("  test_temporal_history...")

    collector = ReliabilityEvidenceCollector(window_size=5)

    # Add frames with varying quality
    for sqi in [90.0, 85.0, 80.0, 75.0, 70.0]:
        collector.set_signal_quality(sqi)
        collector.build_current_evidence()

    temporal = collector.get_temporal_evidence()
    assert temporal is not None

    # Mean SQI over 5 frames should be ~80
    mean_component = temporal.get_component(EvidenceSource.SIGNAL_QUALITY)
    assert 0.75 < mean_component.value < 0.85

    print(f"    temporal mean: {mean_component.value:.3f}")
    print(f"    PASS")
    return True


def test_invalid_evidence() -> bool:
    """Test handling of invalid/missing evidence."""
    print("  test_invalid_evidence...")

    collector = ReliabilityEvidenceCollector()

    # No evidence set
    evidence = collector.build_current_evidence()

    assert not evidence.has_source(EvidenceSource.SIGNAL_QUALITY)
    assert evidence.coverage == 0.0
    assert evidence.total_evidence == 0.0

    print(f"    coverage: {evidence.coverage:.2f}")
    print(f"    PASS")
    return True


def test_evidence_confidence() -> bool:
    """Test evidence confidence computation."""
    print("  test_evidence_confidence...")

    collector = ReliabilityEvidenceCollector()
    collector.set_signal_quality(90.0)

    evidence = collector.build_current_evidence()

    # Coverage should be reasonable
    assert 0.0 < evidence.confidence <= 100.0

    print(f"    confidence: {evidence.confidence:.1f}%")
    print(f"    PASS")
    return True


def run_tests() -> bool:
    """Run all P4.1 tests."""
    print("\n" + "=" * 60)
    print("P4.1: Reliability Evidence Model Tests")
    print("=" * 60)

    tests = [
        ("evidence_collection", test_evidence_collection),
        ("evidence_normalization", test_evidence_normalization),
        ("independent_evidence", test_independent_evidence),
        ("evidence_aggregation", test_evidence_aggregation),
        ("temporal_history", test_temporal_history),
        ("invalid_evidence", test_invalid_evidence),
        ("evidence_confidence", test_evidence_confidence),
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
    print(f"P4.1 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
