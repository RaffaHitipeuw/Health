"""
P4.2: Uncertainty Decomposition

Implements explicit uncertainty decomposition that distinguishes:

1. "the estimate is uncertain" (aleatoric)
2. "the estimate is confidently wrong" (epistemic/model)
3. "there is not enough evidence to produce an estimate" (coverage)

Key distinctions:
- measurement uncertainty: irreducible noise in the signal
- signal/extraction uncertainty: how well we extracted the signal
- spatial uncertainty: P3 spatial quality contribution
- motion uncertainty: P3 motion quality contribution
- temporal uncertainty: instability over time
- spectral uncertainty: cardiac band quality
- model/fusion uncertainty: estimator uncertainty
- invalid/insufficient state: not enough evidence

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


# Physiological bounds for sanity
MIN_BPM = 40.0
MAX_BPM = 200.0
TYPICAL_BPM_STD = 5.0  # Typical variation


class UncertaintySource(Enum):
    """Sources of uncertainty."""
    MEASUREMENT = "measurement"           # Irreducible noise
    SIGNAL_EXTRACTION = "signal_extraction" # Extraction quality
    SPATIAL = "spatial"                   # P3 spatial
    MOTION = "motion"                    # P3 motion
    TEMPORAL = "temporal"                # Temporal instability
    SPECTRAL = "spectral"               # Cardiac band quality
    MODEL = "model"                     # Estimator/fusion
    COVERAGE = "coverage"               # Evidence coverage
    HARMONIC = "harmonic"               # Harmonic contamination


@dataclass
class UncertaintyComponent:
    """Single uncertainty component.

    Attributes:
        source: Uncertainty source
        value: Uncertainty value (std in BPM or normalized)
        unit: Unit type ("bpm", "normalized", "fraction")
        confidence: Confidence in this estimate
        is_aleatoric: Whether irreducible uncertainty
        is_epistemic: Whether reducible with more data
    """
    source: UncertaintySource
    value: float = 0.0
    unit: str = "normalized"  # "bpm", "normalized", "fraction"
    confidence: float = 100.0
    is_aleatoric: bool = False
    is_epistemic: bool = False

    def to_bpm(self, mean_bpm: float) -> float:
        """Convert to BPM units if normalized."""
        if self.unit == "bpm":
            return self.value
        elif self.unit == "normalized":
            return self.value * TYPICAL_BPM_STD
        elif self.unit == "fraction":
            return self.value * mean_bpm
        return self.value


@dataclass
class UncertaintyDecomposition:
    """Complete uncertainty decomposition.

    Attributes:
        measurement_uncertainty: Irreducible measurement noise
        signal_uncertainty: Signal extraction uncertainty
        spatial_uncertainty: P3 spatial quality uncertainty
        motion_uncertainty: P3 motion quality uncertainty
        temporal_uncertainty: Temporal instability
        spectral_uncertainty: Cardiac band quality
        model_uncertainty: Estimator/fusion uncertainty
        coverage_uncertainty: Evidence coverage uncertainty
        total_uncertainty: Combined uncertainty
        aleatoric_total: Total irreducible uncertainty
        epistemic_total: Total reducible uncertainty
        validity: Estimate validity state
    """
    measurement: Optional[UncertaintyComponent] = None
    signal: Optional[UncertaintyComponent] = None
    spatial: Optional[UncertaintyComponent] = None
    motion: Optional[UncertaintyComponent] = None
    temporal: Optional[UncertaintyComponent] = None
    spectral: Optional[UncertaintyComponent] = None
    harmonic: Optional[UncertaintyComponent] = None
    model: Optional[UncertaintyComponent] = None
    coverage: Optional[UncertaintyComponent] = None

    # Aggregated
    total_uncertainty: float = 0.0
    aleatoric_total: float = 0.0
    epistemic_total: float = 0.0

    # Validity
    validity: str = "VALID"  # VALID, UNCERTAIN, INVALID

    # Metadata
    mean_bpm: float = 0.0
    n_frames: int = 0

    def get_component(self, source: UncertaintySource) -> Optional[UncertaintyComponent]:
        """Get uncertainty component by source."""
        mapping = {
            UncertaintySource.MEASUREMENT: self.measurement,
            UncertaintySource.SIGNAL_EXTRACTION: self.signal,
            UncertaintySource.SPATIAL: self.spatial,
            UncertaintySource.MOTION: self.motion,
            UncertaintySource.TEMPORAL: self.temporal,
            UncertaintySource.SPECTRAL: self.spectral,
            UncertaintySource.HARMONIC: self.harmonic,
            UncertaintySource.MODEL: self.model,
            UncertaintySource.COVERAGE: self.coverage,
        }
        return mapping.get(source)

    def get_uncertainty_bpm(self, source: UncertaintySource) -> float:
        """Get uncertainty in BPM units."""
        comp = self.get_component(source)
        if comp is None:
            return 0.0
        return comp.to_bpm(self.mean_bpm)

    def compute_totals(self) -> None:
        """Compute total uncertainties."""
        components = [
            self.measurement, self.signal, self.spatial,
            self.motion, self.temporal, self.spectral,
            self.harmonic, self.model, self.coverage
        ]

        # Total uncertainty (RSS)
        total_var = 0.0
        aleatoric_var = 0.0
        epistemic_var = 0.0

        for comp in components:
            if comp is not None:
                var = comp.to_bpm(self.mean_bpm) ** 2
                total_var += var

                if comp.is_aleatoric:
                    aleatoric_var += var
                if comp.is_epistemic:
                    epistemic_var += var

        self.total_uncertainty = float(np.sqrt(total_var)) if total_var > 0 else 0.0
        self.aleatoric_total = float(np.sqrt(aleatoric_var)) if aleatoric_var > 0 else 0.0
        self.epistemic_total = float(np.sqrt(epistemic_var)) if epistemic_var > 0 else 0.0

    def determine_validity(
        self,
        min_coverage: float = 0.3,
        max_total_std: float = 15.0,
        max_epistemic_std: float = 10.0
    ) -> str:
        """Determine validity state based on uncertainties.

        Distinguishes:
        - VALID: estimate is reliable
        - UNCERTAIN: estimate exists but is uncertain
        - INVALID: not enough evidence for valid estimate
        """
        # Check coverage (use confidence field which stores fraction)
        if self.coverage is not None:
            coverage_fraction = self.coverage.confidence / 100.0
            if coverage_fraction < min_coverage:
                self.validity = "INVALID"
                return self.validity

        # Check if total uncertainty is too high
        if self.total_uncertainty > max_total_std:
            self.validity = "UNCERTAIN"
            return self.validity

        # Check epistemic uncertainty dominates
        if self.epistemic_total > max_epistemic_std:
            self.validity = "UNCERTAIN"
            return self.validity

        self.validity = "VALID"
        return self.validity

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "validity": self.validity,
            "total_uncertainty_bpm": round(self.total_uncertainty, 2),
            "aleatoric_bpm": round(self.aleatoric_total, 2),
            "epistemic_bpm": round(self.epistemic_total, 2),
            "mean_bpm": round(self.mean_bpm, 1),
            "n_frames": self.n_frames,
        }

        # Add individual components
        for source in UncertaintySource:
            comp = self.get_component(source)
            if comp is not None:
                result[f"{source.value}_bpm"] = round(comp.to_bpm(self.mean_bpm), 2)

        return result


class UncertaintyDecomposer:
    """
    Decomposes total uncertainty into components.

    Combines evidence from multiple sources into a structured
    uncertainty representation.
    """

    def __init__(self):
        """Initialize uncertainty decomposer."""
        self._current: Optional[UncertaintyDecomposition] = None

    def decompose(
        self,
        measurement_noise: float,
        sqi: float,
        p3_spatial_quality: float,
        p3_motion_confidence: float,
        temporal_stability: float,
        spectral_snr_db: float,
        model_variance: float,
        evidence_coverage: float,
        mean_bpm: float,
        n_frames: int
    ) -> UncertaintyDecomposition:
        """Decompose uncertainty into components.

        Args:
            measurement_noise: Measurement noise std (BPM units)
            sqi: Signal Quality Index [0, 100]
            p3_spatial_quality: P3 spatial quality [0, 1]
            p3_motion_confidence: P3 motion confidence [0, 1]
            temporal_stability: Temporal stability [0, 1]
            spectral_snr_db: Spectral SNR in cardiac band (dB)
            model_variance: Model/estimator variance
            evidence_coverage: Fraction of evidence sources available [0, 1]
            mean_bpm: Mean heart rate estimate
            n_frames: Number of frames used

        Returns:
            UncertaintyDecomposition with all components
        """
        decomposer = UncertaintyDecomposition(
            mean_bpm=mean_bpm,
            n_frames=n_frames
        )

        # 1. Measurement uncertainty (aleatoric)
        # Higher SQI = lower measurement noise
        sqi_factor = np.clip(sqi / 100.0, 0.0, 1.0)
        measurement_std = measurement_noise * (1.0 - sqi_factor * 0.5)
        decomposer.measurement = UncertaintyComponent(
            source=UncertaintySource.MEASUREMENT,
            value=measurement_std,
            unit="bpm",
            is_aleatoric=True,
            is_epistemic=False
        )

        # 2. Signal extraction uncertainty
        # Based on SQI
        signal_std = (1.0 - sqi_factor) * 10.0  # 0-10 BPM
        decomposer.signal = UncertaintyComponent(
            source=UncertaintySource.SIGNAL_EXTRACTION,
            value=signal_std,
            unit="bpm",
            confidence=sqi_factor * 100.0,
            is_aleatoric=True,
            is_epistemic=True
        )

        # 3. Spatial uncertainty (from P3)
        # Lower spatial quality = higher uncertainty
        spatial_std = (1.0 - p3_spatial_quality) * 8.0  # 0-8 BPM
        decomposer.spatial = UncertaintyComponent(
            source=UncertaintySource.SPATIAL,
            value=spatial_std,
            unit="bpm",
            confidence=p3_spatial_quality * 100.0,
            is_epistemic=True,
            is_aleatoric=False
        )

        # 4. Motion uncertainty (from P3)
        # Lower motion confidence = higher uncertainty
        motion_std = (1.0 - p3_motion_confidence) * 6.0  # 0-6 BPM
        decomposer.motion = UncertaintyComponent(
            source=UncertaintySource.MOTION,
            value=motion_std,
            unit="bpm",
            confidence=p3_motion_confidence * 100.0,
            is_epistemic=True,
            is_aleatoric=False
        )

        # 5. Temporal uncertainty
        # Lower stability = higher uncertainty
        temporal_std = (1.0 - temporal_stability) * 5.0  # 0-5 BPM
        decomposer.temporal = UncertaintyComponent(
            source=UncertaintySource.TEMPORAL,
            value=temporal_std,
            unit="bpm",
            confidence=temporal_stability * 100.0,
            is_epistemic=True,
            is_aleatoric=False
        )

        # 6. Spectral uncertainty
        # Based on SNR in cardiac band
        # Convert dB to normalized
        snr_normalized = np.clip((spectral_snr_db + 20.0) / 40.0, 0.0, 1.0)
        spectral_std = (1.0 - snr_normalized) * 8.0  # 0-8 BPM
        decomposer.spectral = UncertaintyComponent(
            source=UncertaintySource.SPECTRAL,
            value=spectral_std,
            unit="bpm",
            confidence=snr_normalized * 100.0,
            is_epistemic=True,
            is_aleatoric=True
        )

        # 6b. Harmonic uncertainty (placeholder)
        decomposer.harmonic = UncertaintyComponent(
            source=UncertaintySource.HARMONIC,
            value=0.0,
            unit="bpm",
            confidence=100.0,
            is_epistemic=True,
            is_aleatoric=True
        )

        # 7. Model uncertainty (from estimator)
        decomposer.model = UncertaintyComponent(
            source=UncertaintySource.MODEL,
            value=np.sqrt(max(model_variance, 0.01)),
            unit="bpm",
            is_epistemic=True,
            is_aleatoric=False
        )

        # 8. Coverage uncertainty
        # Low coverage = high uncertainty
        coverage_std = (1.0 - evidence_coverage) * 10.0  # 0-10 BPM
        decomposer.coverage = UncertaintyComponent(
            source=UncertaintySource.COVERAGE,
            value=coverage_std,
            unit="bpm",
            confidence=evidence_coverage * 100.0,
            is_epistemic=True,
            is_aleatoric=False
        )

        # Compute totals
        decomposer.compute_totals()

        # Determine validity
        decomposer.determine_validity()

        self._current = decomposer
        return decomposer

    def decompose_from_evidence(
        self,
        reliability_evidence,  # From P4.1
        sqi: float,
        model_variance: float,
        mean_bpm: float,
        n_frames: int
    ) -> UncertaintyDecomposition:
        """Decompose uncertainty from reliability evidence.

        Args:
            reliability_evidence: ReliabilityEvidence object
            sqi: Signal Quality Index
            model_variance: Model variance
            mean_bpm: Mean estimate
            n_frames: Frame count

        Returns:
            UncertaintyDecomposition
        """
        from p4_reliability.reliability_evidence import EvidenceSource

        # Extract components from evidence
        spatial_q = reliability_evidence.get_effective_quality(EvidenceSource.SPATIAL_QUALITY)
        motion_q = reliability_evidence.get_effective_quality(EvidenceSource.MOTION_QUALITY)
        temporal_q = reliability_evidence.get_effective_quality(EvidenceSource.TEMPORAL_QUALITY)
        spectral_q = reliability_evidence.get_effective_quality(EvidenceSource.SPECTRAL_QUALITY)
        coverage = reliability_evidence.coverage

        # Get spectral SNR from metadata if available
        spectral_comp = reliability_evidence.get_component(EvidenceSource.SPECTRAL_QUALITY)
        snr_db = 0.0
        if spectral_comp and "snr_db" in spectral_comp.metadata:
            snr_db = spectral_comp.metadata["snr_db"]

        return self.decompose(
            measurement_noise=3.0,  # Typical measurement noise
            sqi=sqi,
            p3_spatial_quality=spatial_q,
            p3_motion_confidence=motion_q,
            temporal_stability=temporal_q,
            spectral_snr_db=snr_db,
            model_variance=model_variance,
            evidence_coverage=coverage,
            mean_bpm=mean_bpm,
            n_frames=n_frames
        )

    def get_current(self) -> Optional[UncertaintyDecomposition]:
        """Get current decomposition."""
        return self._current


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_uncertainty_decomposition() -> bool:
    """Test basic uncertainty decomposition."""
    print("  test_uncertainty_decomposition...")

    decomposer = UncertaintyDecomposer()

    result = decomposer.decompose(
        measurement_noise=3.0,
        sqi=90.0,
        p3_spatial_quality=0.85,
        p3_motion_confidence=0.9,
        temporal_stability=0.8,
        spectral_snr_db=10.0,
        model_variance=4.0,
        evidence_coverage=0.9,
        mean_bpm=72.0,
        n_frames=100
    )

    assert result.validity in ("VALID", "UNCERTAIN")
    assert result.total_uncertainty > 0
    assert result.measurement is not None

    print(f"    total: {result.total_uncertainty:.2f} BPM")
    print(f"    validity: {result.validity}")
    print(f"    PASS")
    return True


def test_validity_determination() -> bool:
    """Test validity determination."""
    print("  test_validity_determination...")

    decomposer = UncertaintyDecomposer()

    # Low coverage should be INVALID
    result = decomposer.decompose(
        measurement_noise=3.0,
        sqi=90.0,
        p3_spatial_quality=0.85,
        p3_motion_confidence=0.9,
        temporal_stability=0.8,
        spectral_snr_db=10.0,
        model_variance=4.0,
        evidence_coverage=0.1,  # Very low coverage
        mean_bpm=72.0,
        n_frames=100
    )

    assert result.validity == "INVALID", f"Low coverage should be INVALID, got {result.validity}"

    # High quality should be VALID
    result = decomposer.decompose(
        measurement_noise=1.0,
        sqi=95.0,
        p3_spatial_quality=0.95,
        p3_motion_confidence=0.95,
        temporal_stability=0.95,
        spectral_snr_db=15.0,
        model_variance=1.0,
        evidence_coverage=0.95,
        mean_bpm=72.0,
        n_frames=100
    )

    assert result.validity == "VALID", f"High quality should be VALID, got {result.validity}"

    print(f"    PASS")
    return True


def test_uncertainty_bpm_conversion() -> bool:
    """Test BPM conversion."""
    print("  test_uncertainty_bpm_conversion...")

    decomposer = UncertaintyDecomposer()
    result = decomposer.decompose(
        measurement_noise=5.0,
        sqi=80.0,
        p3_spatial_quality=0.7,
        p3_motion_confidence=0.7,
        temporal_stability=0.7,
        spectral_snr_db=5.0,
        model_variance=9.0,
        evidence_coverage=0.7,
        mean_bpm=70.0,
        n_frames=50
    )

    # Check that components have reasonable BPM values
    assert result.get_uncertainty_bpm(UncertaintySource.MEASUREMENT) > 0
    assert result.get_uncertainty_bpm(UncertaintySource.SPATIAL) > 0
    assert result.get_uncertainty_bpm(UncertaintySource.MOTION) > 0

    print(f"    PASS")
    return True


def test_uncertainty_sources() -> bool:
    """Test that all uncertainty sources are present."""
    print("  test_uncertainty_sources...")

    decomposer = UncertaintyDecomposer()
    result = decomposer.decompose(
        measurement_noise=3.0,
        sqi=85.0,
        p3_spatial_quality=0.8,
        p3_motion_confidence=0.8,
        temporal_stability=0.8,
        spectral_snr_db=8.0,
        model_variance=4.0,
        evidence_coverage=0.8,
        mean_bpm=75.0,
        n_frames=60
    )

    for source in UncertaintySource:
        comp = result.get_component(source)
        assert comp is not None, f"Missing component: {source}"

    print(f"    all {len(UncertaintySource)} sources present")
    print(f"    PASS")
    return True


def test_aleatoric_vs_epistemic() -> bool:
    """Test aleatoric/epistemic distinction."""
    print("  test_aleatoric_vs_epistemic...")

    decomposer = UncertaintyDecomposer()
    result = decomposer.decompose(
        measurement_noise=5.0,
        sqi=70.0,
        p3_spatial_quality=0.6,
        p3_motion_confidence=0.6,
        temporal_stability=0.6,
        spectral_snr_db=3.0,
        model_variance=16.0,
        evidence_coverage=0.6,
        mean_bpm=72.0,
        n_frames=50
    )

    # Both types should be present
    assert result.aleatoric_total > 0, "Should have aleatoric uncertainty"
    assert result.epistemic_total > 0, "Should have epistemic uncertainty"

    # Epistemic should be larger for poor quality
    assert result.epistemic_total >= result.aleatoric_total, \
        "Poor quality should have more epistemic than aleatoric"

    print(f"    aleatoric: {result.aleatoric_total:.2f} BPM")
    print(f"    epistemic: {result.epistemic_total:.2f} BPM")
    print(f"    PASS")
    return True


def test_high_quality_low_uncertainty() -> bool:
    """Test that high quality gives low uncertainty."""
    print("  test_high_quality_low_uncertainty...")

    decomposer = UncertaintyDecomposer()

    # High quality
    high_q = decomposer.decompose(
        measurement_noise=1.0,
        sqi=98.0,
        p3_spatial_quality=0.98,
        p3_motion_confidence=0.98,
        temporal_stability=0.98,
        spectral_snr_db=18.0,
        model_variance=0.5,
        evidence_coverage=0.98,
        mean_bpm=72.0,
        n_frames=200
    )

    # Low quality
    low_q = decomposer.decompose(
        measurement_noise=8.0,
        sqi=30.0,
        p3_spatial_quality=0.3,
        p3_motion_confidence=0.3,
        temporal_stability=0.3,
        spectral_snr_db=-5.0,
        model_variance=25.0,
        evidence_coverage=0.3,
        mean_bpm=72.0,
        n_frames=10
    )

    assert high_q.total_uncertainty < low_q.total_uncertainty, \
        "High quality should have lower uncertainty"

    print(f"    high quality: {high_q.total_uncertainty:.2f} BPM")
    print(f"    low quality: {low_q.total_uncertainty:.2f} BPM")
    print(f"    PASS")
    return True


def test_coverage_invalid() -> bool:
    """Test that insufficient coverage results in INVALID."""
    print("  test_coverage_invalid...")

    decomposer = UncertaintyDecomposer()

    result = decomposer.decompose(
        measurement_noise=2.0,
        sqi=95.0,
        p3_spatial_quality=0.9,
        p3_motion_confidence=0.9,
        temporal_stability=0.9,
        spectral_snr_db=12.0,
        model_variance=2.0,
        evidence_coverage=0.1,  # Critical: too little evidence
        mean_bpm=72.0,
        n_frames=5
    )

    assert result.validity == "INVALID", \
        f"Low coverage should be INVALID, got {result.validity}"

    print(f"    validity: {result.validity}")
    print(f"    PASS")
    return True


def run_tests() -> bool:
    """Run all P4.2 tests."""
    print("\n" + "=" * 60)
    print("P4.2: Uncertainty Decomposition Tests")
    print("=" * 60)

    tests = [
        ("uncertainty_decomposition", test_uncertainty_decomposition),
        ("validity_determination", test_validity_determination),
        ("uncertainty_bpm_conversion", test_uncertainty_bpm_conversion),
        ("uncertainty_sources", test_uncertainty_sources),
        ("aleatoric_vs_epistemic", test_aleatoric_vs_epistemic),
        ("high_quality_low_uncertainty", test_high_quality_low_uncertainty),
        ("coverage_invalid", test_coverage_invalid),
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
    print(f"P4.2 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
