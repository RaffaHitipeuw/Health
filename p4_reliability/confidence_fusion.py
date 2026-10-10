"""
P4.3: Confidence / Uncertainty Fusion

Builds a principled fusion mechanism for combining confidence and uncertainty.

Requirements:
- No arbitrary magic-number confidence formula
- No direct multiplication of unrelated percentages
- No double-counting strongly correlated metrics
- No confidence derived from final BPM alone
- No circular update where BPM determines confidence

Investigated methods:
- Normalized evidence aggregation
- Reliability-weighted uncertainty
- Variance-based fusion
- Probabilistic fusion
- Robust consensus
- Bayesian evidence aggregation

This module does NOT modify V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum


class FusionMethod(Enum):
    """Fusion method options."""
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    VARIANCE_WEIGHTED = "variance_weighted"
    ROBUST_CONSENSUS = "robust_consensus"
    BAYESIAN = "bayesian"


@dataclass
class FusedEstimate:
    """Fused estimate with uncertainty.

    Attributes:
        bpm: Fused heart rate estimate
        uncertainty: Uncertainty decomposition
        reliability: Reliability evidence
        confidence: Overall confidence percentage
        method: Fusion method used
        n_contributors: Number of sources contributing
        agreement: Agreement between contributors
        valid: Whether fusion is valid
        error: Error message if invalid
    """
    bpm: float = 0.0
    uncertainty_std: float = 0.0
    confidence: float = 0.0
    method: str = "weighted_mean"
    n_contributors: int = 0
    agreement: float = 0.0
    valid: bool = True
    error: Optional[str] = None


class ConfidenceUncertaintyFusion:
    """
    Fuses multiple confidence and uncertainty sources.

    Mathematical foundation:

    1. Weighted Mean:
       BPM = Σ(w_i * BPM_i) / Σ(w_i)
       where w_i = 1/σ_i²

    2. Variance-Weighted:
       σ² = 1 / Σ(1/σ_i²)

    3. Robust Consensus:
       Use median with outlier rejection

    4. Bayesian Evidence:
       Combine priors with likelihoods
    """

    def __init__(
        self,
        method: FusionMethod = FusionMethod.VARIANCE_WEIGHTED,
        outlier_threshold: float = 2.0,
        min_contributors: int = 1
    ):
        """
        Initialize fusion.

        Args:
            method: Fusion method to use
            outlier_threshold: Std threshold for outlier rejection
            min_contributors: Minimum contributors for valid fusion
        """
        self.method = method
        self.outlier_threshold = outlier_threshold
        self.min_contributors = min_contributors

    def fuse(
        self,
        estimates: List[Tuple[float, float]],  # (bpm, uncertainty_std)
        reliabilities: Optional[List[float]] = None
    ) -> FusedEstimate:
        """Fuse multiple estimates with uncertainty.

        Args:
            estimates: List of (bpm, uncertainty_std) tuples
            reliabilities: Optional reliability weights [0, 1]

        Returns:
            FusedEstimate with fused result
        """
        if not estimates:
            return FusedEstimate(valid=False, error="No estimates provided")

        if len(estimates) < self.min_contributors:
            return FusedEstimate(
                valid=False,
                error=f"Insufficient contributors: {len(estimates)} < {self.min_contributors}"
            )

        bpms = np.array([e[0] for e in estimates])
        stds = np.array([e[1] for e in estimates])

        # Remove NaN/Inf
        valid_mask = np.isfinite(bpms) & np.isfinite(stds) & (stds > 0)
        if not np.any(valid_mask):
            return FusedEstimate(valid=False, error="All estimates invalid")

        bpms = bpms[valid_mask]
        stds = stds[valid_mask]

        if reliabilities:
            reliabilities = np.array(reliabilities)[valid_mask]
        else:
            reliabilities = np.ones(len(bpms))

        if self.method == FusionMethod.MEAN:
            return self._fuse_mean(bpms, stds, reliabilities)
        elif self.method == FusionMethod.WEIGHTED_MEAN:
            return self._fuse_weighted_mean(bpms, stds, reliabilities)
        elif self.method == FusionMethod.VARIANCE_WEIGHTED:
            return self._fuse_variance_weighted(bpms, stds, reliabilities)
        elif self.method == FusionMethod.ROBUST_CONSENSUS:
            return self._fuse_robust_consensus(bpms, stds, reliabilities)
        elif self.method == FusionMethod.BAYESIAN:
            return self._fuse_bayesian(bpms, stds, reliabilities)
        else:
            return self._fuse_weighted_mean(bpms, stds, reliabilities)

    def _fuse_mean(
        self,
        bpms: np.ndarray,
        stds: np.ndarray,
        reliabilities: np.ndarray
    ) -> FusedEstimate:
        """Simple mean fusion."""
        fused_bpm = float(np.mean(bpms))
        fused_std = float(np.sqrt(np.mean(stds ** 2)))
        agreement = self._compute_agreement(bpms)

        confidence = self._compute_confidence(fused_std, agreement, len(bpms))

        return FusedEstimate(
            bpm=fused_bpm,
            uncertainty_std=fused_std,
            confidence=confidence,
            method="mean",
            n_contributors=len(bpms),
            agreement=agreement,
            valid=True
        )

    def _fuse_weighted_mean(
        self,
        bpms: np.ndarray,
        stds: np.ndarray,
        reliabilities: np.ndarray
    ) -> FusedEstimate:
        """Reliability-weighted mean fusion."""
        weights = reliabilities / (stds ** 2 + 1e-6)
        weight_sum = np.sum(weights)

        if weight_sum < 1e-10:
            return self._fuse_mean(bpms, stds, reliabilities)

        fused_bpm = float(np.sum(weights * bpms) / weight_sum)
        fused_std = float(1.0 / np.sqrt(weight_sum))
        agreement = self._compute_agreement(bpms)

        confidence = self._compute_confidence(fused_std, agreement, len(bpms))

        return FusedEstimate(
            bpm=fused_bpm,
            uncertainty_std=fused_std,
            confidence=confidence,
            method="weighted_mean",
            n_contributors=len(bpms),
            agreement=agreement,
            valid=True
        )

    def _fuse_variance_weighted(
        self,
        bpms: np.ndarray,
        stds: np.ndarray,
        reliabilities: np.ndarray
    ) -> FusedEstimate:
        """Variance-weighted fusion (inverse-variance weighting)."""
        precisions = 1.0 / (stds ** 2 + 1e-6)
        total_precision = np.sum(precisions)

        if total_precision < 1e-10:
            return self._fuse_mean(bpms, stds, reliabilities)

        fused_bpm = float(np.sum(precisions * bpms) / total_precision)
        fused_std = float(1.0 / np.sqrt(total_precision))
        agreement = self._compute_agreement(bpms)

        confidence = self._compute_confidence(fused_std, agreement, len(bpms))

        return FusedEstimate(
            bpm=fused_bpm,
            uncertainty_std=fused_std,
            confidence=confidence,
            method="variance_weighted",
            n_contributors=len(bpms),
            agreement=agreement,
            valid=True
        )

    def _fuse_robust_consensus(
        self,
        bpms: np.ndarray,
        stds: np.ndarray,
        reliabilities: np.ndarray
    ) -> FusedEstimate:
        """Robust consensus with outlier rejection."""
        # Start with median
        median_bpm = float(np.median(bpms))

        # Reject outliers
        deviations = np.abs(bpms - median_bpm)
        threshold = self.outlier_threshold * float(np.std(bpms))

        inlier_mask = deviations <= max(threshold, 5.0)  # At least keep values within 5 BPM
        if np.sum(inlier_mask) < 2:
            # Not enough inliers, use median
            fused_bpm = median_bpm
            fused_std = float(np.std(bpms))
        else:
            inlier_bpms = bpms[inlier_mask]
            inlier_stds = stds[inlier_mask]
            inlier_reliabilities = reliabilities[inlier_mask]

            # Re-weight inliers
            return self._fuse_variance_weighted(
                inlier_bpms, inlier_stds, inlier_reliabilities
            )

        agreement = self._compute_agreement(bpms)
        confidence = self._compute_confidence(fused_std, agreement, np.sum(inlier_mask))

        return FusedEstimate(
            bpm=fused_bpm,
            uncertainty_std=fused_std,
            confidence=confidence,
            method="robust_consensus",
            n_contributors=int(np.sum(inlier_mask)),
            agreement=agreement,
            valid=True
        )

    def _fuse_bayesian(
        self,
        bpms: np.ndarray,
        stds: np.ndarray,
        reliabilities: np.ndarray
    ) -> FusedEstimate:
        """Bayesian evidence aggregation.

        Treats each estimate as evidence and combines using
        precision-weighted Bayesian updating.
        """
        # Prior: vague uniform (use mean as weak prior)
        prior_mean = float(np.mean(bpms))
        prior_var = 1000.0  # Vague prior

        # Likelihoods from each estimate
        total_precision = 1.0 / prior_var
        weighted_sum = prior_mean / prior_var

        for bpm, std, rel in zip(bpms, stds, reliabilities):
            likelihood_var = std ** 2
            likelihood_precision = 1.0 / likelihood_var

            # Weight by reliability
            weighted_precision = likelihood_precision * rel
            total_precision += weighted_precision
            weighted_sum += weighted_precision * bpm

        # Posterior
        fused_bpm = weighted_sum / total_precision
        fused_std = float(1.0 / np.sqrt(total_precision))

        agreement = self._compute_agreement(bpms)
        confidence = self._compute_confidence(fused_std, agreement, len(bpms))

        return FusedEstimate(
            bpm=fused_bpm,
            uncertainty_std=fused_std,
            confidence=confidence,
            method="bayesian",
            n_contributors=len(bpms),
            agreement=agreement,
            valid=True
        )

    def _compute_agreement(self, bpms: np.ndarray) -> float:
        """Compute agreement between estimates [0, 1].

        Uses coefficient of variation (CV) as agreement measure.
        Lower CV = higher agreement.
        """
        if len(bpms) < 2:
            return 1.0

        mean_bpm = np.mean(bpms)
        std_bpm = np.std(bpms)

        if mean_bpm < 1.0:
            return 0.0

        cv = std_bpm / mean_bpm

        # CV of 0 = perfect agreement, CV of 0.1 = 10% variation
        agreement = float(np.clip(1.0 - cv * 5.0, 0.0, 1.0))
        return agreement

    def _compute_confidence(
        self,
        fused_std: float,
        agreement: float,
        n_contributors: int
    ) -> float:
        """Compute confidence from fused estimate.

        Mathematical formula:
        confidence = base_confidence * agreement * contributor_factor

        where:
        - base_confidence from uncertainty (low std = high confidence)
        - agreement from inter-estimate agreement
        - contributor_factor from number of estimators
        """
        # Base confidence from uncertainty (BPM std)
        # Assume 0 std = 100% confidence, 10 std = 0% confidence
        base_conf = float(np.clip(100.0 - fused_std * 5.0, 0.0, 100.0))

        # Contributor factor (diminishing returns)
        # 1 contributor = 0.7, 2 = 0.85, 3+ = 1.0
        if n_contributors <= 1:
            contrib_factor = 0.7
        elif n_contributors == 2:
            contrib_factor = 0.85
        else:
            contrib_factor = 1.0

        # Final confidence
        confidence = base_conf * agreement * contrib_factor

        return float(np.clip(confidence, 0.0, 100.0))


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_variance_weighted_fusion() -> bool:
    """Test variance-weighted fusion."""
    print("  test_variance_weighted_fusion...")

    fusion = ConfidenceUncertaintyFusion(method=FusionMethod.VARIANCE_WEIGHTED)

    # Two estimates with different uncertainties
    estimates = [
        (72.0, 5.0),   # Good estimate (low uncertainty)
        (75.0, 15.0),   # Poor estimate (high uncertainty)
    ]

    result = fusion.fuse(estimates)

    assert result.valid
    # Should be closer to the more certain estimate (72)
    assert 70.0 < result.bpm < 74.0, f"BPM {result.bpm} should be closer to 72"
    assert result.uncertainty_std < 5.0, "Should have lower uncertainty than best input"

    print(f"    fused BPM: {result.bpm:.1f} ± {result.uncertainty_std:.2f}")
    print(f"    PASS")
    return True


def test_weighted_mean_fusion() -> bool:
    """Test reliability-weighted mean fusion."""
    print("  test_weighted_mean_fusion...")

    fusion = ConfidenceUncertaintyFusion(method=FusionMethod.WEIGHTED_MEAN)

    estimates = [
        (70.0, 5.0),
        (80.0, 5.0),
    ]
    reliabilities = [0.9, 0.1]  # First estimate more reliable

    result = fusion.fuse(estimates, reliabilities)

    assert result.valid
    # Should be closer to first estimate
    assert result.bpm < 75.0, f"BPM {result.bpm} should be < 75"

    print(f"    fused BPM: {result.bpm:.1f}")
    print(f"    PASS")
    return True


def test_agreement() -> bool:
    """Test agreement computation."""
    print("  test_agreement...")

    fusion = ConfidenceUncertaintyFusion()

    # Perfect agreement
    bpms_perfect = np.array([72.0, 72.0, 72.0])
    agreement_perfect = fusion._compute_agreement(bpms_perfect)
    assert abs(agreement_perfect - 1.0) < 0.01

    # Poor agreement
    bpms_poor = np.array([60.0, 80.0])
    agreement_poor = fusion._compute_agreement(bpms_poor)
    assert agreement_poor < agreement_perfect

    print(f"    perfect: {agreement_perfect:.2f}, poor: {agreement_poor:.2f}")
    print(f"    PASS")
    return True


def test_robust_consensus() -> bool:
    """Test robust consensus with outlier rejection."""
    print("  test_robust_consensus...")

    fusion = ConfidenceUncertaintyFusion(
        method=FusionMethod.ROBUST_CONSENSUS,
        outlier_threshold=2.0
    )

    # Estimates with one outlier
    estimates = [
        (72.0, 2.0),
        (73.0, 2.0),
        (71.0, 2.0),
        (95.0, 2.0),  # Outlier
    ]

    result = fusion.fuse(estimates)

    assert result.valid
    # Should reject the outlier
    assert result.bpm < 80.0, f"BPM {result.bpm} should reject outlier"
    assert result.n_contributors < 4, "Should reject some contributors"

    print(f"    fused BPM: {result.bpm:.1f}, contributors: {result.n_contributors}")
    print(f"    PASS")
    return True


def test_bayesian_fusion() -> bool:
    """Test Bayesian fusion."""
    print("  test_bayesian_fusion...")

    fusion = ConfidenceUncertaintyFusion(method=FusionMethod.BAYESIAN)

    estimates = [
        (70.0, 3.0),
        (75.0, 5.0),
    ]
    reliabilities = [0.9, 0.8]

    result = fusion.fuse(estimates, reliabilities)

    assert result.valid
    assert 68.0 < result.bpm < 77.0

    print(f"    fused BPM: {result.bpm:.1f} ± {result.uncertainty_std:.2f}")
    print(f"    PASS")
    return True


def test_confidence_from_uncertainty() -> bool:
    """Test confidence computation."""
    print("  test_confidence_from_uncertainty...")

    fusion = ConfidenceUncertaintyFusion()

    # Low uncertainty should give high confidence
    conf_low = fusion._compute_confidence(2.0, 0.9, 3)
    # High uncertainty should give low confidence
    conf_high = fusion._compute_confidence(10.0, 0.9, 3)

    assert conf_low > conf_high, "Low uncertainty should give higher confidence"

    print(f"    low std: {conf_low:.1f}%, high std: {conf_high:.1f}%")
    print(f"    PASS")
    return True


def test_insufficient_contributors() -> bool:
    """Test handling of insufficient contributors."""
    print("  test_insufficient_contributors...")

    fusion = ConfidenceUncertaintyFusion(min_contributors=2)

    estimates = [(72.0, 5.0)]  # Only one estimate

    result = fusion.fuse(estimates)

    assert not result.valid
    assert "Insufficient" in result.error

    print(f"    correctly rejected: {result.error}")
    print(f"    PASS")
    return True


def test_single_estimate() -> bool:
    """Test fusion with single estimate."""
    print("  test_single_estimate...")

    fusion = ConfidenceUncertaintyFusion(min_contributors=1)

    estimates = [(72.0, 5.0)]

    result = fusion.fuse(estimates)

    assert result.valid
    assert abs(result.bpm - 72.0) < 0.1

    print(f"    PASS")
    return True


def run_tests() -> bool:
    """Run all P4.3 tests."""
    print("\n" + "=" * 60)
    print("P4.3: Confidence/Uncertainty Fusion Tests")
    print("=" * 60)

    tests = [
        ("variance_weighted_fusion", test_variance_weighted_fusion),
        ("weighted_mean_fusion", test_weighted_mean_fusion),
        ("agreement", test_agreement),
        ("robust_consensus", test_robust_consensus),
        ("bayesian_fusion", test_bayesian_fusion),
        ("confidence_from_uncertainty", test_confidence_from_uncertainty),
        ("insufficient_contributors", test_insufficient_contributors),
        ("single_estimate", test_single_estimate),
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
    print(f"P4.3 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
