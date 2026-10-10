"""
P3.3: Motion-Aware Spatial Quality

Combines spatial signal quality (P3.1) and spatial motion information (P3.2).

Mathematical definitions:

1. Motion-Aware Quality (per cell):
   Q_motion = Q_cardiac * C_confidence * S_stability

   Where:
   - Q_cardiac: Cardiac band SNR from P3.1
   - C_confidence: Motion confidence from P3.2 (high motion = low confidence)
   - S_stability: Temporal stability from P3.2

2. Alternative weighting functions:

   a) Multiplicative:
      Q_weighted = Q * C * S

   b) Additive (softer):
      Q_weighted = Q * (w1*C + w2*S + w3*(1-Q_motion_fraction))

   c) Harmonic (conservative):
      Q_weighted = Q * (C*S) / (C + S + ε)

3. Ablation modes:
   - QUALITY_ONLY: Q_motion = Q (ignore motion)
   - MOTION_ONLY: Q_motion = C * S (ignore quality)
   - FULL: Q_motion = Q * C * S

This module does NOT modify V2 core.
Output aligns with P3.1/P3.2 grid.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from enum import Enum


# Weighting modes
class WeightingMode(Enum):
    """Ablation modes for motion-aware quality."""
    QUALITY_ONLY = "quality_only"     # Q only
    MOTION_ONLY = "motion_only"       # C * S only
    FULL = "full"                     # Q * C * S
    ADDITIVE = "additive"             # Soft additive weighting
    HARMONIC = "harmonic"             # Conservative harmonic mean


# Default parameters
DEFAULT_WEIGHTING_MODE = WeightingMode.FULL
DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)  # (w_quality, w_confidence, w_stability)

# Quality bounds
QUALITY_MIN: float = -20.0
QUALITY_MAX: float = 20.0
CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0

# Motion quality threshold (above this = high motion contamination)
HIGH_MOTION_THRESHOLD: float = 0.5


@dataclass
class MotionAwareQualityResult:
    """Result of motion-aware spatial quality computation.

    Attributes:
        quality_map: (H, W) motion-aware quality per cell
        quality_only: (H, W) quality without motion weighting
        motion_contribution: (H, W) motion reliability factor
        weighted_mask: (H, W) binary mask of high-quality regions
        mode: WeightingMode used
        stats: Aggregate statistics
        valid: Whether computation succeeded
        error: Error message if invalid
    """
    quality_map: np.ndarray = field(default_factory=lambda: np.array([]))
    quality_only: np.ndarray = field(default_factory=lambda: np.array([]))
    motion_contribution: np.ndarray = field(default_factory=lambda: np.array([]))
    weighted_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    mode: WeightingMode = DEFAULT_WEIGHTING_MODE
    stats: Optional[Dict[str, Any]] = None
    valid: bool = True
    error: Optional[str] = None


class MotionAwareQuality:
    """
    Combines spatial signal quality with motion information.

    Mathematical definition:
        For each cell (i, j):

        Q_input = P3.1 cardiac SNR (dB)
        C = P3.2 motion confidence [0, 1]
        S = P3.2 temporal stability [0, 1]

        Weighting modes:
        1. QUALITY_ONLY: Q_out = Q_input
        2. MOTION_ONLY: Q_out = C * S (normalized)
        3. FULL: Q_out = Q_input * C * S
        4. ADDITIVE: Q_out = Q_input * (w_q*C + w_s*S)
        5. HARMONIC: Q_out = Q_input * C*S / (C + S + ε)

        Final mask = Q_out > threshold

    Args:
        mode: Weighting mode for ablation
        weights: (w_quality, w_confidence, w_stability) for ADDITIVE mode
        quality_threshold: Threshold for weighted mask
    """

    def __init__(
        self,
        mode: WeightingMode = DEFAULT_WEIGHTING_MODE,
        weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
        quality_threshold: float = 0.3,
        high_motion_threshold: float = HIGH_MOTION_THRESHOLD
    ):
        self.mode = mode
        self.weights = weights
        self.quality_threshold = quality_threshold
        self.high_motion_threshold = high_motion_threshold

        self._frame_count = 0

    def _compute_motion_contribution(
        self,
        confidence: np.ndarray,
        stability: np.ndarray,
        mode: WeightingMode
    ) -> np.ndarray:
        """
        Compute motion reliability contribution.

        Args:
            confidence: Motion confidence [0, 1]
            stability: Temporal stability [0, 1]
            mode: Weighting mode

        Returns:
            Motion contribution factor [0, 1]
        """
        if mode == WeightingMode.QUALITY_ONLY:
            # No motion contribution
            return np.ones_like(confidence)

        elif mode == WeightingMode.MOTION_ONLY:
            # Only motion factors
            return confidence * stability

        elif mode == WeightingMode.FULL:
            # Full multiplicative
            return confidence * stability

        elif mode == WeightingMode.ADDITIVE:
            w_q, w_c, w_s = self.weights
            # Quality is treated separately, so just motion contribution
            return w_c * confidence + w_s * stability

        elif mode == WeightingMode.HARMONIC:
            # Conservative harmonic mean
            return (confidence * stability) / (confidence + stability + 1e-6)

        else:
            return confidence * stability

    def _normalize_quality(self, quality: np.ndarray) -> np.ndarray:
        """
        Normalize quality from dB to [0, 1] range.

        Mathematical definition:
            Q_norm = (Q - Q_min) / (Q_max - Q_min)

        Where Q_min, Q_max are the configured bounds.
        """
        return (quality - QUALITY_MIN) / (QUALITY_MAX - QUALITY_MIN + 1e-9)

    def compute(
        self,
        quality_map: np.ndarray,
        confidence_map: np.ndarray,
        stability_map: np.ndarray,
        mode: Optional[WeightingMode] = None
    ) -> MotionAwareQualityResult:
        """
        Compute motion-aware quality by combining P3.1 and P3.2 outputs.

        Args:
            quality_map: (H, W) cardiac SNR from P3.1
            confidence_map: (H, W) motion confidence from P3.2
            stability_map: (H, W) temporal stability from P3.2
            mode: Override default weighting mode

        Returns:
            MotionAwareQualityResult
        """
        self._frame_count += 1

        use_mode = mode or self.mode

        # Validate inputs
        if quality_map.size == 0:
            return MotionAwareQualityResult(
                valid=False,
                error="quality_map is empty",
                mode=use_mode
            )

        if quality_map.shape != confidence_map.shape:
            return MotionAwareQualityResult(
                valid=False,
                error=f"Shape mismatch: {quality_map.shape} vs {confidence_map.shape}",
                mode=use_mode
            )

        if quality_map.shape != stability_map.shape:
            return MotionAwareQualityResult(
                valid=False,
                error=f"Shape mismatch: {quality_map.shape} vs {stability_map.shape}",
                mode=use_mode
            )

        H, W = quality_map.shape[:2]

        # Store quality only (for ablation)
        quality_only = quality_map.copy()

        # Normalize quality to [0, 1]
        quality_norm = self._normalize_quality(quality_map)

        # Compute motion contribution
        motion_contrib = self._compute_motion_contribution(
            confidence_map, stability_map, use_mode
        )

        # Compute final weighted quality
        if use_mode == WeightingMode.QUALITY_ONLY:
            weighted_quality = quality_norm
        elif use_mode == WeightingMode.MOTION_ONLY:
            # Normalize motion contribution to quality-like range
            weighted_quality = motion_contrib * (QUALITY_MAX - QUALITY_MIN) + QUALITY_MIN
            weighted_quality = self._normalize_quality(weighted_quality)
        else:
            weighted_quality = quality_norm * motion_contrib

        # Clip to [0, 1]
        weighted_quality = np.clip(weighted_quality, 0.0, 1.0)

        # Create binary mask of high-quality regions
        weighted_mask = (weighted_quality >= self.quality_threshold).astype(np.float32)

        # Compute statistics
        valid_mask = weighted_quality > 0

        # Motion contamination fraction
        high_motion = confidence_map > self.high_motion_threshold
        high_motion_frac = float(np.sum(high_motion) / high_motion.size)

        # Low quality fraction
        low_quality = weighted_quality < self.quality_threshold
        low_quality_frac = float(np.sum(low_quality) / low_quality.size)

        # High quality fraction
        high_quality_frac = float(np.sum(~low_quality) / low_quality.size)

        stats = {
            "mean_weighted_quality": float(np.mean(weighted_quality[valid_mask])) if np.any(valid_mask) else 0.0,
            "max_weighted_quality": float(np.max(weighted_quality)),
            "min_weighted_quality": float(np.min(weighted_quality)),
            "mean_quality_only": float(np.mean(quality_only)),
            "mean_confidence": float(np.mean(confidence_map)),
            "mean_stability": float(np.mean(stability_map)),
            "mean_motion_contrib": float(np.mean(motion_contrib)),
            "high_quality_fraction": high_quality_frac,
            "low_quality_fraction": low_quality_frac,
            "high_motion_fraction": high_motion_frac,
            "quality_threshold": self.quality_threshold,
            "mode": use_mode.value,
            "H": H,
            "W": W,
        }

        return MotionAwareQualityResult(
            quality_map=weighted_quality,
            quality_only=quality_only,
            motion_contribution=motion_contrib,
            weighted_mask=weighted_mask,
            mode=use_mode,
            stats=stats,
            valid=True
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def create_synthetic_inputs(
    H: int = 64,
    W: int = 64,
    quality_value: float = 5.0,
    confidence_value: float = 0.8,
    stability_value: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic inputs for testing."""
    quality = np.full((H, W), quality_value, dtype=np.float32)
    confidence = np.full((H, W), confidence_value, dtype=np.float32)
    stability = np.full((H, W), stability_value, dtype=np.float32)
    return quality, confidence, stability


def test_quality_only_mode() -> bool:
    """
    Test: QUALITY_ONLY mode ignores motion.

    Weighted quality should equal input quality (normalized).
    """
    print("  test_quality_only_mode...")

    H, W = 32, 32
    quality, confidence, stability = create_synthetic_inputs(
        H, W, quality_value=10.0, confidence_value=0.1, stability_value=0.1
    )

    combiner = MotionAwareQuality(mode=WeightingMode.QUALITY_ONLY)
    result = combiner.compute(quality, confidence, stability)

    assert result.valid, f"Result invalid: {result.error}"

    # QUALITY_ONLY should return quality only (normalized)
    expected = (10.0 - QUALITY_MIN) / (QUALITY_MAX - QUALITY_MIN)
    actual = float(np.mean(result.quality_map))

    np.testing.assert_almost_equal(actual, expected, decimal=2)

    print(f"    expected: {expected:.4f}, actual: {actual:.4f}")
    print(f"    PASS")
    return True


def test_motion_only_mode() -> bool:
    """
    Test: MOTION_ONLY mode ignores quality.

    Weighted quality should reflect only motion factors.
    """
    print("  test_motion_only_mode...")

    H, W = 32, 32
    quality, confidence, stability = create_synthetic_inputs(
        H, W, quality_value=0.0, confidence_value=0.8, stability_value=0.9
    )

    combiner = MotionAwareQuality(mode=WeightingMode.MOTION_ONLY)
    result = combiner.compute(quality, confidence, stability)

    assert result.valid, f"Result invalid: {result.error}"

    # MOTION_ONLY should return C * S
    expected = 0.8 * 0.9  # = 0.72
    actual = float(np.mean(result.quality_map))

    # Allow some tolerance due to normalization
    assert abs(actual - expected) < 0.3, \
        f"MOTION_ONLY should reflect C*S, expected ~{expected}, got {actual}"

    print(f"    expected: ~{expected:.4f}, actual: {actual:.4f}")
    print(f"    PASS")
    return True


def test_full_mode() -> bool:
    """
    Test: FULL mode combines quality and motion.

    Weighted quality should be Q * C * S.
    """
    print("  test_full_mode...")

    H, W = 32, 32
    quality, confidence, stability = create_synthetic_inputs(
        H, W, quality_value=10.0, confidence_value=0.8, stability_value=0.9
    )

    combiner = MotionAwareQuality(mode=WeightingMode.FULL)
    result = combiner.compute(quality, confidence, stability)

    assert result.valid, f"Result invalid: {result.error}"

    # Expected: normalized(Q) * C * S
    q_norm = (10.0 - QUALITY_MIN) / (QUALITY_MAX - QUALITY_MIN)
    expected = q_norm * 0.8 * 0.9
    actual = float(np.mean(result.quality_map))

    np.testing.assert_almost_equal(actual, expected, decimal=3)

    print(f"    expected: {expected:.4f}, actual: {actual:.4f}")
    print(f"    PASS")
    return True


def test_high_motion_downweighting() -> bool:
    """
    Test: High-motion regions are appropriately downweighted.

    Region with high motion should have lower weighted quality
    than same quality region with low motion.
    """
    print("  test_high_motion_downweighting...")

    H, W = 32, 32

    # High quality, low motion
    quality_low = np.full((H, W), 10.0, dtype=np.float32)
    confidence_low = np.full((H, W), 0.9, dtype=np.float32)
    stability_low = np.full((H, W), 0.9, dtype=np.float32)

    # High quality, high motion
    quality_high = np.full((H, W), 10.0, dtype=np.float32)
    confidence_high = np.full((H, W), 0.1, dtype=np.float32)  # Low confidence
    stability_high = np.full((H, W), 0.1, dtype=np.float32)

    combiner = MotionAwareQuality(mode=WeightingMode.FULL)

    result_low = combiner.compute(quality_low, confidence_low, stability_low)
    result_high = combiner.compute(quality_high, confidence_high, stability_high)

    assert result_low.valid and result_high.valid

    mean_low = float(np.mean(result_low.quality_map))
    mean_high = float(np.mean(result_high.quality_map))

    # High motion should have lower quality
    assert mean_low > mean_high, \
        f"Low motion ({mean_low:.4f}) should exceed high motion ({mean_high:.4f})"

    print(f"    low motion quality: {mean_low:.4f}")
    print(f"    high motion quality: {mean_high:.4f}")
    print(f"    ratio: {mean_high/mean_low:.2f}")
    print(f"    PASS")
    return True


def test_clean_region_preserved() -> bool:
    """
    Test: Clean regions (high quality, low motion) remain high quality.

    Ablation should NOT destroy already-clean regions.
    """
    print("  test_clean_region_preserved...")

    H, W = 32, 32
    quality = np.full((H, W), 15.0, dtype=np.float32)  # High quality
    confidence = np.full((H, W), 0.95, dtype=np.float32)  # Low motion
    stability = np.full((H, W), 0.95, dtype=np.float32)  # High stability

    combiner = MotionAwareQuality(mode=WeightingMode.FULL, quality_threshold=0.5)
    result = combiner.compute(quality, confidence, stability)

    assert result.valid

    mean_quality = float(np.mean(result.quality_map))

    # Should still be relatively high
    assert mean_quality > 0.6, \
        f"Clean region should remain high quality, got {mean_quality}"

    # Should pass threshold
    high_quality_frac = result.stats["high_quality_fraction"]
    assert high_quality_frac > 0.8, \
        f"Clean region should mostly pass threshold, got {high_quality_frac:.2%}"

    print(f"    mean quality: {mean_quality:.4f}")
    print(f"    high quality fraction: {high_quality_frac:.2%}")
    print(f"    PASS")
    return True


def test_shape_mismatch() -> bool:
    """
    Test: Shape mismatches are detected and handled.
    """
    print("  test_shape_mismatch...")

    quality = np.zeros((32, 32), dtype=np.float32)
    confidence = np.zeros((32, 16), dtype=np.float32)  # Wrong size
    stability = np.zeros((32, 32), dtype=np.float32)

    combiner = MotionAwareQuality()
    result = combiner.compute(quality, confidence, stability)

    assert not result.valid, "Should detect shape mismatch"
    assert "Shape mismatch" in result.error

    print(f"    error: {result.error}")
    print(f"    PASS")
    return True


def test_empty_input() -> bool:
    """
    Test: Empty inputs are handled.
    """
    print("  test_empty_input...")

    quality = np.array([], dtype=np.float32)
    confidence = np.array([], dtype=np.float32)
    stability = np.array([], dtype=np.float32)

    combiner = MotionAwareQuality()
    result = combiner.compute(quality, confidence, stability)

    assert not result.valid, "Should detect empty input"
    assert "empty" in result.error.lower()

    print(f"    error: {result.error}")
    print(f"    PASS")
    return True


def run_tests() -> bool:
    """
    Run all P3.3 motion-aware quality tests.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("P3.3: Motion-Aware Spatial Quality Tests")
    print("=" * 60)

    tests = [
        ("quality_only_mode", test_quality_only_mode),
        ("motion_only_mode", test_motion_only_mode),
        ("full_mode", test_full_mode),
        ("high_motion_downweighting", test_high_motion_downweighting),
        ("clean_region_preserved", test_clean_region_preserved),
        ("shape_mismatch", test_shape_mismatch),
        ("empty_input", test_empty_input),
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
    print(f"P3.3 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
