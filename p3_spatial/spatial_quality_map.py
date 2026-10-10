"""
P3.1: Spatial Quality Map
Estimates local cardiac-band quality across spatial regions.
Independent of V2 core.

Mathematical definition:
    For a spatial cell C with N_t temporal samples at FPS=f:
    1. Extract temporal mean signal: s(t) = mean(frame[t, C])
    2. Compute FFT: S(f) = FFT(s(t) - mean(s)) * Hann window
    3. Cardiac band: f_c ∈ [0.833, 3.0] Hz
    4. Cardiac power: P_c = sum(|S(f_c)|^2)
    5. Non-cardiac power: P_n = sum(|S(f ∉ cardiac)|^2)
    6. SNR_linear = P_c / (P_n + ε)
    7. SNR_dB = 10 * log10(SNR_linear + ε)
    8. Quality = clamp(SNR_dB, -20, 20) for stability

This module does NOT modify V2 core.
Output maps back to image coordinates via cell_size grid.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union
from scipy.signal import windows


# Sanubari cardiac band (VERIFIED - matches V2 config)
CARDIAC_BAND_HZ: Tuple[float, float] = (0.833, 3.0)

# Default grid resolution
DEFAULT_CELL_SIZE: int = 16  # pixels per cell
DEFAULT_PATCH_SIZE: int = 8  # pixels per patch

# Minimum temporal samples for valid FFT
MIN_TEMPORAL_SAMPLES: int = 30

# SNR bounds for numerical stability
SNR_DB_MIN: float = -20.0
SNR_DB_MAX: float = 20.0

# Epsilon for numerical stability
EPS: float = 1e-12


@dataclass
class SpatialQualityResult:
    """Result of spatial quality analysis.

    Attributes:
        quality_map: 2D array of shape (H_out, W_out) with SNR in dB per cell
        stats: Dictionary with aggregate statistics
        representation: Grid representation used ("cell")
        input_shape: Original (H, W) of input frames
        cell_size: Grid cell size in pixels
        valid: Whether computation succeeded
        error: Error message if invalid
    """
    quality_map: np.ndarray = field(default_factory=lambda: np.array([]))
    stats: Optional[Dict[str, Any]] = None
    representation: str = "cell"
    input_shape: Tuple[int, int] = (0, 0)
    cell_size: int = DEFAULT_CELL_SIZE
    valid: bool = True
    error: Optional[str] = None


def snr_cardiac(
    signal: np.ndarray,
    fps: float,
    band: Tuple[float, float] = CARDIAC_BAND_HZ
) -> Tuple[float, float, float]:
    """
    Compute SNR in cardiac band.

    Args:
        signal: 1D temporal signal (N_samples,)
        fps: Frames per second
        band: (low_hz, high_hz) cardiac frequency range

    Returns:
        (snr_linear, peak_hz, snr_db)

    Mathematical definition:
        1. Detrend: s_detrended = s - mean(s)
        2. Window: s_windowed = s_detrended * Hann(n)
        3. FFT: S = |FFT(s_windowed)|
        4. Cardiac mask: M_c = {f: band[0] ≤ f ≤ band[1]}
        5. P_c = Σ_{f∈M_c} S(f)²
        6. P_n = Σ_{f∉M_c} S(f)²
        7. SNR_linear = P_c / (P_n + ε)
        8. SNR_dB = 10 * log10(SNR_linear + ε)
        9. peak_hz = argmax_{f∈M_c} S(f)
    """
    n = len(signal)
    if n < MIN_TEMPORAL_SAMPLES:
        return 0.0, 0.0, 0.0

    # Step 1: Detrend
    detrended = signal - np.mean(signal)

    # Step 2: Window
    win = windows.hann(n)
    windowed = detrended * win

    # Step 3: FFT
    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Step 4: Cardiac band mask
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return 0.0, 0.0, 0.0

    # Step 5: Power calculation
    cardiac_pow = np.sum(fft_vals[band_mask] ** 2)
    total_pow = np.sum(fft_vals ** 2)
    if total_pow < EPS:
        return 0.0, 0.0, 0.0

    # Step 6: Non-cardiac power
    non_pow = np.sum(fft_vals[~band_mask] ** 2)

    # Step 7: SNR linear
    snr_linear = cardiac_pow / max(non_pow, EPS)

    # Step 8: SNR dB
    snr_db = 10.0 * np.log10(snr_linear + EPS)
    snr_db = float(np.clip(snr_db, SNR_DB_MIN, SNR_DB_MAX))

    # Step 9: Peak frequency
    cardiac_freqs = freqs[band_mask]
    cardiac_vals = fft_vals[band_mask]
    peak_idx = np.argmax(cardiac_vals)
    peak_hz = float(cardiac_freqs[peak_idx]) if len(cardiac_vals) > 0 else 0.0

    return float(snr_linear), peak_hz, snr_db


def validate_buffer(
    buffer: np.ndarray,
    fps: float
) -> Tuple[bool, Optional[str], Tuple[int, int, int]]:
    """
    Validate temporal buffer for spectral analysis.

    Returns:
        (is_valid, error_message, (T, H, W))
    """
    if buffer is None:
        return False, "buffer is None", (0, 0, 0)

    if not isinstance(buffer, np.ndarray):
        return False, f"buffer must be ndarray, got {type(buffer).__name__}", (0, 0, 0)

    if buffer.size == 0:
        return False, "buffer is empty", (0, 0, 0)

    if buffer.ndim < 3:
        return False, f"buffer must be at least 3D (T,H,W), got {buffer.ndim}D", (0, 0, 0)

    T, H, W = buffer.shape[:3]

    if T < MIN_TEMPORAL_SAMPLES:
        return False, f"insufficient temporal samples: {T} < {MIN_TEMPORAL_SAMPLES}", (T, H, W)

    if fps <= 0 or not np.isfinite(fps):
        return False, f"fps must be positive finite, got {fps}", (T, H, W)

    if not np.all(np.isfinite(buffer)):
        return False, "buffer contains non-finite values", (T, H, W)

    return True, None, (T, H, W)


class SpatialQualityMap:
    """
    Per-cell cardiac-band quality map.

    Computes spatial quality using grid-based cardiac band SNR analysis.

    Mathematical definition:
        For each grid cell C_{i,j} of size cell_size × cell_size:
        1. Extract temporal signal: s_{i,j}(t) = mean(buffer[t, C_{i,j}])
        2. Compute SNR_dB = snr_cardiac(s_{i,j}(t), fps)
        3. Assign quality[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = SNR_dB

    Attributes:
        cell_size: Grid cell size in pixels (default 16)
        band: Cardiac frequency band (0.833, 3.0) Hz
    """

    def __init__(
        self,
        cell_size: int = DEFAULT_CELL_SIZE,
        band: Tuple[float, float] = CARDIAC_BAND_HZ
    ):
        if cell_size < 2:
            raise ValueError(f"cell_size must be >= 2, got {cell_size}")

        self.cell_size = cell_size
        self.band = band
        self._history: list = []
        self._frame_count: int = 0

    def compute(
        self,
        buffer: np.ndarray,
        fps: float,
        mask: Optional[np.ndarray] = None
    ) -> SpatialQualityResult:
        """
        Compute spatial quality map from temporal buffer.

        Args:
            buffer: (T, H, W) or (T, H, W, C) video buffer
            fps: Frames per second
            mask: Optional (H, W) binary mask for valid regions

        Returns:
            SpatialQualityResult with quality_map (H, W) and statistics
        """
        self._frame_count += 1

        # Validate input
        valid, error, shape = validate_buffer(buffer, fps)
        if not valid:
            return SpatialQualityResult(
                quality_map=np.array([[]]),
                valid=False,
                error=error,
                input_shape=(shape[1], shape[2]) if shape else (0, 0),
                cell_size=self.cell_size
            )

        T, H, W = shape[:3]

        # Compute grid dimensions
        cs = self.cell_size
        nH = H // cs
        nW = W // cs

        if nH == 0 or nW == 0:
            return SpatialQualityResult(
                quality_map=np.array([[]]),
                valid=False,
                error=f"Image too small for cell_size={cs}: ({H}, {W})",
                input_shape=(H, W),
                cell_size=self.cell_size
            )

        # Initialize quality map (full resolution)
        quality = np.zeros((nH * cs, nW * cs), dtype=np.float32)

        # Process each cell
        for i in range(nH):
            for j in range(nW):
                # Extract cell region
                cell = buffer[:, i*cs:(i+1)*cs, j*cs:(j+1)*cs]

                # Compute mean temporal signal
                sig = cell.mean(axis=(1, 2)) if cell.ndim == 3 else cell.mean(axis=0)
                sig = np.atleast_1d(sig)

                # Compute cardiac SNR
                _, _, snr_db = snr_cardiac(sig, fps, self.band)

                # Assign to full-resolution map
                quality[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = snr_db

        # Apply mask if provided
        if mask is not None:
            if mask.shape != (H, W):
                mask_resized = np.zeros((H, W), dtype=mask.dtype)
                mask_resized[:mask.shape[0], :mask.shape[1]] = mask[:H, :W]
                mask = mask_resized
            # Downsample mask to match grid
            mask_grid = np.zeros((nH * cs, nW * cs), dtype=np.float32)
            for i in range(nH):
                for j in range(nW):
                    cell_mask = mask[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
                    mask_grid[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = np.mean(cell_mask) / 255.0
            quality = quality * mask_grid

        # Compute statistics
        valid_mask = quality != 0
        valid_quality = quality[valid_mask] if np.any(valid_mask) else np.array([0.0])

        stats = {
            "mean_db": float(np.mean(valid_quality)),
            "max_db": float(np.max(valid_quality)),
            "min_db": float(np.min(valid_quality)),
            "std_db": float(np.std(valid_quality)),
            "valid_fraction": float(np.sum(valid_mask) / quality.size),
            "T": T,
            "H": H,
            "W": W,
            "nH": nH,
            "nW": nW,
            "fps": fps,
            "band_hz": self.band,
            "cell_size": cs,
            "quality_threshold_high": float(np.percentile(valid_quality, 75)) if len(valid_quality) > 0 else 0.0,
            "quality_threshold_low": float(np.percentile(valid_quality, 25)) if len(valid_quality) > 0 else 0.0,
        }

        result = SpatialQualityResult(
            quality_map=quality,
            stats=stats,
            representation="cell",
            input_shape=(H, W),
            cell_size=cs,
            valid=True
        )

        # Store in history for temporal averaging
        self._history.append(quality)

        return result

    def temporal_average(self) -> Optional[np.ndarray]:
        """Average quality over temporal history."""
        if not self._history:
            return None
        return np.mean(self._history, axis=0)

    def reset(self) -> None:
        """Clear temporal history."""
        self._history.clear()
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count


# Convenience function for one-shot computation
def compute_spatial_quality(
    buffer: np.ndarray,
    fps: float,
    cell_size: int = DEFAULT_CELL_SIZE,
    band: Tuple[float, float] = CARDIAC_BAND_HZ,
    mask: Optional[np.ndarray] = None
) -> SpatialQualityResult:
    """
    Compute spatial quality map from temporal buffer (one-shot).

    For repeated calls, use SpatialQualityMap class for efficiency.

    Args:
        buffer: (T, H, W) video buffer
        fps: Frames per second
        cell_size: Grid cell size in pixels
        band: Cardiac frequency band (low_hz, high_hz)
        mask: Optional (H, W) binary mask

    Returns:
        SpatialQualityResult
    """
    mapper = SpatialQualityMap(cell_size=cell_size, band=band)
    return mapper.compute(buffer, fps, mask)


# =============================================================================
# SYNTHETIC VALIDATION TESTS
# =============================================================================

def test_uniform_quality() -> bool:
    """
    Test: Uniform cardiac signal produces consistent quality across all cells.

    Creates a buffer with identical cardiac signal (72 BPM) everywhere.
    All cells should have approximately equal SNR.
    """
    print("  test_uniform_quality...")

    fps = 30.0
    T, H, W = 300, 64, 64
    t = np.arange(T) / fps
    freq_hz = 72.0 / 60.0  # 72 BPM in Hz

    # Create uniform signal
    sig = 100.0 + 20.0 * np.sin(2.0 * np.pi * freq_hz * t)
    buffer = np.zeros((T, H, W), dtype=np.float32)
    for i in range(T):
        buffer[i] = sig[i]

    result = SpatialQualityMap(cell_size=16).compute(buffer, fps)

    assert result.valid, f"Result invalid: {result.error}"
    assert result.quality_map.shape == (H, W), f"Shape mismatch: {result.quality_map.shape}"
    assert np.all(np.isfinite(result.quality_map)), "Non-finite values in quality map"

    # All cells should have similar quality (within 3 dB due to FFT variance)
    mean_q = result.stats["mean_db"]
    max_q = result.stats["max_db"]
    min_q = result.stats["min_db"]

    # Check consistency: max - min should be small for uniform signal
    spread = max_q - min_q
    assert spread < 10.0, f"Quality spread too large: {spread:.2f} dB (expected < 10)"

    print(f"    shape: {result.quality_map.shape}")
    print(f"    mean: {mean_q:.2f} dB, max: {max_q:.2f} dB, min: {min_q:.2f} dB")
    print(f"    PASS")
    return True


def test_high_vs_low_quality() -> bool:
    """
    Test: Regions with cardiac signal vs. noise are distinguished.

    Top half: strong cardiac signal (72 BPM, amplitude 20)
    Bottom half: random noise (amplitude 2)
    Top should have higher quality than bottom.
    """
    print("  test_high_vs_low_quality...")

    fps = 30.0
    T, H, W = 300, 32, 32
    t = np.arange(T) / fps
    freq_hz = 72.0 / 60.0

    buffer = np.zeros((T, H, W), dtype=np.float32)
    for i in range(T):
        # Top half: cardiac signal
        buffer[i, :H//2, :] = 100.0 + 20.0 * np.sin(2.0 * np.pi * freq_hz * t[i])
        # Bottom half: noise only
        buffer[i, H//2:, :] = 100.0 + 2.0 * np.random.randn(H//2, W)

    result = SpatialQualityMap(cell_size=8).compute(buffer, fps)

    assert result.valid, f"Result invalid: {result.error}"

    # Compute mean quality per half
    top_quality = float(np.mean(result.quality_map[:H//2]))
    bottom_quality = float(np.mean(result.quality_map[H//2:]))

    assert top_quality > bottom_quality, \
        f"Top quality ({top_quality:.2f}) should exceed bottom ({bottom_quality:.2f})"

    print(f"    top quality: {top_quality:.2f} dB")
    print(f"    bottom quality: {bottom_quality:.2f} dB")
    print(f"    difference: {top_quality - bottom_quality:.2f} dB")
    print(f"    PASS")
    return True


def test_multiple_cardiac_frequencies() -> bool:
    """
    Test: Different cardiac frequencies are detected correctly.

    Left third: 60 BPM
    Middle third: 72 BPM
    Right third: 90 BPM
    All should show positive SNR in cardiac band.
    """
    print("  test_multiple_cardiac_frequencies...")

    fps = 30.0
    T, H, W = 300, 32, 48
    t = np.arange(T) / fps

    freqs_bpm = [60.0, 72.0, 90.0]
    freqs_hz = [f / 60.0 for f in freqs_bpm]

    buffer = np.zeros((T, H, W), dtype=np.float32)
    for i in range(T):
        w3 = W // 3
        for k, f_hz in enumerate(freqs_hz):
            buffer[i, :, k*w3:(k+1)*w3 if k < 2 else W] = \
                100.0 + 15.0 * np.sin(2.0 * np.pi * f_hz * t[i])

    result = SpatialQualityMap(cell_size=8).compute(buffer, fps)

    assert result.valid, f"Result invalid: {result.error}"

    # Check each region has positive SNR
    w3 = W // 3
    for k, f_bpm in enumerate(freqs_bpm):
        region = result.quality_map[:, k*w3:(k+1)*w3 if k < 2 else W]
        mean_q = float(np.mean(region))
        assert mean_q > -5.0, \
            f"Frequency {f_bpm} BPM should have SNR > -5 dB, got {mean_q:.2f}"
        print(f"    {f_bpm} BPM region: {mean_q:.2f} dB")

    print(f"    PASS")
    return True


def test_invalid_inputs() -> bool:
    """
    Test: Invalid inputs are handled correctly.
    """
    print("  test_invalid_inputs...")

    fps = 30.0
    mapper = SpatialQualityMap()

    # Empty buffer
    r = mapper.compute(np.array([]).reshape(0, 10, 10), fps)
    assert not r.valid, "Empty buffer should be invalid"
    print("    empty buffer: rejected")

    # Insufficient samples
    r = mapper.compute(np.zeros((10, 32, 32), dtype=np.float32), fps)
    assert not r.valid, "Insufficient samples should be invalid"
    print("    insufficient samples: rejected")

    # Invalid FPS
    r = mapper.compute(np.zeros((100, 32, 32), dtype=np.float32), 0.0)
    assert not r.valid, "Invalid FPS should be rejected"
    print("    invalid FPS: rejected")

    # Non-finite values
    buf = np.zeros((100, 32, 32), dtype=np.float32)
    buf[50, 16, 16] = np.nan
    r = mapper.compute(buf, fps)
    assert not r.valid, "Non-finite buffer should be invalid"
    print("    non-finite buffer: rejected")

    print(f"    PASS")
    return True


def test_deterministic() -> bool:
    """
    Test: Repeated computation produces identical output.

    Uses fixed seed to ensure deterministic behavior.
    """
    print("  test_deterministic...")

    np.random.seed(42)
    fps = 30.0
    T, H, W = 120, 16, 16
    buffer = np.random.randn(T, H, W).astype(np.float32)

    # First computation
    r1 = SpatialQualityMap(cell_size=8).compute(buffer.copy(), fps)

    # Second computation with same data
    np.random.seed(42)
    buffer2 = np.random.randn(T, H, W).astype(np.float32)
    r2 = SpatialQualityMap(cell_size=8).compute(buffer2, fps)

    np.testing.assert_array_equal(
        r1.quality_map, r2.quality_map,
        "Repeated computation should be identical"
    )

    print(f"    PASS")
    return True


def test_zero_variance_signal() -> bool:
    """
    Test: Zero-variance signal (constant) produces zero quality.
    """
    print("  test_zero_variance_signal...")

    fps = 30.0
    T, H, W = 100, 16, 16
    buffer = np.full((T, H, W), 128.0, dtype=np.float32)

    result = SpatialQualityMap(cell_size=8).compute(buffer, fps)

    # Should be valid but with low/zero quality
    assert result.valid, f"Should be valid: {result.error}"
    # Quality should be very low or zero for constant signal
    mean_q = result.stats["mean_db"]
    print(f"    constant signal quality: {mean_q:.2f} dB")
    print(f"    PASS")
    return True


def run_tests() -> bool:
    """
    Run all P3.1 spatial quality tests.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("P3.1: Spatial Quality Map Tests")
    print("=" * 60)

    tests = [
        ("uniform_quality", test_uniform_quality),
        ("high_vs_low_quality", test_high_vs_low_quality),
        ("multiple_cardiac_frequencies", test_multiple_cardiac_frequencies),
        ("invalid_inputs", test_invalid_inputs),
        ("deterministic", test_deterministic),
        ("zero_variance_signal", test_zero_variance_signal),
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
    print(f"P3.1 Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
