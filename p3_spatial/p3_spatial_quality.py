"""
P3.1: Spatial Quality Map

Estimates local cardiac-band signal quality across spatial regions.

This module operates independently of V2 core.
Output maps back to image coordinates.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from scipy.signal import windows


# Sanubari cardiac band
CARDIAC_BAND_HZ = (0.833, 3.0)
SpatialRepresentation = str  # "pixel" | "patch" | "cell"
DEFAULT_CELL_SIZE = 16  # pixels


@dataclass
class SpatialQualityResult:
    """Result of spatial quality analysis."""
    quality_map: np.ndarray
    stats: Optional[Dict[str, Any] = None
    representation: str = "cell"
    input_shape: Tuple[int, int] = (0, 0)
    valid: bool = True
    error: Optional[str] = None


def compute_cardiac_snr(signal: np.ndarray, fps: float,
                       band: Tuple[float, float] = CARDIAC_BAND_HZ) -> Tuple[float, float, float]:
    """Compute SNR in cardiac band. Returns (snr_linear, peak_hz, snr_db)."""
    n = len(signal)
    if n < 30:
        return 0.0, 0.0, 0.0

    signal_detrended = signal - np.mean(signal)
    win = windows.hann(n)
    fft_vals = np.abs(np.fft.rfft(signal_detrended * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return 0.0, 0.0, 0.0

    cardiac_pow = np.sum(fft_vals[band_mask] ** 2)
    total_pow = np.sum(fft_vals ** 2)
    if total_pow < 1e-12:
        return 0.0, 0.0, 0.0

    non_cardiac_mask = ~band_mask
    non_cardiac_pow = np.sum(fft_vals[non_cardiac_mask] ** 2)
    snr_linear = cardiac_pow / max(non_cardiac_pow, 1e-12)
    snr_db = 10 * np.log10(snr_linear + 1e-12)

    cardiac_freqs = freqs[band_mask]
    cardiac_vals = fft_vals[band_mask]
    peak_idx = np.argmax(cardiac_vals)
    peak_hz = cardiac_freqs[peak_idx] if len(cardiac_vals) > 0 else 0.0

    return snr_linear, peak_hz, snr_db


class SpatialQualityMap:
    """Per-region cardiac-band quality estimation."""

    def __init__(self, cell_size: int = DEFAULT_CELL_SIZE,
                 cardiac_band: Tuple[float, float] = CARDIAC_BAND_HZ):
        if cell_size < 2:
            raise ValueError("cell_size must be >= 2")
        self.cell_size = cell_size
        self.cardiac_band = cardiac_band
        self._history = []
        self._frame_count = 0

    def compute(self, buffer: np.ndarray, fps: float,
               mask: Optional[np.ndarray] = None) -> SpatialQualityResult:
        """Compute quality map from (T, H, W) video buffer."""
        self._frame_count += 1

        if buffer is None or buffer.size == 0:
            return SpatialQualityResult(
                quality_map=np.array([[]]),
                valid=False, error="Empty buffer")

        if len(buffer.shape) < 3:
            return SpatialQualityResult(
                quality_map=np.array([[]]),
                valid=False,
                error=f"Expected 3D buffer, got shape {buffer.shape}")

        T, H, W = buffer.shape[:3]
        if T < 30:
            return SpatialQualityResult(
                quality_map=np.zeros((H // self.cell_size, W // self.cell_size)),
                valid=False,
                error=f"Insufficient samples: {T} < 30",
                input_shape=(H, W))

        # Per-cell SNR
        cs = self.cell_size
        nH, nW = H // cs, W // cs
        quality = np.zeros((nH * cs, nW * cs), dtype=np.float32)

        for i in range(nH):
            for j in range(nW):
                cell = buffer[:, i*cs:(i+1)*cs, j*cs:(j+1)*cs]
                sig = cell.mean(axis=(0).mean(axis=0) if cell.ndim == 3 else cell.mean(axis=0)
                _, _, snr_db = compute_cardiac_snr(sig, fps, self.cardiac_band)
                quality[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = snr_db

        if mask is not None:
            quality = quality * mask

        stats = {
            "mean_quality": float(np.mean(quality)) if np.any(quality != 0) else 0.0,
            "max_quality": float(np.max(quality)),
            "min_quality": float(np.min(quality)),
            "frame_count": T,
            "fps": fps,
            "cardiac_band_hz": self.cardiac_band,
        }

        return SpatialQualityResult(
            quality_map=quality,
            stats=stats,
            representation="cell",
            input_shape=(H, W),
            valid=True)

    def temporal_average(self) -> Optional[np.ndarray]:
        if not self._history:
            return None
        return np.mean(self._history, axis=0)

    def reset(self):
        self._history.clear()
        self._frame_count = 0


def test_uniform_quality() -> bool:
    """Test uniform cardiac signal produces consistent quality."""
    print("  Uniform quality...")
    fps, T, h, w = 30.0, 300, 64, 64
    t = np.arange(T) / fps
    freq_hz = 72.0 / 60.0
    sig = 100 + 20 * np.sin(2 * np.pi * freq_hz * t)
    buffer = np.zeros((T, h, w), dtype=np.float32)
    for i in range(T):
        buffer[i] = sig[i]
    result = SpatialQualityMap(cell_size=16).compute(buffer, fps)
    assert result.valid, f"Result invalid: {result.error}"
    assert result.quality_map.shape == (h, w)
    assert np.all(np.isfinite(result.quality_map))
    print(f"    Quality map: {result.quality_map.shape}")
    print(f"    Mean: {result.stats['mean_quality']:.2f} dB")
    return True


def test_high_vs_low() -> bool:
    """Test regions with different cardiac content."""
    print("  High vs low quality...")
    fps, T, h, w = 30.0, 300, 32, 32
    t = np.arange(T) / fps
    freq_hz = 72.0 / 60.0
    buffer = np.zeros((T, h, w), dtype=np.float32)
    for i in range(T):
        buffer[i, :h//2, :] = 100 + 20 * np.sin(2 * np.pi * freq_hz * t[i])
        buffer[i, h//2:, :] = 100 + 2 * np.random.randn(h//2, w)
    result = SpatialQualityMap(cell_size=8).compute(buffer, fps)
    top_q = float(np.mean(result.quality_map[:h//2]))
    bot_q = float(np.mean(result.quality_map[h//2:]))
    print(f"    High-quality: {top_q:.2f} dB")
    print(f"    Low-quality: {bot_q:.2f} dB")
    assert top_q > bot_q, "High-quality should exceed low-quality"
    return True


def test_invalid_inputs() -> bool:
    """Test invalid inputs are handled."""
    print("  Invalid inputs...")
    fps = 30.0
    # Empty buffer
    r = SpatialQualityMap().compute(np.array([]).reshape(0, 10, 10), fps)
    assert not r.valid, "Empty buffer should be invalid"
    # Insufficient samples
    r = SpatialQualityMap().compute(np.zeros((10, 32, 32), fps)
    assert not r.valid, "Insufficient samples should be invalid"
    print("    Invalid inputs handled correctly")
    return True


def test_deterministic() -> bool:
    """Test repeated computation produces identical output."""
    print("  Deterministic...")
    fps, T, h, w = 30.0, 120, 16, 16
    np.random.seed(42)
    sig = np.random.randn(T, h, w).astype(np.float32)
    r1 = SpatialQualityMap().compute(sig, fps)
    np.random.seed(42)
    r2 = SpatialQualityMap().compute(sig.copy(), fps)
    np.testing.assert_array_equal(r1.quality_map, r2.quality_map)
    print("    Deterministic: PASS")
    return True


def run_tests() -> bool:
    """Run all P3.1 tests."""
    print("\nP3.1: Spatial Quality Map")
    print("-" * 40)
    tests = [
        ("Uniform quality", test_uniform_quality),
        ("High vs low quality", test_high_vs_low),
        ("Invalid inputs", test_invalid_inputs),
        ("Deterministic", test_deterministic),
    ]
    passed = 0
    for name, fn in tests:
        try:
            if fn():
                passed += 1
                print(f"  {name}: PASS")
        except AssertionError as e:
            print(f"  {name}: FAIL: {e}")
        except Exception as e:
            print(f"  {name}: ERROR: {e}")
    print(f"\nP3.1: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    run_tests()
