"""
P3.1 Spatial Quality Map
Estimates local cardiac-band quality across spatial regions.
Independent of V2 core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from scipy.signal import windows

CARDIAC_BAND_HZ = (0.833, 3.0)
DEFAULT_CELL_SIZE = 16


@dataclass
class SpatialQualityResult:
    quality_map: np.ndarray = field(default_factory=lambda: np.array([]))
    valid: bool = True
    stats: Optional[Dict[str, Any] = None
    error: Optional[str] = None


def snr_cardiac(signal: np.ndarray, fps: float,
              band: Tuple[float, float] = CARDIAC_BAND_HZ) -> Tuple[float, float, float]:
    """SNR in cardiac band. Returns (linear, peak_hz, db)."""
    if signal is None or len(signal) < 30:
        return 0.0, 0.0, 0.0
    detrended = signal - np.mean(signal)
    win = windows.hann(len(signal))
    fft_vals = np.abs(np.fft.rfft(detrended * win))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    cardiac_pow = np.sum(fft_vals[mask] ** 2)
    total_pow = np.sum(fft_vals ** 2)
    if total_pow < 1e-12:
        return 0.0, 0.0, 0.0
    non_pow = np.sum(fft_vals[~mask] ** 2)
    linear = cardiac_pow / max(non_pow, 1e-12)
    db = 10.0 * np.log10(linear + 1e-12)
    peak_idx = np.argmax(fft_vals[mask])
    peak_hz = float(freqs[mask][peak_idx])
    return float(linear), peak_hz, float(db)


class SpatialQualityMap:
    """Per-cell cardiac-band quality map."""

    def __init__(self, cell_size=DEFAULT_CELL_SIZE, band: Tuple[float, float] = CARDIAC_BAND_HZ):
        if cell_size < 2:
            raise ValueError("cell_size must be >= 2")
        self.cell_size = cell_size
        self.band = band
        self._history = []

    def compute(self, buffer, fps):
        """buffer: (T, H, W) video."""
        if buffer is None or buffer.size == 0:
            return SpatialQualityResult(error="Empty buffer", valid=False)
        if len(buffer.shape) < 3:
            return SpatialQualityResult(error=f"Need (T,H,W), got {buffer.shape}", valid=False)
        T, H, W = buffer.shape[:3]
        if T < 30:
            return SpatialQualityResult(
                error=f"Insufficient temporal samples {T} < 30", valid=False)
        cs = self.cell_size
        nH, nW = H // cs, W // cs
        quality = np.zeros((nH * cs, nW * cs), dtype=np.float32)
        for i in range(nH):
            for j in range(nW):
                cell = buffer[:, i*cs:(i+1)*cs, j*cs:(j+1)*cs]
                sig = cell.mean()
                _, _, db = snr_cardiac(np.atleast_1d(sig), fps, self.band)
                quality[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = db
        stats = {
            "mean_db": float(np.mean(quality)) if np.any(quality != 0) else 0.0,
            "max_db": float(np.max(quality)),
            "min_db": float(np.min(quality)),
            "T": T,
            "fps": fps,
            "band": self.band,
        }
        return SpatialQualityResult(quality_map=quality, stats=stats)

    def average(self):
        if not self._history:
            return None
        return np.mean(self._history, axis=0)

    def reset(self):
        self._history.clear()
