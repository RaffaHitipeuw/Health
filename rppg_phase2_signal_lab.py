"""
Classical Signal Lab for Phase 2 rPPG Research.

Standalone environment for benchmarking classical signal extraction methods
independent of the frozen V2 production code.

Architecture:
    video → ROI extraction → RGB signal → Classical method → BVP
        → BPM extraction → timestamped predictions → Benchmark evaluator

This lab does NOT modify V2. It creates a research environment for
classical method characterization.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from scipy.signal import butter, sosfilt, detrend as scipy_detrend, find_peaks
from scipy.fft import rfft, rfftfreq

from rppg_algorithms import (
    extract_green,
    extract_chrom,
    extract_pos,
    extract_ica,
    get_algorithm_by_name,
)
from rppg_signal import apply_windowing


# =============================================================================
# BPM EXTRACTION LAYER
# =============================================================================

@dataclass
class BPMExtractorConfig:
    """Configuration for BPM extraction."""
    cardiac_low: float = 0.833  # Hz (50 BPM)
    cardiac_high: float = 3.0    # Hz (180 BPM)
    fft_window_sec: float = 10.0  # seconds
    fft_hop_sec: float = 5.0    # seconds
    min_samples: int = 30


class BPMExtractor:
    """
    Common BPM extraction for classical methods.

    Uses the same cardiac bandpass and FFT approach across methods to ensure
    fair comparison. This is configurable but defaults to Sanubari V2 values.
    """

    def __init__(self, fps: float, config: Optional[BPMExtractorConfig] = None):
        self.fps = fps
        self.config = config or BPMExtractorConfig()
        self._window_samples = int(self.config.fft_window_sec * fps)
        self._hop_samples = int(self.config.fft_hop_sec * fps)

    def extract_bpm(
        self,
        signal: np.ndarray,
        method_name: str = "unknown",
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extract BPM from BVP/rPPG signal using FFT.

        Parameters
        ----------
        signal : np.ndarray
            Input BVP/rPPG signal (1D).
        method_name : str
            Method name for diagnostics.

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray, dict]
            (bpm, frequencies, power, diagnostics)
        """
        n = len(signal)
        if n < self.config.min_samples:
            return 0.0, np.array([]), np.array([]), {"error": "insufficient_samples"}

        # Detrend
        sig = scipy_detrend(signal, type='linear')

        # Bandpass filter
        sig = self._bandpass(sig)

        # Standardize
        std = np.std(sig)
        if std > 1e-9:
            sig = (sig - np.mean(sig)) / std
        else:
            sig = sig - np.mean(sig)

        # Apply windowing
        sig_windowed = apply_windowing(sig, window_type='hann')

        # FFT
        fft_v = np.abs(rfft(sig_windowed))
        freqs = rfftfreq(n, d=1.0 / self.fps)

        # Cardiac band mask
        mask = (freqs >= self.config.cardiac_low) & (freqs <= self.config.cardiac_high)
        if not np.any(mask):
            return 0.0, freqs, fft_v, {"error": "no_cardiac_band"}

        # Find peak
        band_freqs = freqs[mask]
        band_power = fft_v[mask]
        peak_idx = np.argmax(band_power)
        peak_hz = band_freqs[peak_idx]
        bpm = peak_hz * 60.0

        # Diagnostics
        snr = self._estimate_snr(sig, peak_hz)

        return float(bpm), freqs, fft_v, {
            "method": method_name,
            "snr_db": snr,
            "n_samples": n,
            "peak_hz": peak_hz,
        }

    def extract_bpm_windows(
        self,
        signal: np.ndarray,
        method_name: str = "unknown",
    ) -> List[Tuple[float, float, Dict[str, Any]]]:
        """
        Extract BPM using sliding window approach.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        method_name : str
            Method name.

        Returns
        -------
        List[Tuple[float, float, dict]]
            List of (bpm, timestamp, diagnostics) tuples.
        """
        n = len(signal)
        if n < self._window_samples:
            bpm, freqs, power, diag = self.extract_bpm(signal, method_name)
            return [(bpm, 0.0, diag)]

        results = []
        hop = self._hop_samples
        window = self._window_samples

        for start in range(0, n - window + 1, hop):
            segment = signal[start:start + window]
            timestamp = start / self.fps
            bpm, _, _, diag = self.extract_bpm(segment, method_name)
            diag["timestamp"] = timestamp
            diag["frame_start"] = start
            results.append((bpm, timestamp, diag))

        return results

    def _bandpass(self, signal: np.ndarray) -> np.ndarray:
        """Apply cardiac bandpass filter."""
        nyq = 0.5 * self.fps
        low = max(self.config.cardiac_low / nyq, 1e-6)
        high = min(self.config.cardiac_high / nyq, 0.999)
        if low >= high:
            return signal
        sos = butter(4, [low, high], btype='band', output='sos')
        return sosfilt(sos, signal)

    def _estimate_snr(self, signal: np.ndarray, peak_hz: float) -> float:
        """Estimate SNR in dB."""
        n = len(signal)
        freqs = rfftfreq(n, d=1.0 / self.fps)
        fft_v = np.abs(rfft(apply_windowing(signal)))

        # Cardiac band
        band_mask = (freqs >= self.config.cardiac_low) & (freqs <= self.config.cardiac_high)
        if not np.any(band_mask):
            return 0.0

        # Peak energy
        peak_mask = (freqs >= peak_hz - 0.1) & (freqs <= peak_hz + 0.1)
        peak_energy = np.sum(fft_v[peak_mask]**2) if np.any(peak_mask) else 0.0

        # Noise floor (excluding cardiac band)
        noise_mask = ~band_mask
        noise_energy = np.sum(fft_v[noise_mask]**2) if np.any(noise_mask) else 1e-9

        if noise_energy < 1e-9:
            return 0.0

        snr_linear = peak_energy / noise_energy
        return 10 * np.log10(snr_linear + 1e-9)


# =============================================================================
# CLASSICAL METHOD INTERFACE
# =============================================================================

@dataclass
class ClassicalMethodResult:
    """Result from classical method processing."""
    method_name: str
    bvp_signal: np.ndarray
    timestamps: np.ndarray
    fps: float

    # BPM extraction
    bpm: float = 0.0
    bpm_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Windowed BPM if available
    bpm_windows: List[Tuple[float, float, Dict]] = field(default_factory=list)

    # Quality metrics
    snr_db: float = 0.0
    signal_valid: bool = True
    error: Optional[str] = None


class ClassicalMethod:
    """
    Common interface for classical rPPG methods.

    All classical methods should implement this interface for fair comparison.
    """

    def __init__(self, name: str, extractor: Callable):
        self.name = name
        self.extractor = extractor
        self._rgb_buffer_r: List[float] = []
        self._rgb_buffer_g: List[float] = []
        self._rgb_buffer_b: List[float] = []
        self._timestamps: List[float] = []
        self._fps = 30.0

    def ingest(self, r: float, g: float, b: float, timestamp: float = 0.0):
        """Ingest RGB sample."""
        self._rgb_buffer_r.append(r)
        self._rgb_buffer_g.append(g)
        self._rgb_buffer_b.append(b)
        self._timestamps.append(timestamp)

    def set_fps(self, fps: float):
        """Set FPS for BPM extraction."""
        self._fps = fps

    def process(self) -> ClassicalMethodResult:
        """Process accumulated RGB signal."""
        if len(self._rgb_buffer_g) < 30:
            return ClassicalMethodResult(
                method_name=self.name,
                bvp_signal=np.array([]),
                timestamps=np.array([]),
                fps=self._fps,
                signal_valid=False,
                error="insufficient_samples",
            )

        r = np.array(self._rgb_buffer_r)
        g = np.array(self._rgb_buffer_g)
        b = np.array(self._rgb_buffer_b)
        timestamps = np.array(self._timestamps)

        # Extract BVP using method
        bvp = self.extractor(r, g, b)

        # Extract BPM
        bpm_ext = BPMExtractor(self._fps)
        bpm, freqs, power, diag = bpm_ext.extract_bpm(bvp, self.name)

        # Windowed BPM
        bpm_windows = bpm_ext.extract_bpm_windows(bvp, self.name)

        return ClassicalMethodResult(
            method_name=self.name,
            bvp_signal=bvp,
            timestamps=timestamps,
            fps=self._fps,
            bpm=bpm,
            bpm_diagnostics=diag,
            bpm_windows=bpm_windows,
            snr_db=diag.get("snr_db", 0.0),
            signal_valid=True,
        )

    def reset(self):
        """Reset buffers."""
        self._rgb_buffer_r.clear()
        self._rgb_buffer_g.clear()
        self._rgb_buffer_b.clear()
        self._timestamps.clear()


# =============================================================================
# CLASSICAL SIGNAL LAB
# =============================================================================

class ClassicalSignalLab:
    """
    Standalone signal lab for classical rPPG method benchmarking.

    This lab operates independently of the frozen V2 production code.
    It provides a controlled environment for classical method characterization.
    """

    # Supported methods
    SUPPORTED_METHODS = ["GREEN", "CHROM", "POS", "ICA"]

    def __init__(
        self,
        fps: float = 30.0,
        methods: Optional[List[str]] = None,
    ):
        """
        Initialize classical signal lab.

        Parameters
        ----------
        fps : float
            Frame rate for BPM extraction.
        methods : List[str], optional
            List of methods to benchmark. Default: all supported.
        """
        self.fps = fps
        self.methods = methods or self.SUPPORTED_METHODS

        # Initialize method processors
        self._processors: Dict[str, ClassicalMethod] = {}
        for method in self.methods:
            if method not in self.SUPPORTED_METHODS:
                continue
            extractor = get_algorithm_by_name(method)
            self._processors[method] = ClassicalMethod(method, extractor)

    def ingest_frame(
        self,
        r: float,
        g: float,
        b: float,
        timestamp: float = 0.0,
    ):
        """Ingest RGB sample into all method processors."""
        for processor in self._processors.values():
            processor.ingest(r, g, b, timestamp)

    def process_all(self) -> Dict[str, ClassicalMethodResult]:
        """Process all accumulated data through all methods."""
        results = {}
        for name, processor in self._processors.items():
            processor.set_fps(self.fps)
            results[name] = processor.process()
        return results

    def reset(self):
        """Reset all processors."""
        for processor in self._processors.values():
            processor.reset()


# =============================================================================
# BPM EXTRACTION VALIDATION
# =============================================================================

def validate_bpm_extraction():
    """Validate BPM extraction produces reasonable results."""
    # Generate synthetic pulse signal at 72 BPM
    fps = 30.0
    duration = 10.0  # seconds
    n = int(duration * fps)
    t = np.arange(n) / fps
    bpm_true = 72.0
    omega = 2 * np.pi * bpm_true / 60.0

    # Synthetic BVP with harmonics
    signal = np.sin(omega * t)
    signal += 0.3 * np.sin(2 * omega * t)  # 2nd harmonic
    signal += 0.1 * np.random.randn(n)  # noise

    # Extract BPM
    extractor = BPMExtractor(fps)
    bpm, freqs, power, diag = extractor.extract_bpm(signal, "test")

    # Validate
    error_bpm = abs(bpm - bpm_true)
    assert error_bpm < 5.0, f"BPM error {error_bpm:.1f} BPM exceeds threshold"
    assert diag["snr_db"] > 0, "SNR should be positive for synthetic signal"

    return {
        "bpm_true": bpm_true,
        "bpm_estimated": bpm,
        "error": error_bpm,
        "snr_db": diag["snr_db"],
    }


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("CLASSICAL SIGNAL LAB VALIDATION")
    print("=" * 60)

    # Validate BPM extraction
    result = validate_bpm_extraction()
    print(f"\nBPM Extraction Test:")
    print(f"  True BPM:   {result['bpm_true']:.1f}")
    print(f"  Estimated:  {result['bpm_estimated']:.1f}")
    print(f"  Error:     {result['error']:.1f} BPM")
    print(f"  SNR:        {result['snr_db']:.2f} dB")
    print(f"  Status:     {'PASS' if result['error'] < 5.0 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Classical Signal Lab Ready")
    print("=" * 60)
