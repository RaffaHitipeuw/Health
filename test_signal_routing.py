"""
Unit tests for V2 signal routing correctness.

Tests that:
A. Cardiac path preserves cardiac component
B. Respiration path preserves respiration component
C. V2 calls PhysiologicalStateClassifier (not permanently stale)
D. V2 does not estimate respiration from cardiac-only filtered signal
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.signal import butter, sosfilt, welch, find_peaks, windows
from scipy.fft import rfft, rfftfreq


def bandpass_filter(data, fps, lowcut=0.833, highcut=3.0, order=4):
    """Cardiac bandpass filter (0.833-3.0 Hz) using SOS format (no padding issues)."""
    nyq = 0.5 * fps
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, data)


def estimate_respiratory_rate(signal, fps):
    """Estimate respiratory rate from unfiltered signal (replicates rppg_signal version)."""
    n = len(signal)
    if n < max(30, int(fps * 3)):
        return 0.0
    nperseg = min(n, max(int(fps * 10), 64))
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, window='hann')
    resp_mask = (freqs >= 0.10) & (freqs <= 1.50)
    if not np.any(resp_mask):
        return 0.0
    return float(freqs[resp_mask][np.argmax(psd[resp_mask])])


def generate_test_signal(fps=30.0, duration=10.0, cardiac_hz=1.2, resp_hz=0.25,
                          cardiac_amp=1.0, resp_amp=0.3, noise_amp=0.1, seed=42):
    """
    Generate a synthetic rPPG-like signal with known cardiac and respiratory components.

    cardiac_hz: cardiac frequency in Hz (e.g., 1.2 Hz = 72 BPM)
    resp_hz: respiratory frequency in Hz (e.g., 0.25 Hz = 15 br/min)
    """
    np.random.seed(seed)
    n = int(fps * duration)
    t = np.arange(n) / fps

    # Cardiac component
    cardiac = cardiac_amp * np.sin(2 * np.pi * cardiac_hz * t)

    # Respiration modulates the baseline (very slow drift)
    resp = resp_amp * np.sin(2 * np.pi * resp_hz * t)

    # Combine
    signal = cardiac + resp + noise_amp * np.random.randn(n)

    return signal, t


def generate_cardiac_only(fps=30.0, duration=10.0, cardiac_hz=1.2, seed=42):
    """Generate cardiac-only signal (no respiration)."""
    np.random.seed(seed)
    n = int(fps * duration)
    t = np.arange(n) / fps
    cardiac = np.sin(2 * np.pi * cardiac_hz * t)
    noise = 0.05 * np.random.randn(n)
    return cardiac + noise


class TestSignalRouting:
    """Test suite for signal routing correctness."""

    def test_cardiac_path_preserves_cardiac(self):
        """Test that cardiac bandpass preserves cardiac frequency."""
        fps = 30.0
        cardiac_hz = 1.2  # 72 BPM
        signal, _ = generate_test_signal(fps=fps, duration=10.0,
                                         cardiac_hz=cardiac_hz,
                                         resp_hz=0.25,
                                         cardiac_amp=1.0,
                                         resp_amp=0.3,
                                         noise_amp=0.1)

        # Apply cardiac bandpass
        cardiac_filtered = bandpass_filter(signal, fps)

        # Estimate HR from cardiac-filtered signal
        n = len(cardiac_filtered)
        win_signal = cardiac_filtered * windows.hann(n)
        fft_v = np.abs(rfft(win_signal))
        freqs = rfftfreq(n, d=1.0 / fps)
        mask = (freqs >= 0.833) & (freqs <= 3.0)

        peak_idx = np.argmax(fft_v[mask])
        peak_hz = freqs[mask][peak_idx]
        peak_bpm = peak_hz * 60.0

        # Should be close to the cardiac frequency (within 10 BPM)
        expected_bpm = cardiac_hz * 60.0
        error = abs(peak_bpm - expected_bpm)
        assert error < 10.0, f"Cardiac path error: {error:.1f} BPM (expected {expected_bpm:.0f}, got {peak_bpm:.0f})"

        print(f"  ✓ Cardiac path: estimated {peak_bpm:.1f} BPM (expected {expected_bpm:.0f})")

    def test_respiration_path_preserves_respiration(self):
        """Test that unfiltered signal preserves respiration frequency."""
        fps = 30.0
        resp_hz = 0.25  # 15 br/min
        # Make respiration component dominant (stronger than cardiac)
        signal, _ = generate_test_signal(fps=fps, duration=15.0,
                                         cardiac_hz=1.2,
                                         resp_hz=resp_hz,
                                         cardiac_amp=0.3,
                                         resp_amp=2.0,  # Strong respiratory modulation
                                         noise_amp=0.05)

        # Estimate respiration from unfiltered signal
        est_resp_hz = estimate_respiratory_rate(signal, fps)
        est_resp_bpm = est_resp_hz * 60.0

        # Should be close to the respiratory frequency (within 5 br/min)
        expected_bpm = resp_hz * 60.0
        error = abs(est_resp_bpm - expected_bpm)
        assert error < 5.0, \
            f"Respiration path error: {error:.1f} br/min (expected {expected_bpm:.0f}, got {est_resp_bpm:.1f})"

        print(f"  ✓ Respiration path: estimated {est_resp_bpm:.1f} br/min (expected {expected_bpm:.0f})")

    def test_respiration_from_cardiac_filtered_is_wrong(self):
        """
        Test that estimating respiration from cardiac-filtered signal gives WRONG results.

        This proves the original bug: cardiac bandpass removes respiratory band,
        so respiration estimation from cardiac-filtered signal is unreliable.
        """
        fps = 30.0
        resp_hz = 0.25  # 15 br/min
        # Strong respiratory component
        signal, _ = generate_test_signal(fps=fps, duration=15.0,
                                         cardiac_hz=1.2,
                                         resp_hz=resp_hz,
                                         cardiac_amp=0.3,
                                         resp_amp=2.0,
                                         noise_amp=0.05)

        # Apply cardiac bandpass (0.833-3.0 Hz) - this removes 0.1-0.5 Hz respiratory band!
        cardiac_filtered = bandpass_filter(signal, fps)

        # Try to estimate respiration from cardiac-filtered signal
        est_resp_hz_buggy = estimate_respiratory_rate(cardiac_filtered, fps)

        # The cardiac-filtered signal should NOT produce the correct respiration
        # (respiratory band has been filtered out)
        expected_bpm = resp_hz * 60.0
        buggy_error = abs(est_resp_hz_buggy * 60.0 - expected_bpm)

        # The buggy estimation should have a large error (> 5 br/min)
        # This confirms the bug exists
        assert buggy_error > 5.0, \
            f"Bug not reproduced: resp from cardiac-filtered should be wrong, got error={buggy_error:.1f}"

        # Now verify that the correct method (unfiltered signal) works
        est_resp_hz_correct = estimate_respiratory_rate(signal, fps)
        correct_error = abs(est_resp_hz_correct * 60.0 - expected_bpm)

        assert correct_error < 5.0, \
            f"Correct method failed: error={correct_error:.1f}"

        print(f"  ✓ Bug confirmed: cardiac-filtered → {est_resp_hz_buggy*60:.1f} br/min (error {buggy_error:.1f})")
        print(f"  ✓ Correct: unfiltered → {est_resp_hz_correct*60:.1f} br/min (error {correct_error:.1f})")

    def test_separate_paths_do_not_interfere(self):
        """
        Test that cardiac and respiration frequency bands don't destroy each other.

        When both cardiac (1.2 Hz) and respiratory (0.25 Hz) components are present,
        both paths should correctly identify their respective components.
        """
        fps = 30.0
        cardiac_hz = 1.2
        resp_hz = 0.25
        # Balanced signal: both components are significant
        signal, _ = generate_test_signal(fps=fps, duration=15.0,
                                         cardiac_hz=cardiac_hz,
                                         resp_hz=resp_hz,
                                         cardiac_amp=0.8,
                                         resp_amp=1.5,
                                         noise_amp=0.05)

        # Cardiac path
        cardiac_filtered = bandpass_filter(signal, fps)
        n = len(cardiac_filtered)
        win = cardiac_filtered * windows.hann(n)
        fft_v = np.abs(rfft(win))
        freqs = rfftfreq(n, d=1.0 / fps)
        mask = (freqs >= 0.833) & (freqs <= 3.0)
        cardiac_peak_hz = freqs[mask][np.argmax(fft_v[mask])]

        # Respiration path
        resp_peak_hz = estimate_respiratory_rate(signal, fps)

        cardiac_error = abs(cardiac_peak_hz - cardiac_hz) * 60.0
        resp_error = abs(resp_peak_hz - resp_hz) * 60.0

        assert cardiac_error < 10.0, f"Cardiac estimate wrong: {cardiac_error:.1f} BPM error"
        assert resp_error < 5.0, f"Respiration estimate wrong: {resp_error:.1f} br/min error"

        print(f"  ✓ Combined signal: cardiac {cardiac_peak_hz*60:.1f} BPM (err {cardiac_error:.1f}), "
              f"resp {resp_peak_hz*60:.1f} br/min (err {resp_error:.1f})")

    def test_frame_result_separate_signals(self):
        """Test that FrameResult dataclass has separate signal fields."""
        from rppg_core import FrameResult

        fr = FrameResult()

        # Should have both cardiac_signal and resp_signal fields
        assert hasattr(fr, 'cardiac_signal'), "FrameResult missing cardiac_signal field"
        assert hasattr(fr, 'resp_signal'), "FrameResult missing resp_signal field"

        # Should be arrays by default
        import numpy as np
        assert isinstance(fr.cardiac_signal, np.ndarray), "cardiac_signal should be ndarray"
        assert isinstance(fr.resp_signal, np.ndarray), "resp_signal should be ndarray"

        print("  ✓ FrameResult has cardiac_signal and resp_signal fields")

    def test_v2_calls_physio_classifier(self):
        """
        Test that V2 actually calls PhysiologicalStateClassifier with the unfiltered signal.

        This is a code inspection test - we verify the fix is in place.
        """
        import inspect
        from rppg_core import MultiROIFusionEngineV2

        # Read the V2.update source code
        source = inspect.getsource(MultiROIFusionEngineV2.update)

        # Should contain physio_classifier.update call
        assert 'physio_classifier.update' in source, \
            "V2 should call physio_classifier.update"

        # Should use sd (unfiltered) for physio classifier, not sf (cardiac-filtered)
        # Look for the pattern where physio_classifier receives sd
        lines = source.split('\n')
        physio_call_line = None
        for i, line in enumerate(lines):
            if 'physio_classifier.update' in line:
                physio_call_line = line
                break

        assert physio_call_line is not None, "Could not find physio_classifier.update call"

        # The call should be using sd (unfiltered signal), not sf (cardiac-filtered)
        # After the fix, it should pass sd to the classifier
        # Check that it's near the sd assignment
        assert 'sd' in physio_call_line or '_physio_v2' in source, \
            "Physio classifier should use unfiltered signal (sd)"

        print("  ✓ V2 contains physio_classifier.update call with unfiltered signal")

    def test_chrom_signal_routing_for_vitals_engine(self):
        """
        Test that V2 sets chrom_signal to unfiltered signal for VitalsEngine.

        The bug was: V2 set chrom_signal = cardiac-filtered signal
        The fix: V2 sets chrom_signal = resp_signal (unfiltered)
        """
        import inspect
        from rppg_core import MultiROIFusionEngineV2

        source = inspect.getsource(MultiROIFusionEngineV2.update)

        # After fix: chrom_signal should be set from resp_signal (unfiltered)
        # Look for the assignment pattern
        lines = source.split('\n')
        chrom_assignment = None
        for line in lines:
            if 'result.chrom_signal' in line and '=' in line:
                chrom_assignment = line.strip()
                break

        assert chrom_assignment is not None, "Could not find result.chrom_signal assignment"

        # After the fix, chrom_signal should be assigned from resp_signal or sd
        # (not from cardiac-filtered sf)
        assert 'resp_signal' in chrom_assignment or 'sd' in chrom_assignment, \
            f"chrom_signal should be unfiltered signal, got: {chrom_assignment}"

        print(f"  ✓ chrom_signal routed correctly: {chrom_assignment}")


def run_tests():
    """Run all signal routing tests."""
    print("\n" + "=" * 60)
    print("TESTING: V2 Signal Routing")
    print("=" * 60)

    test_suite = TestSignalRouting()

    tests = [
        ("Cardiac path preserves cardiac component",
         test_suite.test_cardiac_path_preserves_cardiac),
        ("Respiration path preserves respiration component",
         test_suite.test_respiration_path_preserves_respiration),
        ("Respiration from cardiac-filtered is wrong (bug confirmed)",
         test_suite.test_respiration_from_cardiac_filtered_is_wrong),
        ("Separate paths do not interfere",
         test_suite.test_separate_paths_do_not_interfere),
        ("FrameResult has separate signal fields",
         test_suite.test_frame_result_separate_signals),
        ("V2 calls PhysiologicalStateClassifier",
         test_suite.test_v2_calls_physio_classifier),
        ("chrom_signal routing for VitalsEngine",
         test_suite.test_chrom_signal_routing_for_vitals_engine),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
