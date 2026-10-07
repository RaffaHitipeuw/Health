"""
Classical Evolution Experiments for Phase 2.

Tests scientifically justified modifications to classical methods.

HYPOTHESIS → BASELINE → CHANGE → EXPERIMENT → RESULT → DECISION

Each evolution follows the research loop pattern.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy.signal import butter, sosfilt, detrend as scipy_detrend
from scipy.fft import rfft, rfftfreq

from rppg_phase2_signal_lab import BPMExtractor, ClassicalMethodResult


# =============================================================================
# EVOLUTION HYPOTHESIS
# =============================================================================

@dataclass
class EvolutionHypothesis:
    """
    A scientifically justified hypothesis for method improvement.

    Each hypothesis follows the form:
    - OBSERVATION: What we observe in the baseline
    - HYPOTHESIS: Why this happens
    - CHANGE: What we modify
    - EXPECTED: What we expect to happen
    """
    id: str
    method: str
    observation: str
    hypothesis: str
    baseline_issue: str
    proposed_change: str
    expected_effect: str


# Predefined hypotheses based on common classical method failure modes
EVOLUTION_HYPOTHESES: Dict[str, EvolutionHypothesis] = {

    "CHROM-harmonic": EvolutionHypothesis(
        id="CHROM-harmonic",
        method="CHROM",
        observation="CHROM shows harmonic confusion at 2*f_hr",
        hypothesis="Standard deviation normalization can amplify harmonics",
        baseline_issue="Incorrect BPM (2x true HR)",
        proposed_change="Add harmonic check with sub-harmonic validation",
        expected_effect="Reduce harmonic confusion failures",
    ),

    "POS-detrending": EvolutionHypothesis(
        id="POS-detrending",
        method="POS",
        observation="POS shows drift with slow illumination changes",
        hypothesis="POS is sensitive to baseline drift",
        baseline_issue="Slow drift artifacts",
        proposed_change="Add robust detrending (moving average subtraction)",
        expected_effect="Reduce drift artifacts",
    ),

    "GREEN-normalization": EvolutionHypothesis(
        id="GREEN-normalization",
        method="GREEN",
        observation="GREEN varies with skin reflectance changes",
        hypothesis="Mean normalization insufficient for varying skin tones",
        baseline_issue="Signal amplitude variation",
        proposed_change="Add amplitude normalization",
        expected_effect="More consistent signal strength",
    ),

    "ICA-deterministic": EvolutionHypothesis(
        id="ICA-deterministic",
        method="ICA",
        observation="ICA results vary between runs (random initialization)",
        hypothesis="Random ICA initialization causes non-determinism",
        baseline_issue="Non-reproducible results",
        proposed_change="Use fixed random seed",
        expected_effect="Deterministic results",
    ),
}


# =============================================================================
# EVOLUTION EXPERIMENT
# =============================================================================

@dataclass
class EvolutionExperiment:
    """
    A controlled experiment testing a hypothesis.

    Compares baseline vs. modified method under identical conditions.
    """
    hypothesis_id: str
    method: str

    # Signal under test
    signal: np.ndarray
    fps: float

    # Baseline (no modification)
    baseline_bpm: float = 0.0
    baseline_diagnostics: Dict = field(default_factory=dict)

    # Modified (with proposed change)
    modified_bpm: float = 0.0
    modified_diagnostics: Dict = field(default_factory=dict)

    # Metrics
    bpm_difference: float = 0.0
    improvement_detected: bool = False

    # Decision
    decision: str = "INCONCLUSIVE"  # KEEP, MODIFY, REJECT, INCONCLUSIVE
    reasoning: str = ""


# =============================================================================
# EVOLUTION OPERATORS
# =============================================================================

class EvolutionOperators:
    """
    Modification operators for classical methods.

    These operators apply specific changes to the signal processing chain.
    Each operator should be well-documented with the scientific justification.
    """

    @staticmethod
    def add_harmonic_check(
        signal: np.ndarray,
        fps: float,
        cardiac_low: float = 0.833,
        cardiac_high: float = 3.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Extract BPM with harmonic/sub-harmonic validation.

        Checks if detected peak is a harmonic or sub-harmonic of a stronger peak.
        If yes, returns the more plausible fundamental/sub-harmonic frequency.

        Returns
        -------
        Tuple[float, dict]
            (bpm, diagnostics)
        """
        n = len(signal)
        if n < 30:
            return 0.0, {"error": "insufficient_samples"}

        # Bandpass
        sig = scipy_detrend(signal, type='linear')
        nyq = 0.5 * fps
        low = max(cardiac_low / nyq, 1e-6)
        high = min(cardiac_high / nyq, 0.999)
        sos = butter(4, [low, high], btype='band', output='sos')
        sig = sosfilt(sos, sig)

        # FFT
        from rppg_signal import apply_windowing
        sig_w = apply_windowing(sig)
        fft_v = np.abs(rfft(sig_w))
        freqs = rfftfreq(n, d=1.0 / fps)
        mask = (freqs >= cardiac_low) & (freqs <= cardiac_high)

        if not np.any(mask):
            return 0.0, {"error": "no_cardiac_band"}

        band_freqs = freqs[mask]
        band_power = fft_v[mask]
        peak_idx = np.argmax(band_power)
        peak_hz = band_freqs[peak_idx]
        peak_power = band_power[peak_idx]

        # Check for sub-harmonic (half frequency)
        sub_hz = peak_hz / 2.0
        if sub_hz >= cardiac_low:
            sub_mask = (freqs >= sub_hz - 0.1) & (freqs <= sub_hz + 0.1)
            if np.any(sub_mask):
                sub_power = fft_v[sub_mask].max()
                # If sub-harmonic has >40% of peak power, prefer it
                if sub_power > 0.4 * peak_power:
                    peak_hz = sub_hz

        bpm = peak_hz * 60.0
        return float(bpm), {
            "method": "harmonic_check",
            "peak_hz": peak_hz,
            "harmonic_checked": True,
        }

    @staticmethod
    def add_robust_detrending(
        signal: np.ndarray,
        window_sec: float = 1.5,
        fps: float = 30.0,
    ) -> np.ndarray:
        """
        Apply robust detrending: linear + moving average subtraction.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        window_sec : float
            Window size in seconds for moving average.
        fps : float
            Frame rate.

        Returns
        -------
        np.ndarray
            Detrended signal.
        """
        # Linear detrend
        sig = scipy_detrend(signal, type='linear')

        # Moving average subtraction
        window_samples = int(window_sec * fps)
        if window_samples > 1:
            padded = np.pad(sig, (window_samples//2, window_samples//2), mode='reflect')
            kernel = np.ones(window_samples) / window_samples
            ma = np.convolve(padded, kernel, mode='valid')
            sig = sig - ma[:len(sig)]

        return sig

    @staticmethod
    def add_amplitude_normalization(
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize signal to unit variance and zero mean.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.

        Returns
        -------
        np.ndarray
            Normalized signal.
        """
        std = np.std(signal)
        if std > 1e-9:
            return (signal - np.mean(signal)) / std
        return signal - np.mean(signal)


# =============================================================================
# EVOLUTION RUNNER
# =============================================================================

class EvolutionRunner:
    """
    Runs evolution experiments comparing baseline vs. modified methods.

    Uses the same input signal for fair comparison.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._extractor = BPMExtractor(fps)

    def run_baseline(self, signal: np.ndarray, method: str) -> Tuple[float, Dict]:
        """Run baseline BPM extraction."""
        bpm, freqs, power, diag = self._extractor.extract_bpm(signal, method)
        diag["method"] = f"{method}_baseline"
        return bpm, diag

    def run_evolution(
        self,
        signal: np.ndarray,
        hypothesis_id: str,
        operators: List[Tuple[str, Callable]],
    ) -> Tuple[float, Dict]:
        """
        Run evolution with specified operators.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        hypothesis_id : str
            Hypothesis being tested.
        operators : List[Tuple[name, function]]
            List of (name, operator_function) tuples to apply.

        Returns
        -------
        Tuple[float, dict]
            (modified_bpm, diagnostics)
        """
        modified = signal.copy()
        diag = {
            "hypothesis": hypothesis_id,
            "operators": [op[0] for op in operators],
        }

        for op_name, op_func in operators:
            if op_name == "harmonic_check":
                modified_bpm, op_diag = EvolutionOperators.add_harmonic_check(
                    modified, self.fps
                )
                diag[f"op_{op_name}"] = op_diag
                if modified_bpm > 0:
                    return modified_bpm, diag
            elif op_name == "detrending":
                modified = EvolutionOperators.add_robust_detrending(modified, fps=self.fps)
                diag[f"op_{op_name}"] = {"applied": True}
            elif op_name == "normalization":
                modified = EvolutionOperators.add_amplitude_normalization(modified)
                diag[f"op_{op_name}"] = {"applied": True}

        # Final BPM extraction
        bpm, freqs, power, bpm_diag = self._extractor.extract_bpm(modified, f"evolved_{hypothesis_id}")
        diag.update(bpm_diag)
        return bpm, diag

    def compare(
        self,
        signal: np.ndarray,
        method: str,
        hypothesis_id: str,
        operators: List[Tuple[str, Callable]],
    ) -> EvolutionExperiment:
        """
        Run comparison experiment.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        method : str
            Base method name.
        hypothesis_id : str
            Hypothesis being tested.
        operators : List[Tuple]
            Operators to apply.

        Returns
        -------
        EvolutionExperiment
            Comparison result.
        """
        # Baseline
        baseline_bpm, baseline_diag = self.run_baseline(signal, method)

        # Evolved
        evolved_bpm, evolved_diag = self.run_evolution(signal, hypothesis_id, operators)

        # Compute difference
        bpm_diff = abs(evolved_bpm - baseline_bpm)

        return EvolutionExperiment(
            hypothesis_id=hypothesis_id,
            method=method,
            signal=signal,
            fps=self.fps,
            baseline_bpm=baseline_bpm,
            baseline_diagnostics=baseline_diag,
            modified_bpm=evolved_bpm,
            modified_diagnostics=evolved_diag,
            bpm_difference=bpm_diff,
            improvement_detected=False,  # Requires GT to determine
            decision="INCONCLUSIVE",
            reasoning="No GT available for comparison",
        )


# =============================================================================
# EXAMPLE EVOLUTION EXPERIMENTS
# =============================================================================

def run_chrom_harmonic_evolution(
    chrom_signal: np.ndarray,
    fps: float = 30.0,
    gt_bpm: Optional[float] = None,
) -> EvolutionExperiment:
    """
    Test CHROM with harmonic check.

    Hypothesis: CHROM harmonic confusion can be reduced by sub-harmonic validation.
    """
    runner = EvolutionRunner(fps=fps)

    operators = [("harmonic_check", EvolutionOperators.add_harmonic_check)]

    experiment = runner.compare(
        signal=chrom_signal,
        method="CHROM",
        hypothesis_id="CHROM-harmonic",
        operators=operators,
    )

    # If GT available, determine if evolution helped
    if gt_bpm is not None and gt_bpm > 0:
        baseline_error = abs(experiment.baseline_bpm - gt_bpm)
        evolved_error = abs(experiment.modified_bpm - gt_bpm)

        experiment.improvement_detected = evolved_error < baseline_error

        if experiment.improvement_detected:
            experiment.decision = "KEEP"
            experiment.reasoning = f"Evolved ({evolved_error:.1f} BPM error) closer to GT than baseline ({baseline_error:.1f} BPM error)"
        else:
            experiment.decision = "REJECT"
            experiment.reasoning = f"Evolution did not improve: baseline error {baseline_error:.1f}, evolved error {evolved_error:.1f}"

    return experiment


def run_pos_detrending_evolution(
    pos_signal: np.ndarray,
    fps: float = 30.0,
    gt_bpm: Optional[float] = None,
) -> EvolutionExperiment:
    """
    Test POS with robust detrending.

    Hypothesis: POS is sensitive to drift; robust detrending can help.
    """
    runner = EvolutionRunner(fps=fps)

    operators = [
        ("detrending", EvolutionOperators.add_robust_detrending),
        ("harmonic_check", EvolutionOperators.add_harmonic_check),
    ]

    experiment = runner.compare(
        signal=pos_signal,
        method="POS",
        hypothesis_id="POS-detrending",
        operators=operators,
    )

    # If GT available, determine if evolution helped
    if gt_bpm is not None and gt_bpm > 0:
        baseline_error = abs(experiment.baseline_bpm - gt_bpm)
        evolved_error = abs(experiment.modified_bpm - gt_bpm)

        experiment.improvement_detected = evolved_error < baseline_error

        if experiment.improvement_detected:
            experiment.decision = "KEEP"
            experiment.reasoning = f"Evolved ({evolved_error:.1f} BPM error) closer to GT than baseline ({baseline_error:.1f} BPM error)"
        else:
            experiment.decision = "REJECT"
            experiment.reasoning = f"Evolution did not improve: baseline error {baseline_error:.1f}, evolved error {evolved_error:.1f}"

    return experiment


if __name__ == "__main__":
    print("=" * 60)
    print("CLASSICAL EVOLUTION EXPERIMENTS READY")
    print("=" * 60)

    print("\nAvailable Hypotheses:")
    for hid, hyp in EVOLUTION_HYPOTHESES.items():
        print(f"\n{hyp.id}:")
        print(f"  Method: {hyp.method}")
        print(f"  Observation: {hyp.observation}")
        print(f"  Change: {hyp.proposed_change}")
        print(f"  Expected: {hyp.expected_effect}")

    print("\n" + "=" * 60)
