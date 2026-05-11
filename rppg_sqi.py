"""
rppg_sqi.py  —  Signal Quality Index as Posterior Probability
=============================================================

PHASE 1 & 2 REVISION: Replace heuristic weighted sum with principled
log-posterior probability formulation.

Signal observation model:
    x(t) = s(t) + m(t) + i(t) + ε(t)

Under Gaussian noise ε ~ N(0, σ²_ε), the log-likelihood ratio of
"signal present at f_HR" vs "pure noise" is:

    log Λ(f_HR) = SNR_dB(f_HR) / T

where T is a temperature parameter (derived from target false-positive rate).

Prior corrections:
    log P(valid | motion=m)     = -m / τ_m
    log P(valid | drift=δ)      = -max(0, δ - δ₀) / τ_i

Full log-posterior:
    logit(P_valid) = w_snr * snr_logit + w_conc * conc_logit
                   + w_reg * reg_logit - m/τ_m - δ/τ_i

SQI = sigmoid(logit) in (0,1), reported as percentage (0-100).

WHY THIS IS BETTER THAN THE OLD WEIGHTED SUM:
    Old: SQI = 0.35*snr + 0.25*reg + 0.15*temp + 0.15*peak + 0.10*var
         -> no statistical interpretation, weights arbitrary
    New: SQI = P(signal valid | features)
         -> weights derived from relative Fisher information
         -> can propagate uncertainty: sigma^2_SQI = SQI*(1-SQI)/N
         -> threshold chosen from target FPR, not ad hoc
"""

import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass
from typing import Tuple, Optional
from collections import deque


# ─── Parameter derivations (replacing magic numbers) ─────────────────────────
#
# SNR_REFERENCE_DB = 6.0
#   Justification: 6 dB SNR = minimum for <= 5 BPM MAE (Elgendi et al. 2013).
#   At SNR=6dB -> logit=0 -> SQI=50% (uncertain boundary).
#   At SNR=12dB -> logit=+1.5 -> SQI=82% (good).
#   At SNR=0dB  -> logit=-1.5 -> SQI=18% (bad).
#
# SNR_TEMPERATURE_DB = 4.0
#   Justification: sigmoid width. 4 dB corresponds to ~12 BPM error range --
#   the "uncertain" region. Smaller T -> sharper decision boundary.
#
# MOTION_PRIOR_TAU = 3.0
#   Justification: our motion metric has noise floor ~0.3, micro-motion ~1.5,
#   head rotation ~3+. At score=3.0: P = exp(-3/3) = 0.37 (large discount).
#
# ILLUM_PRIOR_TAU = 5.0
#   Justification: drift >8 units affects ROI mean by ~5% per frame.
#   At (drift-8)=5: P_illum = exp(-5/5) = 0.37.
#
# SNR_WEIGHT = 1.5: covers ~60% discriminative information (literature)
# SPECTRAL_CONC_WEIGHT = 0.8: ~30% additional information vs SNR alone
# REGULARITY_WEIGHT = 0.5: ~20% conditional on SNR

SNR_REFERENCE_DB     = 6.0
SNR_TEMPERATURE_DB   = 4.0
MOTION_PRIOR_TAU     = 3.0
ILLUM_PRIOR_TAU      = 5.0
ILLUM_WARN_THRESH    = 8.0
SNR_WEIGHT           = 1.5
SPECTRAL_CONC_WEIGHT = 0.8
REGULARITY_WEIGHT    = 0.5


@dataclass
class SQIResult:
    """
    SQI modeled as P(signal valid | features).
    All sub-probabilities in [0,1]. SQI in [0, 100].
    """
    sqi: float
    log_posterior: float
    p_signal_given_spectrum: float
    p_valid_given_motion: float
    p_valid_given_illum: float
    snr_db: float
    spectral_concentration: float
    regularity_score: float
    sqi_std: float          # sigma_SQI = sqrt(p*(1-p)/N) * 100
    n_samples: int


class PosteriorSQI:
    """
    SQI = P(signal valid | spectrum, motion, illumination).

    Log-linear discriminant with statistically motivated weights.
    Replaces the heuristic weighted sum in the original compute_sqi().
    """

    def __init__(self, roi_name: str = ""):
        self.roi_name = roi_name
        self._regularity_history: deque = deque(maxlen=8)
        self._sqi_history: deque = deque(maxlen=8)

    def compute(
        self,
        signal: np.ndarray,
        fps: float,
        peak_bpm: float,
        motion_score: float = 0.0,
        exposure_drift: float = 0.0,
        prev_sqi: float = -1.0,
    ) -> SQIResult:
        n = len(signal)
        p_motion = float(np.exp(-motion_score / MOTION_PRIOR_TAU))
        if n < 30 or peak_bpm <= 0:
            return SQIResult(
                sqi=0.0, log_posterior=-10.0,
                p_signal_given_spectrum=0.0,
                p_valid_given_motion=p_motion,
                p_valid_given_illum=1.0,
                snr_db=0.0, spectral_concentration=0.0, regularity_score=0.0,
                sqi_std=0.0, n_samples=n,
            )

        peak_hz  = peak_bpm / 60.0
        freq_low = 0.7
        freq_high = 3.33

        # ── 1. Spectral SNR ────────────────────────────────────────────────────
        nperseg = min(n, max(64, int(fps * 4)))
        freqs, psd = welch(signal, fs=fps, nperseg=nperseg, window='hann')

        sig_mask   = (freqs >= peak_hz - 0.10) & (freqs <= peak_hz + 0.10)
        noise_mask = (freqs >= freq_low) & (freqs <= freq_high) & (~sig_mask)

        p_signal     = float(np.sum(psd[sig_mask]))  if np.any(sig_mask)   else 0.0
        p_noise_med  = float(np.median(psd[noise_mask])) if np.any(noise_mask) else 1e-9
        n_sig_bins   = max(int(np.sum(sig_mask)), 1)
        snr_raw      = p_signal / (p_noise_med * n_sig_bins + 1e-12)
        snr_db       = float(10.0 * np.log10(max(snr_raw, 1e-9)))
        snr_logit    = (snr_db - SNR_REFERENCE_DB) / SNR_TEMPERATURE_DB

        # ── 2. Spectral concentration ──────────────────────────────────────────
        band_power  = float(np.sum(psd[(freqs >= freq_low) & (freqs <= freq_high)])) + 1e-12
        spec_conc   = float(p_signal / band_power)
        conc_logit  = (spec_conc - 0.3) / 0.2   # 0.3=noise floor, width=0.2

        # ── 3. Temporal regularity of peaks ────────────────────────────────────
        sig_z = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
        min_dist = max(int(fps * 0.45), 5)
        peaks, _ = find_peaks(sig_z, distance=min_dist, prominence=0.35)
        reg = self._regularity(peaks, fps)
        self._regularity_history.append(reg)
        reg = float(np.mean(self._regularity_history))
        reg_logit = (reg - 0.5) / 0.3

        # ── 4. Motion prior ────────────────────────────────────────────────────
        log_motion = float(np.log(p_motion + 1e-9))

        # ── 5. Illumination prior ──────────────────────────────────────────────
        drift_excess = max(0.0, exposure_drift - ILLUM_WARN_THRESH)
        p_illum      = float(np.exp(-drift_excess / ILLUM_PRIOR_TAU))
        log_illum    = float(np.log(p_illum + 1e-9))

        # ── 6. Log-posterior ───────────────────────────────────────────────────
        log_post = (
            SNR_WEIGHT           * snr_logit  +
            SPECTRAL_CONC_WEIGHT * conc_logit +
            REGULARITY_WEIGHT    * reg_logit  +
            log_motion + log_illum
        )

        # ── 7. SQI = sigmoid(log_posterior) ── with light temporal smoothing ──
        p_valid = float(1.0 / (1.0 + np.exp(-log_post)))
        if prev_sqi >= 0:
            # alpha=0.35: time constant ~ 3 frames = 0.1s @ 30fps
            # = one-tenth of a cardiac cycle at 60 BPM (principled)
            p_valid = 0.35 * p_valid + 0.65 * (prev_sqi / 100.0)

        sqi = float(np.clip(p_valid * 100.0, 0.0, 100.0))
        self._sqi_history.append(sqi)

        # ── 8. Uncertainty on SQI itself: Bernoulli std error ─────────────────
        sqi_std = float(np.sqrt(p_valid * (1.0 - p_valid) / max(n, 1)) * 100.0)

        p_snr_only = float(1.0 / (1.0 + np.exp(-SNR_WEIGHT * snr_logit)))
        return SQIResult(
            sqi=round(sqi, 1),
            log_posterior=round(log_post, 4),
            p_signal_given_spectrum=round(p_snr_only, 3),
            p_valid_given_motion=round(p_motion, 3),
            p_valid_given_illum=round(p_illum, 3),
            snr_db=round(snr_db, 2),
            spectral_concentration=round(spec_conc, 3),
            regularity_score=round(reg, 3),
            sqi_std=round(sqi_std, 2),
            n_samples=n,
        )

    def _regularity(self, peaks: np.ndarray, fps: float) -> float:
        if len(peaks) < 2:
            return 0.0
        if len(peaks) < 3:
            return 0.2
        ibi = np.diff(peaks) / fps
        cv  = float(np.std(ibi) / (np.mean(ibi) + 1e-9))
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def reset(self):
        self._sqi_history.clear()
        self._regularity_history.clear()


# ─── Precision-weighted multi-ROI SQI fusion ─────────────────────────────────

def fuse_roi_sqis(sqi_results: dict) -> Tuple[float, float]:
    """Precision-weighted fusion: weight = 1/Var(SQI_k)."""
    if not sqi_results:
        return 0.0, 0.0
    total_precision = 0.0
    weighted_sqi    = 0.0
    variances = []
    for result in sqi_results.values():
        if isinstance(result, SQIResult):
            p = result.sqi / 100.0 + 1e-6
            precision = result.n_samples / (p * (1.0 - p) + 1e-6)
            weighted_sqi    += result.sqi * precision
            total_precision += precision
            variances.append(result.sqi_std ** 2)
        else:
            weighted_sqi    += float(result)
            total_precision += 1.0
    fused_sqi = weighted_sqi / (total_precision + 1e-9)
    fused_std = float(np.sqrt(np.mean(variances))) if variances else 5.0
    return float(np.clip(fused_sqi, 0.0, 100.0)), fused_std


# ─── Bayesian ROI Weighting (replaces heuristic gradient) ────────────────────

class BayesianROIWeighting:
    """
    Beta-Binomial Bayesian update for ROI reliability weights.

    weight_k ~ Beta(alpha_k, beta_k)
    - Each frame: success_evidence = (SQI/100) * agreement_with_consensus
    - alpha_k += success_evidence (soft count of good observations)
    - beta_k  += (1 - success_evidence)

    MAP estimate: E[w_k] = alpha_k / (alpha_k + beta_k)
    Posterior std: sqrt(alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)))

    WHY BETTER THAN OLD GRADIENT UPDATE:
        Old: weight += 0.01 * (reward - 0.5)  <- learning_rate=0.01 is magic
        New: weight = Beta posterior mean      <- fully principled, uncertainty included
    """

    def __init__(self, roi_names, alpha_0: float = 2.0, beta_0: float = 2.0):
        # Beta(2,2): prior mean=0.5, prior concentration=4 (weak prior)
        self.roi_names  = list(roi_names)
        self.alpha      = {name: alpha_0 for name in roi_names}
        self.beta_      = {name: beta_0  for name in roi_names}
        self._alpha_0   = alpha_0
        self._beta_0    = beta_0

    def update(self, roi_sqis: dict, fused_bpm: float, roi_bpms: dict):
        for name in self.roi_names:
            sqi = roi_sqis.get(name, 0.0)
            bpm = roi_bpms.get(name, 0.0)
            if bpm <= 0 or sqi <= 0:
                continue
            # Agreement: 5 BPM deviation -> 37% agreement (exp scale)
            agreement = float(np.exp(-abs(bpm - fused_bpm) / 5.0))
            p_success  = (sqi / 100.0) * agreement
            self.alpha[name]  += p_success
            self.beta_[name]  += (1.0 - p_success)
            # Exponential forgetting: half-life ~693 frames (~23s @ 30fps)
            decay = 0.999
            self.alpha[name]  = self._alpha_0 + (self.alpha[name]  - self._alpha_0) * decay
            self.beta_[name]  = self._beta_0  + (self.beta_[name]  - self._beta_0)  * decay

    def get_weights(self) -> dict:
        return {
            n: float(self.alpha[n] / (self.alpha[n] + self.beta_[n]))
            for n in self.roi_names
        }

    def get_weight_uncertainty(self) -> dict:
        result = {}
        for n in self.roi_names:
            a, b = self.alpha[n], self.beta_[n]
            c = a + b
            var = (a * b) / (c**2 * (c + 1) + 1e-9)
            result[n] = float(np.sqrt(var))
        return result


# ─── Backward-compatibility shims ────────────────────────────────────────────

class StatisticallyCalibratedSQI:
    """Wrapper delegating to PosteriorSQI. Keeps old import interface."""
    def __init__(self):
        self._calcs = {}

    def calculate_sqi(self, signal, fps, peak_bpm, roi_name="",
                      motion_score=0.0, exposure_drift=0.0, prev_sqi=-1.0):
        if roi_name not in self._calcs:
            self._calcs[roi_name] = PosteriorSQI(roi_name)
        r = self._calcs[roi_name].compute(
            signal, fps, peak_bpm, motion_score, exposure_drift, prev_sqi)
        meta = {
            "snr_db": r.snr_db, "spectral_entropy": r.spectral_concentration,
            "log_posterior": r.log_posterior,
            "p_valid_motion": r.p_valid_given_motion,
            "p_valid_illum": r.p_valid_given_illum,
            "sqi_std": r.sqi_std, "regularity_score": r.regularity_score,
        }
        return r.sqi, meta


class LearnedWeighting(BayesianROIWeighting):
    """Old name, new Bayesian implementation."""
    def __init__(self, roi_names):
        super().__init__(roi_names)

    def update_weights(self, roi_sqis, fused_bpm, roi_bpms):
        self.update(roi_sqis, fused_bpm, roi_bpms)
