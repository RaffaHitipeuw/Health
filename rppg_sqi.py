


import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass
from typing import Tuple, Optional
from collections import deque


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


    sqi: float
    log_posterior: float
    p_signal_given_spectrum: float
    p_valid_given_motion: float
    p_valid_given_illum: float
    snr_db: float
    spectral_concentration: float
    regularity_score: float
    sqi_std: float
    n_samples: int


class PosteriorSQI:


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


        band_power  = float(np.sum(psd[(freqs >= freq_low) & (freqs <= freq_high)])) + 1e-12
        spec_conc   = float(p_signal / band_power)
        conc_logit  = (spec_conc - 0.3) / 0.2


        sig_z = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
        min_dist = max(int(fps * 0.45), 5)
        peaks, _ = find_peaks(sig_z, distance=min_dist, prominence=0.35)
        reg = self._regularity(peaks, fps, signal=sig_z)
        self._regularity_history.append(reg)
        reg = float(np.mean(self._regularity_history))
        reg_logit = (reg - 0.5) / 0.3


        log_motion = float(np.log(p_motion + 1e-9))


        drift_excess = max(0.0, exposure_drift - ILLUM_WARN_THRESH)
        p_illum      = float(np.exp(-drift_excess / ILLUM_PRIOR_TAU))
        log_illum    = float(np.log(p_illum + 1e-9))


        log_bio = 0.0
        # Petunjuk: HARD REJECT BPM > 130
        if peak_bpm > 130:
            log_bio = -10.0 # Extreme penalty (effective kill)
        elif peak_bpm > 115 and motion_score < 1.0:
            log_bio = -5.0   # Strong prior against resting tachycardia


        if self.roi_name == "forehead" and peak_bpm > 125:
            # Forehead at high BPM is textbook harmonic zombie territory
            log_bio -= 5.0   # Was -4.0 — nuclear double penalty

            


        log_peak = 0.0
        if len(psd) > 0:
            sorted_psd = np.sort(psd[noise_mask])
            if len(sorted_psd) > 2:
                peak_ratio = p_signal / (np.sum(sorted_psd[-3:]) + 1e-9)
                if peak_ratio < 1.5:
                    log_peak = -1.0

        # Petunjuk: TAMBAH TEMPORAL INERTIA
        log_temporal = 0.0
        if len(self._sqi_history) > 0 and hasattr(self, '_last_bpm'):
            delta_bpm = abs(peak_bpm - self._last_bpm)
            if delta_bpm > 25:
                log_temporal = -2.5 # Significant penalty for teleporting BPM
        self._last_bpm = peak_bpm

        log_post = (
            SNR_WEIGHT           * snr_logit  +
            SPECTRAL_CONC_WEIGHT * conc_logit +
            REGULARITY_WEIGHT    * reg_logit  +
            log_motion + log_illum + log_bio + log_peak + log_temporal
        )


        p_valid = float(1.0 / (1.0 + np.exp(-log_post)))
        if prev_sqi >= 0:
            # Only blend with history when the current estimate is not severely penalized.
            # If log_bio is very negative (harmonic/BPM prior fired hard), bypass blending
            # so the zombie can't maintain SQI through historical momentum.
            if log_bio < -4.0:
                pass   # No blending — hard penalty lands immediately
            else:
                p_valid = 0.35 * p_valid + 0.65 * (prev_sqi / 100.0)

        sqi = float(np.clip(p_valid * 100.0, 0.0, 100.0))
        self._sqi_history.append(sqi)


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

    def _regularity(self, peaks: np.ndarray, fps: float, signal: np.ndarray = None) -> float:
        """
        Overhauled Regularity Engine:
        1. Bandpass Filtering (Physiological Range)
        2. Z-score Normalization
        3. Lag-limited Autocorrelation (42-180 BPM)
        4. Smooth Scoring (No binary gating)
        """
        # 1. Peak-to-Peak Consistency (Time Domain)
        reg_peaks = 0.0
        if len(peaks) >= 3:
            ibi = np.diff(peaks) / fps
            cv  = float(np.std(ibi) / (np.mean(ibi) + 1e-9))
            # Smooth CV scoring: Humans are messy, allow some variance
            reg_peaks = float(np.clip(1.2 - (cv / 0.25), 0.0, 1.0))
        elif len(peaks) == 2:
            reg_peaks = 0.3

        # 2. Autocorrelation Periodicity (Waveform Consistency)
        reg_acf = 0.0
        if signal is not None and len(signal) >= 30:
            from scipy.signal import correlate, butter, filtfilt
            
            # Bandpass BEFORE regularity: 0.6 Hz - 4.0 Hz (Wider to preserve morphology)
            # Lower order (2 instead of 4) to avoid over-smoothing/phase distortion
            try:
                b, a = butter(2, [0.6 / (fps / 2), 4.0 / (fps / 2)], btype='band')
                sig_clean = filtfilt(b, a, signal)
            except:
                sig_clean = signal
            
            # Z-score Normalize waveform
            sig_norm = (sig_clean - np.mean(sig_clean)) / (np.std(sig_clean) + 1e-9)
            
            n = len(sig_norm)
            acf = correlate(sig_norm, sig_norm, mode='full')[n-1:]
            acf = acf / (acf[0] + 1e-12) # Normalize by zero-lag
            
            # Find the second peak in physiological lag (40-190 BPM)
            lag_min = int(fps * 60 / 190) # ~0.31s
            lag_max = int(fps * 60 / 40)  # ~1.50s
            
            if len(acf) > lag_min:
                # Limit search to physiological range
                acf_search = acf[lag_min:min(len(acf), lag_max)]
                if len(acf_search) > 0:
                    # Instead of just max, look for a local maximum (peak)
                    # This prevents picking a value from a descending slope
                    from scipy.signal import find_peaks as scipy_find_peaks
                    peaks, props = scipy_find_peaks(acf_search, height=0.1)
                    if len(peaks) > 0:
                        acf_peak = float(np.max(props['peak_heights']))
                    else:
                        acf_peak = float(np.max(acf_search)) * 0.5 # Penalty for no clear peak
                    
                    # More generous scoring: (acf_peak - 0.15) / 0.5
                    # Biological signals are messy, don't expect 0.8+ ACF
                    reg_acf = float(np.clip((acf_peak - 0.15) / 0.5, 0.0, 1.0))
        
        # Combined Regularity: Prioritize ACF for waveform shape, CV for timing
        combined_reg = 0.7 * reg_acf + 0.3 * reg_peaks
        return float(np.clip(combined_reg, 0.0, 1.0))

    def reset(self):
        self._sqi_history.clear()
        self._regularity_history.clear()


def fuse_roi_sqis(sqi_results: dict) -> Tuple[float, float]:

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


class BayesianROIWeighting:


    def __init__(self, roi_names, alpha_0: float = 2.0, beta_0: float = 2.0):

        self.roi_names  = list(roi_names)
        self.alpha      = {name: alpha_0 for name in roi_names}
        self.beta_      = {name: beta_0  for name in roi_names}
        self._alpha_0   = alpha_0
        self._beta_0    = beta_0

    def update(self, roi_sqis: dict, fused_bpm: float, roi_bpms: dict, roi_signals: dict = None):


        for name in self.roi_names:
            sqi = roi_sqis.get(name, 0.0)
            bpm = roi_bpms.get(name, 0.0)
            if bpm <= 0 or sqi <= 0:
                continue
                

            # ── BUG FIX: Agreement Engine (Petunjuk) ───────────────────────────────
            # Formula lama: exp(-diff / 5) -> Masih terlalu permisif.
            # Formula baru: Exponential penalty (np.exp(-diff / 12)).
            # Petunjuk menyarankan np.exp(-diff / 12) untuk agreement brutal.
            bpm_agreement = float(np.exp(-abs(bpm - fused_bpm) / 12.0))
            
            # ── Subharmonic / Half-Frequency Lock (Petunjuk) ───────────────────────
            # If ROI BPM is ~0.5x of fused BPM, it's a subharmonic artifact.
            # Petunjuk: 0.45 < bpm / stable < 0.55 -> penalty *= 0.4
            if fused_bpm > 0:
                ratio = bpm / fused_bpm
                if 0.45 < ratio < 0.55:
                    bpm_agreement *= 0.4
            


            physio_validity = 1.0
            if bpm > 130:
                if sqi < 80:
                    physio_validity *= 0.5
                if name == "forehead":
                    # ── Forehead High BPM Penalty (Petunjuk) ───────────────────────────
                    # Forehead at high BPM is often a zombie lock (harmonic).
                    # Petunjuk: sqi *= 0.3 atau sqi *= 0.2
                    physio_validity *= 0.15 # Even more brutal for Bayesian update
            

            p_success  = (sqi / 100.0) * bpm_agreement * physio_validity
            
            self.alpha[name]  += p_success
            self.beta_[name]  += (1.0 - p_success)

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


class StatisticallyCalibratedSQI:

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

    def __init__(self, roi_names):
        super().__init__(roi_names)

    def update_weights(self, roi_sqis, fused_bpm, roi_bpms):
        self.update(roi_sqis, fused_bpm, roi_bpms)
