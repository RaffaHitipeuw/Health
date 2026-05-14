


import numpy as np
from collections import deque
from dataclasses import dataclass, field
from scipy.signal import welch, butter, sosfilt
from enum import Enum


class PhysioState(Enum):
    RESTING = "RESTING"
    ACTIVE  = "ACTIVE"
    PANTING = "PANTING"


@dataclass
class PhysioStateResult:
    state:              PhysioState  = PhysioState.RESTING
    resp_rate_hz:       float        = 0.0
    resp_rate_bpm:      float        = 0.0
    resp_power_ratio:   float        = 0.0
    cardiac_fraction:   float        = 1.0
    hr_prior_low_bpm:   float        = 50.0
    hr_prior_high_bpm:  float        = 180.0
    confidence:         float        = 0.0
    phase_separation:   dict         = field(default_factory=dict)


class PhysiologicalStateClassifier:


    ACTIVE_ENTER_HZ  = 0.35
    ACTIVE_EXIT_HZ   = 0.28
    PANTING_ENTER_HZ = 0.85
    PANTING_EXIT_HZ  = 0.70


    PANTING_HR_PRIOR_LOW  = 75.0
    PANTING_HR_PRIOR_HIGH = 180.0

    def __init__(self, fps: float = 30.0, history_sec: float = 10.0):
        self.fps = fps
        self._window  = max(30, int(fps * history_sec))
        self._state   = PhysioState.RESTING
        self._resp_hz_history: deque = deque(maxlen=15)
        self._motion_history:  deque = deque(maxlen=30)

    def update(
        self,
        signal: np.ndarray,
        fps: float,
        motion_score: float = 0.0,
        roi_signals: dict   = None,
    ) -> PhysioStateResult:


        self.fps = fps
        self._motion_history.append(motion_score)
        n = len(signal)
        if n < max(30, int(fps * 3)):
            return PhysioStateResult()


        resp_hz, resp_ratio = self._estimate_resp_rate(signal, fps)
        if resp_hz > 0:
            self._resp_hz_history.append(resp_hz)
        smoothed_resp_hz = float(np.median(self._resp_hz_history)) if self._resp_hz_history else 0.0


        self._state = self._transition(self._state, smoothed_resp_hz)


        hr_low, hr_high = 50.0, 180.0
        if self._state == PhysioState.PANTING:
            hr_low  = self.PANTING_HR_PRIOR_LOW
            hr_high = self.PANTING_HR_PRIOR_HIGH


        phase_sep = {}
        cardiac_fraction = 1.0
        if roi_signals and len(roi_signals) >= 2 and self._state != PhysioState.RESTING:
            phase_sep = self._phase_separation(roi_signals, fps, smoothed_resp_hz)
            cardiac_fraction = phase_sep.get("cardiac_fraction", 1.0)


        confidence = min(len(self._resp_hz_history) / 10.0, 1.0)

        return PhysioStateResult(
            state              = self._state,
            resp_rate_hz       = round(smoothed_resp_hz, 3),
            resp_rate_bpm      = round(smoothed_resp_hz * 60.0, 1),
            resp_power_ratio   = round(resp_ratio, 3),
            cardiac_fraction   = round(cardiac_fraction, 3),
            hr_prior_low_bpm   = hr_low,
            hr_prior_high_bpm  = hr_high,
            confidence         = round(confidence, 3),
            phase_separation   = phase_sep,
        )

    def _estimate_resp_rate(self, signal: np.ndarray, fps: float):


        n = len(signal)
        nperseg = min(n, max(int(fps * 10), 64))

        freqs, psd = welch(signal, fs=fps, nperseg=nperseg, window='hann')


        resp_mask  = (freqs >= 0.10) & (freqs <= 1.50)
        pulse_mask = (freqs >= 0.83) & (freqs <= 3.00)

        if not np.any(resp_mask):
            return 0.0, 0.0


        resp_psd   = psd[resp_mask]
        resp_freqs = freqs[resp_mask]
        peak_idx   = np.argmax(resp_psd)
        resp_hz    = float(resp_freqs[peak_idx])

        resp_power  = float(np.sum(resp_psd))
        pulse_power = float(np.sum(psd[pulse_mask])) if np.any(pulse_mask) else 1e-9
        ratio       = resp_power / (pulse_power + 1e-9)

        return resp_hz, ratio

    def _transition(self, current: PhysioState, resp_hz: float) -> PhysioState:

        if current == PhysioState.RESTING:
            if resp_hz >= self.ACTIVE_ENTER_HZ:
                return PhysioState.ACTIVE
        elif current == PhysioState.ACTIVE:
            if resp_hz >= self.PANTING_ENTER_HZ:
                return PhysioState.PANTING
            if resp_hz < self.ACTIVE_EXIT_HZ:
                return PhysioState.RESTING
        elif current == PhysioState.PANTING:
            if resp_hz < self.PANTING_EXIT_HZ:
                return PhysioState.ACTIVE
        return current

    def _phase_separation(
        self,
        roi_signals: dict,
        fps: float,
        resp_hz: float,
    ) -> dict:


        names = list(roi_signals.keys())
        if "forehead" not in roi_signals or len(names) < 2:
            return {"cardiac_fraction": 0.8, "method": "insufficient_rois"}

        sig_fore = np.asarray(roi_signals["forehead"], dtype=float)

        other_name = next((n for n in names if n != "forehead"), None)
        if other_name is None:
            return {"cardiac_fraction": 0.8, "method": "no_second_roi"}
        sig_other = np.asarray(roi_signals[other_name], dtype=float)


        n = min(len(sig_fore), len(sig_other))
        if n < 30:
            return {"cardiac_fraction": 0.8, "method": "too_short"}

        sig_fore  = sig_fore[-n:]
        sig_other = sig_other[-n:]


        F = np.fft.rfft(sig_fore  * np.hanning(n))
        C = np.fft.rfft(sig_other * np.hanning(n))
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        cross = F * np.conj(C)


        if resp_hz > 0:
            resp_idx = int(np.argmin(np.abs(freqs - resp_hz)))
            phase_at_resp = float(np.angle(cross[resp_idx]))
        else:
            phase_at_resp = 0.0


        tau_cardiac_sec = 0.050
        cardiac_freqs = freqs[(freqs >= 1.0) & (freqs <= 3.0)]
        if len(cardiac_freqs) == 0:
            return {"cardiac_fraction": 0.5, "method": "no_cardiac_band"}


        cardiac_scores = []
        for f in cardiac_freqs:
            idx = int(np.argmin(np.abs(freqs - f)))
            observed_phase = float(np.angle(cross[idx]))
            expected_phase = 2.0 * np.pi * f * tau_cardiac_sec

            score = (np.cos(observed_phase - expected_phase) + 1.0) / 2.0
            power = float(np.abs(cross[idx]))
            cardiac_scores.append(score * power)


        resp_cardiac_score = (np.cos(phase_at_resp) + 1.0) / 2.0


        total_cardiac_evidence = sum(cardiac_scores)
        total_evidence = total_cardiac_evidence + resp_cardiac_score * 10
        cardiac_fraction = float(np.clip(
            total_cardiac_evidence / (total_evidence + 1e-9), 0.0, 1.0
        ))

        return {
            "cardiac_fraction": round(cardiac_fraction, 3),
            "resp_phase_diff":  round(phase_at_resp, 3),
            "method":           "cross_spectrum_phase",
            "tau_assumed_ms":   50,
        }

    def reset(self):
        self._state = PhysioState.RESTING
        self._resp_hz_history.clear()
        self._motion_history.clear()


class PantingAdaptiveHR:


    def __init__(self):
        self._panting_frames = 0
        self._bpm_memory: deque = deque(maxlen=20)

    def adjust_fft_weights(
        self,
        fft_v: np.ndarray,
        freqs: np.ndarray,
        mask: np.ndarray,
        state_result: PhysioStateResult,
    ) -> np.ndarray:


        adjusted = fft_v.copy()
        if state_result.state != PhysioState.PANTING:
            return adjusted
        if state_result.resp_rate_hz <= 0:
            return adjusted

        rh = state_result.resp_rate_hz
        half_bw = 0.15


        resp_zone = (freqs >= rh - half_bw) & (freqs <= rh + half_bw) & mask
        adjusted[resp_zone] *= 0.35


        resp2_zone = (freqs >= 2*rh - half_bw) & (freqs <= 2*rh + half_bw) & mask
        adjusted[resp2_zone] *= 0.50


        cardiac_boost = np.where(
            mask,
            np.clip(0.6 + 0.4 * (freqs / 1.5), 0.6, 1.4),
            1.0
        )
        adjusted *= cardiac_boost

        return adjusted

    def update(self, bpm: float, state: PhysioState):
        if bpm > 0:
            self._bpm_memory.append(bpm)
        if state == PhysioState.PANTING:
            self._panting_frames += 1
        else:
            self._panting_frames = max(0, self._panting_frames - 1)

    def is_panting(self) -> bool:
        return self._panting_frames > 5

    def reset(self):
        self._panting_frames = 0
        self._bpm_memory.clear()
