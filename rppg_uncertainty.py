


import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from collections import deque


@dataclass
class HeartRateEstimate:


    mean_bpm: float
    std_bpm: float
    ci_95_lower: float
    ci_95_upper: float
    ci_width: float


    aleatoric_std: float
    epistemic_std: float


    confidence_pct: float
    n_frames_used: int
    sqi_at_estimate: float

    def __str__(self):
        return (
            f"HR = {self.mean_bpm:.1f} ± {self.std_bpm:.1f} BPM "
            f"(95% CI: {self.ci_95_lower:.1f}–{self.ci_95_upper:.1f}) "
            f"[aleatoric: ±{self.aleatoric_std:.1f}, epistemic: ±{self.epistemic_std:.1f}]"
        )

    def is_reliable(self, max_ci_width: float = 20.0) -> bool:

        return self.ci_width <= max_ci_width and self.confidence_pct >= 40.0

    def as_dict(self) -> dict:
        return {
            "mean_bpm": round(self.mean_bpm, 1),
            "std_bpm": round(self.std_bpm, 1),
            "ci_95": [round(self.ci_95_lower, 1), round(self.ci_95_upper, 1)],
            "ci_width": round(self.ci_width, 1),
            "aleatoric_std": round(self.aleatoric_std, 1),
            "epistemic_std": round(self.epistemic_std, 1),
            "confidence_pct": round(self.confidence_pct, 1),
        }


class BayesianHREstimator:


    def __init__(
        self,
        q_bpm:    float = 1.0,
        q_vel:    float = 0.01,
        r_base:   float = 25.0,
        r_floor:  float = 4.0,
        window:   int   = 90,
    ):


        self.q_bpm  = q_bpm
        self.q_vel  = q_vel
        self.r_base = r_base
        self.r_floor = r_floor


        self.x = np.array([75.0, 0.0])
        self.P = np.diag([100.0, 1.0])


        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.Q = np.diag([q_bpm, q_vel])
        self.H = np.array([[1.0, 0.0]])

        self.initialized = False
        self._n_updates  = 0


        self._bpm_history:  deque = deque(maxlen=window)
        self._sqi_history:  deque = deque(maxlen=window)
        self._roi_agree_history: deque = deque(maxlen=window)

    def _measurement_noise(self, sqi: float) -> float:


        t = np.clip(sqi / 100.0, 0.0, 1.0)
        return float(self.r_floor + (self.r_base - self.r_floor) * (1.0 - t))

    def update(self, measured_bpm: float, sqi: float,
               roi_agreement: float = 1.0) -> HeartRateEstimate:


        if measured_bpm <= 0:
            return self._make_estimate(sqi, roi_agreement)

        if not self.initialized:
            self.x[0] = measured_bpm
            self.initialized = True

        R = self._measurement_noise(sqi)


        agreement_penalty = max(1.0, 1.0 / (roi_agreement + 0.1))
        R_effective = float(R * agreement_penalty)


        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q


        innovation = measured_bpm - x_pred[0]
        S = float(P_pred[0, 0]) + R_effective
        K = P_pred[:, 0] / S

        self.x = x_pred + K * innovation
        self.P = (np.eye(2) - np.outer(K, self.H))


        self.x[0] = float(np.clip(self.x[0], 42.0, 200.0))

        self._n_updates += 1
        self._bpm_history.append(self.x[0])
        self._sqi_history.append(sqi)
        self._roi_agree_history.append(roi_agreement)

        return self._make_estimate(sqi, roi_agreement, R_effective)

    def _make_estimate(self, sqi: float, roi_agreement: float,
                       R_effective: float = None) -> HeartRateEstimate:

        mean_bpm = float(self.x[0])


        posterior_var = float(self.P[0, 0])
        posterior_std = float(np.sqrt(max(posterior_var, 0.01)))


        R_now = R_effective if R_effective is not None else self._measurement_noise(sqi)
        aleatoric_var = float(R_now)
        aleatoric_std = float(np.sqrt(aleatoric_var))


        epistemic_var = max(0.0, posterior_var - aleatoric_var)


        if len(self._roi_agree_history) >= 5:
            mean_agree = float(np.mean(self._roi_agree_history))

            disagreement_var = (1.0 - mean_agree) * 25.0
            epistemic_var += disagreement_var

        epistemic_std = float(np.sqrt(max(epistemic_var, 0.0)))


        total_std = float(np.sqrt(aleatoric_var + epistemic_var))
        total_std = max(total_std, 0.5)


        ci_lower = mean_bpm - 1.96 * total_std
        ci_upper = mean_bpm + 1.96 * total_std


        sqi_term   = sqi / 100.0
        agree_term = roi_agreement
        ci_term    = float(np.exp(-(ci_upper - ci_lower) / 30.0))
        confidence = float(np.cbrt(sqi_term * agree_term * ci_term) * 100.0)


        return HeartRateEstimate(
            mean_bpm=round(mean_bpm, 1),
            std_bpm=round(total_std, 2),
            ci_95_lower=round(max(ci_lower, 42.0), 1),
            ci_95_upper=round(min(ci_upper, 200.0), 1),
            ci_width=round(min(ci_upper, 200.0) - max(ci_lower, 42.0), 1),
            aleatoric_std=round(aleatoric_std, 2),
            epistemic_std=round(epistemic_std, 2),
            confidence_pct=round(float(np.clip(confidence, 0.0, 100.0)), 1),
            n_frames_used=self._n_updates,
            sqi_at_estimate=round(sqi, 1),
        )

    def reset(self):
        self.x = np.array([75.0, 0.0])
        self.P = np.diag([100.0, 1.0])
        self.initialized = False
        self._n_updates = 0
        self._bpm_history.clear()
        self._sqi_history.clear()
        self._roi_agree_history.clear()


def propagate_variance(roi_variances, weights) -> Tuple[float, float]:


    weights_arr  = np.array(weights, dtype=float)
    variance_arr = np.array(roi_variances, dtype=float)


    valid = (variance_arr > 0) & (weights_arr > 0)
    if not np.any(valid):
        return 100.0, 10.0

    precisions = 1.0 / variance_arr[valid]
    total_precision = float(np.sum(precisions))
    fused_variance  = 1.0 / total_precision
    fused_std       = float(np.sqrt(fused_variance))
    return fused_variance, fused_std


class UncertaintyAwareConfidence:


    def __init__(self, window_size: int = 60):
        self._estimator = BayesianHREstimator()
        self._last_estimate: Optional[HeartRateEstimate] = None
        self._window_size = window_size

    def update(self, bpm: float, sqi: float, agreement: float):
        if bpm > 0:
            self._last_estimate = self._estimator.update(bpm, sqi, agreement)

    def get_confidence_metrics(self) -> Tuple[float, dict]:

        if self._last_estimate is None:
            return 0.0, {}
        est = self._last_estimate
        return est.confidence_pct, {
            "aleatoric_uncertainty": est.aleatoric_std,
            "epistemic_uncertainty": est.epistemic_std,
            "total_uncertainty":     est.std_bpm,
            "confidence_interval_95": est.ci_width,
            "ci_lower": est.ci_95_lower,
            "ci_upper": est.ci_95_upper,
        }

    def get_estimate(self) -> Optional[HeartRateEstimate]:
        return self._last_estimate

    def reset(self):
        self._estimator.reset()
        self._last_estimate = None
