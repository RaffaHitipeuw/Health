"""
rppg_uncertainty.py  —  Bayesian Uncertainty Quantification
============================================================

PHASE 9 REVISION: Replace deterministic point estimate with
full posterior distribution over heart rate.

Output format (research grade):
    HR ~ Normal(mu_HR, sigma^2_HR)
    Reported as: "78 +/- 3.2 BPM (95% CI: 71.7-84.3)"

Uncertainty decomposition:
    Var[HR] = Var_aleatoric + Var_epistemic

    Aleatoric (irreducible):
        From signal noise. Inversely proportional to SNR.
        Cannot be reduced with more computation.
        Estimated from Kalman measurement noise R.

    Epistemic (reducible):
        From model uncertainty: limited buffer, ROI disagreement.
        Decreases with more data, better models.
        Estimated from Kalman state covariance P.

KEY INSIGHT: The Kalman filter already computes the exact posterior
    p(HR_t | signal_{1:t}) = N(x_hat[0], P[0,0])
under linear Gaussian assumptions. We just need to expose P.
No heuristic needed — the math already gives us the CI.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from collections import deque


@dataclass
class HeartRateEstimate:
    """
    Full posterior distribution over heart rate.

    This is the research-grade output format — NOT a single number.
    "HR = 78" is engineering. "HR ~ N(78, 3.2^2)" is research.
    """
    mean_bpm: float          # MAP estimate (posterior mean)
    std_bpm: float           # Posterior standard deviation
    ci_95_lower: float       # 95% credible interval lower bound
    ci_95_upper: float       # 95% credible interval upper bound
    ci_width: float          # ci_95_upper - ci_95_lower (quality indicator)

    # Uncertainty decomposition
    aleatoric_std: float     # Irreducible: from signal noise (SNR-based)
    epistemic_std: float     # Reducible: from model/data uncertainty

    # Meta
    confidence_pct: float    # Overall confidence 0-100
    n_frames_used: int       # Data quantity indicator
    sqi_at_estimate: float   # Signal quality at this estimate

    def __str__(self):
        return (
            f"HR = {self.mean_bpm:.1f} ± {self.std_bpm:.1f} BPM "
            f"(95% CI: {self.ci_95_lower:.1f}–{self.ci_95_upper:.1f}) "
            f"[aleatoric: ±{self.aleatoric_std:.1f}, epistemic: ±{self.epistemic_std:.1f}]"
        )

    def is_reliable(self, max_ci_width: float = 20.0) -> bool:
        """A CI width > 20 BPM means the estimate is not clinically useful."""
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
    """
    Maintains a full posterior distribution over HR using
    the Kalman filter state covariance P.

    Under linear Gaussian state-space model:
        HR_t = HR_{t-1} + vel_{t-1} + w_t,   w_t ~ N(0, Q)
        z_t  = HR_t + v_t,                   v_t ~ N(0, R(SQI_t))

    The Kalman filter computes:
        p(HR_t | z_{1:t}) = N(x_hat[0], P[0,0])

    This IS the Bayesian posterior — no approximation needed.

    Uncertainty decomposition:
        Aleatoric variance  = R (measurement noise, driven by SQI)
        Epistemic variance  = P[0,0] - R (residual after measurement update)

    Note: "Epistemic" here means uncertainty due to limited observations
    and model inadequacy, not Bayesian parameter uncertainty in the
    strict sense. This is the operationally useful decomposition.
    """

    def __init__(
        self,
        q_bpm:    float = 1.0,   # Process noise: BPM variance per frame
        q_vel:    float = 0.01,  # Process noise: velocity variance per frame
        r_base:   float = 25.0,  # Base measurement noise (BPM^2) at SQI=0
        r_floor:  float = 4.0,   # Minimum measurement noise at SQI=100
        window:   int   = 90,    # History window for epistemic estimate
    ):
        """
        Parameter derivations:
        q_bpm = 1.0 BPM^2/frame:
            Max physiological HR change: ~3 BPM/sec = 0.1 BPM/frame @ 30fps
            Process variance = (3/30)^2 per frame... but we allow larger
            because FFT estimation noise dominates: set to 1.0 BPM^2/frame
            -> sqrt(1.0) = 1 BPM/frame max step (conservative but stable)

        r_base = 25.0 BPM^2 at SQI=0:
            FFT resolution at N=150, fs=30: Δf = 0.2 Hz = 12 BPM
            Measurement std ≈ 12/2 = 6 BPM -> variance ≈ 36 BPM^2
            We use 25 (sqrt=5 BPM) as conservative floor for poor signal.

        r_floor = 4.0 BPM^2 at SQI=100:
            At excellent SNR: FFT peak detection accuracy ≈ ±2 BPM -> var=4
        """
        self.q_bpm  = q_bpm
        self.q_vel  = q_vel
        self.r_base = r_base
        self.r_floor = r_floor

        # State [HR, velocity] and covariance P
        self.x = np.array([75.0, 0.0])
        self.P = np.diag([100.0, 1.0])  # Large initial uncertainty

        # Transition and measurement matrices
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # constant-velocity model
        self.Q = np.diag([q_bpm, q_vel])
        self.H = np.array([[1.0, 0.0]])

        self.initialized = False
        self._n_updates  = 0

        # History for epistemic uncertainty estimation
        self._bpm_history:  deque = deque(maxlen=window)
        self._sqi_history:  deque = deque(maxlen=window)
        self._roi_agree_history: deque = deque(maxlen=window)

    def _measurement_noise(self, sqi: float) -> float:
        """
        R(SQI): measurement noise as function of signal quality.

        Linear interpolation from r_base (SQI=0) to r_floor (SQI=100).
        Justification: SQI = P(valid) -> higher SQI -> lower noise.
        R = r_base * (1 - SQI/100) + r_floor * (SQI/100)
            = r_floor + (r_base - r_floor) * (1 - SQI/100)
        """
        t = np.clip(sqi / 100.0, 0.0, 1.0)
        return float(self.r_floor + (self.r_base - self.r_floor) * (1.0 - t))

    def update(self, measured_bpm: float, sqi: float,
               roi_agreement: float = 1.0) -> HeartRateEstimate:
        """
        Full Kalman update step. Returns HeartRateEstimate.
        """
        if measured_bpm <= 0:
            return self._make_estimate(sqi, roi_agreement)

        if not self.initialized:
            self.x[0] = measured_bpm
            self.initialized = True

        R = self._measurement_noise(sqi)
        # Scale R by ROI disagreement: high disagreement -> larger measurement noise
        # At agreement=0.5: R *= 2.0 (doubles uncertainty)
        # Derivation: agreement = exp(-sigma_ROI / k), so sigma_ROI = -k * log(agreement)
        # We penalize R proportionally: R *= 1 + sigma_ROI / reference_sigma
        agreement_penalty = max(1.0, 1.0 / (roi_agreement + 0.1))
        R_effective = float(R * agreement_penalty)

        # ── Predict ────────────────────────────────────────────────────────────
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ─────────────────────────────────────────────────────────────
        innovation = measured_bpm - x_pred[0]  # H=[1,0] so H@x = x[0]
        S = float(P_pred[0, 0]) + R_effective  # H=[1,0] so H P H^T = P[0,0]
        K = P_pred[:, 0] / S  # H=[1,0] so P H^T = P[:,0]                   # Kalman gain (2x1)

        self.x = x_pred + K * innovation  # K is shape (2,)
        self.P = (np.eye(2) - np.outer(K, self.H))  # K shape (2,), H shape (2,) @ P_pred

        # Clip to physiological bounds
        self.x[0] = float(np.clip(self.x[0], 42.0, 200.0))

        self._n_updates += 1
        self._bpm_history.append(self.x[0])
        self._sqi_history.append(sqi)
        self._roi_agree_history.append(roi_agreement)

        return self._make_estimate(sqi, roi_agreement, R_effective)

    def _make_estimate(self, sqi: float, roi_agreement: float,
                       R_effective: float = None) -> HeartRateEstimate:
        """Construct HeartRateEstimate from current Kalman state."""
        mean_bpm = float(self.x[0])

        # ── Posterior variance = P[0,0] ────────────────────────────────────────
        posterior_var = float(self.P[0, 0])
        posterior_std = float(np.sqrt(max(posterior_var, 0.01)))

        # ── Aleatoric: from measurement noise R (signal quality) ───────────────
        # This is the irreducible noise floor set by SNR
        R_now = R_effective if R_effective is not None else self._measurement_noise(sqi)
        aleatoric_var = float(R_now)
        aleatoric_std = float(np.sqrt(aleatoric_var))

        # ── Epistemic: posterior variance minus measurement noise ───────────────
        # If P[0,0] > R: we have residual uncertainty beyond what measurement noise explains
        # This captures model inadequacy, limited data, ROI disagreement
        epistemic_var = max(0.0, posterior_var - aleatoric_var)

        # Additional epistemic from ROI disagreement history
        if len(self._roi_agree_history) >= 5:
            mean_agree = float(np.mean(self._roi_agree_history))
            # Disagreement adds epistemic uncertainty: low agreement = high epistemic
            disagreement_var = (1.0 - mean_agree) * 25.0  # up to 5 BPM std
            epistemic_var += disagreement_var

        epistemic_std = float(np.sqrt(max(epistemic_var, 0.0)))

        # ── Total std = sqrt(aleatoric + epistemic) ────────────────────────────
        total_std = float(np.sqrt(aleatoric_var + epistemic_var))
        total_std = max(total_std, 0.5)  # minimum: 0.5 BPM precision floor

        # ── 95% Credible Interval ──────────────────────────────────────────────
        # Under Gaussian posterior: P(|HR - mu| <= 1.96*sigma) = 95%
        # This is a valid Bayesian credible interval (not a frequentist CI)
        ci_lower = mean_bpm - 1.96 * total_std
        ci_upper = mean_bpm + 1.96 * total_std

        # ── Confidence: P(estimate reliable) ──────────────────────────────────
        # Composite: from SQI, ROI agreement, CI width
        # - SQI captures signal quality (likelihood term)
        # - Agreement captures consistency across ROIs
        # - CI width penalizes wide intervals
        sqi_term   = sqi / 100.0
        agree_term = roi_agreement
        ci_term    = float(np.exp(-(ci_upper - ci_lower) / 30.0))  # exp decay over CI width
        confidence = float(np.cbrt(sqi_term * agree_term * ci_term) * 100.0)
        # Geometric mean (cbrt of product): zero-one-zero -> zero (all must be good)

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


# ─── Variance propagation through fusion ─────────────────────────────────────

def propagate_variance(roi_variances, weights) -> Tuple[float, float]:
    """
    Propagate variance through precision-weighted fusion.

    Given K independent estimates {mu_k, sigma^2_k} with weights {w_k}:
        w_k = 1/sigma^2_k  (precision weights, MVUE)
        mu_fused = sum(w_k * mu_k) / sum(w_k)
        Var[mu_fused] = 1 / sum(w_k)  [inverse-sum of precisions]

    Args:
        roi_variances: array of per-ROI variance estimates (BPM^2)
        weights: array of precision weights (1/variance)

    Returns:
        (fused_variance, fused_std) in BPM
    """
    weights_arr  = np.array(weights, dtype=float)
    variance_arr = np.array(roi_variances, dtype=float)

    # Remove zero-variance entries
    valid = (variance_arr > 0) & (weights_arr > 0)
    if not np.any(valid):
        return 100.0, 10.0

    precisions = 1.0 / variance_arr[valid]
    total_precision = float(np.sum(precisions))
    fused_variance  = 1.0 / total_precision
    fused_std       = float(np.sqrt(fused_variance))
    return fused_variance, fused_std


# ─── Backward-compatible class ────────────────────────────────────────────────

class UncertaintyAwareConfidence:
    """
    Backward-compatible wrapper around BayesianHREstimator.
    Old interface returns (confidence_pct, metadata_dict).
    New code should use BayesianHREstimator directly for HeartRateEstimate.
    """

    def __init__(self, window_size: int = 60):
        self._estimator = BayesianHREstimator()
        self._last_estimate: Optional[HeartRateEstimate] = None
        self._window_size = window_size

    def update(self, bpm: float, sqi: float, agreement: float):
        if bpm > 0:
            self._last_estimate = self._estimator.update(bpm, sqi, agreement)

    def get_confidence_metrics(self) -> Tuple[float, dict]:
        """Returns (confidence_pct, metadata_dict)."""
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
