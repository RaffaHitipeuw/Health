


import numpy as np
from collections import deque
from typing import Optional


class KalmanBPMFilter:


    def __init__(self, q_bpm: float = 1.0, q_vel: float = 0.01, r_bpm: float = 25.0):
        self.q_bpm = q_bpm
        self.q_vel = q_vel
        self.r_bpm = r_bpm


        self.x = np.array([75.0, 0.0])
        self.P = np.diag([100.0, 1.0])


        self.Q = np.diag([q_bpm, q_vel])
        self.H = np.array([[1.0, 0.0]])

        self.initialized = False

    def update(self, z_bpm: float, dt: float = 1.0,
               sqi: Optional[float] = None) -> float:


        if not self.initialized:
            self.x[0] = z_bpm
            self.initialized = True
            return z_bpm


        r_floor = 4.0
        t = np.clip((sqi or 50.0) / 100.0, 0.0, 1.0)
        R_effective = r_floor + (self.r_bpm - r_floor) * (1.0 - t)
        R = np.array([[R_effective]])


        F = np.array([[1.0, dt], [0.0, 1.0]])


        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + self.Q


        innovation = z_bpm - x_pred[0]
        S = float(P_pred[0, 0]) + R_effective
        K = P_pred[:, 0] / S

        self.x = x_pred + K * innovation
        self.P = (np.eye(2) - np.outer(K, self.H)) @ P_pred


        self.x[0] = float(np.clip(self.x[0], 42.0, 200.0))

        return float(self.x[0])

    @property
    def sigma_bpm(self) -> float:


        return float(np.sqrt(max(self.P[0, 0], 0.01)))

    @property
    def posterior_covariance(self) -> np.ndarray:

        return self.P.copy()

    def reset(self):
        self.x = np.array([75.0, 0.0])
        self.P = np.diag([100.0, 1.0])
        self.initialized = False


class BayesianBPMTracker:


    def __init__(
        self,
        bpm_min:    float = 40.0,
        bpm_max:    float = 200.0,
        resolution: float = 0.5,
        diffusion:  float = 1.0,
    ):


        self.bpm_range  = np.arange(bpm_min, bpm_max + resolution, resolution)
        self.resolution = resolution
        self.diffusion  = diffusion


        K = len(self.bpm_range)
        self.prior = np.ones(K) / K


        half_k = min(20, K // 4)
        kernel_x = np.arange(-half_k, half_k + 1) * resolution
        self.kernel = np.exp(-0.5 * (kernel_x / diffusion)**2)
        self.kernel /= self.kernel.sum()

        self._n_updates = 0

    def update(self, measured_bpm: float, sqi: float) -> float:


        if measured_bpm <= 0:

            self._predict()
            return float(np.dot(self.bpm_range, self.prior))

        sigma = 15.0 * (1.0 - np.clip(sqi / 100.0, 0.0, 1.0)) + 2.0


        likelihood = np.exp(-0.5 * ((self.bpm_range - measured_bpm) / sigma)**2)
        likelihood /= likelihood.sum() + 1e-12


        posterior = self.prior * likelihood
        posterior /= posterior.sum() + 1e-12


        self.prior = self._predict(posterior)
        self._n_updates += 1


        return float(np.dot(self.bpm_range, posterior))

    def _predict(self, posterior: Optional[np.ndarray] = None) -> np.ndarray:

        if posterior is None:
            posterior = self.prior
        propagated = np.convolve(posterior, self.kernel, mode='same')
        propagated /= propagated.sum() + 1e-12
        return propagated

    @property
    def posterior_std(self) -> float:

        mean = float(np.dot(self.bpm_range, self.prior))
        var  = float(np.dot((self.bpm_range - mean)**2, self.prior))
        return float(np.sqrt(max(var, 0.0)))

    @property
    def map_estimate(self) -> float:

        return float(self.bpm_range[np.argmax(self.prior)])

    @property
    def posterior_mean(self) -> float:

        return float(np.dot(self.bpm_range, self.prior))

    def credible_interval(self, confidence: float = 0.95) -> tuple:


        cdf = np.cumsum(self.prior)
        alpha = (1.0 - confidence) / 2.0
        lower_idx = int(np.searchsorted(cdf, alpha))
        upper_idx = int(np.searchsorted(cdf, 1.0 - alpha))
        lower_idx = max(0, min(lower_idx, len(self.bpm_range) - 1))
        upper_idx = max(0, min(upper_idx, len(self.bpm_range) - 1))
        return (float(self.bpm_range[lower_idx]),
                float(self.bpm_range[upper_idx]))

    def reset(self):
        K = len(self.bpm_range)
        self.prior = np.ones(K) / K
        self._n_updates = 0


class ProbabilisticFusion:


    def __init__(self):
        self.tracker = BayesianBPMTracker()

    def fuse(self, roi_estimates) -> float:


        if not roi_estimates:
            return 0.0


        log_joint = np.zeros_like(self.tracker.bpm_range)

        for bpm, sqi in roi_estimates:
            if bpm <= 0 or sqi < 0:
                continue

            sigma = 15.0 * (1.0 - np.clip(sqi / 100.0, 0.0, 1.0)) + 1.0
            log_lik = -0.5 * ((self.tracker.bpm_range - bpm) / sigma)**2
            log_joint += log_lik


        log_joint -= log_joint.max()
        joint_likelihood = np.exp(log_joint)
        joint_likelihood /= joint_likelihood.sum() + 1e-12


        posterior = self.tracker.prior * joint_likelihood
        posterior /= posterior.sum() + 1e-12


        self.tracker.prior = self.tracker._predict(posterior)

        return float(np.dot(self.tracker.bpm_range, posterior))

    def reset(self):
        self.tracker.reset()
