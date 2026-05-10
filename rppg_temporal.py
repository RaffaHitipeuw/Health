import numpy as np
from collections import deque

class KalmanBPMFilter:
    """
    A more rigorous Kalman Filter for BPM tracking.
    State: [BPM, Velocity]
    """
    def __init__(self, q_bpm=0.1, q_vel=0.01, r_bpm=5.0):
        # State vector [bpm, velocity]
        self.x = np.array([75.0, 0.0])
        # State covariance
        self.P = np.eye(2) * 10.0
        # Process noise covariance
        self.Q = np.array([[q_bpm, 0], [0, q_vel]])
        # Measurement noise covariance
        self.R = np.array([[r_bpm]])
        # State transition matrix (assuming dt=1 for simplicity, or update dt)
        self.F = np.array([[1, 1], [0, 1]])
        # Measurement matrix
        self.H = np.array([[1, 0]])
        
        self.initialized = False

    def update(self, z_bpm, dt=1.0):
        if not self.initialized:
            self.x[0] = z_bpm
            self.initialized = True
            return z_bpm
        
        # Update F with dt
        self.F[0, 1] = dt
        
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = z_bpm - (self.H @ self.x) # Innovation
        S = self.H @ self.P @ self.H.T + self.R # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
        return float(self.x[0])

class BayesianBPMTracker:
    """
    Probabilistic state estimation for BPM.
    Uses a discrete grid to represent the probability distribution of BPM.
    """
    def __init__(self, bpm_min=40, bpm_max=180, resolution=0.5):
        self.bpm_range = np.arange(bpm_min, bpm_max + resolution, resolution)
        self.prior = np.ones_like(self.bpm_range) / len(self.bpm_range)
        self.resolution = resolution
        
    def update(self, measured_bpm, measured_sqi):
        if measured_bpm <= 0:
            return float(np.dot(self.bpm_range, self.prior))
        
        # Likelihood: Gaussian centered at measured_bpm with sigma inversely proportional to SQI
        sigma = 20.0 * (1.0 - measured_sqi / 100.0) + 2.0
        likelihood = np.exp(-0.5 * ((self.bpm_range - measured_bpm) / sigma)**2)
        likelihood /= (np.sum(likelihood) + 1e-9)
        
        # Posterior
        posterior = self.prior * likelihood
        posterior /= (np.sum(posterior) + 1e-9)
        
        # Prediction step (diffusion/drift)
        # Transition kernel: Gaussian blur to represent uncertainty over time
        kernel_sigma = 1.0 # BPM change per frame
        kernel = np.exp(-0.5 * (np.arange(-10, 10, self.resolution) / kernel_sigma)**2)
        kernel /= np.sum(kernel)
        
        self.prior = np.convolve(posterior, kernel, mode='same')
        self.prior /= (np.sum(self.prior) + 1e-9)
        
        # MAP or Mean estimate
        estimate = np.dot(self.bpm_range, posterior)
        return float(estimate)

class ProbabilisticFusion:
    """
    Adaptive probabilistic fusion of multiple ROI estimates.
    """
    def __init__(self):
        self.tracker = BayesianBPMTracker()
        
    def fuse(self, roi_estimates):
        """
        roi_estimates: list of (bpm, sqi)
        """
        if not roi_estimates:
            return 0.0
        
        # Combine likelihoods from all ROIs
        combined_likelihood = np.ones_like(self.tracker.bpm_range)
        
        for bpm, sqi in roi_estimates:
            if bpm <= 0 or sqi <= 0:
                continue
            sigma = 15.0 * (1.0 - sqi / 100.0) + 1.0
            likelihood = np.exp(-0.5 * ((self.tracker.bpm_range - bpm) / sigma)**2)
            combined_likelihood *= (likelihood + 1e-9)
            
        combined_likelihood /= (np.sum(combined_likelihood) + 1e-9)
        
        # Update tracker with combined likelihood
        # (Manually doing the update here to handle multiple ROIs)
        posterior = self.tracker.prior * combined_likelihood
        posterior /= (np.sum(posterior) + 1e-9)
        
        # Prediction step for next frame
        kernel_sigma = 0.5
        kernel = np.exp(-0.5 * (np.arange(-5, 5, self.tracker.resolution) / kernel_sigma)**2)
        kernel /= np.sum(kernel)
        self.tracker.prior = np.convolve(posterior, kernel, mode='same')
        self.tracker.prior /= (np.sum(self.tracker.prior) + 1e-9)
        
        return float(np.dot(self.tracker.bpm_range, posterior))
