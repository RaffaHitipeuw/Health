import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import entropy

class StatisticallyCalibratedSQI:
    """
    SQI calibrated using statistical properties of the signal.
    """
    def __init__(self):
        # Baseline statistics for calibration (could be loaded from a file)
        self.snr_mu = 5.0
        self.snr_sigma = 2.0
        self.reg_mu = 0.8
        self.reg_sigma = 0.1
        
    def calculate_snr(self, signal, fps, peak_bpm):
        """Calculate SNR in dB and normalize."""
        n = len(signal)
        freqs, psd = welch(signal, fs=fps, nperseg=n)
        
        peak_hz = peak_bpm / 60.0
        # Signal band
        sig_mask = (freqs >= peak_hz - 0.1) & (freqs <= peak_hz + 0.1)
        # Noise band (within pulse range but outside signal band)
        noise_mask = (freqs >= 0.7) & (freqs <= 4.0) & (~sig_mask)
        
        p_sig = np.sum(psd[sig_mask])
        p_noise = np.sum(psd[noise_mask])
        
        snr = 10 * np.log10(p_sig / (p_noise + 1e-9))
        # Sigmoid normalization based on calibrated mu/sigma
        norm_snr = 1.0 / (1.0 + np.exp(-(snr - self.snr_mu) / self.snr_sigma))
        return norm_snr, snr

    def calculate_spectral_entropy(self, signal, fps):
        """Calculate spectral entropy as a measure of signal cleanliness."""
        n = len(signal)
        freqs, psd = welch(signal, fs=fps, nperseg=n)
        psd_norm = psd / np.sum(psd)
        se = entropy(psd_norm)
        # Normalize: lower entropy is better (more peaked)
        # Max entropy for N bins is log(N)
        max_se = np.log(len(psd_norm))
        norm_se = 1.0 - (se / max_se)
        return norm_se

    def calculate_sqi(self, signal, fps, peak_bpm):
        snr_norm, snr_db = self.calculate_snr(signal, fps, peak_bpm)
        se_norm = self.calculate_spectral_entropy(signal, fps)
        
        # Combined SQI
        sqi = 0.6 * snr_norm + 0.4 * se_norm
        return sqi * 100.0, {"snr_db": snr_db, "spectral_entropy": se_norm}

class LearnedWeighting:
    """
    Learned weighting for ROI fusion based on historical performance.
    """
    def __init__(self, roi_names):
        self.roi_weights = {name: 1.0 for name in roi_names}
        self.learning_rate = 0.01
        
    def update_weights(self, roi_sqis, fused_bpm, roi_bpms):
        """
        Update weights based on how close each ROI was to the consensus/fused BPM,
        weighted by their SQI.
        """
        for name, sqi in roi_sqis.items():
            bpm = roi_bpms.get(name, 0)
            if bpm <= 0: continue
            
            # Error relative to fused result
            error = abs(bpm - fused_bpm)
            # Reward ROIs that are close to consensus and have high SQI
            reward = (sqi / 100.0) * np.exp(-error / 5.0)
            
            # Update weight
            self.roi_weights[name] += self.learning_rate * (reward - 0.5)
            # Clamp weights
            self.roi_weights[name] = np.clip(self.roi_weights[name], 0.1, 5.0)
            
    def get_weights(self):
        return self.roi_weights
