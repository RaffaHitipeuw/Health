
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, correlate
import os

def debug_signal(signal, fps, roi_name="ROI", output_path="debug_plot.png"):
    """
    Plots Raw Signal, Detrended, Bandpassed, FFT Spectrum, and Autocorrelation.
    """
    n = len(signal)
    t = np.arange(n) / fps
    
    # 1. Detrend & Normalize
    sig_detrend = signal - np.mean(signal)
    
    # 2. FFT
    nperseg = min(n, max(64, int(fps * 4)))
    freqs, psd = welch(sig_detrend, fs=fps, nperseg=nperseg, window='hann')
    
    # 3. Autocorrelation
    # Normalize signal for ACF
    sig_norm = (sig_detrend - np.mean(sig_detrend)) / (np.std(sig_detrend) + 1e-9)
    acf = correlate(sig_norm, sig_norm, mode='full')[n-1:]
    acf = acf / acf[0] # Normalize
    lags = np.arange(len(acf)) / fps
    
    # Find ACF peaks
    min_dist = max(int(fps * 0.45), 5) # ~133 BPM max
    peaks, _ = find_peaks(acf, distance=min_dist, prominence=0.1)
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    # Plot Time Domain
    axes[0].plot(t, sig_detrend, label='Detrended Signal')
    axes[0].set_title(f"Time Domain Signal - {roi_name}")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend()
    
    # Plot FFT
    axes[1].plot(freqs, psd, color='red', label='PSD')
    axes[1].set_xlim(0.5, 4.0)
    axes[1].set_title("FFT Spectrum (Power Spectral Density)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].legend()
    
    # Plot Autocorrelation
    axes[2].plot(lags, acf, color='green', label='Autocorrelation')
    if len(peaks) > 0:
        axes[2].plot(lags[peaks], acf[peaks], "x", color='black', label='ACF Peaks')
    axes[2].set_xlim(0, 3.0)
    axes[2].set_title("Autocorrelation (ACF)")
    axes[2].set_xlabel("Lag (s)")
    axes[2].legend()
    
    # Plot Histogram of Signal
    axes[3].hist(sig_detrend, bins=30, alpha=0.7, color='purple')
    axes[3].set_title("Signal Distribution (Histogram)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Debug plot saved to {output_path}")

if __name__ == "__main__":
    # Create dummy signal for testing: 72 BPM (1.2 Hz) + noise
    fs = 30.0
    duration = 10.0
    t = np.arange(int(fs * duration)) / fs
    clean_signal = np.sin(2 * np.pi * 1.2 * t)
    noisy_signal = clean_signal + 0.5 * np.random.normal(size=len(t))
    
    debug_signal(noisy_signal, fs, "Test_72BPM", "/home/ubuntu/rppg_project/test_debug_plot.png")
