
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, correlate, butter, filtfilt
import os

def debug_signal(raw_signal, fps, roi_name="ROI", output_path="debug_plot.png"):
    """
    Plots the 4 MANDATORY plots per ROI:
    1. Raw Signal
    2. Detrended Signal
    3. Filtered Signal (Bandpassed)
    4. Autocorrelation Curve
    """
    n = len(raw_signal)
    t = np.arange(n) / fps
    
    # 1. Detrended
    sig_detrend = raw_signal - np.mean(raw_signal)
    
    # 2. Filtered (Bandpass 0.6 - 4.0 Hz)
    try:
        b, a = butter(2, [0.6 / (fps / 2), 4.0 / (fps / 2)], btype='band')
        sig_filtered = filtfilt(b, a, sig_detrend)
    except:
        sig_filtered = sig_detrend
    
    # 3. Autocorrelation
    # Normalize filtered signal for ACF
    sig_norm = (sig_filtered - np.mean(sig_filtered)) / (np.std(sig_filtered) + 1e-9)
    acf = correlate(sig_norm, sig_norm, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-12)
    lags = np.arange(len(acf)) / fps
    
    # Find ACF peaks in physiological range (40-190 BPM)
    lag_min = int(fps * 60 / 190)
    lag_max = int(fps * 60 / 40)
    
    acf_search = acf[lag_min:min(len(acf), lag_max)]
    peaks = []
    if len(acf_search) > 0:
        rel_peaks, props = find_peaks(acf_search, height=0.1)
        peaks = rel_peaks + lag_min
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    # Plot 1: Raw Signal
    axes[0].plot(t, raw_signal, color='gray', label='Raw Signal')
    axes[0].set_title(f"1. Raw Signal - {roi_name}")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend()
    
    # Plot 2: Detrended
    axes[1].plot(t, sig_detrend, color='blue', label='Detrended')
    axes[1].set_title("2. Detrended Signal")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    
    # Plot 3: Filtered
    axes[2].plot(t, sig_filtered, color='green', label='Filtered (0.6-4.0Hz)')
    axes[2].set_title("3. Filtered Signal (Bandpass)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    
    # Plot 4: Autocorrelation
    axes[3].plot(lags, acf, color='red', label='Autocorrelation')
    if len(peaks) > 0:
        axes[3].plot(lags[peaks], acf[peaks], "x", color='black', label='ACF Peaks')
    axes[3].set_xlim(0, 2.5)
    axes[3].set_ylim(-0.5, 1.1)
    axes[3].axhline(0.15, color='black', linestyle='--', alpha=0.3, label='Threshold')
    axes[3].set_title("4. Autocorrelation Curve (ACF)")
    axes[3].set_xlabel("Lag (s)")
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Debug plot saved to {output_path}")

if __name__ == "__main__":
    # Smoke test
    fs = 30.0
    duration = 10.0
    t = np.arange(int(fs * duration)) / fs
    # 72 BPM signal with some drift and noise
    raw = np.sin(2 * np.pi * 1.2 * t) + 0.05 * t + 0.3 * np.random.normal(size=len(t))
    debug_signal(raw, fs, "Test_72BPM", "/home/ubuntu/rppg_project/test_debug_plot.png")
