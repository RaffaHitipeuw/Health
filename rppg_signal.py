import numpy as np
from scipy.signal import welch, find_peaks, windows

def apply_windowing(signal, window_type='hann'):
    """Apply anti-leakage windowing."""
    if window_type == 'hann':
        win = windows.hann(len(signal))
    elif window_type == 'hamming':
        win = windows.hamming(len(signal))
    elif window_type == 'blackman':
        win = windows.blackman(len(signal))
    else:
        win = np.ones(len(signal))
    return signal * win

def mature_harmonic_rejection(psd, freqs, peak_hz, threshold=0.15):
    """
    Detect and reject harmonic peaks.
    """
    # Check for 2nd and 3rd harmonics
    harmonics = [2 * peak_hz, 3 * peak_hz]
    is_harmonic = False
    
    for h in harmonics:
        if h > freqs[-1]: continue
        
        # Find peak near harmonic
        h_mask = (freqs >= h - 0.1) & (freqs <= h + 0.1)
        if not np.any(h_mask): continue
        
        h_peak = np.max(psd[h_mask])
        main_peak = np.max(psd[(freqs >= peak_hz - 0.1) & (freqs <= peak_hz + 0.1)])
        
        # If harmonic is too strong relative to main peak, it might be a false main peak
        # or the signal is just very harmonic.
        # In rPPG, sometimes the 2nd harmonic is stronger than the fundamental.
        if h_peak > threshold * main_peak:
            # This is a heuristic: if we find a strong sub-harmonic, 
            # the current peak might actually be the 2nd harmonic.
            sub_h = peak_hz / 2.0
            if sub_h >= 0.7: # Still in pulse range
                sub_mask = (freqs >= sub_h - 0.1) & (freqs <= sub_h + 0.1)
                if np.any(sub_mask):
                    sub_peak = np.max(psd[sub_mask])
                    if sub_peak > 0.3 * main_peak:
                        is_harmonic = True
                        break
    return is_harmonic

def respiratory_interference_analysis(signal, fps):
    """
    Analyze if the signal is dominated by respiratory interference.
    Respiratory rate is usually 0.1 - 0.5 Hz (6-30 BPM).
    """
    n = len(signal)
    freqs, psd = welch(signal, fs=fps, nperseg=n)
    
    resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
    pulse_mask = (freqs >= 0.7) & (freqs <= 3.0)
    
    if not np.any(resp_mask) or not np.any(pulse_mask):
        return 0.0
        
    resp_power = np.sum(psd[resp_mask])
    pulse_power = np.sum(psd[pulse_mask])
    
    # Ratio of respiratory power to pulse power
    ratio = resp_power / (pulse_power + 1e-9)
    return float(ratio)
