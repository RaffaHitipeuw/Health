import numpy as np
from scipy.signal import detrend
from sklearn.decomposition import FastICA

def extract_green(r, g, b):
    """Standard GREEN algorithm: Simple green channel averaging."""
    return detrend(g)

def extract_chrom(r, g, b):
    """
    CHROM: Chrominance-based rPPG.
    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG.
    """
    # Normalize
    rn = r / (np.mean(r) + 1e-9)
    gn = g / (np.mean(g) + 1e-9)
    bn = b / (np.mean(b) + 1e-9)
    
    # Chrominance signals
    xs = 3 * rn - 2 * gn
    ys = 1.5 * rn + gn - 1.5 * bn
    
    # Bandpass/Detrending usually happens outside, but we return the raw projection
    # alpha = std(xs) / std(ys)
    alpha = np.std(xs) / (np.std(ys) + 1e-9)
    return xs - alpha * ys

def extract_pos(r, g, b):
    """
    POS: Plane-Orthogonal-to-Skin.
    Wang, W., et al. (2017). Algorithmic Principles of Remote PPG.
    """
    # Step 1: Temporal normalization
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    
    rn = r / (mean_r + 1e-9)
    gn = g / (mean_g + 1e-9)
    bn = b / (mean_b + 1e-9)
    
    # Step 2: Projection
    # S = [rn, gn, bn]^T * P
    # P is the projection matrix
    s1 = gn - bn
    s2 = gn + bn - 2 * rn
    
    # Step 3: Alpha tuning
    alpha = np.std(s1) / (np.std(s2) + 1e-9)
    
    # Step 4: Final signal
    h = s1 + alpha * s2
    return h

def extract_ica(r, g, b):
    """
    ICA: Independent Component Analysis based rPPG.
    Poh, M. Z., et al. (2010). Non-contact, automated cardiac pulse measurements.
    """
    # Stack channels
    X = np.stack([r, g, b], axis=1)
    
    # Normalize
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
    
    try:
        ica = FastICA(n_components=3, max_iter=1000)
        S = ica.fit_transform(X)
        
        # Select the component with the highest periodicity in the pulse range
        # This is a simplified selection: pick the one with highest peak in FFT
        best_comp = 0
        max_peak = -1
        
        for i in range(3):
            comp = S[:, i]
            fft = np.abs(np.fft.rfft(comp * np.hanning(len(comp))))
            # We don't have FPS here, so we just look at the max peak in the whole spectrum
            # (excluding DC)
            peak = np.max(fft[1:])
            if peak > max_peak:
                max_peak = peak
                best_comp = i
        
        return S[:, best_comp]
    except Exception:
        # Fallback to Green if ICA fails
        return g - np.mean(g)

def get_algorithm_by_name(name):
    algos = {
        "GREEN": extract_green,
        "CHROM": extract_chrom,
        "POS": extract_pos,
        "ICA": extract_ica
    }
    return algos.get(name.upper(), extract_pos)
