


import numpy as np
import cv2
from scipy.signal import welch, find_peaks, windows, butter, sosfilt
from collections import deque


def apply_windowing(signal, window_type='hann'):

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


    harmonics = [2 * peak_hz, 3 * peak_hz]
    for h in harmonics:
        if h > freqs[-1]:
            continue
        h_mask = (freqs >= h - 0.1) & (freqs <= h + 0.1)
        if not np.any(h_mask):
            continue
        h_peak   = np.max(psd[h_mask])
        main_mask = (freqs >= peak_hz - 0.1) & (freqs <= peak_hz + 0.1)
        main_peak = np.max(psd[main_mask]) if np.any(main_mask) else 1e-9
        if h_peak > threshold * main_peak:
            sub_h = peak_hz / 2.0
            if sub_h >= 0.7:
                sub_mask = (freqs >= sub_h - 0.1) & (freqs <= sub_h + 0.1)
                if np.any(sub_mask) and np.max(psd[sub_mask]) > 0.3 * main_peak:
                    return True
    return False


def respiratory_interference_analysis(signal: np.ndarray, fps: float) -> float:


    n = len(signal)
    if n < 20:
        return 0.0
    nperseg = min(n, max(int(fps * 8), 64))
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, window='hann')


    resp_mask  = (freqs >= 0.10) & (freqs <= 1.50)
    pulse_mask = (freqs >= 0.83) & (freqs <= 3.00)

    if not np.any(resp_mask) or not np.any(pulse_mask):
        return 0.0

    resp_power  = float(np.sum(psd[resp_mask]))
    pulse_power = float(np.sum(psd[pulse_mask]))
    return float(resp_power / (pulse_power + 1e-9))


def estimate_respiratory_rate(signal: np.ndarray, fps: float) -> float:


    n = len(signal)
    if n < max(30, int(fps * 3)):
        return 0.0
    nperseg = min(n, max(int(fps * 10), 64))
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, window='hann')

    resp_mask = (freqs >= 0.10) & (freqs <= 1.50)
    if not np.any(resp_mask):
        return 0.0
    return float(freqs[resp_mask][np.argmax(psd[resp_mask])])


def extract_pos_with_phase(r: np.ndarray, g: np.ndarray, b: np.ndarray):


    mean_r = np.mean(r) + 1e-9
    mean_g = np.mean(g) + 1e-9
    mean_b = np.mean(b) + 1e-9
    rn = r / mean_r
    gn = g / mean_g
    bn = b / mean_b

    s1 = gn - bn
    s2 = gn + bn - 2.0 * rn

    sigma1 = np.std(s1) + 1e-9
    sigma2 = np.std(s2) + 1e-9
    alpha  = sigma1 / sigma2

    h = s1 + alpha * s2
    m = s1 - (1.0 / alpha) * s2

    return h, m, float(alpha), s1, s2


def phase_coherence_cardiac_fraction(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fps:   float,
    tau_cardiac_ms: float = 50.0,
) -> float:


    n = min(len(sig_a), len(sig_b))
    if n < 30:
        return 0.7

    sa = sig_a[-n:]
    sb = sig_b[-n:]

    win  = np.hanning(n)
    Fa   = np.fft.rfft(sa * win)
    Fb   = np.fft.rfft(sb * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    cross    = Fa * np.conj(Fb)
    phase    = np.angle(cross)
    power    = np.abs(cross)

    tau_sec  = tau_cardiac_ms / 1000.0


    eval_mask = (freqs >= 0.50) & (freqs <= 2.50)
    if not np.any(eval_mask):
        return 0.7

    freqs_e  = freqs[eval_mask]
    phase_e  = phase[eval_mask]
    power_e  = power[eval_mask]


    expected_cardiac = 2.0 * np.pi * freqs_e * tau_sec


    cardiac_sim  = (np.cos(phase_e - expected_cardiac) + 1.0) / 2.0
    resp_sim     = (np.cos(phase_e) + 1.0) / 2.0


    cardiac_score = float(np.sum(cardiac_sim * power_e))
    resp_score    = float(np.sum(resp_sim    * power_e))
    total         = cardiac_score + resp_score + 1e-9

    return float(np.clip(cardiac_score / total, 0.0, 1.0))


class OpticalFlowROIStabilizer:


    LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    FEATURE_PARAMS = dict(
        maxCorners=30,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    def __init__(self):
        self._prev_gray:    np.ndarray = None
        self._prev_pts:     np.ndarray = None
        self._ref_pts:      np.ndarray = None
        self._M_accum:      np.ndarray = np.eye(2, 3, dtype=np.float32)
        self._frame_count:  int = 0
        self._reset_every:  int = 90

    def stabilize(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple,
        mask: np.ndarray = None,
    ) -> np.ndarray:


        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return frame_bgr[y1:y2, x1:x2]

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        crop_gray = gray[y1:y2, x1:x2]
        self._frame_count += 1


        size_changed = False
        if self._prev_gray is not None:
            if self._prev_gray.shape != crop_gray.shape:
                size_changed = True

        if (self._prev_gray is None or size_changed or
                self._frame_count % self._reset_every == 1 or
                self._prev_pts is None or len(self._prev_pts) < 4):

            self._prev_gray = crop_gray.copy()
            self._prev_pts  = cv2.goodFeaturesToTrack(
                crop_gray, mask=mask, **self.FEATURE_PARAMS)
            self._ref_pts   = self._prev_pts.copy() if self._prev_pts is not None else None
            self._M_accum   = np.eye(2, 3, dtype=np.float32)
            return frame_bgr[y1:y2, x1:x2]

        if self._prev_pts is None or len(self._prev_pts) < 4:
            return frame_bgr[y1:y2, x1:x2]


        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, crop_gray, self._prev_pts, None, **self.LK_PARAMS)

        if curr_pts is None or status is None:
            return frame_bgr[y1:y2, x1:x2]

        good_prev = self._prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < 4:
            self._prev_gray = crop_gray.copy()
            return frame_bgr[y1:y2, x1:x2]


        M, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
        if M is None:
            self._prev_gray = crop_gray.copy()
            self._prev_pts  = curr_pts
            return frame_bgr[y1:y2, x1:x2]


        M_inv = np.eye(2, 3, dtype=np.float32)
        M_inv[:2, :2] = M[:2, :2].T
        M_inv[:2, 2]  = -M[:2, :2].T @ M[:2, 2]


        crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
        h_crop, w_crop = crop_bgr.shape[:2]
        stabilised = cv2.warpAffine(
            crop_bgr, M_inv, (w_crop, h_crop),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        self._prev_gray = crop_gray.copy()
        self._prev_pts  = good_curr.reshape(-1, 1, 2)

        return stabilised

    def reset(self):
        self._prev_gray   = None
        self._prev_pts    = None
        self._ref_pts     = None
        self._M_accum     = np.eye(2, 3, dtype=np.float32)
        self._frame_count = 0


class PulseRichPixelSelector:


    UPDATE_EVERY = 30

    def __init__(self, fps: float = 30.0):
        self.fps         = fps
        self._crop_buf:  deque = deque(maxlen=90)
        self._weight_map: np.ndarray = None
        self._frame_idx: int = 0

    def ingest(self, roi_crop_bgr: np.ndarray):

        if roi_crop_bgr is None or roi_crop_bgr.size == 0:
            return
        

        h, w = roi_crop_bgr.shape[:2]
        if len(self._crop_buf) > 0:
            prev_h, prev_w = self._crop_buf[0].shape
            if h != prev_h or w != prev_w:
                self._crop_buf.clear()
                self._weight_map = None

        g = roi_crop_bgr[:, :, 1].astype(np.float32)
        self._crop_buf.append(g)
        self._frame_idx += 1
        if self._frame_idx % self.UPDATE_EVERY == 0 and len(self._crop_buf) >= 30:
            self._update_weights()

    def _update_weights(self):

        crops = np.array(list(self._crop_buf), dtype=np.float32)
        T, H, W = crops.shape
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fps)

        cardiac_mask = (freqs >= 0.83) & (freqs <= 3.0)
        total_mask   = (freqs >= 0.10) & (freqs <= 4.0)


        fft_amp = np.abs(np.fft.rfft(crops, axis=0))

        cardiac_power = np.sum(fft_amp[cardiac_mask], axis=0) + 1e-9
        total_power   = np.sum(fft_amp[total_mask],   axis=0) + 1e-9
        spectral_purity = cardiac_power / total_power


        if H > 4 and W > 4:
            spectral_purity = cv2.GaussianBlur(
                spectral_purity.astype(np.float32), (5, 5), 0)

        self._weight_map = spectral_purity

    def weighted_mean_rgb(self, roi_crop_bgr: np.ndarray):


        if roi_crop_bgr is None or roi_crop_bgr.size == 0:
            return 0.0, 0.0, 0.0

        h, w = roi_crop_bgr.shape[:2]

        if (self._weight_map is None or
                self._weight_map.shape != (h, w)):

            means = cv2.mean(roi_crop_bgr)
            return means[2], means[1], means[0]

        w_map = self._weight_map
        w_sum = float(w_map.sum()) + 1e-9

        r_mean = float(np.sum(roi_crop_bgr[:, :, 2].astype(float) * w_map) / w_sum)
        g_mean = float(np.sum(roi_crop_bgr[:, :, 1].astype(float) * w_map) / w_sum)
        b_mean = float(np.sum(roi_crop_bgr[:, :, 0].astype(float) * w_map) / w_sum)
        return r_mean, g_mean, b_mean

    def has_weights(self) -> bool:
        return self._weight_map is not None

    def reset(self):
        self._crop_buf.clear()
        self._weight_map = None
        self._frame_idx  = 0
