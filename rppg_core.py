"""
rppg_core.py  -  Multi-ROI rPPG fusion engine  (ALL 12 FIXES APPLIED + SQI REBALANCING)
Fix #1  Hard dominance rules in dynamic weight (SNR/agreement kill switches)
Fix #2  Temporal+physiological truth-aware agreement
Fix #3  EMA + sliding-median temporal smoothing at decision layer
Fix #4  Rebalanced SQI component weights (SNR dominant, capped penalties)
Fix #5  Outlier ROI rejection BEFORE fusion (median-distance gate)
Fix #6  Exposure drift as hard weight penalty + output freeze on extreme drift
Fix #7  Session confidence rebuilt from SQI x consistency x agreement
Fix #8  Arrhythmia gated behind SQI >= 65
Fix #9  All constants centralised in rppg_config.py
Fix #10 Calibration phase (first N frames as baseline)
Fix #12 Hierarchical fusion: filter->cluster->fuse
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch, detrend as scipy_detrend, find_peaks, iirnotch
from scipy.stats import pearsonr
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import time

# Fix #9: Central config
from rppg_config import cfg
from rppg_algorithms import get_algorithm_by_name
from rppg_temporal import KalmanBPMFilter, ProbabilisticFusion
from rppg_sqi import StatisticallyCalibratedSQI, LearnedWeighting
from rppg_signal import apply_windowing, mature_harmonic_rejection, respiratory_interference_analysis
from rppg_uncertainty import UncertaintyAwareConfidence, propagate_variance
from rppg_roi import AdaptiveROIManager
from rppg_benchmark import ReproducibilityProtocol

# Backward-compat module-level names
BUFFER_SIZE            = cfg.BUFFER_SIZE
FPS_TARGET             = cfg.FPS_TARGET
BPM_LOW                = cfg.BPM_LOW
BPM_HIGH               = cfg.BPM_HIGH
FREQ_LOW               = BPM_LOW  / 60.0
FREQ_HIGH              = BPM_HIGH / 60.0
RESP_LOW               = cfg.RESP_LOW
RESP_HIGH              = cfg.RESP_HIGH
SQI_HARD_GATE          = cfg.SQI_HARD_GATE
BASELINE_SQI_THRESHOLD = cfg.BASELINE_SQI_THRESHOLD
MOTION_ENTER_THRESHOLD = cfg.MOTION_ENTER_THRESHOLD
MOTION_EXIT_THRESHOLD  = cfg.MOTION_EXIT_THRESHOLD

FOREHEAD_LANDMARKS = [10, 338, 297, 299, 337, 151, 108, 69, 67, 109]
CHEEK_LEFT_LANDMARKS = [
    50, 101, 118, 117, 116, 123, 147, 187, 207, 206, 205, 36, 142, 126, 100, 203]
CHEEK_RIGHT_LANDMARKS = [
    280, 330, 347, 346, 345, 352, 376, 411, 427, 426, 425, 266, 371, 355, 329, 423]
ROI_CONFIGS = {
    "forehead":    {"landmarks": FOREHEAD_LANDMARKS,    "base_weight": 1.5},
    "cheek_left":  {"landmarks": CHEEK_LEFT_LANDMARKS,  "base_weight": 1.0},
    "cheek_right": {"landmarks": CHEEK_RIGHT_LANDMARKS, "base_weight": 1.0},
}


def _rppg_warn(context: str, exc: Exception, once_key: str = "") -> None:
    """Emit a RuntimeWarning instead of swallowing exceptions silently.
    Replaces bare 'except Exception: pass' so errors surface during dev
    without crashing the real-time loop in production."""
    import warnings
    warnings.warn(f"[rPPG] {context}: {type(exc).__name__}: {exc}",
                  RuntimeWarning, stacklevel=3)


# ─── Data containers ──────────────────────────────────────────────────────────

@dataclass
class ROISignal:
    name: str
    buf_r: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    buf_g: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    buf_b: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    sqi: float = 0.0
    bpm: float = 0.0
    pts: Optional[np.ndarray] = None
    bbox: tuple = (0, 0, 0, 0)
    valid: bool = False
    roi_brightness: float = 128.0
    roi_motion_stability: float = 1.0
    roi_snr: float = 0.0
    roi_regularity: float = 0.0
    dynamic_weight: float = 1.0
    skin_coverage: float = 1.0
    agreement_factor: float = 1.0
    _reg_history: object = field(default_factory=lambda: deque(maxlen=8))
    _var_history: object = field(default_factory=lambda: deque(maxlen=8))
    _bpm_history: object = field(default_factory=lambda: deque(maxlen=15))  # Fix T: temporal
    is_harmonic_suspect: bool = False   # Fix H: cross-ROI harmonic flag
    # Priority 2D: auto-exclusion
    _bad_streak: int = 0       # consecutive frames with SQI below threshold
    _excluded_until: int = 0   # frame number until which this ROI is excluded

    def clear(self):
        self.buf_r.clear(); self.buf_g.clear(); self.buf_b.clear()
        self.sqi = 0.0; self.bpm = 0.0; self.valid = False
        self.roi_brightness = 128.0; self.roi_motion_stability = 1.0
        self.roi_snr = 0.0; self.roi_regularity = 0.0
        self.dynamic_weight = 1.0; self.skin_coverage = 1.0
        self.agreement_factor = 1.0; self.is_harmonic_suspect = False
        self._reg_history.clear(); self._var_history.clear()
        self._bpm_history.clear(); self._bad_streak = 0; self._excluded_until = 0


@dataclass
class FrameResult:
    fused_bpm: float         = 0.0
    fused_sqi: float         = 0.0
    roi_signals: dict        = field(default_factory=dict)
    chrom_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    motion_score: float      = 0.0
    motion_rejected: bool    = False
    skin_tone_factor: float  = 1.0
    frame_brightness: float  = 128.0
    freqs: np.ndarray        = field(default_factory=lambda: np.array([]))
    power: np.ndarray        = field(default_factory=lambda: np.array([]))
    sqi_breakdown: dict      = field(default_factory=dict)
    session_confidence: float = 0.0
    roi_agreement: float     = 1.0
    exposure_drift: float    = 0.0
    brightness_normalized: bool = False
    in_calibration: bool     = False   # Fix #10
    output_frozen: bool      = False   # Fix #6
    n_valid_rois: int        = 0       # Fix 4/5: ROI count at fusion
    bpm_locked: bool         = False  # True when temporal trust lock is active


# ─── Exposure compensator ─────────────────────────────────────────────────────

class ExposureCompensator:
    def __init__(self, window: int = 90):
        self.window          = window
        self.brightness_hist = deque(maxlen=window)
        self.drift_threshold = cfg.EXPOSURE_DRIFT_WARN
        self.drift           = 0.0

    def update(self, frame_brightness: float) -> tuple:
        self.brightness_hist.append(frame_brightness)
        if len(self.brightness_hist) < 10:
            return frame_brightness, False, 0.0
        arr    = np.array(self.brightness_hist)
        recent = float(np.mean(arr[-10:]))
        older  = float(np.mean(arr[:-10])) if len(arr) > 10 else recent
        self.drift     = abs(recent - older)
        drift_detected = self.drift > self.drift_threshold
        if len(arr) >= 30:
            t = np.arange(len(arr), dtype=float)
            poly = np.polyfit(t, arr, 1)
            trend_now  = np.polyval(poly, len(arr) - 1)
            normalized = frame_brightness - (trend_now - np.mean(arr))
            normalized = float(np.clip(normalized, 0, 255))
        else:
            normalized = frame_brightness
        return normalized, drift_detected, self.drift

    def normalize_roi_signal(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) < 15:
            return signal
        t = np.arange(len(signal), dtype=float)
        try:
            poly  = np.polyfit(t, signal, 1)
            trend = np.polyval(poly, t)
            return signal - trend + float(np.mean(signal))
        except Exception:
            return signal


# ─── Fix #3: BPM Smoother with Fix #2 plausibility ───────────────────────────

class BPMSmoother:
    """
    Fix #3: Three-stage smoother at decision layer:
      1. Physiological plausibility gate (Fix #2)
      2. Sliding-median window
      3. EMA -> Kalman
    """
    def __init__(self, median_window: int = None, ema_alpha: float = None):
        self.median_window = median_window or cfg.BPM_MEDIAN_WINDOW
        self.ema_alpha     = ema_alpha     or cfg.BPM_EMA_ALPHA
        self.raw_history   = deque(maxlen=self.median_window)
        self.ema_val: Optional[float] = None
        self._last_accepted: Optional[float] = None
        self.kalman = KalmanBPMFilter(q_bpm=0.1, q_vel=0.01, r_bpm=5.0)
        self._recent: deque = deque(maxlen=cfg.TEMPORAL_HISTORY_LEN)
        # Priority 2: Temporal Trust Lock
        self._stable_history: deque = deque(maxlen=int(cfg.STABLE_LOCK_SECONDS * 30))
        self._is_locked: bool = False
        self._lock_center: float = 0.0

    def _is_physiologically_plausible(self, bpm: float) -> bool:
        """Fix #2: Reject wild jumps even when ROI agreement looks good."""
        if len(self._recent) < 3:
            return True
        for prev in list(self._recent)[-3:]:
            if abs(bpm - prev) > cfg.TEMPORAL_BPM_MAX_JUMP_3FRAMES:
                return False
        return True

    def update(self, raw_bpm: float) -> float:
        if raw_bpm <= 0:
            return self.ema_val if self.ema_val else 0.0
        self._recent.append(raw_bpm)
        # Fix #2: physiological plausibility check
        if not self._is_physiologically_plausible(raw_bpm):
            if self.ema_val:
                blended = 0.20 * raw_bpm + 0.80 * self.ema_val
                self._recent[-1] = blended
                raw_bpm = blended
            else:
                return self.ema_val if self.ema_val else 0.0
        # Bug3 fix: wider jump gate during warmup (first 150 updates), tighter after
        effective_jump = cfg.BPM_MAX_JUMP if len(self.raw_history) >= cfg.BPM_MEDIAN_WINDOW else cfg.BPM_MAX_JUMP * 2.5
        if self._last_accepted is not None:
            if abs(raw_bpm - self._last_accepted) > effective_jump:
                self.raw_history.append(raw_bpm)
                return self.ema_val if self.ema_val else 0.0
        self._last_accepted = raw_bpm
        self.raw_history.append(raw_bpm)
        # Fix #3a: sliding median
        median_bpm = float(np.median(self.raw_history))
        # Fix #3b: EMA
        if self.ema_val is None:
            self.ema_val = median_bpm
        self.ema_val = self.ema_alpha * median_bpm + (1 - self.ema_alpha) * self.ema_val
        # Rigorous Kalman Filter Update
        final = self.kalman.update(self.ema_val, dt=1.0)

        # Priority 2: Temporal Trust Lock
        self._stable_history.append(final)
        # Bug6 fix: require raw_history full AND stable before locking
        min_frames_for_lock = int(cfg.STABLE_LOCK_SECONDS * 25)
        if (len(self._stable_history) >= min_frames_for_lock
                and len(self.raw_history) >= self.median_window):
            recent_std = float(np.std(self._stable_history))
            if recent_std <= cfg.STABLE_LOCK_BPM_STD:
                if not self._is_locked:
                    self._is_locked = True
                    self._lock_center = float(np.mean(self._stable_history))
            else:
                self._is_locked = False
        return self.ema_val

    def reset(self):
        self.raw_history.clear(); self.ema_val = None; self._last_accepted = None
        self._kf_x = None; self._kf_p = 10.0; self._recent.clear()
        self._stable_history.clear(); self._is_locked = False


# ─── Session Confidence Scorer ────────────────────────────────────────────────

class SessionConfidenceScorer:
    """Fix #7: confidence = SQI x consistency x agreement, NOT time accumulation."""
    def __init__(self, window: int = None):
        self.window = window or cfg.SESSION_CONF_CONSISTENCY_WINDOW
        self.sqi_samples = deque(maxlen=self.window)
        self.bpm_samples = deque(maxlen=self.window)
        self.agreement_samples = deque(maxlen=self.window)
        self.motion_samples = deque(maxlen=self.window)
        self.confidence = 0.0

    def update(self, sqi, bpm, is_moving, roi_agreement=1.0, n_valid_rois=1):
        if is_moving:
            self.motion_samples.append(1.0)
        else:
            self.motion_samples.append(0.0)
            if sqi >= cfg.CALIBRATION_SQI_FLOOR:
                self.sqi_samples.append(sqi)
                if bpm > 0: self.bpm_samples.append(bpm)
                self.agreement_samples.append(roi_agreement)

        if len(self.sqi_samples) < 10:
            self.confidence = 0.0; return

        mean_sqi = float(np.mean(self.sqi_samples)) / 100.0
        mean_agr = float(np.mean(self.agreement_samples)) if self.agreement_samples else 1.0
        motion_ratio = float(np.mean(self.motion_samples))

        # Consistency: low std in BPM = high confidence
        if len(self.bpm_samples) >= 5:
            bpm_std = float(np.std(self.bpm_samples))
            consistency = float(np.exp(-bpm_std / 10.0))
        else:
            consistency = 0.5

        # Multi-ROI bonus
        roi_bonus = 1.0 if n_valid_rois >= 2 else 0.85

        raw_conf = (
            cfg.SESSION_CONF_WEIGHT_SQI       * mean_sqi      +
            cfg.SESSION_CONF_WEIGHT_AGREEMENT * mean_agr      +
            cfg.SESSION_CONF_WEIGHT_TEMPORAL  * consistency   +
            cfg.SESSION_CONF_WEIGHT_MOTION    * (1.0 - motion_ratio)
        )
        # Hard multiplier: if agreement is low, confidence is capped
        agr_mult = max(cfg.SESSION_CONF_AGREEMENT_FLOOR, mean_agr)
        self.confidence = float(np.clip(raw_conf * agr_mult * roi_bonus * 100.0, 0, 100))

    def get_confidence(self): return self.confidence
    @property
    def confidence_label(self):
        if self.confidence >= 75: return "GOOD"
        if self.confidence >= 45: return "FAIR"
        return "POOR"
    def reset(self):
        self.sqi_samples.clear(); self.bpm_samples.clear()
        self.agreement_samples.clear(); self.motion_samples.clear(); self.confidence = 0.0


# ─── Calibration Phase ────────────────────────────────────────────────────────

class CalibrationPhase:
    """Fix #10: First N frames to establish baseline SQI and BPM."""
    def __init__(self, frames: int = None):
        self.target_frames = frames or cfg.CALIBRATION_FRAMES
        self.sqi_samples = []
        self.bpm_samples = []
        self.done = False
        self.baseline_sqi = 0.0
        self.baseline_bpm = 0.0

    def update(self, sqi, bpm):
        if self.done: return False
        if sqi >= cfg.CALIBRATION_SQI_FLOOR:
            self.sqi_samples.append(sqi)
            if bpm > 0: self.bpm_samples.append(bpm)
        if len(self.sqi_samples) >= self.target_frames:
            self.done = True
            self.baseline_sqi = float(np.mean(self.sqi_samples))
            self.baseline_bpm = float(np.median(self.bpm_samples)) if self.bpm_samples else 0.0
        return True

    def reset(self):
        self.sqi_samples = []; self.bpm_samples = []; self.done = False


# ─── Skin Tone Calibrator ─────────────────────────────────────────────────────

class SkinToneCalibrator:
    def __init__(self):
        self.r_history = deque(maxlen=150); self.g_history = deque(maxlen=150); self.b_history = deque(maxlen=150)
        self.nf_r = 1.0; self.nf_g = 1.0; self.nf_b = 1.0

    def update(self, r, g, b):
        self.r_history.append(r); self.g_history.append(g); self.b_history.append(b)
        if len(self.g_history) >= 30:
            mr = np.mean(self.r_history); mg = np.mean(self.g_history); mb = np.mean(self.b_history)
            # Normalize to green channel
            self.nf_r = mg / (mr + 1e-9); self.nf_b = mg / (mb + 1e-9)

    def get_normalization_factors(self): return self.nf_r, 1.0, self.nf_b

    def skin_type_label(self):
        if not self.g_history: return "UNKNOWN"
        # Very rough heuristic based on brightness
        lum = (np.mean(self.r_history) + np.mean(self.g_history) + np.mean(self.b_history)) / 3.0
        if lum > 180: return "TYPE I-II"
        if lum > 120: return "TYPE III-IV"
        return "TYPE V-VI"

    def reset(self):
        self.r_history.clear(); self.g_history.clear(); self.b_history.clear()
        self.nf_r = 1.0; self.nf_g = 1.0; self.nf_b = 1.0


# ─── Motion Artifact Detector ─────────────────────────────────────────────────

class MotionArtifactDetector:
    """
    Redesigned motion detector.
    Root cause of "always 2.0": np.mean(absdiff) on raw pixels ≈ sensor noise floor.
    Fix:
      1. Pre-blur both frames to kill sensor noise
      2. Use 85th-percentile of diff (not mean) – ignores noise, responds to real motion
      3. Dual signal: pixel-diff + landmark displacement for robustness
      4. EMA temporal smoothing over 5 frames
      5. Thresholds tuned for percentile-based metric (not raw mean)
    """
    def __init__(self):
        self.prev_gray     = None
        self.prev_lm_pts   = None          # landmark positions prev frame
        self.motion_score  = 0.0
        self.is_moving     = False
        self.grace_counter = 0
        self._score_ema    = 0.0           # EMA of raw score (fast, ~5-frame)
        self._ema_alpha    = 0.35          # EMA smoothing
        self._slow_ema     = 0.0           # FIX1-HPF: slow baseline tracker (breathing freq)
        self._slow_alpha   = 0.04          # ~8s time constant @30fps → tracks respiration drift
        self._prev_raw     = 0.0           # FIX1-HPF: previous raw score for HPF difference term
        # Landmark anchor indices (stable, spread across face)
        self._lm_ids = [1, 33, 263, 61, 291, 199, 168, 4, 234, 454, 10, 152]

    def update(self, gray, face_landmarks, h, w):
        if face_landmarks is None:
            self.prev_gray   = None
            self.prev_lm_pts = None
            return 0.0, False

        # ── Signal 1: landmark displacement (immune to lighting changes) ──────
        lm       = face_landmarks.landmark
        curr_pts = np.array([[lm[i].x * w, lm[i].y * h]
                              for i in self._lm_ids if i < len(lm)], dtype=np.float32)
        lm_score = 0.0
        if self.prev_lm_pts is not None and len(curr_pts) == len(self.prev_lm_pts):
            dists    = np.linalg.norm(curr_pts - self.prev_lm_pts, axis=1)
            lm_score = float(np.median(dists))   # pixels of displacement
        self.prev_lm_pts = curr_pts.copy()

        # ── Signal 2: pixel diff on pre-blurred face crop ────────────────────
        lm_x = (curr_pts[:, 0] / w); lm_y = (curr_pts[:, 1] / h)
        x1 = max(0, int(lm_x.min() * w) - 10)
        y1 = max(0, int(lm_y.min() * h) - 10)
        x2 = min(w, int(lm_x.max() * w) + 10)
        y2 = min(h, int(lm_y.max() * h) + 10)
        px_score = 0.0
        if self.prev_gray is not None and x2 > x1 and y2 > y1:
            curr_crop = gray[y1:y2, x1:x2]
            prev_crop = self.prev_gray[y1:y2, x1:x2]
            if curr_crop.shape == prev_crop.shape and curr_crop.size > 0:
                # Pre-blur kills sensor noise (3×3 Gaussian ≈ noise floor → 0)
                curr_b = cv2.GaussianBlur(curr_crop, (3, 3), 0)
                prev_b = cv2.GaussianBlur(prev_crop, (3, 3), 0)
                diff   = cv2.absdiff(curr_b, prev_b)
                # 85th percentile: ignores low-level noise, catches real motion edges
                px_score = float(np.percentile(diff, 85))
        self.prev_gray = gray.copy()

        # ── Combine: lm-gated formula – lighting changes can't fake motion ──────
        # lm_score: noise<0.3px, small move>1px, head turn>5px
        # px_score (p85 blurred): noise<1.5, lighting up to 10, real motion>8
        # Key: px contribution is gated by lm stability → lighting alone = no trigger
        lm_gate   = min(lm_score / 0.8, 1.0)   # 0=still, 1=moving (per landmarks)
        raw_score = lm_score * 1.5 + px_score * 0.15 * lm_gate

        # ── EMA temporal smoothing (5-frame) ──────────────────────────────────
        self._score_ema = self._ema_alpha * raw_score + (1 - self._ema_alpha) * self._score_ema

        # ── FIX1-HPF: High-Pass Filter – removes slow breathing oscillation (∼0.2-0.4 Hz) ───
        # slow_ema is a very long time-constant tracker (tau≈8s @30fps).
        # HPF score = fast_ema – 0.6*slow_ema  → DC drift & respiration component cancelled.
        self._slow_ema = self._slow_alpha * raw_score + (1 - self._slow_alpha) * self._slow_ema
        hpf_score = max(0.0, self._score_ema - self._slow_ema * 0.6)
        self.motion_score = round(hpf_score, 3)

        # ── Hysteresis with grace period ──────────────────────────────────────
        if self.motion_score > cfg.MOTION_ENTER_THRESHOLD:
            self.is_moving     = True
            self.grace_counter = cfg.MOTION_GRACE_FRAMES
        elif self.motion_score < cfg.MOTION_EXIT_THRESHOLD:
            if self.grace_counter > 0:
                self.grace_counter -= 1
            else:
                self.is_moving = False

        return self.motion_score, self.is_moving

    def reset(self):
        self.prev_gray     = None
        self.prev_lm_pts   = None
        self.motion_score  = 0.0
        self.is_moving     = False
        self.grace_counter = 0
        self._score_ema    = 0.0
        self._slow_ema     = 0.0  # FIX1-HPF: reset baseline tracker
        self._prev_raw     = 0.0


# ─── Core Signal Processing ───────────────────────────────────────────────────

def bandpass_filter(data, fps, lowcut=0.7, highcut=3.0, order=4):
    nyq = 0.5 * fps
    low = lowcut / nyq; high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def detrend(data):
    return scipy_detrend(data, type='linear')

def chrom_rppg(r, g, b):
    r = np.array(r); g = np.array(g); b = np.array(b)
    rn = r / (np.mean(r) + 1e-9); gn = g / (np.mean(g) + 1e-9); bn = b / (np.mean(b) + 1e-9)
    xs = 3 * rn - 2 * gn; ys = 1.5 * rn + gn - 1.5 * bn
    return xs - (np.std(xs) / (np.std(ys) + 1e-9)) * ys

def pos_rppg(r, g, b):
    r = np.array(r); g = np.array(g); b = np.array(b)
    rn = r / (np.mean(r) + 1e-9); gn = g / (np.mean(g) + 1e-9); bn = b / (np.mean(b) + 1e-9)
    C = np.vstack([rn, gn, bn]); Pn = np.dot([0, 1, -1], C); S2 = np.dot([0, 1, 1], C)
    return Pn / (np.std(Pn) + 1e-9) - (np.std(Pn) / (np.std(S2) + 1e-9)) * (S2 / (np.std(S2) + 1e-9))

def _pick_one_peak(fft_v, freqs, mask):
    if not np.any(mask): return None
    idxs, _ = find_peaks(fft_v, distance=3)
    idxs = [i for i in idxs if mask[i]]
    if not idxs: return freqs[mask][np.argmax(fft_v[mask])]
    idxs = sorted(idxs, key=lambda i: fft_v[i], reverse=True)[:3]
    def _is_harm(fc, ff, tol=0.05):
        return any(abs(fc - ff * r) / (ff * r + 1e-9) < tol
                   for r in [2.0, 3.0, 0.5, 0.333, 1.5, 0.667])
    for i, pidx in enumerate(idxs):
        fc = freqs[pidx]
        if not any(freqs[pidx2] < fc and _is_harm(fc, freqs[pidx2])
                   for j, pidx2 in enumerate(idxs) if j != i):
            return fc
    return freqs[idxs[0]]

_peak_freq_history = {}

def estimate_bpm_fft(signal, fps, roi_name: str = ""):
    n = len(signal)
    if n < 30: return 0.0, np.array([]), np.array([]), {}
    # Anti-leakage windowing
    win_signal = apply_windowing(signal, window_type='hann')
    fft_v_full = np.abs(np.fft.rfft(win_signal))
    freqs_full  = np.fft.rfftfreq(n, d=1.0 / fps)
    mask_full   = (freqs_full >= FREQ_LOW) & (freqs_full <= FREQ_HIGH)
    win_size   = max(int(fps * cfg.FFT_WINDOW_SEC), 30)
    hop_size   = win_size // 2
    candidates = []
    if n >= win_size:
        for start in range(0, n - win_size + 1, hop_size):
            seg = signal[start: start + win_size]
            seg_win = apply_windowing(seg, window_type='hann')
            seg_fft = np.abs(np.fft.rfft(seg_win))
            seg_freq = np.fft.rfftfreq(len(seg), d=1.0 / fps)
            seg_mask = (seg_freq >= FREQ_LOW) & (seg_freq <= FREQ_HIGH)
            hz = _pick_one_peak(seg_fft, seg_freq, seg_mask)
            if hz is not None: candidates.append(hz)
    if len(candidates) >= 2:
        arr = np.array(candidates); best_hz = None; best_n = 0
        for hz in arr:
            cluster = arr[np.abs(arr - hz) < cfg.FFT_WINDOW_VOTE_TOL]
            if len(cluster) > best_n:
                best_n = len(cluster); best_hz = float(np.median(cluster))
        peak_hz = best_hz if best_hz is not None else float(np.median(arr))
    else:
        hz = _pick_one_peak(fft_v_full, freqs_full, mask_full)
        peak_hz = hz if hz is not None else 0.0
    fft_consistency = 1.0
    if roi_name:
        if roi_name not in _peak_freq_history: _peak_freq_history[roi_name] = deque(maxlen=12)
        hist = _peak_freq_history[roi_name]
        if len(hist) >= 5:
            hist_med = float(np.median(hist)); hist_std = float(np.std(hist)); hz_dev = abs(peak_hz - hist_med)
            fft_consistency = float(np.exp(-hz_dev / 0.20))
            if hist_std < 0.08 and hz_dev > cfg.FFT_HISTORY_OVERRIDE:
                if len(candidates) >= 2:
                    hist_votes = sum(1 for c in candidates if abs(c - hist_med) < cfg.FFT_WINDOW_VOTE_TOL)
                    new_votes  = sum(1 for c in candidates if abs(c - peak_hz) < cfg.FFT_WINDOW_VOTE_TOL)
                    if hist_votes >= new_votes: peak_hz = hist_med
        _peak_freq_history[roi_name].append(peak_hz)
    peak_bpm = peak_hz * 60.0
    fft_m_disp = fft_v_full.copy(); fft_m_disp[~mask_full] = 0.0
    near = (freqs_full >= peak_hz - 0.10) & (freqs_full <= peak_hz + 0.10)
    if np.any(near) and np.any(mask_full):
        pp = fft_m_disp[near].max(); dom = float(pp / (fft_m_disp[mask_full].sum() + 1e-9))
        hp = pp * 0.5; pidx_d = int(np.argmax(fft_m_disp * near.astype(float)))
        li = pidx_d; ri = pidx_d
        while li > 0 and fft_m_disp[li] > hp: li -= 1
        while ri < len(fft_m_disp) - 1 and fft_m_disp[ri] > hp: ri += 1
        width_ok = (ri - li) <= 5
    else: dom = 0.0; width_ok = True
    vi = {
        "dominance": round(dom, 3), "peak_dominance": round(dom, 3), "dominant": dom >= 0.12,
        "peak_width_ok": width_ok, "harmonic_rejected": any(abs(c * 2 - peak_hz) < 0.12 for c in candidates if abs(c - peak_hz) > 0.05) if candidates else False,
        "fft_consistency": round(fft_consistency, 3), "n_windows": len(candidates),
        "is_harmonic": mature_harmonic_rejection(fft_v_full, freqs_full, peak_hz),
        "resp_interference": respiratory_interference_analysis(signal, fps)
    }
    return peak_bpm, freqs_full, fft_v_full, vi


# ─── Fix #4: SQI with rebalanced weights + capped penalties ──────────────────

def compute_sqi(signal, fps, peak_bpm, motion_score=0.0, mean_brightness=128.0,
                roi_brightness=-1.0, fft_validation=None, prev_sqi=-1.0,
                roi_obj=None, exposure_drift=0.0):
    breakdown={"snr":0.0,"regularity":0.0,"variance":0.0,"motion_penalty":0.0,
               "lighting_penalty":0.0,"fft_penalty":0.0,"overall":0.0,
               "temporal_stability": 0.0, "peak_consistency": 0.0}
    if len(signal)<30 or peak_bpm<=0: return 0.0, breakdown

    sig=signal-np.mean(signal); sig=scipy_detrend(sig,type="linear")
    n=len(sig); fft_v=np.abs(np.fft.rfft(sig*np.hanning(n)))**2
    freqs=np.fft.rfftfreq(n,d=1.0/fps)
    mask_p=(freqs>=FREQ_LOW)&(freqs<=FREQ_HIGH)
    if not np.any(mask_p): return 0.0, breakdown

    peak_hz=peak_bpm/60.0
    mask_peak=(freqs>=peak_hz-0.08)&(freqs<=peak_hz+0.08)
    mask_harm2=(freqs>=peak_hz*2-0.08)&(freqs<=peak_hz*2+0.08)
    mask_sb=((freqs>=peak_hz-0.45)&(freqs<peak_hz-0.10))|((freqs>peak_hz+0.10)&(freqs<=peak_hz+0.45))
    power_peak=fft_v[mask_peak].sum() if np.any(mask_peak) else 0.0
    power_harm2=fft_v[mask_harm2].sum() if np.any(mask_harm2) else 0.0

    if np.any(mask_sb):
        sb_med=float(np.median(fft_v[mask_sb]))
        n_pk=max(int(np.sum(mask_peak)),1)
        snr_raw=power_peak/(sb_med*n_pk+1e-9)
        snr_score=min(np.log1p(snr_raw)/np.log1p(8.0),1.0)
    else: snr_score=0.0
    if power_peak>1e-9:
        snr_score=min(snr_score+min((power_harm2/(power_peak+1e-9))*0.10,0.10),1.0)

    sig_norm=(sig-sig.mean())/(sig.std()+1e-9)
    peaks,_=find_peaks(sig_norm,distance=int(fps*0.50),prominence=0.35,height=0.1)
    if len(peaks)>=3:
        intervals=np.diff(peaks); cv=np.std(intervals)/(np.mean(intervals)+1e-9)
        reg_raw=max(0.0,1.0-cv*1.5)
        if roi_obj is not None: roi_obj._reg_history.append(reg_raw)
    elif len(peaks)==2:
        reg_raw=0.20
        if roi_obj is not None: roi_obj._reg_history.append(reg_raw)
    else: reg_raw=None
    if roi_obj is not None and roi_obj._reg_history:
        reg_score=float(np.mean(roi_obj._reg_history))
    else: reg_score=reg_raw if reg_raw is not None else 0.0

    sig_std=float(np.std(sig))
    if sig_std<0.004: var_raw=0.0
    elif sig_std>8.0: var_raw=max(0.0,1.0-(sig_std-8.0)/8.0)
    else: var_raw=min(1.0,sig_std/0.6)
    if sig_std>3.0 and snr_score<0.30: var_raw*=0.4
    if roi_obj is not None:
        roi_obj._var_history.append(var_raw); var_score=float(np.mean(roi_obj._var_history))
    else: var_score=var_raw

    # 🥇 Fix 1: FFT penalty (Non-linear & less aggressive)
    fft_penalty=0.0
    if fft_validation:
        dom = fft_validation.get("dominance", 0.15)
        dom_pen_raw = max(0.0, (cfg.FFT_DOMINANCE_STRONG - dom) / cfg.FFT_DOMINANCE_STRONG)
        dom_pen = (dom_pen_raw ** 0.5) * 0.25
        fft_penalty += dom_pen
        if not fft_validation.get("peak_width_ok", True): fft_penalty += 0.05
        if fft_validation.get("harmonic_rejected", False): fft_penalty += 0.02
        fft_penalty = min(fft_penalty, cfg.SQI_MAX_FFT_PENALTY)
        consistency = fft_validation.get("fft_consistency", 1.0)
        if consistency < 0.8:
            extra_pen = ((1.0 - consistency) ** 0.5) * 0.10
            fft_penalty = min(fft_penalty + extra_pen, cfg.SQI_MAX_FFT_PENALTY)

    # 🥈 Fix 2: Rebalance SQI formula & 🥉 Fix 3: Temporal Stability
    temp_stability = 1.0
    if roi_obj is not None and len(roi_obj._bpm_history) >= 5:
        hist = list(roi_obj._bpm_history); bpm_std = np.std(hist); bpm_mean = np.mean(hist)
        temp_stability = max(0.0, 1.0 - (bpm_std / (bpm_mean + 1e-9)) * 5.0)
    
    peak_consistency = fft_validation.get("fft_consistency", 0.5) if fft_validation else 0.5

    motion_penalty=0.0
    # Fix2: threshold raised to match redesigned motion metric (noise floor ~0.3, real motion >2.0)
    # FIX3: threshold raised to 2.0 (breathing/tremor lives at 0.5-1.5, must not be penalised).
    # Penalty ramps gradually via linear scale /8.0 (was /6.0) → gentler curve below 2.8 enter.
    if motion_score > 2.0: motion_penalty = min((motion_score - 2.0) / 8.0, cfg.SQI_MAX_MOTION_PENALTY)

    bref=roi_brightness if roi_brightness>=0 else mean_brightness
    lighting_penalty=0.0
    if bref<50: lighting_penalty=min((50-bref)/50.0, cfg.SQI_MAX_LIGHTING_PENALTY)
    elif bref>210: lighting_penalty=min((bref-210)/40.0, cfg.SQI_MAX_LIGHTING_PENALTY)
    if exposure_drift > cfg.EXPOSURE_DRIFT_WARN:
        drift_ratio = min((exposure_drift - cfg.EXPOSURE_DRIFT_WARN) / (cfg.EXPOSURE_DRIFT_FREEZE - cfg.EXPOSURE_DRIFT_WARN), 1.0)
        drift_sqi_pen = drift_ratio * cfg.EXPOSURE_DRIFT_SQI_PENALTY
        lighting_penalty = min(lighting_penalty + drift_sqi_pen, cfg.SQI_MAX_LIGHTING_PENALTY)

    weighted_score = (
        cfg.SQI_WEIGHT_SNR * snr_score +
        cfg.SQI_WEIGHT_REGULARITY * reg_score +
        cfg.SQI_WEIGHT_TEMPORAL * temp_stability +
        cfg.SQI_WEIGHT_PEAK_CONSIST * peak_consistency +
        cfg.SQI_WEIGHT_VARIANCE * var_score
    )
    total_penalty = (motion_penalty + lighting_penalty + fft_penalty) / 3.0
    raw = weighted_score - (cfg.SQI_PENALTY_WEIGHT * total_penalty)
    env_mult = (1.0 - 0.3 * motion_penalty) * (1.0 - 0.3 * lighting_penalty)
    raw *= env_mult

    if snr_score>0.35 and reg_score>0.20: raw=max(raw,0.22)
    if roi_obj is not None:
        af = getattr(roi_obj, 'agreement_factor', 1.0)
        if af < 0.7:
            consensus_pen = (0.7 - af) / 0.7 * 0.30
            raw *= (1.0 - consensus_pen)
            
    overall=raw*100.0
    if prev_sqi>0: overall=0.40*prev_sqi+0.60*overall
    overall=round(min(100.0,max(0.0,overall)),1)
    
    breakdown={
        "snr": round(snr_score*100,1), "regularity": round(reg_score*100,1),
        "variance": round(var_score*100,1), "motion_penalty": round(motion_penalty*100,1),
        "lighting_penalty": round(lighting_penalty*100,1), "fft_penalty": round(fft_penalty*100,1),
        "temporal_stability": round(temp_stability*100,1), "peak_consistency": round(peak_consistency*100,1),
        "overall": overall
    }
    return overall, breakdown


def get_roi_polygon_masked(frame, face_landmarks, landmark_ids, h, w):
    pts=[]
    for idx in landmark_ids:
        if idx>=len(face_landmarks.landmark): continue
        lm=face_landmarks.landmark[idx]; pts.append([int(lm.x*w),int(lm.y*h)])
    if len(pts)<3: return None
    pts_arr=np.array(pts,dtype=np.int32); mask=np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask,[pts_arr],255)
    if cv2.countNonZero(mask)<10: return None
    mean_b=cv2.mean(frame[:,:,0],mask=mask)[0]; mean_g=cv2.mean(frame[:,:,1],mask=mask)[0]; mean_r=cv2.mean(frame[:,:,2],mask=mask)[0]
    roi_brightness=(mean_r+mean_g+mean_b)/3.0
    x1=max(0,pts_arr[:,0].min()); y1=max(0,pts_arr[:,1].min()); x2=min(w,pts_arr[:,0].max()); y2=min(h,pts_arr[:,1].max())
    return pts_arr, mean_r, mean_g, mean_b, (x1,y1,x2,y2), roi_brightness


# ─── Fix #1 + #6: Dynamic weight with hard dominance ─────────────────────────

def compute_dynamic_roi_weight(roi, base_weight, motion_score, exposure_drift=0.0):
    """🧠 Fix 5: Dynamic Trust System"""
    snr_val=roi.roi_snr
    # FIX3: Continuous exponential weighting replaces binary kill switch.
    # exp(-k*(thresh-snr)/thresh) gives near-zero weight far below threshold,
    # 1.0 well above it, and smooth curve in between – no cliff edge.
    if snr_val < cfg.SNR_KILL_THRESHOLD:
        snr_score = cfg.SNR_KILL_WEIGHT + (1.0 - cfg.SNR_KILL_WEIGHT) * np.exp(-3.0 * (cfg.SNR_KILL_THRESHOLD - snr_val) / (cfg.SNR_KILL_THRESHOLD + 1e-9))
    else: snr_score = min((snr_val / 100.0) ** 1.5, 1.0)
    temp_stab = 1.0
    if len(roi._bpm_history) >= 5:
        bpm_std = np.std(roi._bpm_history); temp_stab = max(0.1, 1.0 - (bpm_std / 10.0))
    af = roi.agreement_factor
    # FIX3: Continuous exponential weighting for agreement (same pattern as SNR above).
    if af < cfg.AGREEMENT_KILL_BELOW:
        agr_score = cfg.AGREEMENT_KILL_MULT + (af / (cfg.AGREEMENT_KILL_BELOW + 1e-9)) * (cfg.AGREEMENT_KILL_BELOW - cfg.AGREEMENT_KILL_MULT)
    else: agr_score = af
    trust_score = 0.5 * snr_score + 0.3 * temp_stab + 0.2 * agr_score
    weight = base_weight * trust_score
    if motion_score > 3.5: weight *= 0.5   # Fix3: raised to match new metric
    if exposure_drift > cfg.EXPOSURE_DRIFT_FREEZE: weight *= 0.1
    elif exposure_drift > cfg.EXPOSURE_DRIFT_WARN: weight *= cfg.EXPOSURE_DRIFT_WEIGHT_MULT
    return float(np.clip(weight, cfg.DYN_WEIGHT_MIN, cfg.DYN_WEIGHT_MAX))


def cross_roi_harmonic_check(rois_dict):
    if not rois_dict: return
    ref_roi = max(rois_dict.values(), key=lambda r: r.sqi)
    ref_bpm = ref_roi.bpm; harmonic_ratios = [2.0, 3.0, 0.5, 0.333]; tol_ratio = 0.08
    for name, roi in rois_dict.items():
        if roi is ref_roi or roi.bpm <= 0: continue
        ratio = roi.bpm / (ref_bpm + 1e-6)
        is_suspect = min(abs(ratio - hr) for hr in harmonic_ratios) < tol_ratio
        if abs(roi.bpm - ref_bpm * 2.0) < ref_bpm * 0.15: is_suspect = True
        if abs(roi.bpm - ref_bpm * 0.5) < ref_bpm * 0.08: is_suspect = True
        if abs(roi.bpm - ref_bpm) < 6.0: is_suspect = False
        roi.is_harmonic_suspect = is_suspect


def _cluster_rois(roi_items):
    if not roi_items: return []
    sorted_rois = sorted(roi_items, key=lambda x: x[1].bpm)
    clusters = []; current_cluster = [sorted_rois[0]]
    for i in range(1, len(sorted_rois)):
        if abs(sorted_rois[i][1].bpm - current_cluster[-1][1].bpm) <= cfg.CLUSTER_DISTANCE_BPM:
            current_cluster.append(sorted_rois[i])
        else:
            clusters.append(current_cluster); current_cluster = [sorted_rois[i]]
    clusters.append(current_cluster); return clusters


def _fuse_cluster(cluster):
    total_w = sum(c[2] for c in cluster) + 1e-9
    fused_bpm = sum(c[1].bpm * c[2] for c in cluster) / total_w
    return fused_bpm, total_w


def hierarchical_fusion(roi_items, min_cluster_weight_frac=None):
    if not roi_items: return 0.0
    clamped = [(n, r, float(np.clip(w, cfg.DYN_WEIGHT_MIN, cfg.DYN_WEIGHT_MAX))) for n, r, w in roi_items]
    total_w = sum(c[2] for c in clamped)
    if total_w <= 0: return 0.0
    normalized = []
    for n, r, w in clamped:
        frac = w / total_w
        if frac > cfg.DYN_WEIGHT_NORM_MAX: w = cfg.DYN_WEIGHT_NORM_MAX * total_w
        normalized.append((n, r, w))
    roi_items = normalized; min_frac = min_cluster_weight_frac or cfg.MIN_CLUSTER_WEIGHT
    clusters = _cluster_rois(roi_items); total_all = sum(c[2] for c in roi_items)
    cluster_info = []
    for cluster in clusters:
        fb, cw = _fuse_cluster(cluster); frac = cw / (total_all + 1e-9); n_members = len(cluster)
        cluster_info.append((fb, cw, frac, n_members))
    cluster_info.sort(key=lambda x: (x[3], x[1]), reverse=True)
    results = []; majority_n = cluster_info[0][3] if cluster_info else 0
    for fb, cw, frac, n_m in cluster_info:
        if frac >= min_frac or n_m == majority_n: results.append((fb, cw))
    if not results: return cluster_info[0][0]
    total_w = sum(r[1] for r in results); return sum(r[0]*r[1] for r in results)/total_w


# ─── Main fusion engine V1 ─────────────────────────────────────────────────────

class MultiROIFusionEngine:
    def __init__(self):
        self.rois = {name: ROISignal(name=name) for name in ROI_CONFIGS}
        self.skin_calibrator = SkinToneCalibrator(); self.motion_detector = MotionArtifactDetector()
        self.exposure_comp = ExposureCompensator(); self.bpm_smoother = BPMSmoother()
        self.session_scorer = SessionConfidenceScorer(); self.calibration = CalibrationPhase()
        self.prob_fusion = ProbabilisticFusion()
        self.calibrated_sqi = StatisticallyCalibratedSQI()
        self.learned_weighting = LearnedWeighting(ROI_CONFIGS.keys())
        self.uncertainty_engine = UncertaintyAwareConfidence()
        self.roi_manager = AdaptiveROIManager()
        self.protocol = ReproducibilityProtocol()
        self.fps_buffer = deque(maxlen=30); self.bpm_history = deque(maxlen=12)
        self.prev_time = time.time(); self.frame_count = 0; self.MIN_FRAMES = cfg.MIN_FRAMES; self._frozen_bpm = 0.0

    def update(self, frame, face_landmarks, h, w):
        result=FrameResult(); self.frame_count+=1; now=time.time(); dt=now-self.prev_time; self.prev_time=now
        self.fps_buffer.append(1.0/dt if dt>1e-9 else 30.0); fps=float(np.mean(self.fps_buffer))
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        motion_score,is_moving=self.motion_detector.update(frame_gray,face_landmarks,h,w)
        result.motion_score=motion_score; result.motion_rejected=is_moving
        norm_brightness,drift_detected,drift_mag=self.exposure_comp.update(float(frame.mean()))
        result.frame_brightness=norm_brightness; result.exposure_drift=drift_mag; result.brightness_normalized=drift_detected
        if drift_mag>=cfg.EXPOSURE_DRIFT_FREEZE:
            result.output_frozen=True; result.fused_bpm=self._frozen_bpm; result.roi_signals={k:v for k,v in self.rois.items()}; return result
        if face_landmarks is None: return result
        nf_r,nf_g,nf_b=self.skin_calibrator.get_normalization_factors()
        # Adaptive ROI & Skin Segmentation
        self.roi_manager.segment_skin(frame)
        
        for roi_name,cfg_roi in ROI_CONFIGS.items():
            roi=self.rois[roi_name]
            out = self.roi_manager.get_adaptive_roi(frame, face_landmarks, cfg_roi["landmarks"], h, w)
            
            if out is None: roi.valid=False; continue
            mr, mg, mb, refined_mask = out
            roi.valid=True; roi.roi_brightness=float(np.mean([mr, mg, mb]))
            # Update ROI metadata for legacy compatibility
            roi.skin_coverage = cv2.countNonZero(refined_mask) / (refined_mask.size + 1e-9)
            if not is_moving and roi.roi_brightness<=220 and roi.roi_brightness>=40:
                roi.buf_r.append(mr*nf_r); roi.buf_g.append(mg*nf_g); roi.buf_b.append(mb*nf_b); self.skin_calibrator.update(mr,mg,mb)
            elif not is_moving: self.skin_calibrator.update(mr,mg,mb)
        result.skin_tone_factor=nf_g
        valid_rois={}
        for k,v in self.rois.items():
            if not v.valid or len(v.buf_g)<self.MIN_FRAMES: continue
            if v._excluded_until > self.frame_count: continue
            if v.skin_coverage<0.20: continue  # v12: lowered from 0.55
            if v.roi_brightness<cfg.ROI_BRIGHTNESS_MIN or v.roi_brightness>cfg.ROI_BRIGHTNESS_MAX: continue
            # v12: SNR gate removed; regularity gate disabled (REGULARITY_HARD_GATE=0)
            if v.bpm>0 and not (cfg.BPM_PLAUSIBLE_LOW<=v.bpm<=cfg.BPM_PLAUSIBLE_HIGH): continue
            valid_rois[k]=v
        if not valid_rois: result.roi_signals={k:v for k,v in self.rois.items()}; return result
        for roi_name,roi in valid_rois.items():
            arr_r=np.array(roi.buf_r); arr_g=np.array(roi.buf_g); arr_b=np.array(roi.buf_b)
            try:
                if motion_score<=self.motion_detector.enter_threshold:
                    # Standardized Algorithm Pipeline
                    algo_name = getattr(cfg, "RPPG_ALGO", "POS")
                    extractor = get_algorithm_by_name(algo_name)
                    sc = extractor(arr_r, arr_g, arr_b)
                    
                    sc = detrend(sc)
                    sc = self.exposure_comp.normalize_roi_signal(sc)
                    sig_filt = bandpass_filter(sc, fps)
                    _std=float(np.std(sig_filt))
                    if _std>1e-8: sig_filt=(sig_filt-np.mean(sig_filt))/_std
                    bpm_raw,freqs,power,fft_val=estimate_bpm_fft(sig_filt,fps,roi_name=roi_name)
                    # Statistically Calibrated SQI
                    sqi_cal, sqi_meta = self.calibrated_sqi.calculate_sqi(sig_filt, fps, bpm_raw)
                    
                    # Hybrid SQI (Legacy + Calibrated)
                    sqi_legacy, breakdown = compute_sqi(sig_filt,fps,bpm_raw,motion_score=motion_score,mean_brightness=result.frame_brightness,roi_brightness=roi.roi_brightness,fft_validation=fft_val,prev_sqi=roi.sqi,roi_obj=roi,exposure_drift=result.exposure_drift)
                    
                    # Mature Harmonic & Respiratory Rejection
                    if fft_val.get("is_harmonic", False): sqi_cal *= 0.5
                    resp_ratio = fft_val.get("resp_interference", 0.0)
                    if resp_ratio > 2.0: sqi_cal *= 0.7 # Heavy respiratory interference
                    
                    sqi = 0.5 * sqi_cal + 0.5 * sqi_legacy
                    roi.bpm=bpm_raw; roi.sqi=sqi; roi.roi_snr=sqi_meta.get("snr_db", 0.0); roi.roi_regularity=sqi_meta.get("spectral_entropy", 0.0)
                    if bpm_raw>0: roi._bpm_history.append(bpm_raw)
                    if sqi < cfg.ROI_BAD_SQI_THRESH:
                        roi._bad_streak += 1
                        if roi._bad_streak >= cfg.ROI_BAD_STREAK_LIMIT: roi._excluded_until = self.frame_count + cfg.ROI_REHAB_FRAMES; roi._bad_streak = 0
                    else: roi._bad_streak = max(0, roi._bad_streak - 2)
                    if sqi>result.sqi_breakdown.get("overall",0) and motion_score<=self.motion_detector.enter_threshold:
                        result.freqs=freqs; result.power=power; result.sqi_breakdown=breakdown
            except Exception as _exc:
                roi.sqi = 0.0
                _rppg_warn(f'ROI processing (frame {getattr(self,"frame_count",0)})', _exc)
        best_sqi=max((r.sqi for r in valid_rois.values()),default=0.0); best_bpm=max((r.bpm for r in valid_rois.values() if BPM_LOW<=r.bpm<=BPM_HIGH),default=0.0)
        result.in_calibration=self.calibration.update(best_sqi,best_bpm)
        # v12: try with SQI gate first, fallback to best available ROI
        candidates = {k: v for k, v in valid_rois.items() if BPM_LOW <= v.bpm <= BPM_HIGH and v.sqi >= SQI_HARD_GATE}
        if not candidates:
            # Fallback: use any ROI with valid BPM regardless of SQI (show something)
            candidates = {k: v for k, v in valid_rois.items() if BPM_LOW <= v.bpm <= BPM_HIGH}
        if not candidates: result.roi_signals={k:v for k,v in self.rois.items()}; return result
        ref_roi = max(candidates.values(), key=lambda r: r.sqi); ref_bpm = ref_roi.bpm
        HARD_OUTLIER = cfg.FUSION_OUTLIER_THRESHOLD + 10.0; candidates = {k: v for k, v in candidates.items() if abs(v.bpm - ref_bpm) <= HARD_OUTLIER}
        if not candidates: candidates = {ref_roi.name: ref_roi}
        for roi in candidates.values():
            dev = abs(roi.bpm - ref_bpm); roi.agreement_factor = float(np.exp(-dev / cfg.AGREEMENT_DEV_K))
        accepted = {k: v for k, v in candidates.items() if v.agreement_factor >= cfg.ROI_MIN_AGREEMENT_FACTOR}
        if not accepted:
            accepted = {k: v for k, v in valid_rois.items() if v is ref_roi}
            if not accepted: result.roi_signals={k:v for k,v in self.rois.items()}; return result
        accepted_bpms = [r.bpm for r in accepted.values()]
        result.roi_agreement = float(np.exp(-np.std(accepted_bpms) / cfg.AGREEMENT_STD_K)) if len(accepted_bpms) >= 2 else 0.80
        if len(accepted) > 2: accepted = dict(sorted(accepted.items(), key=lambda x: x[1].sqi, reverse=True)[:2])
        cross_roi_harmonic_check(accepted); roi_items=[]
        for roi_name,roi in accepted.items():
            w=compute_dynamic_roi_weight(roi,ROI_CONFIGS[roi_name]["base_weight"],motion_score,result.exposure_drift); roi.dynamic_weight=w; roi_items.append((roi_name,roi,w))
        # Learned Weighting Adjustment
        learned_weights = self.learned_weighting.get_weights()
        roi_items_learned = []
        for name, roi, w in roi_items:
            lw = learned_weights.get(name, 1.0)
            roi_items_learned.append((name, roi, w * lw))
        
        # Explicit Probabilistic Fusion
        roi_data = [(r.bpm, r.sqi) for _, r, _ in roi_items_learned]
        fused_bpm_prob = self.prob_fusion.fuse(roi_data)
        
        # Fallback/Hybrid with hierarchical
        fused_bpm_hier = hierarchical_fusion(roi_items_learned)
        fused_bpm_raw = 0.7 * fused_bpm_prob + 0.3 * fused_bpm_hier
        
        # Update Learned Weights
        roi_sqis = {n: r.sqi for n, r, _ in roi_items_learned}
        roi_bpms = {n: r.bpm for n, r, _ in roi_items_learned}
        self.learned_weighting.update_weights(roi_sqis, fused_bpm_raw, roi_bpms)
        if fused_bpm_raw>0:
            smoothed=self.bpm_smoother.update(fused_bpm_raw); result.fused_bpm=smoothed; self._frozen_bpm=smoothed
            sqi_w_sum=sum(compute_dynamic_roi_weight(r,ROI_CONFIGS[n]["base_weight"],motion_score,result.exposure_drift)*r.sqi for n,r in accepted.items() if r.sqi>0)
            sqi_w_den=sum(compute_dynamic_roi_weight(r,ROI_CONFIGS[n]["base_weight"],motion_score,result.exposure_drift) for n,r in accepted.items() if r.sqi>0)
            result.fused_sqi=sqi_w_sum/sqi_w_den if sqi_w_den>0 else 0.0
            n_valid_rois = len(accepted)
            if n_valid_rois < 2: result.fused_sqi *= cfg.SINGLE_ROI_SQI_MULT
            agr_pen = cfg.AGREEMENT_SQI_PENALTY_COEFF * max(0.0, cfg.AGREEMENT_SQI_PENALTY_BELOW - result.roi_agreement)
            result.fused_sqi *= max(0.0, 1.0 - agr_pen); result.n_valid_rois = n_valid_rois
            sigs=[(np.array(r.buf_g),r.sqi) for r in accepted.values() if r.sqi>0]
            if sigs: result.chrom_signal=max(sigs,key=lambda x:x[1])[0]
        self.session_scorer.update(result.fused_sqi,result.fused_bpm,is_moving,roi_agreement=result.roi_agreement,n_valid_rois=getattr(result,'n_valid_rois',1))
        
        # Uncertainty-Aware Confidence
        self.uncertainty_engine.update(result.fused_bpm, result.fused_sqi, result.roi_agreement)
        conf, meta = self.uncertainty_engine.get_confidence_metrics()
        result.session_confidence = conf
        result.sqi_breakdown.update(meta)
        
        result.roi_signals={k:v for k,v in self.rois.items()}; return result

    def reset(self):
        for roi in self.rois.values(): roi.clear()
        self.bpm_history.clear(); self.fps_buffer.clear(); self.bpm_smoother.reset(); self.session_scorer.reset(); self.motion_detector.reset(); self.calibration.reset(); self.exposure_comp.brightness_hist.clear(); self.exposure_comp.drift=0.0; self.frame_count=0; self.prev_time=time.time(); self._frozen_bpm=0.0


# ─── Skin / quality helpers ───────────────────────────────────────────────────

class SkinConfidenceMap:
    YCRCB_CR_LO,YCRCB_CR_HI=125,185; YCRCB_CB_LO,YCRCB_CB_HI=70,140; HSV_S_LO,HSV_S_HI=15,220; HSV_V_LO,HSV_V_HI=50,255
    def get_skin_mask(self,frame_bgr,polygon_mask):
        ycrcb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2YCrCb); Cr=ycrcb[:,:,1]; Cb=ycrcb[:,:,2]
        mk_y=((Cr>=self.YCRCB_CR_LO)&(Cr<=self.YCRCB_CR_HI)&(Cb>=self.YCRCB_CB_LO)&(Cb<=self.YCRCB_CB_HI)).astype(np.uint8)*255
        hsv=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2HSV); S=hsv[:,:,1]; V=hsv[:,:,2]
        mk_h=((S>=self.HSV_S_LO)&(S<=self.HSV_S_HI)&(V>=self.HSV_V_LO)&(V<=self.HSV_V_HI)).astype(np.uint8)*255
        mf=cv2.bitwise_and(cv2.bitwise_or(mk_y,mk_h),polygon_mask); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        return cv2.morphologyEx(cv2.morphologyEx(mf,cv2.MORPH_OPEN,k,iterations=1),cv2.MORPH_DILATE,k,iterations=1)
    def get_skin_coverage(self,polygon_mask,skin_mask):
        pa=cv2.countNonZero(polygon_mask); return 0.0 if pa==0 else float(cv2.countNonZero(skin_mask)/pa)

class PerPixelQualityFilter:
    def __init__(self,max_saturation=200,min_luminance=40,max_luminance=235):
        self.max_saturation=max_saturation; self.min_luminance=min_luminance; self.max_luminance=max_luminance; self._roi_history={}; self._hist_len=8
    _CANONICAL_H=32; _CANONICAL_W=32
    def filter_roi(self,roi_bgr,roi_name="default"):
        if roi_bgr is None or roi_bgr.size==0 or roi_bgr.shape[0]<4 or roi_bgr.shape[1]<4: return 0.0,0.0,0.0,0.0,None
        h,w=roi_bgr.shape[:2]; hsv=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2HSV); S=hsv[:,:,1].astype(np.float32); V=hsv[:,:,2].astype(np.float32)
        qm=((S<=self.max_saturation)&(V>=self.min_luminance)&(V<=self.max_luminance)).astype(np.uint8)*255
        if roi_name not in self._roi_history: self._roi_history[roi_name]=deque(maxlen=self._hist_len)
        gc=cv2.resize(cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY).astype(np.float32),(self._CANONICAL_W,self._CANONICAL_H),interpolation=cv2.INTER_AREA); self._roi_history[roi_name].append(gc)
        if len(self._roi_history[roi_name])>=4:
            try:
                stack=np.stack(list(self._roi_history[roi_name]),axis=0); tv=cv2.resize(np.std(stack,axis=0),(w,h),interpolation=cv2.INTER_LINEAR)
                qm=cv2.bitwise_and(qm,(tv<=np.percentile(tv,90)).astype(np.uint8)*255)
            except Exception as _exc: _rppg_warn('PerPixelQualityFilter temporal', _exc)
        vc=cv2.countNonZero(qm)
        if vc<5: return float(roi_bgr[:,:,2].mean()),float(roi_bgr[:,:,1].mean()),float(roi_bgr[:,:,0].mean()),0.0,qm
        return float(cv2.mean(roi_bgr[:,:,2],mask=qm)[0]),float(cv2.mean(roi_bgr[:,:,1],mask=qm)[0]),float(cv2.mean(roi_bgr[:,:,0],mask=qm)[0]),vc/max(h*w,1),qm
    def clear(self): self._roi_history.clear()

class ROIMicroMotionDetector:
    def __init__(self): self._prev_roi_grays={}; self._flow_history={}; self._hist_len=10
    def update(self,frame_bgr,roi_name,bbox):
        x1,y1,x2,y2=bbox
        if x2<=x1 or y2<=y1: return 0.0
        roi_bgr=frame_bgr[y1:y2,x1:x2]
        if roi_bgr.size==0: return 0.0
        roi_gray=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY)
        if roi_name not in self._flow_history: self._flow_history[roi_name]=deque(maxlen=self._hist_len); self._prev_roi_grays[roi_name]=roi_gray.copy(); return 0.0
        pg=self._prev_roi_grays[roi_name]
        if pg.shape!=roi_gray.shape:
            try: pg=cv2.resize(pg,(roi_gray.shape[1],roi_gray.shape[0]),interpolation=cv2.INTER_AREA)
            except Exception: self._prev_roi_grays[roi_name]=roi_gray.copy(); return 0.0
        try:
            flow=cv2.calcOpticalFlowFarneback(pg,roi_gray,None,0.5,2,9,2,5,1.1,0); score=float(np.median(np.sqrt(flow[:,:,0]**2+flow[:,:,1]**2)))
        except Exception: score=0.0
        self._prev_roi_grays[roi_name]=roi_gray.copy(); self._flow_history[roi_name].append(score)
        hist=np.array(self._flow_history[roi_name]); wts=np.exp(np.linspace(-1,0,len(hist))); wts/=wts.sum(); return float(np.dot(hist,wts))
    def get_weight_penalty(self,roi_name):
        if roi_name not in self._flow_history or not self._flow_history[roi_name]: return 1.0
        score=float(np.mean(self._flow_history[roi_name]))
        if score<0.3: return 1.0
        elif score<2.0: return 1.0-(score-0.3)/2.3
        else: return max(0.15,1.0-(score-0.3)/2.3)
    def clear(self): self._prev_roi_grays.clear(); self._flow_history.clear()

class MethodArbitrator:
    def __init__(self): self._win_counts={"chrom":0,"pos":0,"green":0}
    def select_best(self,r,g,b,fps):
        eps=1e-6; signals={}
        try:
            r_n=r/(np.mean(r)+eps); g_n=g/(np.mean(g)+eps); b_n=b/(np.mean(b)+eps); Xc=3*r_n-2*g_n; Yc=1.5*r_n+g_n-1.5*b_n; signals["chrom"]=Xc-(np.std(Xc)+eps)/(np.std(Yc)+eps)*Yc
        except Exception: signals["chrom"]=np.zeros(len(g))
        try:
            r_n=r/(np.mean(r)+eps); g_n=g/(np.mean(g)+eps); b_n=b/(np.mean(b)+eps); C=np.vstack([r_n,g_n,b_n]); Pn=np.dot([0,1,-1],C); S2=np.dot([0,1,1],C); s1=np.std(Pn)+eps; s2=np.std(S2)+eps; signals["pos"]=Pn/s1-s1/s2*S2/s2
        except Exception: signals["pos"]=np.zeros(len(g))
        try: signals["green"]=g/(np.mean(g)+eps)-1.0
        except Exception: signals["green"]=np.zeros(len(g))
        nyq=fps/2.0; lo=max(0.7/nyq,1e-6); hi=min(3.0/nyq,1-1e-6); sqi_s={}
        for name,sig in signals.items():
            try:
                sd=scipy_detrend(sig)
                if lo<hi: bc,ac=butter(4,[lo,hi],btype="band"); sf=filtfilt(bc,ac,sd)
                else: sf=sd
                fv=np.abs(np.fft.rfft(sf*np.hanning(len(sf))))**2; fr=np.fft.rfftfreq(len(sf),1/fps); hm=(fr>=0.7)&(fr<=3.0)
                if not np.any(hm): sqi_s[name]=0.0; continue
                ph=fr[hm][np.argmax(fv[hm])]; pp=fv[(fr>=ph-0.08)&(fr<=ph+0.08)].sum(); sb=fv[((fr>=ph-0.4)&(fr<ph-0.09))|((fr>ph+0.09)&(fr<=ph+0.4))].sum()+1e-9; sqi_s[name]=float(pp/sb)
            except Exception: sqi_s[name]=0.0
        winner=max(sqi_s,key=sqi_s.get); self._win_counts[winner]+=1; ranked=sorted(sqi_s,key=sqi_s.get,reverse=True); blended=0.6*signals[ranked[0]]+0.3*signals[ranked[1]]+0.1*signals[ranked[2]]; return blended,winner,sqi_s
    def get_win_stats(self):
        total=sum(self._win_counts.values())+1e-9; return {k:round(v/total,3) for k,v in self._win_counts.items()}

class TimestampResampler:
    def __init__(self,target_fps=30.0): self.target_fps=target_fps; self._timestamps={}; self._maxlen=400
    def push(self,roi_name,ts,r,g,b):
        if roi_name not in self._timestamps: self._timestamps[roi_name]=deque(maxlen=self._maxlen)
        self._timestamps[roi_name].append((ts,r,g,b))
    def get_resampled(self,roi_name):
        if roi_name not in self._timestamps or len(self._timestamps[roi_name])<10: return np.array([]),np.array([]),np.array([]),self.target_fps
        data=list(self._timestamps[roi_name]); ts_arr=np.array([d[0] for d in data]); r_arr=np.array([d[1] for d in data]); g_arr=np.array([d[2] for d in data]); b_arr=np.array([d[3] for d in data]); dur=ts_arr[-1]-ts_arr[0]
        if dur<1.0: return r_arr,g_arr,b_arr,self.target_fps
        ns=int(dur*self.target_fps)
        if ns<10: return r_arr,g_arr,b_arr,self.target_fps
        t_u=np.linspace(ts_arr[0],ts_arr[-1],ns); return np.interp(t_u,ts_arr,r_arr),np.interp(t_u,ts_arr,g_arr),np.interp(t_u,ts_arr,b_arr),float(ns/dur)
    def clear(self): self._timestamps.clear()

class RollingConfidenceTrend:
    def __init__(self,window_sec=20,fps=30.0): self._sqi_history=deque(maxlen=int(window_sec*fps))
    def push(self,sqi): self._sqi_history.append(sqi)
    def get_trend(self):
        if len(self._sqi_history)<10: return {"trend":"insufficient_data","mean":0.0,"slope":0.0}
        arr=np.array(self._sqi_history); mean=float(np.mean(arr)); poly=np.polyfit(np.arange(len(arr)),arr,1) if len(arr)>2 else [0,0]; slope=float(poly[0])*30.0; trend="IMPROVING" if slope>0.5 else "DEGRADING" if slope<-0.5 else "STABLE"
        return {"trend":trend,"mean":round(mean,1),"slope_per_sec":round(slope,2),"stability":round(1.0-np.std(arr)/max(mean,1.0),3),"n_frames":len(arr)}

class RespiratoryArtifactSuppressor:
    def __init__(self): self._resp_freq_hz=0.25; self._resp_history=deque(maxlen=30)
    def update_resp_freq(self,signal,fps):
        try:
            nyq=fps/2.0; lo=max(0.1/nyq,1e-6); hi=min(0.5/nyq,1-1e-6)
            if lo>=hi: return
            br,ar=butter(3,[lo,hi],btype="band"); resp=filtfilt(br,ar,signal); freqs,psd=welch(resp,fs=fps,nperseg=min(len(resp),128)); mask=(freqs>=0.1)&(freqs<=0.5)
            if np.any(mask): self._resp_freq_hz=float(freqs[mask][np.argmax(psd[mask])]); self._resp_history.append(self._resp_freq_hz)
            if self._resp_history: self._resp_freq_hz=float(np.median(self._resp_history))
        except Exception as _exc: _rppg_warn('RespiratoryArtifactSuppressor.update_resp_freq', _exc)
    def suppress(self,signal,fps):
        try:
            nyq=fps/2.0
            if self._resp_freq_hz<=0 or self._resp_freq_hz>=nyq: return signal
            fn=self._resp_freq_hz/nyq
            if 0.01<fn<0.99: bn,an=iirnotch(fn,12.0); signal=filtfilt(bn,an,signal)
            h2=self._resp_freq_hz*2.0
            if FREQ_LOW<=h2<=FREQ_HIGH and h2<nyq:
                h2n=h2/nyq
                if 0.01<h2n<0.99: bh,ah=iirnotch(h2n,9.6); signal=filtfilt(bh,ah,signal)
        except Exception as _exc: _rppg_warn('RespiratoryArtifactSuppressor.suppress -- returning unfiltered signal', _exc)
        return signal

class HeadPoseGate:
    def __init__(self,max_yaw_deg=50.0,max_pitch_deg=40.0): self.max_yaw=max_yaw_deg; self.max_pitch=max_pitch_deg; self._yaw_history=deque(maxlen=8); self._pitch_history=deque(maxlen=8)
    def update(self,face_landmarks,h,w):
        if face_landmarks is None: return 0.0,0.0,True
        try:
            lm=face_landmarks.landmark; mid_ear=(lm[234].x+lm[454].x)/2.0*w; ew=abs(lm[454].x-lm[234].x)*w+1e-6; yaw_deg=float((lm[1].x*w-mid_ear)/ew*90.0); pitch_deg=float((lm[1].y-lm[10].y)/(lm[152].y-lm[10].y+1e-6)*90.0-45.0); self._yaw_history.append(abs(yaw_deg)); self._pitch_history.append(abs(pitch_deg)); sy=float(np.median(self._yaw_history)); sp=float(np.median(self._pitch_history)); return sy,sp,(sy<=self.max_yaw and sp<=self.max_pitch)
        except Exception: return 0.0,0.0,True

class JawBlinkSuppressor:
    UPPER_LIP=13;LOWER_LIP=14;L_EYE_TOP=159;L_EYE_BOT=145;R_EYE_TOP=386;R_EYE_BOT=374
    def __init__(self,jaw_open_threshold=0.05,blink_threshold=0.012): self.jaw_open_threshold=jaw_open_threshold; self.blink_threshold=blink_threshold; self._jaw_history=deque(maxlen=10); self._blink_history=deque(maxlen=6); self._jaw_baseline=None; self._talking_frames=0; self._init_frames=0
    def update(self,face_landmarks,h,w):
        r={"talking":False,"blinking":False,"jaw_open_normalized":0.0,"eye_aperture":0.0,"suppress_cheeks":False}
        if face_landmarks is None: return r
        try:
            lm=face_landmarks.landmark; self._init_frames+=1; fh=abs(lm[152].y-lm[10].y)*h+1e-6; jaw_open=abs(lm[self.LOWER_LIP].y-lm[self.UPPER_LIP].y)*h/fh; self._jaw_history.append(jaw_open)
            if self._jaw_baseline is None and len(self._jaw_history)>=10: self._jaw_baseline=float(np.percentile(self._jaw_history,20))
            baseline=self._jaw_baseline if self._jaw_baseline is not None else jaw_open
            if jaw_open-baseline>self.jaw_open_threshold: self._talking_frames+=1
            else: self._talking_frames=max(0,self._talking_frames-1)
            r["talking"]=self._talking_frames>=3; r["jaw_open_normalized"]=round(float(jaw_open),4); la=abs(lm[self.L_EYE_TOP].y-lm[self.L_EYE_BOT].y)*h/fh; ra=abs(lm[self.R_EYE_TOP].y-lm[self.R_EYE_BOT].y)*h/fh; eye_ap=(la+ra)/2.0; self._blink_history.append(eye_ap); r["blinking"]=eye_ap<self.blink_threshold; r["eye_aperture"]=round(float(eye_ap),4); r["suppress_cheeks"]=(r["talking"] or r["blinking"]) if self._init_frames>=20 else False
        except Exception as _exc: _rppg_warn('JawBlinkSuppressor.update -- returning default dict', _exc)
        return r

class ReproducibilityLogger:
    def __init__(self,session_id=None):
        import os; self.session_id=session_id or str(int(time.time())); self.out_dir=f"rppg_session_{self.session_id}"; os.makedirs(self.out_dir,exist_ok=True); self._raw_log=[]; self._sqi_log=[]; self._event_log=[]; self._max_log=10000
    def log_frame(self,ts,roi_name,r,g,b,bpm,sqi):
        if len(self._raw_log)<self._max_log: self._raw_log.append((ts,roi_name,r,g,b,bpm,sqi))
    def log_sqi(self,ts,roi_name,sqi,breakdown):
        if len(self._sqi_log)<self._max_log: self._sqi_log.append((ts,roi_name,sqi,str(breakdown)))
    def log_event(self,event,detail="",value=0.0): self._event_log.append((time.time(),event,detail,value))
    def save(self):
        import csv,os
        for path,rows,header in [(os.path.join(self.out_dir,"raw_signals.csv"),self._raw_log,["timestamp","roi_name","r","g","b","bpm","sqi"]),(os.path.join(self.out_dir,"sqi_log.csv"),self._sqi_log,["timestamp","roi_name","sqi","breakdown"]),(os.path.join(self.out_dir,"event_log.csv"),self._event_log,["timestamp","event","detail","value"])]:
            with open(path,"w",newline="") as f: w=csv.writer(f); w.writerow(header); w.writerows(rows)
        return self.out_dir
    def clear(self): self._raw_log.clear(); self._sqi_log.clear(); self._event_log.clear()


# ─── V2 Engine: inherits updated V1, adds all advanced subsystems ─────────────

class MultiROIFusionEngineV2(MultiROIFusionEngine):
    def __init__(self,enable_logging=False):
        super().__init__(); self.skin_map=SkinConfidenceMap(); self.pixel_filter=PerPixelQualityFilter(); self.micro_motion=ROIMicroMotionDetector(); self.arbitrator=MethodArbitrator(); self.resampler=TimestampResampler(target_fps=30.0); self.conf_trend=RollingConfidenceTrend(window_sec=20); self.resp_suppressor=RespiratoryArtifactSuppressor(); self.pose_gate=HeadPoseGate(); self.jaw_suppressor=JawBlinkSuppressor(); self.repro_logger=ReproducibilityLogger() if enable_logging else None; self.enable_logging=enable_logging; self._last_jaw_result={}; self._last_pose=(0.0,0.0,True)

    def update(self,frame,face_landmarks,h,w):
        result=FrameResult(); self.frame_count+=1; now=time.time(); dt=now-self.prev_time; self.prev_time=now
        self.fps_buffer.append(1.0/dt if dt>1e-9 else 30.0); fps=float(np.mean(self.fps_buffer))
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); motion_score,is_moving=self.motion_detector.update(frame_gray,face_landmarks,h,w); result.motion_score=motion_score; result.motion_rejected=is_moving
        norm_brightness,drift_detected,drift_mag=self.exposure_comp.update(float(frame.mean())); result.frame_brightness=norm_brightness; result.exposure_drift=drift_mag; result.brightness_normalized=drift_detected
        if drift_mag>=cfg.EXPOSURE_DRIFT_FREEZE:
            result.output_frozen=True; result.fused_bpm=self._frozen_bpm; result.roi_signals={k:v for k,v in self.rois.items()}; return result
        if face_landmarks is None: return result
        yaw,pitch,pose_ok=self.pose_gate.update(face_landmarks,h,w); self._last_pose=(yaw,pitch,pose_ok)
        if not pose_ok: result.roi_signals={k:v for k,v in self.rois.items()}; return result
        jaw_result=self.jaw_suppressor.update(face_landmarks,h,w); self._last_jaw_result=jaw_result; suppress_cheeks=jaw_result.get("suppress_cheeks",False)
        nf_r,nf_g,nf_b=self.skin_calibrator.get_normalization_factors()
        for roi_name,cfg_roi in ROI_CONFIGS.items():
            roi=self.rois[roi_name]
            if suppress_cheeks and roi_name!="forehead": roi.valid=False; continue
            out=get_roi_polygon_masked(frame,face_landmarks,cfg_roi["landmarks"],h,w)
            if out is None: roi.valid=False; continue
            pts_arr,mr_raw,mg_raw,mb_raw,bbox,roi_bright=out; pm=np.zeros((h,w),dtype=np.uint8); cv2.fillPoly(pm,[pts_arr],255); sm=self.skin_map.get_skin_mask(frame,pm); sc=self.skin_map.get_skin_coverage(pm,sm); roi.skin_coverage=sc
            if sc<0.20: roi.valid=False; continue
            roi.pts=pts_arr; roi.bbox=bbox; roi.valid=True; roi.roi_brightness=roi_bright; self.micro_motion.update(frame,roi_name,bbox)
            if not is_moving and roi_bright<=220 and roi_bright>=40:
                x1c=max(0,bbox[0]); y1c=max(0,bbox[1]); x2c=min(frame.shape[1],bbox[2]); y2c=min(frame.shape[0],bbox[3]); rc=frame[y1c:y2c,x1c:x2c]
                if rc.size>0 and rc.shape[0]>=4 and rc.shape[1]>=4:
                    mr,mg,mb,vf,_=self.pixel_filter.filter_roi(rc,roi_name)
                    if vf<0.1 or mr==0.0: mr,mg,mb=mr_raw,mg_raw,mb_raw
                else: mr,mg,mb=mr_raw,mg_raw,mb_raw
                roi.buf_r.append(mr*nf_r); roi.buf_g.append(mg*nf_g); roi.buf_b.append(mb*nf_b); self.resampler.push(roi_name,now,mr*nf_r,mg*nf_g,mb*nf_b); self.skin_calibrator.update(mr,mg,mb)
            elif not is_moving: self.skin_calibrator.update(mr_raw,mg_raw,mb_raw)
        result.skin_tone_factor=nf_g
        valid_rois={}
        for k,v in self.rois.items():
            if len(v.buf_g)<self.MIN_FRAMES or v.skin_coverage<0.18: continue
            if v.roi_brightness<cfg.ROI_BRIGHTNESS_MIN or v.roi_brightness>cfg.ROI_BRIGHTNESS_MAX: continue
            # v12: SNR gate removed; regularity gate disabled (REGULARITY_HARD_GATE=0)
            if v.bpm>0 and not (cfg.BPM_PLAUSIBLE_LOW<=v.bpm<=cfg.BPM_PLAUSIBLE_HIGH): continue
            valid_rois[k]=v
        valid_rois_res={}
        for k,v in valid_rois.items():
            r_re,g_re,b_re,eff_fps=self.resampler.get_resampled(k)
            if len(g_re)>=self.MIN_FRAMES: valid_rois_res[k]=(v,r_re,g_re,b_re,eff_fps)
        if not valid_rois_res:
            for k,v in valid_rois.items():
                ar=np.array(v.buf_r); ag=np.array(v.buf_g); ab=np.array(v.buf_b)
                if len(ag)>=self.MIN_FRAMES: valid_rois_res[k]=(v,ar,ag,ab,fps)
        if not valid_rois_res: result.roi_signals={k:v for k,v in self.rois.items()}; return result
        weighted_sigs=[]
        for roi_name,(roi,arr_r,arr_g,arr_b,eff_fps) in valid_rois_res.items():
            try:
                self.resp_suppressor.update_resp_freq(arr_g,eff_fps); sc,_,_=self.arbitrator.select_best(arr_r,arr_g,arr_b,eff_fps); sd=scipy_detrend(sc,type="linear"); sd=self.exposure_comp.normalize_roi_signal(sd); sd=self.resp_suppressor.suppress(sd,eff_fps); sf=bandpass_filter(sd,eff_fps)
                _std2=float(np.std(sf))
                if _std2>1e-8: sf=(sf-np.mean(sf))/_std2
                bpm_raw,freqs,power,fft_val=estimate_bpm_fft(sf,eff_fps,roi_name=roi_name)
                sqi,breakdown=compute_sqi(sf,eff_fps,bpm_raw,motion_score=motion_score,mean_brightness=result.frame_brightness,roi_brightness=roi.roi_brightness,fft_validation=fft_val,prev_sqi=roi.sqi,roi_obj=roi,exposure_drift=result.exposure_drift)
                roi.bpm=bpm_raw; roi.sqi=sqi; roi.roi_snr=breakdown.get("snr",0.0); roi.roi_regularity=breakdown.get("regularity",0.0)
                if bpm_raw>0: roi._bpm_history.append(bpm_raw)
                self.conf_trend.push(sqi)
                if self.enable_logging and self.repro_logger: self.repro_logger.log_frame(now,roi_name,float(arr_r[-1]),float(arr_g[-1]),float(arr_b[-1]),bpm_raw,sqi); self.repro_logger.log_sqi(now,roi_name,sqi,breakdown)
                if sqi>result.sqi_breakdown.get("overall",0) and len(sf)>0: result.freqs=freqs; result.power=power; result.sqi_breakdown=breakdown
                if len(sf)>0 and sqi>0: weighted_sigs.append((sf,sqi,roi_name))
            except Exception as _exc:
                roi.sqi = 0.0
                _rppg_warn(f'ROI processing (frame {getattr(self,"frame_count",0)})', _exc)
        bs=max((v[0].sqi for v in valid_rois_res.values()),default=0.0); bb=max((v[0].bpm for v in valid_rois_res.values() if BPM_LOW<=v[0].bpm<=BPM_HIGH),default=0.0); result.in_calibration=self.calibration.update(bs,bb)
        avb=[v[0].bpm for v in valid_rois_res.values() if BPM_LOW<=v[0].bpm<=BPM_HIGH and v[0].sqi>=SQI_HARD_GATE]
        if len(avb)>=2:
            cm=float(np.median(avb))
            for k_h,v_h in valid_rois_res.items():
                roi_h=v_h[0]
                if not(BPM_LOW<=roi_h.bpm<=BPM_HIGH and roi_h.sqi>=SQI_HARD_GATE): continue
                ratio=roi_h.bpm/(cm+1e-6)
                if 1.85<=ratio<=2.15 and cm>=BPM_LOW:
                    hb=roi_h.bpm/2.0
                    if BPM_LOW<=hb<=BPM_HIGH: roi_h.bpm=hb
                elif 2.8<=ratio<=3.2 and cm>=BPM_LOW:
                    tb=roi_h.bpm/3.0
                    if BPM_LOW<=tb<=BPM_HIGH: roi_h.bpm=tb
        cands_v2 = {k: v for k, v in valid_rois_res.items() if BPM_LOW <= v[0].bpm <= BPM_HIGH and v[0].sqi >= SQI_HARD_GATE}
        if not cands_v2: result.roi_signals={k:v for k,v in self.rois.items()}; return result
        ref_v2 = max(cands_v2.values(), key=lambda v: v[0].sqi); ref_bpm_v2 = ref_v2[0].bpm; HARD_OUT_V2 = cfg.FUSION_OUTLIER_THRESHOLD + 10.0; cands_v2 = {k: v for k, v in cands_v2.items() if abs(v[0].bpm - ref_bpm_v2) <= HARD_OUT_V2}
        if not cands_v2: cands_v2 = {list(valid_rois_res.keys())[0]: ref_v2}
        for k_a, (roi_a, *_) in cands_v2.items():
            dev = abs(roi_a.bpm - ref_bpm_v2); roi_a.agreement_factor = float(np.exp(-dev / cfg.AGREEMENT_DEV_K))
        accepted = {k: v for k, v in cands_v2.items() if v[0].agreement_factor >= cfg.ROI_MIN_AGREEMENT_FACTOR}
        if not accepted: accepted = {list(cands_v2.keys())[0]: ref_v2}
        accepted_bpms_v2 = [v[0].bpm for v in accepted.values()]; result.roi_agreement = (float(np.exp(-np.std(accepted_bpms_v2) / cfg.AGREEMENT_STD_K)) if len(accepted_bpms_v2) >= 2 else 0.80)
        accepted_rois_only = {k: v[0] for k, v in accepted.items()}; cross_roi_harmonic_check(accepted_rois_only); roi_items=[]
        for roi_name,(roi,*_) in accepted.items():
            mp=self.micro_motion.get_weight_penalty(roi_name); w=compute_dynamic_roi_weight(roi,ROI_CONFIGS[roi_name]["base_weight"],motion_score,result.exposure_drift)*mp; roi.dynamic_weight=w; roi_items.append((roi_name,roi,w))
        # Learned Weighting Adjustment
        learned_weights = self.learned_weighting.get_weights()
        roi_items_learned = []
        for name, roi, w in roi_items:
            lw = learned_weights.get(name, 1.0)
            roi_items_learned.append((name, roi, w * lw))
        
        # Explicit Probabilistic Fusion
        roi_data = [(r.bpm, r.sqi) for _, r, _ in roi_items_learned]
        fused_bpm_prob = self.prob_fusion.fuse(roi_data)
        
        # Fallback/Hybrid with hierarchical
        fused_bpm_hier = hierarchical_fusion(roi_items_learned)
        fused_bpm_raw = 0.7 * fused_bpm_prob + 0.3 * fused_bpm_hier
        
        # Update Learned Weights
        roi_sqis = {n: r.sqi for n, r, _ in roi_items_learned}
        roi_bpms = {n: r.bpm for n, r, _ in roi_items_learned}
        self.learned_weighting.update_weights(roi_sqis, fused_bpm_raw, roi_bpms)
        if fused_bpm_raw>0:
            smoothed=self.bpm_smoother.update(fused_bpm_raw); result.fused_bpm=smoothed; self._frozen_bpm=smoothed; sws=sum(compute_dynamic_roi_weight(v[0],ROI_CONFIGS[k]["base_weight"],motion_score,result.exposure_drift)*v[0].sqi for k,v in accepted.items() if v[0].sqi>=SQI_HARD_GATE); swd=sum(compute_dynamic_roi_weight(v[0],ROI_CONFIGS[k]["base_weight"],motion_score,result.exposure_drift) for k,v in accepted.items() if v[0].sqi>=SQI_HARD_GATE); result.fused_sqi=sws/swd if swd>0 else 0.0
            n_valid_rois_v2 = len(accepted)
            if n_valid_rois_v2 < 2: result.fused_sqi *= cfg.SINGLE_ROI_SQI_MULT
            agr_pen = cfg.AGREEMENT_SQI_PENALTY_COEFF * max(0.0, cfg.AGREEMENT_SQI_PENALTY_BELOW - result.roi_agreement); result.fused_sqi *= max(0.0, 1.0 - agr_pen); result.n_valid_rois = n_valid_rois_v2
            if weighted_sigs: result.chrom_signal=max(weighted_sigs,key=lambda x:x[1])[0]
        self.session_scorer.update(result.fused_sqi,result.fused_bpm,is_moving,roi_agreement=result.roi_agreement); result.session_confidence=self.session_scorer.get_confidence(); result.roi_signals={k:v for k,v in self.rois.items()}; return result

    def get_diagnostics(self):
        return {"method_wins":self.arbitrator.get_win_stats(),"confidence_trend":self.conf_trend.get_trend(),"skin_type":self.skin_calibrator.skin_type_label(),"session_confidence":self.session_scorer.get_confidence(),"calibration_done":self.calibration.done,"baseline_sqi":round(self.calibration.baseline_sqi,1),"last_pose":{"yaw":round(self._last_pose[0],1),"pitch":round(self._last_pose[1],1),"ok":self._last_pose[2]},"jaw_state":self._last_jaw_result,"per_roi_sqi":{k:round(v.sqi,1) for k,v in self.rois.items()},"per_roi_weight":{k:round(v.dynamic_weight,3) for k,v in self.rois.items()}}

    def reset(self):
        super().reset(); self.micro_motion.clear(); self.pixel_filter.clear(); self.resampler.clear(); self.conf_trend._sqi_history.clear(); self.resp_suppressor._resp_history.clear(); self.pose_gate._yaw_history.clear(); self.pose_gate._pitch_history.clear(); self.jaw_suppressor._jaw_history.clear(); self.jaw_suppressor._blink_history.clear(); self.jaw_suppressor._jaw_baseline=None; self.jaw_suppressor._talking_frames=0; self.jaw_suppressor._init_frames=0; self._last_jaw_result={}; self._last_pose=(0.0,0.0,True)
        if self.repro_logger: self.repro_logger.clear()
