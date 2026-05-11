from rppg_config import cfg
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import pearsonr
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import time
# Bug fix: was hardcoded 50.0 — out of sync with cfg.BASELINE_SQI_THRESHOLD (45.0).
# Using cfg ensures the gate matches the rest of the system. When both values
# differ, calibration rejects frames that every other module already accepted.
BASELINE_SQI_GATE = cfg.BASELINE_SQI_THRESHOLD  # 45.0

@dataclass
class VitalsResult:
    bpm: float = 0.0
    resp_rate: float = 0.0
    hrv_sdnn: float = 0.0
    hrv_rmssd: float = 0.0
    hrv_pnn50: float = 0.0
    lf_hf_ratio: float = 0.0
    stress_index: float = 0.0
    arrhythmia_flag: bool = False
    arrhythmia_detail: str = ''
    sqi: float = 0.0
    confidence_label: str = 'LOW'
    mae_estimate: float = 0.0
    pearson_r: float = 0.0
    session_confidence: float = 0.0

@dataclass
class HealthBaseline:
    bpm_mean: float = 72.0
    bpm_std: float = 8.0
    hrv_sdnn_mean: float = 50.0
    resp_mean: float = 15.0
    stress_mean: float = 30.0
    samples_count: int = 0
    last_updated: float = 0.0

class GroundTruthInterface:
    SOURCES = ['manual', 'pulse_oximeter', 'smartwatch_ppg', 'ecg']

    def __init__(self):
        self.gt_bpm_series: list = []
        self.est_bpm_series: list = []
        self.current_gt_bpm: Optional[float] = None
        self.current_source: str = 'manual'
        self._last_update: float = 0.0

    def set_ground_truth(self, bpm: float, source: str='manual'):
        assert source in self.SOURCES, f'Unknown GT source: {source}'
        now = time.time()
        self.current_gt_bpm = bpm
        self.current_source = source
        self.gt_bpm_series.append((now, bpm, source))
        self._last_update = now

    def record_estimate(self, bpm: float):
        if bpm > 0:
            self.est_bpm_series.append((time.time(), bpm))

    def compute_bland_altman(self) -> dict:
        if len(self.gt_bpm_series) < 5 or len(self.est_bpm_series) < 5:
            return {'error': 'insufficient_data', 'n': min(len(self.gt_bpm_series), len(self.est_bpm_series))}
        gt_bpms = [g[1] for g in self.gt_bpm_series]
        est_bpms = [e[1] for e in self.est_bpm_series]
        n = min(len(gt_bpms), len(est_bpms))
        gt_arr = np.array(gt_bpms[-n:])
        est_arr = np.array(est_bpms[-n:])
        diffs = est_arr - gt_arr
        means = (est_arr + gt_arr) / 2.0
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        return {'n': n, 'mean_diff_bpm': round(mean_diff, 2), 'std_diff_bpm': round(std_diff, 2), 'loa_upper': round(loa_upper, 2), 'loa_lower': round(loa_lower, 2), 'mae': round(float(np.mean(np.abs(diffs))), 2), 'rmse': round(float(np.sqrt(np.mean(diffs ** 2))), 2), 'source': self.current_source}

    def compute_pearson(self) -> dict:
        gt_bpms = [g[1] for g in self.gt_bpm_series]
        est_bpms = [e[1] for e in self.est_bpm_series]
        n = min(len(gt_bpms), len(est_bpms))
        if n < 3:
            return {'error': 'insufficient_data'}
        try:
            r, p = pearsonr(gt_bpms[-n:], est_bpms[-n:])
            return {'pearson_r': round(float(r), 4), 'p_value': round(float(p), 4), 'n': n}
        except Exception as e:
            return {'error': str(e)}

    def clear(self):
        self.gt_bpm_series.clear()
        self.est_bpm_series.clear()
        self.current_gt_bpm = None

class FailureModeLogger:

    def __init__(self, maxlen: int=500):
        self.events: deque = deque(maxlen=maxlen)
        self.counts = {'motion_rejected': 0, 'sqi_gated': 0, 'fft_harmonic_reject': 0, 'exposure_drift': 0, 'arrhythmia_suppressed': 0, 'low_snr': 0, 'baseline_rejected': 0, 'total_frames': 0}

    def log(self, event_type: str, detail: str='', value: float=0.0):
        ts = time.time()
        self.events.append({'ts': ts, 'type': event_type, 'detail': detail, 'value': value})
        if event_type in self.counts:
            self.counts[event_type] += 1

    def tick(self):
        self.counts['total_frames'] += 1

    def get_summary(self) -> dict:
        n = max(self.counts['total_frames'], 1)
        return {'total_frames': n, 'motion_reject_rate': round(self.counts['motion_rejected'] / n, 3), 'sqi_gate_rate': round(self.counts['sqi_gated'] / n, 3), 'fft_harmonic_rate': round(self.counts['fft_harmonic_reject'] / n, 3), 'exposure_drift_rate': round(self.counts['exposure_drift'] / n, 3), 'arrhythmia_suppressed': self.counts['arrhythmia_suppressed'], 'low_snr_rate': round(self.counts['low_snr'] / n, 3), 'baseline_rejected': self.counts['baseline_rejected']}

    def recent_events(self, n: int=20) -> list:
        return list(self.events)[-n:]

class VitalsEngine:

    def __init__(self, fps: float=30.0):
        self.fps = fps
        # Bug fix: was hardcoded 120 -- mismatch with cfg.MIN_FRAMES (60).
        # At 60/60 frames the engine returned early so BPM/HRV never appeared
        # until the buffer hit 120. Single source of truth: cfg.MIN_FRAMES.
        self.MIN_FRAMES = cfg.MIN_FRAMES  # 60
        self.ibi_history = deque(maxlen=200)
        self.bpm_history = deque(maxlen=30)
        self.resp_history = deque(maxlen=30)
        self.stress_history = deque(maxlen=20)
        self.signal_buffer = deque(maxlen=300)
        self.resp_buffer = deque(maxlen=300)
        self.prev_peaks = deque(maxlen=50)
        self.baseline = HealthBaseline()
        self._bpm_window = deque(maxlen=60)
        self.ground_truth = GroundTruthInterface()
        self.failure_log = FailureModeLogger()
        self._arrhythmia_cooldown_until: float = 0.0
        self._arrhythmia_candidate_count: int = 0
        self._arrhythmia_windows: deque = deque(maxlen=10)

    def update_fps(self, fps: float):
        self.fps = max(1.0, fps)

    def ingest_signal(self, chrom_signal: np.ndarray, fused_bpm: float, sqi: float):
        for v in chrom_signal[-5:]:
            self.signal_buffer.append(float(v))
        for v in chrom_signal[-5:]:
            self.resp_buffer.append(float(v))
        if fused_bpm > 0:
            self._bpm_window.append(fused_bpm)
            self.ground_truth.record_estimate(fused_bpm)

    def compute(self, chrom_signal: np.ndarray, fused_bpm: float, sqi: float) -> VitalsResult:
        result = VitalsResult(bpm=fused_bpm, sqi=sqi)
        if sqi >= 65:
            result.confidence_label = 'GOOD'
        elif sqi >= 40:
            result.confidence_label = 'FAIR'
        else:
            result.confidence_label = 'LOW'
        if len(chrom_signal) < self.MIN_FRAMES or fused_bpm <= 0:
            return result
        result.resp_rate = self._compute_respiration(chrom_signal)
        ibi_arr = self._extract_ibi(chrom_signal, fused_bpm)
        if len(ibi_arr) >= 5:
            result.hrv_sdnn = self._sdnn(ibi_arr)
            result.hrv_rmssd = self._rmssd(ibi_arr)
            result.hrv_pnn50 = self._pnn50(ibi_arr)
            result.lf_hf_ratio = self._lf_hf(ibi_arr)
            result.arrhythmia_flag, result.arrhythmia_detail = self._arrhythmia_check_sustained(ibi_arr, fused_bpm, sqi)
            result.stress_index = self._stress_index(fused_bpm, result.hrv_sdnn, result.lf_hf_ratio, result.resp_rate)
        return result

    def _compute_respiration(self, signal: np.ndarray) -> float:
        try:
            fps = self.fps
            nyq = fps / 2.0
            lo = max(0.1 / nyq, 1e-05)
            hi = min(0.5 / nyq, 0.999)
            if lo >= hi:
                return 0.0
            b, a = butter(3, [lo, hi], btype='band')
            resp = filtfilt(b, a, signal)
            freqs, psd = welch(resp, fs=fps, nperseg=min(len(resp), 128))
            mask = (freqs >= 0.1) & (freqs <= 0.5)
            if not np.any(mask):
                return 0.0
            peak_f = freqs[mask][np.argmax(psd[mask])]
            rr = peak_f * 60.0
            if 6 <= rr <= 30:
                self.resp_history.append(rr)
                return float(np.median(self.resp_history))
        except Exception:
            pass
        return float(np.median(self.resp_history)) if self.resp_history else 0.0

    def _extract_ibi(self, signal: np.ndarray, bpm: float) -> np.ndarray:
        try:
            fps = self.fps
            min_dist = int(fps * 0.35)
            sig_norm = (signal - signal.mean()) / (signal.std() + 1e-09)
            peaks, _ = find_peaks(sig_norm, distance=min_dist, prominence=0.25)
            if len(peaks) < 3:
                return np.array([])
            ibi_samples = np.diff(peaks)
            ibi_ms = ibi_samples / fps * 1000.0
            med = np.median(ibi_ms)
            ibi_ms = ibi_ms[(ibi_ms > med * 0.6) & (ibi_ms < med * 1.6)]
            for v in ibi_ms:
                self.ibi_history.append(v)
            return ibi_ms
        except Exception:
            return np.array([])

    def _sdnn(self, ibi: np.ndarray) -> float:
        return float(np.std(ibi))

    def _rmssd(self, ibi: np.ndarray) -> float:
        if len(ibi) < 2:
            return 0.0
        diffs = np.diff(ibi)
        return float(np.sqrt(np.mean(diffs ** 2)))

    def _pnn50(self, ibi: np.ndarray) -> float:
        if len(ibi) < 2:
            return 0.0
        diffs = np.abs(np.diff(ibi))
        return float(np.mean(diffs > 50.0) * 100.0)

    def _lf_hf(self, ibi: np.ndarray) -> float:
        try:
            if len(ibi) < 20:
                return 0.0
            fps_ibi = 4.0
            t_orig = np.cumsum(ibi) / 1000.0
            t_resamp = np.arange(t_orig[0], t_orig[-1], 1.0 / fps_ibi)
            ibi_resamp = np.interp(t_resamp, t_orig, ibi)
            freqs, psd = welch(ibi_resamp, fs=fps_ibi, nperseg=min(len(ibi_resamp), 64))
            _trapz = getattr(np, 'trapezoid', np.trapz)
            lf = _trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
            hf = _trapz(psd[(freqs >= 0.15) & (freqs < 0.4)])
            if hf < 1e-09:
                return 0.0
            ratio = lf / hf
            return round(float(ratio), 3)
        except Exception:
            return 0.0

    def _arrhythmia_check_sustained(self, ibi: np.ndarray, bpm: float, sqi: float) -> tuple:
        if sqi < cfg.ARRHYTHMIA_SQI_GATE:  # Fix #8: only active when signal stable
            return (False, '')
        if len(ibi) < 10:
            return (False, '')
        now = time.time()
        if now < self._arrhythmia_cooldown_until:
            self.failure_log.log('arrhythmia_suppressed', 'cooldown active')
            return (False, '')
        window_size = 5
        anomaly_votes = 0
        total_windows = 0
        for i in range(len(ibi) - window_size + 1):
            seg = ibi[i:i + window_size]
            cv = np.std(seg) / (np.mean(seg) + 1e-09)
            max_jump = np.max(np.abs(np.diff(seg)))
            is_anomaly = cv > 0.28 or max_jump > 180
            if is_anomaly:
                anomaly_votes += 1
            total_windows += 1
        if total_windows > 0:
            anomaly_ratio = anomaly_votes / total_windows
            self._arrhythmia_windows.append(anomaly_ratio)
        else:
            return (False, '')
        recent_windows = list(self._arrhythmia_windows)[-5:]
        sustained_count = sum((1 for r in recent_windows if r > 0.4))
        sustained = sustained_count >= 3 and len(recent_windows) >= 3
        if not sustained:
            return (False, '')
        details = []
        flags = []
        cv = np.std(ibi) / (np.mean(ibi) + 1e-09)
        if cv > 0.3:
            details.append('sustained high RR variability')
            flags.append('HRV_HIGH')
        diffs = np.abs(np.diff(ibi))
        large_jumps = int(np.sum(diffs > 180))
        if large_jumps >= 3:
            details.append(f'repeated RR irregularity ({large_jumps} events)')
            flags.append('RR_JUMP')
        rr_bpm = 60000.0 / (np.median(ibi) + 1e-09)
        if bpm < 45 and rr_bpm < 48:
            details.append('possible bradycardia')
            flags.append('BRADY')
        elif bpm > 150 and rr_bpm > 148:
            details.append('possible tachycardia')
            flags.append('TACHY')
        if len(set(flags)) >= 2:
            self._arrhythmia_cooldown_until = now + 45.0
            self._arrhythmia_candidate_count += 1
            return (True, ' | '.join(details))
        if bpm < 38 or bpm > 175:
            self._arrhythmia_cooldown_until = now + 30.0
            return (True, details[0] if details else f'extreme HR: {bpm:.0f} BPM')
        return (False, '')

    def _stress_index(self, bpm: float, sdnn: float, lf_hf: float, resp: float) -> float:
        score = 0.0
        bpm_score = 0.0
        if bpm > 100:
            bpm_score = min((bpm - 100) / 40.0, 1.0)
        elif bpm < 55:
            bpm_score = min((55 - bpm) / 15.0, 1.0)
        score += bpm_score * 30.0
        hrv_score = 0.0
        if sdnn > 0:
            hrv_score = max(0.0, 1.0 - sdnn / 80.0)
        score += hrv_score * 35.0
        lf_score = 0.0
        if lf_hf > 0:
            lf_score = min(lf_hf / 4.0, 1.0)
        score += lf_score * 20.0
        resp_score = 0.0
        if resp > 0:
            if resp > 20:
                resp_score = min((resp - 20) / 10.0, 1.0)
            elif resp < 10:
                resp_score = min((10 - resp) / 5.0, 0.5)
        score += resp_score * 15.0
        self.stress_history.append(score)
        return round(float(np.mean(self.stress_history)), 1)

    def update_baseline(self, result: VitalsResult):
        if result.sqi < BASELINE_SQI_GATE or result.bpm <= 0:
            self.failure_log.log('baseline_rejected', f'SQI={result.sqi:.1f} < {BASELINE_SQI_GATE}')
            return
        bl = self.baseline
        lr = 0.12 if bl.samples_count < 30 else 0.04
        old_mean = bl.bpm_mean
        bl.bpm_mean = bl.bpm_mean * (1 - lr) + result.bpm * lr
        bl.bpm_std = max(1.5, bl.bpm_std * (1 - lr) + abs(result.bpm - old_mean) * lr)
        if result.hrv_sdnn > 0:
            bl.hrv_sdnn_mean = bl.hrv_sdnn_mean * (1 - lr) + result.hrv_sdnn * lr
        if result.resp_rate > 0:
            bl.resp_mean = bl.resp_mean * (1 - lr) + result.resp_rate * lr
        if result.stress_index > 0:
            bl.stress_mean = bl.stress_mean * (1 - lr) + result.stress_index * lr
        bl.samples_count += 1
        bl.last_updated = time.time()

    def baseline_deviation(self, result: VitalsResult) -> dict:
        bl = self.baseline
        if bl.samples_count < 20:
            return {'calibrating': True, 'samples': bl.samples_count}
        devs = {}
        if result.bpm > 0 and bl.bpm_std > 0:
            z_bpm = (result.bpm - bl.bpm_mean) / bl.bpm_std
            devs['bpm_z'] = round(z_bpm, 2)
            devs['bpm_label'] = 'HIGH' if z_bpm > 2 else 'LOW' if z_bpm < -2 else 'ELEVATED' if z_bpm > 1 else 'SUPPRESSED' if z_bpm < -1 else 'NORMAL'
        if result.hrv_sdnn > 0 and bl.hrv_sdnn_mean > 5:
            hrv_ratio = result.hrv_sdnn / bl.hrv_sdnn_mean
            devs['hrv_ratio'] = round(hrv_ratio, 2)
            devs['hrv_label'] = 'REDUCED' if hrv_ratio < 0.7 else 'ELEVATED' if hrv_ratio > 1.5 else 'NORMAL'
        if result.stress_index > 0:
            devs['stress_elevated'] = result.stress_index > bl.stress_mean * 1.25
            devs['stress_z'] = round((result.stress_index - bl.stress_mean) / max(5.0, bl.stress_mean * 0.2), 2)
        devs['baseline_samples'] = bl.samples_count
        return devs

    def get_paper_metrics(self, ground_truth_bpm: Optional[float]=None, estimated_bpm_series: Optional[list]=None, ground_truth_series: Optional[list]=None) -> dict:
        metrics = {}
        hist = list(self._bpm_window)
        if not hist:
            return metrics
        metrics['mean_bpm'] = round(float(np.mean(hist)), 2)
        metrics['std_bpm'] = round(float(np.std(hist)), 2)
        if ground_truth_bpm is not None and hist:
            errors = [abs(b - ground_truth_bpm) for b in hist]
            metrics['MAE'] = round(float(np.mean(errors)), 2)   
            metrics['RMSE'] = round(float(np.sqrt(np.mean([e ** 2 for e in errors]))), 2)
        if estimated_bpm_series is not None and ground_truth_series is not None and (len(estimated_bpm_series) == len(ground_truth_series)) and (len(estimated_bpm_series) > 2):
            try:
                r, p = pearsonr(estimated_bpm_series, ground_truth_series)
                metrics['pearson_r'] = round(float(r), 4)
                metrics['pearson_p'] = round(float(p), 4)
            except Exception:
                pass
        ba = self.ground_truth.compute_bland_altman()
        if 'error' not in ba:
            metrics['bland_altman'] = ba
        pe = self.ground_truth.compute_pearson()
        if 'error' not in pe:
            metrics['pearson_gt'] = pe
        metrics['failure_modes'] = self.failure_log.get_summary()
        return metrics

    def start_robustness_test(self, test_name: str):
        self.failure_log.log('robustness_test_start', test_name)
        print(f'[ROBUSTNESS] Starting test: {test_name}')

    def get_robustness_report(self) -> dict:
        events = self.failure_log.recent_events(500)
        report = {'tests': [], 'failure_summary': self.failure_log.get_summary()}
        test_events = [e for e in events if e['type'] == 'robustness_test_start']
        for ev in test_events:
            report['tests'].append({'name': ev['detail'], 'timestamp': ev['ts'], 'note': 'manual GT comparison required for full MAE/RMSE'})
        return report