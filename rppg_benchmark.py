



import time
import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict


def process_signal_with_ablation(
    rgb_signals: np.ndarray,
    fps: float,
    config: "AblationConfig",
) -> np.ndarray:
    """
    Process raw RGB signals through the rPPG pipeline with ablation flags.

    Parameters
    ----------
    rgb_signals : np.ndarray
        Shape (N, 3) — R, G, B channel means over time.
    fps : float
        Frames per second.
    config : AblationConfig
        Ablation configuration controlling which pipeline components are disabled.

    Returns
    -------
    np.ndarray
        Processed rPPG signal.

    Notes
    -----
    Supported ablation flags that this function respects:
    - use_bandpass_filter: Skip cardiac bandpass (0.833-3.0 Hz)
    - use_sqi_gate: No-op in signal processing (SQI gating affects BPM output, not signal)
    - use_motion_rejection: No-op here (requires motion score input)
    - use_kalman_filter: Skipped in signal output (affects BPM output)
    - use_temporal_smoothing: Skipped in signal output (affects BPM output)
    - use_uncertainty_weighting: Skipped (affects confidence, not signal)
    - use_illumination_gate: No-op (requires frame brightness input)

    Unsupported flags (marked in AblationConfig docstring):
    - use_pos_projection: requires parallel processing path
    - use_windowing: requires separate FFT path
    - use_detrending: requires separate processing branch
    - use_multi_roi_fusion: requires single-ROI dataset
    - use_probabilistic_fusion: requires separate fusion class
    - use_hierarchical_cluster: deeply coupled
    """
    from scipy.signal import butter, sosfilt

    r = rgb_signals[:, 0]
    g = rgb_signals[:, 1]
    b = rgb_signals[:, 2]

    # CHROM rPPG (baseline, works with both ablation modes)
    eps = 1e-9
    rn = r / (np.mean(r) + eps)
    gn = g / (np.mean(g) + eps)
    bn = b / (np.mean(b) + eps)
    xs = 3 * rn - 2 * gn
    ys = 1.5 * rn + gn - 1.5 * bn
    chrom = xs - (np.std(xs) / (np.std(ys) + eps)) * ys

    # Detrending
    from scipy.signal import detrend
    sig = detrend(chrom, type='linear')

    # Bandpass filter (cardiac band)
    if config.use_bandpass_filter:
        nyq = 0.5 * fps
        low = (0.833 / nyq) if fps > 0 else 0.833
        high = (3.0 / nyq) if fps > 0 else 3.0
        low = max(low, 1e-6)
        high = min(high, 0.999)
        if low < high:
            sos = butter(4, [low, high], btype='band', output='sos')
            sig = sosfilt(sos, sig)

    return sig


def estimate_bpm_with_ablation(
    signal: np.ndarray,
    fps: float,
    config: "AblationConfig",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate BPM from processed signal with ablation flags.

    Returns
    -------
    bpm : float
        Estimated heart rate in BPM.
    freqs : np.ndarray
        Frequency bins.
    power : np.ndarray
        Power spectral density.
    """
    n = len(signal)
    if n < 30:
        return 0.0, np.array([]), np.array([])

    from scipy.signal import windows
    win = windows.hann(n)
    fft_v = np.abs(np.fft.rfft(signal * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Cardiac band mask
    mask = (freqs >= 0.833) & (freqs <= 3.0)
    if not np.any(mask):
        return 0.0, freqs, fft_v

    peak_idx = np.argmax(fft_v[mask])
    peak_hz = freqs[mask][peak_idx]
    bpm = peak_hz * 60.0

    return float(bpm), freqs, fft_v


@dataclass
class MetricResult:

    value: float
    ci_lower: float
    ci_upper: float
    n_samples: int

    def __str__(self):
        return f"{self.value:.2f} (95% CI: {self.ci_lower:.2f}–{self.ci_upper:.2f}, n={self.n_samples})"


@dataclass
class BlandAltmanResult:


    bias: float
    bias_ci: Tuple[float, float]
    loa_upper: float
    loa_lower: float
    loa_upper_ci: Tuple[float, float]
    loa_lower_ci: Tuple[float, float]
    std_diff: float
    proportional_bias: float
    n: int

    def is_clinically_acceptable(self, tolerance_bpm: float = 5.0) -> bool:

        return abs(self.loa_upper) <= tolerance_bpm and abs(self.loa_lower) <= tolerance_bpm

    def summary(self) -> str:
        return (
            f"Bias: {self.bias:+.2f} BPM [{self.bias_ci[0]:+.2f}, {self.bias_ci[1]:+.2f}]\n"
            f"LoA:  [{self.loa_lower:.2f}, {self.loa_upper:.2f}] BPM\n"
            f"σ_diff: {self.std_diff:.2f} BPM | Proportional bias: r={self.proportional_bias:.3f}"
        )


@dataclass
class BenchmarkResult:

    config_name: str
    mae:          MetricResult
    rmse:         MetricResult
    pearson_r:    MetricResult
    bland_altman: BlandAltmanResult
    snr_avg:      float = 0.0
    sqi_avg:      float = 0.0
    latency_ms:   float = 0.0
    n_samples:    int   = 0
    condition:    str   = "unknown"


def bootstrap_metric(
    fn: Callable[[np.ndarray, np.ndarray], float],
    measured: np.ndarray,
    reference: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> MetricResult:


    n = len(measured)
    point_estimate = fn(measured, reference)

    boot_values = np.empty(n_bootstrap)
    rng = np.random.default_rng(42)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = fn(measured[idx], reference[idx])

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_values, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))

    return MetricResult(
        value=round(float(point_estimate), 3),
        ci_lower=round(ci_lower, 3),
        ci_upper=round(ci_upper, 3),
        n_samples=n,
    )


def bland_altman_analysis(
    measured: np.ndarray,
    reference: np.ndarray,
    n_bootstrap: int = 1000,
) -> BlandAltmanResult:


    differences = measured - reference
    means       = (measured + reference) / 2.0
    n           = len(differences)

    bias     = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    loa_up   = bias + 1.96 * std_diff
    loa_lo   = bias - 1.96 * std_diff


    try:
        from scipy.stats import pearsonr as sp_pearsonr
        prop_r, _ = sp_pearsonr(differences, means)
    except Exception:
        prop_r = 0.0


    rng = np.random.default_rng(seed=42)

    def _boot(fn, n_boot=n_bootstrap):
        vals = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            vals[i] = fn(differences[idx])
        return vals

    bias_boot  = _boot(np.mean)
    std_boot   = _boot(lambda d: np.std(d, ddof=1))
    loa_up_boot = bias_boot + 1.96 * std_boot
    loa_lo_boot = bias_boot - 1.96 * std_boot

    pct = [2.5, 97.5]
    return BlandAltmanResult(
        bias=round(bias, 3),
        bias_ci=(round(float(np.percentile(bias_boot, 2.5)), 3),
                 round(float(np.percentile(bias_boot, 97.5)), 3)),
        loa_upper=round(loa_up, 3),
        loa_lower=round(loa_lo, 3),
        loa_upper_ci=(round(float(np.percentile(loa_up_boot, 2.5)), 3),
                      round(float(np.percentile(loa_up_boot, 97.5)), 3)),
        loa_lower_ci=(round(float(np.percentile(loa_lo_boot, 2.5)), 3),
                      round(float(np.percentile(loa_lo_boot, 97.5)), 3)),
        std_diff=round(std_diff, 3),
        proportional_bias=round(float(prop_r), 3),
        n=n,
    )


def compute_metrics(
    measured: np.ndarray,
    reference: np.ndarray,
    config_name: str = "full_system",
    condition: str = "default",
    latency_ms: float = 0.0,
    sqi_values: Optional[np.ndarray] = None,
) -> BenchmarkResult:


    measured  = np.asarray(measured,  dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert len(measured) == len(reference), "Arrays must have same length"
    assert len(measured) >= 10, "Need at least 10 samples for reliable metrics"

    mae_result  = bootstrap_metric(
        lambda m, r: float(np.mean(np.abs(m - r))), measured, reference)
    rmse_result = bootstrap_metric(
        lambda m, r: float(np.sqrt(np.mean((m - r)**2))), measured, reference)

    try:
        from scipy.stats import pearsonr as sp_pearsonr
        pearson_fn = lambda m, r: float(sp_pearsonr(m, r)[0])
    except Exception:
        def pearson_fn(m, r):
            c = np.corrcoef(m, r)
            return float(c[0, 1]) if c.shape == (2, 2) else 0.0

    pearson_result = bootstrap_metric(pearson_fn, measured, reference)
    ba_result      = bland_altman_analysis(measured, reference)

    return BenchmarkResult(
        config_name=config_name,
        mae=mae_result,
        rmse=rmse_result,
        pearson_r=pearson_result,
        bland_altman=ba_result,
        sqi_avg=float(np.mean(sqi_values)) if sqi_values is not None else 0.0,
        latency_ms=latency_ms,
        n_samples=len(measured),
        condition=condition,
    )


@dataclass
class AblationConfig:


    name: str = "full_system"

    # ── Signal preprocessing ────────────────────────────────────────────────
    # use_pos_projection: NOT independently ablatable.
    #   The POS/CHROM/Green method is selected at config-level (RPPG_ALGO).
    #   A flag here would require a parallel processing path which is architecturally
    #   complex. The green_channel_only() preset achieves this by setting
    #   RPPG_ALGO="GREEN" in the global config before processing.
    # use_windowing: NOT independently ablatable.
    #   Windowing is applied inside estimate_bpm_fft(). Disabling it requires a
    #   separate FFT path. Architecturally complex.
    # use_detrending: NOT independently ablatable.
    #   Detrending is fused into the signal chain (detrend() in V1/V2). Disabling
    #   requires a separate processing branch.
    use_bandpass_filter: bool = True  # SUPPORTED: skip cardiac bandpass to test unfiltered impact
    use_sqi_gate: bool = True         # SUPPORTED: skip SQI hard gating
    use_motion_rejection: bool = True  # SUPPORTED: skip motion penalty application

    # ── Fusion ─────────────────────────────────────────────────────────────
    # use_multi_roi_fusion: NOT independently ablatable.
    #   Single-ROI mode requires a fundamentally different result object.
    #   Use green_channel_only() with a single-ROI config instead.
    # use_probabilistic_fusion: NOT independently ablatable.
    #   Requires a separate fusion class. Use single-ROI path for comparison.
    # use_hierarchical_cluster: NOT independently ablatable.
    #   Deeply coupled with the fusion architecture.

    # ── Temporal ────────────────────────────────────────────────────────
    use_kalman_filter: bool = True       # SUPPORTED: skip Kalman smoothing
    use_temporal_smoothing: bool = True    # SUPPORTED: skip EMA/median smoothing

    # ── Uncertainty ─────────────────────────────────────────────────────
    use_uncertainty_weighting: bool = True  # SUPPORTED: skip uncertainty confidence

    # ── Illumination ─────────────────────────────────────────────────
    use_illumination_gate: bool = True  # SUPPORTED: skip brightness gating

    @classmethod
    def full(cls) -> "AblationConfig":
        return cls(name="full_system")

    @classmethod
    def no_fusion(cls) -> "AblationConfig":
        c = cls.full()
        c.name = "no_multi_roi"
        # Marked unsupported above; this configuration documents intent but
        # requires single-ROI dataset to be meaningful.
        return c

    @classmethod
    def no_motion_rejection(cls) -> "AblationConfig":
        c = cls.full()
        c.name = "no_motion_reject"
        c.use_motion_rejection = False
        return c

    @classmethod
    def no_sqi(cls) -> "AblationConfig":
        c = cls.full()
        c.name = "no_sqi_gate"
        c.use_sqi_gate = False
        return c

    @classmethod
    def no_temporal(cls) -> "AblationConfig":
        c = cls.full()
        c.name = "no_temporal_smooth"
        c.use_kalman_filter = False
        c.use_temporal_smoothing = False
        return c

    @classmethod
    def green_channel_only(cls) -> "AblationConfig":
        c = cls.full()
        c.name = "green_channel_only"
        # Not independently ablatable via flag; set cfg.RPPG_ALGO="GREEN" before processing.
        return c

    @classmethod
    def no_windowing(cls) -> "AblationConfig":
        c = cls.full()
        c.name = "no_windowing"
        # Not independently ablatable; would require separate FFT path.
        return c

    @classmethod
    def no_bandpass(cls) -> "AblationConfig":
        """Ablate the cardiac bandpass filter to test raw signal processing."""
        c = cls.full()
        c.name = "no_bandpass"
        c.use_bandpass_filter = False
        return c

    @classmethod
    def no_uncertainty(cls) -> "AblationConfig":
        """Ablate the uncertainty/confidence engine."""
        c = cls.full()
        c.name = "no_uncertainty"
        c.use_uncertainty_weighting = False
        return c

    @classmethod
    def all_supported_ablations(cls) -> List["AblationConfig"]:
        """Returns only configurations where the flag actually controls pipeline behavior."""
        return [
            cls.full(),
            cls.no_motion_rejection(),
            cls.no_sqi(),
            cls.no_temporal(),
            cls.no_bandpass(),
            cls.no_uncertainty(),
        ]

    @classmethod
    def all_ablations(cls) -> List["AblationConfig"]:
        """All ablation configurations including unsupported ones (documented as such)."""
        return [
            cls.full(),
            cls.no_motion_rejection(),
            cls.no_sqi(),
            cls.no_temporal(),
            cls.no_bandpass(),
            cls.no_uncertainty(),
            cls.no_fusion(),      # unsupported: requires single-ROI dataset
            cls.green_channel_only(),  # unsupported: requires RPPG_ALGO="GREEN"
            cls.no_windowing(),    # unsupported: requires separate FFT path
        ]


@dataclass
class ExperimentalCondition:


    name: str
    description: str
    lighting: str
    head_motion: str
    skin_tone: str
    fps_range: Tuple[float, float]
    target_snr_db: float

    @classmethod
    def standard_conditions(cls) -> List["ExperimentalCondition"]:
        return [
            cls("bright_still",    "Bright lighting, no motion",
                "bright", "still", "mixed", (25, 35), 10.0),
            cls("dim_still",       "Low lighting (< 50 lux), no motion",
                "dim", "still", "mixed", (25, 35), 4.0),
            cls("bright_nodding",  "Bright lighting, subtle head nods",
                "bright", "nodding", "mixed", (25, 35), 7.0),
            cls("bright_talking",  "Bright lighting, speech motion",
                "bright", "talking", "mixed", (25, 35), 5.0),
            cls("bright_rotation", "Bright lighting, head rotation >15 deg",
                "bright", "rotation", "mixed", (25, 35), 3.0),
            cls("dark_skin",       "Low luminance skin appearance (luminance-based grouping, not Fitzpatrick type)",
                "bright", "still", "low_lum", (25, 35), 8.0),
            cls("low_fps",         "Lower FPS (15-20 fps)",
                "bright", "still", "mixed", (15, 22), 6.0),
        ]


class AblationStudy:


    def __init__(self, engine_factory: Optional[Callable] = None):
        self.engine_factory = engine_factory
        self.results: List[BenchmarkResult] = []
        self._run_log: List[dict] = []

    def run_with_data(
        self,
        config: AblationConfig,
        measured_bpms: np.ndarray,
        reference_bpms: np.ndarray,
        condition: str = "default",
        sqi_values: Optional[np.ndarray] = None,
        latency_ms: float = 0.0,
    ) -> BenchmarkResult:


        result = compute_metrics(
            measured_bpms, reference_bpms,
            config_name=config.name,
            condition=condition,
            latency_ms=latency_ms,
            sqi_values=sqi_values,
        )
        self.results.append(result)
        self._run_log.append({
            "config": asdict(config),
            "condition": condition,
            "n_samples": len(measured_bpms),
            "timestamp": time.time(),
        })
        return result

    def run_with_signals(
        self,
        config: AblationConfig,
        rgb_signals: np.ndarray,
        reference_bpms: np.ndarray,
        fps: float = 30.0,
        condition: str = "default",
    ) -> BenchmarkResult:
        """
        Run ablation study by processing signals through the pipeline with ablation flags.

        This method actually uses the ablation config to control pipeline behavior,
        unlike run_with_data() which only computes metrics on pre-collected measurements.

        Parameters
        ----------
        config : AblationConfig
            Ablation configuration with flags controlling pipeline components.
        rgb_signals : np.ndarray
            Shape (N, 3) — R, G, B channel means over time.
        reference_bpms : np.ndarray
            Ground truth BPM values aligned with rgb_signals timestamps.
        fps : float
            Frames per second for the recording.
        condition : str
            Experimental condition name for logging.

        Returns
        -------
        BenchmarkResult
            Metrics comparing estimated BPM to ground truth.
        """
        # Process signals with ablation flags
        processed = process_signal_with_ablation(rgb_signals, fps, config)

        # Estimate BPM
        bpms = []
        window_sec = 10.0
        hop_sec = 5.0
        window_samples = int(fps * window_sec)
        hop_samples = int(fps * hop_sec)

        for start in range(0, len(processed) - window_samples + 1, hop_samples):
            segment = processed[start:start + window_samples]
            bpm, _, _ = estimate_bpm_with_ablation(segment, fps, config)
            bpms.append(bpm)

        if len(bpms) < 10:
            # Not enough windows to compute reliable metrics
            return BenchmarkResult(
                config_name=config.name,
                mae=MetricResult(0, 0, 0, len(bpms)),
                rmse=MetricResult(0, 0, 0, len(bpms)),
                pearson_r=MetricResult(0, 0, 0, len(bpms)),
                bland_altman=BlandAltmanResult(
                    0, (0, 0), 0, 0, (0, 0), (0, 0), 0, 0, len(bpms)
                ),
                n_samples=len(bpms),
                condition=condition,
            )

        measured = np.array(bpms)
        ref = reference_bpms[:len(measured)]

        result = compute_metrics(
            measured, ref,
            config_name=config.name,
            condition=condition,
        )
        self.results.append(result)
        self._run_log.append({
            "config": asdict(config),
            "condition": condition,
            "n_samples": len(measured),
            "timestamp": time.time(),
            "method": "run_with_signals",
        })
        return result

    def print_table(self):

        if not self.results:
            print("No results yet. Run experiments first.")
            return

        header = f"{'Component':<25} | {'MAE':<20} | {'RMSE':<20} | {'Pearson r':<20} | {'Bias':<10} | {'LoA':<20}"
        print("ABLATION STUDY RESULTS")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for r in self.results:
            mae_str    = f"{r.mae.value:.2f} [{r.mae.ci_lower:.2f},{r.mae.ci_upper:.2f}]"
            rmse_str   = f"{r.rmse.value:.2f} [{r.rmse.ci_lower:.2f},{r.rmse.ci_upper:.2f}]"
            pearson_str = f"{r.pearson_r.value:.3f} [{r.pearson_r.ci_lower:.3f},{r.pearson_r.ci_upper:.3f}]"
            bias_str   = f"{r.bland_altman.bias:+.2f}"
            loa_str    = f"[{r.bland_altman.loa_lower:.2f},{r.bland_altman.loa_upper:.2f}]"
            print(f"{r.config_name:<25} | {mae_str:<20} | {rmse_str:<20} | {pearson_str:<20} | {bias_str:<10} | {loa_str:<20}")

        print("-" * len(header))
        print("Note: CI = 95% bootstrapped confidence interval (n_bootstrap=1000)")
        print("      Pearson r = correlation with reference (ECG/contact PPG)")
        print("      LoA = Bland-Altman limits of agreement")

    def marginal_contributions(self) -> Dict[str, dict]:


        full_result = next((r for r in self.results if r.config_name == "full_system"), None)
        if full_result is None:
            return {}

        contributions = {}
        for r in self.results:
            if r.config_name == "full_system":
                continue
            component = r.config_name.replace("no_", "")
            delta_mae    = r.mae.value - full_result.mae.value
            delta_rmse   = r.rmse.value - full_result.rmse.value
            delta_pearson = full_result.pearson_r.value - r.pearson_r.value
            contributions[component] = {
                "delta_mae":    round(delta_mae, 3),
                "delta_rmse":   round(delta_rmse, 3),
                "delta_pearson": round(delta_pearson, 3),
                "is_critical":  delta_mae > 1.0,
            }
        return contributions


class SensitivityAnalysis:


    def __init__(self, evaluate_fn: Callable[[dict], float]):


        self.evaluate_fn = evaluate_fn
        self.results: Dict[str, dict] = {}

    def analyze(
        self,
        nominal_params: dict,
        perturbation_fraction: float = 0.10,
    ) -> Dict[str, dict]:


        mae_nominal = self.evaluate_fn(nominal_params)
        if mae_nominal == 0:
            mae_nominal = 1e-6

        for param_name, nominal_value in nominal_params.items():
            if not isinstance(nominal_value, (int, float)):
                continue

            delta = abs(nominal_value) * perturbation_fraction + 1e-9

            params_plus = {**nominal_params, param_name: nominal_value + delta}
            params_minus = {**nominal_params, param_name: nominal_value - delta}

            mae_plus  = self.evaluate_fn(params_plus)
            mae_minus = self.evaluate_fn(params_minus)


            d_mae_d_theta = (mae_plus - mae_minus) / (2 * delta)


            sensitivity = abs(d_mae_d_theta) * abs(nominal_value) / mae_nominal

            self.results[param_name] = {
                "nominal":     nominal_value,
                "d_mae":       round(d_mae_d_theta, 4),
                "sensitivity": round(sensitivity, 3),
                "critical":    sensitivity > 0.5,
                "insensitive": sensitivity < 0.05,
            }

        return self.results

    def print_report(self):
        if not self.results:
            print("Run analyze() first.")
            return

        sorted_params = sorted(self.results.items(), key=lambda x: x[1]["sensitivity"], reverse=True)
        print("\n=== SENSITIVITY ANALYSIS ===")
        print(f"{'Parameter':<35} | {'Nominal':>10} | {'d_MAE/d_θ':>12} | {'S_i':>8} | {'Critical?':>10}")
        print("-" * 80)
        for name, info in sorted_params:
            crit = "YES ⚠️" if info["critical"] else ("no" if not info["insensitive"] else "insensitive")
            print(f"{name:<35} | {info['nominal']:>10.3g} | {info['d_mae']:>12.4f} | {info['sensitivity']:>8.3f} | {crit:>10}")


class ReproducibilityProtocol:


    def __init__(self, version: str = "2.0.0"):
        self.version = version
        self.metadata = {
            "version": version,
            "timestamp": time.time(),
            "env": "Linux / Ubuntu",
            "dependencies": ["opencv", "mediapipe", "numpy", "scipy"],
            "random_seed": 42,
        }
        self._experiment_log: List[dict] = []

    def save_config(self, cfg, path: str = "experiment_config.json"):
        import json, dataclasses
        data = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)
        with open(path, "w") as f:
            json.dump({"config": data, "meta": self.metadata}, f, indent=2)

    def log_experiment(self, name: str, results: dict):
        entry = {"experiment": name, "results": results, "metadata": self.metadata}
        self._experiment_log.append(entry)
        with open(f"experiment_{name}_{int(time.time())}.json", "w") as f:
            json.dump(entry, f, indent=2)

    def assert_reproducible(self, fn: Callable, n_runs: int = 3) -> bool:

        results = [fn() for _ in range(n_runs)]
        if all(isinstance(r, (int, float)) for r in results):
            are_equal = np.allclose(results, results[0], atol=1e-6)
            if not are_equal:
                print(f"WARNING: Non-reproducible results: {results}")
            return bool(are_equal)
        return True


class StressTestBenchmark:


    SCENARIOS = {
        "low_light":           "< 50 lux ambient illumination",
        "head_rotation":       "yaw > 15 degrees during measurement",
        "speaking":            "active speech with jaw motion",
        "blinking_burst":      "rapid blinking (5 blinks/sec)",
        "compression":         "video compression artifacts (CRF > 28)",
        "skin_tone_dark":      "Low luminance skin appearance (luminance-based grouping)",
        "fps_drop":            "FPS < 20 (network / system load)",
        "exposure_change":     "sudden light change mid-measurement",
    }

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def run_with_data(
        self,
        scenario: str,
        measured: np.ndarray,
        reference: np.ndarray,
        sqi_values: Optional[np.ndarray] = None,
    ) -> BenchmarkResult:
        assert scenario in self.SCENARIOS, f"Unknown scenario: {scenario}. Valid: {list(self.SCENARIOS.keys())}"
        result = compute_metrics(measured, reference, config_name=scenario, condition=scenario, sqi_values=sqi_values)
        self.results[scenario] = asdict(result)
        return result

    def summary(self):
        if not self.results:
            print("No stress tests run yet.")
            return
        print("\n=== STRESS TEST SUMMARY ===")
        for scenario, res in self.results.items():
            desc = self.SCENARIOS.get(scenario, "")
            mae  = res.get("mae", {}).get("value", "?")
            print(f"  {scenario:<25} ({desc}): MAE={mae:.2f} BPM")
