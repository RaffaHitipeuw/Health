


import time
import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict


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


    use_pos_projection:     bool = True
    use_windowing:          bool = True
    use_bandpass_filter:    bool = True
    use_detrending:         bool = True


    use_sqi_gate:           bool = True
    use_motion_rejection:   bool = True
    use_illumination_gate:  bool = True


    use_multi_roi_fusion:   bool = True
    use_probabilistic_fusion: bool = True
    use_hierarchical_cluster: bool = True


    use_kalman_filter:      bool = True
    use_temporal_smoothing: bool = True


    use_uncertainty_weighting: bool = True

    @classmethod
    def full(cls) -> "AblationConfig":
        return cls(name="full_system")

    @classmethod
    def no_fusion(cls) -> "AblationConfig":
        c = cls.full(); c.name = "no_multi_roi"
        c.use_multi_roi_fusion = False; return c

    @classmethod
    def no_motion_rejection(cls) -> "AblationConfig":
        c = cls.full(); c.name = "no_motion_reject"
        c.use_motion_rejection = False; return c

    @classmethod
    def no_sqi(cls) -> "AblationConfig":
        c = cls.full(); c.name = "no_sqi_gate"
        c.use_sqi_gate = False; return c

    @classmethod
    def no_temporal(cls) -> "AblationConfig":
        c = cls.full(); c.name = "no_temporal_smooth"
        c.use_kalman_filter = False; c.use_temporal_smoothing = False; return c

    @classmethod
    def green_channel_only(cls) -> "AblationConfig":
        c = cls.full(); c.name = "green_channel_only"
        c.use_pos_projection = False; return c

    @classmethod
    def no_windowing(cls) -> "AblationConfig":
        c = cls.full(); c.name = "no_windowing"
        c.use_windowing = False; return c

    @classmethod
    def all_ablations(cls) -> List["AblationConfig"]:
        return [
            cls.full(),
            cls.no_fusion(),
            cls.no_motion_rejection(),
            cls.no_sqi(),
            cls.no_temporal(),
            cls.green_channel_only(),
            cls.no_windowing(),
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
            cls("dark_skin",       "Dark skin tone (Fitzpatrick V-VI)",
                "bright", "still", "type_V_VI", (25, 35), 8.0),
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
        "skin_tone_dark":      "Fitzpatrick type V-VI skin",
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
