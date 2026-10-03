"""
Ablation audit for Sanubari V2 benchmark.

Documents which ablation configurations actually affect the frozen V2 execution path.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class AblationRecord:
    """Record of an ablation configuration's status."""
    name: str
    config_field: str
    affects_v2: bool
    tested: bool
    notes: str
    path_affected: Optional[str] = None


# Ablation audit for Sanubari V2
ABLATION_AUDIT: List[AblationRecord] = [
    # Signal preprocessing ablations
    AblationRecord(
        name="use_bandpass_filter",
        config_field="use_bandpass_filter",
        affects_v2=True,
        tested=True,
        notes="Affects cardiac bandpass filter in FusionEngineV2. "
              "In V2, bandpass is applied via bandpass_filter() function.",
        path_affected="rppg_core.py::bandpass_filter"
    ),
    AblationRecord(
        name="use_sqi_gate",
        config_field="use_sqi_gate",
        affects_v2=True,
        tested=True,
        notes="Affects SQI gating in FusionEngineV2 candidate selection. "
              "V2 uses SQI_HARD_GATE in candidate filtering.",
        path_affected="rppg_core.py::MultiROIFusionEngineV2.update"
    ),
    AblationRecord(
        name="use_motion_rejection",
        config_field="use_motion_rejection",
        affects_v2=True,
        tested=True,
        notes="Affects motion penalty in V2 signal processing. "
              "Motion state affects ROI SQI and fusion weights.",
        path_affected="rppg_core.py::MotionArtifactDetector, compute_sqi"
    ),
    AblationRecord(
        name="use_kalman_filter",
        config_field="use_kalman_filter",
        affects_v2=True,
        tested=True,
        notes="Affects Kalman BPM smoothing in BPMSmoother. "
              "V2 uses KalmanBPMFilter.update() in BPM smoothing.",
        path_affected="rppg_core.py::BPMSmoother.update, KalmanBPMFilter"
    ),
    AblationRecord(
        name="use_temporal_smoothing",
        config_field="use_temporal_smoothing",
        affects_v2=True,
        tested=True,
        notes="Affects EMA/median smoothing in BPMSmoother. "
              "V2 uses EMA and median window for BPM smoothing.",
        path_affected="rppg_core.py::BPMSmoother.update"
    ),
    AblationRecord(
        name="use_uncertainty_weighting",
        config_field="use_uncertainty_weighting",
        affects_v2=True,
        tested=True,
        notes="Affects UncertaintyAwareConfidence in V2. "
              "Used in fusion weight calculation.",
        path_affected="rppg_core.py::UncertaintyAwareConfidence"
    ),

    # Not directly connected to V2
    AblationRecord(
        name="use_pos_projection",
        config_field="use_pos_projection",
        affects_v2=False,
        tested=False,
        notes="Requires parallel processing path. V2 uses MethodArbitrator. "
              "Select by setting RPPG_ALGO config, not ablation flag.",
        path_affected=None
    ),
    AblationRecord(
        name="use_windowing",
        config_field="use_windowing",
        affects_v2=False,
        tested=False,
        notes="Affects FFT windowing in estimate_bpm_fft(). "
              "NOT independently ablatable in V2 - would require separate FFT path.",
        path_affected=None
    ),
    AblationRecord(
        name="use_detrending",
        config_field="use_detrending",
        affects_v2=False,
        tested=False,
        notes="Detrending is fused into signal chain in V2. "
              "Cannot be disabled without separate processing branch.",
        path_affected=None
    ),
    AblationRecord(
        name="use_multi_roi_fusion",
        config_field="use_multi_roi_fusion",
        affects_v2=False,
        tested=False,
        notes="Requires single-ROI mode which changes result object. "
              "Use single-ROI config for comparison, not ablation flag.",
        path_affected=None
    ),
    AblationRecord(
        name="use_probabilistic_fusion",
        config_field="use_probabilistic_fusion",
        affects_v2=False,
        tested=False,
        notes="Requires separate fusion class. V2 always uses "
              "ProbabilisticFusion + hierarchical_fusion.",
        path_affected=None
    ),
    AblationRecord(
        name="use_hierarchical_cluster",
        config_field="use_hierarchical_cluster",
        affects_v2=False,
        tested=False,
        notes="Deeply coupled with fusion architecture in V2. "
              "Cannot be disabled independently.",
        path_affected=None
    ),
    AblationRecord(
        name="use_illumination_gate",
        config_field="use_illumination_gate",
        affects_v2=True,
        tested=True,
        notes="Affects brightness gating in ROI processing. "
              "V2 applies brightness penalties in compute_sqi().",
        path_affected="rppg_core.py::compute_sqi"
    ),
]


def get_supported_ablations() -> List[AblationRecord]:
    """Get ablations that are actually connected to V2 execution path."""
    return [a for a in ABLATION_AUDIT if a.affects_v2]


def get_unsupported_ablations() -> List[AblationRecord]:
    """Get ablations not connected to V2 execution path."""
    return [a for a in ABLATION_AUDIT if not a.affects_v2]


def get_untested_ablations() -> List[AblationRecord]:
    """Get ablations marked as not tested."""
    return [a for a in ABLATION_AUDIT if not a.tested]


def audit_ablations() -> Dict[str, Any]:
    """
    Generate ablation audit report.

    Returns
    -------
    dict
        Audit report with supported, unsupported, and untested ablating configs.
    """
    supported = get_supported_ablations()
    unsupported = get_unsupported_ablations()
    untested = get_untested_ablations()

    return {
        "summary": {
            "total": len(ABLATION_AUDIT),
            "supported": len(supported),
            "unsupported": len(unsupported),
            "untested": len(untested),
        },
        "supported_ablations": [
            {
                "name": a.name,
                "field": a.config_field,
                "path": a.path_affected,
                "notes": a.notes,
            }
            for a in supported
        ],
        "unsupported_ablations": [
            {
                "name": a.name,
                "field": a.config_field,
                "notes": a.notes,
            }
            for a in unsupported
        ],
        "untested_ablations": [
            {
                "name": a.name,
                "field": a.config_field,
                "notes": a.notes,
            }
            for a in untested
        ],
        "recommendation": (
            "Only use ablation configurations marked as 'supported' for V2 benchmarks. "
            "Unsupported ablating configs either require separate processing paths "
            "or cannot be disabled without modifying V2 architecture."
        ),
    }


if __name__ == "__main__":
    import json
    report = audit_ablations()
    print(json.dumps(report, indent=2))
