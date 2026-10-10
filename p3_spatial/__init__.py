"""
P3 Spatial/Motion Evolution Package

P3 extends Sanubari from coarse fixed-landmark ROI processing toward
spatially and motion-aware physiological signal extraction.

Modules:
    spatial_quality_map (P3.1): Per-cell cardiac-band SNR quality
    motion_field (P3.2): Spatially varying motion representation
    motion_aware_quality (P3.3): Quality × motion weighting
    dynamic_candidates (P3.4): Dynamic candidate region selection
    spatial_temporal (P3.5): Temporal consistency tracking
    p3_pipeline (P3.6): Complete pipeline integration
    p3_failures (P3.7): Failure detection and analysis

Usage:
    from p3_spatial import P3Pipeline, ExperimentID

    pipeline = P3Pipeline(experiment_id=ExperimentID.E6_FULL)
    result = pipeline.process_buffer(buffer, fps)

Note: P3 operates PARALLEL to V2, not replacing it.
      V2 core remains frozen and unchanged.
"""

# Version
__version__ = "1.0.0"

# P3 Stage modules
from p3_spatial.spatial_quality_map import (
    SpatialQualityMap,
    SpatialQualityResult,
    snr_cardiac,
    CARDIAC_BAND_HZ,
    DEFAULT_CELL_SIZE,
    compute_spatial_quality
)

from p3_spatial.motion_field import (
    MotionFieldEstimator,
    MotionFieldResult,
    SimpleMotionDetector,
    DEFAULT_CELL_SIZE as MOTION_CELL_SIZE
)

from p3_spatial.motion_aware_quality import (
    MotionAwareQuality,
    MotionAwareQualityResult,
    WeightingMode
)

from p3_spatial.dynamic_candidates import (
    DynamicCandidateSelector,
    Candidate,
    CandidateSet
)

from p3_spatial.spatial_temporal import (
    SpatialTemporalConsistency,
    TemporalConsistencyResult,
    CandidateState,
    TemporalCandidate
)

from p3_spatial.p3_pipeline import (
    P3Pipeline,
    P3PipelineResult,
    ExperimentID,
    run_ablation_experiment,
    run_all_ablations
)

from p3_spatial.p3_failures import (
    P3FailureAnalyzer,
    FailureAnalysisResult,
    Failure,
    FailureType,
    FailureSeverity,
    FailureResponse
)

# All public names
__all__ = [
    # Version
    "__version__",
    # P3.1
    "SpatialQualityMap",
    "SpatialQualityResult",
    "snr_cardiac",
    "CARDIAC_BAND_HZ",
    "DEFAULT_CELL_SIZE",
    "compute_spatial_quality",
    # P3.2
    "MotionFieldEstimator",
    "MotionFieldResult",
    "SimpleMotionDetector",
    "MOTION_CELL_SIZE",
    # P3.3
    "MotionAwareQuality",
    "MotionAwareQualityResult",
    "WeightingMode",
    # P3.4
    "DynamicCandidateSelector",
    "Candidate",
    "CandidateSet",
    # P3.5
    "SpatialTemporalConsistency",
    "TemporalConsistencyResult",
    "CandidateState",
    "TemporalCandidate",
    # P3.6
    "P3Pipeline",
    "P3PipelineResult",
    "ExperimentID",
    "run_ablation_experiment",
    "run_all_ablations",
    # P3.7
    "P3FailureAnalyzer",
    "FailureAnalysisResult",
    "Failure",
    "FailureType",
    "FailureSeverity",
    "FailureResponse",
]
