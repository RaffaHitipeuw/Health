"""
P4 Reliability & Uncertainty Evolution Package

Extends Sanubari with a rigorous reliability and uncertainty layer.

Modules:
    reliability_evidence (P4.1): Formal reliability evidence model
    uncertainty_decomposition (P4.2): Uncertainty decomposition
    confidence_fusion (P4.3): Confidence/uncertainty fusion
    temporal_reliability (P4.4): Temporal reliability state machine
    plausibility_gate (P4.5): Physiological plausibility gate
    p4_pipeline (P4.6): Complete P4 estimation pipeline
    calibration (P4.7): Calibration infrastructure
    p4_failures (P4.9): P4-specific failure analysis

Usage:
    from p4_reliability import P4Estimator, P4Mode

    estimator = P4Estimator(mode=P4Mode.FULL_P4)
    result = estimator.estimate(bpm_estimate=72.0, sqi=85.0)

Note: P4 operates PARALLEL to V2 and P3, not replacing them.
      V2 core remains frozen and unchanged.
"""

__version__ = "1.0.0"

# P4.1: Reliability Evidence
from p4_reliability.reliability_evidence import (
    EvidenceSource,
    EvidenceComponent,
    ReliabilityEvidence,
    ReliabilityEvidenceCollector,
)

# P4.2: Uncertainty Decomposition
from p4_reliability.uncertainty_decomposition import (
    UncertaintySource,
    UncertaintyComponent,
    UncertaintyDecomposition,
    UncertaintyDecomposer,
)

# P4.3: Confidence Fusion
from p4_reliability.confidence_fusion import (
    FusionMethod,
    FusedEstimate,
    ConfidenceUncertaintyFusion,
)

# P4.4: Temporal Reliability
from p4_reliability.temporal_reliability import (
    TemporalState,
    TemporalReliabilityState,
    TemporalReliabilityResult,
    TemporalReliabilityTracker,
)

# P4.5: Plausibility Gate
from p4_reliability.plausibility_gate import (
    PlausibilityState,
    PlausibilityCheck,
    PlausibilityResult,
    PhysiologicalPlausibilityGate,
)

# P4.6: P4 Pipeline
from p4_reliability.p4_pipeline import (
    P4Mode,
    P4EstimateResult,
    P4Estimator,
)

# P4.7: Calibration
from p4_reliability.calibration import (
    CalibrationMetrics,
    CalibrationEvaluator,
)

# P4.9: Failures
from p4_reliability.p4_failures import (
    P4FailureType,
    P4Failure,
    P4FailureAnalysisResult,
    P4FailureAnalyzer,
)

__all__ = [
    # Version
    "__version__",
    # P4.1
    "EvidenceSource",
    "EvidenceComponent",
    "ReliabilityEvidence",
    "ReliabilityEvidenceCollector",
    # P4.2
    "UncertaintySource",
    "UncertaintyComponent",
    "UncertaintyDecomposition",
    "UncertaintyDecomposer",
    # P4.3
    "FusionMethod",
    "FusedEstimate",
    "ConfidenceUncertaintyFusion",
    # P4.4
    "TemporalState",
    "TemporalReliabilityState",
    "TemporalReliabilityResult",
    "TemporalReliabilityTracker",
    # P4.5
    "PlausibilityState",
    "PlausibilityCheck",
    "PlausibilityResult",
    "PhysiologicalPlausibilityGate",
    # P4.6
    "P4Mode",
    "P4EstimateResult",
    "P4Estimator",
    # P4.7
    "CalibrationMetrics",
    "CalibrationEvaluator",
    # P4.9
    "P4FailureType",
    "P4Failure",
    "P4FailureAnalysisResult",
    "P4FailureAnalyzer",
]
