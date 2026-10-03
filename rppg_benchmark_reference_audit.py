"""
Reference method audit for Sanubari V2 benchmark.

Documents the candidate reference methods for comparison benchmarking,
including their requirements, license, and execution status.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ReferenceMethod:
    """Reference method specification."""
    name: str
    repo: str
    license: str
    execution_status: str  # READY, NOT_EXECUTED, BLOCKED, NOT_APPLICABLE
    blocking_reason: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    input_format: str = ""
    output_format: str = ""
    preprocessing: str = ""
    training_required: bool = False
    notes: str = ""
    # Execution status values:
    # READY: Can be executed in current environment
    # NOT_EXECUTED: Can be set up but hasn't been run yet
    # BLOCKED: Cannot execute due to dependency/model/constraint issues
    # NOT_APPLICABLE: Does not apply to this benchmark context


# Reference method audit
REFERENCE_METHODS: List[ReferenceMethod] = [
    ReferenceMethod(
        name="MMPD",
        repo="https://github.com/BlandAndr/mppd",
        license="MIT",
        execution_status="BLOCKED",
        blocking_reason="Requires specific dataset format (VIPL). "
                      "Implementation uses MATLAB/Octave. Not integrated.",
        requirements=[
            "MATLAB or Octave",
            "VIPL dataset",
            "Face detection model"
        ],
        input_format="Processed video sequences",
        output_format="BPM estimates",
        preprocessing="Face detection, ROI tracking",
        training_required=False,
        notes="Uses blind source separation (ICA/PCA) for rPPG extraction."
    ),
    ReferenceMethod(
        name="rPPG-Toolbox",
        repo="https://github.com/ubicomplab/rPPG-Toolbox",
        license="MIT",
        execution_status="BLOCKED",
        blocking_reason="Requires PyTorch. Dataset format differs from Sanubari. "
                      "Model training required for neural methods.",
        requirements=[
            "PyTorch >= 1.9",
            "Dataset in Toolbox format",
            "GPU recommended"
        ],
        input_format="Raw video",
        output_format="BPM, BVP",
        preprocessing="Face detection (RetinaFace/MTCNN), ROI extraction",
        training_required=True,
        notes="Includes CHROME, POS, GREEN, and deep learning methods (PhysNet, MTTS)."
    ),
    ReferenceMethod(
        name="RemoteBio",
        repo="https://github.com/mit-han-lab/remote photoplethysmography",
        license="MIT",
        execution_status="BLOCKED",
        blocking_reason="Requires specific benchmark dataset. "
                      "Dataset not available in current environment.",
        requirements=[
            "RemoteBio dataset",
            "MATLAB Runtime"
        ],
        input_format="Preprocessed video",
        output_format="BPM",
        preprocessing="Face detection, motion tracking",
        training_required=False,
        notes="Academic benchmark for remote PPG methods."
    ),
    ReferenceMethod(
        name="PhysBench",
        repo="N/A (Academic benchmark)",
        license="Research only",
        execution_status="BLOCKED",
        blocking_reason="Benchmark dataset not publicly available. "
                      "Requires institutional access.",
        requirements=[
            "PhysBench dataset",
            "Academic institution"
        ],
        input_format="Standardized video format",
        output_format="BPM, SpO2, HRV",
        preprocessing="Standardized preprocessing",
        training_required=False,
        notes="Comprehensive physiological measurement benchmark. "
              "Limited public availability."
    ),
    ReferenceMethod(
        name="kargs/rPPG-Toolbox-Benchmark",
        repo="https://github.com/kargs/rPPG-Toolbox-Benchmark",
        license="MIT",
        execution_status="BLOCKED",
        blocking_reason="Fork of rPPG-Toolbox with different dataset format. "
                      "Requires dataset conversion.",
        requirements=[
            "Python >= 3.8",
            "Converted dataset",
            "Deep learning framework"
        ],
        input_format="Video sequences",
        output_format="BPM",
        preprocessing="Face detection",
        training_required=True,
        notes="Benchmark fork with standardized evaluation. "
              "Requires dataset format conversion."
    ),
    ReferenceMethod(
        name="CHROME",
        repo="Integrated in Sanubari",
        license="N/A",
        execution_status="NOT_APPLICABLE",
        blocking_reason=None,
        requirements=[],
        input_format="ROI color signals",
        output_format="rPPG signal",
        preprocessing="Skin segmentation, ROI extraction",
        training_required=False,
        notes="Baseline method. Already available in Sanubari V2 via "
              "MethodArbitrator. Compare V2 full system vs CHROME alone."
    ),
    ReferenceMethod(
        name="POS",
        repo="Integrated in Sanubari",
        license="N/A",
        execution_status="NOT_APPLICABLE",
        blocking_reason=None,
        requirements=[],
        input_format="ROI color signals",
        output_format="rPPG signal",
        preprocessing="Skin segmentation, ROI extraction",
        training_required=False,
        notes="Phase-based method. Already available in Sanubari V2 via "
              "MethodArbitrator. Compare V2 full system vs POS alone."
    ),
    ReferenceMethod(
        name="GREEN",
        repo="Integrated in Sanubari",
        license="N/A",
        execution_status="NOT_APPLICABLE",
        blocking_reason=None,
        requirements=[],
        input_format="ROI color signals",
        output_format="rPPG signal",
        preprocessing="Skin segmentation, ROI extraction",
        training_required=False,
        notes="Simple green channel method. Already available in Sanubari V2 via "
              "MethodArbitrator. Compare V2 full system vs GREEN alone."
    ),
]


def get_executable_references() -> List[ReferenceMethod]:
    """Get reference methods that can be executed."""
    return [m for m in REFERENCE_METHODS if m.execution_status == "READY"]


def get_blocked_references() -> List[ReferenceMethod]:
    """Get reference methods that are blocked."""
    return [m for m in REFERENCE_METHODS if m.execution_status == "BLOCKED"]


def get_not_executed_references() -> List[ReferenceMethod]:
    """Get reference methods that can be set up but haven't been run."""
    return [m for m in REFERENCE_METHODS if m.execution_status == "NOT_EXECUTED"]


def get_internal_methods() -> List[ReferenceMethod]:
    """Get methods already integrated in Sanubari V2."""
    return [m for m in REFERENCE_METHODS if m.execution_status == "NOT_APPLICABLE"]


def audit_references() -> Dict[str, Any]:
    """
    Generate reference method audit report.

    Returns
    -------
    dict
        Audit report with method status and recommendations.
    """
    executable = get_executable_references()
    blocked = get_blocked_references()
    not_executed = get_not_executed_references()
    internal = get_internal_methods()

    return {
        "summary": {
            "total": len(REFERENCE_METHODS),
            "executable": len(executable),
            "blocked": len(blocked),
            "not_executed": len(not_executed),
            "internal": len(internal),
        },
        "executable_methods": [
            {
                "name": m.name,
                "repo": m.repo,
                "requirements": m.requirements,
            }
            for m in executable
        ],
        "blocked_methods": [
            {
                "name": m.name,
                "repo": m.repo,
                "reason": m.blocking_reason,
                "requirements": m.requirements,
            }
            for m in blocked
        ],
        "internal_methods": [
            {
                "name": m.name,
                "notes": m.notes,
            }
            for m in internal
        ],
        "recommendation": (
            "Phase 1 focuses on infrastructure. "
            "External references are BLOCKED due to dataset/model dependencies. "
            "For Phase 1, compare Sanubari V2 vs internal methods (CHROME/POS/GREEN) "
            "using AblationConfig to disable fusion components. "
            "External reference integration requires Phase 2 dataset preparation."
        ),
        "phase2_preparation": {
            "required_actions": [
                "Convert benchmark dataset to rPPG-Toolbox format",
                "Install PyTorch and required dependencies",
                "Obtain dataset licenses if required",
                "Train or obtain pretrained models if required",
                "Create adapter for common evaluation protocol"
            ],
            "priority": "LOW - Phase 1 infrastructure takes precedence"
        },
    }


# Reference adapter interface
class ReferenceMethodAdapter:
    """
    Common interface for reference method integration.

    All reference methods should be adapted to this interface for
    unified benchmark evaluation.
    """

    def __init__(self, method_name: str):
        self.method_name = method_name

    def process_video(
        self,
        video_path: str,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        Process video and return timestamped predictions.

        Returns
        -------
        List[Dict]
            List of predictions with keys:
            - timestamp: float
            - bpm: float
            - confidence: float (optional)
            - method: str
        """
        raise NotImplementedError(
            f"Reference adapter for {self.method_name} not implemented"
        )


if __name__ == "__main__":
    import json
    report = audit_references()
    print(json.dumps(report, indent=2))
