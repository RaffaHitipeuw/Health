"""
Phase 2: Classical Signal Evolution - Source Audit

Documents classical rPPG methods from academic literature and implementations.
Each method is traced to its source with license, algorithm details, and integration status.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class MethodStatus(Enum):
    """Integration status for classical methods."""
    INTEGRATED = "integrated"  # Implemented and working
    REFERENCE_ONLY = "reference_only"  # Documented but not integrated
    BLOCKED = "blocked"  # Cannot be integrated due to dependencies/license
    NOT_FOUND = "not_found"  # Source not available


class License(Enum):
    """License types for referenced implementations."""
    MIT = "MIT"
    APACHE2 = "Apache-2.0"
    BSD = "BSD"
    GPL = "GPL"
    RESEARCH_ONLY = "research_only"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


@dataclass
class ClassicalMethodSource:
    """Source documentation for a classical rPPG method."""
    name: str
    paper: Optional[str] = None
    repo: Optional[str] = None
    license: License = License.UNKNOWN
    status: MethodStatus = MethodStatus.NOT_FOUND

    # Algorithm characteristics
    core_idea: str = ""
    input_channels: List[str] = field(default_factory=list)
    projection: str = ""
    normalization: str = ""
    temporal_filtering: str = ""
    detrending: str = ""

    # Implementation details
    implementation: str = ""
    dependencies: List[str] = field(default_factory=list)

    # Sanubari relevance
    integrated_in_sanubari: bool = False
    sanubari_module: Optional[str] = None

    # Known characteristics
    strengths: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    computational_complexity: str = ""

    # Research notes
    notes: str = ""
    blocking_reason: Optional[str] = None


# =============================================================================
# CLASSICAL METHOD INVENTORY
# =============================================================================

CLASSICAL_METHODS: Dict[str, ClassicalMethodSource] = {

    # -------------------------------------------------------------------------
    # GREEN CHANNEL METHOD
    # -------------------------------------------------------------------------
    "GREEN": ClassicalMethodSource(
        name="GREEN",
        paper="Verkruysse et al. 2007",
        license=License.UNKNOWN,
        status=MethodStatus.INTEGRATED,
        core_idea="Extract pulse signal from green channel reflectance",
        input_channels=["G"],
        projection="None (raw green channel)",
        normalization="Mean subtraction",
        temporal_filtering="Bandpass 0.5-4 Hz",
        detrending="Linear detrend",
        implementation="rppg_algorithms.py::extract_green",
        dependencies=["numpy", "scipy.signal"],
        integrated_in_sanubari=True,
        sanubari_module="rppg_algorithms.py",
        strengths=[
            "Simplicity",
            "Green channel has peak absorption in blood",
            "Low computational cost",
        ],
        failure_modes=[
            "Susceptible to illumination changes",
            "Motion artifacts",
            "Weak pulsatile signal in dark skin",
        ],
        computational_complexity="O(n)",
        notes="Baseline method. Performance highly dependent on ROI quality.",
    ),

    # -------------------------------------------------------------------------
    # CHROMINANCE METHOD
    # -------------------------------------------------------------------------
    "CHROM": ClassicalMethodSource(
        name="CHROM",
        paper="Haan & Jeanne 2013",
        license=License.UNKNOWN,
        status=MethodStatus.INTEGRATED,
        core_idea="Chrominance-based signal separation exploiting skin optics",
        input_channels=["R", "G", "B"],
        projection="Xc = 3Rn - 2Gn, Ys = 1.5Rn + Gn - 1.5Bn",
        normalization="Channel normalization by mean",
        temporal_filtering="Bandpass 0.5-4 Hz after projection",
        detrending="Linear + moving average",
        implementation="rppg_algorithms.py::extract_chrom",
        dependencies=["numpy", "scipy.signal"],
        integrated_in_sanubari=True,
        sanubari_module="rppg_algorithms.py",
        strengths=[
            "Robust to illumination changes",
            "Skin tone independence (theoretical)",
            "Good performance under motion",
        ],
        failure_modes=[
            "Harmonic confusion at 2f_hr",
            "Breathing interference",
            "Exposure drift",
        ],
        computational_complexity="O(n)",
        notes="Widely used baseline. CHROM in Sanubari uses alpha = std(Xs)/std(Ys) normalization.",
    ),

    # -------------------------------------------------------------------------
    # PLANE-ORTHOGONAL-TO-SKIN METHOD
    # -------------------------------------------------------------------------
    "POS": ClassicalMethodSource(
        name="POS",
        paper="Wang et al. 2017",
        license=License.UNKNOWN,
        status=MethodStatus.INTEGRATED,
        core_idea="Plane-orthogonal-to-skin color subspace projection",
        input_channels=["R", "G", "B"],
        projection="S = (G-B), P = G + B - 2R with alpha normalization",
        normalization="Standard deviation normalization",
        temporal_filtering="Bandpass 0.5-4 Hz",
        detrending="Linear detrend",
        implementation="rppg_algorithms.py::extract_pos",
        dependencies=["numpy", "scipy.signal"],
        integrated_in_sanubari=True,
        sanubari_module="rppg_algorithms.py",
        strengths=[
            "Robust to motion artifacts",
            "Good skin-tone independence",
            "Strong pulsatile signal extraction",
        ],
        failure_modes=[
            "Sensitive to ROI quality",
            "Harmonic/sub-harmonic confusion",
            "Computational overhead vs GREEN",
        ],
        computational_complexity="O(n)",
        notes="Default method in Sanubari V2. MethodArbitrator can select between POS/CHROM/GREEN.",
    ),

    # -------------------------------------------------------------------------
    # INDEPENDENT COMPONENT ANALYSIS
    # -------------------------------------------------------------------------
    "ICA": ClassicalMethodSource(
        name="ICA",
        paper="Poh et al. 2010",
        license=License.UNKNOWN,
        status=MethodStatus.INTEGRATED,
        core_idea="Blind source separation using FastICA on RGB channels",
        input_channels=["R", "G", "B"],
        projection="ICA decomposition, select component with cardiac peak",
        normalization="Z-score before ICA",
        temporal_filtering="None (source separation provides filtering)",
        detrending="None (ICA handles)",
        implementation="rppg_algorithms.py::extract_ica",
        dependencies=["numpy", "sklearn.decomposition.FastICA"],
        integrated_in_sanubari=True,
        sanubari_module="rppg_algorithms.py",
        strengths=[
            "Theoretically optimal source separation",
            "Handles multiple signal sources",
        ],
        failure_modes=[
            "Non-deterministic (random initialization)",
            "Cardiac component selection heuristic",
            "Computationally expensive",
            "Requires sufficient samples",
        ],
        computational_complexity="O(n * k^2) where k=3 components",
        notes="Uses sklearn FastICA. Best component selected by FFT peak in cardiac band.",
    ),

    # -------------------------------------------------------------------------
    # PBV (Pulse Blood Volume) - REFERENCE ONLY
    # -------------------------------------------------------------------------
    "PBV": ClassicalMethodSource(
        name="PBV",
        paper="Jimenez et al. 2019",
        license=License.RESEARCH_ONLY,
        status=MethodStatus.REFERENCE_ONLY,
        core_idea="Pulse Blood Volume signature using rgb pulse spectrum",
        input_channels=["R", "G", "B"],
        projection="PBV vector = [rgb_pulse_spectrum]",
        normalization="PBV subspace projection",
        temporal_filtering="Spectral analysis",
        detrending="High-pass filter",
        implementation="Not integrated",
        dependencies=["numpy", "scipy"],
        integrated_in_sanubari=False,
        sanubari_module=None,
        strengths=[
            "Theoretical basis in blood volume changes",
            "Good for PPG waveform analysis",
        ],
        failure_modes=[
            "Complex implementation",
            "Requires spectral analysis pipeline",
        ],
        computational_complexity="O(n log n)",
        notes="REFERENCE ONLY. Requires specific spectral processing pipeline not in Sanubari.",
    ),

    # -------------------------------------------------------------------------
    # LGI (Local Group Invariance) - REFERENCE ONLY
    # -------------------------------------------------------------------------
    "LGI": ClassicalMethodSource(
        name="LGI",
        paper="Pilz et al. 2018",
        license=License.UNKNOWN,
        status=MethodStatus.REFERENCE_ONLY,
        core_idea="Local Group Invariance across spatial locations",
        input_channels=["R", "G", "B"],
        projection="Spatial invariance testing",
        normalization="Temporal normalization",
        temporal_filtering="Spatial averaging + bandpass",
        detrending="Running average subtraction",
        implementation="Not integrated",
        dependencies=["numpy", "scipy"],
        integrated_in_sanubari=False,
        sanubari_module=None,
        strengths=[
            "Spatial robustness",
            "Handles partial occlusion",
        ],
        failure_modes=[
            "Computationally intensive (spatial processing)",
            "Complex implementation",
        ],
        computational_complexity="O(n * m) where m = spatial locations",
        notes="REFERENCE ONLY. Requires spatial invariance testing pipeline.",
    ),

    # -------------------------------------------------------------------------
    # EXTERNAL: Sameed66 - REFERENCE ONLY
    # -------------------------------------------------------------------------
    "Sameed": ClassicalMethodSource(
        name="Sameed66/Contactless-Heart-Rate-Estimation",
        repo="https://github.com/Sameed66/Contactless-Heart-Rate-Estimation-from-Facial-Video-rPPG",
        license=License.UNKNOWN,
        status=MethodStatus.REFERENCE_ONLY,
        core_idea="Contactless HR estimation from facial video",
        input_channels=["R", "G", "B"],
        projection="Green channel + filtering",
        normalization="Standard normalization",
        temporal_filtering="Bandpass + smoothing",
        detrending="Linear detrend",
        implementation="Not integrated (no direct code access)",
        dependencies=["OpenCV", "NumPy"],
        integrated_in_sanubari=False,
        sanubari_module=None,
        strengths=[
            "Simple implementation",
            "Fast processing",
        ],
        failure_modes=[
            "Basic method, limited robustness",
            "No advanced signal processing",
        ],
        computational_complexity="O(n)",
        notes="REFERENCE ONLY. GitHub repo not analyzed in detail. License unknown.",
    ),

    # -------------------------------------------------------------------------
    # EXTERNAL: marnixnaber - REFERENCE ONLY
    # -------------------------------------------------------------------------
    "marnixnaber": ClassicalMethodSource(
        name="marnixnaber/rPPG",
        repo="https://github.com/marnixnaber/rPPG",
        license=License.MIT,
        status=MethodStatus.REFERENCE_ONLY,
        core_idea="Contactless heart rate measurement",
        input_channels=["R", "G", "B"],
        projection="Green channel extraction",
        normalization="Mean normalization",
        temporal_filtering="Butterworth bandpass",
        detrending="Polynomial detrending",
        implementation="Not integrated",
        dependencies=["OpenCV", "NumPy", "SciPy"],
        integrated_in_sanubari=False,
        sanubari_module=None,
        strengths=[
            "Clean implementation",
            "Bandpass filtering",
        ],
        failure_modes=[
            "Basic approach",
            "Motion sensitive",
        ],
        computational_complexity="O(n)",
        notes="REFERENCE ONLY. MIT licensed. Implementation principles align with GREEN baseline.",
    ),

    # -------------------------------------------------------------------------
    # EXTERNAL: rPPG-Toolbox - BLOCKED
    # -------------------------------------------------------------------------
    "rPPG-Toolbox": ClassicalMethodSource(
        name="ubicomplab/rPPG-Toolbox",
        repo="https://github.com/ubicomplab/rPPG-Toolbox",
        license=License.MIT,
        status=MethodStatus.BLOCKED,
        core_idea="Comprehensive rPPG toolbox with deep learning methods",
        input_channels=["R", "G", "B"],
        projection="Multiple methods (CHROM, POS, GREEN) + deep learning",
        normalization="Method-specific",
        temporal_filtering="Comprehensive signal processing",
        detrending="Multiple approaches",
        implementation="BLOCKED",
        dependencies=["PyTorch", "OpenCV", "NumPy", "face detection models"],
        integrated_in_sanubari=False,
        sanubari_module=None,
        blocking_reason="Requires PyTorch + GPU + face detection models. Dataset format differs from Sanubari.",
        strengths=[
            "State-of-the-art methods",
            "Comprehensive benchmarking",
            "Deep learning integration",
        ],
        failure_modes=[
            "GPU required for deep learning",
            "Complex setup",
            "Dataset format incompatibility",
        ],
        computational_complexity="O(n) to O(n * GPU) depending on method",
        notes="BLOCKED. Requires extensive setup. Classical methods align with Sanubari but deep learning is Phase 5.",
    ),

    # -------------------------------------------------------------------------
    # EXTERNAL: RemoteBio - BLOCKED
    # -------------------------------------------------------------------------
    "RemoteBio": ClassicalMethodSource(
        name="RemoteBio / remotebiosensing/rppg",
        repo="https://github.com/remotebiosensing/rppg",
        license=License.UNKNOWN,
        status=MethodStatus.BLOCKED,
        core_idea="Academic benchmark for remote photoplethysmography",
        input_channels=["R", "G", "B"],
        projection="Classical methods + benchmark protocol",
        normalization="Benchmark standardized",
        temporal_filtering="Comprehensive",
        detrending="Multiple approaches",
        implementation="BLOCKED",
        dependencies=["MATLAB Runtime", "Benchmark dataset"],
        integrated_in_sanubari=False,
        sanubari_module=None,
        blocking_reason="Requires MATLAB Runtime. Benchmark dataset not publicly available.",
        strengths=[
            "Academic benchmark protocol",
            "Standardized evaluation",
        ],
        failure_modes=[
            "Dataset availability",
            "MATLAB dependency",
        ],
        computational_complexity="Varies by method",
        notes="BLOCKED. Academic benchmark. Limited public access.",
    ),

    # -------------------------------------------------------------------------
    # EXTERNAL: MMPD - BLOCKED
    # -------------------------------------------------------------------------
    "MMPD": ClassicalMethodSource(
        name="MMPD (Multi-Environment rPPG)",
        repo="https://github.com/BlandAndr/mppd",
        license=License.UNKNOWN,
        status=MethodStatus.BLOCKED,
        core_idea="Multi-environment PPG detection using blind source separation",
        input_channels=["R", "G", "B"],
        projection="ICA/PCA-based separation",
        normalization="Statistical normalization",
        temporal_filtering="ICA separation + filtering",
        detrending="Component analysis",
        implementation="BLOCKED",
        dependencies=["MATLAB/Octave", "VIPL dataset"],
        integrated_in_sanubari=False,
        sanubari_module=None,
        blocking_reason="MATLAB/Octave implementation. VIPL dataset required.",
        strengths=[
            "Multi-environment robustness",
            "Blind source separation",
        ],
        failure_modes=[
            "Dataset dependency",
            "MATLAB required",
        ],
        computational_complexity="O(n * components)",
        notes="BLOCKED. ICA-based approach similar to Sanubari's ICA method.",
    ),
}


def get_integrated_methods() -> Dict[str, ClassicalMethodSource]:
    """Get methods that are integrated in Sanubari."""
    return {
        name: method
        for name, method in CLASSICAL_METHODS.items()
        if method.status == MethodStatus.INTEGRATED
    }


def get_reference_methods() -> Dict[str, ClassicalMethodSource]:
    """Get methods that are reference-only."""
    return {
        name: method
        for name, method in CLASSICAL_METHODS.items()
        if method.status == MethodStatus.REFERENCE_ONLY
    }


def get_blocked_methods() -> Dict[str, ClassicalMethodSource]:
    """Get methods that are blocked."""
    return {
        name: method
        for name, method in CLASSICAL_METHODS.items()
        if method.status == MethodStatus.BLOCKED
    }


def audit_sources() -> Dict[str, Any]:
    """
    Generate comprehensive source audit report.

    Returns
    -------
    dict
        Source audit with method inventory, status, and recommendations.
    """
    integrated = get_integrated_methods()
    reference = get_reference_methods()
    blocked = get_blocked_methods()

    return {
        "summary": {
            "total_methods": len(CLASSICAL_METHODS),
            "integrated": len(integrated),
            "reference_only": len(reference),
            "blocked": len(blocked),
        },
        "integrated_methods": {
            name: {
                "paper": m.paper,
                "core_idea": m.core_idea,
                "implementation": m.implementation,
                "sanubari_module": m.sanubari_module,
                "strengths": m.strengths,
                "failure_modes": m.failure_modes,
            }
            for name, m in integrated.items()
        },
        "reference_methods": {
            name: {
                "repo": m.repo,
                "license": m.license.value,
                "core_idea": m.core_idea,
                "notes": m.notes,
            }
            for name, m in reference.items()
        },
        "blocked_methods": {
            name: {
                "repo": m.repo,
                "blocking_reason": m.blocking_reason,
                "dependencies": m.dependencies,
            }
            for name, m in blocked.items()
        },
        "recommendations": {
            "for_phase2": (
                "Focus on integrated methods (GREEN, CHROM, POS, ICA). "
                "These can be benchmarked using Phase 1 infrastructure. "
                "External methods (rPPG-Toolbox, RemoteBio, MMPD) are blocked by dependencies/datasets."
            ),
            "for_phase3": (
                "Consider spatial/motion methods from reference implementations "
                "after Phase 2 classical evolution is validated."
            ),
        },
    }


if __name__ == "__main__":
    import json
    report = audit_sources()
    print(json.dumps(report, indent=2, default=str))
