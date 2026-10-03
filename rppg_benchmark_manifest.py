"""
Experiment manifest for benchmark reproducibility.

Records the exact configuration used in each experiment so it can be
reconstructed from recorded metadata.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ExperimentManifest:
    """
    Experiment manifest for reproducible benchmark execution.

    Every experiment should have a corresponding manifest that allows
    reconstruction of the exact conditions.
    """
    # Identity
    experiment_id: str = ""
    dataset: str = "unknown"
    subject_id: Optional[str] = None
    video_id: Optional[str] = None
    run_id: Optional[str] = None

    # Paths
    video_path: str = ""
    gt_path: str = ""

    # Method
    method: str = "sanubari_v2"
    method_version: str = "1.0.0"

    # Benchmark configuration
    alignment_tolerance: float = 2.0
    window_duration: float = 0.0  # 0 = single full-video window

    # Infrastructure
    timestamp: str = ""  # ISO format
    code_version: str = "unknown"
    git_commit: str = "unknown"
    seed: int = 42

    # Validation
    gt_sample_count: int = 0
    gt_timestamp_range: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        return d

    def save(self, path: str) -> str:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentManifest":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "ExperimentManifest":
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def get_git_info() -> Dict[str, str]:
    """Get git repository information if available."""
    info = {"commit": "unknown", "branch": "unknown", "clean": True}

    try:
        import subprocess
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()[:8]

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check if clean
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["clean"] = len(result.stdout.strip()) == 0
    except Exception:
        pass

    return info


def create_manifest(
    experiment_id: str,
    video_path: str,
    gt_path: str,
    dataset: str = "unknown",
    subject_id: Optional[str] = None,
    video_id: Optional[str] = None,
    run_id: Optional[str] = None,
    alignment_tolerance: float = 2.0,
    window_duration: float = 0.0,
    gt_sample_count: int = 0,
    gt_timestamp_range: Optional[List[float]] = None,
) -> ExperimentManifest:
    """
    Create an experiment manifest with current environment information.

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier.
    video_path : str
        Path to video file.
    gt_path : str
        Path to ground truth file.
    dataset : str
        Dataset name.
    subject_id : Optional[str]
        Subject identifier.
    video_id : Optional[str]
        Video identifier.
    run_id : Optional[str]
        Run identifier.
    alignment_tolerance : float
        Alignment tolerance in seconds.
    window_duration : float
        Window duration in seconds (0 = full video).
    gt_sample_count : int
        Number of GT samples.
    gt_timestamp_range : Optional[List[float]]
        GT timestamp range [min, max].

    Returns
    -------
    ExperimentManifest
        Populated manifest with environment information.
    """
    git_info = get_git_info()

    return ExperimentManifest(
        experiment_id=experiment_id,
        dataset=dataset,
        subject_id=subject_id,
        video_id=video_id,
        run_id=run_id,
        video_path=os.path.abspath(video_path),
        gt_path=os.path.abspath(gt_path),
        method="sanubari_v2",
        method_version="1.0.0",
        alignment_tolerance=alignment_tolerance,
        window_duration=window_duration,
        timestamp=datetime.now().isoformat(),
        code_version="1.0.0",
        git_commit=git_info["commit"],
        seed=42,
        gt_sample_count=gt_sample_count,
        gt_timestamp_range=gt_timestamp_range or [],
    )
