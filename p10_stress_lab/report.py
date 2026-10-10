"""
P10 Real-World Stress Lab - Report Generation

Report generation for stress condition experiments.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .base import ExperimentRecord
from .metrics import ExperimentMetrics, SessionMetrics, ConditionMetrics


@dataclass
class ExperimentReport:
    """Report for a stress condition experiment."""
    experiment_id: str
    name: str
    description: str
    created_at: str

    # Counts
    total_sessions: int
    total_windows: int
    valid_windows: int
    failed_windows: int
    overall_valid_rate: float

    # HR metrics
    hr_mae: Optional[float]
    hr_rmse: Optional[float]
    hr_bias: Optional[float]
    hr_std: Optional[float]

    # Per-session summary
    session_summaries: List[Dict]

    # Per-condition summary
    condition_summaries: List[Dict]

    # Per-quality summary
    quality_summaries: List[Dict]

    # Limitations
    limitations: List[str]

    # Methodology
    methodology: str

    @classmethod
    def from_metrics(
        cls,
        experiment: ExperimentRecord,
        metrics: ExperimentMetrics,
    ) -> "ExperimentReport":
        """Create report from experiment and metrics."""
        limitations = []
        methodology_parts = []

        # Check limitations
        if metrics.valid_windows == 0:
            limitations.append("No valid windows for evaluation")

        if metrics.total_windows < 10:
            limitations.append(f"Very small sample size ({metrics.total_windows} windows)")

        if not experiment.sessions:
            limitations.append("No sessions in experiment")
        else:
            # Check for missing reference data
            windows_with_ref = [
                w for s in experiment.sessions
                for w in s.windows
                if w.hr_reference is not None
            ]
            if len(windows_with_ref) < metrics.total_windows * 0.5:
                limitations.append(
                    f"Only {len(windows_with_ref)}/{metrics.total_windows} windows have reference HR"
                )

        # Methodology description
        methodology_parts.append("Stress condition evaluation using synthetic test data")
        methodology_parts.append(f"HR error threshold: 10.0 BPM")
        methodology_parts.append(f"Condition detection based on SQI and confidence thresholds")

        # Build session summaries
        session_summaries = []
        for session_id, sm in metrics.session_metrics.items():
            session_summaries.append({
                "session_id": session_id,
                "total_windows": sm.total_windows,
                "valid_windows": sm.valid_windows,
                "valid_rate": sm.valid_rate,
                "hr_mae": sm.hr_mae,
                "mean_sqi": sm.mean_sqi,
            })

        # Build condition summaries
        condition_summaries = []
        for cond_type, cm in metrics.condition_metrics.items():
            if cm.total_windows > 0:
                condition_summaries.append({
                    "condition": cond_type,
                    "total_windows": cm.total_windows,
                    "valid_windows": cm.valid_windows,
                    "valid_rate": cm.valid_rate,
                    "reliable_rate": cm.reliable_rate,
                    "hr_mae": cm.hr_mae,
                    "hr_rmse": cm.hr_rmse,
                })

        # Build quality summaries
        quality_summaries = []
        for quality, qm in metrics.quality_metrics.items():
            quality_summaries.append({
                "quality_level": quality,
                "total_windows": qm.total_windows,
                "valid_windows": qm.valid_windows,
                "valid_rate": qm.valid_rate,
                "hr_mae": qm.hr_mae,
            })

        return cls(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            description=experiment.description,
            created_at=(
                experiment.created_at.isoformat()
                if hasattr(experiment.created_at, 'isoformat')
                else str(experiment.created_at)
            ),
            total_sessions=metrics.total_sessions,
            total_windows=metrics.total_windows,
            valid_windows=metrics.valid_windows,
            failed_windows=metrics.failed_windows,
            overall_valid_rate=metrics.overall_valid_rate,
            hr_mae=metrics.hr_mae,
            hr_rmse=metrics.hr_rmse,
            hr_bias=metrics.hr_bias,
            hr_std=metrics.hr_std,
            session_summaries=session_summaries,
            condition_summaries=condition_summaries,
            quality_summaries=quality_summaries,
            limitations=limitations,
            methodology="; ".join(methodology_parts),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "summary": {
                "total_sessions": self.total_sessions,
                "total_windows": self.total_windows,
                "valid_windows": self.valid_windows,
                "failed_windows": self.failed_windows,
                "overall_valid_rate": self.overall_valid_rate,
            },
            "hr_metrics": {
                "mae": self.hr_mae,
                "rmse": self.hr_rmse,
                "bias": self.hr_bias,
                "std": self.hr_std,
            },
            "session_summaries": self.session_summaries,
            "condition_summaries": self.condition_summaries,
            "quality_summaries": self.quality_summaries,
            "limitations": self.limitations,
            "methodology": self.methodology,
        }

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# Stress Lab Experiment Report",
            f"",
            f"**Experiment**: {self.name}",
            f"**ID**: {self.experiment_id}",
            f"**Date**: {self.created_at}",
            f"",
        ]

        if self.description:
            lines.extend([f"## Description", f"{self.description}", f""])

        # Summary
        lines.extend([
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Sessions | {self.total_sessions} |",
            f"| Total Windows | {self.total_windows} |",
            f"| Valid Windows | {self.valid_windows} |",
            f"| Failed Windows | {self.failed_windows} |",
            f"| Valid Rate | {self.overall_valid_rate:.1%} |",
            f"",
        ])

        # HR Metrics
        if self.hr_mae is not None:
            lines.extend([
                f"## HR Estimation Metrics",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| MAE | {self.hr_mae:.2f} BPM |",
                f"| RMSE | {self.hr_rmse:.2f} BPM |" if self.hr_rmse else "",
                f"| Bias | {self.hr_bias:+.2f} BPM |" if self.hr_bias is not None else "",
                f"| Std | {self.hr_std:.2f} BPM |" if self.hr_std else "",
                f"",
            ])

        # Session Summary
        if self.session_summaries:
            lines.extend([
                f"## Per-Session Summary",
                f"",
                f"| Session | Windows | Valid | Valid Rate | HR MAE | Mean SQI |",
                f"|---------|---------|-------|------------|--------|----------|",
            ])
            for s in self.session_summaries:
                lines.append(
                    f"| {s['session_id']} | {s['total_windows']} | "
                    f"{s['valid_windows']} | {s['valid_rate']:.1%} | "
                    f"{s.get('hr_mae', 'N/A'):.2f} | "
                    f"{s.get('mean_sqi', 'N/A'):.2f} |"
                )
            lines.append("")

        # Condition Summary
        if self.condition_summaries:
            lines.extend([
                f"## Per-Condition Summary",
                f"",
                f"| Condition | Windows | Valid | HR MAE |",
                f"|-----------|---------|-------|--------|",
            ])
            for c in self.condition_summaries:
                hr_str = f"{c['hr_mae']:.2f}" if c.get('hr_mae') else "N/A"
                lines.append(
                    f"| {c['condition']} | {c['total_windows']} | "
                    f"{c['valid_windows']} | {hr_str} |"
                )
            lines.append("")

        # Limitations
        if self.limitations:
            lines.extend([
                f"## Limitations",
                f"",
            ])
            for lim in self.limitations:
                lines.append(f"- {lim}")
            lines.append("")

        # Methodology
        lines.extend([
            f"## Methodology",
            f"",
            f"{self.methodology}",
            f"",
            f"---",
            f"",
            f"*Note: This report contains synthetic test data. Real-world validation*",
            f"*requires experiments with diverse recording conditions.*",
        ])

        return "\n".join(lines)


def generate_report(
    experiment: ExperimentRecord,
    metrics: ExperimentMetrics,
) -> ExperimentReport:
    """
    Generate experiment report.

    Args:
        experiment: ExperimentRecord
        metrics: ExperimentMetrics

    Returns:
        ExperimentReport
    """
    return ExperimentReport.from_metrics(experiment, metrics)


def save_report(
    report: ExperimentReport,
    filepath: str,
    format: str = "json",
) -> None:
    """
    Save report to file.

    Args:
        report: ExperimentReport
        filepath: Output path
        format: Output format ("json" or "markdown")
    """
    if format == "json":
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    elif format == "markdown":
        with open(filepath, 'w') as f:
            f.write(report.to_markdown())
    else:
        raise ValueError(f"Unknown format: {format}")


def load_report(filepath: str) -> ExperimentReport:
    """
    Load report from file.

    Args:
        filepath: Path to report file

    Returns:
        ExperimentReport
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return ExperimentReport(
        experiment_id=data["experiment_id"],
        name=data["name"],
        description=data.get("description", ""),
        created_at=data.get("created_at", ""),
        total_sessions=data["summary"]["total_sessions"],
        total_windows=data["summary"]["total_windows"],
        valid_windows=data["summary"]["valid_windows"],
        failed_windows=data["summary"]["failed_windows"],
        overall_valid_rate=data["summary"]["overall_valid_rate"],
        hr_mae=data["hr_metrics"].get("mae"),
        hr_rmse=data["hr_metrics"].get("rmse"),
        hr_bias=data["hr_metrics"].get("bias"),
        hr_std=data["hr_metrics"].get("std"),
        session_summaries=data.get("session_summaries", []),
        condition_summaries=data.get("condition_summaries", []),
        quality_summaries=data.get("quality_summaries", []),
        limitations=data.get("limitations", []),
        methodology=data.get("methodology", ""),
    )
