"""
P10 Real-World Stress Lab Tests

Tests for stress condition evaluation infrastructure.
"""

import pytest
import torch
import numpy as np
import sys
import os
import json
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p10_stress_lab import (
    ExperimentRecord,
    SessionRecord,
    WindowRecord,
    ConditionType,
    StressCondition,
    ExperimentConfig,
    QualityLevel,
    detect_lighting_condition,
    detect_motion_level,
    detect_roi_stability,
    classify_window_quality,
    get_supported_conditions,
    WindowMetrics,
    SessionMetrics,
    ConditionMetrics,
    ExperimentMetrics,
    compute_window_metrics,
    aggregate_session_metrics,
    aggregate_condition_metrics,
    aggregate_experiment_metrics,
    StressLabEvaluator,
    StressLabRunner,
    run_stress_experiment,
    generate_report,
    save_report,
    load_report,
    ExperimentReport,
)


class TestBaseStructures:
    """Test base data structures."""

    def test_condition_type_enum(self):
        """Test ConditionType enum values."""
        assert ConditionType.LIGHTING_NORMAL.value == "lighting_normal"
        assert ConditionType.MOTION_SEVERE.value == "motion_severe"
        assert ConditionType.SQI_FAILED.value == "sqi_failed"

    def test_stress_condition(self):
        """Test StressCondition creation."""
        cond = StressCondition(
            condition_type=ConditionType.LIGHTING_LOW,
            severity=0.7,
            evidence={"brightness": 30.0},
            description="Low lighting detected",
        )
        assert cond.condition_type == ConditionType.LIGHTING_LOW
        assert cond.severity == 0.7
        assert cond.is_detected(0.5) is True
        assert cond.is_detected(0.8) is False

    def test_window_record(self):
        """Test WindowRecord creation."""
        window = WindowRecord(
            window_id="w1",
            session_id="s1",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            hr_estimate=72.0,
            hr_reference=70.0,
            signal_quality=0.8,
            is_valid=True,
        )
        assert window.window_id == "w1"
        assert window.duration == 10.0
        assert window.is_valid is True

    def test_window_to_dict(self):
        """Test WindowRecord serialization."""
        window = WindowRecord(
            window_id="w1",
            session_id="s1",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            hr_estimate=72.0,
            is_valid=True,
        )
        d = window.to_dict()
        assert d["window_id"] == "w1"
        assert d["hr_estimate"] == 72.0

    def test_session_record(self):
        """Test SessionRecord creation."""
        session = SessionRecord(
            session_id="s1",
            experiment_id="exp1",
            subject_id="subj1",
            dataset="test_data",
        )
        assert session.session_id == "s1"
        assert session.num_windows == 0
        assert session.num_valid_windows == 0

    def test_experiment_record(self):
        """Test ExperimentRecord creation."""
        exp = ExperimentRecord(
            experiment_id="exp1",
            name="Test Experiment",
            description="Test description",
        )
        assert exp.experiment_id == "exp1"
        assert exp.num_sessions == 0
        assert exp.num_windows == 0


class TestConditionDetection:
    """Test condition detection functions."""

    def test_detect_lighting_normal(self):
        """Test normal lighting detection."""
        rgb = np.array([120.0, 125.0, 130.0])
        cond = detect_lighting_condition(rgb)
        assert cond.condition_type == ConditionType.LIGHTING_NORMAL
        assert cond.severity == 0.0

    def test_detect_lighting_low(self):
        """Test low lighting detection."""
        rgb = np.array([20.0, 20.0, 20.0])  # Very low brightness
        cond = detect_lighting_condition(rgb)
        assert cond.condition_type == ConditionType.LIGHTING_LOW
        assert cond.severity > 0.5

    def test_detect_lighting_no_data(self):
        """Test lighting detection with no data."""
        cond = detect_lighting_condition(None)
        assert cond.condition_type == ConditionType.UNKNOWN

    def test_detect_motion_none(self):
        """Test no motion detection."""
        cond = detect_motion_level()
        assert cond.condition_type == ConditionType.MOTION_NONE
        assert cond.severity == 0.0

    def test_detect_motion_severe(self):
        """Test severe motion detection."""
        motion = np.array([0.2, 0.3, 0.25])  # High motion
        cond = detect_motion_level(motion_vectors=motion)
        assert cond.condition_type == ConditionType.MOTION_SEVERE
        assert cond.severity > 0.7

    def test_detect_roi_stable(self):
        """Test ROI stability."""
        cond = detect_roi_stability(roi_coverage=1.0)
        assert cond.condition_type == ConditionType.ROI_STABLE
        assert cond.severity == 0.0

    def test_detect_roi_partial_loss(self):
        """Test partial ROI loss."""
        cond = detect_roi_stability(roi_coverage=0.85)
        assert cond.condition_type == ConditionType.ROI_PARTIAL_LOSS
        assert cond.severity > 0.0

    def test_classify_window_quality_high(self):
        """Test high quality classification."""
        sqi_type, score = classify_window_quality(sqi=0.9)
        assert sqi_type == ConditionType.SQI_HIGH
        assert score > 0.8

    def test_classify_window_quality_low(self):
        """Test low quality classification."""
        sqi_type, score = classify_window_quality(sqi=0.25)  # Between low and failed
        assert sqi_type == ConditionType.SQI_FAILED
        assert score == 0.25  # Returns original SQI value

    def test_classify_window_quality_medium(self):
        """Test medium quality classification."""
        sqi_type, score = classify_window_quality(sqi=0.6)  # Medium range
        assert sqi_type == ConditionType.SQI_MEDIUM
        assert 0.5 <= score <= 0.8

    def test_classify_window_quality_no_data(self):
        """Test quality classification with no data."""
        sqi_type, score = classify_window_quality()
        assert sqi_type == ConditionType.SQI_FAILED
        assert score == 0.0

    def test_get_supported_conditions(self):
        """Test supported conditions list."""
        conditions = get_supported_conditions()
        assert ConditionType.LIGHTING_NORMAL in conditions
        assert ConditionType.MOTION_SEVERE in conditions
        assert ConditionType.SQI_HIGH in conditions


class TestMetrics:
    """Test metric computation."""

    def test_compute_window_metrics_valid(self):
        """Test window metrics with valid window."""
        window = WindowRecord(
            window_id="w1",
            session_id="s1",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            hr_estimate=72.0,
            hr_reference=70.0,
            signal_quality=0.8,
            is_valid=True,
        )
        metrics = compute_window_metrics(window)
        assert metrics.window_id == "w1"
        assert metrics.is_valid is True
        assert metrics.hr_error == 2.0
        assert metrics.hr_bias == 2.0

    def test_compute_window_metrics_invalid(self):
        """Test window metrics with invalid window."""
        window = WindowRecord(
            window_id="w1",
            session_id="s1",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            is_valid=False,
            failure_reason="Signal lost",
        )
        metrics = compute_window_metrics(window)
        assert metrics.is_valid is False

    def test_aggregate_session_metrics(self):
        """Test session metrics aggregation."""
        session = SessionRecord(
            session_id="s1",
            experiment_id="exp1",
        )
        # Add windows
        for i in range(5):
            window = WindowRecord(
                window_id=f"w{i}",
                session_id="s1",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                duration=10.0,
                hr_estimate=70.0 + i,
                hr_reference=70.0,
                signal_quality=0.8,
                is_valid=True,
            )
            session.windows.append(window)

        metrics = aggregate_session_metrics(session)
        assert metrics.session_id == "s1"
        assert metrics.total_windows == 5
        assert metrics.valid_windows == 5
        assert metrics.hr_mae is not None

    def test_aggregate_condition_metrics(self):
        """Test condition metrics aggregation."""
        windows = []
        for i in range(10):
            window = WindowRecord(
                window_id=f"w{i}",
                session_id="s1",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                duration=10.0,
                hr_estimate=70.0 + i,
                hr_reference=70.0,
                signal_quality=0.8 if i < 5 else 0.3,
                is_valid=True,
                conditions=[
                    StressCondition(
                        condition_type=ConditionType.LIGHTING_LOW if i < 5 else ConditionType.LIGHTING_NORMAL,
                        severity=0.8 if i < 5 else 0.0,
                    )
                ],
            )
            windows.append(window)

        metrics = aggregate_condition_metrics(windows, ConditionType.LIGHTING_LOW)
        assert metrics.condition_type == "lighting_low"
        assert metrics.total_windows == 5

    def test_aggregate_experiment_metrics(self):
        """Test experiment metrics aggregation."""
        exp = ExperimentRecord(
            experiment_id="exp1",
            name="Test",
        )
        session = SessionRecord(
            session_id="s1",
            experiment_id="exp1",
        )
        for i in range(3):
            window = WindowRecord(
                window_id=f"w{i}",
                session_id="s1",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                duration=10.0,
                hr_estimate=70.0,
                hr_reference=70.0,
                signal_quality=0.8,
                is_valid=True,
            )
            session.windows.append(window)
        exp.sessions.append(session)

        metrics = aggregate_experiment_metrics(exp)
        assert metrics.experiment_id == "exp1"
        assert metrics.total_sessions == 1
        assert metrics.total_windows == 3


class TestRunner:
    """Test experiment runner."""

    def test_create_experiment(self):
        """Test experiment creation."""
        runner = StressLabRunner()
        exp = runner.create_experiment("Test Experiment", "Test description")
        assert exp.name == "Test Experiment"
        assert exp.experiment_id is not None

    def test_add_session(self):
        """Test session addition."""
        runner = StressLabRunner()
        runner.create_experiment("Test")
        session = runner.add_session(
            session_id="s1",
            subject_id="subj1",
            dataset="test_data",
        )
        assert session.session_id == "s1"
        assert session.subject_id == "subj1"

    def test_add_window(self):
        """Test window addition."""
        runner = StressLabRunner()
        runner.create_experiment("Test")
        runner.add_session(session_id="s1")
        window = runner.add_window(
            session_id="s1",
            window_id="w1",
            start_time=0.0,
            end_time=10.0,
            hr_estimate=72.0,
            hr_reference=70.0,
        )
        assert window.window_id == "w1"
        assert window.hr_estimate == 72.0

    def test_add_synthetic_windows(self):
        """Test synthetic window generation."""
        runner = StressLabRunner()
        runner.create_experiment("Test")
        runner.add_session(session_id="s1")
        windows = runner.add_synthetic_windows(session_id="s1", num_windows=5)
        assert len(windows) == 5
        assert all(w.hr_estimate is not None for w in windows)
        assert all(w.hr_reference is not None for w in windows)

    def test_run_experiment(self):
        """Test experiment execution."""
        runner = StressLabRunner()
        runner.create_experiment("Test")
        runner.add_session(session_id="s1")
        runner.add_synthetic_windows(session_id="s1", num_windows=5)

        exp, metrics = runner.run()
        assert exp.num_windows == 5
        assert metrics.total_windows == 5
        assert metrics.hr_mae is not None


class TestEvaluator:
    """Test evaluator."""

    def test_evaluate_session(self):
        """Test session evaluation."""
        session = SessionRecord(
            session_id="s1",
            experiment_id="exp1",
        )
        for i in range(3):
            window = WindowRecord(
                window_id=f"w{i}",
                session_id="s1",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                duration=10.0,
                hr_estimate=70.0,
                hr_reference=70.0,
                signal_quality=0.8,
                is_valid=True,
            )
            session.windows.append(window)

        evaluator = StressLabEvaluator()
        session, metrics = evaluator.evaluate_session(session)
        assert metrics.valid_windows == 3
        assert len(session.windows[0].conditions) > 0

    def test_evaluate_experiment(self):
        """Test experiment evaluation."""
        exp = ExperimentRecord(experiment_id="exp1", name="Test")
        session = SessionRecord(session_id="s1", experiment_id="exp1")
        for i in range(3):
            window = WindowRecord(
                window_id=f"w{i}",
                session_id="s1",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                duration=10.0,
                hr_estimate=70.0,
                hr_reference=70.0,
                signal_quality=0.8,
                is_valid=True,
            )
            session.windows.append(window)
        exp.sessions.append(session)

        evaluator = StressLabEvaluator()
        exp, metrics = evaluator.evaluate_experiment(exp)
        assert metrics.total_windows == 3
        assert "lighting_normal" in metrics.condition_metrics or "all" in metrics.condition_metrics


class TestReport:
    """Test report generation."""

    def test_generate_report(self):
        """Test report generation."""
        exp = ExperimentRecord(experiment_id="exp1", name="Test")
        session = SessionRecord(session_id="s1", experiment_id="exp1")
        for i in range(3):
            window = WindowRecord(
                window_id=f"w{i}",
                session_id="s1",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                duration=10.0,
                hr_estimate=70.0,
                hr_reference=70.0,
                signal_quality=0.8,
                is_valid=True,
            )
            session.windows.append(window)
        exp.sessions.append(session)

        metrics = aggregate_experiment_metrics(exp)
        report = generate_report(exp, metrics)

        assert isinstance(report, ExperimentReport)
        assert report.experiment_id == "exp1"
        assert report.total_windows == 3

    def test_report_to_dict(self):
        """Test report serialization."""
        report = ExperimentReport(
            experiment_id="exp1",
            name="Test",
            description="Test description",
            created_at="2024-01-01",
            total_sessions=1,
            total_windows=3,
            valid_windows=3,
            failed_windows=0,
            overall_valid_rate=1.0,
            hr_mae=2.0,
            hr_rmse=2.5,
            hr_bias=0.5,
            hr_std=1.0,
            session_summaries=[],
            condition_summaries=[],
            quality_summaries=[],
            limitations=["Small sample"],
            methodology="Test methodology",
        )
        d = report.to_dict()
        assert d["experiment_id"] == "exp1"
        assert d["summary"]["total_windows"] == 3

    def test_report_to_markdown(self):
        """Test markdown report generation."""
        report = ExperimentReport(
            experiment_id="exp1",
            name="Test",
            description="Test",
            created_at="2024-01-01",
            total_sessions=1,
            total_windows=3,
            valid_windows=3,
            failed_windows=0,
            overall_valid_rate=1.0,
            hr_mae=2.0,
            hr_rmse=None,
            hr_bias=None,
            hr_std=None,
            session_summaries=[],
            condition_summaries=[],
            quality_summaries=[],
            limitations=[],
            methodology="Test",
        )
        md = report.to_markdown()
        assert "# Stress Lab Experiment Report" in md
        assert "exp1" in md


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test complete evaluation pipeline."""
        runner = StressLabRunner()

        # Create experiment
        exp = runner.create_experiment(
            "Integration Test",
            "Testing full pipeline",
        )

        # Add sessions with different conditions
        # Session 1: Good conditions
        runner.add_session(session_id="s1", subject_id="subj1", dataset="good_conditions")
        runner.add_synthetic_windows(
            session_id="s1",
            num_windows=5,
            sqi_range=(0.7, 1.0),
            hr_error=3.0,
        )

        # Session 2: Poor conditions
        runner.add_session(session_id="s2", subject_id="subj2", dataset="poor_conditions")
        runner.add_synthetic_windows(
            session_id="s2",
            num_windows=5,
            sqi_range=(0.2, 0.4),
            hr_error=8.0,
        )

        # Run evaluation
        exp, metrics = runner.run()

        # Verify results
        assert metrics.total_sessions == 2
        assert metrics.total_windows == 10
        assert metrics.hr_mae is not None

        # Generate report
        report = generate_report(exp, metrics)
        assert isinstance(report, ExperimentReport)
        assert report.total_windows == 10


def run_all_tests():
    """Run all tests."""
    test_classes = [
        TestBaseStructures,
        TestConditionDetection,
        TestMetrics,
        TestRunner,
        TestEvaluator,
        TestReport,
        TestIntegration,
    ]

    total_passed = 0
    total_failed = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"  {test_class.__name__}")
        print(f"{'='*60}")

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in test_methods:
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  {method_name}... PASS")
                total_passed += 1
            except Exception as e:
                print(f"  {method_name}... FAIL ({str(e)[:80]})")
                total_failed += 1
                failed_tests.append(f"{test_class.__name__}.{method_name}")

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Tests: {total_passed + total_failed}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    if failed_tests:
        print(f"\n  Failed tests:")
        for t in failed_tests:
            print(f"    - {t}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
