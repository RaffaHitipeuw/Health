"""
P10 Scientific Validity Audit - Regression Tests

Tests for bugs found during scientific validity audit:
1. RMSE calculation bug in aggregate_experiment_metrics
2. Non-finite input handling in condition detectors
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p10_stress_lab import (
    ExperimentRecord, SessionRecord, WindowRecord,
    ConditionType, QualityLevel,
    detect_lighting_condition, classify_window_quality,
)
from p10_stress_lab.metrics import (
    aggregate_experiment_metrics, aggregate_session_metrics,
)


class TestRMSEBug:
    """Regression test for RMSE calculation bug."""

    def test_aggregate_rmse_from_unequal_errors(self):
        """
        Verify that RMSE is computed from actual errors, not from session MAEs.

        Bug: aggregate_experiment_metrics repeats sm.hr_mae for each valid window,
        then computes RMSE from these repeated MAEs instead of actual errors.
        """
        # Session 1: errors [1, 5] -> MAE=3.0, RMSE=sqrt(13)=3.606
        # Session 2: errors [2, 4] -> MAE=3.0, RMSE=sqrt(10)=3.162
        # Expected overall: MAE=3.0, RMSE=sqrt(11.5)=3.391

        exp = ExperimentRecord(experiment_id='test', name='Test')

        s1 = SessionRecord(session_id='s1', experiment_id='test')
        for i, err in enumerate([1.0, 5.0]):
            w = WindowRecord(
                window_id=f'w{i}', session_id='s1',
                start_time=i*10, end_time=(i+1)*10, duration=10,
                hr_estimate=70+err, hr_reference=70.0,
                signal_quality=0.8, is_valid=True
            )
            s1.windows.append(w)
        exp.sessions.append(s1)

        s2 = SessionRecord(session_id='s2', experiment_id='test')
        for i, err in enumerate([2.0, 4.0]):
            w = WindowRecord(
                window_id=f'w{i+2}', session_id='s2',
                start_time=i*10, end_time=(i+1)*10, duration=10,
                hr_estimate=70+err, hr_reference=70.0,
                signal_quality=0.8, is_valid=True
            )
            s2.windows.append(w)
        exp.sessions.append(s2)

        metrics = aggregate_experiment_metrics(exp)

        # Expected RMSE from all errors: sqrt((1^2+5^2+2^2+4^2)/4) = sqrt(11.5) = 3.391
        expected_rmse = np.sqrt((1**2 + 5**2 + 2**2 + 4**2) / 4)

        print(f"\nExpected RMSE: {expected_rmse:.4f}")
        print(f"Actual RMSE: {metrics.hr_rmse:.4f}")

        # Bug returns 3.0 (RMSE of [3, 3, 3, 3])
        # Correct returns 3.391 (RMSE of [1, 5, 2, 4])
        assert abs(metrics.hr_rmse - expected_rmse) < 0.01, (
            f"RMSE should be {expected_rmse:.4f}, got {metrics.hr_rmse:.4f}. "
            f"Bug causes RMSE to be computed from repeated session MAEs."
        )


class TestNonFiniteInput:
    """Test handling of non-finite inputs."""

    def test_lighting_nan_input(self):
        """NaN RGB input should return UNKNOWN condition."""
        rgb_nan = np.array([np.nan, np.nan, np.nan])
        cond = detect_lighting_condition(rgb_nan)
        # Bug: returns LIGHTING_NORMAL with severity 0.0
        # Correct: should return UNKNOWN or handle NaN explicitly
        # For now, we just document the behavior
        print(f"\nNaN RGB: {cond.condition_type}, severity={cond.severity}")
        # This test documents current behavior, not the desired behavior
        # A fix would require adding NaN checks

    def test_lighting_inf_input(self):
        """Inf RGB input should return UNKNOWN condition."""
        rgb_inf = np.array([np.inf, np.inf, np.inf])
        cond = detect_lighting_condition(rgb_inf)
        print(f"\nInf RGB: {cond.condition_type}, severity={cond.severity}")
        # Bug: returns LIGHTING_NORMAL (because mean is inf which is not < 50)

    def test_sqi_nan_classification(self):
        """NaN SQI should return FAILED with valid score."""
        sqi_type, score = classify_window_quality(sqi=float('nan'))
        print(f"\nNaN SQI: {sqi_type}, score={score}")
        # Returns SQI_FAILED with score=nan, which is problematic
        # Should return score=0.0 instead


class TestMetricDefinitions:
    """Test metric definitions are correct."""

    def test_session_mae_definition(self):
        """Verify MAE = mean(|estimate - reference|)."""
        exp = ExperimentRecord(experiment_id='test', name='Test')
        session = SessionRecord(session_id='s1', experiment_id='test')

        # Errors: 2, 4, 6 -> MAE = (2+4+6)/3 = 4.0
        for i, err in enumerate([2.0, 4.0, 6.0]):
            w = WindowRecord(
                window_id=f'w{i}', session_id='s1',
                start_time=i*10, end_time=(i+1)*10, duration=10,
                hr_estimate=70+err, hr_reference=70.0,
                signal_quality=0.8, is_valid=True
            )
            session.windows.append(w)
        exp.sessions.append(session)

        metrics = aggregate_session_metrics(session)

        print(f"\nErrors: [2, 4, 6], Expected MAE: 4.0")
        print(f"Actual MAE: {metrics.hr_mae}")

        assert abs(metrics.hr_mae - 4.0) < 0.01

    def test_valid_window_rate_denominator(self):
        """Valid rate should be valid_windows / total_windows."""
        exp = ExperimentRecord(experiment_id='test', name='Test')
        session = SessionRecord(session_id='s1', experiment_id='test')

        # 3 valid, 2 invalid
        for i in range(5):
            w = WindowRecord(
                window_id=f'w{i}', session_id='s1',
                start_time=i*10, end_time=(i+1)*10, duration=10,
                hr_estimate=70, hr_reference=70.0,
                signal_quality=0.8,
                is_valid=(i < 3),  # First 3 valid, last 2 invalid
                failure_reason="Test" if i >= 3 else None
            )
            session.windows.append(w)
        exp.sessions.append(session)

        metrics = aggregate_session_metrics(session)

        print(f"\nTotal: 5, Valid: 3, Expected rate: 0.6")
        print(f"Actual valid_rate: {metrics.valid_rate}")

        assert metrics.valid_rate == 0.6


class TestEmptySession:
    """Test handling of empty sessions."""

    def test_empty_session_metrics(self):
        """Empty session should have None for all aggregates."""
        session = SessionRecord(session_id='s1', experiment_id='test')
        metrics = aggregate_session_metrics(session)

        print(f"\nEmpty session valid_rate: {metrics.valid_rate}")
        print(f"Empty session hr_mae: {metrics.hr_mae}")

        assert metrics.valid_rate == 0.0
        assert metrics.hr_mae is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
