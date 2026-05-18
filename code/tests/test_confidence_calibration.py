"""
tests/test_confidence_calibration.py — Unit tests for ConfidenceCalibrator.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.calibration import ConfidenceCalibrator


class TestConfidenceCalibrator:

    def test_default_params(self):
        cal = ConfidenceCalibrator()
        p = cal.params()
        assert p["method"] == "sigmoid"
        assert p["a"] == -4.0
        assert p["b"] == 2.0
        assert p["fitted"] is False

    def test_calibrate_output_bounded(self):
        cal = ConfidenceCalibrator()
        for raw in [0.0, 0.1, 0.3, 0.5, 0.65, 0.8, 0.9, 1.0]:
            out = cal.calibrate(raw)
            assert 0.0 <= out <= 1.0, f"Out of range for raw={raw}: {out}"

    def test_calibrate_clamps_below_zero(self):
        cal = ConfidenceCalibrator()
        # Passing a value just below 0 should be clamped to 0
        out = cal.calibrate(-0.1)
        assert 0.0 <= out <= 1.0

    def test_calibrate_clamps_above_one(self):
        cal = ConfidenceCalibrator()
        out = cal.calibrate(1.5)
        assert 0.0 <= out <= 1.0

    def test_calibrate_monotone(self):
        """Calibration must be monotonically increasing."""
        cal = ConfidenceCalibrator()
        scores = [i / 20 for i in range(21)]
        calibrated = [cal.calibrate(s) for s in scores]
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1], (
                f"Not monotone at index {i}: {calibrated[i]} > {calibrated[i+1]}"
            )

    def test_calibrate_identity_method(self):
        cal = ConfidenceCalibrator(method="identity")
        for raw in [0.0, 0.3, 0.65, 0.9, 1.0]:
            assert cal.calibrate(raw) == raw

    def test_calibrate_returns_float(self):
        cal = ConfidenceCalibrator()
        out = cal.calibrate(0.75)
        assert isinstance(out, float)

    def test_calibrate_precision_6_decimals(self):
        cal = ConfidenceCalibrator()
        out = cal.calibrate(0.75)
        # Should be rounded to 6 decimal places
        assert out == round(out, 6)

    def test_is_fitted_false_initially(self):
        cal = ConfidenceCalibrator()
        assert cal.is_fitted() is False

    def test_fit_requires_equal_length(self):
        cal = ConfidenceCalibrator()
        with pytest.raises(ValueError):
            cal.fit([0.5, 0.7], [1])

    def test_fit_skips_on_too_few_samples(self):
        """fit() with < 4 samples should not raise and leave defaults unchanged."""
        cal = ConfidenceCalibrator()
        cal.fit([0.5, 0.7, 0.3], [1, 0, 1])
        # Still uses defaults because < 4 samples
        assert cal.is_fitted() is False

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sklearn"),
        reason="sklearn not installed",
    )
    def test_fit_with_sufficient_data(self):
        cal = ConfidenceCalibrator()
        raw_scores = [0.2, 0.4, 0.6, 0.8, 0.9, 0.3, 0.7, 0.5]
        labels = [0, 0, 1, 1, 1, 0, 1, 0]
        cal.fit(raw_scores, labels)
        assert cal.is_fitted() is True

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sklearn"),
        reason="sklearn not installed",
    )
    def test_fitted_calibrator_still_monotone(self):
        cal = ConfidenceCalibrator()
        raw_scores = [0.2, 0.4, 0.6, 0.8, 0.9, 0.3, 0.7, 0.5]
        labels = [0, 0, 1, 1, 1, 0, 1, 0]
        cal.fit(raw_scores, labels)
        scores = [i / 20 for i in range(21)]
        calibrated = [cal.calibrate(s) for s in scores]
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1]

    def test_params_method(self):
        cal = ConfidenceCalibrator(a=-3.0, b=1.5, method="sigmoid")
        p = cal.params()
        assert p["a"] == -3.0
        assert p["b"] == 1.5
        assert p["method"] == "sigmoid"
        assert "fitted" in p

    def test_custom_ab_params(self):
        cal = ConfidenceCalibrator(a=-2.0, b=1.0)
        out = cal.calibrate(0.5)
        expected = 1.0 / (1.0 + math.exp(-2.0 * 0.5 + 1.0))
        assert abs(out - expected) < 1e-5

    def test_overflow_handled_gracefully(self):
        """Extreme A/B values should not raise OverflowError."""
        cal = ConfidenceCalibrator(a=-1000.0, b=0.0)
        out = cal.calibrate(1.0)
        assert 0.0 <= out <= 1.0

    def test_high_confidence_score_preserved_above_threshold(self):
        """A raw score of 0.9 should produce calibrated > 0.5 (not inverted)."""
        cal = ConfidenceCalibrator()
        out_high = cal.calibrate(0.9)
        out_low = cal.calibrate(0.1)
        assert out_high > out_low
