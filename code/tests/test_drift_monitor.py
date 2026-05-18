"""
tests/test_drift_monitor.py — Unit tests for DriftMonitor.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.drift_monitor import DriftMonitor, _total_variation_distance


# ---------------------------------------------------------------------------
# TVD helper tests
# ---------------------------------------------------------------------------

class TestTotalVariationDistance:

    def test_identical_distributions(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.5, "b": 0.5}
        assert _total_variation_distance(p, q) == 0.0

    def test_completely_different_distributions(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        assert abs(_total_variation_distance(p, q) - 1.0) < 1e-9

    def test_empty_dicts(self):
        assert _total_variation_distance({}, {}) == 0.0

    def test_one_empty_dict(self):
        p = {"a": 1.0}
        q = {}
        # All mass in p not in q → TVD = 0.5 * 1.0 = 0.5 (normalized)
        result = _total_variation_distance(p, q)
        assert 0.0 <= result <= 1.0

    def test_symmetric(self):
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.4, "b": 0.6}
        assert abs(_total_variation_distance(p, q) - _total_variation_distance(q, p)) < 1e-9

    def test_bounded_zero_to_one(self):
        p = {"a": 0.6, "b": 0.4}
        q = {"b": 0.3, "c": 0.7}
        result = _total_variation_distance(p, q)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# DriftMonitor tests
# ---------------------------------------------------------------------------

class TestDriftMonitor:

    def test_compute_drift_empty(self):
        dm = DriftMonitor()
        result = dm.compute_drift()
        assert result["n_tickets"] == 0
        assert result["drifted"] is False
        assert result["alerts"] == []

    def test_record_increments_counters(self):
        dm = DriftMonitor()
        dm.record("billing_payment", "low", "replied")
        dm.record("technical_bug", "medium", "escalated")
        result = dm.compute_drift()
        assert result["n_tickets"] == 2

    def test_no_drift_with_matching_baseline(self):
        """Feeding the exact baseline proportions should yield near-zero TVD."""
        baseline = {
            "intents": {"billing_payment": 1.0},
            "risk_levels": {"low": 1.0},
            "escalation_rate": 0.0,
        }
        dm = DriftMonitor(baseline=baseline, threshold=0.15)
        for _ in range(10):
            dm.record("billing_payment", "low", "replied")
        result = dm.compute_drift()
        assert result["intent_drift"] < 0.01
        assert result["risk_drift"] < 0.01

    def test_alerts_on_high_drift(self):
        """A single intent when baseline has many intents → drift alert."""
        dm = DriftMonitor(threshold=0.10)
        for _ in range(20):
            dm.record("account_access", "high", "escalated")
        result = dm.compute_drift()
        assert result["drifted"] is True
        assert len(result["alerts"]) > 0

    def test_escalation_drift_calculation(self):
        dm = DriftMonitor()
        # All tickets escalated → escalation_rate = 1.0 vs baseline 0.40
        for _ in range(10):
            dm.record("billing_payment", "low", "escalated")
        result = dm.compute_drift()
        assert abs(result["escalation_rate"] - 1.0) < 0.01
        assert result["escalation_drift"] > 0.4

    def test_current_distributions_structure(self):
        dm = DriftMonitor()
        dm.record("billing_payment", "low", "replied")
        dm.record("technical_bug", "medium", "escalated")
        dist = dm.current_distributions()
        assert "intents" in dist
        assert "risk_levels" in dist
        assert "actions" in dist

    def test_current_distributions_sums_to_one(self):
        dm = DriftMonitor()
        dm.record("billing_payment", "low", "replied")
        dm.record("technical_bug", "medium", "replied")
        dm.record("account_access", "high", "escalated")
        dist = dm.current_distributions()
        # Values are rounded to 4 decimal places, so allow tolerance of 1e-3
        assert abs(sum(dist["intents"].values()) - 1.0) < 1e-3
        assert abs(sum(dist["risk_levels"].values()) - 1.0) < 1e-3
        assert abs(sum(dist["actions"].values()) - 1.0) < 1e-3

    def test_save_report_creates_file(self):
        dm = DriftMonitor()
        dm.record("billing_payment", "low", "replied")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drift_report.json"
            dm.save_report(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "n_tickets" in data
            assert "drift_detected" in data

    def test_save_report_json_structure(self):
        dm = DriftMonitor()
        dm.record("billing_payment", "low", "replied")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drift_report.json"
            dm.save_report(path)
            data = json.loads(path.read_text())
            for key in ("timestamp", "n_tickets", "threshold",
                        "drift_detected", "intent_tvd", "risk_tvd",
                        "escalation_drift", "alerts", "current_distributions"):
                assert key in data, f"Missing key: {key}"

    def test_custom_threshold(self):
        """With threshold=0, any non-baseline data should trigger drift."""
        dm = DriftMonitor(threshold=0.0)
        dm.record("billing_payment", "low", "replied")
        result = dm.compute_drift()
        # At threshold=0, even tiny divergence is an alert
        assert result["drifted"] is True

    def test_drift_fields_bounded(self):
        dm = DriftMonitor()
        for _ in range(5):
            dm.record("billing_payment", "low", "escalated")
        result = dm.compute_drift()
        assert 0.0 <= result["intent_drift"] <= 1.0
        assert 0.0 <= result["risk_drift"] <= 1.0
        assert 0.0 <= result["escalation_drift"] <= 1.0
