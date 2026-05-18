"""
Unit tests for modules/metrics.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import pytest
from modules.metrics import MetricsTracker, RunMetrics, TicketMetric


# ---------------------------------------------------------------------------
# RunMetrics derived properties
# ---------------------------------------------------------------------------

def test_escalation_rate_zero_tickets():
    m = RunMetrics()
    assert m.escalation_rate == 0.0


def test_escalation_rate_all_escalated():
    m = RunMetrics(ticket_count=5, escalation_count=5)
    assert m.escalation_rate == 1.0


def test_escalation_rate_partial():
    m = RunMetrics(ticket_count=10, escalation_count=3)
    assert abs(m.escalation_rate - 0.3) < 1e-9


def test_average_latency_no_tickets():
    m = RunMetrics()
    assert m.average_latency_ms == 0.0


def test_average_latency_single_ticket():
    m = RunMetrics(ticket_count=1, total_latency_ms=42.0)
    assert m.average_latency_ms == 42.0


def test_avg_confidence_empty():
    m = RunMetrics()
    assert m.avg_confidence == 0.0


def test_avg_confidence_with_tickets():
    m = RunMetrics()
    m.tickets = [
        TicketMetric("ts", 1, "bug", "hackerrank", "low", 0.8, "replied", 10.0),
        TicketMetric("ts", 2, "bug", "hackerrank", "low", 0.6, "replied", 10.0),
    ]
    assert abs(m.avg_confidence - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# MetricsTracker.record_ticket
# ---------------------------------------------------------------------------

def test_record_ticket_increments_count():
    t = MetricsTracker()
    t.record_ticket(1, "bug", "hackerrank", "low", 0.8, "replied", 15.0)
    assert t.run.ticket_count == 1


def test_record_escalation_increments_escalation_count():
    t = MetricsTracker()
    t.record_ticket(1, "product_issue", "visa", "high", 0.3, "escalated", 10.0)
    assert t.run.escalation_count == 1


def test_record_reply_does_not_increment_escalation():
    t = MetricsTracker()
    t.record_ticket(1, "bug", "claude", "low", 0.9, "replied", 5.0)
    assert t.run.escalation_count == 0


def test_record_error_increments_error_count():
    t = MetricsTracker()
    t.record_ticket(1, "bug", "hackerrank", "low", 0.0, "escalated", 1.0, error=True)
    assert t.run.error_count == 1


def test_record_no_error_by_default():
    t = MetricsTracker()
    t.record_ticket(1, "bug", "hackerrank", "low", 0.9, "replied", 5.0)
    assert t.run.error_count == 0


def test_multiple_tickets_accumulate():
    t = MetricsTracker()
    for i in range(5):
        t.record_ticket(i + 1, "bug", "hackerrank", "low", 0.8, "replied", 20.0)
    assert t.run.ticket_count == 5
    assert abs(t.run.total_latency_ms - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# MetricsTracker.snapshot
# ---------------------------------------------------------------------------

def test_snapshot_contains_required_keys():
    t = MetricsTracker()
    snap = t.snapshot()
    for key in ("ticket_count", "escalation_count", "error_count",
                "average_latency_ms", "escalation_rate", "elapsed_ms"):
        assert key in snap, f"Missing key: {key}"


def test_snapshot_elapsed_increases():
    t = MetricsTracker()
    snap1 = t.snapshot()
    time.sleep(0.01)
    snap2 = t.snapshot()
    assert snap2["elapsed_ms"] > snap1["elapsed_ms"]


def test_summary_escalation_rate_computed():
    t = MetricsTracker()
    t.record_ticket(1, "bug", "hackerrank", "low", 0.8, "escalated", 10.0)
    t.record_ticket(2, "bug", "hackerrank", "low", 0.8, "replied", 10.0)
    snap = t.snapshot()
    assert abs(snap["escalation_rate"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_peak_memory_non_negative():
    t = MetricsTracker()
    t.record_ticket(1, "bug", "hackerrank", "low", 0.8, "replied", 5.0)
    assert t.run.peak_memory_mb >= 0.0


def test_zero_confidence_allowed():
    t = MetricsTracker()
    t.record_ticket(1, "invalid", "unknown", "high", 0.0, "escalated", 1.0)
    assert t.run.tickets[0].confidence == 0.0
