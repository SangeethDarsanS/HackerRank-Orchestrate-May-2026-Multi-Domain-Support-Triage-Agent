"""
Unit tests for W001 fix: MetricsTracker.get_memory_usage_mb()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from modules.metrics import MetricsTracker, _rss_mb


# ---------------------------------------------------------------------------
# get_memory_usage_mb — normal operation
# ---------------------------------------------------------------------------

def test_get_memory_usage_mb_returns_number():
    tracker = MetricsTracker()
    result = tracker.get_memory_usage_mb()
    # psutil is installed, so we expect a real float
    assert result is not None
    assert isinstance(result, float)


def test_get_memory_usage_mb_positive():
    tracker = MetricsTracker()
    result = tracker.get_memory_usage_mb()
    if result is not None:
        assert result > 0.0, "RSS should be positive for a live process"


def test_get_memory_usage_mb_reasonable_range():
    """Sanity check: process should use between 1 MB and 4 GB."""
    tracker = MetricsTracker()
    result = tracker.get_memory_usage_mb()
    if result is not None:
        assert 1.0 <= result <= 4096.0


def test_get_memory_usage_mb_two_calls_consistent():
    """Two rapid calls should return values in the same ballpark (within 200 MB)."""
    tracker = MetricsTracker()
    a = tracker.get_memory_usage_mb()
    b = tracker.get_memory_usage_mb()
    if a is not None and b is not None:
        assert abs(a - b) < 200.0


# ---------------------------------------------------------------------------
# get_memory_usage_mb — psutil unavailable (graceful fallback)
# ---------------------------------------------------------------------------

def test_get_memory_usage_mb_returns_none_when_psutil_missing(monkeypatch):
    """When psutil is not importable, returns None instead of crashing."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("mocked absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    tracker = MetricsTracker()
    result = tracker.get_memory_usage_mb()
    assert result is None


# ---------------------------------------------------------------------------
# _rss_mb helper (private, used by record_ticket)
# ---------------------------------------------------------------------------

def test_rss_mb_returns_float():
    assert isinstance(_rss_mb(), float)


def test_rss_mb_non_negative():
    assert _rss_mb() >= 0.0


def test_rss_mb_returns_zero_when_psutil_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _rss_mb() == 0.0


# ---------------------------------------------------------------------------
# peak_memory_mb is updated in record_ticket
# ---------------------------------------------------------------------------

def test_peak_memory_updated_after_record():
    tracker = MetricsTracker()
    tracker.record_ticket(
        ticket_id=1, request_type="bug", domain="hackerrank",
        risk_level="low", confidence=0.8, action="replied",
        latency_ms=100.0,
    )
    # peak_memory_mb should now be > 0 since psutil is available
    assert tracker.run.peak_memory_mb > 0.0


def test_snapshot_contains_peak_memory():
    tracker = MetricsTracker()
    tracker.record_ticket(
        ticket_id=1, request_type="product_issue", domain="claude",
        risk_level="low", confidence=0.7, action="replied",
        latency_ms=50.0,
    )
    snap = tracker.snapshot()
    assert "peak_memory_mb" in snap
    assert snap["peak_memory_mb"] > 0.0
