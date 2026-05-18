"""
tests/test_failure_reporter.py — Unit tests for FailureReporter.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.failure_reporter import (
    FailureReporter,
    FailureEvent,
    ERROR_RETRIEVAL,
    ERROR_CLASSIFICATION,
    ERROR_RERANKER,
    ERROR_GUARDRAIL,
    ERROR_TIMEOUT,
    ERROR_EXCEPTION,
    ERROR_POLICY,
)


class TestErrorTypeConstants:

    def test_error_type_constants_exist(self):
        assert ERROR_RETRIEVAL == "retrieval_error"
        assert ERROR_CLASSIFICATION == "classification_error"
        assert ERROR_RERANKER == "reranker_error"
        assert ERROR_GUARDRAIL == "guardrail_error"
        assert ERROR_TIMEOUT == "timeout"
        assert ERROR_EXCEPTION == "exception"
        assert ERROR_POLICY == "policy_error"


class TestFailureEvent:

    def test_to_dict_keys(self):
        ev = FailureEvent(
            ticket_id=1,
            error_type=ERROR_RETRIEVAL,
            error_message="test error",
            action_taken="escalated",
        )
        d = ev.to_dict()
        assert "ticket_id" in d
        assert "error_type" in d
        assert "error_message" in d
        assert "action_taken" in d
        assert "timestamp" in d

    def test_timestamp_not_empty(self):
        ev = FailureEvent(1, ERROR_EXCEPTION, "msg", "escalated")
        assert ev.timestamp != ""


class TestFailureReporter:

    def test_initially_no_failures(self):
        rep = FailureReporter()
        assert rep.has_failures() is False
        assert rep.count() == 0

    def test_report_adds_event(self):
        rep = FailureReporter()
        rep.report(1, ERROR_RETRIEVAL, "connection refused", "escalated")
        assert rep.has_failures() is True
        assert rep.count() == 1

    def test_report_multiple_events(self):
        rep = FailureReporter()
        rep.report(1, ERROR_RETRIEVAL, "err1", "escalated")
        rep.report(2, ERROR_TIMEOUT, "timed out", "escalated")
        rep.report(3, ERROR_EXCEPTION, "some exc", "skipped")
        assert rep.count() == 3

    def test_events_returns_copy(self):
        rep = FailureReporter()
        rep.report(1, ERROR_RETRIEVAL, "err", "escalated")
        events = rep.events()
        assert len(events) == 1
        # Modifying the returned list doesn't affect internal state
        events.clear()
        assert rep.count() == 1

    def test_by_type_aggregation(self):
        rep = FailureReporter()
        rep.report(1, ERROR_RETRIEVAL, "e1", "escalated")
        rep.report(2, ERROR_RETRIEVAL, "e2", "escalated")
        rep.report(3, ERROR_TIMEOUT, "e3", "escalated")
        counts = rep.by_type()
        assert counts[ERROR_RETRIEVAL] == 2
        assert counts[ERROR_TIMEOUT] == 1

    def test_by_type_empty(self):
        rep = FailureReporter()
        assert rep.by_type() == {}

    def test_error_message_capped_at_500(self):
        rep = FailureReporter()
        long_msg = "x" * 1000
        rep.report(1, ERROR_EXCEPTION, long_msg, "escalated")
        ev = rep.events()[0]
        assert len(ev.error_message) <= 500

    def test_report_never_raises(self):
        """report() must not raise even with bad input."""
        rep = FailureReporter()
        # Should not raise:
        rep.report(None, None, None, None)  # type: ignore

    def test_default_action_taken(self):
        rep = FailureReporter()
        rep.report(1, ERROR_RETRIEVAL, "err")
        ev = rep.events()[0]
        assert ev.action_taken == "escalated"

    def test_save_creates_file(self):
        rep = FailureReporter()
        rep.report(1, ERROR_RETRIEVAL, "err", "escalated")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "failure_report.json"
            rep.save(path)
            assert path.exists()

    def test_save_json_structure(self):
        rep = FailureReporter()
        rep.report(1, ERROR_RETRIEVAL, "connection refused", "escalated")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "failure_report.json"
            rep.save(path)
            data = json.loads(path.read_text())
            assert "total_failures" in data
            assert "failures_by_type" in data
            assert "events" in data
            assert data["total_failures"] == 1

    def test_save_with_no_failures(self):
        """save() must write a valid JSON even with zero failures."""
        rep = FailureReporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "failure_report.json"
            rep.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["total_failures"] == 0
            assert data["events"] == []

    def test_save_event_fields(self):
        rep = FailureReporter()
        rep.report(42, ERROR_POLICY, "no citation", "escalated")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "failure_report.json"
            rep.save(path)
            data = json.loads(path.read_text())
            ev = data["events"][0]
            assert ev["ticket_id"] == 42
            assert ev["error_type"] == ERROR_POLICY
            assert ev["action_taken"] == "escalated"

    def test_save_creates_parent_dir(self):
        rep = FailureReporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "failure_report.json"
            rep.save(path)
            assert path.exists()
