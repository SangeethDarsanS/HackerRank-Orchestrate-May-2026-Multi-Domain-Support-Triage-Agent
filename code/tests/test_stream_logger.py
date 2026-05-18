"""
Unit tests for modules/stream_logger.py.
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from modules.stream_logger import StreamLogger


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def test_log_creates_file():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "stream.log"
        sl = StreamLogger(log_file=log_path, to_console=False)
        sl.log(1, "retrieval", "complete", 42.5)
        assert log_path.exists()


def test_log_writes_expected_fields():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "stream.log"
        sl = StreamLogger(log_file=log_path, to_console=False)
        sl.log(7, "guardrails", "safe", 3.1)
        content = log_path.read_text(encoding="utf-8")
        assert "Ticket 7" in content
        assert "guardrails" in content
        assert "safe" in content
        assert "3.1 ms" in content


def test_multiple_log_entries_appended():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "stream.log"
        sl = StreamLogger(log_file=log_path, to_console=False)
        for i in range(5):
            sl.log(i + 1, "retrieval", "complete", float(i * 10))
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5


def test_log_start_writes_entry():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "stream.log"
        sl = StreamLogger(log_file=log_path, to_console=False)
        sl.log_start(42)
        content = log_path.read_text(encoding="utf-8")
        assert "Ticket 42" in content
        assert "START" in content


def test_log_end_writes_action():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "stream.log"
        sl = StreamLogger(log_file=log_path, to_console=False)
        sl.log_end(10, 312.5, "escalated")
        content = log_path.read_text(encoding="utf-8")
        assert "Ticket 10" in content
        assert "END" in content
        assert "escalated" in content


# ---------------------------------------------------------------------------
# No-file mode
# ---------------------------------------------------------------------------

def test_no_file_does_not_crash():
    sl = StreamLogger(log_file=None, to_console=False)
    sl.log(1, "decision", "replied", 1.0)
    sl.log_start(2)
    sl.log_end(2, 100.0, "replied")


# ---------------------------------------------------------------------------
# Parent directory auto-creation
# ---------------------------------------------------------------------------

def test_creates_parent_directory():
    with tempfile.TemporaryDirectory() as td:
        nested = Path(td) / "a" / "b" / "stream.log"
        sl = StreamLogger(log_file=nested, to_console=False)
        sl.log(1, "test", "ok", 0.0)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Multiple instantiations don't duplicate handlers
# ---------------------------------------------------------------------------

def test_no_handler_duplication():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "stream.log"
        for _ in range(3):
            sl = StreamLogger(log_file=log_path, to_console=False)
        sl.log(1, "retrieval", "complete", 5.0)
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        # Should be exactly 1 line (no duplicate handlers writing the same entry)
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Log format validation
# ---------------------------------------------------------------------------

def test_format_has_pipe_separators():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "stream.log"
        sl = StreamLogger(log_file=log_path, to_console=False)
        sl.log(5, "reranker", "score=0.85", 310.0)
        content = log_path.read_text(encoding="utf-8")
        assert "|" in content


def test_latency_zero_is_written():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "stream.log"
        sl = StreamLogger(log_file=log_path, to_console=False)
        sl.log(1, "classification", "bug", 0.0)
        content = log_path.read_text(encoding="utf-8")
        assert "0.0 ms" in content
