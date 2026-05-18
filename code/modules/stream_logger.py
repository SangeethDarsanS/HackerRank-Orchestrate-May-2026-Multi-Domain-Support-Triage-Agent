"""
Streaming stage logger.

Provides real-time, per-stage visibility into ticket processing.

Log format (each line):
    <timestamp> | Ticket <id> | <stage> | <status> | <latency_ms> ms

Destinations:
    - Console  (sys.stderr, so it doesn't interfere with Rich progress on stdout)
    - File     (appended to a dedicated stream log, default: support_tickets/stream.log)

Usage:
    sl = StreamLogger(log_file=Path("support_tickets/stream.log"), to_console=True)
    sl.log(ticket_id=1, stage="retrieval",    status="complete",  latency_ms=45.2)
    sl.log(ticket_id=1, stage="reranker",     status="complete",  latency_ms=310.0)
    sl.log(ticket_id=1, stage="guardrails",   status="safe",      latency_ms=1.1)
    sl.log(ticket_id=1, stage="decision",     status="escalated", latency_ms=0.1)
"""

import sys
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# IST = UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))


def _ts() -> str:
    return datetime.now(tz=_IST).strftime("%Y-%m-%d %H:%M:%S")


class _FileHandler(logging.Handler):
    """
    Buffered append-to-file handler.

    Lines are accumulated in memory and written to disk in a single
    syscall once the buffer reaches FLUSH_EVERY entries.  This removes
    per-log-line open/close overhead when many tickets are processed.

    Thread-safe: a Lock serialises buffer access so the handler can be
    shared across worker threads (Python logging routes to it from any
    thread).
    """

    FLUSH_EVERY = 50  # flush after accumulating this many log lines

    def __init__(self, path: Path):
        super().__init__()
        self._path = path
        self._buf: list = []
        self._buf_lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with self._buf_lock:
                self._buf.append(line)
                if len(self._buf) >= self.FLUSH_EVERY:
                    self._flush_locked()
        except Exception:
            self.handleError(record)

    def _flush_locked(self) -> None:
        """Write buffer to disk. Must be called with self._buf_lock held."""
        if not self._buf:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write("\n".join(self._buf) + "\n")
            self._buf.clear()
        except Exception:
            pass

    def flush(self) -> None:
        with self._buf_lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()
        super().close()


class StreamLogger:
    """
    Real-time per-stage logger.

    Parameters
    ----------
    log_file   : Path to the stream log file (parent dirs created automatically).
                 Pass None to disable file output.
    to_console : Whether to also echo to stderr (default True).
    """

    _FMT = "%(message)s"

    def __init__(
        self,
        log_file: Optional[Path] = None,
        to_console: bool = True,
    ):
        self._logger = logging.getLogger("stream_logger")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False   # don't bubble to root logger

        # Prevent duplicate handlers if re-instantiated
        self._logger.handlers.clear()

        formatter = logging.Formatter(self._FMT)

        if to_console:
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(formatter)
            self._logger.addHandler(sh)

        if log_file is not None:
            fh = _FileHandler(log_file)
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        ticket_id: int,
        stage: str,
        status: str,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Emit a single stage-log entry.

        Parameters
        ----------
        ticket_id  : Sequential ticket number (1-based).
        stage      : Pipeline stage name, e.g. "retrieval", "reranker".
        status     : Short outcome string, e.g. "complete", "safe", "escalated".
        latency_ms : Time taken for this stage (milliseconds).
        """
        msg = (
            f"{_ts()} | Ticket {ticket_id} | {stage:<20} | "
            f"{status:<15} | {latency_ms:.1f} ms"
        )
        self._logger.info(msg)

    def log_start(self, ticket_id: int) -> None:
        """Emit a ticket-start separator."""
        self._logger.info(
            f"{_ts()} | Ticket {ticket_id} | {'START':<20} | {'':<15} |"
        )

    def log_end(self, ticket_id: int, total_ms: float, action: str) -> None:
        """Emit a ticket-end summary line."""
        self._logger.info(
            f"{_ts()} | Ticket {ticket_id} | {'END':<20} | "
            f"{action:<15} | {total_ms:.1f} ms"
        )

    def flush(self) -> None:
        """Flush all buffered log entries to disk immediately."""
        for handler in self._logger.handlers:
            handler.flush()
