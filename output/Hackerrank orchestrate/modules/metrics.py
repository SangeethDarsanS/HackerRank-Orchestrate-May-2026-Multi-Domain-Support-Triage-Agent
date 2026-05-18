"""
Observability metrics tracker.

Tracks per-run and per-ticket metrics:
  ticket_count       — total tickets processed
  escalation_count   — how many were escalated
  error_count        — retrieval / processing errors
  total_latency_ms   — cumulative processing time
  average_latency_ms — mean per-ticket latency
  peak_memory_mb     — peak RSS (requires psutil; 0 if unavailable)

Per-ticket log format (emitted via Python logging at INFO level)
---------------------------------------------------------------
  timestamp | ticket_id=N | intent=X | domain=X | risk=X
            | conf=X.XXXX | action=X | latency=X.Xms
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

# IST = UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))


def _now_iso() -> str:
    return datetime.now(tz=_IST).isoformat(timespec="seconds")


def _rss_mb() -> float:
    """Current process RSS in MB. Returns 0.0 if psutil is not installed."""
    try:
        import psutil  # type: ignore
        import os
        return psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    except Exception:
        return 0.0


@dataclass
class TicketMetric:
    timestamp:   str
    ticket_id:   int
    intent:      str    # request_type
    domain:      str
    risk_level:  str
    confidence:  float
    action:      str    # replied | escalated
    latency_ms:  float


@dataclass
class RunMetrics:
    ticket_count:     int   = 0
    escalation_count: int   = 0
    error_count:      int   = 0
    total_latency_ms: float = 0.0
    peak_memory_mb:   float = 0.0
    tickets: List[TicketMetric] = field(default_factory=list)

    # ----- derived -----

    @property
    def average_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.ticket_count, 1)

    @property
    def escalation_rate(self) -> float:
        return self.escalation_count / max(self.ticket_count, 1)

    @property
    def avg_confidence(self) -> float:
        if not self.tickets:
            return 0.0
        return sum(t.confidence for t in self.tickets) / len(self.tickets)

    def summary(self) -> dict:
        return {
            "ticket_count":      self.ticket_count,
            "escalation_count":  self.escalation_count,
            "error_count":       self.error_count,
            "escalation_rate":   round(self.escalation_rate, 4),
            "avg_confidence":    round(self.avg_confidence, 4),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "total_latency_ms":  round(self.total_latency_ms, 2),
            "peak_memory_mb":    round(self.peak_memory_mb, 2),
        }


class MetricsTracker:
    """
    Collects and logs observability metrics across a single pipeline run.

    Usage
    -----
    tracker = MetricsTracker()
    ...
    tracker.record_ticket(ticket_id=1, request_type="bug", ...)
    ...
    summary = tracker.snapshot()
    """

    def __init__(self):
        self.run = RunMetrics()
        self._start = time.perf_counter()

    # ------------------------------------------------------------------
    # Memory helper
    # ------------------------------------------------------------------

    def get_memory_usage_mb(self) -> Optional[float]:
        """
        Return current process RSS memory usage in MB.

        Returns None (and logs a warning) if psutil is unavailable,
        so callers can distinguish "not installed" from a genuine 0 MB reading.
        """
        try:
            import psutil  # type: ignore
            import os
            process = psutil.Process(os.getpid())
            memory_bytes = process.memory_info().rss
            memory_mb = memory_bytes / (1024 * 1024)
            return round(memory_mb, 2)
        except Exception as exc:
            logger.warning("[metrics] get_memory_usage_mb unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_ticket(
        self,
        ticket_id: int,
        request_type: str,
        domain: str,
        risk_level: str,
        confidence: float,
        action: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record metrics for a single processed ticket."""
        self.run.ticket_count += 1
        self.run.total_latency_ms += latency_ms
        if action == "escalated":
            self.run.escalation_count += 1
        if error:
            self.run.error_count += 1

        mem = _rss_mb()
        if mem > self.run.peak_memory_mb:
            self.run.peak_memory_mb = mem

        metric = TicketMetric(
            timestamp=_now_iso(),
            ticket_id=ticket_id,
            intent=request_type,
            domain=domain,
            risk_level=risk_level,
            confidence=round(confidence, 4),
            action=action,
            latency_ms=round(latency_ms, 2),
        )
        self.run.tickets.append(metric)

        logger.info(
            "[metrics] %s | ticket_id=%d | intent=%s | domain=%s | "
            "risk=%s | conf=%.4f | action=%s | latency=%.1fms",
            metric.timestamp,
            ticket_id,
            request_type,
            domain,
            risk_level,
            confidence,
            action,
            latency_ms,
        )

    def snapshot(self) -> dict:
        """Return current run metrics including elapsed wall-clock time."""
        elapsed = (time.perf_counter() - self._start) * 1000
        snap = self.run.summary()
        snap["elapsed_ms"] = round(elapsed, 2)
        return snap
