"""
modules/failure_reporter.py — Structured failure capture and reporting.

Records any exception, retrieval failure, classification failure, or timeout
that occurs during ticket processing.  Never raises; never blocks the main
pipeline.

Output: failure_report.json
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_iso() -> str:
    return datetime.now(tz=_IST).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Error type constants
# ---------------------------------------------------------------------------

ERROR_RETRIEVAL        = "retrieval_error"
ERROR_CLASSIFICATION   = "classification_error"
ERROR_RERANKER         = "reranker_error"
ERROR_GUARDRAIL        = "guardrail_error"
ERROR_TIMEOUT          = "timeout"
ERROR_EXCEPTION        = "exception"
ERROR_POLICY           = "policy_error"


@dataclass
class FailureEvent:
    """One recorded failure event."""
    ticket_id:     int
    error_type:    str      # one of the ERROR_* constants
    error_message: str
    action_taken:  str      # e.g. "escalated", "fallback", "skipped"
    timestamp:     str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


class FailureReporter:
    """
    Collects structured failure events during a pipeline run.

    Usage
    -----
    reporter = FailureReporter()

    try:
        result = risky_operation()
    except Exception as exc:
        reporter.report(ticket_id=5, error_type=ERROR_RETRIEVAL,
                        error_message=str(exc), action_taken="escalated")

    reporter.save(Path("support_tickets/failure_report.json"))
    """

    def __init__(self) -> None:
        self._events: List[FailureEvent] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def report(
        self,
        ticket_id:     int,
        error_type:    str,
        error_message: str,
        action_taken:  str = "escalated",
    ) -> None:
        """
        Record a failure event.  Never raises.

        Parameters
        ----------
        ticket_id     : ID of the ticket that caused the failure.
        error_type    : One of the ERROR_* constants (or any string label).
        error_message : str(exception) or brief description.
        action_taken  : What the pipeline did in response (default: "escalated").
        """
        try:
            event = FailureEvent(
                ticket_id=ticket_id,
                error_type=error_type,
                error_message=str(error_message)[:500],   # cap length
                action_taken=action_taken,
            )
            self._events.append(event)
            logger.warning(
                "[failure] ticket=%d type=%s action=%s — %s",
                ticket_id, error_type, action_taken,
                str(error_message)[:120],
            )
        except Exception as exc:
            logger.error("[failure_reporter] failed to record event: %s", exc)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def has_failures(self) -> bool:
        """True if at least one failure has been recorded."""
        return bool(self._events)

    def count(self) -> int:
        return len(self._events)

    def events(self) -> List[FailureEvent]:
        """Return a copy of all recorded events."""
        return list(self._events)

    def by_type(self) -> dict:
        """Return {error_type: count} mapping."""
        counts: dict = {}
        for ev in self._events:
            counts[ev.error_type] = counts.get(ev.error_type, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Write failure_report.json to `path`.
        Always writes (even if no failures) so dashboards can rely on the file.
        """
        try:
            report = {
                "total_failures": len(self._events),
                "failures_by_type": self.by_type(),
                "events": [ev.to_dict() for ev in self._events],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("[failure_reporter] saved %d events to %s",
                        len(self._events), path)
        except Exception as exc:
            logger.error("[failure_reporter] could not save report: %s", exc)
