"""
Logging module.

Two logs are maintained:

1. AGENTS.md log  ($HOME/hackerrank_orchestrate/log.txt)
   Written to by the Claude Code session (already started above).
   The triage pipeline appends a final summary entry when done.

2. Triage operation log  (support_tickets/log.txt)
   Written per-ticket with timestamp, classification, risk, decision,
   retrieved docs, and response.
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from .config import AGENTS_LOG_FILE, TRIAGE_LOG_FILE, AGENTS_LOG_DIR


# IST = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))


def _now_iso() -> str:
    return datetime.now(tz=IST).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Triage operation logger (support_tickets/log.txt)
# ---------------------------------------------------------------------------

class TriageLogger:
    """
    Append-only structured logger for ticket processing.

    Write buffering
    ---------------
    Entries are accumulated in an in-memory buffer and flushed to disk
    every _FLUSH_EVERY tickets (default 10) and unconditionally at the
    end of the run (log_summary).  This eliminates one open/close/write
    syscall per ticket, reducing I/O overhead at high throughput.
    """

    _FLUSH_EVERY = 10  # flush buffer to disk after this many ticket entries

    def __init__(self, log_path: Path = TRIAGE_LOG_FILE):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._buf: list = []
        self._ticket_count = 0
        # Write header once if file is new (immediate, bypasses buffer)
        if not self.log_path.exists():
            self._write_direct(
                "# Support Triage Agent — Operation Log\n"
                f"# Run started: {_now_iso()}\n"
                "# ============================================================\n\n"
            )

    def log_ticket(
        self,
        ticket_id: int,
        issue: str,
        subject: str,
        company: str,
        domain: str,
        request_type: str,
        risk_level: str,
        risk_reasons: list,
        action: str,
        confidence: float,
        product_area: str,
        retrieved_doc_ids: list,
        response: str,
        justification: str,
        latency_ms: float,
    ) -> None:
        short_response = (response[:200] + "…") if len(response) > 200 else response
        entry = (
            f"## [{_now_iso()}] Ticket #{ticket_id}\n\n"
            f"Subject: {subject or '(none)'}\n"
            f"Company: {company}\n"
            f"Issue: {issue[:150]}{'…' if len(issue) > 150 else ''}\n\n"
            f"Classification:\n"
            f"  domain       = {domain}\n"
            f"  request_type = {request_type}\n"
            f"  product_area = {product_area}\n\n"
            f"Risk Assessment:\n"
            f"  level   = {risk_level}\n"
            f"  reasons = {'; '.join(risk_reasons) or 'none'}\n\n"
            f"Decision:\n"
            f"  action     = {action}\n"
            f"  confidence = {confidence:.4f}\n\n"
            f"Retrieved docs: {', '.join(retrieved_doc_ids) or 'none'}\n\n"
            f"Response (preview):\n{short_response}\n\n"
            f"Justification: {justification}\n\n"
            f"Latency: {latency_ms:.1f} ms\n"
            "---\n\n"
        )
        self._write(entry)
        self._ticket_count += 1
        if self._ticket_count % self._FLUSH_EVERY == 0:
            self.flush()

    def log_summary(
        self,
        total: int,
        replied: int,
        escalated: int,
        total_ms: float,
    ) -> None:
        entry = (
            f"## [{_now_iso()}] RUN COMPLETE\n\n"
            f"Total tickets : {total}\n"
            f"Replied       : {replied}\n"
            f"Escalated     : {escalated}\n"
            f"Avg latency   : {total_ms / max(total, 1):.1f} ms/ticket\n"
            f"Total time    : {total_ms:.0f} ms\n"
            "---\n\n"
        )
        self._write(entry)
        self.flush()

    def flush(self) -> None:
        """Write all buffered entries to disk and clear the buffer."""
        if not self._buf:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("".join(self._buf))
        self._buf.clear()

    def _write(self, text: str) -> None:
        """Append text to the in-memory buffer."""
        self._buf.append(text)

    def _write_direct(self, text: str) -> None:
        """Write text directly to disk, bypassing the buffer (init only)."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text)


# ---------------------------------------------------------------------------
# AGENTS.md session logger
# ---------------------------------------------------------------------------

def agents_log_session_start(repo_root: str, branch: str = "main") -> None:
    """Append a SESSION START entry to the AGENTS.md log."""
    AGENTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = _now_iso()
    # Time remaining until 2026-05-02T11:00:00+05:30
    deadline = datetime(2026, 5, 2, 11, 0, 0, tzinfo=IST)
    delta = deadline - datetime.now(tz=IST)
    if delta.total_seconds() > 0:
        total_minutes = int(delta.total_seconds() // 60)
        days = total_minutes // (60 * 24)
        hours = (total_minutes % (60 * 24)) // 60
        mins = total_minutes % 60
        remaining = f"{days}d {hours}h {mins}m"
    else:
        remaining = "Challenge ended"

    entry = (
        f"\n## [{now}] SESSION START — triage pipeline run\n\n"
        f"Agent: triage-pipeline\n"
        f"Repo Root: {repo_root}\n"
        f"Branch: {branch}\n"
        f"Worktree: main\n"
        f"Parent Agent: claude-sonnet-4-6\n"
        f"Language: py\n"
        f"Time Remaining: {remaining}\n"
        "---\n"
    )
    with open(AGENTS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def agents_log_run_complete(
    output_csv: str,
    triage_log: str,
    total: int,
    replied: int,
    escalated: int,
    repo_root: str,
) -> None:
    """Append a per-turn log entry after the pipeline finishes."""
    now = _now_iso()
    entry = (
        f"\n## [{now}] Triage pipeline run complete\n\n"
        f"User Prompt (verbatim, secrets redacted):\n"
        f"python main.py --tickets support_tickets/support_tickets.csv\n\n"
        f"Agent Response Summary:\n"
        f"Processed {total} tickets. Replied: {replied}, Escalated: {escalated}. "
        f"Output written to {output_csv} and {triage_log}.\n\n"
        f"Actions:\n"
        f"* Wrote {output_csv}\n"
        f"* Wrote {triage_log}\n\n"
        f"Context:\n"
        f"tool=triage-pipeline\n"
        f"branch=main\n"
        f"repo_root={repo_root}\n"
        f"worktree=main\n"
        f"parent_agent=claude-sonnet-4-6\n"
        "---\n"
    )
    with open(AGENTS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
