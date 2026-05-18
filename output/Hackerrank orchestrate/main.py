#!/usr/bin/env python3
"""
Support Triage Agent — main entry point.

Usage (normal mode):
    python main.py --tickets ../support_tickets/support_tickets.csv

Usage (backtest mode):
    python main.py --mode backtest

Outputs:
    ../support_tickets/output.csv          — predictions
    ../support_tickets/log.txt             — per-ticket operation log
    ../support_tickets/evaluation_report.json  — (backtest mode only)
"""

import sys
import os
import time
import json
import argparse
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure the code/ directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.config import (
    DATA_DIR, BASE_DIR, SUPPORT_TICKETS_DIR,
    OUTPUT_CSV, TRIAGE_LOG_FILE,
    RANDOM_SEED, TOP_K_CHUNKS,
)
from modules.loader import DocumentLoader
from modules.indexer import VectorIndex
from modules.retriever import Retriever
from modules.classifier import DomainClassifier, RequestTypeClassifier, ProductAreaResolver
from modules.risk_engine import RiskEngine
from modules.decision_engine import DecisionEngine
from modules.response_generator import ResponseGenerator
from modules.logger import TriageLogger, agents_log_session_start, agents_log_run_complete
from modules.reranker import Reranker
from modules.guardrails import Guardrails
from modules.confidence import ConfidenceScorer
from modules.metrics import MetricsTracker
from modules.evaluator import Evaluator
from modules.hybrid_retriever import HybridRetriever
from modules.intent_classifier import IntentClassifier
from modules.policy_enforcer import PolicyEnforcer
from modules.stream_logger import StreamLogger
from modules.cache import QueryCache
from modules.drift_monitor import DriftMonitor
from modules.failure_reporter import FailureReporter, ERROR_RETRIEVAL, ERROR_EXCEPTION
from modules.calibration import ConfidenceCalibrator
from modules.response_generator import build_decision_summary
import gc
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load config.yaml; return {} on any failure (all values have defaults)."""
    try:
        import yaml  # type: ignore
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except ImportError:
        pass
    except Exception as exc:
        print(f"[warn] Could not load config.yaml: {exc}", file=sys.stderr)
    return {}


# ---------------------------------------------------------------------------
# Deterministic setup
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Seed all known RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Rich terminal UI (graceful fallback if rich not installed)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    RICH = True
    console = Console(force_terminal=False, highlight=False)
except ImportError:
    RICH = False
    console = None


def cprint(msg: str, style: str = "") -> None:
    if RICH:
        console.print(msg, style=style)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Performance optimization constants
# ---------------------------------------------------------------------------
_BATCH_SIZE = 8   # tickets submitted per batch to the thread pool
_MAX_WORKERS = 4  # parallel worker threads


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent"
    )
    parser.add_argument(
        "--tickets",
        type=Path,
        default=SUPPORT_TICKETS_DIR / "support_tickets.csv",
        help="Path to input CSV (Issue, Subject, Company)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Path for output CSV",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=TRIAGE_LOG_FILE,
        help="Path for triage operation log",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild FAISS index (ignores cache)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_CHUNKS,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "backtest"],
        default="normal",
        help="normal: process tickets; backtest: run + evaluate vs ground truth",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=SUPPORT_TICKETS_DIR / "sample_support_tickets.csv",
        help="Ground-truth CSV for backtest mode",
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=SUPPORT_TICKETS_DIR / "evaluation_report.json",
        help="Output path for evaluation_report.json (backtest mode)",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run a lightweight system health check and exit",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def run_health_check(cfg: dict) -> None:
    """
    Perform a lightweight system health check without loading heavy models.
    Prints a status summary and exits with code 0 (OK) or 1 (failure).
    """
    checks = {}

    # 1. Config loaded
    checks["Config"] = "OK" if cfg else "WARN (empty config)"

    # 2. Corpus documents present
    try:
        doc_count = sum(1 for _ in DATA_DIR.rglob("*.md")) + \
                    sum(1 for _ in DATA_DIR.rglob("*.txt"))
        checks["Corpus"] = f"OK ({doc_count} files)" if doc_count > 0 else "WARN (no docs found)"
    except Exception as exc:
        checks["Corpus"] = f"FAIL ({exc})"

    # 3. Core modules importable
    try:
        from modules.loader import DocumentLoader
        from modules.indexer import VectorIndex
        from modules.retriever import Retriever
        from modules.confidence import ConfidenceScorer
        checks["Modules"] = "OK"
    except Exception as exc:
        checks["Modules"] = f"FAIL ({exc})"

    # 4. Memory (psutil)
    try:
        import psutil
        import os as _os
        rss = psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
        checks["Memory"] = f"OK ({rss:.1f} MB RSS)"
    except ImportError:
        checks["Memory"] = "WARN (psutil not installed)"
    except Exception as exc:
        checks["Memory"] = f"WARN ({exc})"

    # 5. Logging active
    try:
        import logging as _logging
        _logging.getLogger(__name__).debug("health-check")
        checks["Logging"] = "OK"
    except Exception as exc:
        checks["Logging"] = f"FAIL ({exc})"

    # 6. FAISS index cache
    index_cache = DATA_DIR.parent / "code" / ".cache" / "faiss.index"
    if not index_cache.exists():
        # Try relative to current file
        index_cache = Path(__file__).resolve().parent / ".cache" / "faiss.index"
    checks["Index Cache"] = "OK (cached)" if index_cache.exists() else "WARN (not built yet)"

    # --- Print results ---
    all_ok = all(v.startswith("OK") for v in checks.values())
    cprint("\n[bold]System Health Check[/bold]", style="")
    cprint("─" * 40, style="dim")
    for name, status in checks.items():
        icon = "✓" if status.startswith("OK") else ("!" if status.startswith("WARN") else "✗")
        style = "green" if status.startswith("OK") else ("yellow" if status.startswith("WARN") else "red")
        cprint(f"  {icon} {name:<18} {status}", style=style)
    cprint("─" * 40, style="dim")
    overall = "OK" if all_ok else ("WARN" if all(not v.startswith("FAIL") for v in checks.values()) else "FAIL")
    cprint(f"\nSystem Status: {overall}", style="bold green" if overall == "OK" else "bold yellow")
    cprint(f"Models: {'Loaded' if checks.get('Modules', '').startswith('OK') else 'Unavailable'}", style="dim")
    cprint(f"Corpus: {'Indexed' if checks.get('Corpus', '').startswith('OK') else 'Missing'}", style="dim")
    cprint(f"Memory: {'Available' if not checks.get('Memory', '').startswith('FAIL') else 'Unavailable'}\n", style="dim")

    import sys as _sys
    _sys.exit(0 if overall != "FAIL" else 1)


# ---------------------------------------------------------------------------
# Model warm-up
# ---------------------------------------------------------------------------

def preload_models(reranker_obj, intent_clf_obj) -> None:
    """
    Trigger model loading before the ticket loop to eliminate cold-start
    latency spikes on the first ticket.

    Both objects are already initialised by run_pipeline(); this function
    only forces any lazy-loaded weights into memory.
    """
    cprint("[bold]Warming up models...[/bold]", style="")
    # Cross-encoder: lazy-loaded on first rank() call — force it now.
    if reranker_obj is not None:
        reranker_obj._load_model()
    # IntentClassifier: model loaded in __init__ (train or cache hit),
    # but calling classify() once ensures the pipeline is exercised.
    if intent_clf_obj is not None:
        try:
            intent_clf_obj.classify("warmup")
        except Exception:
            pass
    cprint("      Models loaded successfully.", style="dim")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_tickets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower == "issue":
            rename[col] = "issue"
        elif lower == "subject":
            rename[col] = "subject"
        elif lower == "company":
            rename[col] = "company"
    df.rename(columns=rename, inplace=True)
    for col in ("issue", "subject", "company"):
        if col not in df.columns:
            df[col] = ""
    return df


def build_query(issue: str, subject: str) -> str:
    parts = [p.strip() for p in [subject, issue] if p.strip()]
    return " ".join(parts)


def _retrieve_with_retry(
    retriever: Retriever,
    query: str,
    domain_hint,
    k_chunks: int,
    max_attempts: int,
    ticket_id: int,
):
    """
    Attempt retrieval up to max_attempts times.
    Returns (RetrievalResult | None, error_occurred: bool).
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = retriever.retrieve(
                query=query,
                domain_hint=domain_hint,
                k_chunks=k_chunks,
            )
            return result, False
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                cprint(
                    f"[warn] Retrieval attempt {attempt}/{max_attempts} failed "
                    f"for ticket #{ticket_id}: {exc}. Retrying …",
                    style="yellow",
                )
    cprint(
        f"[error] All {max_attempts} retrieval attempts failed for ticket "
        f"#{ticket_id}: {last_exc}. Escalating.",
        style="red",
    )
    return None, True


# ---------------------------------------------------------------------------
# Per-ticket processing (parallel-safe, no shared-state mutations)
# ---------------------------------------------------------------------------

def _process_single_ticket(args_tuple) -> dict:
    """
    Process one support ticket through the full pipeline.

    Designed to run safely inside a ThreadPoolExecutor:
    - Reads from shared objects via ctx (all thread-safe or truly read-only).
    - Never mutates shared counters, log files, metrics, or drift state.
    - All stateful side-effects (triage logger, metrics, drift, fail_rep)
      are returned as data and applied serially by the caller in ticket order.

    Parameters
    ----------
    args_tuple : (idx, row, ctx)
        idx  — 0-based DataFrame index
        row  — pandas Series with issue / subject / company fields
        ctx  — SimpleNamespace of shared read-only pipeline components

    Returns
    -------
    dict containing 'row' (CSV), 'log_args' (triage log), 'drift_record',
    'failure_event', timings, and decision objects for the batch collector.
    """
    idx, row, ctx = args_tuple
    t0 = time.perf_counter()

    issue       = str(row.get("issue",   "")).strip()
    subject     = str(row.get("subject", "")).strip()
    company     = str(row.get("company", "")).strip()
    ticket_id   = idx + 1
    query       = build_query(issue, subject)
    ticket_text = issue + " " + subject

    retrieval_error = False
    failure_event   = None  # applied serially by the caller via fail_rep.report()

    ctx.stream_log.log_start(ticket_id)
    t_stage = time.perf_counter()

    # --- 1. Request type classification ---
    request_type = ctx.type_clf.classify(ticket_text)
    ctx.stream_log.log(ticket_id, "classification", request_type,
                       (time.perf_counter() - t_stage) * 1000)

    # --- 2. Domain classification ---
    domain = ctx.domain_clf.classify(company, ticket_text)

    # --- 3. Risk assessment ---
    risk = ctx.risk_engine.assess(ticket_text, domain, request_type)

    # --- 4. Retrieval (hybrid or FAISS-only) with retry + cache ---
    retrieval = None
    t_stage = time.perf_counter()
    if not risk.is_injection and request_type != "invalid":
        domain_hint = domain if domain != "unknown" else None

        cached_ret, cache_hit = (
            ctx.query_cache.get_retrieval(query, domain_hint, ctx.top_k)
            if ctx.query_cache else (None, False)
        )
        if cache_hit:
            retrieval = cached_ret
        elif ctx.use_hybrid and ctx.hybrid_ret is not None:
            try:
                retrieval = ctx.hybrid_ret.search(
                    query, domain_hint=domain_hint, top_k=ctx.top_k
                )
                if ctx.query_cache:
                    ctx.query_cache.put_retrieval(query, domain_hint, ctx.top_k, retrieval)
            except Exception as exc:
                cprint(
                    f"[warn] Hybrid retrieval failed #{ticket_id}: {exc}", style="yellow"
                )
                retrieval_error = True
                failure_event = {
                    "ticket_id":  ticket_id,
                    "error_type": ERROR_RETRIEVAL,
                    "message":    str(exc),
                    "action":     "escalated",
                }
        else:
            retrieval, retrieval_error = _retrieve_with_retry(
                ctx.retriever, query, domain_hint, ctx.top_k,
                ctx.max_attempts, ticket_id,
            )
            if retrieval_error:
                failure_event = {
                    "ticket_id":  ticket_id,
                    "error_type": ERROR_RETRIEVAL,
                    "message":    "All retrieval attempts failed",
                    "action":     "escalated",
                }
            elif retrieval and ctx.query_cache:
                ctx.query_cache.put_retrieval(query, domain_hint, ctx.top_k, retrieval)

    ret_status = f"hits={len(retrieval.chunks)}" if retrieval else "none"
    ctx.stream_log.log(ticket_id, "retrieval", ret_status,
                       (time.perf_counter() - t_stage) * 1000)

    # --- 5. Refine domain from retrieval ---
    if domain == "unknown" and retrieval:
        domain = ctx.domain_clf.classify(company, ticket_text, retrieval)

    # --- 6. Product area ---
    product_area = ctx.area_resolver.resolve(retrieval, domain, ticket_text)

    # --- 7. Reranker (with cache, input cap, and timeout guard) ---
    reranker_score = 0.0
    t_stage = time.perf_counter()
    if retrieval and retrieval.chunks:
        # Limit the chunks sent to the cross-encoder to reduce latency.
        rr_input = retrieval.chunks[:ctx.reranker_input_lim]
        cached_rnk, rnk_hit = (
            ctx.query_cache.get_reranker(query, rr_input, ctx.reranker_top)
            if ctx.query_cache else (None, False)
        )
        if rnk_hit:
            reranker_score = cached_rnk
        else:
            reranker_score = ctx.reranker.top_reranker_score(
                query, rr_input, top_n=ctx.reranker_top,
                timeout_ms=ctx.reranker_timeout,
            )
            if ctx.query_cache:
                ctx.query_cache.put_reranker(
                    query, rr_input, ctx.reranker_top, reranker_score
                )
    ctx.stream_log.log(ticket_id, "reranker", f"score={reranker_score:.3f}",
                       (time.perf_counter() - t_stage) * 1000)

    # --- 8. Intent classifier (with cache) ---
    t_stage = time.perf_counter()
    cached_intent, intent_hit = (
        ctx.query_cache.get_intent(ticket_text)
        if ctx.query_cache else (None, False)
    )
    if intent_hit:
        intent_res = cached_intent
    else:
        intent_res = ctx.intent_clf.classify(ticket_text)
        if ctx.query_cache:
            ctx.query_cache.put_intent(ticket_text, intent_res)
    ctx.stream_log.log(ticket_id, "intent_classifier",
                       f"{intent_res.intent}({intent_res.source})",
                       (time.perf_counter() - t_stage) * 1000)

    # --- 9. Guardrails ---
    t_stage = time.perf_counter()
    guardrail_result = ctx.guardrails.validate(ticket_text, retrieval, request_type)
    ctx.stream_log.log(ticket_id, "guardrails", guardrail_result.status,
                       (time.perf_counter() - t_stage) * 1000)

    # --- 10. Composite confidence scoring + calibration ---
    conf_dict = ctx.conf_scorer.score(
        retrieval, request_type, reranker_score, guardrail_result
    )
    composite_confidence = conf_dict["confidence"]
    if ctx.calibrator is not None:
        composite_confidence = ctx.calibrator.calibrate(composite_confidence)

    # --- 11. Decision ---
    # Only pass the real reranker score when the reranker actually ran (enabled
    # and returned a non-neutral score).  Passing None skips Rule 7.5 so the
    # threshold check does not fire when the reranker is disabled.
    rr_score_for_decision = (
        reranker_score
        if ctx.reranker.enabled and retrieval and retrieval.chunks
        else None
    )
    t_stage = time.perf_counter()
    decision = ctx.decision_engine.decide(
        risk, retrieval, request_type,
        composite_confidence=composite_confidence,
        guardrail_result=guardrail_result,
        reranker_score=rr_score_for_decision,
    )
    ctx.stream_log.log(ticket_id, "decision", decision.action,
                       (time.perf_counter() - t_stage) * 1000)

    # --- 12. Response ---
    response = ctx.response_gen.generate(issue, decision, risk, retrieval, request_type)

    # --- 13. Policy enforcement ---
    t_stage = time.perf_counter()
    is_esc = decision.is_escalated or risk.is_out_of_scope or request_type == "invalid"
    response, policy_res = ctx.policy_en.validate_and_enforce(
        response, retrieval, is_escalation=is_esc
    )
    if not is_esc and policy_res.escalate:
        from modules.decision_engine import Decision
        decision = Decision(
            action="escalated",
            reason="Policy enforcement: no grounding documents available.",
            confidence=composite_confidence,
        )
        from modules.config import ESCALATION_RESPONSE
        response = ESCALATION_RESPONSE
    ctx.stream_log.log(ticket_id, "policy_enforcer",
                       "citation_added" if policy_res.added_citation else policy_res.reason[:20],
                       (time.perf_counter() - t_stage) * 1000)

    # --- 14. Justification ---
    justification = _build_justification(
        domain, request_type, risk, decision, retrieval,
        product_area, conf_dict, guardrail_result, intent_res,
    )

    # --- 15. Decision summary ---
    decision_summary = ""
    if ctx.use_decision_sum:
        decision_summary = build_decision_summary(
            intent=intent_res.intent,
            risk_level=risk.level,
            confidence=composite_confidence,
            action=decision.action,
            is_injection=risk.is_injection,
            is_out_of_scope=risk.is_out_of_scope,
            guardrail_safe=guardrail_result.safe if guardrail_result else True,
            has_retrieval=bool(retrieval and retrieval.chunks),
            request_type=request_type,
            conf_threshold=ctx.conf_threshold,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    ctx.stream_log.log_end(ticket_id, elapsed_ms, decision.action)

    doc_ids = retrieval.doc_ids if retrieval else []

    return {
        "idx":                  idx,
        "ticket_id":            ticket_id,
        "issue":                issue,
        "subject":              subject,
        "company":              company,
        "domain":               domain,
        "request_type":         request_type,
        "product_area":         product_area,
        "risk":                 risk,
        "decision":             decision,
        "composite_confidence": composite_confidence,
        "retrieval_error":      retrieval_error,
        "elapsed_ms":           elapsed_ms,
        "failure_event":        failure_event,
        # Drift data applied serially by the caller
        "drift_record": {
            "intent":     intent_res.intent,
            "risk_level": risk.level,
            "action":     decision.action,
        },
        # Pre-assembled output CSV row
        "row": {
            "issue":            issue,
            "subject":          subject,
            "company":          company,
            "response":         response,
            "product_area":     product_area,
            "status":           decision.action,
            "request_type":     request_type,
            "intent":           intent_res.intent,
            "justification":    justification,
            "confidence":       round(composite_confidence, 4),
            "decision_summary": decision_summary,
        },
        # Kwargs for triage_logger.log_ticket(), applied serially by the caller
        "log_args": {
            "ticket_id":         ticket_id,
            "issue":             issue,
            "subject":           subject,
            "company":           company,
            "domain":            domain,
            "request_type":      request_type,
            "risk_level":        risk.level,
            "risk_reasons":      risk.reasons,
            "action":            decision.action,
            "confidence":        composite_confidence,
            "product_area":      product_area,
            "retrieved_doc_ids": doc_ids,
            "response":          response,
            "justification":     justification,
            "latency_ms":        elapsed_ms,
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace, cfg: dict) -> MetricsTracker:
    """
    Process all tickets and write output.csv + log.txt.

    Returns the MetricsTracker so callers can include run metrics in the
    evaluation report.
    """
    # --- AGENTS.md log ---
    try:
        agents_log_session_start(str(BASE_DIR))
    except Exception:
        pass

    cprint("\n[bold cyan]Support Triage Agent[/bold cyan]", style="")
    cprint(f"Tickets : {args.tickets}", style="dim")
    cprint(f"Output  : {args.output}", style="dim")
    cprint(f"Log     : {args.log}\n", style="dim")

    # --- Config values ---
    ret_cfg   = cfg.get("retrieval", {})
    rnk_cfg   = cfg.get("reranker", {})
    grd_cfg   = cfg.get("guardrails", {})
    ret_cfg2  = cfg.get("retry", {})
    clf_cfg   = cfg.get("classifier", {})
    pol_cfg   = cfg.get("policy", {})
    log_cfg   = cfg.get("logging", {})
    feat_cfg  = cfg.get("features", {})

    top_k              = ret_cfg.get("top_k", args.top_k)
    reranker_en        = rnk_cfg.get("enabled", True)
    reranker_top       = rnk_cfg.get("top_n", 3)
    reranker_input_lim = rnk_cfg.get("input_limit", 5)   # cap cross-encoder input
    reranker_timeout   = rnk_cfg.get("timeout_ms", 700)  # abort if too slow
    strict_mode        = grd_cfg.get("strict_mode", True)
    max_attempts       = ret_cfg2.get("max_attempts", 3)
    use_hybrid         = ret_cfg.get("method", "embedding") == "hybrid"
    bm25_weight        = ret_cfg.get("bm25_weight",      0.4)
    embed_weight       = ret_cfg.get("embedding_weight", 0.6)

    # --- Load corpus ---
    cprint("[1/5] Loading corpus …", style="bold")
    loader = DocumentLoader(DATA_DIR)
    docs   = loader.load_all()
    chunks = loader.chunk_documents(docs)
    cprint(f"      {len(docs)} documents -> {len(chunks)} chunks", style="dim")
    # Release the full document objects — only chunks are needed downstream.
    del docs
    gc.collect()

    # --- Build / load FAISS index ---
    cprint("[2/5] Building vector index …", style="bold")
    index = VectorIndex()
    index.build(chunks, force_rebuild=args.rebuild_index)
    index.optimize()   # set nprobe=8 on IVF indexes; no-op for FlatIP

    # --- Initialise modules ---
    faiss_retriever = Retriever(index)
    domain_clf      = DomainClassifier()
    type_clf        = RequestTypeClassifier()
    area_resolver   = ProductAreaResolver()
    risk_engine     = RiskEngine()
    decision_engine = DecisionEngine()
    response_gen    = ResponseGenerator()
    triage_logger   = TriageLogger(args.log)

    # --- Hybrid retriever (new) ---
    if use_hybrid:
        hybrid_ret = HybridRetriever(
            faiss_retriever, chunks,
            bm25_weight=bm25_weight, embedding_weight=embed_weight,
        )
        cprint("      Hybrid retrieval: BM25 + FAISS enabled.", style="dim")
    else:
        hybrid_ret = None

    # Unified retriever alias used in the loop
    retriever = faiss_retriever   # kept for retry helper; hybrid path is separate

    # --- Advanced modules ---
    reranker     = Reranker(enabled=reranker_en)
    guardrails   = Guardrails(strict_mode=strict_mode)
    conf_scorer  = ConfidenceScorer()
    metrics      = MetricsTracker()
    intent_clf   = IntentClassifier(
        use_rules=clf_cfg.get("use_rules", True),
        use_ml=clf_cfg.get("use_ml", True),
    )
    policy_en    = PolicyEnforcer(
        require_citation=pol_cfg.get("require_citation", True)
    )
    stream_log   = StreamLogger(
        log_file=args.log.parent / "stream.log" if log_cfg.get("streaming", True) else None,
        to_console=log_cfg.get("stream_to_console", False),
    )

    # --- Production features (Phase 4) ---
    use_cache        = feat_cfg.get("caching",               True)
    use_drift        = feat_cfg.get("drift_monitoring",      True)
    use_failure_rep  = feat_cfg.get("failure_reporting",     True)
    use_calibration  = feat_cfg.get("confidence_calibration", True)
    use_decision_sum = feat_cfg.get("decision_summary",      True)

    query_cache  = QueryCache()          if use_cache       else None
    # Dynamic baseline path: persisted alongside output artifacts
    drift_baseline_path = args.output.parent / "drift_baseline.json"
    drift_mon    = DriftMonitor(baseline_path=drift_baseline_path) if use_drift else None
    fail_rep     = FailureReporter()     if use_failure_rep else None
    calibrator   = ConfidenceCalibrator() if use_calibration else None

    conf_threshold = cfg.get("confidence", {}).get("threshold", 0.65)

    # --- Preload models (eliminates first-ticket cold-start spike) ---
    preload_models(reranker, intent_clf)

    # --- Index warm-up: one dummy query forces any lazy FAISS state ---
    cprint("      Running index warm-up …", style="dim")
    try:
        _wup = index.query("support ticket warmup", k=1)
        del _wup
    except Exception:
        pass

    # --- Build shared read-only pipeline context for worker threads ---
    ctx = SimpleNamespace(
        hybrid_ret=hybrid_ret,
        retriever=retriever,
        domain_clf=domain_clf,
        type_clf=type_clf,
        area_resolver=area_resolver,
        risk_engine=risk_engine,
        decision_engine=decision_engine,
        response_gen=response_gen,
        reranker=reranker,
        guardrails=guardrails,
        conf_scorer=conf_scorer,
        intent_clf=intent_clf,
        policy_en=policy_en,
        calibrator=calibrator,
        query_cache=query_cache,
        top_k=top_k,
        reranker_top=reranker_top,
        reranker_input_lim=reranker_input_lim,
        reranker_timeout=reranker_timeout,
        max_attempts=max_attempts,
        use_hybrid=use_hybrid,
        conf_threshold=conf_threshold,
        use_decision_sum=use_decision_sum,
        stream_log=stream_log,
    )

    # --- Load tickets ---
    cprint("[3/5] Processing tickets …", style="bold")
    tickets = load_tickets(args.tickets)
    n = len(tickets)
    cprint(f"      {n} tickets loaded\n", style="dim")

    rows = []
    total_ms = 0.0
    reply_count = 0
    escalate_count = 0

    ticket_list  = list(tickets.iterrows())
    run_wall_t0  = time.perf_counter()   # wall-clock start for throughput calc

    if RICH:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        )
        task_id = progress.add_task("Triaging …", total=n)
        progress.start()

    # --- Batched parallel processing ---
    # Computation (retrieval, reranking, classification) runs in up to
    # _MAX_WORKERS threads per batch of _BATCH_SIZE tickets.
    # All stateful side-effects (triage log, metrics, drift) are applied
    # serially after each future resolves, preserving ticket order.
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        for batch_start in range(0, n, _BATCH_SIZE):
            batch = ticket_list[batch_start:batch_start + _BATCH_SIZE]

            # Submit batch; futures are ordered by ticket index
            futures = [
                executor.submit(_process_single_ticket, (idx, row, ctx))
                for idx, row in batch
            ]

            # Collect in original order — preserves deterministic CSV/log output
            for future in futures:
                result = future.result()

                elapsed_ms  = result["elapsed_ms"]
                total_ms   += elapsed_ms

                if result["decision"].action == "replied":
                    reply_count += 1
                else:
                    escalate_count += 1

                # --- Metrics (serial) ---
                metrics.record_ticket(
                    ticket_id=result["ticket_id"],
                    request_type=result["request_type"],
                    domain=result["domain"],
                    risk_level=result["risk"].level,
                    confidence=result["composite_confidence"],
                    action=result["decision"].action,
                    latency_ms=elapsed_ms,
                    error=result["retrieval_error"],
                )

                # --- Drift monitoring (serial) ---
                if drift_mon is not None:
                    drift_mon.record(**result["drift_record"])

                # --- Failure reporting (serial) ---
                if fail_rep is not None and result["failure_event"] is not None:
                    fe = result["failure_event"]
                    fail_rep.report(
                        fe["ticket_id"], fe["error_type"],
                        fe["message"], fe["action"],
                    )

                # --- Triage log (serial, buffered — flushes every 10 tickets) ---
                triage_logger.log_ticket(**result["log_args"])

                # --- Output CSV row ---
                rows.append(result["row"])

                if RICH:
                    progress.advance(task_id)

            # Release per-batch objects to keep peak RSS low
            gc.collect()

    if RICH:
        progress.stop()

    # --- Flush any remaining buffered log entries ---
    try:
        stream_log.flush()
    except Exception:
        pass

    # --- Performance report ---
    run_wall_elapsed = time.perf_counter() - run_wall_t0
    _latencies = [t.latency_ms for t in metrics.run.tickets]
    perf_report = {
        "avg_latency_ms":  round(float(np.mean(_latencies))           if _latencies else 0.0, 2),
        "p95_latency_ms":  round(float(np.percentile(_latencies, 95)) if _latencies else 0.0, 2),
        "memory_usage_mb": round(metrics.run.peak_memory_mb, 2),
        "throughput_tps":  round(n / max(run_wall_elapsed, 1e-9), 4),
    }
    _perf_path = args.output.parent / "performance_report.json"
    _perf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_perf_path, "w", encoding="utf-8") as _pf:
            json.dump(perf_report, _pf, indent=2)
        cprint(f"      Perf report : {_perf_path}", style="dim")
    except Exception as _exc:
        cprint(f"[warn] Could not write performance_report.json: {_exc}", style="yellow")

    # --- Write output CSV ---
    cprint("[4/5] Writing output …", style="bold")
    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    cprint(f"      Output CSV  : {args.output}", style="green")

    # --- Save drift report ---
    if drift_mon is not None:
        drift_path = args.output.parent / "drift_report.json"
        try:
            drift_mon.save_report(drift_path)
            cprint(f"      Drift report: {drift_path}", style="dim")
        except Exception as exc:
            cprint(f"[warn] Could not save drift report: {exc}", style="yellow")

    # --- Save failure report ---
    if fail_rep is not None:
        fail_path = args.output.parent / "failure_report.json"
        try:
            fail_rep.save(fail_path)
            if fail_rep.has_failures():
                cprint(
                    f"      Failure report: {fail_path} ({fail_rep.count()} events)",
                    style="yellow",
                )
            else:
                cprint(f"      Failure report: {fail_path} (no failures)", style="dim")
        except Exception as exc:
            cprint(f"[warn] Could not save failure report: {exc}", style="yellow")

    # --- Cache stats ---
    if query_cache is not None:
        cs = query_cache.stats()
        cprint(
            f"      Cache: size={cs['size']} hits={cs['hits']} "
            f"misses={cs['misses']} hit_rate={cs['hit_rate']:.2%}",
            style="dim",
        )

    # --- Triage summary ---
    triage_logger.log_summary(n, reply_count, escalate_count, total_ms)

    # --- Observability snapshot ---
    cprint("[5/5] Finalising metrics …", style="bold")
    run_snap = metrics.snapshot()
    cprint(
        f"      Metrics: tickets={run_snap['ticket_count']} "
        f"escalations={run_snap['escalation_count']} "
        f"errors={run_snap['error_count']} "
        f"peak_mem={run_snap['peak_memory_mb']:.1f}MB",
        style="dim",
    )

    # --- AGENTS.md log ---
    try:
        agents_log_run_complete(
            str(args.output),
            str(args.log),
            n, reply_count, escalate_count,
            str(BASE_DIR),
        )
    except Exception:
        pass

    # --- Terminal summary table ---
    avg_ms = total_ms / max(n, 1)
    if RICH:
        table = Table(title="Run Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric",  style="cyan")
        table.add_column("Value",   style="white")
        table.add_row("Total tickets",   str(n))
        table.add_row("Replied",         str(reply_count))
        table.add_row("Escalated",       str(escalate_count))
        table.add_row("Avg confidence",  f"{run_snap['avg_confidence']:.3f}")
        table.add_row("Avg latency",     f"{avg_ms:.1f} ms")
        table.add_row("Total time",      f"{total_ms / 1000:.2f} s")
        table.add_row("Peak memory",     f"{run_snap['peak_memory_mb']:.1f} MB")
        console.print(table)
    else:
        print(f"\nTotal: {n}  |  Replied: {reply_count}  |  Escalated: {escalate_count}")
        print(f"Avg latency: {avg_ms:.1f} ms  |  Total: {total_ms/1000:.2f} s")

    cprint(f"\nOutput CSV : {args.output}", style="bold green")
    cprint(f"Log file   : {args.log}\n",   style="bold green")

    return metrics


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

def run_backtest(args: argparse.Namespace, cfg: dict) -> None:
    """
    Run the pipeline then compare predictions against ground truth.
    Saves evaluation_report.json to args.eval_report.
    """
    cprint("\n[bold yellow]=== BACKTEST MODE ===[/bold yellow]", style="")

    # --- Use backtest tickets: prefer ground-truth CSV if it has input columns,
    #     otherwise fall back to the normal tickets file ---
    bt_tickets = args.tickets
    if args.ground_truth.exists():
        # Check if ground truth has issue/subject/company (can serve as input)
        gt_df = pd.read_csv(args.ground_truth, dtype=str, nrows=1).fillna("")
        gt_cols = [c.lower().strip() for c in gt_df.columns]
        if "issue" in gt_cols or "subject" in gt_cols:
            bt_tickets = args.ground_truth
            cprint(
                f"[backtest] Using ground-truth file as input: {bt_tickets}",
                style="dim",
            )

    # Save original tickets path and swap for pipeline run
    original_tickets = args.tickets
    args.tickets = bt_tickets

    # Redirect output to a temporary path so we don't overwrite the real output
    bt_output = args.output.parent / "backtest_output.csv"
    original_output = args.output
    args.output = bt_output

    # Run full pipeline
    metrics = run_pipeline(args, cfg)

    # Restore original paths
    args.tickets = original_tickets
    args.output  = original_output

    # --- Evaluate against ground truth ---
    if not args.ground_truth.exists():
        cprint(
            f"[backtest] Ground-truth file not found: {args.ground_truth}\n"
            "Saving run metrics only.",
            style="yellow",
        )
        run_snap = metrics.snapshot()
        args.eval_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.eval_report, "w", encoding="utf-8") as f:
            json.dump(run_snap, f, indent=2)
        cprint(f"[backtest] Saved metrics to {args.eval_report}", style="green")
        return

    cprint(
        f"\n[backtest] Comparing {bt_output} vs {args.ground_truth} …",
        style="dim",
    )
    evaluator = Evaluator()
    try:
        results = evaluator.evaluate(bt_output, args.ground_truth)
        evaluator.print_report(results)
        run_snap = metrics.snapshot()
        evaluator.save_report(results, args.eval_report, run_metrics=run_snap)
        cprint(f"[backtest] Saved evaluation_report.json → {args.eval_report}", style="bold green")
    except Exception as exc:
        cprint(f"[backtest] Evaluation failed: {exc}", style="red")
        # Still save run metrics as fallback
        run_snap = metrics.snapshot()
        args.eval_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.eval_report, "w", encoding="utf-8") as f:
            json.dump(run_snap, f, indent=2)
        cprint(f"[backtest] Saved run metrics to {args.eval_report}", style="yellow")


# ---------------------------------------------------------------------------
# Justification builder
# ---------------------------------------------------------------------------

def _build_justification(
    domain: str,
    request_type: str,
    risk,
    decision,
    retrieval,
    product_area: str,
    conf_dict: dict,
    guardrail_result,
    intent_res=None,
) -> str:
    parts = []

    # Domain + area
    parts.append(f"Domain: {domain}; product_area: {product_area}.")

    # Request type + intent
    if intent_res is not None:
        parts.append(
            f"Classified as '{request_type}'; intent='{intent_res.intent}'"
            f"({intent_res.confidence:.2f}, {intent_res.source})."
        )
    else:
        parts.append(f"Classified as '{request_type}'.")

    # Risk
    if risk.is_injection:
        parts.append("Prompt injection detected.")
    elif risk.is_out_of_scope:
        parts.append("Request is out of scope.")
    elif risk.level != "low":
        top_reason = risk.reasons[0] if risk.reasons else risk.level
        parts.append(f"Risk level: {risk.level} ({top_reason}).")

    # Guardrails
    if guardrail_result and not guardrail_result.safe:
        parts.append(f"Guardrail violations: {guardrail_result.violation_summary()}.")

    # Confidence breakdown
    conf = conf_dict.get("confidence", 0.0)
    parts.append(
        f"Confidence: {conf:.3f} "
        f"(ret={conf_dict.get('retrieval_score', 0):.2f} "
        f"clf={conf_dict.get('classifier_prob', 0):.2f} "
        f"rnk={conf_dict.get('reranker_score', 0):.2f} "
        f"grd={conf_dict.get('guardrail_score', 0):.2f})."
    )

    # Decision
    if decision.is_escalated:
        parts.append(decision.reason)
    else:
        if retrieval and retrieval.top_docs:
            top_title = retrieval.top_docs[0][1][:80]
            parts.append(
                f"Best match: '{top_title}'. Response grounded in corpus."
            )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Force UTF-8 on Windows to avoid cp1252 encode errors (terminal only)
    import io as _io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # Configure basic logging (metrics / reranker / guardrails use it)
    logging.basicConfig(
        level=logging.WARNING,          # suppress DEBUG/INFO by default
        format="%(levelname)s:%(name)s: %(message)s",
    )

    args = parse_args()

    # Load config.yaml from code/ directory
    cfg = load_config(Path(__file__).resolve().parent / "config.yaml")

    # Apply seed from config (overrides module-level constant if set)
    seed = cfg.get("seed", RANDOM_SEED)
    set_seeds(seed)

    # Health check mode (lightweight, exits before model loading)
    if args.health_check:
        run_health_check(cfg)
        # run_health_check calls sys.exit internally; line below is a safety net
        sys.exit(0)

    if args.mode == "backtest":
        run_backtest(args, cfg)
    else:
        # Normal mode
        if not args.tickets.exists():
            print(f"[error] Tickets file not found: {args.tickets}", file=sys.stderr)
            sys.exit(1)
        run_pipeline(args, cfg)
