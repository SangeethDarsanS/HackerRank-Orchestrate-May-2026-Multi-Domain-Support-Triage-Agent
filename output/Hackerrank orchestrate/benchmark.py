#!/usr/bin/env python3
"""
Benchmark harness for the Support Triage Agent.

Runs the full pipeline on a dataset, measures performance metrics, and
saves benchmark_results.json.

Usage:
    python benchmark.py
    python benchmark.py --tickets  ../support_tickets/support_tickets.csv
    python benchmark.py --ground-truth ../support_tickets/sample_support_tickets.csv
    python benchmark.py --no-reranker          # faster run (skips cross-encoder)

Metrics reported:
    accuracy         — status (replied/escalated) classification accuracy
    precision        — binary precision (pos_label=replied)
    recall           — binary recall
    f1               — F1 score
    escalation_rate  — fraction of tickets escalated
    avg_latency_ms   — mean per-ticket latency
    throughput       — tickets / second (wall-clock)
    error_count      — retrieval failures
    peak_memory_mb   — peak process RSS (requires psutil)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.config import (
    DATA_DIR, BASE_DIR, SUPPORT_TICKETS_DIR,
    OUTPUT_CSV, TRIAGE_LOG_FILE, RANDOM_SEED, TOP_K_CHUNKS,
)
from modules.loader import DocumentLoader
from modules.indexer import VectorIndex
from modules.retriever import Retriever
from modules.hybrid_retriever import HybridRetriever
from modules.classifier import DomainClassifier, RequestTypeClassifier, ProductAreaResolver
from modules.risk_engine import RiskEngine
from modules.decision_engine import DecisionEngine
from modules.response_generator import ResponseGenerator
from modules.reranker import Reranker
from modules.guardrails import Guardrails
from modules.confidence import ConfidenceScorer
from modules.intent_classifier import IntentClassifier
from modules.policy_enforcer import PolicyEnforcer
from modules.metrics import MetricsTracker
from modules.evaluator import Evaluator
from main import load_config, set_seeds, load_tickets, build_query, _retrieve_with_retry


# ---------------------------------------------------------------------------
# Benchmark output path
# ---------------------------------------------------------------------------
_BENCHMARK_OUT = SUPPORT_TICKETS_DIR / "benchmark_results.json"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Support Triage Agent — Benchmark Harness")
    p.add_argument(
        "--tickets",
        type=Path,
        default=SUPPORT_TICKETS_DIR / "support_tickets.csv",
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=SUPPORT_TICKETS_DIR / "sample_support_tickets.csv",
        help="CSV with ground-truth labels (optional).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_BENCHMARK_OUT,
        help="Path for benchmark_results.json.",
    )
    p.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable cross-encoder reranker for faster benchmarking.",
    )
    p.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Use pure FAISS retrieval instead of hybrid.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_CHUNKS,
    )
    p.add_argument(
        "--rebuild-index",
        action="store_true",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace, cfg: dict) -> dict:
    """
    Run the full pipeline on all tickets.
    Returns a flat metrics dict suitable for benchmark_results.json.
    """
    wall_start = time.perf_counter()

    # ---- Load corpus ----
    print("[benchmark] Loading corpus …")
    loader = DocumentLoader(DATA_DIR)
    docs   = loader.load_all()
    chunks = loader.chunk_documents(docs)
    print(f"[benchmark] {len(docs)} docs → {len(chunks)} chunks.")

    # ---- Build index ----
    print("[benchmark] Building / loading index …")
    index = VectorIndex()
    index.build(chunks, force_rebuild=args.rebuild_index)

    # ---- Init modules ----
    faiss_retriever = Retriever(index)

    use_hybrid = (
        not args.no_hybrid
        and cfg.get("retrieval", {}).get("method", "hybrid") == "hybrid"
    )
    if use_hybrid:
        bm25_w  = cfg.get("retrieval", {}).get("bm25_weight",      0.4)
        emb_w   = cfg.get("retrieval", {}).get("embedding_weight", 0.6)
        hybrid_retriever = HybridRetriever(
            faiss_retriever, chunks,
            bm25_weight=bm25_w, embedding_weight=emb_w,
        )
        print("[benchmark] Using HYBRID retrieval (BM25 + FAISS).")
    else:
        hybrid_retriever = None
        print("[benchmark] Using FAISS-only retrieval.")

    domain_clf      = DomainClassifier()
    type_clf        = RequestTypeClassifier()
    area_resolver   = ProductAreaResolver()
    risk_engine     = RiskEngine()
    decision_engine = DecisionEngine()
    response_gen    = ResponseGenerator()

    reranker_en = (not args.no_reranker) and cfg.get("reranker", {}).get("enabled", True)
    reranker     = Reranker(enabled=reranker_en)
    reranker_top = cfg.get("reranker", {}).get("top_n", 3)
    guardrails   = Guardrails(strict_mode=cfg.get("guardrails", {}).get("strict_mode", True))
    conf_scorer  = ConfidenceScorer()
    intent_clf   = IntentClassifier(
        use_rules=cfg.get("classifier", {}).get("use_rules", True),
        use_ml=cfg.get("classifier", {}).get("use_ml", True),
    )
    policy_cfg   = cfg.get("policy", {})
    policy_en    = PolicyEnforcer(require_citation=policy_cfg.get("require_citation", True))
    metrics      = MetricsTracker()
    max_attempts = cfg.get("retry", {}).get("max_attempts", 3)

    # ---- Load tickets ----
    if not args.tickets.exists():
        print(f"[error] Tickets file not found: {args.tickets}", file=sys.stderr)
        sys.exit(1)

    tickets = load_tickets(args.tickets)
    n = len(tickets)
    print(f"[benchmark] Processing {n} tickets …")

    rows = []
    for idx, row in tickets.iterrows():
        t0 = time.perf_counter()

        issue       = str(row.get("issue",   "")).strip()
        subject     = str(row.get("subject", "")).strip()
        company     = str(row.get("company", "")).strip()
        ticket_id   = idx + 1
        query       = build_query(issue, subject)
        ticket_text = issue + " " + subject

        request_type = type_clf.classify(ticket_text)
        domain       = domain_clf.classify(company, ticket_text)
        risk         = risk_engine.assess(ticket_text, domain, request_type)

        retrieval = None
        retrieval_error = False
        if not risk.is_injection and request_type != "invalid":
            domain_hint = domain if domain != "unknown" else None
            if use_hybrid and hybrid_retriever:
                try:
                    retrieval = hybrid_retriever.search(
                        query, domain_hint=domain_hint, top_k=args.top_k
                    )
                    retrieval_error = False
                except Exception as exc:
                    print(f"[warn] Hybrid retrieval failed for #{ticket_id}: {exc}")
                    retrieval_error = True
            else:
                retrieval, retrieval_error = _retrieve_with_retry(
                    faiss_retriever, query, domain_hint, args.top_k, max_attempts, ticket_id
                )

        if domain == "unknown" and retrieval:
            domain = domain_clf.classify(company, ticket_text, retrieval)

        product_area = area_resolver.resolve(retrieval, domain, ticket_text)

        reranker_score = 0.0
        if retrieval and retrieval.chunks:
            reranker_score = reranker.top_reranker_score(
                query, retrieval.chunks, top_n=reranker_top
            )

        intent_res     = intent_clf.classify(ticket_text)
        guardrail_res  = guardrails.validate(ticket_text, retrieval, request_type)

        conf_dict = conf_scorer.score(
            retrieval, request_type, reranker_score, guardrail_res
        )
        composite_conf = conf_dict["confidence"]

        decision = decision_engine.decide(
            risk, retrieval, request_type,
            composite_confidence=composite_conf,
            guardrail_result=guardrail_res,
        )

        response = response_gen.generate(issue, decision, risk, retrieval, request_type)
        is_esc   = decision.is_escalated or risk.is_out_of_scope or request_type == "invalid"
        response, policy_res = policy_en.validate_and_enforce(response, retrieval, is_escalation=is_esc)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        metrics.record_ticket(
            ticket_id=ticket_id,
            request_type=request_type,
            domain=domain,
            risk_level=risk.level,
            confidence=composite_conf,
            action=decision.action,
            latency_ms=elapsed_ms,
            error=retrieval_error,
        )

        rows.append({
            "issue":        issue,
            "subject":      subject,
            "company":      company,
            "status":       decision.action,
            "request_type": request_type,
            "confidence":   round(composite_conf, 4),
            "intent":       intent_res.intent,
            "latency_ms":   round(elapsed_ms, 2),
        })

        if ticket_id % 10 == 0 or ticket_id == n:
            snap = metrics.snapshot()
            print(
                f"  [{ticket_id}/{n}] avg_latency={snap['average_latency_ms']:.0f}ms  "
                f"escalations={snap['escalation_count']}"
            )

    wall_elapsed = time.perf_counter() - wall_start
    throughput   = n / max(wall_elapsed, 0.001)

    # ---- Save predictions CSV ----
    pred_path = args.tickets.parent / "benchmark_pred.csv"
    pred_df   = pd.DataFrame(rows)
    pred_df.to_csv(pred_path, index=False)
    print(f"[benchmark] Predictions → {pred_path}")

    # ---- Compute quality metrics (if ground truth available) ----
    evaluator = Evaluator()
    quality   = {}
    if args.ground_truth.exists():
        try:
            eval_results = evaluator.evaluate(pred_path, args.ground_truth)
            flat         = evaluator.generate_flat_report(eval_results)
            quality      = flat
            print("\n[benchmark] Quality metrics:")
            evaluator.print_report(eval_results)
        except Exception as exc:
            print(f"[warn] Could not compute quality metrics: {exc}")

    # ---- Build final benchmark_results.json ----
    run_snap = metrics.snapshot()

    result = {
        "n_tickets":       n,
        "accuracy":        quality.get("accuracy",       0.0),
        "precision":       quality.get("precision",      0.0),
        "recall":          quality.get("recall",         0.0),
        "f1":              quality.get("f1",             0.0),
        "escalation_rate": run_snap["escalation_rate"],
        "avg_confidence":  run_snap["avg_confidence"],
        "avg_latency_ms":  round(run_snap["average_latency_ms"], 2),
        "throughput_tps":  round(throughput, 2),
        "error_count":     run_snap["error_count"],
        "peak_memory_mb":  run_snap["peak_memory_mb"],
        "wall_time_s":     round(wall_elapsed, 2),
        "reranker_enabled": reranker_en,
        "hybrid_enabled":   use_hybrid,
    }

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Force UTF-8 on Windows
    import io as _io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    args = _parse_args()
    cfg  = load_config(Path(__file__).resolve().parent / "config.yaml")
    seed = cfg.get("seed", RANDOM_SEED)
    set_seeds(seed)

    print("\n" + "=" * 60)
    print("  SUPPORT TRIAGE AGENT — BENCHMARK HARNESS")
    print("=" * 60)

    result = run_benchmark(args, cfg)

    # ---- Save benchmark_results.json ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")
    print("=" * 60)
    print(f"\n  Saved → {args.output}\n")


if __name__ == "__main__":
    main()
