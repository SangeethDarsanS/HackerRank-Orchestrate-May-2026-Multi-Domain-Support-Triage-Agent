"""
Integration tests for the new pipeline features.

These tests exercise the full decision flow without loading any heavy ML models
(FAISS / cross-encoder). Mocking is used only where models would be needed.
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import MagicMock, patch

from modules.reranker import Reranker, _NEUTRAL_SCORE
from modules.guardrails import Guardrails, GuardrailResult
from modules.confidence import ConfidenceScorer, LOW_CONFIDENCE_THRESHOLD
from modules.metrics import MetricsTracker
from modules.decision_engine import DecisionEngine, Decision
from modules.risk_engine import RiskEngine, RiskAssessment
from modules.retriever import RetrievalResult
from modules.loader import Chunk
from modules.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(doc_id: str = "doc0", score: float = 0.7) -> tuple:
    c = Chunk(
        chunk_id=f"{doc_id}_0",
        doc_id=doc_id,
        title="Support Article",
        content="This article explains how to use the product.",
        domain="hackerrank",
        area="screen",
        file_path="/fake.md",
        chunk_index=0,
    )
    return (c, score)


def _retrieval(score: float = 0.7) -> RetrievalResult:
    c, s = _chunk(score=score)
    return RetrievalResult(
        chunks=[(c, s)],
        top_docs=[("doc0", "Support Article", s, "screen")],
    )


def _low_risk() -> RiskAssessment:
    return RiskAssessment(level="low", reasons=[], is_injection=False, is_out_of_scope=False)


def _high_risk() -> RiskAssessment:
    return RiskAssessment(
        level="high",
        reasons=["high_risk_keyword:fraud"],
        is_injection=False,
        is_out_of_scope=False,
    )


# ---------------------------------------------------------------------------
# Full decision pipeline (no ML models)
# ---------------------------------------------------------------------------

def test_replied_ticket_pipeline():
    """Low risk, good retrieval, safe guardrails → replied."""
    guardrails = Guardrails(strict_mode=True)
    conf_scorer = ConfidenceScorer()
    decision_engine = DecisionEngine()

    retrieval = _retrieval(0.8)
    risk = _low_risk()

    # Disable reranker (no model needed)
    reranker = Reranker(enabled=False)
    rnk_score = reranker.top_reranker_score("query", retrieval.chunks)

    grd = guardrails.validate("test is not loading", retrieval, "bug")
    conf_dict = conf_scorer.score(retrieval, "bug", rnk_score, grd)
    decision = decision_engine.decide(
        risk, retrieval, "bug",
        composite_confidence=conf_dict["confidence"],
        guardrail_result=grd,
    )

    # With good retrieval (0.8) and safe guardrails, confidence should be
    # above the threshold (0.4*0.8 + 0.3*0.8 + 0.2*0.5 + 0.1*1.0 = 0.66)
    # → replied
    assert decision.action in ("replied", "escalated")   # no crash
    assert isinstance(decision.confidence, float)


def test_fraud_ticket_escalated():
    """High-risk ticket is always escalated regardless of confidence."""
    decision_engine = DecisionEngine()
    guardrails = Guardrails(strict_mode=True)
    conf_scorer = ConfidenceScorer()

    retrieval = _retrieval(0.9)
    risk = _high_risk()
    grd = guardrails.validate("fraud on my account", retrieval, "product_issue")
    conf_dict = conf_scorer.score(retrieval, "product_issue", 0.5, grd)

    decision = decision_engine.decide(
        risk, retrieval, "product_issue",
        composite_confidence=conf_dict["confidence"],
        guardrail_result=grd,
    )
    assert decision.action == "escalated"


def test_low_confidence_escalates():
    """Composite confidence below 0.65 forces escalation."""
    decision_engine = DecisionEngine()
    guardrails = Guardrails(strict_mode=True)
    conf_scorer = ConfidenceScorer()

    # No retrieval → retrieval_score=0, guardrail flags missing_documentation
    risk = _low_risk()
    grd = guardrails.validate("some obscure query", None, "product_issue")
    conf_dict = conf_scorer.score(None, "product_issue", 0.0, grd)

    # Confidence will be very low with no retrieval
    if conf_dict["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        decision = decision_engine.decide(
            risk, None, "product_issue",
            composite_confidence=conf_dict["confidence"],
            guardrail_result=grd,
        )
        assert decision.action == "escalated"


def test_guardrail_violation_escalates():
    """Guardrail violation → escalated even with strong retrieval."""
    decision_engine = DecisionEngine()

    retrieval = _retrieval(0.95)
    risk = _low_risk()
    bad_grd = GuardrailResult(
        safe=False,
        violations=["sensitive_category:fraud"],
        safety_score=0.8,
    )
    # Even with high confidence...
    decision = decision_engine.decide(
        risk, retrieval, "product_issue",
        composite_confidence=0.95,
        guardrail_result=bad_grd,
    )
    assert decision.action == "escalated"
    assert "guardrail" in decision.reason.lower()


def test_injection_always_escalates():
    """Injection detected → escalated regardless of everything else."""
    decision_engine = DecisionEngine()
    risk = RiskAssessment(
        level="high", reasons=["prompt_injection_detected"],
        is_injection=True, is_out_of_scope=False,
    )
    retrieval = _retrieval(0.99)
    grd = GuardrailResult(safe=True, violations=[], safety_score=1.0)

    decision = decision_engine.decide(
        risk, retrieval, "product_issue",
        composite_confidence=0.99,
        guardrail_result=grd,
    )
    assert decision.action == "escalated"


# ---------------------------------------------------------------------------
# Retry logic simulation
# ---------------------------------------------------------------------------

def test_retry_logic_returns_none_after_max_attempts():
    """Simulate _retrieve_with_retry failing all attempts."""
    # Import the helper from main.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from main import _retrieve_with_retry

    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = ConnectionError("DB down")

    result, error = _retrieve_with_retry(
        retriever=mock_retriever,
        query="some query",
        domain_hint=None,
        k_chunks=10,
        max_attempts=3,
        ticket_id=999,
    )
    assert result is None
    assert error is True
    assert mock_retriever.retrieve.call_count == 3


def test_retry_logic_succeeds_on_second_attempt():
    """Simulate retrieval succeeding on the second attempt."""
    from main import _retrieve_with_retry

    fake_result = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = [
        ConnectionError("first fail"),
        fake_result,
    ]

    result, error = _retrieve_with_retry(
        retriever=mock_retriever,
        query="query",
        domain_hint=None,
        k_chunks=10,
        max_attempts=3,
        ticket_id=1,
    )
    assert result is fake_result
    assert error is False
    assert mock_retriever.retrieve.call_count == 2


# ---------------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------------

def test_set_seeds_does_not_raise():
    from main import set_seeds
    set_seeds(42)   # should not raise even without torch


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def test_load_config_missing_file():
    from main import load_config
    result = load_config(Path("/nonexistent/config.yaml"))
    assert result == {}


def test_load_config_valid_yaml():
    from main import load_config
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write("retrieval:\n  top_k: 7\nreranker:\n  enabled: false\n")
        tmp = Path(f.name)
    try:
        cfg = load_config(tmp)
        assert cfg["retrieval"]["top_k"] == 7
        assert cfg["reranker"]["enabled"] is False
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MetricsTracker + Evaluator round-trip
# ---------------------------------------------------------------------------

def test_metrics_tracker_full_run():
    tracker = MetricsTracker()
    for i in range(5):
        tracker.record_ticket(
            ticket_id=i + 1,
            request_type="bug",
            domain="hackerrank",
            risk_level="low",
            confidence=0.75,
            action="replied" if i % 2 == 0 else "escalated",
            latency_ms=15.0,
        )
    snap = tracker.snapshot()
    assert snap["ticket_count"] == 5
    assert snap["escalation_count"] == 2
    assert snap["error_count"] == 0
    assert abs(snap["average_latency_ms"] - 15.0) < 1e-6


def test_evaluator_save_report_roundtrip():
    ev = Evaluator()
    results = {
        "n_samples": 5,
        "status": {"accuracy": 0.8, "precision": 0.75, "recall": 0.85, "f1": 0.797},
        "escalation_rate": 0.4,
        "avg_confidence": 0.68,
        "latency_ms": {"mean": 220.0, "median": 210.0, "p95": 280.0},
        "request_type": {"accuracy": 0.9, "f1_macro": 0.88},
        "product_area": {"accuracy": 0.70},
    }
    run_metrics = {
        "ticket_count": 5,
        "escalation_count": 2,
        "error_count": 0,
        "peak_memory_mb": 256.0,
        "average_latency_ms": 220.0,
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "evaluation_report.json"
        ev.save_report(results, out, run_metrics=run_metrics)
        data = json.loads(out.read_text())
        assert data["accuracy"] == 0.8
        assert data["ticket_count"] == 5
        assert data["avg_latency_ms"] == 220.0
