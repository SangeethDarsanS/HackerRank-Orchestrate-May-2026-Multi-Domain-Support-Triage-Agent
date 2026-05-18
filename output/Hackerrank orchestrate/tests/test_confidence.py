"""
Unit tests for modules/confidence.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from modules.confidence import (
    ConfidenceScorer,
    LOW_CONFIDENCE_THRESHOLD,
    _W_RETRIEVAL, _W_CLASSIFIER, _W_RERANKER, _W_GUARDRAIL,
    _TYPE_PROB,
)
from modules.guardrails import GuardrailResult
from modules.retriever import RetrievalResult
from modules.loader import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_guardrail(score: float = 1.0) -> GuardrailResult:
    return GuardrailResult(safe=True, violations=[], safety_score=score)


def _unsafe_guardrail(score: float = 0.6) -> GuardrailResult:
    return GuardrailResult(
        safe=False,
        violations=["sensitive_category:fraud"],
        safety_score=score,
    )


def _make_retrieval(score: float) -> RetrievalResult:
    chunk = Chunk(
        chunk_id="c0", doc_id="doc0", title="T", content="C",
        domain="hackerrank", area="screen", file_path="/f.md", chunk_index=0,
    )
    return RetrievalResult(
        chunks=[(chunk, score)],
        top_docs=[("doc0", "T", score, "screen")],
    )


# ---------------------------------------------------------------------------
# Weights sum check
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    total = _W_RETRIEVAL + _W_CLASSIFIER + _W_RERANKER + _W_GUARDRAIL
    assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------

def test_score_returns_dict_with_required_keys():
    cs = ConfidenceScorer()
    retrieval = _make_retrieval(0.7)
    result = cs.score(retrieval, "bug", 0.6, _safe_guardrail())
    for key in ("confidence", "retrieval_score", "classifier_prob",
                "reranker_score", "guardrail_score", "low_confidence"):
        assert key in result


def test_score_range_0_to_1():
    cs = ConfidenceScorer()
    for ret_s in [0.0, 0.3, 0.7, 1.0]:
        retrieval = _make_retrieval(ret_s)
        r = cs.score(retrieval, "bug", 0.5, _safe_guardrail())
        assert 0.0 <= r["confidence"] <= 1.0


def test_score_exact_formula():
    cs = ConfidenceScorer()
    ret_s  = 0.6
    clf_s  = _TYPE_PROB["bug"]    # 0.80
    rnk_s  = 0.7
    grd_s  = 1.0
    expected = (
        _W_RETRIEVAL * ret_s
        + _W_CLASSIFIER * clf_s
        + _W_RERANKER   * rnk_s
        + _W_GUARDRAIL  * grd_s
    )
    retrieval = _make_retrieval(ret_s)
    result = cs.score(retrieval, "bug", rnk_s, _safe_guardrail(grd_s))
    assert abs(result["confidence"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# None retrieval
# ---------------------------------------------------------------------------

def test_none_retrieval_gives_zero_retrieval_score():
    cs = ConfidenceScorer()
    result = cs.score(None, "product_issue", 0.5, _safe_guardrail())
    assert result["retrieval_score"] == 0.0


# ---------------------------------------------------------------------------
# Low-confidence detection
# ---------------------------------------------------------------------------

def test_low_confidence_flagged_below_threshold():
    cs = ConfidenceScorer()
    # Force a very low composite score
    result = cs.score(None, "invalid", 0.0, _unsafe_guardrail(0.0))
    assert result["low_confidence"] is True
    assert result["confidence"] < LOW_CONFIDENCE_THRESHOLD


def test_high_confidence_not_flagged():
    cs = ConfidenceScorer()
    retrieval = _make_retrieval(0.9)
    result = cs.score(retrieval, "bug", 0.9, _safe_guardrail(1.0))
    if result["confidence"] >= LOW_CONFIDENCE_THRESHOLD:
        assert result["low_confidence"] is False


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

def test_scores_clamped_below_zero():
    cs = ConfidenceScorer()
    retrieval = _make_retrieval(-5.0)   # abnormal FAISS score
    result = cs.score(retrieval, "bug", -1.0, _safe_guardrail(-0.5))
    assert result["confidence"] >= 0.0


def test_scores_clamped_above_one():
    cs = ConfidenceScorer()
    retrieval = _make_retrieval(10.0)   # abnormal FAISS score
    result = cs.score(retrieval, "bug", 5.0, _safe_guardrail(3.0))
    assert result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Type probabilities
# ---------------------------------------------------------------------------

def test_invalid_type_has_lowest_prob():
    assert _TYPE_PROB["invalid"] < _TYPE_PROB["bug"]
    assert _TYPE_PROB["invalid"] < _TYPE_PROB["product_issue"]


def test_unknown_type_defaults_gracefully():
    cs = ConfidenceScorer()
    retrieval = _make_retrieval(0.5)
    result = cs.score(retrieval, "unknown_type", 0.5, _safe_guardrail())
    assert 0.0 <= result["confidence"] <= 1.0
