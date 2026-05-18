"""
Unit tests for modules/policy_enforcer.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from modules.policy_enforcer import PolicyEnforcer, PolicyResult
from modules.retriever import RetrievalResult
from modules.loader import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(doc_id: str = "abc123") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_0", doc_id=doc_id,
        title="Test Doc", content="Policy-relevant support content.",
        domain="hackerrank", area="screen", file_path="/f.md", chunk_index=0,
    )


def _retrieval(score: float = 0.7, doc_id: str = "abc123") -> RetrievalResult:
    c = _chunk(doc_id)
    return RetrievalResult(
        chunks=[(c, score)],
        top_docs=[(doc_id, "Test Doc", score, "screen")],
    )


_PLAIN_RESPONSE = "Hi,\n\nThe issue is that your account needs to be verified."
_CITED_RESPONSE = "Hi,\n\nSee details below.\n\nSource: abc123"
_ESCALATION    = "Thank you for contacting support.\n\nYour request requires assistance from a human support specialist."


# ---------------------------------------------------------------------------
# validate() — basic rules
# ---------------------------------------------------------------------------

def test_escalation_always_passes():
    pe = PolicyEnforcer(require_citation=True)
    res = pe.validate(_ESCALATION, retrieval=None, is_escalation=True)
    assert res.passed is True
    assert res.escalate is False


def test_existing_citation_passes():
    pe = PolicyEnforcer()
    res = pe.validate(_CITED_RESPONSE, retrieval=_retrieval())
    assert res.passed is True
    assert res.has_citation is True


def test_no_citation_good_retrieval_passes():
    pe = PolicyEnforcer()
    res = pe.validate(_PLAIN_RESPONSE, retrieval=_retrieval(score=0.7))
    assert res.passed is True


def test_no_citation_no_retrieval_fails_strict():
    pe = PolicyEnforcer(require_citation=True)
    res = pe.validate(_PLAIN_RESPONSE, retrieval=None)
    assert res.passed is False
    assert res.escalate is True


def test_no_citation_no_retrieval_passes_non_strict():
    pe = PolicyEnforcer(require_citation=False)
    res = pe.validate(_PLAIN_RESPONSE, retrieval=None)
    assert res.passed is True


def test_no_citation_low_score_fails():
    pe = PolicyEnforcer(require_citation=True)
    res = pe.validate(_PLAIN_RESPONSE, retrieval=_retrieval(score=0.05))
    assert res.passed is False
    assert res.escalate is True


# ---------------------------------------------------------------------------
# Citation detection patterns
# ---------------------------------------------------------------------------

def test_source_pattern_detected():
    pe = PolicyEnforcer()
    res = pe.validate("Some text.\n\nSource: HR-0001", retrieval=None)
    assert res.has_citation is True


def test_reference_pattern_detected():
    pe = PolicyEnforcer()
    res = pe.validate("For more info, Reference: doc_42", retrieval=None)
    assert res.has_citation is True


def test_see_also_pattern_detected():
    pe = PolicyEnforcer()
    res = pe.validate("See also: documentation page", retrieval=None)
    assert res.has_citation is True


# ---------------------------------------------------------------------------
# enforce() — citation enrichment
# ---------------------------------------------------------------------------

def test_enforce_adds_citation_when_missing():
    pe = PolicyEnforcer()
    enriched = pe.enforce(_PLAIN_RESPONSE, retrieval=_retrieval(doc_id="xyz789"))
    assert "Source: xyz789" in enriched


def test_enforce_does_not_duplicate_citation():
    pe = PolicyEnforcer()
    enriched = pe.enforce(_CITED_RESPONSE, retrieval=_retrieval())
    assert enriched.count("Source:") == 1


def test_enforce_escalation_unchanged():
    pe = PolicyEnforcer()
    enriched = pe.enforce(_ESCALATION, retrieval=_retrieval(), is_escalation=True)
    assert enriched == _ESCALATION


def test_enforce_no_retrieval_returns_original():
    pe = PolicyEnforcer()
    enriched = pe.enforce(_PLAIN_RESPONSE, retrieval=None)
    assert enriched == _PLAIN_RESPONSE


# ---------------------------------------------------------------------------
# validate_and_enforce()
# ---------------------------------------------------------------------------

def test_validate_and_enforce_adds_citation():
    pe = PolicyEnforcer()
    enriched, result = pe.validate_and_enforce(
        _PLAIN_RESPONSE, retrieval=_retrieval(doc_id="docA"), is_escalation=False
    )
    assert "Source: docA" in enriched
    assert result.added_citation is True


def test_validate_and_enforce_escalation_exempt():
    pe = PolicyEnforcer()
    enriched, result = pe.validate_and_enforce(
        _ESCALATION, retrieval=None, is_escalation=True
    )
    assert enriched == _ESCALATION
    assert result.passed is True


def test_validate_and_enforce_no_docs_escalate_flag():
    pe = PolicyEnforcer(require_citation=True)
    enriched, result = pe.validate_and_enforce(
        _PLAIN_RESPONSE, retrieval=None, is_escalation=False
    )
    assert result.escalate is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_response():
    pe = PolicyEnforcer()
    res = pe.validate("", retrieval=_retrieval())
    assert isinstance(res, PolicyResult)


def test_empty_response_enforce():
    pe = PolicyEnforcer()
    enriched = pe.enforce("", retrieval=_retrieval(doc_id="d0"))
    assert "Source: d0" in enriched


def test_very_long_response():
    pe = PolicyEnforcer()
    long_resp = "This is a detailed support response. " * 300
    enriched = pe.enforce(long_resp, retrieval=_retrieval(doc_id="longdoc"))
    assert "Source: longdoc" in enriched
