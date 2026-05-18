"""
tests/test_decision_summary.py — Unit tests for build_decision_summary().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.response_generator import build_decision_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(summary: str) -> dict:
    """Parse 'K=V | K=V | …' into a dict."""
    parts = {}
    for token in summary.split(" | "):
        k, _, v = token.partition("=")
        parts[k.strip()] = v.strip()
    return parts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildDecisionSummary:

    def test_returns_string(self):
        result = build_decision_summary("billing_payment", "low", 0.80, "replied")
        assert isinstance(result, str)

    def test_all_fields_present(self):
        result = build_decision_summary("billing_payment", "low", 0.80, "replied")
        for field in ("Intent", "Risk", "Confidence", "Action", "Rule"):
            assert field in result, f"Missing field: {field}"

    def test_intent_field(self):
        result = build_decision_summary("account_access", "low", 0.80, "replied")
        p = _parse(result)
        assert p["Intent"] == "account_access"

    def test_risk_field(self):
        result = build_decision_summary("billing_payment", "high", 0.30, "escalated")
        p = _parse(result)
        assert p["Risk"] == "high"

    def test_confidence_format(self):
        result = build_decision_summary("general_inquiry", "low", 0.753, "replied")
        p = _parse(result)
        # confidence should be 2 decimal places
        assert p["Confidence"] == "0.75"

    def test_action_uppercased(self):
        result = build_decision_summary("billing_payment", "low", 0.80, "replied")
        p = _parse(result)
        assert p["Action"] == "REPLIED"

    def test_action_escalated_uppercased(self):
        result = build_decision_summary("billing_payment", "high", 0.30, "escalated")
        p = _parse(result)
        assert p["Action"] == "ESCALATED"

    def test_rule_injection_detected(self):
        result = build_decision_summary(
            "general_inquiry", "high", 0.10, "escalated",
            is_injection=True,
        )
        p = _parse(result)
        assert p["Rule"] == "InjectionDetected"

    def test_rule_guardrail_violation(self):
        result = build_decision_summary(
            "general_inquiry", "medium", 0.40, "escalated",
            guardrail_safe=False,
        )
        p = _parse(result)
        assert p["Rule"] == "GuardrailViolation"

    def test_rule_high_risk(self):
        result = build_decision_summary(
            "security_concern", "high", 0.50, "escalated",
        )
        p = _parse(result)
        assert p["Rule"] == "HighRiskDetected"

    def test_rule_out_of_scope(self):
        result = build_decision_summary(
            "out_of_scope", "low", 0.70, "escalated",
            is_out_of_scope=True,
        )
        p = _parse(result)
        assert p["Rule"] == "OutOfScope"

    def test_rule_invalid_request_type(self):
        result = build_decision_summary(
            "general_inquiry", "low", 0.70, "escalated",
            request_type="invalid",
        )
        p = _parse(result)
        assert p["Rule"] == "OutOfScope"

    def test_rule_no_corpus_match(self):
        result = build_decision_summary(
            "technical_bug", "low", 0.70, "escalated",
            has_retrieval=False,
        )
        p = _parse(result)
        assert p["Rule"] == "NoCorpusMatch"

    def test_rule_low_confidence(self):
        result = build_decision_summary(
            "billing_payment", "low", 0.40, "escalated",
            conf_threshold=0.65,
        )
        p = _parse(result)
        assert p["Rule"] == "LowConfidence"

    def test_rule_grounded_reply(self):
        result = build_decision_summary(
            "billing_payment", "low", 0.80, "replied",
            has_retrieval=True,
            conf_threshold=0.65,
        )
        p = _parse(result)
        assert p["Rule"] == "GroundedReply"

    def test_injection_takes_priority_over_high_risk(self):
        result = build_decision_summary(
            "general_inquiry", "high", 0.10, "escalated",
            is_injection=True,
            guardrail_safe=False,
        )
        p = _parse(result)
        assert p["Rule"] == "InjectionDetected"

    def test_guardrail_takes_priority_over_high_risk(self):
        result = build_decision_summary(
            "general_inquiry", "high", 0.50, "escalated",
            guardrail_safe=False,
        )
        p = _parse(result)
        assert p["Rule"] == "GuardrailViolation"

    def test_pipe_separated_format(self):
        result = build_decision_summary("billing_payment", "low", 0.80, "replied")
        assert " | " in result
        segments = result.split(" | ")
        assert len(segments) == 5

    def test_confidence_zero(self):
        result = build_decision_summary("general_inquiry", "low", 0.0, "escalated")
        p = _parse(result)
        assert p["Confidence"] == "0.00"

    def test_confidence_one(self):
        result = build_decision_summary("billing_payment", "low", 1.0, "replied")
        p = _parse(result)
        assert p["Confidence"] == "1.00"
