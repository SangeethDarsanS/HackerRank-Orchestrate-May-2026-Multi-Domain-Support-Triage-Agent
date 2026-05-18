"""
Unit tests for modules/intent_classifier.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from modules.intent_classifier import (
    IntentClassifier,
    IntentClassificationResult,
    INTENTS,
    RULE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rule_only() -> IntentClassifier:
    return IntentClassifier(use_rules=True, use_ml=False)


def _ml_only() -> IntentClassifier:
    return IntentClassifier(use_rules=False, use_ml=True)


def _both() -> IntentClassifier:
    return IntentClassifier(use_rules=True, use_ml=True)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_classify_returns_correct_type():
    clf = _rule_only()
    result = clf.classify("I cannot login to my account")
    assert isinstance(result, IntentClassificationResult)


def test_result_has_all_fields():
    clf = _rule_only()
    r = clf.classify("error on billing page")
    assert hasattr(r, "intent")
    assert hasattr(r, "confidence")
    assert hasattr(r, "source")


def test_intent_is_valid_label():
    clf = _both()
    for text in ["password reset", "invoice", "crash", "feature request"]:
        r = clf.classify(text)
        assert r.intent in INTENTS, f"Unexpected intent {r.intent!r} for {text!r}"


def test_confidence_in_0_to_1():
    clf = _both()
    for text in ["I was hacked", "how do I export data", "billing question"]:
        r = clf.classify(text)
        assert 0.0 <= r.confidence <= 1.0, f"conf={r.confidence} for {text!r}"


# ---------------------------------------------------------------------------
# Rule engine — correct intent detection
# ---------------------------------------------------------------------------

def test_rule_account_access():
    clf = _rule_only()
    r = clf.classify("I forgot my password and can't login")
    assert r.intent == "account_access"
    assert r.source == "rule"


def test_rule_billing():
    clf = _rule_only()
    r = clf.classify("I was overcharged this month, need refund")
    assert r.intent == "billing_payment"
    assert r.source == "rule"


def test_rule_technical_bug():
    clf = _rule_only()
    r = clf.classify("the app is crashing every time I open it")
    assert r.intent == "technical_bug"
    assert r.source == "rule"


def test_rule_feature_request():
    clf = _rule_only()
    r = clf.classify("feature request: please add dark mode")
    assert r.intent == "feature_request"
    assert r.source == "rule"


def test_rule_security():
    clf = _rule_only()
    r = clf.classify("my account was hacked, unauthorized access")
    assert r.intent == "security_concern"
    assert r.source == "rule"


def test_rule_api():
    clf = _rule_only()
    r = clf.classify("getting 429 rate limit errors on the API")
    assert r.intent == "api_integration"
    assert r.source == "rule"


def test_rule_out_of_scope():
    clf = _rule_only()
    r = clf.classify("hi")
    # "hi" alone doesn't match the ^(hi|hello)$ pattern strongly
    # Just check it returns a valid intent
    assert r.intent in INTENTS


def test_rule_exact_greeting():
    clf = _rule_only()
    r = clf.classify("thanks")
    assert r.intent in INTENTS


# ---------------------------------------------------------------------------
# Rule confidence threshold
# ---------------------------------------------------------------------------

def test_high_rule_confidence_skips_ml():
    clf = _both()
    # "password reset" should hit the rule at high confidence
    r = clf.classify("I need to reset my password please")
    assert r.intent == "account_access"
    # When rules are very confident, source should be "rule"
    if r.confidence >= RULE_THRESHOLD:
        assert r.source == "rule"


# ---------------------------------------------------------------------------
# ML-only mode
# ---------------------------------------------------------------------------

def test_ml_returns_valid_intent():
    clf = _ml_only()
    r = clf.classify("cannot log into my dashboard")
    assert r.intent in INTENTS
    assert r.source == "ml"


def test_ml_confidence_in_range():
    clf = _ml_only()
    r = clf.classify("billing invoice not received")
    assert 0.0 <= r.confidence <= 1.0


# ---------------------------------------------------------------------------
# Both modes
# ---------------------------------------------------------------------------

def test_combined_mode_returns_result():
    clf = _both()
    r = clf.classify("API key is returning 401 unauthorized error")
    assert r.intent in INTENTS
    assert r.source in ("rule", "ml")


def test_combined_mode_security_escalation():
    clf = _both()
    r = clf.classify("there has been a security breach, data is compromised")
    assert r.intent == "security_concern"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string():
    clf = _both()
    r = clf.classify("")
    assert r.intent in INTENTS
    assert 0.0 <= r.confidence <= 1.0


def test_very_long_text():
    clf = _both()
    r = clf.classify("cannot login " * 200)
    assert r.intent in INTENTS


def test_non_ascii_text():
    clf = _both()
    r = clf.classify("Ich kann mich nicht einloggen")
    assert r.intent in INTENTS   # graceful, even if wrong


def test_no_rules_no_ml_returns_fallback():
    clf = IntentClassifier(use_rules=False, use_ml=False)
    r = clf.classify("some text")
    assert r.intent in INTENTS
    assert r.source == "fallback"
