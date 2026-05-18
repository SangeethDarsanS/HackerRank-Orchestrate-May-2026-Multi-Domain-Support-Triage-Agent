"""
Unit tests for W004 fix: enhanced ProductAreaResolver
(rule boosting + ML classifier + confidence threshold).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from modules.classifier import ProductAreaResolver, _PA_ML_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolver() -> ProductAreaResolver:
    return ProductAreaResolver()


def _mock_retrieval(area: str = "general_support", domain: str = "hackerrank",
                    score: float = 0.5):
    from modules.loader import Chunk
    from modules.retriever import RetrievalResult
    chunk = Chunk(
        chunk_id="c0", doc_id="d0", title="Doc",
        content="content", domain=domain, area=area,
        file_path="/f.md", chunk_index=0,
    )
    return RetrievalResult(
        chunks=[(chunk, score)],
        top_docs=[("d0", "Doc", score, area)],
    )


# ---------------------------------------------------------------------------
# Rule boosting — high-priority keywords
# ---------------------------------------------------------------------------

def test_rule_boost_mock_interview_returns_skillup():
    r = _resolver()
    area, conf = r._rule_boost("my mock interview stopped midway")
    assert area == "skillup"
    assert conf >= 0.85


def test_rule_boost_inactivity_returns_interviews():
    r = _resolver()
    area, conf = r._rule_boost("candidate was kicked due to inactivity")
    assert area == "interviews"


def test_rule_boost_assessment_returns_screen():
    r = _resolver()
    area, conf = r._rule_boost("hackerrank assessment link expired")
    assert area == "screen"


def test_rule_boost_bedrock_returns_amazon_bedrock():
    r = _resolver()
    area, conf = r._rule_boost("requests to claude with aws bedrock is failing")
    assert area == "amazon_bedrock"


def test_rule_boost_lti_returns_education():
    r = _resolver()
    area, conf = r._rule_boost("setup a claude lti key for my students")
    assert area == "education"


def test_rule_boost_bug_bounty_returns_safeguards():
    r = _resolver()
    area, conf = r._rule_boost("I found a security vulnerability in claude bug bounty")
    assert area == "safeguards"


def test_rule_boost_crawl_returns_privacy():
    r = _resolver()
    area, conf = r._rule_boost("please stop crawling my website")
    assert area == "privacy"


def test_rule_boost_chargeback_returns_dispute_resolution():
    r = _resolver()
    area, conf = r._rule_boost("I want to initiate a chargeback for a visa transaction")
    assert area == "dispute_resolution"


def test_rule_boost_card_stolen_returns_consumer_support():
    r = _resolver()
    area, conf = r._rule_boost("my visa card stolen help me block it")
    assert area == "consumer_support"


def test_rule_boost_no_match_returns_none():
    r = _resolver()
    area, conf = r._rule_boost("general inquiry about something")
    assert area is None
    assert conf == 0.0


# ---------------------------------------------------------------------------
# Rule boost takes priority over retrieval
# ---------------------------------------------------------------------------

def test_rule_boost_overrides_wrong_retrieval_area():
    """If retrieval says 'integrations' but text clearly says 'assessment',
    rule boost should win."""
    r = _resolver()
    retrieval = _mock_retrieval(area="integrations", domain="hackerrank", score=0.9)
    result = r.resolve(retrieval, "hackerrank", "reschedule my assessment")
    assert result == "screen"


def test_rule_boost_overrides_bedrock_retrieval():
    r = _resolver()
    retrieval = _mock_retrieval(area="account_management", domain="claude", score=0.8)
    result = r.resolve(retrieval, "claude", "aws bedrock requests failing")
    assert result == "amazon_bedrock"


# ---------------------------------------------------------------------------
# ML classifier
# ---------------------------------------------------------------------------

def test_ml_model_loaded_or_none():
    """ML model should be loaded (training CSV present) or None — never crash."""
    r = _resolver()
    assert r._ml_model is None or isinstance(r._ml_model, tuple)


def test_ml_predict_returns_tuple():
    r = _resolver()
    if r._ml_model is None:
        pytest.skip("ML model not available")
    area, conf = r._ml_predict("mock interview stopped midway")
    assert isinstance(area, str)
    assert 0.0 <= conf <= 1.0


def test_ml_predict_bedrock_text():
    r = _resolver()
    if r._ml_model is None:
        pytest.skip("ML model not available")
    area, conf = r._ml_predict("aws bedrock requests to claude failing")
    assert area == "amazon_bedrock"


def test_ml_predict_privacy_text():
    r = _resolver()
    if r._ml_model is None:
        pytest.skip("ML model not available")
    area, conf = r._ml_predict("delete my claude conversation history")
    assert area == "privacy"


def test_ml_predict_education_text():
    r = _resolver()
    if r._ml_model is None:
        pytest.skip("ML model not available")
    area, conf = r._ml_predict("professor setting up claude lti key for students")
    assert area == "education"


# ---------------------------------------------------------------------------
# Confidence threshold — low-confidence ML falls back to next signal
# ---------------------------------------------------------------------------

def test_low_confidence_ml_uses_retrieval_fallback():
    """When ML confidence is below threshold, retrieval-based area is used."""
    r = _resolver()
    if r._ml_model is None:
        pytest.skip("ML model not available")

    # Patch _ml_predict to return low confidence
    original = r._ml_predict

    def low_conf_predict(text):
        area, _ = original(text)
        return area, 0.10   # force below threshold

    r._ml_predict = low_conf_predict

    retrieval = _mock_retrieval(area="privacy", domain="claude", score=0.8)
    result = r.resolve(retrieval, "claude", "some vague text")
    assert result == "privacy"


# ---------------------------------------------------------------------------
# End-to-end resolve() — spot checks for key tickets
# ---------------------------------------------------------------------------

def test_resolve_mock_interview_ticket():
    r = _resolver()
    result = r.resolve(None, "hackerrank", "my mock interviews stopped midway refund")
    assert result == "skillup"


def test_resolve_bedrock_ticket():
    r = _resolver()
    result = r.resolve(None, "claude", "all requests to claude with aws bedrock failing")
    assert result == "amazon_bedrock"


def test_resolve_education_ticket():
    r = _resolver()
    result = r.resolve(None, "claude", "professor setup claude lti key for students")
    assert result == "education"


def test_resolve_dispute_ticket():
    r = _resolver()
    result = r.resolve(None, "visa", "how do I dispute a charge on my visa card chargeback")
    assert result == "dispute_resolution"


def test_resolve_no_retrieval_no_crash():
    r = _resolver()
    result = r.resolve(None, "hackerrank", "assessment link not working")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Training data file exists and has required columns
# ---------------------------------------------------------------------------

def test_training_csv_exists():
    training_path = Path(__file__).resolve().parent.parent.parent / \
                    "data" / "product_area_training.csv"
    assert training_path.exists(), f"product_area_training.csv not found at {training_path}"


def test_training_csv_has_required_columns():
    import pandas as pd
    training_path = Path(__file__).resolve().parent.parent.parent / \
                    "data" / "product_area_training.csv"
    if training_path.exists():
        df = pd.read_csv(training_path, dtype=str)
        cols = [c.strip().lower() for c in df.columns]
        assert "text" in cols
        assert "product_area" in cols


def test_training_csv_sufficient_classes():
    import pandas as pd
    training_path = Path(__file__).resolve().parent.parent.parent / \
                    "data" / "product_area_training.csv"
    if training_path.exists():
        df = pd.read_csv(training_path, dtype=str).fillna("")
        n_classes = df["product_area"].nunique()
        assert n_classes >= 10, f"Expected ≥10 classes, got {n_classes}"
