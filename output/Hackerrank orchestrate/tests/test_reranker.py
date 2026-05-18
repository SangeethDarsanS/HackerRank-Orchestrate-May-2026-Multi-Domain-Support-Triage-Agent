"""
Unit tests for modules/reranker.py.

Tests run without downloading any model (disabled=True or patched predict).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from modules.reranker import Reranker, _sigmoid, _NEUTRAL_SCORE
from modules.loader import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, content: str = "sample content") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_0",
        doc_id=doc_id,
        title="Test Document",
        content=content,
        domain="hackerrank",
        area="screen",
        file_path="/fake/path.md",
        chunk_index=0,
    )


# ---------------------------------------------------------------------------
# Unit tests — _sigmoid
# ---------------------------------------------------------------------------

def test_sigmoid_zero():
    assert abs(_sigmoid(0.0) - 0.5) < 1e-6


def test_sigmoid_large_positive():
    assert _sigmoid(100.0) > 0.999


def test_sigmoid_large_negative():
    assert _sigmoid(-100.0) < 0.001


def test_sigmoid_output_range():
    for x in [-5, -1, 0, 1, 5]:
        s = _sigmoid(x)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Unit tests — disabled reranker
# ---------------------------------------------------------------------------

def test_disabled_reranker_returns_original_order():
    r = Reranker(enabled=False)
    chunks = [(_make_chunk(f"doc{i}"), float(i) * 0.1) for i in range(5)]
    result = r.rank("some query", chunks, top_n=3)
    assert len(result) == 3
    for idx, (c, ret_s, rnk_s) in enumerate(result):
        assert rnk_s == _NEUTRAL_SCORE
        assert c.doc_id == f"doc{idx}"


def test_disabled_reranker_empty_input():
    r = Reranker(enabled=False)
    result = r.rank("query", [], top_n=3)
    assert result == []


def test_disabled_reranker_top_score():
    r = Reranker(enabled=False)
    chunks = [(_make_chunk("a"), 0.8), (_make_chunk("b"), 0.6)]
    score = r.top_reranker_score("query", chunks)
    assert score == _NEUTRAL_SCORE


# ---------------------------------------------------------------------------
# Unit tests — enabled reranker with patched predict
# ---------------------------------------------------------------------------

def _make_reranker_with_mock(raw_scores):
    """Create an enabled Reranker with a mocked CrossEncoder."""
    r = Reranker(enabled=True)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(raw_scores, dtype=float)
    r._model = mock_model
    return r


def test_rank_sorts_by_reranker_score():
    chunks = [
        (_make_chunk("low"),  0.5),
        (_make_chunk("high"), 0.3),
        (_make_chunk("mid"),  0.4),
    ]
    # raw logits: low=1, high=10, mid=5  → sigmoid → high > mid > low
    r = _make_reranker_with_mock([1.0, 10.0, 5.0])
    result = r.rank("query", chunks, top_n=3)
    assert result[0][0].doc_id == "high"
    assert result[1][0].doc_id == "mid"
    assert result[2][0].doc_id == "low"


def test_rank_returns_top_n():
    chunks = [(_make_chunk(f"d{i}"), 0.5) for i in range(10)]
    r = _make_reranker_with_mock([float(i) for i in range(10)])
    result = r.rank("q", chunks, top_n=3)
    assert len(result) == 3


def test_rank_scores_in_0_1():
    chunks = [(_make_chunk("a"), 0.9), (_make_chunk("b"), 0.1)]
    r = _make_reranker_with_mock([-50.0, 50.0])  # extreme logits
    result = r.rank("q", chunks, top_n=2)
    for _, _, s in result:
        assert 0.0 <= s <= 1.0


def test_top_reranker_score_is_highest():
    chunks = [(_make_chunk("a"), 0.9), (_make_chunk("b"), 0.1)]
    r = _make_reranker_with_mock([0.0, 5.0])
    score = r.top_reranker_score("q", chunks, top_n=2)
    assert score == _sigmoid(5.0)


# ---------------------------------------------------------------------------
# Failure / edge-case tests
# ---------------------------------------------------------------------------

def test_rank_predict_failure_falls_back():
    r = Reranker(enabled=True)
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("GPU OOM")
    r._model = mock_model
    chunks = [(_make_chunk("a"), 0.8), (_make_chunk("b"), 0.6)]
    result = r.rank("q", chunks, top_n=2)
    # Must not raise; must return neutral scores
    assert len(result) == 2
    for _, _, s in result:
        assert s == _NEUTRAL_SCORE


def test_rank_empty_chunks_always_returns_empty():
    r = Reranker(enabled=True)
    mock_model = MagicMock()
    r._model = mock_model
    result = r.rank("query", [], top_n=3)
    assert result == []
    mock_model.predict.assert_not_called()


def test_model_load_failure_returns_neutral():
    r = Reranker(enabled=True, model_name="nonexistent/model-xyz")
    # _load_model should catch the error and set _model=None
    model = r._load_model()
    assert model is None
    # rank() should still work
    chunks = [(_make_chunk("a"), 0.5)]
    result = r.rank("q", chunks, top_n=1)
    assert result[0][2] == _NEUTRAL_SCORE
