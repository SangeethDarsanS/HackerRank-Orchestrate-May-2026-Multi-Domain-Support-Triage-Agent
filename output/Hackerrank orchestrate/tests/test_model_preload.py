"""
Unit tests for W002 fix: preload_models() in main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import MagicMock, patch, call
from modules.reranker import Reranker
from modules.intent_classifier import IntentClassifier


# ---------------------------------------------------------------------------
# preload_models function exists and is importable
# ---------------------------------------------------------------------------

def test_preload_models_importable():
    from main import preload_models
    assert callable(preload_models)


# ---------------------------------------------------------------------------
# preload_models triggers reranker._load_model()
# ---------------------------------------------------------------------------

def test_preload_calls_reranker_load_model():
    from main import preload_models
    mock_reranker = MagicMock()
    mock_intent = MagicMock()
    preload_models(mock_reranker, mock_intent)
    mock_reranker._load_model.assert_called_once()


def test_preload_calls_intent_classify():
    from main import preload_models
    mock_reranker = MagicMock()
    mock_intent = MagicMock()
    preload_models(mock_reranker, mock_intent)
    mock_intent.classify.assert_called_once_with("warmup")


# ---------------------------------------------------------------------------
# preload_models handles None gracefully
# ---------------------------------------------------------------------------

def test_preload_none_reranker_no_crash():
    from main import preload_models
    mock_intent = MagicMock()
    preload_models(None, mock_intent)   # should not raise


def test_preload_none_intent_no_crash():
    from main import preload_models
    mock_reranker = MagicMock()
    preload_models(mock_reranker, None)  # should not raise


def test_preload_both_none_no_crash():
    from main import preload_models
    preload_models(None, None)          # should not raise


# ---------------------------------------------------------------------------
# preload_models handles exceptions in classify gracefully
# ---------------------------------------------------------------------------

def test_preload_classify_exception_does_not_propagate():
    from main import preload_models
    mock_reranker = MagicMock()
    mock_intent = MagicMock()
    mock_intent.classify.side_effect = RuntimeError("model not ready")
    preload_models(mock_reranker, mock_intent)  # should not raise


# ---------------------------------------------------------------------------
# Reranker._load_model is called only once per instance
# ---------------------------------------------------------------------------

def test_reranker_load_model_idempotent():
    """Calling _load_model twice on a disabled reranker does not crash."""
    r = Reranker(enabled=False)
    r._load_model()
    r._load_model()   # second call must be safe
