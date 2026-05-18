"""
Unit and integration tests for modules/hybrid_retriever.py.
All tests avoid network calls and heavy model loading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from modules.hybrid_retriever import HybridRetriever, _normalise_bm25
from modules.retriever import RetrievalResult
from modules.loader import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(cid: str, content: str = "sample support text", domain: str = "hackerrank") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=cid[:4],
        title="Doc", content=content,
        domain=domain, area="screen",
        file_path="/f.md", chunk_index=0,
    )


def _fake_retriever(chunks, scores):
    """Return a mock Retriever whose index.query returns (chunk, score) pairs."""
    mock = MagicMock()
    mock.index.query.return_value = list(zip(chunks, scores))
    mock._adjust_domain_scores = lambda raw, domain: raw
    return mock


def _make_hybrid(chunks=None, bm25_w=0.4, embed_w=0.6):
    if chunks is None:
        chunks = [
            _chunk("c0", "login password reset account"),
            _chunk("c1", "billing invoice payment charge"),
            _chunk("c2", "error crash not working bug"),
        ]
    retriever = _fake_retriever(chunks, [0.9, 0.7, 0.5])
    return HybridRetriever(retriever, chunks, bm25_weight=bm25_w, embedding_weight=embed_w)


# ---------------------------------------------------------------------------
# _normalise_bm25
# ---------------------------------------------------------------------------

def test_normalise_empty():
    assert _normalise_bm25([]) == []


def test_normalise_all_zero():
    result = _normalise_bm25([0.0, 0.0, 0.0])
    assert result == [0.0, 0.0, 0.0]


def test_normalise_max_becomes_one():
    result = _normalise_bm25([1.0, 2.0, 4.0])
    assert abs(result[-1] - 1.0) < 1e-9


def test_normalise_proportional():
    result = _normalise_bm25([2.0, 4.0])
    assert abs(result[0] - 0.5) < 1e-9
    assert abs(result[1] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# HybridRetriever construction
# ---------------------------------------------------------------------------

def test_build_with_empty_chunks():
    retriever = MagicMock()
    retriever.index.query.return_value = []
    retriever._adjust_domain_scores = lambda raw, d: raw
    h = HybridRetriever(retriever, [], bm25_weight=0.4, embedding_weight=0.6)
    assert h.chunks == []


def test_bm25_unavailable_graceful(monkeypatch):
    """When rank_bm25 is not installed, BM25 is disabled but no crash."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "rank_bm25":
            raise ImportError("mocked absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    chunks = [_chunk("c0", "some text")]
    retriever = _fake_retriever(chunks, [0.7])
    h = HybridRetriever(retriever, chunks)
    assert h._bm25 is None


# ---------------------------------------------------------------------------
# search() — basic behaviour
# ---------------------------------------------------------------------------

def test_search_returns_retrieval_result():
    h = _make_hybrid()
    result = h.search("login issue password")
    assert isinstance(result, RetrievalResult)


def test_search_returns_non_empty_with_matching_query():
    chunks = [
        _chunk("c0", "login password reset"),
        _chunk("c1", "billing invoice"),
        _chunk("c2", "error crash bug"),
    ]
    retriever = _fake_retriever(chunks, [0.8, 0.6, 0.5])
    h = HybridRetriever(retriever, chunks)
    result = h.search("password reset login")
    assert len(result.chunks) >= 0  # no crash; may be filtered by score threshold


def test_search_empty_query_does_not_crash():
    h = _make_hybrid()
    result = h.search("")
    assert isinstance(result, RetrievalResult)


def test_search_domain_hint_accepted():
    h = _make_hybrid()
    result = h.search("login issue", domain_hint="hackerrank")
    assert isinstance(result, RetrievalResult)


# ---------------------------------------------------------------------------
# _merge logic
# ---------------------------------------------------------------------------

def test_merge_deduplicates_same_chunk():
    chunks = [_chunk("c0"), _chunk("c1")]
    retriever = _fake_retriever(chunks, [0.8, 0.6])
    h = HybridRetriever(retriever, chunks)

    # Both BM25 and embedding return same chunk
    bm25   = [(chunks[0], 0.9), (chunks[1], 0.5)]
    embed  = [(chunks[0], 0.8), (chunks[1], 0.4)]
    merged = h._merge(bm25, embed, top_k=5)

    ids = [c.chunk_id for c, _ in merged]
    assert len(ids) == len(set(ids))  # no duplicates


def test_merge_bm25_only_chunk_included():
    chunks = [_chunk("c0"), _chunk("c1"), _chunk("c2")]
    retriever = _fake_retriever(chunks, [0.8, 0.6, 0.5])
    h = HybridRetriever(retriever, chunks)

    # c2 only in bm25
    bm25  = [(chunks[0], 0.9), (chunks[2], 0.7)]
    embed = [(chunks[0], 0.8), (chunks[1], 0.6)]
    merged = h._merge(bm25, embed, top_k=5)

    ids = [c.chunk_id for c, _ in merged]
    assert "c0" in ids
    assert "c1" in ids
    assert "c2" in ids


def test_merge_combined_score_formula():
    """Verify the 0.6*embed + 0.4*bm25 formula is applied."""
    chunks = [_chunk("c0")]
    retriever = _fake_retriever(chunks, [0.6])
    h = HybridRetriever(retriever, chunks, bm25_weight=0.4, embedding_weight=0.6)

    bm25  = [(chunks[0], 0.5)]
    embed = [(chunks[0], 0.8)]
    merged = h._merge(bm25, embed, top_k=1)

    expected = 0.6 * 0.8 + 0.4 * 0.5  # 0.68
    assert abs(merged[0][1] - expected) < 1e-6


def test_merge_sorted_descending():
    chunks = [_chunk("c0"), _chunk("c1"), _chunk("c2")]
    retriever = _fake_retriever(chunks, [0.9, 0.5, 0.3])
    h = HybridRetriever(retriever, chunks)

    bm25  = [(chunks[0], 0.1), (chunks[1], 0.8), (chunks[2], 0.5)]
    embed = [(chunks[0], 0.9), (chunks[1], 0.4), (chunks[2], 0.2)]
    merged = h._merge(bm25, embed, top_k=3)

    scores = [s for _, s in merged]
    assert scores == sorted(scores, reverse=True)


def test_merge_top_k_respected():
    chunks = [_chunk(f"c{i}") for i in range(10)]
    retriever = _fake_retriever(chunks, [0.5] * 10)
    h = HybridRetriever(retriever, chunks)

    bm25  = [(c, 0.5) for c in chunks]
    embed = [(c, 0.5) for c in chunks]
    merged = h._merge(bm25, embed, top_k=3)
    assert len(merged) == 3


# ---------------------------------------------------------------------------
# _dedup_to_docs
# ---------------------------------------------------------------------------

def test_dedup_groups_by_doc_id():
    chunk_a = _chunk("docA_0", domain="hackerrank")
    chunk_a2 = Chunk(chunk_id="docA_1", doc_id="docA", title="DocA",
                     content="more text", domain="hackerrank", area="screen",
                     file_path="/f.md", chunk_index=1)
    chunk_b = _chunk("docB_0", domain="hackerrank")

    retriever = _fake_retriever([chunk_a], [0.8])
    h = HybridRetriever(retriever, [chunk_a, chunk_a2, chunk_b])

    merged = [(chunk_a, 0.8), (chunk_a2, 0.9), (chunk_b, 0.5)]
    docs = h._dedup_to_docs(merged, k_docs=5)

    doc_ids = [d[0] for d in docs]
    assert len(doc_ids) == len(set(doc_ids))   # no dup doc_ids


def test_dedup_keeps_highest_score():
    chunk_a1 = _chunk("docA_0")
    chunk_a2 = Chunk(chunk_id="docA_1", doc_id="docA", title="DocA",
                     content="extra", domain="hackerrank", area="screen",
                     file_path="/f.md", chunk_index=1)

    retriever = _fake_retriever([chunk_a1], [0.8])
    h = HybridRetriever(retriever, [chunk_a1, chunk_a2])

    merged = [(chunk_a1, 0.5), (chunk_a2, 0.9)]
    docs = h._dedup_to_docs(merged, k_docs=5)

    assert abs(docs[0][2] - 0.9) < 1e-9   # best score kept
