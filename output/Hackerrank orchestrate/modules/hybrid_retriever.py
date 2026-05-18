"""
Hybrid retriever: lexical BM25 + semantic embedding fusion.

Strategy
--------
1. Retrieve top_k candidates via BM25Okapi (rank-bm25).
2. Retrieve top_k candidates via the existing FAISS Retriever.
3. Merge, de-duplicate by chunk_id, score with weighted formula.
4. Apply optional domain-aware re-scoring.
5. Return a standard RetrievalResult so downstream code is unchanged.

Scoring
-------
final_score = embedding_weight * embedding_score + bm25_weight * bm25_score_norm

bm25_score_norm is batch-normalised to [0, 1] using max-normalisation
so it is on the same scale as the cosine similarity from FAISS.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .loader import Chunk
from .retriever import Retriever, RetrievalResult
from .config import TOP_K_CHUNKS, TOP_K_DOCS, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _normalise_bm25(scores: List[float]) -> List[float]:
    """Max-normalise BM25 scores to [0, 1]."""
    if not scores:
        return scores
    mx = max(scores)
    if mx < _EPS:
        return [0.0] * len(scores)
    return [s / mx for s in scores]


class HybridRetriever:
    """
    Combines BM25 lexical search and FAISS semantic search.

    Parameters
    ----------
    retriever        : Existing Retriever (wraps VectorIndex).
    chunks           : Full list of Chunk objects — needed for BM25 indexing.
    bm25_weight      : Weight of BM25 score  (default 0.4).
    embedding_weight : Weight of FAISS score (default 0.6).
    """

    def __init__(
        self,
        retriever: Retriever,
        chunks: List[Chunk],
        bm25_weight: float = 0.4,
        embedding_weight: float = 0.6,
    ):
        self.retriever        = retriever
        self.chunks           = chunks
        self.bm25_weight      = bm25_weight
        self.embedding_weight = embedding_weight
        self._bm25            = None   # lazy-built
        # Pre-computed boolean masks for fast per-domain BM25 filtering.
        # Built once at init; key = domain string, value = bool ndarray len(chunks).
        self._domain_masks: Dict[str, "np.ndarray"] = {}

        self._build_bm25_index()
        self._build_domain_masks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        domain_hint: Optional[str] = None,
        top_k: int = TOP_K_CHUNKS,
        k_docs: int = TOP_K_DOCS,
    ) -> RetrievalResult:
        """
        Run hybrid search and return a RetrievalResult compatible with the
        existing pipeline.

        Parameters
        ----------
        query       : Combined ticket text (subject + issue).
        domain_hint : Optional domain filter hint.
        top_k       : Number of chunks to surface.
        k_docs      : Number of unique documents to include in top_docs.
        """
        # -- Step 1: BM25 retrieval (domain-scoped when hint is known) --
        bm25_results = self._bm25_search(query, top_k=top_k, domain_hint=domain_hint)

        # -- Step 2: Embedding retrieval (uses existing Retriever, with retry
        #    handled upstream, so call inner index directly to get raw scores) --
        try:
            embed_raw = self.retriever.index.query(query, k=top_k * 2)
        except Exception as exc:
            logger.warning("[hybrid] FAISS query failed: %s — using BM25 only", exc)
            embed_raw = []

        # Apply domain adjustment to embedding results (mirrors Retriever._adjust)
        if domain_hint and embed_raw:
            embed_raw = self.retriever._adjust_domain_scores(embed_raw, domain_hint)

        # -- Step 3 & 4: Merge, de-duplicate, score --
        merged = self._merge(bm25_results, embed_raw, top_k=top_k)

        # -- Step 5: Apply similarity floor --
        merged = [(c, s) for c, s in merged if s >= SIMILARITY_THRESHOLD]

        # -- Deduplicate to document level --
        top_docs = self._dedup_to_docs(merged, k_docs=k_docs)

        result = RetrievalResult(chunks=merged, top_docs=top_docs)
        logger.debug(
            "[hybrid] query=%r  bm25_hits=%d  embed_hits=%d  merged=%d",
            query[:60], len(bm25_results), len(embed_raw), len(merged),
        )
        return result

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def _build_domain_masks(self) -> None:
        """Pre-compute a boolean numpy mask for each domain.

        These are used in ``_bm25_search`` to zero out cross-domain scores in
        a single vectorised operation instead of a per-element Python loop.
        """
        if not self.chunks:
            return
        domains = {c.domain for c in self.chunks}
        chunk_domains = np.array([c.domain for c in self.chunks])
        for domain in domains:
            self._domain_masks[domain] = (chunk_domains == domain)
        logger.debug("[hybrid] Pre-built domain masks for: %s", sorted(domains))

    def _build_bm25_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError:
            logger.warning(
                "[hybrid] rank-bm25 not installed. BM25 step will be skipped. "
                "Run: pip install rank-bm25"
            )
            return

        corpus = [c.text_for_embedding.lower().split() for c in self.chunks]
        if not corpus:
            logger.warning("[hybrid] No chunks provided — BM25 index will be empty.")
            return
        self._bm25 = BM25Okapi(corpus)
        logger.info("[hybrid] BM25 index built over %d chunks.", len(self.chunks))

    def _bm25_search(
        self, query: str, top_k: int, domain_hint: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """Return top_k (Chunk, normalised_bm25_score) pairs.

        When *domain_hint* is provided, BM25 scores for chunks from other
        domains are zeroed before ranking so term-overlap cannot surface
        cross-domain documents.
        """
        if self._bm25 is None:
            return []

        tokens = query.lower().split()
        raw    = self._bm25.get_scores(tokens)          # ndarray, len == len(chunks)

        # --- Domain filtering: vectorised mask (pre-computed at init) ---
        if domain_hint:
            mask = self._domain_masks.get(domain_hint)
            if mask is not None:
                raw = raw * mask   # zero out non-matching chunks; avoids Python loop

        normed = _normalise_bm25(raw.tolist())

        # Pick top_k indices by normalised score
        indexed = sorted(enumerate(normed), key=lambda x: -x[1])[:top_k * 2]
        results = [(self.chunks[i], s) for i, s in indexed if s > _EPS]
        return results[:top_k]

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def _merge(
        self,
        bm25_results:  List[Tuple[Chunk, float]],
        embed_results: List[Tuple[Chunk, float]],
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """Fuse BM25 and embedding results with weighted combination."""
        combined: dict[str, dict] = {}

        for chunk, score in bm25_results:
            combined[chunk.chunk_id] = {
                "chunk": chunk,
                "bm25":  score,
                "embed": 0.0,
            }

        for chunk, score in embed_results:
            cid = chunk.chunk_id
            if cid in combined:
                combined[cid]["embed"] = max(combined[cid]["embed"], score)
            else:
                combined[cid] = {"chunk": chunk, "bm25": 0.0, "embed": score}

        scored: List[Tuple[Chunk, float]] = []
        for item in combined.values():
            final = (
                self.embedding_weight * item["embed"]
                + self.bm25_weight     * item["bm25"]
            )
            scored.append((item["chunk"], final))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _dedup_to_docs(
        self,
        chunks: List[Tuple[Chunk, float]],
        k_docs: int,
    ) -> List[Tuple[str, str, float, str]]:
        seen: dict[str, Tuple[str, float, str]] = {}
        for chunk, score in chunks:
            if chunk.doc_id not in seen or score > seen[chunk.doc_id][1]:
                seen[chunk.doc_id] = (chunk.title, score, chunk.area)
        top = sorted(seen.items(), key=lambda x: -x[1][1])[:k_docs]
        return [
            (doc_id, title, score, area)
            for doc_id, (title, score, area) in top
        ]
