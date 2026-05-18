"""
Cross-encoder document reranker.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to re-score retrieved chunks by
relevance to the query.  Lazy-loads the model on first call so startup time
is unaffected when the reranker is disabled.

Workflow (per ticket):
    retrieved = retriever.retrieve(query, top_k=10)
    reranked  = reranker.rank(query, retrieved.chunks, top_n=3)
    # → List[(Chunk, retrieval_score, reranker_score)] sorted by reranker_score
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

from .loader import Chunk

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Max chars per document fed to the cross-encoder (avoid token overflow)
_MAX_DOC_CHARS = 512
# Score assigned when reranker is disabled or model unavailable
_NEUTRAL_SCORE = 0.5


def _sigmoid(x: float) -> float:
    """Map raw cross-encoder logit → [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-float(x))))


class Reranker:
    """
    Cross-encoder reranker.

    Parameters
    ----------
    enabled : bool
        When False, ``rank()`` returns the original retrieval order with a
        neutral reranker score (0.5) — useful for ablation or offline testing.
    model_name : str
        HuggingFace model identifier for the cross-encoder.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = _MODEL_NAME,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        query: str,
        chunks: List[Tuple[Chunk, float]],
        top_n: int = 3,
        input_limit: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ) -> List[Tuple[Chunk, float, float]]:
        """
        Rerank ``chunks`` by relevance to ``query``.

        Parameters
        ----------
        query       : ticket text (issue + subject).
        chunks      : list of (Chunk, retrieval_score) from FAISS.
        top_n       : how many top results to return.
        input_limit : cap the number of chunks fed to the cross-encoder.
                      Reduces latency when retrieval top_k is large.
                      e.g. input_limit=5 with top_k=10 halves encoder work.
        timeout_ms  : abort predict() after this many milliseconds and return
                      neutral scores.  Safe for low-latency budgets.

        Returns
        -------
        List of (Chunk, retrieval_score, reranker_score) sorted by
        reranker_score descending.  reranker_score is in [0, 1].
        """
        if not chunks:
            return []

        if not self.enabled:
            result = [(c, s, _NEUTRAL_SCORE) for c, s in chunks[:top_n]]
            logger.debug("[reranker] disabled — returning original top-%d", top_n)
            return result

        # --- Apply input cap before sending to cross-encoder ---
        if input_limit is not None and len(chunks) > input_limit:
            chunks = chunks[:input_limit]

        model = self._load_model()
        if model is None:
            # Graceful fallback if model unavailable
            return [(c, s, _NEUTRAL_SCORE) for c, s in chunks[:top_n]]

        # Build query-document pairs
        pairs = [(query, c.content[:_MAX_DOC_CHARS]) for c, _ in chunks]

        try:
            if timeout_ms:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
                _exec = ThreadPoolExecutor(max_workers=1)
                _f = _exec.submit(model.predict, pairs)
                try:
                    raw_scores = _f.result(timeout=timeout_ms / 1000.0)
                    _exec.shutdown(wait=False)
                except _FutureTimeout:
                    logger.warning(
                        "[reranker] predict() timed out after %dms — using retrieval order",
                        timeout_ms,
                    )
                    _exec.shutdown(wait=False)
                    return [(c, s, _NEUTRAL_SCORE) for c, s in chunks[:top_n]]
            else:
                raw_scores = model.predict(pairs)
        except Exception as exc:
            logger.warning(
                "[reranker] predict() failed (%s) — falling back to original order", exc
            )
            return [(c, s, _NEUTRAL_SCORE) for c, s in chunks[:top_n]]

        # Normalise raw logits → [0, 1] via sigmoid
        scored: List[Tuple[Chunk, float, float]] = [
            (chunk, ret_score, _sigmoid(float(raw)))
            for (chunk, ret_score), raw in zip(chunks, raw_scores)
        ]

        # Sort descending by reranker score
        scored.sort(key=lambda x: -x[2])

        top = scored[:top_n]
        if top:
            logger.debug(
                "[reranker] top scores: %s",
                [(c.doc_id[:8], f"{r:.3f}") for c, _, r in top],
            )
        return top

    def top_reranker_score(
        self,
        query: str,
        chunks: List[Tuple[Chunk, float]],
        top_n: int = 3,
        input_limit: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ) -> float:
        """
        Convenience wrapper — returns only the highest reranker score (0–1).
        Safe to call with an empty chunk list (returns 0.0).
        """
        ranked = self.rank(query, chunks, top_n=top_n,
                           input_limit=input_limit, timeout_ms=timeout_ms)
        return ranked[0][2] if ranked else 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load the CrossEncoder model; return None on failure."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            logger.info("[reranker] Loading cross-encoder '%s' …", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("[reranker] Model loaded.")
        except Exception as exc:
            logger.warning(
                "[reranker] Could not load model '%s' (%s). "
                "Reranker will return neutral scores.",
                self.model_name,
                exc,
            )
            self._model = None
        return self._model
