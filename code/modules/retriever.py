"""
Retrieval module: wraps VectorIndex with domain-aware filtering,
keyword re-ranking, and deduplication across chunks → documents.
"""

import re
from typing import List, Tuple, Optional

from .indexer import VectorIndex
from .loader import Chunk, Document
from .config import (
    TOP_K_CHUNKS, TOP_K_DOCS, SIMILARITY_THRESHOLD
)


class RetrievalResult:
    """Aggregated retrieval result for a single query."""

    def __init__(
        self,
        chunks: List[Tuple[Chunk, float]],
        top_docs: List[Tuple[str, str, float, str]],  # (doc_id, title, score, area)
    ):
        self.chunks = chunks        # raw (Chunk, score) list
        self.top_docs = top_docs    # deduplicated doc-level results

    @property
    def best_score(self) -> float:
        return self.top_docs[0][2] if self.top_docs else 0.0

    @property
    def best_area(self) -> str:
        return self.top_docs[0][3] if self.top_docs else "general_support"

    @property
    def best_domain(self) -> str:
        return self.chunks[0][0].domain if self.chunks else "unknown"

    @property
    def doc_ids(self) -> List[str]:
        return [d[0] for d in self.top_docs]

    def top_chunks(self, n: int = 3) -> List[Chunk]:
        """Return top-n chunks sorted by score."""
        return [c for c, _ in sorted(self.chunks, key=lambda x: -x[1])[:n]]


class Retriever:
    """Semantic + keyword retrieval over the VectorIndex."""

    def __init__(self, index: VectorIndex):
        self.index = index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        domain_hint: Optional[str] = None,
        k_chunks: int = TOP_K_CHUNKS,
        k_docs: int = TOP_K_DOCS,
    ) -> RetrievalResult:
        """
        Run semantic search + optional domain filter + keyword re-rank.

        Args:
            query:       Raw ticket text (issue + subject).
            domain_hint: "hackerrank" | "claude" | "visa" | None.
                         When set, results from other domains are penalised
                         but not removed entirely (cross-domain tickets exist).
            k_chunks:    Number of chunks to retrieve from FAISS.
            k_docs:      Number of unique documents to surface.
        """
        # 1. Fetch more than needed to allow filtering
        raw_chunks = self.index.query(query, k=k_chunks * 2)

        # 2. Domain-aware score adjustment
        if domain_hint:
            raw_chunks = self._adjust_domain_scores(raw_chunks, domain_hint)

        # 3. Keyword re-rank: boost chunks that contain query terms
        raw_chunks = self._keyword_rerank(query, raw_chunks)

        # 4. Filter by minimum similarity
        raw_chunks = [
            (c, s) for c, s in raw_chunks if s >= SIMILARITY_THRESHOLD
        ]

        # 5. Keep top k_chunks
        raw_chunks = raw_chunks[:k_chunks]

        # 6. Deduplicate to document level
        top_docs = self._dedup_to_docs(raw_chunks, k_docs)

        return RetrievalResult(chunks=raw_chunks, top_docs=top_docs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adjust_domain_scores(
        self,
        chunks: List[Tuple[Chunk, float]],
        domain: str,
    ) -> List[Tuple[Chunk, float]]:
        adjusted = []
        for chunk, score in chunks:
            if chunk.domain == domain:
                adjusted.append((chunk, score * 1.25))   # boost
            else:
                adjusted.append((chunk, score * 0.70))   # penalise
        # Re-sort after adjustment
        adjusted.sort(key=lambda x: -x[1])
        return adjusted

    def _keyword_rerank(
        self,
        query: str,
        chunks: List[Tuple[Chunk, float]],
    ) -> List[Tuple[Chunk, float]]:
        query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        boosted = []
        for chunk, score in chunks:
            chunk_terms = set(re.findall(r'\b\w{3,}\b', chunk.content.lower()))
            overlap = len(query_terms & chunk_terms)
            # Up to +0.15 boost for keyword overlap
            keyword_boost = min(overlap / max(len(query_terms), 1) * 0.15, 0.15)
            boosted.append((chunk, score + keyword_boost))
        boosted.sort(key=lambda x: -x[1])
        return boosted

    def _dedup_to_docs(
        self,
        chunks: List[Tuple[Chunk, float]],
        k_docs: int,
    ) -> List[Tuple[str, str, float, str]]:
        """Aggregate chunk-level results into unique document summaries."""
        seen: dict[str, Tuple[str, float, str]] = {}
        for chunk, score in chunks:
            if chunk.doc_id not in seen or score > seen[chunk.doc_id][1]:
                seen[chunk.doc_id] = (chunk.title, score, chunk.area)

        top = sorted(seen.items(), key=lambda x: -x[1][1])[:k_docs]
        return [(doc_id, title, score, area) for doc_id, (title, score, area) in top]
