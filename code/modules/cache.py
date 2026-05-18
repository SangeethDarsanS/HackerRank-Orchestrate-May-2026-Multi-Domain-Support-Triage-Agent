"""
modules/cache.py — Query-level LRU cache for retrieval, reranker, and
intent-classifier results.

Design:
  - Single LRUCache class backed by collections.OrderedDict.
  - Max 1 000 entries; evicts the least-recently-used entry on overflow.
  - Thread-safe via a reentrant lock (safe for future async use).
  - QueryCache wraps LRUCache with typed helpers for each pipeline stage.
"""

import logging
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAXSIZE = 1_000


class LRUCache:
    """
    Generic Least-Recently-Used cache backed by OrderedDict.

    Parameters
    ----------
    maxsize : int
        Maximum number of entries before eviction begins.
    """

    def __init__(self, maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self._maxsize = max(1, maxsize)
        self._cache: OrderedDict = OrderedDict()
        self._hits   = 0
        self._misses = 0
        self._lock   = threading.RLock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, key: Any) -> Tuple[Any, bool]:
        """Return (value, True) on cache hit, (None, False) on miss."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key], True
            self._misses += 1
            return None, False

    def put(self, key: Any, value: Any) -> None:
        """Insert or update an entry, evicting LRU entry if at capacity."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    evicted_key, _ = self._cache.popitem(last=False)
                    logger.debug("[cache] evicted key=%s", str(evicted_key)[:60])
            self._cache[key] = value

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        with self._lock:
            self._cache.clear()
            self._hits   = 0
            self._misses = 0

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._cache

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def stats(self) -> dict:
        """Return hit/miss counters and current size."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size":       len(self._cache),
                "maxsize":    self._maxsize,
                "hits":       self._hits,
                "misses":     self._misses,
                "hit_rate":   round(self._hits / max(total, 1), 4),
            }


class QueryCache:
    """
    Typed cache facade for the three pipeline cache targets:

    1. retrieval  — (query, domain_hint, top_k)  → RetrievalResult
    2. reranker   — (query, chunk_ids, top_n)     → float (top score)
    3. intent     — (normalised text)             → IntentClassificationResult
    """

    def __init__(self, maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self._cache = LRUCache(maxsize=maxsize)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_retrieval(self, query: str, domain_hint: Optional[str], top_k: int):
        key = ("ret", query[:200], domain_hint or "", top_k)
        return self._cache.get(key)

    def put_retrieval(self, query: str, domain_hint: Optional[str], top_k: int,
                      value: Any) -> None:
        key = ("ret", query[:200], domain_hint or "", top_k)
        self._cache.put(key, value)

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------

    def get_reranker(self, query: str, chunks: list, top_n: int):
        key = self._reranker_key(query, chunks, top_n)
        return self._cache.get(key)

    def put_reranker(self, query: str, chunks: list, top_n: int,
                     score: float) -> None:
        key = self._reranker_key(query, chunks, top_n)
        self._cache.put(key, score)

    @staticmethod
    def _reranker_key(query: str, chunks: list, top_n: int) -> tuple:
        # Limit to first 20 chunks to keep key bounded
        chunk_ids = tuple(c.chunk_id for c, _ in chunks[:20])
        return ("rnk", query[:100], chunk_ids, top_n)

    # ------------------------------------------------------------------
    # Intent classifier
    # ------------------------------------------------------------------

    def get_intent(self, text: str):
        key = ("clf", text[:200].lower().strip())
        return self._cache.get(key)

    def put_intent(self, text: str, value: Any) -> None:
        key = ("clf", text[:200].lower().strip())
        self._cache.put(key, value)

    # ------------------------------------------------------------------
    # Unified stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()
