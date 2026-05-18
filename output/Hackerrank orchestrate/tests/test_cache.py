"""
tests/test_cache.py — Unit tests for LRUCache and QueryCache.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.cache import LRUCache, QueryCache


# ---------------------------------------------------------------------------
# LRUCache tests
# ---------------------------------------------------------------------------

class TestLRUCache:

    def test_get_miss(self):
        cache = LRUCache(maxsize=10)
        value, hit = cache.get("missing")
        assert hit is False
        assert value is None

    def test_put_and_get_hit(self):
        cache = LRUCache(maxsize=10)
        cache.put("key1", "value1")
        value, hit = cache.get("key1")
        assert hit is True
        assert value == "value1"

    def test_evicts_lru_on_overflow(self):
        cache = LRUCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Access "a" to make it recently used
        cache.get("a")
        # Add "d" — should evict "b" (LRU)
        cache.put("d", 4)
        _, hit_b = cache.get("b")
        _, hit_a = cache.get("a")
        _, hit_d = cache.get("d")
        assert hit_b is False
        assert hit_a is True
        assert hit_d is True

    def test_len(self):
        cache = LRUCache(maxsize=10)
        assert len(cache) == 0
        cache.put("x", 1)
        assert len(cache) == 1

    def test_contains(self):
        cache = LRUCache(maxsize=10)
        cache.put("hello", 99)
        assert "hello" in cache
        assert "world" not in cache

    def test_clear(self):
        cache = LRUCache(maxsize=10)
        cache.put("k", "v")
        cache.clear()
        assert len(cache) == 0
        _, hit = cache.get("k")
        assert hit is False

    def test_stats_hit_rate(self):
        cache = LRUCache(maxsize=10)
        cache.put("k", "v")
        cache.get("k")   # hit
        cache.get("z")   # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_stats_empty(self):
        cache = LRUCache(maxsize=10)
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["hit_rate"] == 0.0

    def test_update_existing_key(self):
        cache = LRUCache(maxsize=10)
        cache.put("k", "v1")
        cache.put("k", "v2")
        val, hit = cache.get("k")
        assert hit is True
        assert val == "v2"
        assert len(cache) == 1

    def test_thread_safety(self):
        cache = LRUCache(maxsize=100)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 20):
                    cache.put(f"key_{i}", i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_maxsize_one(self):
        cache = LRUCache(maxsize=1)
        cache.put("a", 1)
        cache.put("b", 2)
        _, hit_a = cache.get("a")
        _, hit_b = cache.get("b")
        assert hit_a is False
        assert hit_b is True


# ---------------------------------------------------------------------------
# QueryCache tests
# ---------------------------------------------------------------------------

class TestQueryCache:

    def test_retrieval_miss(self):
        qc = QueryCache()
        val, hit = qc.get_retrieval("query", "domain", 10)
        assert hit is False

    def test_retrieval_hit(self):
        qc = QueryCache()
        qc.put_retrieval("my query", "hackerrank", 5, "RESULT_OBJ")
        val, hit = qc.get_retrieval("my query", "hackerrank", 5)
        assert hit is True
        assert val == "RESULT_OBJ"

    def test_retrieval_key_includes_domain(self):
        qc = QueryCache()
        qc.put_retrieval("q", "domain_a", 5, "A")
        val, hit = qc.get_retrieval("q", "domain_b", 5)
        assert hit is False

    def test_retrieval_key_includes_top_k(self):
        qc = QueryCache()
        qc.put_retrieval("q", "d", 5, "A")
        val, hit = qc.get_retrieval("q", "d", 10)
        assert hit is False

    def test_retrieval_none_domain(self):
        qc = QueryCache()
        qc.put_retrieval("q", None, 5, "OBJ")
        val, hit = qc.get_retrieval("q", None, 5)
        assert hit is True

    def test_reranker_miss(self):
        qc = QueryCache()
        val, hit = qc.get_reranker("q", [], 3)
        assert hit is False

    def test_reranker_hit(self):
        qc = QueryCache()

        class FakeChunk:
            def __init__(self, cid):
                self.chunk_id = cid

        chunks = [(FakeChunk("c1"), 0.9), (FakeChunk("c2"), 0.8)]
        qc.put_reranker("query", chunks, 3, 0.95)
        val, hit = qc.get_reranker("query", chunks, 3)
        assert hit is True
        assert abs(val - 0.95) < 1e-9

    def test_intent_miss(self):
        qc = QueryCache()
        val, hit = qc.get_intent("some text")
        assert hit is False

    def test_intent_hit(self):
        qc = QueryCache()
        qc.put_intent("my ticket text", "INTENT_OBJ")
        val, hit = qc.get_intent("my ticket text")
        assert hit is True
        assert val == "INTENT_OBJ"

    def test_intent_normalizes_case(self):
        qc = QueryCache()
        qc.put_intent("Hello World", "OBJ")
        val, hit = qc.get_intent("HELLO WORLD")
        assert hit is True

    def test_stats(self):
        qc = QueryCache()
        qc.put_retrieval("q", None, 5, "X")
        qc.get_retrieval("q", None, 5)  # hit
        qc.get_retrieval("missing", None, 5)  # miss
        stats = qc.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert stats["hits"] >= 1

    def test_clear(self):
        qc = QueryCache()
        qc.put_intent("text", "obj")
        qc.clear()
        _, hit = qc.get_intent("text")
        assert hit is False
