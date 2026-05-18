"""
FAISS index builder and persistence layer.

On first run the index is built from chunks and saved to .cache/.
Subsequent runs load from cache (fast).  Pass force_rebuild=True to
regenerate.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, CACHE_DIR, RANDOM_SEED
)
from .loader import Chunk


class VectorIndex:
    """FAISS flat-IP index over document chunks."""

    _INDEX_FILE = CACHE_DIR / "faiss.index"
    _CHUNKS_FILE = CACHE_DIR / "chunks.pkl"
    _MODEL_FILE = CACHE_DIR / "model_name.txt"

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk], force_rebuild: bool = False) -> None:
        """Embed all chunks and build the FAISS index. Caches to disk."""
        if not force_rebuild and self._cache_valid():
            self._load_from_cache()
            return

        print(f"[indexer] Building FAISS index over {len(chunks)} chunks …")
        self._model = self._load_model()
        embeddings = self._embed(chunks)

        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)

        self._index = index
        self._chunks = chunks

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._INDEX_FILE))
        with open(self._CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
        self._MODEL_FILE.write_text(self.model_name)
        print(f"[indexer] Index saved to {CACHE_DIR}")

    def query(self, text: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs for a query string."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")

        vec = self._embed_query(text)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_valid(self) -> bool:
        if not (self._INDEX_FILE.exists() and self._CHUNKS_FILE.exists()):
            return False
        if not self._MODEL_FILE.exists():
            return False
        return self._MODEL_FILE.read_text().strip() == self.model_name

    def optimize(self) -> None:
        """
        Apply search-time parameters for better throughput.
        Sets nprobe=8 on IVF-family indexes (larger probe set → better recall
        with approximate search). No-op for IndexFlatIP which already does
        exhaustive exact search.
        Must be called after build().
        """
        if self._index is None:
            return
        try:
            self._index.nprobe = 8
            print("[indexer] nprobe=8 applied (IVF index).")
        except AttributeError:
            pass  # IndexFlatIP: exhaustive search — nprobe not applicable

    def _load_from_cache(self) -> None:
        print(f"[indexer] Loading cached index from {CACHE_DIR} …")
        self._index = faiss.read_index(str(self._INDEX_FILE))
        with open(self._CHUNKS_FILE, "rb") as f:
            self._chunks = pickle.load(f)
        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        print(f"[indexer] Loading embedding model '{self.model_name}' …")
        return SentenceTransformer(self.model_name)

    def _embed(self, chunks: List[Chunk]) -> np.ndarray:
        texts = [c.text_for_embedding for c in chunks]
        vecs = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        if self._model is None:
            self._model = self._load_model()
        vec = self._model.encode([text], normalize_embeddings=False, convert_to_numpy=True)
        return vec.astype(np.float32)
