"""
Document loader: reads all .md files from data/, parses frontmatter,
determines domain + product_area, and splits into retrieval chunks.
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import (
    DATA_DIR, HACKERRANK_AREA_MAP, CLAUDE_AREA_MAP, VISA_AREA_MAP
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    title: str
    content: str        # full text (stripped of frontmatter)
    domain: str         # hackerrank | claude | visa | unknown
    area: str           # product area (e.g. screen, billing, travel_support)
    file_path: str
    source_url: str = ""

    def preview(self, n: int = 120) -> str:
        text = self.content.replace("\n", " ")
        return text[:n] + "…" if len(text) > n else text


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    content: str
    domain: str
    area: str
    file_path: str
    chunk_index: int = 0

    @property
    def text_for_embedding(self) -> str:
        """Title + content used when building the embedding."""
        return f"{self.title}\n{self.content}"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """Loads markdown files from DATA_DIR and chunks them for FAISS."""

    MAX_CHUNK_CHARS = 1_400   # keeps chunks within ~256 tokens

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> List[Document]:
        """Return every non-empty document in data/."""
        docs: List[Document] = []
        for md_file in sorted(self.data_dir.rglob("*.md")):
            doc = self._parse_file(md_file)
            if doc and doc.content.strip():
                docs.append(doc)
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Chunk]:
        """Split each document into paragraph-level chunks."""
        chunks: List[Chunk] = []
        for doc in docs:
            chunks.extend(self._chunk_document(doc))
        return chunks

    # ------------------------------------------------------------------
    # File parsing
    # ------------------------------------------------------------------

    def _parse_file(self, filepath: Path) -> Optional[Document]:
        try:
            raw = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        title, source_url, content = self._parse_frontmatter(raw)
        if not content.strip():
            return None

        domain, area = self._classify_path(filepath)
        rel = str(filepath.relative_to(self.data_dir))
        doc_id = hashlib.md5(rel.encode()).hexdigest()[:12]

        return Document(
            doc_id=doc_id,
            title=title,
            content=content,
            domain=domain,
            area=area,
            file_path=str(filepath),
            source_url=source_url,
        )

    def _parse_frontmatter(self, text: str) -> Tuple[str, str, str]:
        title = ""
        source_url = ""
        content = text

        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                fm = text[3:end]
                content = text[end + 4:].strip()

                m = re.search(
                    r'^title:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE
                )
                if m:
                    title = m.group(1).strip().strip('"\'')

                m = re.search(
                    r'^(?:source_url|final_url):\s*["\']?(.+?)["\']?\s*$',
                    fm,
                    re.MULTILINE,
                )
                if m:
                    source_url = m.group(1).strip().strip('"\'')

        if not title:
            m = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if m:
                title = m.group(1).strip()

        if not title:
            title = "Untitled"

        return title, source_url, content

    # ------------------------------------------------------------------
    # Path → domain + area
    # ------------------------------------------------------------------

    def _classify_path(self, filepath: Path) -> Tuple[str, str]:
        parts = filepath.parts
        try:
            data_idx = next(i for i, p in enumerate(parts) if p == "data")
        except StopIteration:
            return "unknown", "general_support"

        tail = parts[data_idx + 1:]   # parts after data/
        if not tail:
            return "unknown", "general_support"

        top = tail[0].lower()

        if "hackerrank" in top:
            sub = tail[1].lower() if len(tail) > 1 else ""
            area = HACKERRANK_AREA_MAP.get(sub, sub.replace("-", "_") or "general_support")
            return "hackerrank", area

        if "claude" in top:
            return "claude", self._claude_area(tail[1:])

        if "visa" in top:
            return "visa", self._visa_area(tail[1:])

        return "unknown", "general_support"

    def _claude_area(self, parts: tuple) -> str:
        if not parts:
            return "claude"
        top = parts[0].lower()
        if top in CLAUDE_AREA_MAP:
            return CLAUDE_AREA_MAP[top]
        if top == "claude" and len(parts) > 1:
            sub = parts[1].lower()
            return CLAUDE_AREA_MAP.get(sub, sub.replace("-", "_"))
        return top.replace("-", "_")

    def _visa_area(self, parts: tuple) -> str:
        joined = "/".join(p.lower() for p in parts)
        for key, val in VISA_AREA_MAP.items():
            if key in joined:
                return val
        return "general_support"

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_document(self, doc: Document) -> List[Chunk]:
        # Split on markdown headers first
        sections = re.split(r'\n(?=#{1,4}\s)', doc.content)
        chunks: List[Chunk] = []
        idx = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) <= self.MAX_CHUNK_CHARS:
                chunks.append(self._make_chunk(doc, section, idx))
                idx += 1
            else:
                # Split large sections into paragraph chunks
                paras = re.split(r'\n{2,}', section)
                current = ""
                for para in paras:
                    para = para.strip()
                    if not para:
                        continue
                    if len(current) + len(para) + 2 <= self.MAX_CHUNK_CHARS:
                        current = (current + "\n\n" + para).lstrip()
                    else:
                        if current:
                            chunks.append(self._make_chunk(doc, current, idx))
                            idx += 1
                        current = para
                if current:
                    chunks.append(self._make_chunk(doc, current, idx))
                    idx += 1

        return [c for c in chunks if c.content.strip()]

    def _make_chunk(self, doc: Document, content: str, idx: int) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.doc_id}_{idx}",
            doc_id=doc.doc_id,
            title=doc.title,
            content=content,
            domain=doc.domain,
            area=doc.area,
            file_path=doc.file_path,
            chunk_index=idx,
        )
