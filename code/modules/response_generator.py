"""
Response generator: produces grounded, extractive responses from retrieved
corpus chunks.  Never hallucinate – every sentence must come from the docs.

Strategy:
  1. Take the top N chunks from retrieval.
  2. From those chunks, extract the most relevant paragraphs (keyword overlap).
  3. Format into a professional support reply.
  4. For escalated/out-of-scope tickets, return the appropriate template.
"""

import re
from typing import List, Optional

from .config import (
    ESCALATION_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    INJECTION_RESPONSE,
)
from .retriever import RetrievalResult
from .loader import Chunk
from .decision_engine import Decision
from .risk_engine import RiskAssessment


def build_decision_summary(
    intent: str,
    risk_level: str,
    confidence: float,
    action: str,
    is_injection: bool = False,
    is_out_of_scope: bool = False,
    guardrail_safe: bool = True,
    has_retrieval: bool = True,
    request_type: str = "",
    conf_threshold: float = 0.65,
) -> str:
    """
    Build a concise one-line decision summary for explainability.

    Format:
        Intent=X | Risk=high | Confidence=0.42 | Action=ESCALATE | Rule=HighRiskDetected

    Rule priority:
        InjectionDetected   — prompt injection flagged
        GuardrailViolation  — guardrails blocked the request
        HighRiskDetected    — risk_level is "high"
        OutOfScope          — out-of-scope or invalid request type
        NoCorpusMatch       — no retrieval results available
        LowConfidence       — confidence below threshold
        GroundedReply       — normal grounded reply
    """
    # Determine primary rule triggered
    if is_injection:
        rule = "InjectionDetected"
    elif not guardrail_safe:
        rule = "GuardrailViolation"
    elif risk_level == "high":
        rule = "HighRiskDetected"
    elif is_out_of_scope or request_type == "invalid":
        rule = "OutOfScope"
    elif not has_retrieval:
        rule = "NoCorpusMatch"
    elif confidence < conf_threshold:
        rule = "LowConfidence"
    else:
        rule = "GroundedReply"

    action_label = action.upper()
    return (
        f"Intent={intent} | Risk={risk_level} | "
        f"Confidence={confidence:.2f} | Action={action_label} | Rule={rule}"
    )


class ResponseGenerator:

    # Max paragraphs extracted from corpus per response
    _MAX_PARAS = 6
    # Min paragraph length (chars) to include
    _MIN_PARA_LEN = 40

    def generate(
        self,
        issue: str,
        decision: Decision,
        risk: RiskAssessment,
        retrieval: Optional[RetrievalResult],
        request_type: str,
    ) -> str:
        """Return the user-facing response string."""

        # --- Injection ---
        if risk.is_injection:
            return INJECTION_RESPONSE

        # --- Out of scope / invalid ---
        if risk.is_out_of_scope or request_type == "invalid":
            return OUT_OF_SCOPE_RESPONSE

        # --- Escalated (non-injection) ---
        if decision.is_escalated:
            return ESCALATION_RESPONSE

        # --- Replied: extract from corpus ---
        if not retrieval or not retrieval.chunks:
            return ESCALATION_RESPONSE   # safety fallback

        return self._build_corpus_response(issue, retrieval)

    # ------------------------------------------------------------------
    # Extractive response construction
    # ------------------------------------------------------------------

    def _build_corpus_response(
        self, issue: str, retrieval: RetrievalResult
    ) -> str:
        # Pick the best chunks (up to 3 unique docs)
        chunks = retrieval.top_chunks(n=5)
        if not chunks:
            return ESCALATION_RESPONSE

        query_terms = set(re.findall(r'\b\w{3,}\b', issue.lower()))

        # Extract relevant paragraphs from those chunks
        paras = self._extract_paragraphs(chunks, query_terms)

        if not paras:
            # Fallback: use first chunk verbatim (cleaned)
            paras = [self._clean_text(chunks[0].content[:800])]

        # Compose the response
        body = "\n\n".join(paras[: self._MAX_PARAS])

        # Minimal professional wrapper
        response = f"Hi,\n\n{body}"

        # Append source reference if available (not a hallucination – it
        # comes from the doc's own source_url field)
        source_urls = self._collect_source_urls(chunks)
        if source_urls:
            response += "\n\nFor more information, please refer to: " + source_urls[0]

        return response.strip()

    # Headings/patterns to skip in extractive response
    _SKIP_PATTERNS = [
        r'^#+\s*related\s+articles',
        r'^_last\s+updated',
        r'^\s*last\s+updated',
        r'^\s*\*\*related',
        r'^\s*see\s+also',
        r'^\s*further\s+reading',
    ]

    def _is_junk_paragraph(self, para: str) -> bool:
        p = para.lower().strip()
        for pat in self._SKIP_PATTERNS:
            if re.match(pat, p, re.IGNORECASE):
                return True
        # Skip paragraphs that are just a list of bare URLs
        lines = [l.strip() for l in p.split('\n') if l.strip()]
        if lines and all(re.match(r'https?://', l) for l in lines):
            return True
        return False

    def _extract_paragraphs(
        self,
        chunks: List[Chunk],
        query_terms: set,
    ) -> List[str]:
        """
        Score every paragraph in every chunk by keyword overlap with the query,
        return the top paragraphs in document order.
        """
        scored: List[tuple] = []   # (score, chunk_idx, para_idx, text)

        for ci, chunk in enumerate(chunks):
            paragraphs = re.split(r'\n{2,}', chunk.content)
            for pi, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) < self._MIN_PARA_LEN:
                    continue
                # Skip pure markdown fences / images / links
                if para.startswith("```") or para.startswith("!["):
                    continue
                # Skip junk sections (related articles, last updated, etc.)
                if self._is_junk_paragraph(para):
                    continue
                para_terms = set(re.findall(r'\b\w{3,}\b', para.lower()))
                overlap = len(query_terms & para_terms)
                # Boost if paragraph contains numbered/bulleted steps
                if re.search(r'^\s*(?:\d+\.|[-*•])\s', para, re.MULTILINE):
                    overlap += 2
                scored.append((overlap, ci, pi, para))

        # Sort: first by overlap desc, then by document order
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))

        # Collect unique paragraphs (avoid exact duplicates)
        seen: set = set()
        result: List[str] = []
        for _, _, _, text in scored:
            cleaned = self._clean_text(text)
            key = cleaned[:80]
            if key not in seen:
                seen.add(key)
                result.append(cleaned)
            if len(result) >= self._MAX_PARAS:
                break

        # Re-order by original document position so the response reads naturally
        # (Simple heuristic: chunks with higher original score first)
        return result

    def _clean_text(self, text: str) -> str:
        """Light markdown cleanup for terminal / plain-text output."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Convert markdown links [text](url) → text (url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
        # Remove image markdown
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace per line
        lines = [l.rstrip() for l in text.split('\n')]
        return '\n'.join(lines).strip()

    def _collect_source_urls(self, chunks: List[Chunk]) -> List[str]:
        """Gather unique source URLs from the chunk file paths (not hallucinated)."""
        urls: List[str] = []
        seen: set = set()
        for chunk in chunks:
            # Extract source_url from file if possible – we stored it in title
            # For safety, we only surface URLs that are present in the corpus
            # (the doc's own content may contain a source_url reference)
            content = chunk.content
            found = re.findall(
                r'https?://(?:support\.hackerrank\.com|support\.claude\.com|'
                r'privacy\.claude\.com|www\.visa\.co\.in|visa\.co\.in)'
                r'[^\s\)\"\']*',
                content,
            )
            for url in found:
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        return urls
