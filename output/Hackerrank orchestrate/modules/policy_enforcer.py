"""
Policy citation enforcer.

Ensures every agent response that is NOT an escalation or out-of-scope reply
is grounded in at least one retrieved document.

Rules
-----
1. If the response already contains a "Source:" citation → PASS.
2. If retrieved_doc_ids is non-empty AND the best score is acceptable
   → APPEND a citation and PASS.
3. If no docs are available (retrieval empty or None) → FAIL (escalate).

The enforcer both VALIDATES and ENRICHES:
  - validate()  returns a PolicyResult (pass / fail with reason).
  - enforce()   mutates the response string to add a citation when missing,
                or returns the original string on failure.

Citation format
---------------
    \n\nSource: <doc_id>

where <doc_id> is the first element of retrieved_doc_ids.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional

from .retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Minimum retrieval score for a citation to be considered valid
_MIN_CITATION_SCORE = 0.15

_CITATION_RE = re.compile(
    r'(?:source|ref(?:erence)?|see\s+also|cited?\s+from)\s*:',
    re.IGNORECASE,
)


@dataclass
class PolicyResult:
    """
    Outcome of policy validation.

    Attributes
    ----------
    passed        : True when policy is satisfied (citation present or added).
    has_citation  : True when the response already had a citation before checking.
    added_citation: True when the enforcer appended a citation.
    escalate      : True when no grounding docs exist → caller should escalate.
    reason        : Human-readable description.
    """
    passed:         bool
    has_citation:   bool  = False
    added_citation: bool  = False
    escalate:       bool  = False
    reason:         str   = ""


class PolicyEnforcer:
    """
    Citation-based response grounding enforcer.

    Parameters
    ----------
    require_citation : bool
        When True (default), responses must be grounded.  Set False to
        validate-only without triggering escalation.
    """

    def __init__(self, require_citation: bool = True):
        self.require_citation = require_citation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        response: str,
        retrieval: Optional[RetrievalResult],
        is_escalation: bool = False,
    ) -> PolicyResult:
        """
        Validate that *response* is properly cited.

        Parameters
        ----------
        response      : Generated response text.
        retrieval     : Retrieval result (may be None for injection / OOS).
        is_escalation : True for escalation/OOS responses (exempt from rule).
        """
        # Escalation and OOS replies are pre-defined templates — exempt.
        if is_escalation:
            return PolicyResult(
                passed=True, has_citation=False,
                reason="escalation template — citation not required.",
            )

        # Check existing citation
        if _CITATION_RE.search(response):
            return PolicyResult(
                passed=True, has_citation=True,
                reason="Response already contains a citation.",
            )

        # Check retrieval quality
        if (
            retrieval is not None
            and retrieval.top_docs
            and retrieval.best_score >= _MIN_CITATION_SCORE
        ):
            return PolicyResult(
                passed=True, has_citation=False, added_citation=False,
                reason="Retrieval docs available; citation will be appended by enforce().",
            )

        # No docs — policy fail
        if self.require_citation:
            return PolicyResult(
                passed=False, escalate=True,
                reason="No grounding documents retrieved; response cannot be cited.",
            )

        return PolicyResult(
            passed=True,
            reason="require_citation=False; policy check skipped.",
        )

    def enforce(
        self,
        response: str,
        retrieval: Optional[RetrievalResult],
        is_escalation: bool = False,
    ) -> str:
        """
        Return the (possibly citation-enriched) response string.

        If the response already has a citation or is exempt, return as-is.
        If docs are available but citation is missing, append one.
        If policy fails (no docs), return the original (caller decides on escalation).
        """
        if is_escalation:
            return response

        if _CITATION_RE.search(response):
            return response  # already has citation

        if (
            retrieval is not None
            and retrieval.top_docs
            and retrieval.best_score >= _MIN_CITATION_SCORE
        ):
            doc_id = retrieval.top_docs[0][0]   # top doc_id
            citation = f"\n\nSource: {doc_id}"
            enriched = response.rstrip() + citation
            logger.debug("[policy] appended citation Source:%s", doc_id)
            return enriched

        # No docs — return as-is; escalation decision is in validate()
        return response

    def validate_and_enforce(
        self,
        response: str,
        retrieval: Optional[RetrievalResult],
        is_escalation: bool = False,
    ) -> tuple:
        """
        Combined call: validate, enrich, and return (enriched_response, PolicyResult).
        """
        result   = self.validate(response, retrieval, is_escalation)
        enriched = self.enforce(response, retrieval, is_escalation)
        if enriched != response and not result.added_citation:
            result = PolicyResult(
                passed=True,
                has_citation=False,
                added_citation=True,
                reason=result.reason,
            )
        return enriched, result
