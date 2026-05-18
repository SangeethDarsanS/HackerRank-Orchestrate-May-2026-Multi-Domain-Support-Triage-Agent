"""
Safety guardrails module.

Rule-based validation that runs *after* retrieval to catch unsafe or
unsupported response categories before the response generator sees them.

Detected violations
-------------------
sensitive_category:<name>  — one or more sensitive topic detected in ticket text
missing_documentation      — no relevant docs retrieved
low_evidence_score         — retrieved docs below minimum similarity floor
unsupported_claim          — absolute guarantee language in ticket text

Decision
--------
GuardrailResult.safe=False  →  caller should override action to ESCALATE.
GuardrailResult.safety_score is used as one component of the composite
confidence formula (weight 0.1).
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .retriever import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive-category patterns
# Each entry: (category_name, [regex_patterns])
# One hit in any pattern → that category is flagged.
# ---------------------------------------------------------------------------
_SENSITIVE_CATEGORIES: List[tuple] = [
    (
        "financial_advice",
        [
            r"\bfinancial\s+advice\b",
            r"\binvest\s+in\b",
            r"\bstock\s+market\b",
            r"\btax\s+advice\b",
        ],
    ),
    (
        "legal_advice",
        [
            r"\blegal\s+advice\b",
            r"\bconsult\s+(?:a\s+)?(?:lawyer|attorney|solicitor)\b",
            r"\blegal\s+action\b",
            r"\bfiling\s+a\s+complaint\b",
        ],
    ),
    (
        "account_recovery",
        [
            r"\baccount\s+recover(?:y|ed)?\b",
            r"\brecover\s+(?:my\s+)?account\b",
            r"\bregain\s+access\b",
        ],
    ),
    (
        "fraud",
        [
            r"\bfraud(?:ulent)?\b",
            r"\bphishing\b",
            r"\bransomware\b",
            r"\bmoney\s+laundering\b",
            r"\bscam(?:med)?\b",
        ],
    ),
    (
        "password_reset",
        [
            r"\bpassword\s+reset\b",
            r"\breset\s+(?:my\s+)?password\b",
            r"\bforgot\s+(?:my\s+)?password\b",
            r"\bchange\s+(?:my\s+)?password\b",
        ],
    ),
    (
        "identity_verification",
        [
            r"\bidentity\s+verif(?:ication|y|ied)\b",
            r"\bverify\s+(?:my\s+)?identity\b",
            r"\bidentity\s+(?:theft|stolen)\b",
            r"\bkyc\b",
        ],
    ),
    (
        "payment_dispute",
        [
            r"\bpayment\s+dispute\b",
            r"\bdispute\s+(?:a\s+)?(?:charge|payment|transaction)\b",
            r"\bchargeback\b",
            r"\bunauthorized\s+(?:charge|transaction|payment)\b",
            r"\bdouble\s+charged\b",
            r"\bovercharged\b",
        ],
    ),
    (
        "security_breach",
        [
            r"\bsecurity\s+breach\b",
            r"\bdata\s+breach\b",
            r"\bhacked\b",
            r"\bcompromised\s+account\b",
            r"\baccount\s+compromised\b",
        ],
    ),
    (
        "medical_advice",
        [
            r"\bmedical\s+advice\b",
            r"\bdiagnos(?:e|is)\b",
            r"\bprescri(?:be|ption)\b",
            r"\bsymptom\b",
        ],
    ),
]

# Minimum similarity score required to consider docs as valid evidence
_MIN_EVIDENCE_SCORE = 0.15

# Patterns that indicate unsupported/over-confident claims
_UNSUPPORTED_CLAIM_PATTERNS = [
    r"\bguarantee\b",
    r"\bwe\s+promise\b",
    r"\b100%\s+(?:sure|certain|guaranteed)\b",
    r"\bdefinitely\s+(?:will|can|should)\b",
]


@dataclass
class GuardrailResult:
    """
    Result of guardrail validation.

    Attributes
    ----------
    safe         : True when no violations found (or only non-blocking ones).
    violations   : List of violation tags.
    safety_score : Numeric score in [0, 1]; 1.0 = fully safe.
    """

    safe: bool
    violations: List[str] = field(default_factory=list)
    safety_score: float = 1.0

    @property
    def status(self) -> str:
        return "safe" if self.safe else "unsafe"

    def violation_summary(self) -> str:
        return "; ".join(self.violations) if self.violations else "none"


class Guardrails:
    """
    Rule-based safety validator.

    Parameters
    ----------
    strict_mode : bool
        When True, *any* violation marks the result unsafe (recommended for
        production).  When False, only violations other than
        ``missing_documentation`` cause ``safe=False``.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        ticket_text: str,
        retrieval: Optional[RetrievalResult],
        request_type: str = "product_issue",
    ) -> GuardrailResult:
        """
        Run all guardrail checks against a ticket.

        Parameters
        ----------
        ticket_text  : combined issue + subject text.
        retrieval    : retrieval result (may be None for injection/invalid).
        request_type : classifier output; affects evidence checks.

        Returns
        -------
        GuardrailResult with safe=True iff all checks pass.
        """
        violations: List[str] = []
        text_lower = ticket_text.lower()

        # 1. Sensitive category detection
        for category, patterns in _SENSITIVE_CATEGORIES:
            for pat in patterns:
                if re.search(pat, text_lower):
                    tag = f"sensitive_category:{category}"
                    if tag not in violations:
                        violations.append(tag)
                    break  # one hit per category suffices

        # 2. Evidence quality check (skip for invalid/injection request types)
        if request_type not in ("invalid",):
            if retrieval is None or not retrieval.top_docs:
                violations.append("missing_documentation")
            elif retrieval.best_score < _MIN_EVIDENCE_SCORE:
                violations.append("low_evidence_score")

        # 3. Unsupported-claim check (applies to ticket text)
        for pat in _UNSUPPORTED_CLAIM_PATTERNS:
            if re.search(pat, text_lower):
                violations.append("unsupported_claim")
                break

        # Compute safety score: deduct 0.2 per violation, floor at 0.0
        safety_score = max(0.0, 1.0 - len(violations) * 0.2)

        # Determine safe / unsafe
        if self.strict_mode:
            safe = len(violations) == 0
        else:
            # Non-strict: only category-level or evidence violations are blocking
            blocking = [
                v for v in violations
                if not v.startswith("unsupported_claim")
            ]
            safe = len(blocking) == 0

        if violations:
            logger.debug(
                "[guardrails] violations=%s safe=%s score=%.2f",
                violations,
                safe,
                safety_score,
            )

        return GuardrailResult(
            safe=safe,
            violations=violations,
            safety_score=safety_score,
        )
