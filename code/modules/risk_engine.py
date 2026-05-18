"""
Risk engine: assesses the risk level of a support ticket and detects
prompt injection / malicious content.

Risk levels:
  high   → always escalate
  medium → escalate if confidence is below threshold, else reply
  low    → reply (subject to confidence check)
"""

import re
from dataclasses import dataclass
from typing import List

from .config import (
    HIGH_RISK_KEYWORDS,
    MEDIUM_RISK_KEYWORDS,
    INJECTION_SIGNATURES,
    ALWAYS_ESCALATE_PATTERNS,
)


@dataclass
class RiskAssessment:
    level: str                  # "high" | "medium" | "low"
    reasons: List[str]          # human-readable list of triggered rules
    is_injection: bool          # prompt injection detected
    is_out_of_scope: bool       # completely outside supported domains

    @property
    def is_high(self) -> bool:
        return self.level == "high"

    @property
    def is_medium(self) -> bool:
        return self.level == "medium"

    def summary(self) -> str:
        parts = [f"risk={self.level}"]
        if self.is_injection:
            parts.append("injection_detected")
        if self.is_out_of_scope:
            parts.append("out_of_scope")
        if self.reasons:
            parts.append("reasons=" + "|".join(self.reasons))
        return ", ".join(parts)


class RiskEngine:
    """
    Evaluates a ticket for risk level, injection, and scope.

    Rules are evaluated top-down; first match wins the level.
    """

    # Out-of-scope patterns: content with no relevance to any domain
    _OOS_PATTERNS = [
        r"iron\s*man",
        r"\bactor\b.*\bmovie\b",
        r"(?:^|\b)give\s+me\s+(?:the\s+)?code\s+to\s+(?:delete|destroy|wipe)\b",
        r"(?:^|\b)(?:thanks|thank\s+you|ty)\s*[.!]?\s*$",
        r"(?:^|\b)(?:hi|hello|hey)\s*[.!]?\s*$",
        r"^\s*none\s+of\s+the\s+(?:pages?|websites?)\s+(?:are\s+)?(?:accessible|working)\s*$",
    ]

    def assess(self, text: str, domain: str, request_type: str) -> RiskAssessment:
        reasons: List[str] = []

        # --- Injection detection ---
        is_injection = self._detect_injection(text)
        if is_injection:
            reasons.append("prompt_injection_detected")

        # --- Out-of-scope detection ---
        is_oos = self._detect_out_of_scope(text, domain, request_type)
        if is_oos:
            reasons.append("out_of_scope")

        # If injection → treat as high risk
        if is_injection:
            return RiskAssessment(
                level="high",
                reasons=reasons,
                is_injection=True,
                is_out_of_scope=is_oos,
            )

        t = text.lower()

        # --- Always-escalate patterns (high risk) ---
        for pattern in ALWAYS_ESCALATE_PATTERNS:
            if re.search(pattern, t, re.IGNORECASE):
                rule = pattern[:60]
                if rule not in reasons:
                    reasons.append(f"always_escalate:{rule}")

        # --- High-risk keyword check (word-boundary safe) ---
        # Sort matched keywords so reason ordering is deterministic regardless
        # of Python set iteration order.
        high_hits = sorted(
            kw for kw in HIGH_RISK_KEYWORDS
            if re.search(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', t)
        )
        if high_hits:
            reasons.extend([f"high_risk_keyword:{kw}" for kw in high_hits[:3]])

        # --- Medium-risk keyword check (word-boundary safe) ---
        med_hits = sorted(
            kw for kw in MEDIUM_RISK_KEYWORDS
            if re.search(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', t)
        )
        if med_hits:
            reasons.extend([f"medium_risk_keyword:{kw}" for kw in med_hits[:3]])

        # --- Determine final level ---
        if high_hits or any("always_escalate" in r for r in reasons):
            level = "high"
        elif med_hits:
            level = "medium"
        else:
            level = "low"

        return RiskAssessment(
            level=level,
            reasons=reasons,
            is_injection=is_injection,
            is_out_of_scope=is_oos,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_injection(self, text: str) -> bool:
        t = text.lower()
        return any(sig in t for sig in INJECTION_SIGNATURES)

    def _detect_out_of_scope(self, text: str, domain: str, request_type: str) -> bool:
        if request_type == "invalid":
            return True
        t = text.lower().strip()
        for pattern in self._OOS_PATTERNS:
            if re.search(pattern, t, re.IGNORECASE):
                return True
        # Very short or extremely vague inputs with no domain signal
        words = t.split()
        if len(words) <= 6 and domain == "unknown":
            return True
        return False
