"""
Decision engine: decides whether a ticket should be replied to or escalated.

Decision hierarchy (evaluated top-to-bottom; first match wins):

1. Prompt injection or clearly malicious            → ESCALATE
2. Out-of-scope (invalid request type)              → REPLY (out-of-scope msg)
3. Guardrail violation                              → ESCALATE
4. High-risk (fraud, identity theft …)             → ESCALATE
5. Low composite confidence (<threshold)            → ESCALATE
6. Medium-risk AND low raw confidence               → ESCALATE
7. Low/no corpus match (best_score < threshold)     → ESCALATE
7.5 Low reranker relevance score (<0.55)            → ESCALATE
8. Otherwise                                        → REPLY
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .config import (
    SIMILARITY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)
from .risk_engine import RiskAssessment
from .retriever import RetrievalResult

if TYPE_CHECKING:
    from .guardrails import GuardrailResult
    from .confidence import ConfidenceScorer

# Composite confidence threshold (for the weighted score from confidence.py)
_LOW_CONFIDENCE_THRESHOLD = 0.65


@dataclass
class Decision:
    action: str             # "replied" | "escalated"
    reason: str             # brief explanation used in justification
    confidence: float       # 0–1 composite or retrieval score

    @property
    def is_escalated(self) -> bool:
        return self.action == "escalated"

    @property
    def is_replied(self) -> bool:
        return self.action == "replied"


class DecisionEngine:
    """
    Stateless decision engine.

    The ``decide()`` method accepts optional keyword arguments:

    composite_confidence
        Pre-computed weighted confidence from ConfidenceScorer.  When
        provided it replaces the raw retrieval score for threshold checks,
        and the low-confidence rule uses 0.65 instead of CONFIDENCE_THRESHOLD.

    guardrail_result
        Result from Guardrails.validate().  When provided and ``safe=False``,
        the ticket is escalated (Rule 3) before any other checks.

    reranker_score
        Score from the cross-encoder reranker (0–1).  When provided and the
        reranker actually ran (i.e. not a neutral placeholder), tickets with a
        score below 0.55 are escalated to avoid low-relevance replies.
        Pass ``None`` (default) when the reranker is disabled.

    All parameters are optional so existing call sites still work unchanged.
    """

    # Threshold below which a real reranker score triggers escalation
    _RERANKER_THRESHOLD = 0.55

    def decide(
        self,
        risk: RiskAssessment,
        retrieval: Optional[RetrievalResult],
        request_type: str,
        composite_confidence: Optional[float] = None,
        guardrail_result: Optional["GuardrailResult"] = None,
        reranker_score: Optional[float] = None,
    ) -> Decision:

        # --- Effective confidence (raw retrieval score if composite unavailable) ---
        raw_score = retrieval.best_score if retrieval else 0.0
        raw_score = min(max(raw_score, 0.0), 1.0)

        effective_conf = composite_confidence if composite_confidence is not None else raw_score
        effective_conf = min(max(effective_conf, 0.0), 1.0)

        # --- Rule 1: Injection / malicious ---
        if risk.is_injection:
            return Decision(
                action="escalated",
                reason="Prompt injection or malicious content detected; escalated for safety review.",
                confidence=effective_conf,
            )

        # --- Rule 2: Out-of-scope (invalid request type) ---
        if risk.is_out_of_scope or request_type == "invalid":
            return Decision(
                action="replied",
                reason="Request is outside the scope of supported domains; out-of-scope response provided.",
                confidence=effective_conf,
            )

        # --- Rule 3: Guardrail violation ---
        if guardrail_result is not None and not guardrail_result.safe:
            violations = guardrail_result.violation_summary()
            return Decision(
                action="escalated",
                reason=f"Safety guardrail violation ({violations}); escalated for specialist review.",
                confidence=effective_conf,
            )

        # --- Rule 4: High risk ---
        if risk.is_high:
            reasons_str = "; ".join(risk.reasons[:3]) if risk.reasons else "high-risk content"
            return Decision(
                action="escalated",
                reason=f"High-risk content detected ({reasons_str}); requires human specialist.",
                confidence=effective_conf,
            )

        # --- Rule 5: Low composite confidence ---
        if composite_confidence is not None and composite_confidence < _LOW_CONFIDENCE_THRESHOLD:
            return Decision(
                action="escalated",
                reason=(
                    f"Low composite confidence ({composite_confidence:.3f} < "
                    f"{_LOW_CONFIDENCE_THRESHOLD}); escalated to avoid ungrounded response."
                ),
                confidence=effective_conf,
            )

        # --- Rule 6: Medium risk + low raw confidence (legacy path when no composite) ---
        if composite_confidence is None and risk.is_medium and raw_score < CONFIDENCE_THRESHOLD:
            return Decision(
                action="escalated",
                reason=(
                    f"Medium-risk ticket with low retrieval confidence ({raw_score:.2f}); "
                    "escalated to ensure accurate handling."
                ),
                confidence=effective_conf,
            )

        # --- Rule 7: No corpus match ---
        if not retrieval or raw_score < SIMILARITY_THRESHOLD:
            return Decision(
                action="escalated",
                reason=(
                    "No sufficiently relevant documentation found in the corpus; "
                    "escalated to avoid ungrounded response."
                ),
                confidence=effective_conf,
            )

        # --- Rule 7.5: Low reranker relevance ---
        # Only applied when caller passes a real reranker score (not None and not
        # the neutral 0.5 placeholder used when the reranker is disabled).
        if (
            reranker_score is not None
            and retrieval
            and retrieval.chunks
            and reranker_score < self._RERANKER_THRESHOLD
        ):
            return Decision(
                action="escalated",
                reason=(
                    f"Low reranker relevance score ({reranker_score:.3f} < "
                    f"{self._RERANKER_THRESHOLD}); best matching document may not "
                    "be accurate enough for an automated reply."
                ),
                confidence=effective_conf,
            )

        # --- Rule 8: Reply ---
        return Decision(
            action="replied",
            reason=f"Relevant documentation found (confidence={effective_conf:.2f}); grounded reply generated.",
            confidence=effective_conf,
        )
