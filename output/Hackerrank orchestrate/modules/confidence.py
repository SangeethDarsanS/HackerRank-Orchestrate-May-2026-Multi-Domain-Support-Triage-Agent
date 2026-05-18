"""
Composite confidence scorer.

Formula
-------
confidence = 0.4 * retrieval_score
           + 0.3 * classifier_probability
           + 0.2 * reranker_score
           + 0.1 * guardrail_safety_score

All components are in [0, 1]; the composite score is also in [0, 1].

Low-confidence rule
-------------------
If composite confidence < LOW_CONFIDENCE_THRESHOLD (0.65) the ticket
should be escalated to a human specialist.
"""

import logging
from typing import Optional

from .retriever import RetrievalResult
from .guardrails import GuardrailResult

logger = logging.getLogger(__name__)

# Component weights (must sum to 1.0)
_W_RETRIEVAL   = 0.4
_W_CLASSIFIER  = 0.3
_W_RERANKER    = 0.2
_W_GUARDRAIL   = 0.1

# Pseudo-probability per request_type (keyword classifier is not probabilistic)
_TYPE_PROB: dict = {
    "bug":             0.80,
    "feature_request": 0.75,
    "product_issue":   0.65,
    "invalid":         0.10,
}

# Composite score below this threshold → ESCALATE
LOW_CONFIDENCE_THRESHOLD: float = 0.65


class ConfidenceScorer:
    """
    Stateless composite confidence calculator.

    Call ``score()`` after retrieval, reranking, and guardrail validation.
    """

    def score(
        self,
        retrieval: Optional[RetrievalResult],
        request_type: str,
        reranker_score: float,
        guardrail_result: GuardrailResult,
    ) -> dict:
        """
        Compute the composite confidence score.

        Parameters
        ----------
        retrieval       : FAISS retrieval result (None → retrieval_score=0).
        request_type    : classifier output (one of the REQUEST_TYPES).
        reranker_score  : top cross-encoder score after sigmoid normalisation.
        guardrail_result: result from Guardrails.validate().

        Returns
        -------
        dict with keys:
          confidence      float [0, 1]
          retrieval_score float
          classifier_prob float
          reranker_score  float
          guardrail_score float
          low_confidence  bool
        """
        # Retrieval component
        raw_ret = retrieval.best_score if retrieval else 0.0
        retrieval_s = float(min(max(raw_ret, 0.0), 1.0))

        # Classifier component (pseudo-probability)
        classifier_s = float(_TYPE_PROB.get(request_type, 0.65))

        # Reranker component
        reranker_s = float(min(max(reranker_score, 0.0), 1.0))

        # Guardrail component
        guardrail_s = float(min(max(guardrail_result.safety_score, 0.0), 1.0))

        # Composite
        composite = (
            _W_RETRIEVAL  * retrieval_s
            + _W_CLASSIFIER * classifier_s
            + _W_RERANKER   * reranker_s
            + _W_GUARDRAIL  * guardrail_s
        )
        composite = float(min(max(composite, 0.0), 1.0))

        result = {
            "confidence":      composite,
            "retrieval_score": retrieval_s,
            "classifier_prob": classifier_s,
            "reranker_score":  reranker_s,
            "guardrail_score": guardrail_s,
            "low_confidence":  composite < LOW_CONFIDENCE_THRESHOLD,
        }

        logger.debug("[confidence] %s", result)
        return result
