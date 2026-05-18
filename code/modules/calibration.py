"""
modules/calibration.py — Confidence score calibration.

Applies Platt (sigmoid) scaling to the raw composite confidence score produced
by ConfidenceScorer so that the reported value better reflects true outcome
probability.

Platt scaling formula:
    calibrated = 1 / (1 + exp(A * raw + B))

Defaults (A = -4.0, B = 2.0) produce a gentle S-curve that:
  - compresses over-confident scores near 1.0 slightly downward
  - keeps the 0.65 decision threshold region well-separated
  - is monotone and invertible

If labeled calibration data is available (calibrate_fit() with binary labels
and raw scores), optimal A and B are fitted via LogisticRegression (matching
sklearn's CalibratedClassifierCV sigmoid approach).

Integration
-----------
Applied AFTER ConfidenceScorer.score() in main.py when
  config.yaml → features.confidence_calibration = true
"""

import logging
import math
from typing import List

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Sigmoid (Platt) calibrator for composite confidence scores.

    Parameters
    ----------
    a : float
        Slope parameter.  Negative values produce a standard sigmoid.
    b : float
        Intercept parameter.
    method : str
        "sigmoid" (Platt scaling) or "identity" (no-op / pass-through).
    """

    def __init__(
        self,
        a: float   = -4.0,
        b: float   =  2.0,
        method: str = "sigmoid",
    ) -> None:
        self.a      = a
        self.b      = b
        self.method = method
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, raw_score: float) -> float:
        """
        Map a raw composite confidence score ∈ [0, 1] to a calibrated
        probability ∈ [0, 1].

        The calibration is monotone: ranking order is preserved.
        """
        raw_score = float(min(max(raw_score, 0.0), 1.0))
        if self.method == "identity":
            return raw_score
        # Platt sigmoid
        try:
            val = 1.0 / (1.0 + math.exp(self.a * raw_score + self.b))
        except OverflowError:
            val = 0.0 if self.a * raw_score + self.b > 0 else 1.0
        return round(float(min(max(val, 0.0), 1.0)), 6)

    def fit(self, raw_scores: List[float], labels: List[int]) -> None:
        """
        Fit Platt parameters from labeled data.

        Parameters
        ----------
        raw_scores : list of composite confidence scores (floats in [0, 1]).
        labels     : list of binary outcomes (1 = correct/replied, 0 = wrong/escalated).
        """
        if len(raw_scores) != len(labels):
            raise ValueError("raw_scores and labels must have equal length")
        if len(raw_scores) < 4:
            logger.warning("[calibration] too few samples (%d) to fit; using defaults",
                           len(raw_scores))
            return

        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression

            X = np.array(raw_scores, dtype=float).reshape(-1, 1)
            y = np.array(labels, dtype=int)

            lr = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
            lr.fit(X, y)

            # Platt: P(y=1) = 1 / (1 + exp(A*x + B))
            # sklearn LogReg: coef_ and intercept_ give the decision function
            # decision = X @ coef_.T + intercept_  →  P(y=1) = sigmoid(decision)
            # sigmoid(decision) = 1/(1+exp(-decision)) = 1/(1+exp(A*x+B)) where A=-coef_, B=-intercept_
            self.a = float(-lr.coef_[0, 0])
            self.b = float(-lr.intercept_[0])
            self._fitted = True
            logger.info("[calibration] fitted Platt params: A=%.4f B=%.4f", self.a, self.b)

        except ImportError:
            logger.warning("[calibration] sklearn/numpy not available; fit() skipped")
        except Exception as exc:
            logger.warning("[calibration] fit() failed: %s; using defaults", exc)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        return self._fitted

    def params(self) -> dict:
        return {"method": self.method, "a": self.a, "b": self.b, "fitted": self._fitted}
