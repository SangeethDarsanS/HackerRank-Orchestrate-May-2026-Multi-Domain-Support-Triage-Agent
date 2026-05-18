"""
modules/drift_monitor.py — Detects distributional drift in ticket streams.

Tracks three distributions across a run:
  - intent distribution (per-ticket intent label)
  - risk distribution   (low | medium | high)
  - escalation rate     (fraction of escalated tickets)

Computes drift vs. a baseline using Total Variation Distance (TVD).
TVD ∈ [0, 1]; 0 = identical distributions.

Alert rule: if TVD for any distribution exceeds `threshold` (default 0.15),
a WARNING is logged and the drift report flags it.

Output file: drift_report.json
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# Default baseline distributions (uniform-ish, reflects expected healthy mix)
_DEFAULT_BASELINE: Dict[str, dict] = {
    "intents": {
        "account_access":   0.20,
        "billing_payment":  0.10,
        "technical_bug":    0.20,
        "feature_request":  0.05,
        "security_concern": 0.10,
        "api_integration":  0.10,
        "data_privacy":     0.10,
        "general_inquiry":  0.10,
        "out_of_scope":     0.05,
    },
    "risk_levels": {
        "low":    0.50,
        "medium": 0.35,
        "high":   0.15,
    },
    "escalation_rate": 0.40,   # scalar baseline
}

_DEFAULT_THRESHOLD = 0.15   # TVD threshold for alerting


def _total_variation_distance(p: dict, q: dict) -> float:
    """
    Compute TVD = 0.5 * Σ |p(x) - q(x)| over the union of keys.
    Both dicts are treated as unnormalised; they are normalised internally.
    """
    all_keys = set(p) | set(q)
    if not all_keys:
        return 0.0

    # Normalise
    p_total = sum(p.values()) or 1.0
    q_total = sum(q.values()) or 1.0
    p_norm  = {k: p.get(k, 0.0) / p_total for k in all_keys}
    q_norm  = {k: q.get(k, 0.0) / q_total for k in all_keys}

    return 0.5 * sum(abs(p_norm[k] - q_norm[k]) for k in all_keys)


class DriftMonitor:
    """
    Records per-ticket distributions and detects drift vs. a baseline.

    Parameters
    ----------
    baseline      : dict with keys "intents", "risk_levels", "escalation_rate".
                    Defaults to _DEFAULT_BASELINE when None.
    threshold     : TVD threshold above which a drift alert is raised.
    baseline_path : Optional path to a JSON file where the baseline is
                    persisted across runs.  When the file exists its contents
                    are used as the baseline (overrides the ``baseline``
                    parameter).  After every ``save_report()`` call the file is
                    updated as a running mean so the baseline adapts to
                    real observed traffic rather than staying hardcoded.
    """

    def __init__(
        self,
        baseline: Optional[dict] = None,
        threshold: float = _DEFAULT_THRESHOLD,
        baseline_path: Optional[Path] = None,
    ) -> None:
        self._threshold    = threshold
        self._baseline_path = baseline_path
        self._intents:     Counter = Counter()
        self._risk_levels: Counter = Counter()
        self._actions:     Counter = Counter()
        self._total        = 0

        # Try loading a persisted baseline first, then fall back to the
        # caller-supplied value, and finally to the hardcoded default.
        self._baseline = self._load_baseline(baseline_path) or baseline or _DEFAULT_BASELINE

    # ------------------------------------------------------------------
    # Baseline persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_baseline(path: Optional[Path]) -> Optional[dict]:
        """Load baseline from *path* if it exists and is valid; else None."""
        if path is None or not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Validate expected keys
            if "intents" in data and "risk_levels" in data:
                logger.info("[drift] Loaded dynamic baseline from %s", path)
                return data
        except Exception as exc:
            logger.warning("[drift] Could not load baseline from %s: %s", path, exc)
        return None

    def _update_baseline_file(self, path: Path) -> None:
        """
        Update the persisted baseline as a running mean of all previous runs
        plus the current one.

        The file stores accumulated sums and a run count so the mean can be
        recomputed exactly without holding the full history.
        """
        if self._total == 0:
            return
        try:
            # Load existing accumulator or start fresh
            acc: dict = {}
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        acc = json.load(f)
                except Exception:
                    acc = {}

            n_prev    = int(acc.get("_n_runs", 0))
            n_new     = n_prev + 1
            total_t   = max(self._total, 1)

            # Current run's normalised distributions
            cur_intents    = {k: v / total_t for k, v in self._intents.items()}
            cur_risk       = {k: v / total_t for k, v in self._risk_levels.items()}
            cur_esc_rate   = self._actions.get("escalated", 0) / total_t

            # Merge into running mean
            def _update_dist(old: dict, cur: dict, n: int) -> dict:
                """Running mean: new_mean[k] = (old_mean[k]*(n-1) + cur[k]) / n"""
                all_keys = set(old) | set(cur)
                return {
                    k: round(
                        (old.get(k, 0.0) * (n - 1) + cur.get(k, 0.0)) / n, 6
                    )
                    for k in all_keys
                }

            new_intents = _update_dist(
                acc.get("intents", {}), cur_intents, n_new
            )
            new_risk = _update_dist(
                acc.get("risk_levels", {}), cur_risk, n_new
            )
            old_esc  = float(acc.get("escalation_rate", cur_esc_rate))
            new_esc  = round((old_esc * (n_new - 1) + cur_esc_rate) / n_new, 6)

            updated = {
                "_n_runs":        n_new,
                "intents":        new_intents,
                "risk_levels":    new_risk,
                "escalation_rate": new_esc,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(updated, f, indent=2)
            logger.info("[drift] Baseline updated (%d runs) at %s", n_new, path)
        except Exception as exc:
            logger.warning("[drift] Could not update baseline file %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record(self, intent: str, risk_level: str, action: str) -> None:
        """Record one ticket's distribution contribution."""
        self._intents[intent]         += 1
        self._risk_levels[risk_level] += 1
        self._actions[action]         += 1
        self._total                   += 1

    # ------------------------------------------------------------------
    # Drift computation
    # ------------------------------------------------------------------

    def compute_drift(self) -> dict:
        """
        Compare current batch distributions to baseline.

        Returns a dict with:
          intent_drift       float [0, 1] TVD
          risk_drift         float [0, 1] TVD
          escalation_drift   float absolute difference in escalation rate
          alerts             list[str]  human-readable warnings
          drifted            bool
        """
        if self._total == 0:
            return {
                "intent_drift": 0.0, "risk_drift": 0.0,
                "escalation_drift": 0.0, "alerts": [], "drifted": False,
                "n_tickets": 0,
            }

        # Intent TVD
        intent_baseline = self._baseline.get("intents", {})
        intent_tvd = _total_variation_distance(dict(self._intents), intent_baseline)

        # Risk TVD
        risk_baseline = self._baseline.get("risk_levels", {})
        risk_tvd = _total_variation_distance(dict(self._risk_levels), risk_baseline)

        # Escalation rate difference
        current_esc  = self._actions.get("escalated", 0) / self._total
        baseline_esc = float(self._baseline.get("escalation_rate", 0.40))
        esc_drift    = abs(current_esc - baseline_esc)

        alerts = []
        if intent_tvd > self._threshold:
            msg = (
                f"Intent distribution drift detected (TVD={intent_tvd:.3f} "
                f"> threshold={self._threshold})."
            )
            logger.warning("[drift] %s", msg)
            alerts.append(msg)

        if risk_tvd > self._threshold:
            msg = (
                f"Risk distribution drift detected (TVD={risk_tvd:.3f} "
                f"> threshold={self._threshold})."
            )
            logger.warning("[drift] %s", msg)
            alerts.append(msg)

        if esc_drift > self._threshold:
            msg = (
                f"Escalation rate drift detected "
                f"(current={current_esc:.2f} baseline={baseline_esc:.2f} "
                f"delta={esc_drift:.3f} > threshold={self._threshold})."
            )
            logger.warning("[drift] %s", msg)
            alerts.append(msg)

        drifted = bool(alerts)
        return {
            "intent_drift":     round(intent_tvd, 4),
            "risk_drift":       round(risk_tvd, 4),
            "escalation_drift": round(esc_drift, 4),
            "escalation_rate":  round(current_esc, 4),
            "alerts":           alerts,
            "drifted":          drifted,
            "n_tickets":        self._total,
        }

    # ------------------------------------------------------------------
    # Current distributions (for reporting)
    # ------------------------------------------------------------------

    def current_distributions(self) -> dict:
        total = max(self._total, 1)
        return {
            "intents":     {k: round(v / total, 4) for k, v in self._intents.items()},
            "risk_levels": {k: round(v / total, 4) for k, v in self._risk_levels.items()},
            "actions":     {k: round(v / total, 4) for k, v in self._actions.items()},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, path: Path) -> None:
        """Write drift_report.json to ``path``.

        Also updates the persisted baseline file (if ``baseline_path`` was
        provided at construction) so future runs compare against real observed
        distributions rather than the hardcoded default.
        """
        drift = self.compute_drift()
        report = {
            "timestamp":           datetime.now(tz=_IST).isoformat(timespec="seconds"),
            "n_tickets":           self._total,
            "threshold":           self._threshold,
            "drift_detected":      drift["drifted"],
            "intent_tvd":          drift["intent_drift"],
            "risk_tvd":            drift["risk_drift"],
            "escalation_drift":    drift["escalation_drift"],
            "escalation_rate":     drift["escalation_rate"],
            "alerts":              drift["alerts"],
            "current_distributions": self.current_distributions(),
            "baseline":            {
                k: v for k, v in self._baseline.items()
                if k != "escalation_rate"
            },
            "baseline_source": (
                "dynamic" if (
                    self._baseline_path is not None
                    and self._baseline_path.exists()
                )
                else "default"
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("[drift] report saved to %s", path)

        # Update persisted baseline with this run's data
        if self._baseline_path is not None:
            self._update_baseline_file(self._baseline_path)
