"""
Evaluator: compares model output against ground-truth labels.

Metrics computed per column:
  - status        : accuracy, precision, recall, F1 (binary: replied vs escalated)
  - request_type  : accuracy, macro precision/recall/F1
  - product_area  : accuracy (exact match)
  - response      : (qualitative — not auto-scored here)

Also computes:
  - escalation_rate
  - avg_confidence   (if confidence column present)
  - avg_latency_ms   (if latency column present)

Backtesting
-----------
Call ``save_report(results, path)`` to persist the flat evaluation_report.json
that the backtest mode produces.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class Evaluator:
    """Compare output.csv against sample_support_tickets.csv (ground truth)."""

    def evaluate(
        self,
        predictions_path: Path,
        ground_truth_path: Path,
    ) -> dict:
        pred_df = pd.read_csv(predictions_path, dtype=str).fillna("")
        gt_df = pd.read_csv(ground_truth_path, dtype=str).fillna("")

        # Normalise column names
        pred_df.columns = [c.lower().strip().replace(" ", "_") for c in pred_df.columns]
        gt_df.columns = [c.lower().strip().replace(" ", "_") for c in gt_df.columns]

        # Align on issue text (inner join by index if same length)
        n = min(len(pred_df), len(gt_df))
        pred = pred_df.iloc[:n].reset_index(drop=True)
        gt = gt_df.iloc[:n].reset_index(drop=True)

        # --- Coverage: identify rows where ground-truth labels are present ---
        # A row is considered labelled when its "status" column is non-empty.
        status_col_gt = next((c for c in gt.columns if "status" in c), None)
        if status_col_gt:
            labelled_mask = gt[status_col_gt].str.strip().str.lower().ne("")
        else:
            labelled_mask = pd.Series([True] * n)  # assume all labelled

        labelled_count = int(labelled_mask.sum())
        coverage_ratio = round(labelled_count / n, 4) if n > 0 else 0.0

        # Filter both frames to labelled rows for metric computation
        pred_labelled = pred[labelled_mask].reset_index(drop=True)
        gt_labelled   = gt[labelled_mask].reset_index(drop=True)
        n_labelled    = len(pred_labelled)

        results: dict = {
            "n_samples":             n,
            "n_labelled":            n_labelled,
            "ground_truth_coverage": coverage_ratio,
        }

        # --- Status (replied / escalated) — evaluated on labelled rows only ---
        if "status" in pred_labelled.columns and status_col_gt in gt_labelled.columns:
            y_pred = pred_labelled["status"].str.lower().str.strip()
            y_true = gt_labelled[status_col_gt].str.lower().str.strip()
            results["status"] = self._binary_metrics(y_true, y_pred, pos_label="replied")

        # --- Request type ---
        pred_rt_col = next((c for c in pred_labelled.columns if "request" in c), None)
        gt_rt_col   = next((c for c in gt_labelled.columns if "request" in c), None)
        if pred_rt_col and gt_rt_col:
            y_pred = pred_labelled[pred_rt_col].str.lower().str.strip()
            y_true = gt_labelled[gt_rt_col].str.lower().str.strip()
            # Skip rows where ground-truth request_type is also empty
            rt_mask = y_true.ne("")
            if rt_mask.any():
                results["request_type"] = self._multiclass_metrics(
                    y_true[rt_mask], y_pred[rt_mask]
                )

        # --- Product area ---
        pred_pa_col = next((c for c in pred_labelled.columns if "product" in c or "area" in c), None)
        gt_pa_col   = next((c for c in gt_labelled.columns if "product" in c or "area" in c), None)
        if pred_pa_col and gt_pa_col:
            y_pred = pred_labelled[pred_pa_col].str.lower().str.strip()
            y_true = gt_labelled[gt_pa_col].str.lower().str.strip()
            pa_mask = y_true.ne("")
            if pa_mask.any():
                results["product_area"] = {
                    "accuracy": float(accuracy_score(y_true[pa_mask], y_pred[pa_mask]))
                }

        # --- Escalation rate (over all predictions, not just labelled) ---
        if "status" in pred.columns:
            esc = (pred["status"].str.lower() == "escalated").sum()
            results["escalation_rate"] = float(esc / n)

        # --- Average confidence (if present in predictions) ---
        conf_col = next(
            (c for c in pred.columns if "confidence" in c),
            None,
        )
        if conf_col:
            confs = pd.to_numeric(pred[conf_col], errors="coerce").dropna()
            results["avg_confidence"] = float(confs.mean()) if len(confs) else 0.0

        # --- Latency (if present) ---
        lat_col = next(
            (c for c in pred.columns if "latency" in c),
            None,
        )
        if lat_col:
            lats = pd.to_numeric(pred[lat_col], errors="coerce").dropna()
            results["latency_ms"] = {
                "mean":   float(lats.mean()),
                "median": float(lats.median()),
                "p95":    float(np.percentile(lats, 95)) if len(lats) else 0.0,
            }

        return results

    # ------------------------------------------------------------------
    # Flat report helpers (for evaluation_report.json)
    # ------------------------------------------------------------------

    def generate_flat_report(
        self,
        results: dict,
        run_metrics: Optional[dict] = None,
    ) -> dict:
        """
        Produce a flat, JSON-serialisable report dictionary.

        Shape matches the spec:
          accuracy, precision, recall, f1,
          escalation_rate, avg_confidence, avg_latency_ms
        """
        status = results.get("status", {})
        lat = results.get("latency_ms", {})

        flat = {
            "n_samples":              results.get("n_samples", 0),
            "n_labelled":             results.get("n_labelled", results.get("n_samples", 0)),
            "ground_truth_coverage":  round(results.get("ground_truth_coverage", 1.0), 4),
            "accuracy":               round(status.get("accuracy", 0.0), 4),
            "precision":              round(status.get("precision", 0.0), 4),
            "recall":                 round(status.get("recall", 0.0), 4),
            "f1":                     round(status.get("f1", 0.0), 4),
            "escalation_rate":        round(results.get("escalation_rate", 0.0), 4),
            "avg_confidence":         round(results.get("avg_confidence", 0.0), 4),
            "avg_latency_ms":         round(lat.get("mean", 0.0), 2),
        }

        # request_type metrics
        rt = results.get("request_type", {})
        flat["request_type_accuracy"]   = round(rt.get("accuracy", 0.0), 4)
        flat["request_type_f1_macro"]   = round(rt.get("f1_macro", 0.0), 4)

        # product_area accuracy
        pa = results.get("product_area", {})
        flat["product_area_accuracy"] = round(pa.get("accuracy", 0.0), 4)

        # Merge run-level observability metrics if provided
        if run_metrics:
            flat["ticket_count"]    = run_metrics.get("ticket_count", 0)
            flat["escalation_count"] = run_metrics.get("escalation_count", 0)
            flat["error_count"]     = run_metrics.get("error_count", 0)
            flat["peak_memory_mb"]  = run_metrics.get("peak_memory_mb", 0.0)
            # prefer run_metrics latency if no latency column in CSV
            if flat["avg_latency_ms"] == 0.0:
                flat["avg_latency_ms"] = round(
                    run_metrics.get("average_latency_ms", 0.0), 2
                )

        return flat

    def save_report(
        self,
        results: dict,
        output_path: Path,
        run_metrics: Optional[dict] = None,
    ) -> None:
        """
        Generate and save evaluation_report.json to ``output_path``.

        Parameters
        ----------
        results     : dict returned by ``evaluate()``.
        output_path : destination path (parent directory must exist).
        run_metrics : optional dict from MetricsTracker.snapshot().
        """
        flat = self.generate_flat_report(results, run_metrics=run_metrics)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(flat, f, indent=2)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------

    def print_report(self, results: dict) -> None:
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        n_samples  = results.get("n_samples", 0)
        n_labelled = results.get("n_labelled", n_samples)
        coverage   = results.get("ground_truth_coverage", 1.0)
        print(f"Samples total     : {n_samples}")
        print(f"Labelled rows     : {n_labelled}  (coverage {coverage:.0%})")
        print(f"Escalation rate   : {results.get('escalation_rate', 0):.1%}")
        if "avg_confidence" in results:
            print(f"Avg confidence    : {results.get('avg_confidence', 0):.4f}")
        print()

        for key in ("status", "request_type", "product_area"):
            if key not in results:
                continue
            m = results[key]
            print(f"--- {key.upper()} ---")
            print(f"  Accuracy  : {m.get('accuracy', 0):.3f}")
            if "precision" in m:
                print(f"  Precision : {m.get('precision', 0):.3f}")
                print(f"  Recall    : {m.get('recall', 0):.3f}")
                print(f"  F1        : {m.get('f1', 0):.3f}")
            if "precision_macro" in m:
                print(f"  Precision (macro): {m.get('precision_macro', 0):.3f}")
                print(f"  Recall (macro)   : {m.get('recall_macro', 0):.3f}")
                print(f"  F1 (macro)       : {m.get('f1_macro', 0):.3f}")
            if "report" in m:
                print()
                print(m["report"])

        if "latency_ms" in results:
            lat = results["latency_ms"]
            print(f"--- LATENCY (ms) ---")
            print(f"  Mean   : {lat['mean']:.1f}")
            print(f"  Median : {lat['median']:.1f}")
            print(f"  P95    : {lat['p95']:.1f}")

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Internal metric helpers
    # ------------------------------------------------------------------

    def _binary_metrics(self, y_true, y_pred, pos_label: str) -> dict:
        classes = sorted(set(y_true.tolist() + y_pred.tolist()))
        try:
            return {
                "accuracy":  float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
                "recall":    float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
                "f1":        float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
                "report":    classification_report(y_true, y_pred, labels=classes, zero_division=0),
            }
        except Exception as e:
            return {"error": str(e)}

    def _multiclass_metrics(self, y_true, y_pred) -> dict:
        classes = sorted(set(y_true.tolist() + y_pred.tolist()))
        try:
            return {
                "accuracy":        float(accuracy_score(y_true, y_pred)),
                "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "report":          classification_report(y_true, y_pred, labels=classes, zero_division=0),
            }
        except Exception as e:
            return {"error": str(e)}
