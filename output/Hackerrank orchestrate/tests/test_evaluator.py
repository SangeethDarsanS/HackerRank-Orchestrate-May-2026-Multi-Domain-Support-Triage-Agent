"""
Unit tests for the new save_report / generate_flat_report additions to
modules/evaluator.py.
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd

from modules.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list, columns: list) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# generate_flat_report
# ---------------------------------------------------------------------------

def test_flat_report_has_required_keys():
    ev = Evaluator()
    results = {
        "n_samples": 10,
        "status": {"accuracy": 0.9, "precision": 0.88, "recall": 0.87, "f1": 0.875},
        "escalation_rate": 0.2,
        "avg_confidence": 0.72,
        "latency_ms": {"mean": 310.0, "median": 305.0, "p95": 400.0},
        "request_type": {"accuracy": 0.85, "f1_macro": 0.83},
        "product_area": {"accuracy": 0.78},
    }
    flat = ev.generate_flat_report(results)
    for key in ("accuracy", "precision", "recall", "f1",
                "escalation_rate", "avg_confidence", "avg_latency_ms"):
        assert key in flat, f"Missing: {key}"


def test_flat_report_values_are_floats():
    ev = Evaluator()
    results = {
        "n_samples": 5,
        "status": {"accuracy": 0.8, "precision": 0.75, "recall": 0.82, "f1": 0.78},
        "escalation_rate": 0.4,
    }
    flat = ev.generate_flat_report(results)
    assert isinstance(flat["accuracy"],  float)
    assert isinstance(flat["precision"], float)
    assert isinstance(flat["recall"],    float)
    assert isinstance(flat["f1"],        float)


def test_flat_report_run_metrics_merged():
    ev = Evaluator()
    results = {"n_samples": 3, "status": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}}
    run_metrics = {
        "ticket_count": 3,
        "escalation_count": 1,
        "error_count": 0,
        "peak_memory_mb": 512.5,
        "average_latency_ms": 220.0,
    }
    flat = ev.generate_flat_report(results, run_metrics=run_metrics)
    assert flat["ticket_count"]    == 3
    assert flat["escalation_count"] == 1
    assert flat["peak_memory_mb"]  == 512.5


def test_flat_report_latency_from_run_metrics_when_no_latency_col():
    ev = Evaluator()
    results = {"n_samples": 2, "status": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}}
    run_metrics = {"average_latency_ms": 350.0}
    flat = ev.generate_flat_report(results, run_metrics=run_metrics)
    assert flat["avg_latency_ms"] == 350.0


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def test_save_report_creates_valid_json():
    ev = Evaluator()
    results = {
        "n_samples": 5,
        "status": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
        "escalation_rate": 0.2,
    }
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "evaluation_report.json"
        ev.save_report(results, out_path)
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "escalation_rate" in data


def test_save_report_json_serialisable():
    ev = Evaluator()
    results = {
        "n_samples": 1,
        "status": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "escalation_rate": 0.0,
        "avg_confidence": 0.80,
        "latency_ms": {"mean": 10.0, "median": 9.0, "p95": 12.0},
    }
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "report.json"
        ev.save_report(results, out_path)
        data = json.loads(out_path.read_text())
        assert isinstance(data, dict)


def test_save_report_creates_parent_dir():
    ev = Evaluator()
    results = {"n_samples": 0, "status": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}}
    with tempfile.TemporaryDirectory() as td:
        nested = Path(td) / "a" / "b" / "report.json"
        ev.save_report(results, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# evaluate() integration (using temp CSVs)
# ---------------------------------------------------------------------------

def _minimal_pred_csv(path: Path, status="replied", request_type="bug"):
    _write_csv(
        path,
        [{"issue": "x", "status": status, "request_type": request_type}],
        ["issue", "status", "request_type"],
    )


def _minimal_gt_csv(path: Path, status="replied", request_type="bug"):
    _write_csv(
        path,
        [{"issue": "x", "status": status, "request_type": request_type}],
        ["issue", "status", "request_type"],
    )


def test_evaluate_perfect_agreement():
    ev = Evaluator()
    with tempfile.TemporaryDirectory() as td:
        pred = Path(td) / "pred.csv"
        gt   = Path(td) / "gt.csv"
        _minimal_pred_csv(pred)
        _minimal_gt_csv(gt)
        results = ev.evaluate(pred, gt)
        assert results["status"]["accuracy"] == 1.0


def test_evaluate_total_disagreement():
    ev = Evaluator()
    with tempfile.TemporaryDirectory() as td:
        pred = Path(td) / "pred.csv"
        gt   = Path(td) / "gt.csv"
        _minimal_pred_csv(pred, status="replied")
        _minimal_gt_csv(gt,    status="escalated")
        results = ev.evaluate(pred, gt)
        assert results["status"]["accuracy"] == 0.0
