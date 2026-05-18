"""
Unit tests for W003 fix: ground_truth_coverage and partial-label handling
in modules/evaluator.py
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

def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


_PRED_ROWS = [
    {"issue": "issue 1", "status": "replied",   "request_type": "product_issue", "product_area": "screen"},
    {"issue": "issue 2", "status": "escalated", "request_type": "bug",           "product_area": "privacy"},
    {"issue": "issue 3", "status": "replied",   "request_type": "product_issue", "product_area": "billing"},
    {"issue": "issue 4", "status": "escalated", "request_type": "invalid",       "product_area": "general_support"},
]

_GT_FULL = [
    {"issue": "issue 1", "Status": "replied",   "Request Type": "product_issue", "Product Area": "screen"},
    {"issue": "issue 2", "Status": "escalated", "Request Type": "bug",           "Product Area": "privacy"},
    {"issue": "issue 3", "Status": "replied",   "Request Type": "product_issue", "Product Area": "billing"},
    {"issue": "issue 4", "Status": "escalated", "Request Type": "invalid",       "Product Area": "general_support"},
]

_GT_PARTIAL = [
    {"issue": "issue 1", "Status": "replied",   "Request Type": "product_issue", "Product Area": "screen"},
    {"issue": "issue 2", "Status": "",           "Request Type": "",              "Product Area": ""},
    {"issue": "issue 3", "Status": "replied",    "Request Type": "product_issue", "Product Area": "billing"},
    {"issue": "issue 4", "Status": "",           "Request Type": "",              "Product Area": ""},
]


# ---------------------------------------------------------------------------
# ground_truth_coverage when all rows labelled
# ---------------------------------------------------------------------------

def test_full_coverage_is_one():
    with tempfile.TemporaryDirectory() as td:
        pred_path = Path(td) / "pred.csv"
        gt_path   = Path(td) / "gt.csv"
        _write_csv(pred_path, _PRED_ROWS)
        _write_csv(gt_path, _GT_FULL)
        ev = Evaluator()
        results = ev.evaluate(pred_path, gt_path)
        assert results["ground_truth_coverage"] == 1.0


def test_full_coverage_n_labelled_equals_n_samples():
    with tempfile.TemporaryDirectory() as td:
        pred_path = Path(td) / "pred.csv"
        gt_path   = Path(td) / "gt.csv"
        _write_csv(pred_path, _PRED_ROWS)
        _write_csv(gt_path, _GT_FULL)
        ev = Evaluator()
        results = ev.evaluate(pred_path, gt_path)
        assert results["n_labelled"] == results["n_samples"]


# ---------------------------------------------------------------------------
# ground_truth_coverage when only 2 of 4 rows labelled
# ---------------------------------------------------------------------------

def test_partial_coverage_ratio():
    with tempfile.TemporaryDirectory() as td:
        pred_path = Path(td) / "pred.csv"
        gt_path   = Path(td) / "gt.csv"
        _write_csv(pred_path, _PRED_ROWS)
        _write_csv(gt_path, _GT_PARTIAL)
        ev = Evaluator()
        results = ev.evaluate(pred_path, gt_path)
        assert results["ground_truth_coverage"] == pytest.approx(0.5, abs=1e-4)


def test_partial_coverage_n_labelled():
    with tempfile.TemporaryDirectory() as td:
        pred_path = Path(td) / "pred.csv"
        gt_path   = Path(td) / "gt.csv"
        _write_csv(pred_path, _PRED_ROWS)
        _write_csv(gt_path, _GT_PARTIAL)
        ev = Evaluator()
        results = ev.evaluate(pred_path, gt_path)
        assert results["n_labelled"] == 2


# ---------------------------------------------------------------------------
# Metrics are computed only on labelled rows
# ---------------------------------------------------------------------------

def test_partial_coverage_accuracy_on_labelled_only():
    """With 2 labelled rows (both matching), accuracy should be 1.0."""
    with tempfile.TemporaryDirectory() as td:
        pred_path = Path(td) / "pred.csv"
        gt_path   = Path(td) / "gt.csv"
        _write_csv(pred_path, _PRED_ROWS)
        _write_csv(gt_path, _GT_PARTIAL)
        ev = Evaluator()
        results = ev.evaluate(pred_path, gt_path)
        # Labelled rows 0 and 2: pred replied/replied vs gt replied/replied → acc 1.0
        assert results["status"]["accuracy"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# flat report contains ground_truth_coverage
# ---------------------------------------------------------------------------

def test_flat_report_has_coverage_key():
    with tempfile.TemporaryDirectory() as td:
        pred_path = Path(td) / "pred.csv"
        gt_path   = Path(td) / "gt.csv"
        _write_csv(pred_path, _PRED_ROWS)
        _write_csv(gt_path, _GT_FULL)
        ev = Evaluator()
        results = ev.evaluate(pred_path, gt_path)
        flat = ev.generate_flat_report(results)
        assert "ground_truth_coverage" in flat
        assert flat["ground_truth_coverage"] == 1.0


def test_flat_report_partial_coverage_value():
    with tempfile.TemporaryDirectory() as td:
        pred_path = Path(td) / "pred.csv"
        gt_path   = Path(td) / "gt.csv"
        _write_csv(pred_path, _PRED_ROWS)
        _write_csv(gt_path, _GT_PARTIAL)
        ev = Evaluator()
        results = ev.evaluate(pred_path, gt_path)
        flat = ev.generate_flat_report(results)
        assert flat["ground_truth_coverage"] == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# save_report writes coverage to JSON
# ---------------------------------------------------------------------------

def test_save_report_coverage_in_json():
    with tempfile.TemporaryDirectory() as td:
        pred_path   = Path(td) / "pred.csv"
        gt_path     = Path(td) / "gt.csv"
        report_path = Path(td) / "report.json"
        _write_csv(pred_path, _PRED_ROWS)
        _write_csv(gt_path, _GT_FULL)
        ev = Evaluator()
        results = ev.evaluate(pred_path, gt_path)
        ev.save_report(results, report_path)
        data = json.loads(report_path.read_text())
        assert "ground_truth_coverage" in data
        assert data["ground_truth_coverage"] == 1.0


# ---------------------------------------------------------------------------
# full_ground_truth.csv exists and has correct row count
# ---------------------------------------------------------------------------

def test_full_ground_truth_csv_exists():
    gt_path = Path(__file__).resolve().parent.parent.parent / \
              "support_tickets" / "full_ground_truth.csv"
    assert gt_path.exists(), f"full_ground_truth.csv not found at {gt_path}"


def test_full_ground_truth_csv_row_count():
    gt_path = Path(__file__).resolve().parent.parent.parent / \
              "support_tickets" / "full_ground_truth.csv"
    if gt_path.exists():
        df = pd.read_csv(gt_path, dtype=str).fillna("")
        assert len(df) == 29, f"Expected 29 rows, got {len(df)}"


def test_full_ground_truth_csv_has_required_columns():
    gt_path = Path(__file__).resolve().parent.parent.parent / \
              "support_tickets" / "full_ground_truth.csv"
    if gt_path.exists():
        df = pd.read_csv(gt_path, dtype=str).fillna("")
        cols = [c.lower().strip() for c in df.columns]
        assert "status" in cols
        assert any("request" in c for c in cols)
        assert any("product" in c or "area" in c for c in cols)
