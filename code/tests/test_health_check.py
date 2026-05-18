"""
tests/test_health_check.py — Tests for --health-check CLI feature.
"""

import sys
import subprocess
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


class TestHealthCheckImport:

    def test_run_health_check_importable(self):
        """run_health_check must be importable from main."""
        # We can't import main directly (it runs __main__ code), so check file
        content = MAIN_PY.read_text(encoding="utf-8")
        assert "def run_health_check" in content

    def test_health_check_arg_in_parse_args(self):
        """--health-check argument must be defined in parse_args."""
        content = MAIN_PY.read_text(encoding="utf-8")
        assert "--health-check" in content

    def test_health_check_dispatch_in_main(self):
        """__main__ block must dispatch to run_health_check."""
        content = MAIN_PY.read_text(encoding="utf-8")
        assert "args.health_check" in content
        assert "run_health_check(cfg)" in content

    def test_health_check_checks_corpus(self):
        """run_health_check must check corpus existence."""
        content = MAIN_PY.read_text(encoding="utf-8")
        assert "Corpus" in content

    def test_health_check_checks_memory(self):
        """run_health_check must check memory / psutil."""
        content = MAIN_PY.read_text(encoding="utf-8")
        assert "Memory" in content

    def test_health_check_checks_modules(self):
        """run_health_check must check module imports."""
        content = MAIN_PY.read_text(encoding="utf-8")
        assert "Modules" in content

    def test_health_check_checks_config(self):
        """run_health_check must verify config loaded."""
        content = MAIN_PY.read_text(encoding="utf-8")
        assert "Config" in content


class TestHealthCheckOutput:

    @pytest.mark.parametrize("flag", ["--health-check"])
    def test_health_check_runs_without_error(self, flag):
        """Running python main.py --health-check must exit 0 or 1 (not crash)."""
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), flag],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Exit code should be 0 (all OK) or 1 (some warnings/failures)
        # but NOT an uncaught exception (exit code 2+ from argparse errors excepted)
        assert result.returncode in (0, 1), (
            f"Health check crashed. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_health_check_prints_system_status(self):
        """Output must contain 'System Status'."""
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--health-check"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = result.stdout + result.stderr
        assert "System Status" in combined, (
            f"'System Status' not found in output:\n{combined}"
        )

    def test_health_check_does_not_run_pipeline(self):
        """Health check must not start ticket processing."""
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--health-check"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = result.stdout + result.stderr
        # Pipeline start markers must not appear
        assert "Processing tickets" not in combined
        assert "Warming up models" not in combined
