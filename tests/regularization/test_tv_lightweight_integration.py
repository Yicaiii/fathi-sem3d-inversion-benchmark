from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_tv_lightweight_validation() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "scripts.regularization.validate_tv_lightweight"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.returncode == 0, process.stdout + "\n" + process.stderr
    assert "RESULT = PASS_TV_LIGHTWEIGHT_VALIDATION" in process.stdout
