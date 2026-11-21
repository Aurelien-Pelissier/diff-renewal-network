"""End-to-end reproducibility test checking SHA-256 archives and figure assets."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_reproduce_all_script():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce_all.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(proc.stdout)
    assert summary["status"] == "PASSED"
    assert summary["verified_archive_files"] == 23
    assert summary["erlang16_replay"]["nominal_max_error"] < 1e-12
    assert summary["stem_cell_replay"]["abs_drift"] < 1e-8
