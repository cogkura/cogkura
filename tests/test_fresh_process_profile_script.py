"""The fresh-process fixture preserves results after reopening the database."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_fresh_process_profile_reopens_and_checks_output(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "profile_fresh_process.py"
    output = tmp_path / "fresh.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--observations",
            "3",
            "--revisions",
            "2",
            "--processes",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    report = json.loads(output.read_text())
    assert len(report["samples"]) == 2
    for sample in report["samples"]:
        assert len(sample["recalls"]) == 2
        assert all(recall["outputs"] == report["fixture_output"] for recall in sample["recalls"])
