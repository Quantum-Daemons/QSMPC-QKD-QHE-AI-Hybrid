"""Every tracked shell script parses.

WHY THIS EXISTS (Round 86). An apostrophe inside a `${VAR:?message}` broke `scripts/deploy_demo.sh`, and all 18
gates passed, because nothing runs that script before it is needed. `bash -n` costs nothing and would have
said so.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _scripts() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.sh"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return out.split()


def test_there_are_scripts_to_check() -> None:
    assert "scripts/deploy_demo.sh" in _scripts() and "scripts/ci_local.sh" in _scripts()


@pytest.mark.parametrize("script", _scripts())
def test_it_parses(script: str) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash absent")
    r = subprocess.run(["bash", "-n", script], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_check_can_fail(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sh"
    bad.write_text(": \"${X:?the host's network}\"\nage=$(echo \"a 'b' c\")\n")
    assert subprocess.run(["bash", "-n", str(bad)], capture_output=True).returncode != 0
