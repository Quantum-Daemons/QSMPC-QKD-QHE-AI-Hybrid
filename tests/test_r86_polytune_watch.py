"""The scheduled check of Polytune upstream fails on what it exists to catch, and on nothing else (Round 86, audit item D3).

The pin is watched by `tests/test_the_polytune_integration_reports_what_happened.py`. This watches the other direction: a
change to the protocol's source past the pin, or an answer on `sine-fdn/polytune` issues 63 and 100, which the threat model
waits on. It uses a stand-in for the GitHub API, so it needs no network and the failure paths are all exercised.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_deps.sh"
WORKFLOW = REPO / ".github" / "workflows" / "audit.yml"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="the check reads the responses with node")


def _check(tmp_path: Path, *, issue63: tuple[str, int] = ("open", 2), issue100: tuple[str, int] = ("open", 0),
           files: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    issue = lambda state, comments: json.dumps({"state": state, "comments": comments})   # noqa: E731
    compare = json.dumps({"ahead_by": len(files), "files": [{"filename": f} for f in files]})
    stub = tmp_path / "api.sh"
    stub.write_text("#!/bin/sh\ncase \"$1\" in\n"
                    f"  */issues/63) cat <<'J'\n{issue(*issue63)}\nJ\n  ;;\n"
                    f"  */issues/100) cat <<'J'\n{issue(*issue100)}\nJ\n  ;;\n"
                    f"  */compare/*) cat <<'J'\n{compare}\nJ\n  ;;\nesac\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return subprocess.run(["sh", str(SCRIPT), "polytune"], capture_output=True, text=True, timeout=60, cwd=REPO,
                          env={**os.environ, "POLYTUNE_API": str(stub)})


def test_nothing_changed_passes(tmp_path: Path) -> None:
    r = _check(tmp_path)
    assert r.returncode == 0 and "no protocol source among them" in r.stdout, r.stdout + r.stderr


def test_dependency_bumps_and_test_utilities_do_not_count(tmp_path: Path) -> None:
    r = _check(tmp_path, files=("Cargo.lock", ".github/workflows/ci.yml", "crates/polytune-test-utils/src/peak_alloc.rs"))
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize(("kwargs", "says"), [
    ({"issue63": ("open", 3)}, "issue #63"),
    ({"issue63": ("closed", 2)}, "issue #63"),
    ({"issue100": ("open", 1)}, "issue #100"),
    ({"issue100": ("closed", 0)}, "issue #100"),
    ({"files": ("src/mpc/faand.rs",)}, "src/mpc/faand.rs"),
    ({"files": ("crates/polytune-core/src/ot.rs",)}, "crates/polytune-core/src/ot.rs"),
], ids=["a reply on 63", "63 closed", "a comment on 100", "100 closed", "the crate's source", "a member crate's source"])
def test_what_it_exists_for_fails_and_says_which(tmp_path: Path, kwargs: dict, says: str) -> None:
    r = _check(tmp_path, **kwargs)
    assert r.returncode == 1 and says in r.stderr and "docs/threat-model.md" in r.stderr, r.stdout + r.stderr


def test_it_runs_on_the_schedule() -> None:
    script, workflow = SCRIPT.read_text(), WORKFLOW.read_text()
    assert "polytune) audit_polytune ;;" in script and "audit_polytune && audit_python" in script
    assert "bash scripts/audit_deps.sh polytune" in workflow and 'cron: "17 3 * * *"' in workflow
    assert "scripts/audit_deps.sh polytune" in " ".join((REPO / "docs" / "threat-model.md").read_text().split())
