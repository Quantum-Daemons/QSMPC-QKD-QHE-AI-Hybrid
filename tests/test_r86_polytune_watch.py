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
           files: tuple[str | tuple[str, str], ...] = (), patch_bytes: int = 0) -> subprocess.CompletedProcess[str]:
    issue = lambda state, comments: json.dumps({"state": state, "comments": comments})   # noqa: E731
    entries = [(f, "0" * 40) if isinstance(f, str) else f for f in files]
    compare = json.dumps({"ahead_by": len(entries),
                          "files": [{"filename": f, "sha": sha, "patch": "+" * patch_bytes} for f, sha in entries]})
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
    assert r.returncode == 0 and "no unread protocol source among them" in r.stdout, r.stdout + r.stderr


def test_a_compare_past_the_kernel_s_limit_on_one_environment_string_is_read(tmp_path: Path) -> None:
    """2026-11-23: the real compare past the pin was 139,155 bytes; passed as an environment variable it was over Linux's
    131,072-byte limit on one string, and node never started ("Argument list too long", exit 126). The responses go to
    files now."""
    r = _check(tmp_path, files=("Cargo.lock", "README.md"), patch_bytes=100_000)
    assert r.returncode == 0 and "no unread protocol source among them" in r.stdout, r.stdout + r.stderr
    r = _check(tmp_path, files=("src/mpc/faand.rs", "README.md"), patch_bytes=100_000)
    assert r.returncode == 1 and "src/mpc/faand.rs" in r.stderr, r.stdout + r.stderr


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


READ_PORTABLE = ("src/transpose/portable.rs", "1c706c121d13cf93153df990e0bb6a08c63ee62f")


def test_a_source_file_read_at_its_blob_passes_and_a_later_change_to_it_fails(tmp_path: Path) -> None:
    """2026-11-24: upstream's d0921018 rewrote one line of src/transpose/portable.rs (`*v = *v << 1` as `*v <<= 1`), read
    and recorded by that file's blob SHA. The same blob passes; any later change to the file is a new blob and fails."""
    assert _check(tmp_path, files=(READ_PORTABLE, "Cargo.lock")).returncode == 0
    r = _check(tmp_path, files=(("src/transpose/portable.rs", "f" * 40),))
    assert r.returncode == 1 and "src/transpose/portable.rs" in r.stderr, r.stdout + r.stderr
    r = _check(tmp_path, files=(READ_PORTABLE, "src/mpc/faand.rs"))
    assert r.returncode == 1 and "src/mpc/faand.rs" in r.stderr and "portable.rs" not in r.stderr, r.stdout + r.stderr
