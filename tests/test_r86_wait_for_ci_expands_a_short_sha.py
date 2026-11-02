"""`scripts/wait_for_ci.sh` reads the outcome of the runs on a commit, whether it is named in full or short.

WHY THIS EXISTS (Round 86)

The script selects the runs whose `headSha` equals its argument, and GitHub reports `headSha` in full.
Given a short sha (`ac7702b`) it matched no run, so it waited out its whole forty-minute deadline for a
commit whose three runs had all passed - on PR #83, the release, while `gh run list` already showed the
three successes. The deadline then fails safe (no verdict is not a pass), so nothing merged wrongly; the
cost was the wait, and an operator learning to ignore the script.

WHAT IS ASSERTED, with a stub `gh` serving fixed runs and the real script

- A short sha is expanded and reaches GREEN at once; so does the full sha.
- A failed run is still NOT GREEN (exit 1): the fix did not loosen the verdict.
- A short sha this clone does not know fails at once, rather than waiting for runs that cannot match.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "wait_for_ci.sh"

pytestmark = pytest.mark.skipif(shutil.which("jq") is None or shutil.which("git") is None,
                                reason="jq and git are needed to run the script")

STUB = """#!/usr/bin/env bash
# A stand-in for `gh run list ... --jq EXPR`: applies EXPR to the runs in $GH_CANNED.
expr=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--jq" ]; then expr="$2"; shift; fi
  shift
done
printf '%s' "$GH_CANNED" | jq -c "$expr"
"""


@pytest.fixture()
def repo(tmp_path: Path) -> tuple[Path, str, dict]:
    run = lambda *a: subprocess.run(a, cwd=tmp_path, capture_output=True, text=True, check=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.invalid")
    run("git", "config", "user.name", "test")
    (tmp_path / "f").write_text("x\n")
    run("git", "add", "f")
    run("git", "commit", "-q", "-m", "c")
    sha = run("git", "rev-parse", "HEAD").stdout.strip()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(STUB)
    (bin_dir / "gh").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "WAIT_FOR_CI_TIMEOUT": "3"}
    return tmp_path, sha, env


def _runs(sha: str, *conclusions: str) -> str:
    return json.dumps([{"headSha": sha, "status": "completed", "conclusion": c} for c in conclusions])


def _wait(cwd: Path, env: dict, sha_arg: str, canned: str) -> tuple[subprocess.CompletedProcess[str], float]:
    t0 = time.monotonic()
    r = subprocess.run(["bash", str(SCRIPT), "main", sha_arg, "2"], cwd=cwd, capture_output=True, text=True,
                       env={**env, "GH_CANNED": canned}, timeout=120)
    return r, time.monotonic() - t0


def test_a_short_sha_reaches_its_verdict_at_once(repo) -> None:
    cwd, sha, env = repo
    r, secs = _wait(cwd, env, sha[:7], _runs(sha, "success", "success"))
    assert r.returncode == 0 and f"GREEN: 2/2 runs on {sha[:7]}" in r.stdout, r.stdout + r.stderr
    assert secs < 30, f"took {secs:.0f}s: the script waited for runs it could not match"


def test_the_full_sha_still_works(repo) -> None:
    cwd, sha, env = repo
    r, _ = _wait(cwd, env, sha, _runs(sha, "success", "success"))
    assert r.returncode == 0 and "GREEN" in r.stdout, r.stdout + r.stderr


def test_a_failed_run_is_still_not_green(repo) -> None:
    cwd, sha, env = repo
    r, _ = _wait(cwd, env, sha[:7], _runs(sha, "success", "cancelled"))
    assert r.returncode == 1 and "NOT GREEN: 1 of 2" in r.stdout, r.stdout + r.stderr


def test_an_unknown_short_sha_fails_at_once(repo) -> None:
    cwd, sha, env = repo
    unknown = "0000000" if not sha.startswith("0000000") else "fffffff"
    r, secs = _wait(cwd, env, unknown, _runs(sha, "success", "success"))
    assert r.returncode != 0 and "GREEN" not in r.stdout, r.stdout + r.stderr
    assert secs < 30, f"took {secs:.0f}s for a sha no run can match"
