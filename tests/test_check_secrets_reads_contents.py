"""`make check-secrets` reads file CONTENTS, not only file names.

WHY THIS EXISTS (2026-10-29)

The check used to look at tracked file NAMES only, and passed for months while three things leaked
through contents: the demo's host name in a compose comment and the verification log, a local home
path (whose user name is the deploy login) inside two tracked LaTeX build files, and the private
notes named in three prose documents. They were removed from the whole git history the same day,
and `scripts/check_secrets.sh` gained content checks. This seeds each leak in a throwaway repository
and requires the check to fail on it, and to pass on the clean repository and on the placeholder.

The seeded strings are built at run time, so this file itself carries no home path for the real
check to find.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_secrets.sh"
HOME = "/" + "home" + "/"


def _repo(tmp_path: Path, files: dict[str, str], env_demo: str | None = None) -> subprocess.CompletedProcess[str]:
    if shutil.which("git") is None:
        pytest.skip("git absent")
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "check_secrets.sh")
    for name, text in {"README.md": "a public readme\n", **files}.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    if env_demo is not None:
        (tmp_path / ".env.demo").write_text(env_demo)
    run = lambda *a: subprocess.run(a, cwd=tmp_path, capture_output=True, text=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "add", "README.md", "scripts", *files)
    return subprocess.run(["bash", "scripts/check_secrets.sh"], cwd=tmp_path, capture_output=True, text=True)


def test_a_clean_repository_passes(tmp_path: Path) -> None:
    r = _repo(tmp_path, {"docs/a.md": "see " + HOME + "user/project for the layout\n"})
    assert r.returncode == 0, r.stdout


def test_an_absolute_home_path_fails(tmp_path: Path) -> None:
    r = _repo(tmp_path, {"report/out/x.fls": "PWD " + HOME + "alice/build\n"})
    assert r.returncode == 1 and "absolute home path" in r.stdout and "report/out/x.fls" in r.stdout, r.stdout


def test_naming_the_private_notes_fails(tmp_path: Path) -> None:
    r = _repo(tmp_path, {"docs/b.md": "Background for `MONETIZATION.md` only.\n"})
    assert r.returncode == 1 and "private notes are named in: docs/b.md" in r.stdout, r.stdout


def test_a_deploy_value_from_the_env_file_fails_and_is_not_printed(tmp_path: Path) -> None:
    box = "demo-box.example.test"
    r = _repo(tmp_path, {"docs/c.md": f"Observed on https://{box}/usecase\n"},
              env_demo=f"QSH_DEMO_URL=https://{box}\nQSH_DEMO_SSH=deployer@{box}\n")
    assert r.returncode == 1 and "value from .env.demo appears in tracked files: docs/c.md" in r.stdout, r.stdout
    assert box not in r.stdout and "deployer" not in r.stdout, r.stdout
