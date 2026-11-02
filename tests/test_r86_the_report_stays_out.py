"""The technical report left this repository in Round 86, and nothing brings it back.

WHY THIS EXISTS (Round 86, by the owner's decision)

The owner chose to keep the technical report private from Round 86, with a new summary of the
verification gates beside it, in a separate private repository. The report's history moved there
with it. In this repository the history is kept rather than rewritten, also by the owner's choice,
so the report's four PDFs are still in old commits. Two things follow:

- Nothing may track the report or the summary again, under any path. `make check-secrets` fails on
  that now, beside its other name checks.
- `make check-secrets` also fails on any PDF the history holds and the tree has dropped, a rule
  written after four third-party PDFs without a redistribution grant were removed from the whole
  history on 2026-10-31. The report's own PDFs are an exception, named path by path, so that the
  exception cannot cover a new PDF under `report/` or anywhere else.

WHAT IS ASSERTED

- No tracked path is under `report/`, and no tracked file is a technical report or a summary.
- In a throwaway repository: a tracked report file fails the check; the four named PDFs dropped
  from the tree pass it; any other dropped PDF, under `report/` or not, still fails it.
- No tracked document links into `report/`, and the Makefile has no `report` target: the report is
  built in its own repository.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_secrets.sh"
KEPT = ("report/technical-report.pdf", "report/generated/fig_baselines.pdf",
        "report/generated/fig_decision_curve.pdf", "report/generated/fig_permutation_null.pdf")


def _tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()


def test_nothing_tracks_the_report_or_the_summary() -> None:
    files = _tracked()
    assert len(files) > 300, f"only {len(files)} tracked files; the listing is looking at nothing"
    back = [f for f in files if f.startswith("report/")
            or re.search(r"(^|/)(technical-report|verification-summary)\.(pdf|tex)$", f)]
    assert not back, f"the report or the summary is tracked again: {back}"


def _history_repo(tmp_path: Path, added: dict[str, str], removed: list[str]) -> subprocess.CompletedProcess[str]:
    """A throwaway repository whose first commit adds `added` and whose second removes `removed`."""
    if shutil.which("git") is None:
        pytest.skip("git absent")
    run = lambda *a: subprocess.run(a, cwd=tmp_path, capture_output=True, text=True, check=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.invalid")
    run("git", "config", "user.name", "test")
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "check_secrets.sh")
    for name, text in {"README.md": "a public readme\n", **added}.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    run("git", "add", "-A")
    run("git", "commit", "-q", "-m", "add")
    if removed:
        run("git", "rm", "-q", *removed)
        run("git", "commit", "-q", "-m", "remove")
    return subprocess.run(["bash", "scripts/check_secrets.sh"], cwd=tmp_path, capture_output=True, text=True)


def test_a_tracked_report_file_fails_the_check(tmp_path: Path) -> None:
    r = _history_repo(tmp_path, {"report/technical-report.tex": "\\documentclass{article}\n"}, [])
    assert r.returncode == 1 and "both left this repository in Round 86" in r.stdout, r.stdout


def test_only_the_named_report_pdfs_may_stay_in_the_history(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    clean = _history_repo(tmp_path / "a", dict.fromkeys(KEPT, "%PDF-1.5\n"), list(KEPT))
    assert clean.returncode == 0 and "no PDF in the history that the tree has dropped" in clean.stdout, clean.stdout
    others = ["report/generated/fig_new.pdf", "docs/paper.pdf"]
    seeded = _history_repo(tmp_path / "b", dict.fromkeys([*KEPT, *others], "%PDF-1.5\n"), [*KEPT, *others])
    line = next((ln for ln in seeded.stdout.splitlines() if "removed from the tree" in ln), "")
    assert seeded.returncode == 1 and all(o in line for o in others), seeded.stdout
    assert not any(k in line for k in KEPT), f"a named report PDF was reported, so the exception is not applied: {line}"


def test_no_document_links_into_report_and_make_has_no_report_target() -> None:
    links = []
    for f in _tracked():
        if f.endswith(".md") and f not in ("docs/deliverable-verification.md", "docs/verification-log.md"):
            text = (REPO / f).read_text(errors="replace")
            links += [f"{f}: {m.group(0)}" for m in re.finditer(r"\]\((?:\.\./)*report/[^)]*\)", text)]
    assert not links, f"links into report/, which is not in this repository: {links}"
    make = (REPO / "Makefile").read_text()
    assert not re.search(r"^report:", make, flags=re.M), "the Makefile builds a report that is not here"
