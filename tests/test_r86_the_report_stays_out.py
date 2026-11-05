"""The technical report left this repository in Round 86, and nothing brings it back.

WHY THIS EXISTS (Round 86, by the owner's decision)

The owner chose to keep the technical report private from Round 86, with a new summary of the
verification gates beside it, in a separate private repository. On 2026-11-02 the owner chose to take
it out of this repository's history too, so no commit here touches `report/`. Two things follow:

- Nothing may track the report or the summary again, under any path. `make check-secrets` fails on
  that now, beside its other name checks.
- `make check-secrets` also fails on any PDF the history holds and the tree has dropped, a rule
  written after four third-party PDFs without a redistribution grant were removed from the whole
  history on 2026-10-31. Until the rewrite of 2026-11-02 the report's own four PDFs were an exception to
  it. That exception is gone: a branch from the old history carrying them now fails the check, which
  is the hazard the rewrite left behind.

WHAT IS ASSERTED

- No tracked path is under `report/`, and no tracked file is a technical report or a summary.
- In a throwaway repository: a tracked report file fails the check, and any PDF dropped from the tree,
  the report's own four included, fails it.
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
OLD_REPORT_PDFS = ("report/technical-report.pdf", "report/generated/fig_baselines.pdf",
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


def test_no_pdf_dropped_from_the_tree_may_stay_in_the_history(tmp_path: Path) -> None:
    others = [*OLD_REPORT_PDFS, "report/generated/fig_new.pdf", "docs/paper.pdf"]
    seeded = _history_repo(tmp_path, dict.fromkeys(others, "%PDF-1.5\n"), others)
    line = next((ln for ln in seeded.stdout.splitlines() if "removed from the tree" in ln), "")
    assert seeded.returncode == 1 and any(o in line for o in others), seeded.stdout


def test_the_history_here_holds_none_of_the_reports_pdfs() -> None:
    if subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=REPO, capture_output=True,
                      text=True).stdout.strip() == "true":
        pytest.skip("a shallow clone cannot see the history")
    added = subprocess.run(["git", "log", "--all", "--format=", "--name-only", "--diff-filter=A", "--", "report/"],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    assert not added, f"the history touches report/: {added[:3]}"


def test_no_document_links_into_report_and_make_has_no_report_target() -> None:
    links = []
    for f in _tracked():
        if f.endswith(".md") and f not in ("docs/deliverable-verification.md", "docs/verification-log.md"):
            text = (REPO / f).read_text(errors="replace")
            links += [f"{f}: {m.group(0)}" for m in re.finditer(r"\]\((?:\.\./)*report/[^)]*\)", text)]
    assert not links, f"links into report/, which is not in this repository: {links}"
    make = (REPO / "Makefile").read_text()
    assert not re.search(r"^report:", make, flags=re.M), "the Makefile builds a report that is not here"
