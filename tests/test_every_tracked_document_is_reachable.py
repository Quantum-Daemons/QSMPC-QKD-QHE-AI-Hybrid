"""A tracked document nobody links to is a document nobody reads.

WHY THIS EXISTS

Two files were unreachable from anywhere a cloner could start, and one of them mattered:

- **`docs/deployment.md`** is the ONLY document carrying the demo's port-collision workaround - the
  `-p qsh-demo` project name and the behind-proxy compose override. `README.md` has a `## Deployment`
  section that inlines Docker commands and never linked it, so somebody hitting the collision had no
  way to find the answer that already existed in the repository.
- **`report/technical-report.pdf`**, the 14-section generated deliverable, had no link from
  `README.md` and no `make` target. It was discoverable only by browsing the directory tree.

This is the same class as `test_model_artifact_inventory.py`'s orphan check, which asks the question
of `references/*.md`, and it was found by asking the same question one directory over. The existing
check was correct and its SCOPE was one folder.

WHAT COUNTS AS A LINK

Any mention of the path in another tracked `.md`, `.tex` or in the `Makefile` - a markdown link, a
`\\href`, a backticked path or a make recipe. That is deliberately loose: the question is whether a
reader following the repository's own documents can arrive at the file, not whether the reference is
formatted a particular way.

SCOPE, AND WHY IT IS NOT EVERY TRACKED FILE

`docs/` and `report/` only. Source files are reached by reading source; model cards are covered by
`test_model_artifact_inventory.py`; and the gitignored checklist is excluded automatically because it
is untracked, which is the correct reason rather than a special case.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Where a reader can plausibly start. `Makefile` counts: `make report` is a way to arrive at the PDF.
LINKERS = ("README.md", "Makefile", "DATA_LICENSES.md")


def _tracked(*globs: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", *globs], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [f for f in out.stdout.split() if f]


def _corpus() -> str:
    """Every tracked document a reference could live in, concatenated."""
    files = set(_tracked("*.md", "docs", "references", "report")) | set(LINKERS)
    parts = []
    for f in sorted(files):
        p = REPO / f
        if p.is_file() and p.suffix in {".md", ".tex"} or f == "Makefile":
            parts.append(f"<<<{f}>>>\n" + p.read_text(errors="ignore"))
    return "\n".join(parts)


def _inbound(rel: str, corpus: str) -> bool:
    """Is `rel` mentioned anywhere other than inside itself?"""
    blocks = corpus.split("<<<")
    name = Path(rel).name
    for b in blocks:
        if not b or b.startswith(rel + ">>>"):
            continue                      # a file referring to itself is not an inbound link
        body = b.split(">>>", 1)[-1]
        if rel in body or name in body:
            return True
    return False


def test_there_are_documents_to_check() -> None:
    """A parse that finds nothing passes the real check for the wrong reason."""
    docs = _tracked("docs", "report")
    assert len(docs) >= 8, f"only {len(docs)} tracked files under docs/ and report/ - has the tree moved?"


def test_every_tracked_document_has_an_inbound_reference() -> None:
    """docs/ and report/: one inbound mention, from somewhere a reader can start."""
    corpus = _corpus()
    watched = [
        f for f in _tracked("docs", "report")
        if Path(f).suffix in {".md", ".pdf", ".tex"} and "/generated/" not in f
    ]
    orphans = sorted(f for f in watched if not _inbound(f, corpus))
    assert not orphans, (
        "these tracked documents are referenced by no other tracked document:\n  "
        + "\n  ".join(orphans)
        + "\n\nA document nobody links to is a document nobody reads. `docs/deployment.md` was the "
        "live case: the only file carrying the demo's port-collision workaround, with no inbound link "
        "from anywhere, while README's Deployment section inlined commands instead."
    )


def test_the_gate_fires_on_an_unreferenced_path() -> None:
    """Seed the axis being ADDED - reachability, not content.

    The seed is a path nobody mentions, checked against the same corpus. The negative control is a
    path that IS linked: without it, a check that reported everything would look identical to one
    that works.
    """
    corpus = _corpus()
    assert not _inbound("docs/a-file-nobody-mentions.md", corpus), (
        "an unreferenced path was reported as reachable; the check would pass on any orphan"
    )
    assert _inbound("docs/deployment.md", corpus), (
        "docs/deployment.md is unreferenced again - it is the only document carrying the demo's "
        "port-collision workaround, and Round 43 linked it from README.md for that reason"
    )
