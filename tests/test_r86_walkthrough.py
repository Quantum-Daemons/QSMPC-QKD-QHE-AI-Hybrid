"""The Python walkthrough loads nothing from another origin, and its excerpt says what it claims (Round 86).

A marimo export reads Pyodide from a third-party CDN and its lock file from marimo's own host by default. The demo says its code
comes from this site, so the build fetches every file against a recorded sha256 and points marimo at the copies. The checks here
hold that record together; the build itself runs in the image.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_walkthrough as B  # noqa: E402


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("LOCK", "SOURCES"):
        original = getattr(B, name)
        copy = tmp_path / original.name
        shutil.copy(original, copy)
        monkeypatch.setattr(B, name, copy)
    return tmp_path


def _edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_the_record_is_consistent() -> None:
    assert B.check() == []


def test_a_package_fetched_from_another_host_is_refused(record: Path) -> None:
    _edit(B.LOCK, lambda d: d["packages"]["micropip"].__setitem__("file_name", "https://files.pythonhosted.org/x/micropip.whl"))
    assert any("local files only" in p for p in B.check())


def test_a_changed_checksum_is_refused(record: Path) -> None:
    _edit(B.LOCK, lambda d: d["packages"]["micropip"].__setitem__("sha256", "0" * 64))
    assert any("disagree on the sha256" in p for p in B.check())


def test_a_file_without_a_checksum_is_refused(record: Path) -> None:
    _edit(B.SOURCES, lambda d: d["files"]["pyodide.asm.wasm"].pop("sha256"))
    assert any("has no sha256" in p for p in B.check())


def test_a_missing_dependency_is_refused(record: Path) -> None:
    _edit(B.LOCK, lambda d: d["packages"].pop("parso"))
    assert any("depends on parso" in p for p in B.check())


def test_a_record_for_another_marimo_is_refused(record: Path) -> None:
    _edit(B.SOURCES, lambda d: d.__setitem__("marimo", "0.0.1"))
    assert any("requirements.lock pins" in p for p in B.check())


def test_the_image_builds_it_and_serves_it() -> None:
    docker = (REPO / "docker" / "Dockerfile.web").read_text()
    assert "scripts/build_walkthrough.py /out" in docker
    assert "--require-hashes -r walkthroughs/requirements.lock" in docker
    assert "COPY --from=walkthrough /out /srv/walkthrough" in docker


def test_the_build_points_marimo_at_its_own_copies() -> None:
    source = (REPO / "scripts" / "build_walkthrough.py").read_text()
    assert 'data-pyodide-index-url="pyodide/"' in source and 'data-pyodide-lockfile-url="pyodide/pyodide-lock.json"' in source


def test_the_excerpt_holds_what_the_page_says() -> None:
    excerpt = json.loads((REPO / "walkthroughs" / "public" / "nidd_copies_excerpt.json").read_text())
    assert excerpt["licence"] == "CC-BY-4.0" and excerpt["attribution"] and excerpt["modifications"]
    columns = excerpt["columns"]
    assert len(columns) == 113 and len(excerpt["labelled_benign"]) == 200 == len(excerpt["labelled_udpflood"])
    differ = {c for k, c in enumerate(columns) if any(a[k] != b[k] for a, b in zip(excerpt["labelled_benign"], excerpt["labelled_udpflood"], strict=True))}
    assert differ == {"Unnamed: 0", "Rank", "Seq", "Offset", "IdleTime", "Label", "Attack Type", "Attack Tool"}
    where = columns.index("Attack Type")
    assert {r[where] for r in excerpt["labelled_benign"]} == {"Benign"} and {r[where] for r in excerpt["labelled_udpflood"]} == {"UDPFlood"}


def test_the_card_links_to_it() -> None:
    assert 'href="/walkthrough/"' in (REPO / "web" / "src" / "components" / "ModelPanel.tsx").read_text()
