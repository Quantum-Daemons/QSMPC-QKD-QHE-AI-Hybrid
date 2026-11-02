"""The release version is written once, in `qsh.__version__`, and every other copy reads it or matches it.

WHY THIS EXISTS (Round 86)

The project had never cut a release, and its version was 0.1.0 in five places that nothing held
together: `pyproject.toml`, `qsh.__version__`, two literals in `api/main.py` (the OpenAPI document
and `/api/health`) and `web/package.json` with its lockfile root. The owner chose to release 0.85.0,
the minor naming the verification round the release closes, as a tag with release notes and the
source archives GitHub generates. A bump that reaches some of five copies is this project's
most-repeated defect, so there is now one copy and the rest read it or are compared with it.

WHAT IS ASSERTED

- `qsh.__version__` is a plain X.Y.Z, and `pyproject.toml` declares the version dynamic, read from it,
  with no literal of its own.
- The installed distribution reports the same version. A dynamic version that does not resolve, or
  an environment installed before the bump, fails here; `pip install -e .` refreshes it.
- `api/main.py` imports `__version__` and uses it at both sites, with no version literal left.
- `web/package.json` and both version fields of `web/package-lock.json` equal it (npm reads those two
  and cannot import Python).

The Rust crates under `wasm/` and `prototypes/` keep their own versions on purpose: they are internal,
never published, and versioned as crates rather than as this release.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import tomllib
from pathlib import Path

import qsh

REPO = Path(__file__).resolve().parents[1]


def test_the_version_is_written_once_and_pyproject_reads_it() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", qsh.__version__), qsh.__version__
    project = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert "version" not in project["project"], "pyproject.toml carries a version literal of its own again"
    assert "version" in project["project"].get("dynamic", []), "pyproject.toml no longer declares the version dynamic"
    attr = project["tool"]["setuptools"]["dynamic"]["version"]
    assert attr == {"attr": "qsh.__version__"}, attr
    init = (REPO / "src" / "qsh" / "__init__.py").read_text()
    assert len(re.findall(r"^__version__ = ", init, flags=re.M)) == 1


def test_the_installed_distribution_reports_it() -> None:
    installed = importlib.metadata.version("qsh")
    assert installed == qsh.__version__, (
        f"the installed qsh reports {installed}, the source says {qsh.__version__}: run `pip install -e .`, "
        "or the dynamic version in pyproject.toml no longer resolves")


def test_the_api_reports_it_at_both_sites() -> None:
    src = (REPO / "api" / "main.py").read_text()
    assert "from qsh import __version__" in src
    assert 'FastAPI(title="QSMPC-QKD-QHE-AI-Hybrid API", version=__version__)' in src
    assert '"version": __version__' in src
    assert not re.search(r'version\s*[=:]\s*"\d+\.\d+\.\d+"', src), "a version literal is back in api/main.py"
    assert not re.search(r'"version":\s*"\d', src), "/api/health carries a version literal again"


def test_the_web_package_and_its_lockfile_match_it() -> None:
    package = json.loads((REPO / "web" / "package.json").read_text())
    lock = json.loads((REPO / "web" / "package-lock.json").read_text())
    assert package["version"] == qsh.__version__, package["version"]
    assert lock["version"] == qsh.__version__, lock["version"]
    assert lock["packages"][""]["version"] == qsh.__version__, lock["packages"][""]["version"]
