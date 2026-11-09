#!/usr/bin/env python
"""Build the Python walkthrough as static files that load nothing from another origin.

A marimo WebAssembly export loads Pyodide and its packages from third parties by default. The demo says its code comes from
this site, so everything the walkthrough runs is fetched here at build time, against a sha256 recorded in
`walkthroughs/pyodide-sources.json`, and served from the walkthrough's own folder. marimo reads where to find it from two
attributes on its root element.

    python scripts/build_walkthrough.py OUT      build into OUT (needs marimo, see walkthroughs/requirements.lock)
    python scripts/build_walkthrough.py --check  the recorded files are consistent; no network, no marimo
    python scripts/build_walkthrough.py --refresh  rebuild the record from marimo's lock file, for a version bump

See references/notebook-walkthroughs.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from email.parser import Parser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "walkthroughs"
NOTEBOOK = HERE / "nidd_copies.py"
LOCK = HERE / "pyodide-lock.json"
SOURCES = HERE / "pyodide-sources.json"
REQUIREMENTS = HERE / "requirements.lock"

PYODIDE = "v314.0.0"
BASE = f"https://cdn.jsdelivr.net/pyodide/{PYODIDE}/full/"
CORE = ("pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip")
#: What marimo's worker asks for by name, and what its kernel imports as it starts.
ROOTS = ("micropip", "msgspec", "marimo-base", "markdown", "pymdown-extensions", "narwhals", "packaging", "pyodide-http", "jedi", "pygments", "docutils")


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "qsh-walkthrough-build"})  # marimo's host refuses urllib's default
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310  (fixed https hosts)
        return response.read()


def _marimo_version() -> str:
    match = re.search(r"^marimo==([0-9.]+)", REQUIREMENTS.read_text(), re.M)
    assert match, f"{REQUIREMENTS} does not pin marimo"
    return match.group(1)


def refresh() -> int:
    """Rewrite the two records from marimo's lock file. A development step; the build never does this."""
    version = _marimo_version()
    full = json.loads(_fetch(f"https://wasm.marimo.app/pyodide-lock.json?v={version}&pyodide={PYODIDE}"))
    packages = full["packages"]
    keep: dict[str, dict] = {}
    stack = list(ROOTS)
    while stack:
        name = stack.pop().lower()
        if name not in keep:
            keep[name] = dict(packages[name])
            stack += keep[name].get("depends", [])
    sources: dict[str, dict[str, str]] = {}
    for name in CORE:
        sources[name] = {"url": BASE + name, "sha256": hashlib.sha256(_fetch(BASE + name)).hexdigest()}
    for package in keep.values():
        origin = package["file_name"]
        local = origin.rsplit("/", 1)[-1]
        sources[local] = {"url": origin if origin.startswith("http") else BASE + origin, "sha256": package["sha256"]}
        package["file_name"] = local
    LOCK.write_text(json.dumps({"info": full["info"], "packages": dict(sorted(keep.items()))}, indent=1, sort_keys=True) + "\n")
    SOURCES.write_text(json.dumps({"marimo": version, "pyodide": PYODIDE, "files": dict(sorted(sources.items()))}, indent=1) + "\n")
    print(f"recorded {len(sources)} files for marimo {version} and Pyodide {PYODIDE}")
    return 0


def check() -> list[str]:
    """Everything that can be wrong without the network. An empty list means the record is consistent."""
    problems: list[str] = []
    lock = json.loads(LOCK.read_text())
    sources = json.loads(SOURCES.read_text())
    if sources["marimo"] != _marimo_version():
        problems.append(f"the record is for marimo {sources['marimo']} and requirements.lock pins {_marimo_version()}")
    if sources["pyodide"] != PYODIDE:
        problems.append(f"the record is for Pyodide {sources['pyodide']} and this script asks for {PYODIDE}")
    written = re.search(r'__generated_with = "([0-9.]+)"', NOTEBOOK.read_text())
    if not written or written.group(1) != _marimo_version():
        problems.append("the notebook's __generated_with does not match the pinned marimo")
    files = sources["files"]
    for name in CORE:
        if name not in files:
            problems.append(f"{name} is not recorded")
    for name, entry in files.items():
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            problems.append(f"{name} has no sha256")
        if not entry.get("url", "").startswith("https://"):
            problems.append(f"{name} is fetched from {entry.get('url')!r}")
    for key, package in lock["packages"].items():
        name = package["file_name"]
        if "/" in name or name.startswith("http"):
            problems.append(f"{key} would be fetched from {name}: the lock file must name local files only")
        elif name not in files:
            problems.append(f"{key}: {name} is not among the recorded files")
        elif files[name]["sha256"] != package["sha256"]:
            problems.append(f"{key}: the lock file and the record disagree on the sha256")
        for needed in package.get("depends", []):
            if needed.lower() not in lock["packages"]:
                problems.append(f"{key} depends on {needed}, which is not in the lock file")
    for root in ROOTS:
        if root not in lock["packages"]:
            problems.append(f"{root} is not in the lock file")
    return problems


def _third_party(pyodide_dir: Path) -> str:
    lines = [
        "The Python walkthrough serves these third-party works from this folder.",
        "",
        "Pyodide (pyodide.asm.*, python_stdlib.zip), https://pyodide.org, MPL-2.0; it includes CPython, under the PSF License.",
        "The marimo front end (assets/), https://marimo.io, Apache-2.0.",
        "",
        "Python packages, in pyodide/ as wheels, each with its licence text inside:",
    ]
    for wheel in sorted(pyodide_dir.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            fields = Parser().parsestr(archive.read(metadata).decode("utf8", "replace"), headersonly=True)
        licence = fields.get("License-Expression") or fields.get("License") or "; ".join(c for c in fields.get_all("Classifier", []) if c.startswith("License")) or "see the wheel"
        lines.append(f"  {fields['Name']} {fields['Version']}: {' '.join(licence.split())[:120]}")
    return "\n".join(lines) + "\n"


def build(out: Path) -> int:
    problems = check()
    assert not problems, problems
    if out.exists():
        shutil.rmtree(out)
    subprocess.run([sys.executable, "-m", "marimo", "export", "html-wasm", str(NOTEBOOK), "-o", str(out), "--mode", "run", "--show-code"], check=True)
    (out / "CLAUDE.md").unlink(missing_ok=True)  # the export copies marimo's own development note
    pyodide = out / "pyodide"
    pyodide.mkdir()
    for name, entry in json.loads(SOURCES.read_text())["files"].items():
        content = _fetch(entry["url"])
        digest = hashlib.sha256(content).hexdigest()
        assert digest == entry["sha256"], f"{name}: sha256 {digest} is not the recorded {entry['sha256']}"
        (pyodide / name).write_bytes(content)
    shutil.copy(LOCK, pyodide / "pyodide-lock.json")
    index = out / "index.html"
    html = index.read_text()
    assert html.count("<marimo-wasm") == 1, "the export's root element has changed"
    index.write_text(html.replace("<marimo-wasm", '<marimo-wasm data-pyodide-index-url="pyodide/" data-pyodide-lockfile-url="pyodide/pyodide-lock.json"', 1))
    (out / "THIRD_PARTY.txt").write_text(_third_party(pyodide))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out}: {total / 1e6:.1f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh:
        return refresh()
    if args.check:
        problems = check()
        for problem in problems:
            print(f"  FAIL {problem}")
        print("walkthrough record consistent" if not problems else "walkthrough record NOT consistent")
        return 1 if problems else 0
    if not args.out:
        ap.error("give an output folder, --check or --refresh")
    return build(args.out)


if __name__ == "__main__":
    sys.exit(main())
