#!/usr/bin/env python
"""Give the four SBOMs the metadata their generators leave out, without making them irreproducible (Round 86, 2026-11-23).

The owner's choices of 2026-11-23, after sbomqs (report only, in CI) graded the SBOMs F, F, D and F. Some of what it
missed is deliberate: the generators run reproducibly, so they write no creation time and no random serial number.
Some of it was wrong: the WASM SBOM said it was made on 1970-01-01, because `SOURCE_DATE_EPOCH=0` stood in for a time.
This sets, in place, only what is true and the same on every run of the same commit:

- the creation time: the commit's own time (`git log -1 --format=%ct`), replacing a missing or 1970-01-01 one;
- a serial number derived from the document's content (a UUID v5, as `scripts/generate_cbom.py` derives the CBOM's),
  where the generator wrote none;
- the supplier, the project's own organisation; the same organisation as the BOM's manufacturer, CycloneDX's field
  for the creator of a BOM made by an automated process (a hand-written one names its authors); and the lifecycle
  phase "build";
- the Python SBOM's primary component, the `qsh` package `requirements.lock` is locked for, which cyclonedx-py's
  requirements mode does not name; and the threshold-aggregation module's version, which is the repository's;
- the web SBOM's component hashes: each package's SHA-512 from `web/package-lock.json`'s `integrity`, the hash of the
  tarball npm installs. cyclonedx-npm writes the same value on the package's distribution reference, where sbomqs
  does not look; the step stops if the two differ, or if the lockfile's version is not the SBOM's;
- the Python SBOM's licences, when `--python-environment` names cyclonedx-py's environment-mode SBOM of an interpreter
  where `requirements.lock` is installed (CI's python job writes one): each licence as that generator read it from the
  installed package's metadata, copied onto the same package at the same version. A package not found at that version
  keeps none.

It reads no hash and no licence itself: the hashes are the lockfile's and the licences the environment generator's.
The Python SBOM gets no component hash: `requirements.lock` lists every file of a version, and which one is installed
depends on the machine, so they stay on the distribution references where cyclonedx-py puts them.

    python scripts/sbom_metadata.py [--python-environment FILE] [--summary FILE] \
        sbom-out/python.cdx.json sbom-out/web.cdx.json sbom-out/wasm.cdx.json sbom-out/fedthresh.cdx.json
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import re
import subprocess
import sys
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SUPPLIER = {"name": "Quantum Daemons", "url": ["https://quantum.daemons.jp/"]}
EPOCH = ("1970-01-01T00:00:00Z", "1970-01-01T00:00:00.000Z", "1970-01-01T00:00:00.000000000Z")
#: Subresource Integrity's algorithm names (the form of npm's `integrity`) and CycloneDX's, strongest first.
SRI = {"sha512": "SHA-512", "sha384": "SHA-384", "sha256": "SHA-256", "sha1": "SHA-1"}


def commit_time() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%ct"], capture_output=True, text=True, check=True).stdout.strip()
    return datetime.fromtimestamp(int(out), tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def complete(doc: dict, name: str, when: str, version: str) -> dict:
    md = doc.setdefault("metadata", {})
    if md.get("timestamp") in (None, *EPOCH):
        md["timestamp"] = when
    md.setdefault("supplier", SUPPLIER)
    md.setdefault("manufacturer", SUPPLIER)
    md.setdefault("lifecycles", [{"phase": "build"}])
    comp = md.get("component")
    if name == "python" and not comp:
        md["component"] = {"type": "application", "bom-ref": "qsh", "name": "qsh", "version": version,
                           "description": "The Python package requirements.lock is locked for"}
    if name == "fedthresh" and comp and not comp.get("version"):
        comp["version"] = version
    if not doc.get("serialNumber"):
        body = {k: v for k, v in doc.items() if k != "serialNumber"}
        doc["serialNumber"] = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, f'qsh-sbom:{name}:' + json.dumps(body, sort_keys=True))}"
    return doc


def walk(components: list[dict]) -> Iterator[dict]:
    """Every component, nested ones included (cyclonedx-npm nests a package installed under another's node_modules)."""
    for c in components:
        yield c
        yield from walk(c.get("components", []))


def sri(integrity: str) -> tuple[str, str]:
    """An npm `integrity` ("sha512-<base64>", possibly several, space-separated) as CycloneDX's algorithm name and the
    hex digest, the strongest it lists."""
    found = {}
    for part in integrity.split():
        alg, _, digest = part.partition("-")
        if alg in SRI:
            found[alg] = base64.b64decode(digest, validate=True).hex()
    for alg, cdx in SRI.items():
        if alg in found:
            return cdx, found[alg]
    raise ValueError(f"no algorithm this step knows in the integrity {integrity[:16]}...")


def lockfile_hashes(doc: dict, packages: dict) -> None:
    """The web SBOM's component hashes from `packages`, `web/package-lock.json`'s map from a package's path to its entry;
    a component's entry is the one at its `cdx:npm:package:path` property."""
    for c in walk(doc.get("components", [])):
        path = next((p.get("value") for p in c.get("properties", []) if p.get("name") == "cdx:npm:package:path"), None)
        entry = packages.get(path) if path else None
        if c.get("hashes") or not entry or not entry.get("integrity"):
            continue
        ref = c.get("bom-ref", c.get("name"))
        if entry.get("version") != c.get("version"):
            raise ValueError(f"{ref}: web/package-lock.json has {path} at {entry.get('version')}, the SBOM at {c.get('version')}")
        alg, digest = sri(entry["integrity"])
        written = [h.get("content") for r in c.get("externalReferences", []) for h in r.get("hashes", []) if h.get("alg") == alg]
        if any(w != digest for w in written):
            raise ValueError(f"{ref}: the lockfile's {alg} is not the one cyclonedx-npm wrote on its distribution reference")
        c["hashes"] = [{"alg": alg, "content": digest}]


def _key(name: str, version: str | None) -> tuple[str, str | None]:
    """A Python package's name as PEP 503 normalises it, with its version."""
    return re.sub(r"[-_.]+", "-", name).lower(), version


def environment_licences(doc: dict, environment: dict) -> None:
    """The Python SBOM's licences from `environment`, cyclonedx-py's environment-mode SBOM: each copied, as that generator
    wrote it, onto the component of the same normalised name at the same version."""
    read = {_key(c.get("name", ""), c.get("version")): c["licenses"] for c in walk(environment.get("components", [])) if c.get("licenses")}
    for c in walk(doc.get("components", [])):
        found = read.get(_key(c.get("name", ""), c.get("version")))
        if found and not c.get("licenses"):
            c["licenses"] = copy.deepcopy(found)


def _with(doc: dict, field: str) -> tuple[int, int]:
    comps = list(walk(doc.get("components", [])))
    return sum(1 for c in comps if c.get(field)), len(comps)


def main(argv: list[str]) -> int:
    import qsh

    ap = argparse.ArgumentParser(description="Fill in the SBOMs' metadata, the web SBOM's hashes and the Python SBOM's licences, in place.")
    ap.add_argument("sboms", nargs="+", type=Path)
    ap.add_argument("--python-environment", type=Path, help="cyclonedx-py's environment-mode SBOM of an interpreter where requirements.lock is installed")
    ap.add_argument("--summary", type=Path, help="append one Markdown line per SBOM to this file (CI's job summary)")
    args = ap.parse_args(argv)
    when = commit_time()
    packages = json.loads((REPO / "web" / "package-lock.json").read_text(encoding="utf-8"))["packages"]
    environment = None
    if args.python_environment is not None and args.python_environment.is_file():
        environment = json.loads(args.python_environment.read_text(encoding="utf-8"))
    lines = []
    for path in args.sboms:
        name = path.name.split(".")[0]
        doc = json.loads(path.read_text(encoding="utf-8"))
        try:
            if name == "web":
                lockfile_hashes(doc, packages)
            if name == "python" and environment is not None:
                environment_licences(doc, environment)
        except ValueError as e:
            print(f"{path.name}: {e}", file=sys.stderr)
            return 1
        complete(doc, name, when, qsh.__version__)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        hashed, total = _with(doc, "hashes")
        licensed, _ = _with(doc, "licenses")
        line = f"{path.name}: created {doc['metadata']['timestamp']}; component hashes on {hashed} of {total}, licences on {licensed} of {total}"
        if name == "python" and environment is None:
            line += f" (no environment SBOM {'at ' + str(args.python_environment) if args.python_environment else 'was given'}, so no licences)"
        print(line)
        lines.append(line)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write("".join(f"- {line}\n" for line in lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
