"""Writes the CycloneDX 1.7 SBOM of the qsh-fedthresh wasm: the modules compiled into it, and Go's standard library.

Run by scripts/generate_sbom.sh with Go on the path, only the standard library besides:

    python3 sbom.py <output.cdx.json>

The module set is the one notices.py lists (`GOOS=js GOARCH=wasm go list -deps ./cmd/wasm`), with the licences detected
the same way from each module's own licence files. Each module's hash is the SHA-256 of its zip as `go mod download`
left it in the module cache, the file Go fetched and checked against go.sum (Round 86, 2026-11-23); the standard
library ships with Go and has none. The document is deterministic: no timestamp, and a serial number derived from its
content, as the other SBOMs are made with --output-reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

from notices import LICENCE_FILE, kinds_of

SPDX = {"Apache-2.0": "Apache-2.0", "MIT": "MIT", "ISC": "ISC"}


def licence_of(d: Path) -> dict:
    texts = [f.read_text(errors="replace") for f in sorted(d.iterdir())
             if f.is_file() and LICENCE_FILE.match(f.name) and not f.name.upper().startswith(("NOTICE", "PATENTS"))]
    ids = set()
    for t in texts:
        for k in kinds_of(t):
            ids.add(("BSD-3-Clause" if "endorse or promote" in t else "BSD-2-Clause") if k == "BSD" else SPDX[k])
    if not ids:
        sys.exit(f"sbom: no licence of a known kind in {d}")
    ids = sorted(ids)
    return {"license": {"id": ids[0]}} if len(ids) == 1 else {"expression": " AND ".join(ids)}


def json_objects(text: str) -> list[dict]:
    """The JSON objects a go command prints one after another with -json."""
    decoder, objects, at = json.JSONDecoder(), [], 0
    while at < len(text):
        if text[at].isspace():
            at += 1
            continue
        obj, at = decoder.raw_decode(text, at)
        objects.append(obj)
    return objects


def zip_hashes(modules: dict[str, tuple[str, Path]]) -> dict[str, str]:
    """The SHA-256 of each module's zip in the module cache, by module path. `go mod download` checks a zip against
    go.sum before it keeps it, so this is the hash of the file Go verified. A module it reports an error for, or one
    with no zip (a directory replacement), stops the script: the SBOM states a hash for every module or for none."""
    run = subprocess.run(["go", "mod", "download", "-json", *(f"{p}@{v}" for p, (v, _) in sorted(modules.items()))],
                         capture_output=True, text=True)
    found = {}
    # It exits 1 when a module fails, and still prints that module's object with its Error.
    for info in json_objects(run.stdout):
        if info.get("Error") or not info.get("Zip"):
            sys.exit(f"sbom: no verified zip for {info.get('Path')}@{info.get('Version')}: {info.get('Error', 'no Zip')}")
        found[info["Path"]] = (info["Version"], hashlib.sha256(Path(info["Zip"]).read_bytes()).hexdigest())
    if run.returncode != 0:
        sys.exit(f"sbom: go mod download exited {run.returncode}: {run.stderr.strip()[:300]}")
    missing = sorted(p for p, (v, _) in modules.items() if found.get(p, (None,))[0] != v)
    if missing:
        sys.exit(f"sbom: go mod download did not report {', '.join(missing)} at the listed versions")
    return {p: h for p, (_, h) in found.items()}


def main(out: str) -> None:
    env = dict(os.environ, GOOS="js", GOARCH="wasm")
    listing = subprocess.run(["go", "list", "-deps", "-json", "./cmd/wasm"], env=env, check=True,
                             capture_output=True, text=True).stdout
    modules = {}
    for blob in re.split(r"(?m)^}\n(?={)", listing.strip()):
        mod = json.loads(blob if blob.endswith("}") else blob + "}").get("Module")
        if mod and not mod.get("Main"):
            modules[mod["Path"]] = (mod["Version"], Path(mod["Dir"]))
    goroot = Path(subprocess.run(["go", "env", "GOROOT"], check=True, capture_output=True, text=True).stdout.strip())
    gover = subprocess.run(["go", "env", "GOVERSION"], check=True, capture_output=True, text=True).stdout.strip()
    components = [{"type": "library", "bom-ref": f"pkg:golang/std@{gover}", "name": "std", "version": gover,
                   "description": "Go standard library and runtime, compiled into the module; wasm_exec.js ships beside it",
                   "purl": f"pkg:golang/std@{gover}", "licenses": [licence_of(goroot)]}]
    hashes = zip_hashes(modules)
    for path, (version, d) in sorted(modules.items()):
        purl = f"pkg:golang/{path}@{version}"
        components.append({"type": "library", "bom-ref": purl, "name": path, "version": version, "purl": purl,
                           "hashes": [{"alg": "SHA-256", "content": hashes[path]}], "licenses": [licence_of(d)]})
    body = {
        "bomFormat": "CycloneDX", "specVersion": "1.7", "version": 1,
        "metadata": {"component": {"type": "library", "bom-ref": "qsh-fedthresh", "name": "qsh-fedthresh",
                                   "description": "Threshold CKKS and BGV aggregation for the medical page, on Lattigo, compiled to WebAssembly",
                                   "licenses": [{"license": {"id": "Apache-2.0"}}]}},
        "components": components,
        "dependencies": [{"ref": "qsh-fedthresh", "dependsOn": [c["bom-ref"] for c in components]}],
    }
    serial = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(body, sort_keys=True))
    doc = {"bomFormat": "CycloneDX", "specVersion": "1.7", "serialNumber": f"urn:uuid:{serial}", **{k: v for k, v in body.items() if k not in ("bomFormat", "specVersion")}}
    Path(out).write_text(json.dumps(doc, indent=2) + "\n")
    print(f"sbom: {len(components)} components -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
