"""Writes the notices for everything compiled into the qsh-fedthresh wasm, or fails the build.

Run by the Go stage of docker/Dockerfile.web after `go mod download`, with only the standard library:

    python3 notices.py <output.txt>

WHAT IS COVERED. Every module whose packages `GOOS=js GOARCH=wasm go list -deps ./cmd/wasm` reports, and Go itself:
the standard library and runtime are compiled into the module and `wasm_exec.js` is copied from Go's tree. Each
module's own licence and notice files are copied in, copyright lines included; Apache-2.0 asks for NOTICE files to
travel with a redistribution, and BSD and MIT for their notices.

WHAT FAILS THE BUILD. A module without a licence file, or a licence text that is none of Apache-2.0, BSD, ISC or MIT. Lattigo's
non-test packages import testify and go-cmp, so they and their dependencies are compiled in and listed too.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

LICENCE_FILE = re.compile(r"^(licen[cs]e|copying|notice|patents)(\..*)?$", re.I)
KINDS = {
    "Apache-2.0": "Apache License",
    "BSD": "Redistribution and use in source and binary forms",
    "MIT": "Permission is hereby granted, free of charge",
    "ISC": "Permission to use, copy, modify, and/or distribute this software",
}


def kinds_of(text: str) -> set[str]:
    """Every licence the text carries: a file may hold two (gopkg.in/yaml.v3's holds MIT and Apache-2.0)."""
    return {k for k, marker in KINDS.items() if marker in text}


def main(out: str) -> None:
    env = dict(os.environ, GOOS="js", GOARCH="wasm")
    listing = subprocess.run(["go", "list", "-deps", "-json", "./cmd/wasm"], env=env, check=True,
                             capture_output=True, text=True).stdout
    modules = {}
    for blob in re.split(r"(?m)^}\n(?={)", listing.strip()):
        pkg = json.loads(blob if blob.endswith("}") else blob + "}")
        mod = pkg.get("Module")
        if mod and not mod.get("Main"):
            modules[mod["Path"]] = (mod["Version"], mod["Dir"])
    goroot = subprocess.run(["go", "env", "GOROOT"], check=True, capture_output=True, text=True).stdout.strip()
    goversion = subprocess.run(["go", "env", "GOVERSION"], check=True, capture_output=True, text=True).stdout.strip()
    sections = [("Go " + goversion + " (standard library, runtime, wasm_exec.js)", Path(goroot))]
    sections += [(f"{path} {version}", Path(d)) for path, (version, d) in sorted(modules.items())]
    parts, problems = [], []
    for name, d in sections:
        files = sorted(f for f in d.iterdir() if f.is_file() and LICENCE_FILE.match(f.name))
        texts = [(f.name, f.read_text(errors="replace")) for f in files]
        kinds = set().union(*(kinds_of(t) for n, t in texts if not n.upper().startswith(("NOTICE", "PATENTS"))))
        if not texts or not kinds:
            problems.append(f"{name}: no licence file of a known kind in {d}")
            continue
        parts.append(f"== {name} ({', '.join(sorted(kinds))})\n")
        parts += [f"-- {n}\n{t.rstrip()}\n" for n, t in texts]
    if problems:
        sys.exit("notices: " + "; ".join(problems))
    Path(out).write_text("Third-party notices for qsh_fedthresh.wasm and wasm_exec.js\n\n" + "\n".join(parts))
    print(f"notices: {len(sections)} sections -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
