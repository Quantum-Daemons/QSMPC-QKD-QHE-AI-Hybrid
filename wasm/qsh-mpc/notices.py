"""Writes the third-party notices for every crate compiled into the qsh-mpc wasm, or fails the build.

Run by the Rust stage of docker/Dockerfile.web after `cargo fetch`, with only the standard library:

    python3 notices.py <output.txt>

WHAT IS COVERED. The crates that end up in the shipped binary: the normal-dependency closure of this
package for wasm32-unknown-unknown, without proc-macro crates and build dependencies, which run at
compile time and ship nothing. MIT and BSD require their notices to travel with a binary, which a wasm
module served to every visitor is; so each crate's OWN licence files are copied in, copyright lines
included, rather than a generic text.

WHAT FAILS THE BUILD. A licence expression that no permissive alternative satisfies, a crate with no
licence file to quote, or an output that does not cover the whole closure. A new dependency therefore
cannot ship under terms nobody reviewed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ALLOWED = {
    "MIT", "MIT-0", "Apache-2.0", "Apache-2.0 WITH LLVM-exception", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "Zlib", "Unicode-3.0", "Unicode-DFS-2016", "CC0-1.0", "Unlicense", "0BSD",
}
LICENSE_FILE = re.compile(r"^(licen[cs]e|copying|notice|unlicense|copyright)", re.I)


def acceptable(expr: str) -> bool:
    """True when some OR-alternative of the SPDX expression consists only of allowed licences."""
    expr = expr.replace("/", " OR ")
    for alt in re.split(r"\s+OR\s+", expr.replace("(", " ").replace(")", " ")):
        parts = [p.strip() for p in re.split(r"\s+AND\s+", alt) if p.strip()]
        if parts and all(p in ALLOWED for p in parts):
            return True
    # An AND that spans an OR group, e.g. "(MIT OR Apache-2.0) AND Unicode-3.0".
    m = re.fullmatch(r"\((.+)\)\s+AND\s+(.+)", expr.strip())
    return bool(m) and acceptable(m.group(1)) and acceptable(m.group(2))


def main(out: Path) -> int:
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked", "--filter-platform", "wasm32-unknown-unknown"],
        check=True, capture_output=True, text=True).stdout)
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for dep in nodes[pid]["deps"]:
            if any(k["kind"] is None for k in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    shipped = sorted(
        (packages[p] for p in seen if p != root
         and not any("proc-macro" in t["kind"] for t in packages[p]["targets"])),
        key=lambda p: (p["name"], p["version"]))

    problems, blocks = [], []
    for p in shipped:
        expr = p.get("license") or ""
        if not acceptable(expr):
            problems.append(f"{p['name']} {p['version']}: licence {expr or 'NONE'!r} is not on the permissive list")
        folder = Path(p["manifest_path"]).parent
        files = sorted(f for f in folder.iterdir() if f.is_file() and LICENSE_FILE.match(f.name))
        if not files:
            problems.append(f"{p['name']} {p['version']}: no licence file to quote")
            continue
        texts = "\n".join(f"--- {f.name} ---\n{f.read_text(errors='replace').strip()}\n" for f in files)
        blocks.append(f"== {p['name']} {p['version']} ==\nLicence: {expr}\n"
                      f"Source: {p.get('repository') or p['source'] or 'unknown'}\n\n{texts}")
    if problems:
        print("third-party notices: refusing to build\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    if len(blocks) != len(shipped):
        print(f"third-party notices cover {len(blocks)} of {len(shipped)} crates", file=sys.stderr)
        return 1
    header = (
        "Third-party notices for the qsh-mpc WebAssembly module (the demo's garbled-circuit tier).\n"
        f"{len(blocks)} crates are compiled into it. Each entry quotes the crate's own licence files.\n"
        "Generated at build time by wasm/qsh-mpc/notices.py from the locked dependency graph.\n\n")
    out.write_text(header + "\n".join(blocks))
    print(f"third-party notices: {len(blocks)} crates -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
