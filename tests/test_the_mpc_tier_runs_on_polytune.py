"""The demo's garbled-circuit tier runs on Polytune, built and tested in the web image.

WHY THIS EXISTS

Round 65 replaced mpc-framework + emp-wasm-engine + summon-ts - all three archived upstream, and the
first of them pulling a zod release with an open advisory and no fix coming - with Polytune, after a
prototype (Round 63) and a measurement over netem-shaped TCP (Round 64). The decisions recorded with
the change are each one line of configuration that could quietly come undone:

- the archived packages are gone from the dependency tree, not merely unused;
- clientMPC.ts runs the wasm built from wasm/qsh-mpc and imports nothing from the old stack;
- the web image builds that wasm in a Rust stage whose base image is pinned by digest, whose
  parallelism is limited because the demo host is shared, and which runs `cargo test` BEFORE it builds
  the module - so every image, the public demo's included, ships an argmax that was just checked;
- the stage writes third-party notices for every crate compiled in (MIT and BSD require them with a
  distributed binary), refusing licences outside a permissive list, and the page links them;
- the built module and its notices are generated, never committed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
OLD = ("mpc-framework", "emp-wasm-engine", "summon-ts")


def _strip(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?<![:\"'`])//[^\n]*", " ", src)


def test_the_archived_packages_are_gone() -> None:
    pkg = json.loads((WEB / "package.json").read_text())
    declared = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert not [p for p in OLD if p in declared], "an archived MPC package is declared again"
    lock = json.loads((WEB / "package-lock.json").read_text())["packages"]
    assert not [k for k in lock if any(k.endswith(f"node_modules/{p}") for p in OLD + ("zod",))], (
        "an archived MPC package (or the zod it pulled) is back in the lockfile")


def test_client_mpc_runs_the_polytune_module() -> None:
    src = _strip((WEB / "src" / "lib" / "clientMPC.ts").read_text())
    assert 'import("../wasm/qsh_mpc/qsh_mpc.js")' in src, "clientMPC.ts no longer loads the qsh_mpc wasm"
    assert "secure_argmax(" in src
    assert not [p for p in OLD if p in src], "clientMPC.ts imports from the archived stack again"


def _stage(dockerfile: str, name: str) -> str:
    m = re.search(rf"^FROM\s+\S+\s+AS\s+{name}\s*$(.*?)(?=^FROM\s)", dockerfile, re.M | re.S)
    assert m, f"docker/Dockerfile.web has no `{name}` stage"
    return m.group(0)


def test_the_image_tests_then_builds_the_module() -> None:
    dockerfile = (REPO / "docker" / "Dockerfile.web").read_text()
    wasm = _stage(dockerfile, "wasm")
    assert re.search(r"^FROM rust:\S+@sha256:[0-9a-f]{64}\s+AS wasm", wasm, re.M), (
        "the Rust stage's base image is not pinned by digest")
    assert re.search(r"^ARG CARGO_JOBS=2$", wasm, re.M) and "CARGO_BUILD_JOBS=${CARGO_JOBS}" in wasm, (
        "the Rust stage no longer limits its parallelism (the demo host is shared)")
    test_at, build_at = wasm.find("cargo test"), wasm.find("wasm-pack build")
    assert 0 <= test_at < build_at, "`cargo test` must run in the image, BEFORE `wasm-pack build`"
    assert "--locked" in wasm[test_at:] and "python3 notices.py" in wasm, (
        "the stage must build from the lockfile and write the third-party notices")
    build = _stage(dockerfile + "\nFROM end\n", "build")
    assert "COPY --from=wasm /out/qsh_mpc ./src/wasm/qsh_mpc" in build
    assert "COPY --from=wasm /out/qsh-mpc-notices.txt ./public/licenses/qsh-mpc.txt" in build


def test_the_notices_refuse_copyleft_and_are_linked() -> None:
    notices = (REPO / "wasm" / "qsh-mpc" / "notices.py").read_text()
    allowed = re.search(r"ALLOWED = \{(.*?)\}", notices, re.S).group(1)
    assert not re.search(r"\b(A?GPL|LGPL|MPL|EPL|CDDL|EUPL|SSPL|OSL)\b", allowed), (
        "a copyleft licence was added to the permissive list the notices enforce")
    assert "refusing to build" in notices and "len(blocks) != len(shipped)" in notices, (
        "the generator no longer fails on an unlisted licence or an incomplete closure")
    card = (WEB / "src" / "components" / "MpcNetBench.tsx").read_text()
    assert 'href="/licenses/qsh-mpc.txt"' in card, "the page no longer links the third-party notices"


def test_the_built_module_is_never_committed() -> None:
    import subprocess
    for path in ("web/src/wasm/qsh_mpc/qsh_mpc_bg.wasm", "web/public/licenses/qsh-mpc.txt",
                 "wasm/qsh-mpc/target/x"):
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
        assert r.returncode == 0, f"{path} is not gitignored"
    dockerignore = (REPO / ".dockerignore").read_text().splitlines()
    for rule in ("web/src/wasm/", "web/public/licenses/qsh-mpc.txt", "prototypes/"):
        assert rule in dockerignore, f".dockerignore no longer excludes {rule}"
