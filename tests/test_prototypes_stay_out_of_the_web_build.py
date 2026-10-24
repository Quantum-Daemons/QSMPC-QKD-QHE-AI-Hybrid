"""A prototype under prototypes/ is measured beside the demo, never shipped inside it.

WHY THIS EXISTS

Round 63 added `prototypes/polytune-argmax/`: the demo's garbled-circuit argmax rebuilt on Polytune, so
that a replacement for the archived mpc-framework + emp-wasm + summon tier could be measured before
anything is replaced. The decision recorded with it is that a prototype is NOT part of the production
build until a later change adopts it on the strength of its measurements. That is a property of the
tree, and a single import would quietly break it - the bundle would ship code that was only ever
benchmarked, and the web image would start depending on a Rust toolchain it does not have.

So this holds three things: nothing under web/ refers to prototypes/ or to the prototype's package;
the web image's Dockerfile does not copy or build prototypes/; and the prototype's build outputs are
gitignored, so a built wasm package cannot be committed by accident.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROTOTYPES = REPO / "prototypes"


def _web_sources():
    for p in sorted((REPO / "web").rglob("*")):
        if "node_modules" in p.parts or "dist" in p.parts or not p.is_file():
            continue
        if p.suffix in {".ts", ".tsx", ".js", ".mjs", ".json", ".html"}:
            yield p


def test_nothing_under_web_refers_to_a_prototype() -> None:
    names = [d.name.replace("-", "_") for d in PROTOTYPES.iterdir() if d.is_dir()]
    assert names, "no prototype found - if prototypes/ was removed, remove this test with it"
    pattern = re.compile(r"prototypes/|" + "|".join(re.escape(n) for n in names))
    hits = [str(p.relative_to(REPO)) for p in _web_sources()
            if p.name != "package-lock.json" and pattern.search(p.read_text(errors="replace"))]
    assert not hits, (
        "web/ refers to a prototype, so it would ship in the demo bundle. Prototypes are measured "
        f"beside the demo and adopted by a deliberate change, not by an import: {hits}")


def test_the_web_image_does_not_build_prototypes() -> None:
    dockerfile = (REPO / "docker" / "Dockerfile.web").read_text()
    assert "prototypes" not in dockerfile, "docker/Dockerfile.web copies or builds prototypes/"
    assert not re.search(r"\b(cargo|wasm-pack|rustup)\b", dockerfile), (
        "the web image invokes a Rust toolchain; the demo's build must not depend on one")


def test_prototype_build_outputs_are_ignored() -> None:
    # The ROOT rules, not only the effective state: wasm-pack writes a `.gitignore` of `*` into pkg/,
    # so `git check-ignore` passes while pkg/ exists even with no root rule, and stops passing the day
    # the directory is regenerated without it. Seeding the root rule away found this.
    root = (REPO / ".gitignore").read_text().splitlines()
    for rule in ("prototypes/*/target/", "prototypes/*/pkg/", "prototypes/*/.cargo-home/"):
        assert rule in root, f"the root .gitignore no longer carries {rule!r}"
    for d in PROTOTYPES.iterdir():
        if not d.is_dir():
            continue
        for out in ("target/x", "pkg/x.wasm", ".cargo-home/x"):
            probe = d / out
            r = subprocess.run(["git", "check-ignore", "-q", str(probe.relative_to(REPO))], cwd=REPO)
            assert r.returncode == 0, f"{probe.relative_to(REPO)} is not gitignored"
