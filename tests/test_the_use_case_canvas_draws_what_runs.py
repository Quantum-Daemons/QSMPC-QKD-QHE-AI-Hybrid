"""The use-case canvas draws each stage where the code runs it, the way its data goes.

WHY THIS EXISTS (Round 80)

The owner held the merge for an animated architecture on the five use-case pages, in place of the
two-party sequence diagram. Drawing it meant stating, stage by stage, which component on which side
does what and which way the data goes, and the old figure had three of those facts wrong:
- the satellite's key and its QSDC telemetry went UP, ground to satellite, on a link the code models
  as a downlink;
- genomics drew two cohorts for a secure sum the run makes over four;
- under a PQC-only profile both figures drew a quantum QSDC arrow, though the browser skips QSDC
  there and sends the telemetry under the session key.

`web/scripts/check_use_case_canvas.mjs` asserts those facts on `canvasRow` (src/lib/useCaseFlows.ts),
the one place the canvas reads them from, through the real module bundled by esbuild. This file runs
it from the Python suite, where CI sets `QSH_REQUIRE_BROWSER_GATES=1`, so the check cannot pass by
being skipped. The seeded negative restores the old drawing's direction for the satellite's key.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
SCRIPT = WEB / "scripts" / "check_use_case_canvas.mjs"
MODULE = WEB / "src" / "lib" / "useCaseFlows.ts"


def _require_browser_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = (f"{' and '.join(missing)} absent, so the canvas check cannot run. Run `npm ci` in web/. "
           "This is a SKIP, not a pass.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def _run(module: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if module is not None:
        env["CHECK_MODULE"] = str(module.with_suffix(""))
    return subprocess.run(["node", str(SCRIPT)], cwd=WEB, capture_output=True, text=True, timeout=300, env=env)


def test_the_canvas_places_every_stage_where_it_runs() -> None:
    _require_browser_or_skip()
    r = _run()
    assert r.returncode == 0, f"check:use-case-canvas failed:\n{r.stdout}\n{r.stderr}"
    assert "check:use-case-canvas OK" in r.stdout, r.stdout


def test_the_check_fails_on_the_old_drawings_uplink() -> None:
    """The seeded negative: the satellite's key drawn ground to satellite, as the old figure drew it."""
    _require_browser_or_skip()
    src = MODULE.read_text()
    downlink = '{ a: "QKD receiver: telescope and detectors", b: "QKD source on board", flow: "ba", channel: "quantum" }'
    assert src.count(downlink) == 1, "the satellite key's canvas row moved; re-derive this seed"
    seeded = MODULE.with_name("zz_seeded_useCaseFlows.ts")
    try:
        seeded.write_text(src.replace(downlink, downlink.replace('flow: "ba"', 'flow: "ab"')))
        r = _run(seeded)
    finally:
        seeded.unlink(missing_ok=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1, f"the uplink passed the check:\n{out}"
    assert "the satellite's QKD key is drawn ab" in out, out
