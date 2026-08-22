"""The QSDC secrecy capacity must be one expression, not two that happen to match.

WHY THIS EXISTS

Three corrections have now been shipped to this single formula:

    Round 24  the abort threshold was a literal 0.11 - the g = 1 root of the same expression, which
              is why it coincided with the Shor-Preskill bound and looked right
    Round 55  `g` omitted the detector efficiency, so it was too small and C_s too large
    Round 55  the second entropy took a POOLED error rate where Qi et al. Eq. (8) takes `e_x + e_z`,
              understating Eve by 2.24x and overstating C_s by 26% at the paper's own operating point

All three were permissive. Each had to be applied by hand in `src/qsh/qkd/qsdc.py` AND in
`web/src/lib/clientCompute.ts`, and nothing could check that the second application happened - the
browser copy lived in a `switch` arm inside a generator, which is not importable.

AND THE THIRD FIX MISSED A SITE, IN THE FILE IT WAS BEING APPLIED TO. `clientCompute.ts` evaluates
the capacity twice: once for the measured channel and once for the same channel with Eve removed, to
attribute the abort. The commit message said the change had been made in both languages; within the
browser file it had been made in one of two places, three hundred lines apart. That is the defect
class this project has recorded nine times, at a granularity below the one it had been watching.

WHAT IS ASSERTED

That `qsdcSecrecyCapacity(e, g)` in the browser and the Python expression agree across a grid
spanning the demo's own operating range, and that the paper's own published number is reproduced.
The reproduction is the part that cannot drift silently: two implementations can agree with each
other while both being wrong, which is exactly what happened for the pooled argument.

WHAT IS DELIBERATELY NOT ASSERTED

Not the finite-size bound - that is Python-only, because the browser path has no sampled check
subset and reports the asymptotic limit. Not `g`; `tests/test_the_browser_computes_the_bound_the_
page_names.py` covers the channel model.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

#: (e, g) pairs spanning what the demo can produce: the free-space misalignment floor at 0.025, the
#: fibre floor near 0.001, Eve-driven rates up to 0.14, and `g` from Qi's measured 2.57 up to the
#: 701 this project's own downlink reaches once detector efficiency is included.
GRID = [(e, g) for e in (0.0, 0.001, 0.006, 0.008, 0.016, 0.025, 0.05, 0.14, 0.3, 0.5)
        for g in (1.0, 2.57, 4.446, 40.0, 280.43, 701.06)]


def _h2(x: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def _python(e: float, g: float) -> float:
    """The Python expression, written out rather than imported.

    Deliberate. Importing `qsdc.py` would make this a check that one file equals itself plus a
    bundler round-trip. Written here, the test states what the formula IS, so a change to the Python
    that is not also a change to the paper has to come past this line.
    """
    return 1.0 - _h2(e) - g * _h2(min(0.5, 2.0 * e))


def _require_browser_or_skip() -> None:
    """Skip when the browser toolchain is absent - unless the environment forbids skipping.

    ADDED ROUND 55, HOURS AFTER THIS FILE WAS WRITTEN, AND THE ORIGINAL WAS WRONG TWICE.

    It used a bare `pytest.skip`, making this the only one of the three cross-language browser gates
    not escalated by `QSH_REQUIRE_BROWSER_GATES=1`. Measured with the variable set and esbuild
    absent: the two twins FAIL and this one SKIPs. The skip is exactly the hole those twins exist to
    close - `web/node_modules` is gitignored and the CI python job installs no Node, so a
    one-language fix ships green on a skip.

    And its message claimed compensation that does not exist: it said "this skip is why `check:qkd`
    is also a CI gate", but `qsdcSecrecyCapacity` appears **zero** times anywhere in `web/scripts`.
    Nothing in the web job checks this identity. Citing a compensating control without checking it
    is the same move as citing an authority for a number that is not in it, which `RI11` exists for.
    """
    import os

    if ESBUILD.exists():
        return
    msg = (
        "web/node_modules/.bin/esbuild is absent, so the browser expression cannot be executed. "
        "Run `npm ci` in web/. This is a SKIP rather than a pass: the identity is unchecked, not "
        "satisfied - and NOTHING in web/scripts covers `qsdcSecrecyCapacity`, so no other gate is "
        "watching this one."
    )
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(
            msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE: the cross-language "
            "checks are mandatory in this environment."
        )
    pytest.skip(msg)


def _browser() -> list[float]:
    _require_browser_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            "export { qsdcSecrecyCapacity } from "
            f"{json.dumps(str(WEB / 'src/lib/quantumSim'))};\n")
        subprocess.run(
            [str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
             f"--outfile={bundle}", "--log-level=error"],
            cwd=WEB, check=True, capture_output=True)
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            f"const grid = {json.dumps(GRID)};\n"
            "console.log(JSON.stringify(grid.map(([e, g]) => m.qsdcSecrecyCapacity(e, g))));\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def test_the_two_implementations_agree_across_the_operating_range():
    got = _browser()
    assert len(got) == len(GRID)
    worst = max(abs(b - _python(e, g)) for (e, g), b in zip(GRID, got))
    assert worst < 1e-12, (
        f"the browser and Python secrecy capacities differ by up to {worst:.3e}. This formula has "
        "taken three corrections and each one had to be applied twice by hand; a divergence here "
        "means one of the two copies took a fix the other did not.")


def test_the_paper_s_own_number_is_reproduced():
    """Two implementations can agree and both be wrong. This is the check that does not rely on them.

    Qi et al. state `I(A:E) = g * Q_Bob * h(e_x + e_z) = 9.1e-4` with `g = 2.57`, `Q_Bob = 0.3%`,
    `e_x = e_z = 0.8%`. Only the SUM reproduces it - the pooled `h(e)` at e = 0.6% gives 4.08e-4,
    which is what this project computed for several rounds.
    """
    g, q_bob = 2.57, 0.003
    leak_sum = g * q_bob * _h2(0.008 + 0.008)
    assert abs(leak_sum - 9.1e-4) < 5e-6, (
        f"h(e_x + e_z) gives I(A:E) = {leak_sum:.5e}, and the paper states 9.1e-4. If this no longer "
        "matches, the entropy argument has changed and the change is not the paper's.")

    leak_pooled = g * q_bob * _h2(0.006)
    assert abs(leak_pooled - 9.1e-4) > 1e-4, (
        "the pooled argument now also reproduces the paper's figure, which would mean this test can "
        "no longer tell the two forms apart - re-derive it")
    assert leak_sum / leak_pooled > 2.0, (
        f"the sum form leaks {leak_sum / leak_pooled:.2f}x what the pooled form does; this was 2.24x "
        "and it is the size of the error that was shipped")


def test_neither_language_evaluates_the_capacity_off_the_shared_helper():
    """The reason the third fix missed a site: two call sites, one of them out of sight.

    Asserted on the source because the output cannot carry it - a second inlined copy that happens to
    be correct today produces identical numbers and drifts on the next correction.
    """
    src = (WEB / "src" / "lib" / "clientCompute.ts").read_text()
    assert "qsdcSecrecyCapacity" in src, "clientCompute no longer calls the shared helper"
    import re
    inline = re.findall(r"1\s*-\s*h2\([^)]*\)\s*-\s*g\s*\*\s*h2\(", src)
    assert not inline, (
        f"{len(inline)} inlined secrecy-capacity expression(s) remain in clientCompute.ts. One of "
        "these was missed by the very commit that corrected the formula, in the same file, because "
        "the two sites are three hundred lines and a switch arm apart. Call the helper.")
