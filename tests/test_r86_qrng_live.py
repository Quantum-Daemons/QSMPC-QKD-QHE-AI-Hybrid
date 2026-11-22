"""The live QRNG view's health tests are SP 800-90B's, sample for sample, as the Python package runs them (2026-11-23).

WHY THIS EXISTS

On the owner's choice of 2026-11-22 the telecom page streams simulated samples through the two continuous health tests
of SP 800-90B section 4.4 and lets a visitor inject a fault (`web/src/lib/qrngHealthStream.ts`, `QrngHealthLive`). A
live counter that drifts from the batch test it illustrates teaches the wrong cutoff, so the stream is bundled with
esbuild, run in node on the same samples as `qsh.qrng.health`, and held to it.

WHAT IS ASSERTED

- On a fair stream, a stream that sticks at 1, and one that comes up 1 three times in four, all a whole number of
  windows long: both cutoffs, the longest run, the worst window count and each test's verdict are Python's.
- Each test fails at the sample where its count first reaches the cutoff, found here independently.
- The fair stream passes, the stuck one fails the repetition count, the biased one fails the adaptive proportion.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from qsh.qrng import health

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
H, W = 0.9, 1024

RUNNER = r"""
import { readFileSync } from "node:fs";
const m = await import(%(bundle)s);
const streams = JSON.parse(readFileSync(%(data)s, "utf8"));
const out = {};
for (const [name, bits] of Object.entries(streams)) {
  const s = m.newHealthState(%(h)s, %(w)s);
  for (const b of bits) m.feed(s, b);
  out[name] = s;
}
console.log(JSON.stringify(out));
"""


def _streams() -> dict[str, list[int]]:
    rng = np.random.default_rng(20261123)
    n = 64 * W
    fair = rng.integers(0, 2, n)
    stuck = fair.copy()
    stuck[n // 2:] = 1
    biased = (rng.random(n) < 0.75).astype(int)
    return {"fair": fair.tolist(), "stuck": stuck.tolist(), "biased": biased.tolist()}


def _first_failure(bits: np.ndarray, rct_c: int, apt_c: int) -> tuple[int | None, int | None]:
    rct = apt = None
    run, prev = 0, None
    for i, b in enumerate(bits):
        run = run + 1 if b == prev else 1
        prev = b
        if rct is None and run >= rct_c:
            rct = i + 1
    for start in range(0, bits.size, W):
        win = bits[start:start + W]
        hits = np.flatnonzero(np.cumsum(win == win[0]) >= apt_c)
        if hits.size:
            apt = start + int(hits[0]) + 1
            break
    return rct, apt


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()), ("node", shutil.which("node")))
               if not ok]
    if missing:
        msg = f"{' and '.join(missing)} absent, so the browser code cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = tmp_path_factory.mktemp("qrnglive")
    entry, bundle, runner, data = d / "entry.ts", d / "bundle.mjs", d / "run.mjs", d / "streams.json"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/qrngHealthStream'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node", f"--outfile={bundle}",
                    "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    data.write_text(json.dumps(_streams()))
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri()), "data": json.dumps(str(data)), "h": H, "w": W})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"the browser module did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", ["fair", "stuck", "biased"])
def test_the_stream_is_the_python_health_tests(browser, name: str) -> None:
    bits = np.asarray(_streams()[name])
    js = browser[name]
    rct_ok, rct_c, longest = health.repetition_count_test(bits, H)
    apt_ok, apt_c, worst = health.adaptive_proportion_test(bits, H, W)
    assert (js["rctCutoff"], js["aptCutoff"]) == (rct_c, apt_c)
    assert js["longestRun"] == longest and js["worstCount"] == worst
    assert (js["rctFailedAt"] is None) == rct_ok and (js["aptFailedAt"] is None) == apt_ok
    assert (js["rctFailedAt"], js["aptFailedAt"]) == _first_failure(bits, rct_c, apt_c)


def test_each_fault_is_caught_by_its_own_test(browser) -> None:
    assert browser["fair"]["rctFailedAt"] is None and browser["fair"]["aptFailedAt"] is None
    assert browser["stuck"]["rctFailedAt"] is not None
    assert browser["biased"]["aptFailedAt"] is not None
