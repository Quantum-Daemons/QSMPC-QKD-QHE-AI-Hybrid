"""The pulse-by-pulse BB84 view measures the QBER the page's model states (Round 86, 2026-11-23).

WHY THIS EXISTS

On the owner's choice of 2026-11-22 the telecom page draws BB84 pulses one by one, with an intercept-resend
eavesdropper to switch on, and puts each press's measured QBER beside the page's closed-form model
(`web/src/lib/bb84Photons.ts`, `Bb84PhotonPanel`). A simulation that disagreed with the model, or a model line the
simulation could not reach, would teach a wrong number either way. The module is bundled with esbuild and run in node
on fixed seeds.

WHAT IS ASSERTED

- With no misalignment and no eavesdropper, no kept pulse is wrong.
- At Eve's shares 0, 0.5 and 1, the measured QBER is within four standard errors of the model, which is the page's own
  composition of misalignment with a quarter of Eve's share (Python's twin, `compose_independent_errors`).
- Each intensity's gain is 1 - exp(-transmittance x mean photons) within four standard errors, intercepted or not.
- The table holds pulses Bob detected, and Eve touches only pulses that carry a photon.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from qsh.qkd.channel import compose_independent_errors

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
ETA, MU, NU, MIS = 0.05, 0.3416, 0.1417, 0.01

RUNNER = r"""
const m = await import(%(bundle)s);
const base = { pulses: 400000, transmittance: %(eta)s, mu: %(mu)s, nu: %(nu)s, pSignal: 0.5, keep: 2000 };
const out = { clean: m.simulateBb84({ ...base, eve: 0, misalignment: 0, seed: 11 }), runs: {} };
for (const eve of [0, 0.5, 1]) out.runs[eve] = m.simulateBb84({ ...base, eve, misalignment: %(mis)s, seed: 100 + eve * 10 });
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()), ("node", shutil.which("node")))
               if not ok]
    if missing:
        msg = f"{' and '.join(missing)} absent, so the browser code cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = tmp_path_factory.mktemp("bb84")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/bb84Photons'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node", f"--outfile={bundle}",
                    "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri()), "eta": ETA, "mu": MU, "nu": NU, "mis": MIS})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"the browser module did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_clean_channel_has_no_errors(browser) -> None:
    c = browser["clean"]
    assert c["signal"]["sifted"] + c["decoy"]["sifted"] > 1000
    assert c["signal"]["errors"] + c["decoy"]["errors"] == 0 and c["qber"] == 0


@pytest.mark.parametrize("eve", ["0", "0.5", "1"])
def test_the_measured_qber_is_the_models(browser, eve: str) -> None:
    r = browser["runs"][eve]
    want = compose_independent_errors(MIS, 0.25 * float(eve))
    assert r["model"] == pytest.approx(want, rel=1e-12)
    n = r["signal"]["sifted"] + r["decoy"]["sifted"]
    sd = math.sqrt(want * (1 - want) / n)
    assert abs(r["qber"] - want) < 4 * sd, (eve, r["qber"], want, n)
    assert r["ci95"][0] <= r["qber"] <= r["ci95"][1]


@pytest.mark.parametrize("eve", ["0", "1"])
def test_the_gains_are_the_channels(browser, eve: str) -> None:
    r = browser["runs"][eve]
    for kind, mean in (("signal", MU), ("decoy", NU)):
        t = r[kind]
        q = 1 - math.exp(-ETA * mean)
        sd = math.sqrt(q * (1 - q) / t["sent"])
        assert abs(t["detected"] / t["sent"] - q) < 4 * sd, (eve, kind)


def test_the_table_shows_detected_pulses_and_eve_touches_only_photons(browser) -> None:
    r = browser["runs"]["1"]
    first = r["first"]
    assert len(first) == 2000 and all(p["clicked"] for p in first)
    assert r["firstSpan"] >= len(first)
    assert any(p["eve"] for p in first)
    assert all(p["photons"] > 0 for p in first if p["eve"])
