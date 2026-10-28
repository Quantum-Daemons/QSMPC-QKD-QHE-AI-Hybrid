"""The trained QCNN's answer depends on its input and its weights, and on nothing random.

WHY THIS EXISTS

Checklist row QC2 asks that the class be stable across run seeds for a fixed input, because a class
that flips with the seed is noise. It was carried as UNVERIFIED from Round 70 because the page cannot
show it. The pipeline draws its input row with the run's seeded generator, so a new seed brings a new
row, and no control holds the row fixed while the seed changes. A live walk can only ever compare two
runs with one seed.

So the property is checked where it lives: in the function. `qcnnInferTrained(features, w)` takes no
seed. This executes the browser's own module on every shipped QCNN artifact, twice on the same
input, with `Math.random` and `crypto.getRandomValues` replaced by different generators between the
calls. Identical outputs mean nothing random reaches the answer, so the class for a fixed input is
the same whatever the seed.

Seeded by making the readout depend on `Math.random`: the comparison fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
ARTIFACTS = sorted((WEB / "public" / "data").glob("*/qcnn_weights*.json"))

_RUNNER = r"""
const m = await import(BUNDLE);
const fs = await import("node:fs");
const out = [];
for (const path of PATHS) {
  const w = JSON.parse(fs.readFileSync(path, "utf8"));
  // A fixed input of the artifact's own width, in the units it declares.
  const x = Array.from({ length: w.n_qubits }, (_, i) => Math.sin(1.7 * i + 0.3) * (w.input_space === "raw" ? 3 : 1));
  const runs = [];
  for (const seed of [1, 99991]) {
    let s = seed;
    const next = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    Math.random = next;
    globalThis.crypto.getRandomValues = (a) => { for (let i = 0; i < a.length; i++) a[i] = Math.floor(next() * 4294967296) >>> 0; return a; };
    const r = m.qcnnInferTrained(x, w);
    runs.push({ cls: r.cls, expectation: r.expectation, logit: r.logit, probs: r.probs ?? null });
  }
  out.push({ path: path.split("/").slice(-2).join("/"), runs });
}
console.log(JSON.stringify(out));
"""


def _run() -> list[dict]:
    if not ESBUILD.exists():
        msg = "web/node_modules/.bin/esbuild is absent; run `npm ci` in web/. This is a SKIP rather than a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE.")
        pytest.skip(msg)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "qsim.mjs"
        subprocess.run([str(ESBUILD), str(WEB / "src" / "lib" / "quantumSim.ts"), "--bundle", "--format=esm",
                        "--platform=node", f"--outfile={bundle}", "--log-level=error"],
                       cwd=WEB, check=True, capture_output=True)
        code = (_RUNNER.replace("BUNDLE", json.dumps(bundle.as_uri()))
                .replace("PATHS", json.dumps([str(p) for p in ARTIFACTS])))
        out = subprocess.run(["node", "--input-type=module", "-e", code], check=True, capture_output=True,
                             text=True)
        return json.loads(out.stdout)


def test_every_shipped_artifact_is_found() -> None:
    """Anti-tautology: an empty glob would make the comparison below vacuous."""
    assert len(ARTIFACTS) >= 6, f"only {len(ARTIFACTS)} QCNN artifacts found"


def test_the_answer_is_the_same_whatever_the_randomness() -> None:
    results = _run()
    assert len(results) == len(ARTIFACTS)
    for r in results:
        first, second = r["runs"]
        assert first == second, (
            f"{r['path']}: the same input gave {first} and then {second} with different random "
            "generators installed, so something random reaches the QCNN's answer and the class for "
            "a fixed input would depend on the run seed")
