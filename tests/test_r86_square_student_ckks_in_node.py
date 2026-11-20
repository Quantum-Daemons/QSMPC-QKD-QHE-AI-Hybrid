"""The fully encrypted medical student under the page's own node-seal, end to end, in node (Round 86, 2026-11-19).

`web/src/lib/clientFHESquare.ts` runs `medical_fl.square_student` with no layer decrypted in between, in its own CKKS
context, under each parameter set its published `weights.json` lists. This bundles the module with esbuild, as
tests/test_r86_satellite_student_ckks_in_node.py bundles the forecaster, and runs it against the REAL node-seal on held-out
test images, so the page's claim is checked where it can be: the pre-registered limit (every decrypted logit within 1e-3
of the plaintext one, every decision unchanged) holds in the browser's own library, not only in TenSEAL; the published
sets are exactly the two that passed; and the module's float64 reference is the Python one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
MODEL = REPO / "models" / "medical_breastmnist_square_student"
ROWS = (0, 1, 2, 3, 40, 77, 118, 155)          # fixed, spread over the split, both classes among them

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], MODEL = process.argv[4], ROWS = JSON.parse(process.argv[5]);
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runSquareStudent, plainSquareLogits } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquare"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  // The one Vite-only specifier: the SEAL wasm's `?url` import becomes its path on disk.
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const sample = JSON.parse(readFileSync(MODEL + "/eval_sample.json", "utf8"));
const weights = JSON.parse(readFileSync(MODEL + "/weights.json", "utf8"));
const out = { sets: Object.keys(weights.ckks).sort(), plain: [], runs: {} };
for (const i of ROWS) out.plain.push(m.plainSquareLogits(sample.samples[i], weights));
for (const set of out.sets) {
  out.runs[set] = [];
  for (const i of ROWS) {
    const r = await m.runSquareStudent(sample.samples[i], weights, set);
    out.runs[set].push({ row: i, logits: r.logits, maxAbs: r.maxAbsError, agrees: r.decisionAgrees,
                         ms: r.ms, bytes: r.bytes, N: r.N });
  }
}
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def run() -> dict:
    if not (WEB / "node_modules" / "node-seal").exists():
        pytest.skip("web/node_modules is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.mjs"
        harness.write_text(HARNESS)
        proc = subprocess.run(["node", str(harness), str(WEB), str(Path(tmp) / "bundle.mjs"), str(MODEL),
                               json.dumps(list(ROWS))], capture_output=True, text=True, timeout=600,
                              env={**os.environ, "NODE_OPTIONS": "--max-old-space-size=4096"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_published_sets_are_the_two_that_passed(run) -> None:
    assert run["sets"] == ["A2", "B"], run["sets"]


def test_every_decrypted_logit_is_within_the_preregistered_limit_and_no_decision_changes(run) -> None:
    for name, rows in run["runs"].items():
        assert len(rows) == len(ROWS)
        worst = max(r["maxAbs"] for r in rows)
        assert worst <= 1e-3, f"{name}: a decrypted logit is {worst:.2e} from the plaintext one"
        assert all(r["agrees"] for r in rows), f"{name}: a decision changed under CKKS"
    assert {r["N"] for r in run["runs"]["A2"]} == {8192} and {r["N"] for r in run["runs"]["B"]} == {16384}


def test_the_module_float64_reference_is_the_python_one(run) -> None:
    sys.path.insert(0, str(REPO / "scripts" / "train"))
    from medical_square_student import plain_logits

    w = json.loads((MODEL / "weights.json").read_text())
    sample = json.loads((MODEL / "eval_sample.json").read_text())
    p = {"conv_w": np.asarray(w["conv"]["weight"], dtype=float).reshape(4, 9), "conv_b": np.asarray(w["conv"]["bias"]),
         "fc_w": np.asarray(w["dense"][0]["weight"]), "fc_b": np.asarray(w["dense"][0]["bias"])}
    for row, js in zip(ROWS, run["plain"], strict=True):
        py = plain_logits(np.asarray(sample["samples"][row], dtype=float).reshape(14, 14), p)
        assert np.max(np.abs(py - np.asarray(js))) < 1e-9, (row, py, js)


def test_the_decrypted_vector_holds_the_logit_in_every_slot_so_no_partial_sum_is_decrypted() -> None:
    """The module tiles every vector with period 1,024 so a rotate-and-add leaves the full sum in every slot; the
    period must divide N/2 for every published set, or the windows near the end would wrap onto a different phase."""
    w = json.loads((MODEL / "weights.json").read_text())
    src = (WEB / "src" / "lib" / "clientFHESquare.ts").read_text()
    assert "const PERIOD = 1024;" in src and "1 << i" in src
    for name, p in w["ckks"].items():
        assert (p["N"] // 2) % 1024 == 0, name
