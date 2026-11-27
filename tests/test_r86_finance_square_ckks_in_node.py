"""The finance square student's encrypted pass under the page's real CKKS, in node (Round 86, batch 4).

`web/src/lib/clientFHESquareTabular.ts` evaluates dense -> square -> dense under CKKS, one row a slot, with no layer
decrypted in between, under the two sets `references/finance-square-student-preregistration.md` names. This bundles it
with esbuild, as tests/test_r86_nodule3d_square_ckks_in_node.py bundles the 3D student, runs it against the REAL
node-seal, and checks it against a NumPy reference of the network written here, independently of the TypeScript:

- the TypeScript's float64 logits are the reference's (W1 z + b1, squared, through W2 and b2);
- each decrypted logit is within the pre-registered $`10^{-3}`$ of the plaintext one, and no decision changes;
- the rows span more than one block, and a block's inputs travel as one ciphertext an input, its logits as two;
- the client's scaling clips at the stated bound.

The weights are drawn here from a stated seed, at about the size a trained network's are (a batch-normalised hidden
unit near unit variance, so its square near 1, and logits of a few units), with a few rows clipped at the bound.

Since Round 86, batch 12, the module takes any number of output classes from two, and a nine-class network of the telecom
shape (30 inputs, 64 units, a unit with no weights) is held to the same reference, its decision the first largest
logit, and malformed layouts are refused before any key is made. Finance's two outputs are the case k = 2 of the same
code, and its test above is unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
N_IN, WIDTH = 114, 128
SETS = {"A2": {"N": 8192, "coeff_mod_bits": [48, 34, 34, 34, 52], "scale_bits": 34},
        "B": {"N": 16384, "coeff_mod_bits": [60, 40, 40, 40, 60], "scale_bits": 40}}

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], DIR = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export * from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquareTabular"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const w = JSON.parse(readFileSync(DIR + "/weights.json", "utf8"));
const raw = JSON.parse(readFileSync(DIR + "/raw.json", "utf8"));
const z = new Float64Array(raw.length * w.n_in);
raw.forEach((x, r) => z.set(m.scaleRow(w, x), r * w.n_in));
const out = { z: Array.from(z.subarray(0, 3 * w.n_in)), runs: {} };
for (const set of Object.keys(w.ckks)) {
  const r = await m.runSquareTabular(w, z, raw.length, set);
  out.runs[set] = { logits: Array.from(r.logits), plain: Array.from(r.plain), maxAbsError: r.maxAbsError,
                    decisionsDiffer: r.decisionsDiffer, blocks: r.blocks, per_block: r.per_block };
}
console.log(JSON.stringify(out));
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", (WEB / "node_modules" / ".bin" / "esbuild").exists()),
                                     ("web/node_modules/node-seal", (WEB / "node_modules" / "node-seal").exists()))
               if not ok]
    if not missing:
        return
    msg = f"{', '.join(missing)} absent, so the finance square student's CKKS path cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a failure.")
    pytest.skip(msg)


def _weights_and_rows(rows: int, seed: int = 20261125) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    centre, scale = rng.normal(0, 1, N_IN), rng.uniform(0.5, 2.0, N_IN)
    raw = centre + scale * rng.normal(0, 1, (rows, N_IN))
    raw[:3, :5] = centre[:5] + 40 * scale[:5]            # clipped at +10 by the client
    w1 = rng.normal(0, 1 / np.sqrt(N_IN), (WIDTH, N_IN))
    w1[:, 0] += 0.3                                      # not symmetric in the inputs
    w = {"kind": "dense-bn-square-dense", "n_in": N_IN, "width": WIDTH, "centre": centre.tolist(),
         "scale": scale.tolist(), "clip": 10.0, "w1": w1.tolist(), "b1": rng.normal(0, 0.3, WIDTH).tolist(),
         "w2": rng.normal(0, 0.05, (2, WIDTH)).tolist(), "b2": [0.4, -0.6], "classes": ["licit", "laundering"],
         "ckks": SETS}
    return w, raw


def _reference(w: dict, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip((raw - np.asarray(w["centre"])) / np.asarray(w["scale"]), -w["clip"], w["clip"])
    h = z @ np.asarray(w["w1"]).T + np.asarray(w["b1"])
    return z, (h * h) @ np.asarray(w["w2"]).T + np.asarray(w["b2"])


def test_the_encrypted_finance_pass_gives_the_networks_logits_under_both_sets() -> None:
    _require_node_or_skip()
    rows = 4096 + 37                                     # two blocks under A2, one under B
    w, raw = _weights_and_rows(rows)
    z, ref = _reference(w, raw)
    assert np.abs(ref[:, 1] - ref[:, 0]).max() > 0.5     # decisions that CKKS noise cannot flip at random
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "weights.json").write_text(json.dumps(w))
        (d / "raw.json").write_text(json.dumps(raw.tolist()))
        (d / "harness.mjs").write_text(HARNESS)
        out = subprocess.run(["node", str(d / "harness.mjs"), str(WEB), str(d / "bundle.mjs"), str(d)],
                             check=True, capture_output=True, text=True, cwd=WEB, timeout=900)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    np.testing.assert_allclose(np.asarray(r["z"]).reshape(3, N_IN), z[:3], rtol=0, atol=1e-12)
    assert np.all(np.asarray(r["z"]).reshape(3, N_IN)[:, :5] == 10.0)
    assert set(r["runs"]) == set(SETS)
    for name, run in r["runs"].items():
        plain = np.asarray(run["plain"]).reshape(-1, 2)
        dec = np.asarray(run["logits"]).reshape(-1, 2)
        np.testing.assert_allclose(plain, ref, rtol=1e-12, atol=1e-9)
        assert run["maxAbsError"] <= 1e-3, (name, run["maxAbsError"])
        assert np.abs(dec - ref).max() <= 1e-3, name
        assert run["decisionsDiffer"] == 0 and np.array_equal(dec.argmax(1), ref.argmax(1)), name
        assert run["blocks"] == {"A2": 2, "B": 1}[name]
        assert run["per_block"]["ciphertexts"] == {"input": N_IN, "output": 2}
        assert run["per_block"]["bytes"]["input"] > 0 and run["per_block"]["bytes"]["output"] > 0


def _nine_class_weights_and_rows(rows: int, seed: int = 20261128) -> tuple[dict, np.ndarray]:
    """A network of the shape the telecom square network's pre-registration names: 30 inputs, 64 units, nine classes."""
    n_in, width, k = 30, 64, 9
    rng = np.random.default_rng(seed)
    centre, scale = rng.normal(0, 1, n_in), rng.uniform(0.5, 2.0, n_in)
    raw = centre + scale * rng.normal(0, 1, (rows, n_in))
    raw[:3, :4] = centre[:4] - 40 * scale[:4]            # clipped at -20 by the client
    w1 = rng.normal(0, 1 / np.sqrt(n_in), (width, n_in))
    w1[:, 0] += 0.3                                      # not symmetric in the inputs
    w1[5] = 0.0                                          # a unit with no weights: its b1^2 folds into the offset
    w = {"kind": "dense-bn-square-dense", "n_in": n_in, "width": width, "centre": centre.tolist(),
         "scale": scale.tolist(), "clip": 20.0, "w1": w1.tolist(), "b1": rng.normal(0, 0.3, width).tolist(),
         "w2": rng.normal(0, 0.1, (k, width)).tolist(), "b2": rng.normal(0, 0.5, k).tolist(),
         "classes": [f"class {c}" for c in range(k)], "ckks": SETS}
    return w, raw


def _run_harness(w: dict, raw: np.ndarray, harness: str = HARNESS) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "weights.json").write_text(json.dumps(w))
        (d / "raw.json").write_text(json.dumps(raw.tolist()))
        (d / "harness.mjs").write_text(harness)
        out = subprocess.run(["node", str(d / "harness.mjs"), str(WEB), str(d / "bundle.mjs"), str(d)],
                             check=True, capture_output=True, text=True, cwd=WEB, timeout=900)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_nine_class_network_gives_its_logits_and_its_argmax_under_both_sets() -> None:
    _require_node_or_skip()
    rows = 4096 + 11                                     # two blocks under A2, one under B
    w, raw = _nine_class_weights_and_rows(rows)
    k, n_in = len(w["classes"]), w["n_in"]
    z, ref = _reference(w, raw)
    top2 = np.sort(ref, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    assert (margin > 2e-3).mean() > 0.99                 # nearly every decision clear of CKKS noise at the bound
    r = _run_harness(w, raw)
    np.testing.assert_allclose(np.asarray(r["z"]).reshape(3, n_in), z[:3], rtol=0, atol=1e-12)
    assert np.all(np.asarray(r["z"]).reshape(3, n_in)[:, :4] == -20.0)
    assert set(r["runs"]) == set(SETS)
    for name, run in r["runs"].items():
        plain = np.asarray(run["plain"]).reshape(-1, k)
        dec = np.asarray(run["logits"]).reshape(-1, k)
        assert plain.shape == dec.shape == (rows, k), name
        np.testing.assert_allclose(plain, ref, rtol=1e-12, atol=1e-9)
        err = np.abs(dec - ref).max()
        assert err <= 1e-3 and run["maxAbsError"] <= 1e-3, (name, err, run["maxAbsError"])
        clear = margin > 2 * err
        assert np.array_equal(dec.argmax(1)[clear], ref.argmax(1)[clear]), name
        assert run["decisionsDiffer"] == int((dec.argmax(1) != plain.argmax(1)).sum()), name
        assert run["blocks"] == {"A2": 2, "B": 1}[name]
        assert run["per_block"]["ciphertexts"] == {"input": n_in, "output": k}


REFUSE_HARNESS = HARNESS.split("const raw = ")[0] + r"""
const out = {};
const cases = {
  one_output: { ...w, w2: w.w2.slice(0, 1), b2: w.b2.slice(0, 1), classes: w.classes.slice(0, 1) },
  classes_short: { ...w, classes: w.classes.slice(1) },
  ragged_w2: { ...w, w2: w.w2.map((row, c) => (c === 2 ? row.slice(1) : row)) },
};
for (const [name, bad] of Object.entries(cases)) {
  try { await m.runSquareTabular(bad, new Float64Array(bad.n_in), 1, "A2"); out[name] = "accepted"; }
  catch (e) { out[name] = String(e.message); }
}
console.log(JSON.stringify(out));
"""


def test_a_layout_the_module_cannot_decide_is_refused_before_any_key() -> None:
    _require_node_or_skip()
    w, raw = _nine_class_weights_and_rows(1)
    r = _run_harness(w, raw, REFUSE_HARNESS)
    assert set(r) == {"one_output", "classes_short", "ragged_w2"}
    for name, message in r.items():
        assert "not a tabular square network's layout" in message, (name, message)

