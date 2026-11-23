"""The FHE budget views show what the ciphertexts hold, on the page's own node-seal, in node (Round 86, 2026-11-23).

WHY THIS EXISTS

On the owner's choice of 2026-11-22 the medical page shows the square student's CKKS ladder, each ciphertext's place in
the modulus chain and its scale after each step of the real encrypted pass (`clientFHESquare.ts` records it), and a BFV
noise-budget meter with a "one more multiply" button (`lib/bfvMeter.ts`). A ladder is only worth showing if it is the
arithmetic the parameters imply, and a meter only if its budget is the library's and its "wrong" is a real failure.
Both are bundled with esbuild and run against the REAL node-seal, as tests/test_r86_square_student_ckks_in_node.py runs
the pass itself.

WHAT IS ASSERTED

- For each published parameter set, the ladder starts with every data prime (all but the special one), at the scale
  the set names; each product doubles the scale; each rescale drops one prime, exactly that prime's bits, and brings the
  scale back to within 0.05 bits; the pass ends on one prime.
- The meter's budget falls with every squaring until it reaches 0; while it is above 0 every decryption is start^(2^k)
  mod t, computed here in Python; and the step that spends it decrypts wrong, which is what the panel says.
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
MODEL = REPO / "models" / "medical_breastmnist_square_student"

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], MODEL = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runSquareStudent, loadSeal } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquare"))};
export { makeBfvMeter, BFV_DEMO } from ${JSON.stringify(resolve(WEB, "src/lib/bfvMeter"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const sample = JSON.parse(readFileSync(MODEL + "/eval_sample.json", "utf8"));
const weights = JSON.parse(readFileSync(MODEL + "/weights.json", "utf8"));
const out = { ladders: {}, sets: weights.ckks, meter: [] };
for (const set of Object.keys(weights.ckks).sort()) out.ladders[set] = (await m.runSquareStudent(sample.samples[0], weights, set)).ladder;
const meter = m.makeBfvMeter(await m.loadSeal());
let step = meter.read();
out.t = String(meter.t);
out.meter.push({ ...step, decrypted: String(step.decrypted), expected: String(step.expected) });
for (let i = 0; i < 12 && step.budget > 0; i++) {
  step = meter.square();
  out.meter.push({ ...step, decrypted: String(step.decrypted), expected: String(step.expected) });
}
meter.dispose();
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def run() -> dict:
    if not (WEB / "node_modules" / "node-seal").exists():
        pytest.skip("web/node_modules is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.mjs"
        harness.write_text(HARNESS)
        proc = subprocess.run(["node", str(harness), str(WEB), str(Path(tmp) / "bundle.mjs"), str(MODEL)],
                              capture_output=True, text=True, timeout=600,
                              env={**os.environ, "NODE_OPTIONS": "--max-old-space-size=4096"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_ladder_is_the_arithmetic_the_parameters_imply(run) -> None:
    assert run["ladders"], "no parameter set ran"
    for name, ladder in run["ladders"].items():
        p = run["sets"][name]
        data = p["coeff_mod_bits"][:-1]                       # the last prime is the special one, kept for keys
        s = p["scale_bits"]
        first, last = ladder[0], ladder[-1]
        assert first["primes"] == len(data) and first["modulusBits"] == sum(data), name
        assert abs(first["scaleBits"] - s) < 1e-9, name
        for before, after in zip(ladder, ladder[1:], strict=False):
            if after["primes"] == before["primes"] - 1:       # a rescale: one prime and its bits go, the scale comes back
                assert before["modulusBits"] - after["modulusBits"] == data[before["primes"] - 1], (name, after["step"])
                assert abs(after["scaleBits"] - s) < 0.05, (name, after["step"])
            elif "products" in after["step"] or "squared" in after["step"]:
                assert after["primes"] == before["primes"] and abs(after["scaleBits"] - 2 * before["scaleBits"]) < 1e-6
        assert last["primes"] == 1, name


def test_the_meter_spends_its_budget_and_then_decrypts_wrong(run) -> None:
    steps, t = run["meter"], int(run["t"])
    assert len(steps) >= 3 and steps[-1]["budget"] == 0
    for k, s in enumerate(steps):
        assert s["squarings"] == k
        assert int(s["expected"]) == pow(3, 2 ** k, t)
    for before, after in zip(steps, steps[1:], strict=False):
        assert after["budget"] < before["budget"]
    for s in steps[:-1]:
        assert int(s["decrypted"]) == int(s["expected"]), s
    assert int(steps[-1]["decrypted"]) != int(steps[-1]["expected"])
