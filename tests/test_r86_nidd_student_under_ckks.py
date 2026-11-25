"""The 5G-NIDD detector's pre-registered CKKS student, run by the browser's own code against the real node-seal (Round 86, batch 6).

The owner chose an encrypted path for the telecom page's real-data detector; its pre-registration
(references/telecom-nidd-ckks-student-preregistration.md) asks, as criterion 2, that under CKKS every decision on a row
whose plaintext top-two margin exceeds twice the largest deviation equals the plaintext one, and that the deviation is
below 1e-3, on the 2,048 held-out flows the page ships. `niddStudentHeldOut` (web/src/lib/realData.ts) into
`encryptedHeldOutAgreement` (web/src/lib/clientFHE.ts) is the path the page's `scoreNiddStudent` takes; this runs it in
node, as tests/test_r84_c3_encrypted_heldout_in_node.py runs the other held-out figures.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
PUBLIC = WEB / "public" / "data" / "telecom_nidd"

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { niddStudentHeldOut } from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};
export { encryptedHeldOutAgreement } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHE"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  // The one Vite-only specifier: the SEAL wasm's `?url` import becomes its path on disk.
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
globalThis.fetch = async (url) => {
  const p = join(WEB, "public", String(url).replace(/^\/+/, ""));
  try { const t = readFileSync(p, "utf8"); return { ok: true, status: 200, json: async () => JSON.parse(t) }; }
  catch { return { ok: false, status: 404, json: async () => { throw new Error("404"); } }; }
};
const m = await import(`file://${OUT}`);
const held = await m.niddStudentHeldOut();
if (!held) { console.log(JSON.stringify(null)); process.exit(0); }
const r = await m.encryptedHeldOutAgreement(held);
const p = r.plot;
let decidedFlips = 0;
p.points.forEach((q) => { if (!q.agree && Math.abs(q.plain) > 2 * r.batch.maxAbsError) decidedFlips++; });
console.log(JSON.stringify({ rows: p.rows, ciphertexts: p.ciphertexts, maxAbsError: r.batch.maxAbsError, mismatches: r.mismatches,
  decidedFlips, plaintextAccuracy: r.plaintextAccuracy, encryptedAccuracy: r.encryptedAccuracy, modelKey: held.modelKey,
  licence: p.licence, heldOut: p.heldOut, classes: held.classNames.length }));
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the encrypted path cannot run. Run `npm ci` in web/. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    _require_node_or_skip()
    tmp = tmp_path_factory.mktemp("nidd")
    script = tmp / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(tmp / "bundle.mjs")], cwd=WEB,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, "the 5G-NIDD encrypted path did not run:\n" + proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out is not None, "niddStudentHeldOut returned nothing"
    return out


def test_every_shipped_flow_is_scored_under_ckks(run: dict) -> None:
    assert run["rows"] == 2048 and run["ciphertexts"] == 32, run      # 36 inputs: 64-slot blocks, 64 rows a ciphertext
    assert run["modelKey"] == "telecom_ids.nidd_student" and run["classes"] == 9
    assert run["heldOut"] is True and run["licence"]["name"] == "CC-BY-4.0"


def test_criterion_2_holds(run: dict) -> None:
    assert run["maxAbsError"] < 1e-3, run["maxAbsError"]
    assert run["decidedFlips"] == 0, run


def test_the_plaintext_reference_is_the_site_files_own() -> None:
    w = json.loads((PUBLIC / "student_weights.json").read_text())
    h = json.loads((PUBLIC / "student_heldout.json").read_text())
    z = np.asarray(h["rows"], dtype=float)
    pred = (z @ np.asarray(w["W"], dtype=float).T + np.asarray(w["b"], dtype=float)).argmax(1)
    acc = float(np.mean(pred == np.asarray(h["labels"])))
    assert 0.95 <= acc <= 1.0, acc
    assert np.all(np.abs(z) <= w["clip"] + 1e-9), "a shipped row lies outside the clip the model reads"
    rec = json.loads((REPO / "docs" / "measurements" / "telecom-nidd-student-2026-11-25.json").read_text())
    assert rec["chosen"] == w["arm"] == "direct" and rec["criterion_1_test_macro_f1_at_least_0_95"] is True
    assert w["preregistration"]["sha256"] == rec["preregistration"]["sha256"]


def test_the_browser_accuracy_is_the_plaintext_one(run: dict) -> None:
    w = json.loads((PUBLIC / "student_weights.json").read_text())
    h = json.loads((PUBLIC / "student_heldout.json").read_text())
    z = np.asarray(h["rows"], dtype=float)
    pred = (z @ np.asarray(w["W"], dtype=float).T + np.asarray(w["b"], dtype=float)).argmax(1)
    assert abs(run["plaintextAccuracy"] - float(np.mean(pred == np.asarray(h["labels"])))) < 1e-12
    assert run["encryptedAccuracy"] == run["plaintextAccuracy"]
