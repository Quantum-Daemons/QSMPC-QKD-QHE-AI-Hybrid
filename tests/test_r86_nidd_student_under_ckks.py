"""The telecom page's encrypted 5G-NIDD path, run by the browser's own code against the real node-seal (Round 86).

Batch 6 gave the page's real-data detector an encrypted path, a pre-registered linear CKKS student
(references/telecom-nidd-ckks-student-preregistration.md). Batch 12 replaced it there, on the owner's choice of
2026-11-28, with the square network whose pre-registered run met every criterion
(references/telecom-nidd-square-student-preregistration.md): `niddSquareHeldOut` (web/src/lib/realData.ts) into
`squareHeldOutAgreement` (web/src/lib/clientSquareAgreement.ts) is now the path `scoreNiddStudent` takes, and this runs
it in node, holding it to the pre-registration's criterion 3 on the 2,048 test flows the page ships and to the run's
recorded accuracy. The linear student's two files stay on the site: the federation scores its model on their rows, and
they are batch 6's record; the last two tests hold them to it.
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
  stdin: { contents: `export { niddSquareHeldOut } from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};
export { squareHeldOutAgreement } from ${JSON.stringify(resolve(WEB, "src/lib/clientSquareAgreement"))};`,
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
const held = await m.niddSquareHeldOut();
if (!held) { console.log(JSON.stringify(null)); process.exit(0); }
const r = await m.squareHeldOutAgreement(held, held.squareK, held.squareK.default_set);
const p = r.plot;
const gaps = r.batch.plainLogits.map((l) => { const s = [...l].sort((x, y) => y - x); return s[0] - s[1]; });
let decidedFlips = 0;
p.points.forEach((q, i) => { if (!q.agree && gaps[i] > 2 * r.batch.maxAbsError) decidedFlips++; });
console.log(JSON.stringify({ rows: p.rows, ciphertexts: p.ciphertexts, maxAbsError: r.batch.maxAbsError, mismatches: r.mismatches,
  decidedFlips, plaintextAccuracy: r.plaintextAccuracy, encryptedAccuracy: r.encryptedAccuracy, modelKey: held.modelKey,
  licence: p.licence, heldOut: p.heldOut, classes: held.classNames.length, set: held.squareK.default_set,
  scoreKind: p.scoreKind }));
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
    assert out is not None, "niddSquareHeldOut returned nothing"
    return out


def test_every_shipped_flow_is_scored_under_ckks(run: dict) -> None:
    assert run["rows"] == 2048 and run["ciphertexts"] == 30, run      # one block of 4,096 slots, one ciphertext an input
    assert run["modelKey"] == "telecom_ids.square_network" and run["classes"] == 9 and run["set"] == "B"
    assert run["heldOut"] is True and run["licence"]["name"] == "CC-BY-4.0" and run["scoreKind"] == "winning-score"


def test_criterion_3_holds(run: dict) -> None:
    assert run["maxAbsError"] <= 1e-3, run["maxAbsError"]
    assert run["decidedFlips"] == 0, run


def test_the_page_scores_the_accuracy_the_run_recorded(run: dict) -> None:
    rec = json.loads((REPO / "docs" / "measurements" / "telecom-nidd-square-student-2026-11-28.json").read_text())
    want = rec["ckks"]["sets"][run["set"]]
    assert round(run["encryptedAccuracy"], 6) == want["encrypted_accuracy"] == round(run["plaintextAccuracy"], 6), run


def test_the_linear_students_files_are_batch_6s_record() -> None:
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


def test_the_page_no_longer_runs_the_linear_student_on_this_path() -> None:
    """The replacement is complete where it was chosen: the stage reads the square network and nothing else."""
    src = (WEB / "src" / "lib" / "clientCompute.ts").read_text()
    body = src[src.index("async function scoreNiddStudent("):src.index("async function scoreNiddFederated(")]
    assert "niddSquareHeldOut()" in body and "squareHeldOutAgreement(" in body
    assert "niddStudentHeldOut" not in body and "encryptedHeldOutAgreement(" not in body
