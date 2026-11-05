"""The private-inference figure encrypts a PUBLISHED model on rows it was never fitted on, every row.

WHY THIS EXISTS (Round 84, checklist PI7-PI9)

The agreement figure called finance's and medical's 400 rows "held-out" while the plotted model had
been fitted on exactly those rows, and on finance the model was a PaySim surrogate rather than the
AMLworld student the page names; telecom scored its student on rows a JavaScript mirror of the
generator drew in the tab instead of the split the student was evaluated on. The browser path is now
`heldOutCorpus` (web/src/lib/realData.ts) into `encryptedHeldOutAgreement` (web/src/lib/clientFHE.ts).

WHAT THIS ASSERTS, BY RUNNING THE BROWSER'S OWN CODE

The two modules are bundled with esbuild and run in node against the REAL node-seal CKKS the page
uses, with `fetch` answering from web/public. Per use case: every held-out row is scored (8,192 on
finance, which takes 64 ciphertexts; 171 on medical; 600 on telecom, two ciphertexts), the figure says
`heldOut` with a licence link, the plaintext reference reproduces the decisions the payload's builder
computed in Python, and no row flips. Finance's PaySim fitting rows no longer reach the batch, and
medical's corpus is the held-out side of its split.

On the baseline this fails at the first step: neither function exists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
PUBLIC = WEB / "public" / "data"

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { heldOutCorpus, realInputCorpus } from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};
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
const out = {};
for (const uc of ["finance_aml", "medical_fl", "telecom_ids", "satellite_comms"]) {
  const held = await m.heldOutCorpus(uc);
  if (!held) { out[uc] = null; continue; }
  const r = await m.encryptedHeldOutAgreement(held);
  const p = r.plot;
  out[uc] = {
    corpusRows: held.rows.length, modelKey: held.modelKey,
    rows: p.rows, rowsOffered: p.rowsOffered, ciphertexts: p.ciphertexts,
    rowsPerCiphertext: p.rowsPerCiphertext, msPerCiphertext: p.msPerCiphertext,
    heldOut: p.heldOut, fitNote: p.fitNote, licence: p.licence, scoreKind: p.scoreKind,
    planePoints: p.points.filter((q) => typeof q.x === "number" && typeof q.y === "number").length,
    plane: p.plane, mismatches: r.mismatches, maxAbsError: r.batch.maxAbsError,
    medianAbsError: r.medianAbsError, smallestMargin: r.smallestMargin,
    flipImpossible: r.flipImpossible, plaintextAccuracy: r.plaintextAccuracy,
    plainFlagged: r.batch.plainLogits.filter((row) => row.indexOf(Math.max(...row)) === 1).length,
    encFlagged: p.points.filter((q) => q.encCls === 1).length,
    labelsAligned: p.points.every((q, i) => q.label === held.labels[i]),
  };
}
out.financeCorpus = await m.realInputCorpus("finance_aml");
out.medicalCorpus = await m.realInputCorpus("medical_fl");
console.log(JSON.stringify(out));
"""


def _require_node_or_skip() -> None:
    """Skip without the toolchain, unless the environment says a skip is a failure (CI does)."""
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = (f"{' and '.join(missing)} absent, so the encrypted held-out path cannot run. Run `npm ci` "
           "in web/. This is a SKIP, not a pass.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    _require_node_or_skip()
    tmp = tmp_path_factory.mktemp("heldout")
    script = tmp / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(tmp / "bundle.mjs")], cwd=WEB,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        "the browser's held-out path did not run:\n" + proc.stderr[-3000:]
        + "\n(`heldOutCorpus` and `encryptedHeldOutAgreement` are the Round 84 entry points)")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _payload(rel: str) -> dict:
    return json.loads((PUBLIC / rel).read_text())


@pytest.mark.parametrize("usecase, rows, ciphertexts", [
    ("finance_aml", 8192, 64), ("medical_fl", 171, 1), ("telecom_ids", 600, 2), ("satellite_comms", 529, 34)])
def test_every_held_out_row_is_scored(run: dict, usecase: str, rows: int, ciphertexts: int) -> None:
    r = run[usecase]
    assert r is not None, f"{usecase}: heldOutCorpus returned nothing"
    assert r["corpusRows"] == r["rows"] == r["rowsOffered"] == rows, (
        f"{usecase}: {r['rows']} of {r['rowsOffered']} rows scored, expected all {rows}. A denominator "
        "smaller than the split it names is what the cap disclosure exists for; the held-out path "
        "pays for every row instead.")
    assert r["ciphertexts"] == ciphertexts, (
        f"{usecase}: {r['ciphertexts']} ciphertexts for {rows} rows; at "
        f"{r['rowsPerCiphertext']} rows each it takes {ciphertexts}")
    assert r["msPerCiphertext"] and r["msPerCiphertext"] > 0, "the per-ciphertext cost is not reported"


@pytest.mark.parametrize("usecase", ["finance_aml", "medical_fl", "telecom_ids", "satellite_comms"])
def test_the_figure_says_held_out_and_names_its_licence(run: dict, usecase: str) -> None:
    r = run[usecase]
    assert r["heldOut"] is True and r["fitNote"], f"{usecase}: the plot does not say what the model was fitted on"
    lic = r["licence"]
    assert lic["name"] and lic["modifications"], f"{usecase}: licence name or change note missing: {lic}"
    if usecase != "telecom_ids":                     # the generated corpus licenses nothing to link
        assert lic["url"] and lic["url"].startswith("https://"), f"{usecase}: no licence link: {lic}"
    assert r["planePoints"] == r["rows"] and r["plane"], (
        f"{usecase}: {r['planePoints']} of {r['rows']} rows carry overlay-plane coordinates")
    assert r["labelsAligned"], f"{usecase}: the plotted true labels are not the corpus labels in order"


@pytest.mark.parametrize("usecase", ["finance_aml", "medical_fl", "telecom_ids", "satellite_comms"])
def test_encryption_moves_no_decision_and_says_why(run: dict, usecase: str) -> None:
    r = run[usecase]
    assert r["mismatches"] == 0, f"{usecase}: {r['mismatches']} encrypted decisions differ from plaintext"
    assert 0 < r["maxAbsError"] < 1e-6, f"{usecase}: CKKS error {r['maxAbsError']} is out of its range"
    assert r["medianAbsError"] is not None and r["medianAbsError"] <= r["maxAbsError"]
    assert r["flipImpossible"] is True and r["smallestMargin"] > r["maxAbsError"], (
        f"{usecase}: smallest margin {r['smallestMargin']} against worst error {r['maxAbsError']}")


def test_finance_scores_the_published_student_and_reproduces_its_decisions(run: dict) -> None:
    r = run["finance_aml"]
    exp = _payload("finance/student_heldout.json")["expected"]
    assert r["modelKey"] == "finance_aml.student"
    assert r["scoreKind"] == "logit"
    # The plaintext student in the browser flags exactly the rows the Python builder's student does,
    # and the encrypted one agrees with it.
    assert r["plainFlagged"] == r["encFlagged"] == exp["plaintext_flagged"], (r, exp)


def test_satellite_scores_the_published_student_and_reproduces_its_decisions(run: dict) -> None:
    """The expansion and the scaler run in TypeScript here and in scikit-learn in the builder; one wrong term flips rows."""
    r = run["satellite_comms"]
    exp = _payload("satellite/student_heldout.json")["expected"]
    assert r["modelKey"] == "satellite_comms.opssat_student"
    assert r["scoreKind"] == "logit"
    assert r["plainFlagged"] == r["encFlagged"] == exp["plaintext_flagged"], (r, exp)
    assert abs(r["plaintextAccuracy"] - exp["plaintext_accuracy"]) < 1e-6


def test_medical_and_telecom_reproduce_the_builders_plaintext_accuracy(run: dict) -> None:
    med = _payload("medical/breast_cancer_sample.json")["heldout"]["plaintext_accuracy"]
    tel = _payload("telecom/student_heldout.json")["expected"]["plaintext_accuracy"]
    assert abs(run["medical_fl"]["plaintextAccuracy"] - med) < 1e-6
    assert abs(run["telecom_ids"]["plaintextAccuracy"] - tel) < 1e-6
    assert run["telecom_ids"]["modelKey"] == "telecom_ids.student"


def test_the_old_corpus_path_no_longer_hands_over_fitting_rows(run: dict) -> None:
    """`realInputCorpus` feeds the existing batch path; it must not carry a model's fitting rows."""
    assert run["financeCorpus"] is None, (
        "realInputCorpus('finance_aml') still returns the PaySim rows the surrogate was fitted on")
    med = run["medicalCorpus"]
    ds = _payload("medical/breast_cancer_sample.json")
    fit = {tuple(r[:8]) for r in ds["samples"]}
    assert med and len(med) == ds["heldout"]["n"], "the medical corpus is not the held-out split"
    assert not fit & {tuple(r) for r in med}, "a fitting row reached the medical corpus"
