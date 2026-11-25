"""The 3D nodule student's card on the medical page, and the files it ships (Round 86, batch 4).

`scripts/export_nodule3d_student.py` writes `models/medical_lidc_nodule3d_square_student/weights.json` and
`web/public/data/medical/nodule3d_square_student.json` from the record of the pre-registered run. This checks:

- the web file is what the exporter writes from the committed weights and record (`--check`), so every figure the card
  quotes is the record's and every crop is the one its SHA-256 names;
- the twelve crops are the addendum's section 4 (six of each label), and no patient or series identifier is shipped;
- only the parameter sets that met criterion 3 are offered;
- the card is mounted on the medical page, reads that file, and credits LIDC-IDRI as TCIA's collection page asks;
- the page's own module, under node-seal, runs the first crop of each label under every offered set within the
  pre-registered 1e-3 with the decision unchanged, and gives the shipped plaintext logits in the clear.
"""

from __future__ import annotations

import base64
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
DATA = WEB / "public" / "data" / "medical" / "nodule3d_square_student.json"
WEIGHTS = REPO / "models" / "medical_lidc_nodule3d_square_student" / "weights.json"
RECORD = REPO / "docs" / "measurements" / "medical-nodule3d-square-student.json"
CARD = WEB / "src" / "components" / "Nodule3dCard.tsx"
PAGE = WEB / "src" / "pages" / "UseCasePage.tsx"

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], DIR = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export * from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquare3d"))};`,
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
const crops = JSON.parse(readFileSync(DIR + "/crops.json", "utf8"));
const out = [];
for (const v of crops) {
  const runs = {};
  for (const set of Object.keys(w.ckks)) runs[set] = await m.runSquare3d(v, w, set);
  out.push({ plain: m.plainSquare3dLogits(v, w), runs });
}
console.log(JSON.stringify(out));
"""


def _shipped() -> dict:
    return json.loads(DATA.read_text())


def _voxels(c: dict) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(c["u8_base64"]), dtype=np.uint8)
    return (raw.astype(np.float64) / 255.0 - 0.5) / 0.5


def _malignant(logits, decision: dict) -> bool:
    a, b = decision["platt"]
    return 1 / (1 + np.exp(-(a * (logits[1] - logits[0]) + b))) >= decision["threshold"]


def test_the_web_file_is_the_exporters() -> None:
    r = subprocess.run([sys.executable, "scripts/export_nodule3d_student.py", "--check"], cwd=REPO,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr


def test_the_twelve_crops_and_no_identifier() -> None:
    d = _shipped()
    labels = [c["label"] for c in d["crops"]]
    assert len(labels) == 12 and labels.count(1) == 6 and labels.count(0) == 6, labels
    assert len({c["sha256"] for c in d["crops"]}) == 12
    text = DATA.read_text()
    for marker in ("LIDC-IDRI-", "1.3.6.1.4.1.14519", '"patient"', '"series"'):
        assert marker not in text, f"the shipped file holds {marker!r}"


def test_only_the_sets_that_met_criterion_3_are_offered() -> None:
    record = json.loads(RECORD.read_text())
    w = json.loads(WEIGHTS.read_text())
    assert _shipped()["weights"] == w
    assert set(w["ckks"]) == {k for k, v in record["ckks"].items() if v["criterion_3"]}
    assert record["criteria"]["1_accuracy"] and record["criteria"]["2_no_collapse"]


def test_the_card_is_mounted_reads_the_file_and_credits_the_data() -> None:
    page, card = PAGE.read_text(), CARD.read_text()
    assert 'usecase === "medical_fl" && <Nodule3dCard />' in page
    assert "data/medical/nodule3d_square_student.json" in card
    assert "not for clinical use" in card
    d = _shipped()
    for must in ("10.7937/K9/TCIA.2015.LO9QL9SX", "10.1118/1.3528204", "10.1117/1.JMI.3.4.044504",
                 "National Cancer Institute", "Foundation for the National Institutes of Health"):
        assert must in d["attribution"], must
    assert d["data_usage_policy"] == "https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/"
    assert (d["licence"], d["licence_url"]) == ("CC-BY-3.0", "https://creativecommons.org/licenses/by/3.0/")


def test_the_pages_module_runs_a_crop_of_each_label_under_every_offered_set() -> None:
    if not ((WEB / "node_modules" / ".bin" / "esbuild").exists() and (WEB / "node_modules" / "node-seal").exists()):
        msg = "web/node_modules lacks esbuild or node-seal, so the CKKS path cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = _shipped()
    picks = [next(c for c in d["crops"] if c["label"] == 1), next(c for c in d["crops"] if c["label"] == 0)]
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "weights.json").write_text(json.dumps(d["weights"]))
        (t / "crops.json").write_text(json.dumps([_voxels(c).tolist() for c in picks]))
        (t / "harness.mjs").write_text(HARNESS)
        out = subprocess.run(["node", str(t / "harness.mjs"), str(WEB), str(t / "bundle.mjs"), str(t)],
                             check=True, capture_output=True, text=True, cwd=WEB, timeout=900)
    runs = json.loads(out.stdout.strip().splitlines()[-1])
    for c, r in zip(picks, runs, strict=True):
        np.testing.assert_allclose(r["plain"], c["plain_logits"], rtol=1e-12, atol=1e-9)
        for name, run in r["runs"].items():
            assert run["maxAbsError"] <= 1e-3, (name, run["maxAbsError"])
            assert _malignant(run["logits"], d["weights"]["decision"]) == _malignant(run["plain"], d["weights"]["decision"])
