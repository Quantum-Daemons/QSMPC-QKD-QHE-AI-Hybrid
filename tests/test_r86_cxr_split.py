"""Split inference on a chest X-ray: the last layer the page encrypts against is the published model's (Round 86).

The medical page runs the published network up to its pooled features in the tab, encrypts the 512 features, and evaluates
the network's own last layer on them under CKKS (`web/src/components/CxrSplitInference.tsx`). Held here:

1. `web/public/data/medical/cxr_head.json` is for the model's pinned revision, holds 512 weights and a bias, takes its
   threshold from the published sample, records the check `scripts/export_cxr_head.py` makes against the pinned graphs
   (the float32 logit to 1e-4, all 16 int8 decisions the same), and carries NIH's terms and credit;
2. the page's own CKKS layer (`encryptedLinear`, unchanged) evaluates this head on 512 features to the plaintext logit's
   precision, and the decision at the operating point is the same, in node;
3. the film helpers scale and lay out the 224 x 224 pixels as the graph and a canvas expect;
4. the medical page mounts the card, and the card reads the head file and the features graph it names.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
HEAD = WEB / "public" / "data" / "medical" / "cxr_head.json"
SAMPLE = REPO / "models" / "medical_cxr14_fedavg_resnet18" / "eval_sample.json"

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const [WEB, headPath] = process.argv.slice(2);
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r86-split-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { encryptedLinear } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHE"))};
export { headLogit, decide, filmInput, filmRgba } from ${JSON.stringify(resolve(WEB, "src/lib/cxrSplit"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/","VITE_CLIENT_ONLY":"1"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "seal", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
  } }],
});
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const head = JSON.parse(readFileSync(headPath, "utf8"));
// Features shaped like pooled ReLU outputs: non-negative, a few large. Drawn from a fixed generator.
let s = 12345;
const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
const runs = [];
for (let k = 0; k < 3; k++) {
  const f = Array.from({ length: 512 }, () => (rnd() < 0.3 ? 0 : 3 * rnd() * rnd()));
  const r = await m.encryptedLinear([head.weights], [head.bias], f);
  const zPlain = m.headLogit(head, f);
  runs.push({ zEnc: r.logits[0], zPlain, same: m.decide(head, r.logits[0]).effusion === m.decide(head, zPlain).effusion,
              rotations: r.rotations, ctBytes: r.ctBytes });
}
const px = [0, 128, 255];
console.log(JSON.stringify({ runs, input: Array.from(m.filmInput(px)), rgba: Array.from(m.filmRgba(px)) }));
"""


def test_the_head_file_is_the_pinned_model_s_last_layer() -> None:
    from qsh.models import registry

    head = json.loads(HEAD.read_text())
    spec = registry.get("medical_fl.cxr_fedavg")
    assert head["model"] == spec.key and head["hf_revision"] == spec.hf_revision and head["hf_repo"] == spec.hf_repo_id
    assert len(head["weights"]) == head["features"] == 512 and all(math.isfinite(w) for w in head["weights"])
    sample = json.loads(SAMPLE.read_text())
    thr = sample["expected"]["decision_threshold"]
    assert head["decision_threshold"] == thr
    assert math.isclose(head["operating_point_logit"], math.log(thr / (1 - thr)), rel_tol=1e-12)
    c = head["check"]
    assert c["films"] == 16 and c["max_abs_logit_difference_float32"] <= 1e-4 and c["decisions_agreeing_int8"] == 16
    assert head["features_graph"] == "onnx/features_int8.onnx"
    assert head["licence"].startswith("NIH ChestX-ray14 terms") and "NIH Clinical Center" in head["attribution"]


def test_the_page_s_ckks_layer_evaluates_this_head(tmp_path: Path) -> None:
    if not (WEB / "node_modules" / ".bin" / "esbuild").exists() or shutil.which("node") is None:
        msg = "node or esbuild absent, so the CKKS layer cannot run here. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    (tmp_path / "h.mjs").write_text(HARNESS)
    r = subprocess.run(["node", str(tmp_path / "h.mjs"), str(WEB), str(HEAD)], cwd=WEB, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for run in out["runs"]:
        assert abs(run["zEnc"] - run["zPlain"]) < 1e-5, run
        assert run["same"] and run["rotations"] == 9 and run["ctBytes"] and run["ctBytes"] > 0, run
    assert out["input"][0] == 0 and out["input"][2] == 1 and math.isclose(out["input"][1], 128 / 255, rel_tol=1e-7), \
        "the graph reads pixel / 255, in float32"
    assert out["rgba"] == [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]


def test_the_medical_page_mounts_the_card_and_the_card_reads_its_files() -> None:
    page = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert '{usecase === "medical_fl" && <CxrSplitInference />}' in page
    card = (WEB / "src" / "components" / "CxrSplitInference.tsx").read_text()
    assert "data/medical/cxr_head.json" in card and "openPinnedGraph(spec, head.features_graph)" in card
    assert "publishedSampleOf(spec)" in card and 'import("../lib/clientFHE")' in card
