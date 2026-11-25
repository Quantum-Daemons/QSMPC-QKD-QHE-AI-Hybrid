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
export { headLogit, decide, filmInput, filmRgba, filmsSummary, logitAxis } from ${JSON.stringify(resolve(WEB, "src/lib/cxrSplit"))};`,
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
// The all-films plot's figures on three stated films: one above the operating point, NIH and the int8 model agreeing;
// one below it, NIH saying effusion and the int8 model not; one with neither record.
const t = head.operating_point_logit;
const films = m.filmsSummary(head, [
  { film: 0, zPlain: t + 1, zEnc: t + 1 + 2e-9, effusion: true, label: 1, recorded: 1 },
  { film: 1, zPlain: t - 1, zEnc: t - 1 - 1e-9, effusion: false, label: 1, recorded: 0 },
  { film: 2, zPlain: t - 2, zEnc: t - 2, effusion: false, label: null, recorded: null },
]);
// The plot's axis: on the deployed site's 16 films (2026-11-24) the logits spanned about -5.1 to 1.2, with the operating
// point at -2.285; recharts labelled -5.5, -3.5, -1.5 and 1.5. A narrow span takes a finer step.
const axes = [m.logitAxis([-5.1, 1.2, -2.285]), m.logitAxis([-2.4, -2.2, -2.285])];
console.log(JSON.stringify({ runs, input: Array.from(m.filmInput(px)), rgba: Array.from(m.filmRgba(px)), films, axes }));
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
    f = out["films"]
    assert f["films"] == 3 and abs(f["worst"] - 2e-9) < 1e-12, f
    assert (f["sameDecision"], f["recorded"], f["agreeRecorded"], f["labelled"], f["agreeLabel"]) == (3, 2, 2, 2, 1), f
    wide, narrow = out["axes"]
    assert (wide["lo"], wide["hi"], wide["ticks"], wide["decimals"]) == (-6, 2, [-6, -4, -2, 0, 2], 0), wide
    for ax in (wide, narrow):
        steps = {round(b - a, 9) for a, b in zip(ax["ticks"], ax["ticks"][1:], strict=False)}
        assert len(steps) == 1 and ax["ticks"][0] == ax["lo"] and ax["ticks"][-1] == ax["hi"], ax
        assert 4 <= len(ax["ticks"]) <= 7, ax


def test_the_medical_page_mounts_the_card_and_the_card_reads_its_files() -> None:
    page = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert '{usecase === "medical_fl" && <CxrSplitInference />}' in page
    card = (WEB / "src" / "components" / "CxrSplitInference.tsx").read_text()
    assert "data/medical/cxr_head.json" in card and "openPinnedGraph(spec, h.features_graph)" in card
    assert "publishedSampleOf(spec)" in card and 'import("../lib/clientFHE")' in card


def test_the_card_says_the_page_runs_the_last_layer_encrypted() -> None:
    """The card said "No code path evaluates this model under encryption", false once the medical page runs the split."""
    import importlib.util
    import re

    from qsh.models import registry

    loader = importlib.util.spec_from_file_location("publish_hf_split", REPO / "scripts" / "publish_hf.py")
    publish_hf = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(publish_hf)
    card_key = re.search(r'const KEY = "([^"]+)"', (WEB / "src" / "components" / "CxrSplitInference.tsx").read_text()).group(1)
    assert card_key in publish_hf.SPLIT_MODELS, "the page runs a split the card does not mention"
    spec = registry.get(card_key)
    card = publish_hf.model_card(spec, registry.achieved(spec.key))
    assert "No code path" not in card and "its own last layer under CKKS" in card and "512 pooled features" in card
    assert "runs it split" not in publish_hf.model_card(spec, registry.achieved(spec.key), private=True)


def test_all_sixteen_films_go_through_the_single_film_steps() -> None:
    """The owner's choice of 2026-11-23: a film's point on the all-films plot is what its own run gives, so both runs call
    one function, and the plot draws what that function returned for every film."""
    card = (WEB / "src" / "components" / "CxrSplitInference.tsx").read_text()
    assert "try { setOutcome(await inferFilm(file, head, chosen, setBusy)); }" in card
    assert "const o = await inferFilm(file, head, i, (s) => setBusy(`Film ${i + 1} of ${file.samples.length}: ${s}`));" in card
    assert "for (let i = 0; i < file.samples.length; i++)" in card and "<CxrFilmsChart head={head} runs={all} />" in card
    chart = (WEB / "src" / "components" / "CxrFilmsChart.tsx").read_text()
    assert "filmsSummary(head, runs)" in chart and "head.operating_point_logit" in chart
    assert "<ReferenceLine x={t}" in chart and "<ReferenceLine y={t}" in chart
