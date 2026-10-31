"""The browser's plaintext anomaly step and the full deployment's CKKS stage score one model on one segment.

WHY THIS EXISTS (Round 84)

Until this round the two satellite anomaly checks shared a name and nothing else: the browser scored
OPS-SAT-AD segments with the demo regression in `web/public/data/satellite/opssat_segments.json`,
and `TelemetryAnomalyStage` scored NASA SMAP/MSL windows with a regression it fitted itself. The
Python stage now reads the same payload (tests/test_r84_satpy_opssat_ckks.py). This file checks the
claim the way a reader would, by running the browser's own code: `realSatelliteAnomaly` and
`linearShap` are bundled with esbuild and run in node with `fetch` answering from web/public, and
each held-out position is compared with the Python side.

WHAT IS ASSERTED

  - The browser's model, reference point, feature names and notes are the Python loader's, and every
    one of its 159 segments is the Python loader's segment at the same position, with the same label.
  - On all 159, the Python plaintext margin gives the browser's score to its four decimals and the
    browser's verdict.
  - On a spread of positions (a true positive, a true negative, a false positive, a false negative,
    the smallest margin, and the seed-1 draw), the Python stage under CKKS reports the browser's
    verdict and score, and the SHAP bars, reference value and margin the browser computes.

At the base commit (aa293ab) the Python side fails first: there is no `load_opssat_segments`, and
the stage scores a 48-wide SMAP/MSL window.
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

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2], RESULT = process.argv[3];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-satpy-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { linearShap } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};
export { realSatelliteAnomaly, SATELLITE_SEGMENTS_PATH } from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
globalThis.fetch = async (url) => {
  const p = String(url);
  const miss = { ok: false, status: 404, json: async () => { throw new Error("404"); } };
  if (/^[a-z]+:/i.test(p)) return miss;
  let text;
  try { text = readFileSync(join(WEB, "public", p.replace(/^\/+/, "")), "utf8"); } catch { return miss; }
  return { ok: true, status: 200, json: async () => JSON.parse(text) };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const n = JSON.parse(readFileSync(join(WEB, "public", m.SATELLITE_SEGMENTS_PATH.replace(/^\/+/, "")), "utf8")).segments.length;
const rows = [];
let model = null;
for (let k = 0; k < n; k++) {
  // The step draws k = floor(u() * n) % n from the run's uniform; this u lands on position k.
  const a = await m.realSatelliteAnomaly(() => (k + 0.5) / n);
  if (!a) { writeFileSync(RESULT, JSON.stringify({ error: `realSatelliteAnomaly returned null at ${k}` })); process.exit(0); }
  const shap = m.linearShap(a.model.W, a.model.b, a.window, a.featureNames, 5, a.baseline);
  rows.push({ score: a.anomalyScore, isAnomaly: a.isAnomaly, trueLabel: a.trueLabel, window: a.window,
              top: shap.top_features, baseValue: shap.base_value, margin: shap.margin });
  model ??= { W: a.model.W, b: a.model.b, baseline: a.baseline, names: a.featureNames,
              provenance: a.provenance, modelNote: a.modelNote, baselineNote: a.baselineNote,
              licence: a.licence, licenceUrl: a.licenceUrl };
}
// To a file, because the exit below ends the run whatever handles the bundled modules leave open,
// and an exit can cut a piped stdout at 64 KB.
writeFileSync(RESULT, JSON.stringify({ model, rows }));
process.exit(0);
"""


def _require_node_or_skip() -> None:
    """Skip without the toolchain, unless the environment says a skip is a failure (CI does)."""
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = (f"{' and '.join(missing)} absent, so the browser's satellite anomaly step cannot run. Run "
           "`npm ci` in web/. This is a SKIP, not a pass.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    _require_node_or_skip()
    tmp = tmp_path_factory.mktemp("r84satpy")
    script, result = tmp / "harness.mjs", tmp / "result.json"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(result)], cwd=WEB, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0 and result.exists(), (
        f"the browser's anomaly step did not run:\n{proc.stderr[-3000:]}")
    out = json.loads(result.read_text())
    assert "error" not in out, out
    return out


@pytest.fixture(scope="module")
def python_side():
    from qsh.data.loaders import load_opssat_segments

    return load_opssat_segments()


def _margins(p) -> np.ndarray:
    w_eff, b_eff = p.weight[1] - p.weight[0], float(p.bias[1] - p.bias[0])
    return p.segments @ w_eff + b_eff


def _spread(p) -> dict[str, int]:
    """Positions covering each outcome of the demo regression, its closest call and the default draw."""
    margins = _margins(p)
    pred, y = (margins > 0).astype(int), p.labels
    first = lambda mask: int(np.flatnonzero(mask)[0])  # noqa: E731
    n = len(margins)
    return {"true positive": first((pred == 1) & (y == 1)), "true negative": first((pred == 0) & (y == 0)),
            "false positive": first((pred == 1) & (y == 0)), "false negative": first((pred == 0) & (y == 1)),
            "smallest margin": int(np.argmin(np.abs(margins))),
            "seed 1": int(np.random.default_rng(1).random() * n) % n}


def test_the_browser_scores_the_python_loaders_model_and_rows(browser: dict, python_side) -> None:
    p, mdl = python_side, browser["model"]
    np.testing.assert_array_equal(np.asarray(mdl["W"], dtype=float), p.weight)
    np.testing.assert_array_equal(np.asarray(mdl["b"], dtype=float), p.bias)
    np.testing.assert_array_equal(np.asarray(mdl["baseline"], dtype=float), p.baseline)
    assert tuple(mdl["names"]) == p.names
    assert mdl["modelNote"] == p.model_note and mdl["baselineNote"] == p.baseline_note
    assert mdl["provenance"] == p.provenance and mdl["licence"] == p.licence
    rows = browser["rows"]
    assert len(rows) == len(p.segments) == 159
    for k, r in enumerate(rows):
        np.testing.assert_array_equal(np.asarray(r["window"], dtype=float), p.segments[k], err_msg=f"position {k}")
        assert r["trueLabel"] == int(p.labels[k]), k


def test_every_segment_gets_the_browsers_score_and_verdict_from_the_python_margin(browser: dict,
                                                                                    python_side) -> None:
    margins = _margins(python_side)
    for k, r in enumerate(browser["rows"]):
        score = 0.5 * (1.0 + np.tanh(0.5 * margins[k]))
        assert abs(round(float(score), 4) - r["score"]) <= 1.0001e-4, (k, score, r["score"])
        assert r["isAnomaly"] is bool(margins[k] > 0), k


def test_the_ckks_stage_agrees_with_the_browser_on_a_spread_of_segments(browser: dict, python_side) -> None:
    from qsh.orchestration.crypto_agility import default_registry
    from qsh.orchestration.events import EventBus, EventType
    from qsh.usecases import satellite_comms

    spread = _spread(python_side)
    assert len(set(spread.values())) >= 5, spread
    for what, k in spread.items():
        _, ctx = satellite_comms.build({"seed": 1, "telemetry_segment": k})
        bus = EventBus()
        satellite_comms.TelemetryAnomalyStage().run(ctx, bus, default_registry())
        m = {e.data["name"]: e.data["value"] for e in bus.history if e.type == EventType.METRIC}
        r = browser["rows"][k]
        assert m["telemetry_anomaly_detected"] == int(r["isAnomaly"]), (what, k)
        assert abs(m["telemetry_anomaly_score"] - r["score"]) <= 1.0001e-4, (what, k, m, r["score"])
        assert m["telemetry_anomaly_label"] == r["trueLabel"] and m["telemetry_anomaly_ckks_matches_plaintext"] == 1
        exp = ctx.artifacts["explanation"]
        assert [t["feature"] for t in exp["top_features"]] == [t["feature"] for t in r["top"][:3]], (what, k)
        for t, u in zip(exp["top_features"], r["top"][:3], strict=True):
            assert abs(t["contribution"] - u["contribution"]) <= 2e-6, (what, k, t, u)
        assert abs(exp["base_value"] - r["baseValue"]) <= 2e-6 and abs(exp["margin"] - r["margin"]) <= 2e-6
