"""Round 84: the changes the three drafters handed over as notes do what the notes said, end to end.

WHY THIS EXISTS

The drafters of this round could not edit each other's files, so the wiring between their pieces
arrived as notes and was applied by the integrator: `clientCompute.ts` (the held-out encrypted figure,
the QCNN verdicts, the OPS-SAT log line), `stages.py` and `usecases/medical_fl` (the Python QCNN's
input columns, its verdict, the secure sum's threshold) and the benchmark row's masking bounds. A
note applied by hand is exactly where a twin goes stale, so each change is checked by RUNNING it:
the Python stages directly, and the browser pipeline bundled with esbuild and run in node.

WHAT IS ASSERTED, AND WHAT EACH FAILS ON AT THE BASELINE (0bd9db9)

- Python medical QCNN: the circuit is handed the patient's RAW values in the artifact's own columns
  (`feature_names`), not the CKKS model's 30-column z-score row. Baseline: it read the z-score row's
  first 8 columns, other measurements in other units, and gave class 0 (benign, in the trainer's
  1 = malignant) for the malignant patient the use case scores; now class 1, as `infer` gives on the
  right row.
- Python and browser QCNN verdicts are three-way: medical TIES logistic regression at the shipped
  seed (0.953488 each). Baseline: Python printed "the classical model is ahead", the browser "the
  circuit is ahead here".
- The secure sum's log prints the threshold the run used. Baseline: `t={min(3, n)}`, the retired
  literal, so a configured threshold of 2 printed t=3.
- The benchmark row carries the whole-view bound. Baseline: only the slot-0 bound existed.
- The browser's telecom QCNN log describes the shipped circuit in the artifact's words. Baseline: it
  called the SU(4) circuit "two-qubit blocks on disjoint pairs".
- A bundled sample without the space the QCNN declares is refused, not fed. Baseline: the `?? feats`
  fallback handed the raw-input model z-scores and reported its class.
- Finance, medical and telecom encrypt held-out rows through the pipeline, every row, and no metric
  is overwritten with a different value. Baseline: no held-out figure ran on any of the three.
- The satellite verdict names its model and the segment's label (OPS-SAT-AD). Baseline: neither.

The browser runs use node-seal's real CKKS. The garbled-circuit argmax is a gitignored wasm build, so
a plaintext argmax stands in for it here; it has its own tests (the crate's `cargo test` and
`check:qcnn-metrics`), and nothing below asserts anything about it.
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


# ---------------------------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------------------------

def _run_private_inference(config: dict):
    from qsh.orchestration.crypto_agility import CryptoRegistry
    from qsh.orchestration.events import EventBus
    from qsh.orchestration.stages import PrivateInferenceStage
    from qsh.usecases.medical_fl import build

    _, ctx = build(config)
    bus = EventBus()
    PrivateInferenceStage().run(ctx, bus, CryptoRegistry())
    logs = [e.message for e in bus.history if e.type.value == "log"]
    return ctx, bus.metrics(), logs


def test_the_python_medical_qcnn_reads_its_own_columns_in_raw_units() -> None:
    from sklearn.datasets import load_breast_cancer

    from qsh.qhe.qcnn_trained import infer, load_weights

    w = load_weights("medical_fl")
    assert w["input_space"] == "raw" and len(w["feature_names"]) == 8
    ctx, metrics, _ = _run_private_inference({"inference": "qcnn"})
    d = load_breast_cancer()
    patient = int(np.flatnonzero(d.target == 0)[0])          # the use case's malignant patient
    cols = [list(d.feature_names).index(n) for n in w["feature_names"]]
    want = d.data[patient, cols]
    got = ctx.data.get("qcnn_features")
    assert got is not None, "the use case hands the QCNN no row of its own; it falls back to `features`"
    np.testing.assert_allclose(got, want, rtol=0, atol=0)
    z, cls, _ = infer(want, w)
    assert ctx.artifacts["qhe_prediction"] == cls and metrics["qhe_expectation_z0"] == round(z, 4), (
        "the stage's class is not the circuit's on the patient's own columns")
    # The z-score row's first 8 columns are other measurements; on them the circuit says otherwise,
    # which is what made the old path wrong rather than merely different.
    assert infer(ctx.data["features"], w)[1] != cls


def test_the_python_qcnn_verdict_is_computed_and_says_level_on_a_tie() -> None:
    from qsh.qhe.qcnn_trained import load_weights

    w = load_weights("medical_fl")
    assert w["test_accuracy"] == w["classical_lr_accuracy"], "the shipped medical artifact no longer ties"
    _, _, logs = _run_private_inference({"inference": "qcnn"})
    line = next(m for m in logs if m.startswith("QCNN inference (trained"))
    assert "level on this split" in line and "classical model is ahead" not in line, line


def test_the_secure_sum_log_prints_the_threshold_the_run_used() -> None:
    from qsh.orchestration.crypto_agility import CryptoRegistry
    from qsh.orchestration.events import EventBus
    from qsh.orchestration.stages import Context, QSMPCComputeStage

    ctx = Context(usecase="demo", config={"qsmpc_op": "sum", "qsmpc_threshold": 2, "seed": 0})
    ctx.data.update(secrets=[3, 4, 5], session_key=b"\x01" * 32)
    bus = EventBus()
    QSMPCComputeStage().run(ctx, bus, CryptoRegistry())
    line = next(e.message for e in bus.history if e.type.value == "log" and "QSMS" in e.message)
    assert "t=2" in line and "t=3" not in line, line
    # Over the field the stage derived from the party count (n <= d <= 2n), as it records.
    assert ctx.artifacts["secure_sum"] == 12 % ctx.artifacts["qsms_prime"]
    assert isinstance(ctx.artifacts.get("qsms_field_within_paper_bound"), bool)


def test_the_benchmark_row_reports_the_bound_on_the_whole_view() -> None:
    pytest.importorskip("tenseal")
    pytest.importorskip("mpyc")
    from qsh.benchmark.classical_mpc_fhe import run_hybrid

    rng = np.random.default_rng(84)
    layers = [(rng.normal(size=(4, 8)) * 0.3, rng.normal(size=4) * 0.1),
              (rng.normal(size=(2, 4)) * 0.3, rng.normal(size=2) * 0.1)]
    row = run_hybrid(layers, rng.normal(size=(2, 8)), np.array([0, 1]), n_samples=2)
    ex = row.extra
    assert 0 < ex["mask_statistical_distance_max"] < ex["view_statistical_distance_max"] < 1e-6
    assert ex["masked_slots"] == "every slot, real and imaginary parts"
    assert ex["flood_margin_bits"] >= 40


# ---------------------------------------------------------------------------------------------
# Browser, run in node
# ---------------------------------------------------------------------------------------------

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const RUNS = JSON.parse(readFileSync(process.argv[3], "utf8"));
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-int-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    // node-seal's `?url` wasm import becomes its path on disk, so the CKKS tier is the real one.
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    // The garbled-circuit argmax is a gitignored wasm build; a plaintext argmax stands in for it.
    b.onResolve({ filter: /(^|\/)clientMPC$/ }, () => ({ path: "fakempc", namespace: "fakempc" }));
    b.onLoad({ filter: /.*/, namespace: "fakempc" }, () => ({ loader: "js", contents: `
      export async function warmMPC() {}
      export async function secureArgmax(l) {
        let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i;
        return { out: [b], compares: Math.max(0, l.length - 1), commBytes: 0, mpcMs: 0, messages: 0, clearOut: [b] };
      }` }));
    b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
// Every fetch answers from web/public; anything remote (the Hub) is a 404, as offline. A run may
// drop keys from one payload to show what the page does without them.
let drop = null;
globalThis.fetch = async (url) => {
  const p = String(url);
  const miss = { ok: false, status: 404, headers: { get: () => null },
                 json: async () => { throw new Error("404"); }, text: async () => "" };
  if (/^[a-z]+:/i.test(p)) return miss;
  let text;
  try { text = readFileSync(join(WEB, "public", p.replace(/^\/+/, "")), "utf8"); } catch { return miss; }
  if (drop && drop.path === "/" + p.replace(/^\/+/, "")) {
    const j = JSON.parse(text);
    for (const k of drop.keys) delete j[k];
    for (const k of drop.heldout ?? []) if (j.heldout) delete j.heldout[k];
    text = JSON.stringify(j);
  }
  return { ok: true, status: 200, headers: { get: () => "application/json" },
           json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const result = {};
for (const run of RUNS) {
  drop = run.drop ?? null;
  const logs = [], metrics = {}, seen = [];
  await new Promise((done) => {
    let ctl = null, finished = false;
    const finish = () => { if (!finished) { finished = true; clearTimeout(timer); ctl?.stop?.(); done(); } };
    const timer = setTimeout(finish, 240_000);
    ctl = m.runClientPipeline(run.usecase, { seed: 0, ...run.config }, {
      onEvent: (e) => {
        if (finished) return;
        if (e.stage && !seen.includes(e.stage)) {
          if (seen.length && seen[seen.length - 1] === run.stopAfter) { finish(); return; }
          seen.push(e.stage);
        }
        if (e.type === "log") logs.push([e.stage, String(e.message ?? "")]);
        if (e.type === "metric") (metrics[e.stage] ??= {})[e.data.name] = e.data.value;
      },
      onEnd: finish,
    });
  });
  result[run.name] = { stages: seen, logs, metrics };
}
console.log(JSON.stringify(result));
process.exit(0);
"""

#: One node process per group: `realData` caches each payload for the tab's lifetime, so a run that
#: drops keys from a payload must not share a process with one that reads it whole.
GROUPS = {
    "whole": [
        {"name": "medical_qcnn", "usecase": "medical_fl", "config": {"inference": "qcnn"},
         "stopAfter": "private_inference"},
        {"name": "finance", "usecase": "finance_aml", "config": {}, "stopAfter": "private_inference"},
        {"name": "telecom_qcnn", "usecase": "telecom_ids", "config": {"inference": "qcnn"},
         "stopAfter": "qml_ids"},
        {"name": "satellite", "usecase": "satellite_comms", "config": {}, "stopAfter": "qsdc_telemetry"},
        {"name": "satellite_small_pool", "usecase": "satellite_comms", "config": {"sat_stike_pool_mbit": 1},
         "stopAfter": "qsdc_telemetry"},
    ],
    "no_raw_rows": [
        {"name": "medical_no_raw", "usecase": "medical_fl", "config": {"inference": "qcnn"},
         "stopAfter": "private_inference",
         "drop": {"path": "/data/medical/breast_cancer_sample.json", "keys": ["samples_by_space"],
                  "heldout": ["samples_by_space"]}},
    ],
}


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the browser pipeline cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    _require_node_or_skip()
    tmp = tmp_path_factory.mktemp("r84int")
    script = tmp / "harness.mjs"
    script.write_text(HARNESS)
    out: dict = {}
    for group, runs in GROUPS.items():
        spec = tmp / f"{group}.json"
        spec.write_text(json.dumps(runs))
        proc = subprocess.run(["node", str(script), str(WEB), str(spec)], cwd=WEB,
                              capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, f"the browser pipeline did not run ({group}):\n{proc.stderr[-3000:]}"
        out.update(json.loads(proc.stdout.strip().splitlines()[-1]))
    return out


def _stage(run: dict, stage: str) -> tuple[dict, list[str]]:
    assert stage in run["stages"], f"the run never reached {stage}: {run['stages']}"
    return run["metrics"].get(stage, {}), [m for s, m in run["logs"] if s == stage]


def test_the_browser_medical_qcnn_tie_reads_as_level(browser) -> None:
    m, logs = _stage(browser["medical_qcnn"], "private_inference")
    assert m.get("primary_inference_backend") == "qcnn-trained"
    line = next(x for x in logs if x.startswith("inference backend = QCNN"))
    assert "level on this split" in line and "the circuit is ahead here" not in line, line


def test_the_browser_telecom_qcnn_names_its_own_circuit(browser) -> None:
    w = json.loads((WEB / "public" / "data" / "telecom" / "qcnn_weights.json").read_text())
    _, logs = _stage(browser["telecom_qcnn"], "qml_ids")
    line = next(x for x in logs if x.startswith("QCNN (trained"))
    assert "disjoint pairs" not in line and w["circuit"]["simulability"] in line, line
    verdict = next(x for x in logs if x.startswith("and it "))
    assert verdict.startswith("and it LOSES") == (w["test_macro_f1"] < w["classical_lr_macro_f1"]), verdict


def test_a_sample_without_the_declared_space_is_refused_not_fed(browser) -> None:
    m, logs = _stage(browser["medical_no_raw"], "private_inference")
    assert m.get("primary_inference_backend") == "ckks", (
        "the QCNN ran on a sample that ships no rows in the space it declares")
    assert any("ships no rows in the space the trained QCNN declares (raw)" in x for x in logs), logs


@pytest.mark.parametrize("name, stage, model, rows, ciphertexts", [
    ("finance", "private_inference", "finance_aml.student", 8192, 64),
    ("medical_qcnn", "private_inference", None, 171, 1),
    ("telecom_qcnn", "qml_ids", "telecom_ids.student", 600, 2),
])
def test_the_pipeline_encrypts_every_held_out_row(browser, name, stage, model, rows, ciphertexts) -> None:
    m, logs = _stage(browser[name], stage)
    assert m.get("encrypted_rows_held_out") is True, f"{name}: no held-out figure ran"
    assert m["encrypted_rows_scored"] == m["encrypted_rows_available"] == m["encrypted_agreement_rows"] == rows
    assert m["encrypted_ciphertexts"] == ciphertexts and m["encrypted_rows_dropped_by_cap"] == 0
    if model:
        assert m["encrypted_model"] == model
    assert m["encrypted_top1_mismatches_batch"] == 0
    # The rate is over the held-out rows: the one-row figure did not overwrite it, and nothing else
    # wrote a key twice with two values.
    assert "metric_overwritten" not in m, m.get("metric_overwritten")
    assert any("HELD-OUT rows" in x for x in logs)


def test_the_telecom_page_still_reports_its_per_rotation_cost(browser) -> None:
    m, _ = _stage(browser["telecom_qcnn"], "qml_ids")
    assert m["fhe_ms_per_rotation"] == m["fhe_batch_ms_per_rotation"] > 0


def test_the_browser_log_says_what_became_of_the_telemetry(browser) -> None:
    """Round 86: when one-way quasi-QSDC carries the telemetry the verdict above it must not say it fell back."""
    _, logs = _stage(browser["satellite"], "qsdc_telemetry")
    i = next(n for n, x in enumerate(logs) if x.startswith("QSDC (simulated) NOT AVAILABLE"))
    assert "carries the telemetry instead" in logs[i] and "falls back" not in logs[i]
    assert logs[i + 1].startswith("Quasi-QSDC (simulated, one-way STIKE"), logs[i + 1]
    _, small = _stage(browser["satellite_small_pool"], "qsdc_telemetry")
    j = next(n for n, x in enumerate(small) if x.startswith("QSDC (simulated) NOT AVAILABLE"))
    assert "falls back to the PQC session key" in small[j] and "carries the telemetry instead" not in small[j]
    assert small[j + 1].startswith("Quasi-QSDC (STIKE) not attempted"), small[j + 1]


def test_the_satellite_verdict_names_its_model_and_the_segments_label(browser) -> None:
    seg = json.loads((WEB / "public" / "data" / "satellite" / "opssat_segments.json").read_text())
    _, logs = _stage(browser["satellite"], "qsdc_telemetry")
    line = next(x for x in logs if x.startswith("telemetry anomaly detection"))
    assert "OPS-SAT-AD" in line and "NASA" not in line
    assert "the segment is labelled" in line and f"Model: {seg['real_model']}" in line, line
