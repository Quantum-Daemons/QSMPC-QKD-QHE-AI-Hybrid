"""Round 84: the browser's finance run encrypts and explains the four-column model, and the QCNN
still gets its own eight columns.

WHY THIS EXISTS

`realData.realInput("finance_aml")` sliced every row, every space and the SHAP reference point to
eight columns, and `FEATURE_NAMES` named the eight. With the model refitted on four columns
(`tests/test_r84_finance_paysim_model.py`), a slice to the model's own width is what keeps the CKKS
row, the SHAP bars and the reference point in one space, while the finance QCNN artifact, trained on
all eight and fed by position, must still receive its eight raw values. Both are checked by RUNNING
the browser code in node: `realInput` on its own, and the pipeline through `private_inference` with
the CKKS default and with the trained QCNN selected.

WHAT EACH FAILS ON AT THE BASE COMMIT (de1111c)

- `realInput` returns eight values, eight names and an eight-wide baseline (the payload's), so
  `values`, `names` and `baseline` are not four wide.
- The explanation's bars include `device_risk` (the base model's top driver) and the run's XAI log
  line names it.
- The trained QCNN still running on finance is a no-regression guard: it held at the base commit,
  where the payload's only block was eight wide, and must hold now that `samples` is four wide.

The garbled-circuit argmax is a gitignored wasm build, so a plaintext argmax stands in for it here,
as in tests/test_r84_integrator_wiring.py; nothing below asserts anything about it.
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
PAYLOAD = WEB / "public" / "data" / "finance" / "paysim_sample.json"
#: The model's four columns. Written out, not read from the registry, so this module still collects
#: (and its tests fail one by one) on a tree that predates `registry.PAYSIM_MODEL_FEATURES`;
#: tests/test_r84_finance_paysim_model.py pins the registry to the same list.
FOUR = ["amount", "cross_border", "night_tx", "new_payee"]

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-fin-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};
export { realInput } from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
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
globalThis.fetch = async (url) => {
  const p = String(url);
  const miss = { ok: false, status: 404, headers: { get: () => null },
                 json: async () => { throw new Error("404"); }, text: async () => "" };
  if (/^[a-z]+:/i.test(p)) return miss;
  let text;
  try { text = readFileSync(join(WEB, "public", p.replace(/^\/+/, "")), "utf8"); } catch { return miss; }
  return { ok: true, status: 200, headers: { get: () => "application/json" },
           json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
// A fixed draw: realInput takes the run's seeded uniform, so u = 0.37 picks row floor(0.37 * 400).
const one = await m.realInput("finance_aml", () => 0.37);
const result = { realInput: { values: one.values, names: one.names, baseline: one.baseline,
                              bySpace: one.bySpace, model: one.model } };
for (const run of [{ name: "ckks", config: { seed: 0, qfl: true } },
                   { name: "qcnn", config: { seed: 0, qfl: true, inference: "qcnn" } }]) {
  const logs = [], metrics = {};
  const artifacts = await new Promise((done) => {
    const timer = setTimeout(() => done(null), 600_000);
    m.runClientPipeline("finance_aml", run.config, {
      onEvent: (e) => {
        if (e.type === "log") logs.push([e.stage, String(e.message ?? "")]);
        if (e.type === "metric") (metrics[e.stage] ??= {})[e.data.name] = e.data.value;
      },
      onEnd: (s) => { clearTimeout(timer); done(s.artifacts); },
    });
  });
  result[run.name] = { logs, metrics, explanation: artifacts?.explanation ?? null,
                       primary: artifacts?.primary_inference ?? null };
}
console.log(JSON.stringify(result));
process.exit(0);
"""


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
def run(tmp_path_factory) -> dict:
    _require_node_or_skip()
    script = tmp_path_factory.mktemp("r84fin") / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB)], cwd=WEB, capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, f"the browser pipeline did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_real_input_is_the_models_four_columns_and_the_qcnns_eight(run) -> None:
    d = json.loads(PAYLOAD.read_text())
    one = run["realInput"]
    idx = int(0.37 * len(d["samples"]))
    assert one["names"] == FOUR
    assert one["values"] == d["samples"][idx] and len(one["values"]) == 4
    assert one["bySpace"]["zscore"] == one["values"]
    assert one["bySpace"]["raw"] == d["samples_by_space"]["raw"][idx]
    assert len(one["bySpace"]["raw"]) == 8 == len(d["names_by_space"]["raw"])
    assert len(one["baseline"]) == 4
    assert [len(r) for r in one["model"]["W"]] == [4, 4]


def test_the_ckks_run_encrypts_and_explains_the_four_column_model(run) -> None:
    r = run["ckks"]
    pi = r["metrics"]["private_inference"]
    assert pi["explanation_model"] == "real bundled model"
    assert str(pi["fhe_model"]).startswith("bundled real model")
    assert pi["primary_inference_backend"] == "ckks"
    exp = r["explanation"]
    assert exp and exp["method"] == "exact-linear-shap"
    assert sorted(f["feature"] for f in exp["top_features"]) == sorted(FOUR)
    assert exp["efficiency_gap"] < 1e-9
    driver = next(m for s, m in r["logs"] if m.startswith("XAI: top driver = "))
    assert driver.removeprefix("XAI: top driver = ") in FOUR
    # Nothing on the run's log names a balance-derived column.
    assert not [m for _, m in r["logs"] if "device_risk" in m or "geo_mismatch" in m]
    # The federated stage federates over the same four-column rows.
    fed = r["metrics"]["federated_training"]
    assert fed["federated_accuracy_source"] == "bundled finance_aml sample, 400 rows"


def test_the_trained_qcnn_still_runs_on_its_own_eight_raw_columns(run) -> None:
    pi = run["qcnn"]["metrics"]["private_inference"]
    assert pi["primary_inference_backend"] == "qcnn-trained", pi
    assert pi["qcnn_qubits"] == 8
    assert run["qcnn"]["primary"]["backend"] == "qcnn-trained"
