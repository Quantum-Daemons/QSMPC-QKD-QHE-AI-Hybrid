"""A tab opens each published model's ONNX session once, however many runs it makes.

WHY THIS EXISTS (Round 85)

`onnxInference.ts` kept one module-level `session` and overwrote it whenever a different model was
opened, without `release()`. ONNX Runtime Web frees a session's weights and arena only on `release()`,
so every overwritten session stayed in the WebAssembly heap for the life of the tab. A run re-scores
every published model of its use case and then runs the selected one, so each run opened the models
again: the medical page's renderer grew by about 130 MB a run in headless Chrome once the CKKS leak
(tests/test_r85_ckks_frees_its_wasm_heap.py) was fixed. Sessions are now kept by model and backend
and reused.

The same change ends a read through the shared slot: a caller awaited `session.run` and then read
`session.outputNames`, and a load by another caller in between repointed `session`, so outputs could
be read by another model's names. Each caller now reads the session it opened.

WHAT IS ASSERTED, on runs of the browser pipeline in node built as the client-only demo, with a
stand-in for ONNX Runtime Web that counts the sessions created and serves the evaluation samples the
repository ships:

- Three finance runs open two sessions, one per published model, where the base commit opened a
  session for every model every run.
- Selecting the other model opens nothing (verification opened both); asking for WebGPU opens both
  again for that backend, since verification re-scores on the run's backend and a backend is part of
  what a session is; and nothing reopens after that.
- Every run still re-scores both models: the counted sessions are the ones verification used.
- Every session is created logging errors only (`logSeverityLevel: 3`): at its default ONNX Runtime
  printed two node-placement warnings per WebGPU session through `console.error`, red in a visitor's
  console, about a placement the page already states.

The stand-in's scores are constant, so verification FAILS here by design; what is under test is how
often a session is created, not what it computes. tests/test_r84_finance_browser_run.py checks the
numbers against the real runtime.

On the base commit (6bb8657) the first assertion fails: the three runs open six sessions.
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

# The finance models' Hub repos, and the directories holding the samples they publish.
SAMPLES = {"Daemons-Q/qsh-finance-aml-gbm": "models/finance_aml_gbm/eval_sample.json",
           "Daemons-Q/qsh-finance-aml-ckks-student": "models/finance_aml_student/eval_sample.json"}

HARNESS = r"""
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2], REPO = process.argv[3];
const SAMPLES = JSON.parse(process.argv[4]);
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r85-ort-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
// ONNX Runtime Web's two builds, replaced by one that counts what is created and released.
const FAKE_ORT = `
  export const env = { wasm: {} };
  export class Tensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }
  export const InferenceSession = { create: async (_buf, options) => {
    globalThis.__ort.created++;
    globalThis.__ort.severity.push(options?.logSeverityLevel ?? null);
    return { inputNames: ["input"], outputNames: ["label", "probabilities"],
      run: async (feeds) => {
        globalThis.__ort.runs++;
        const n = feeds.input.dims[0], p = new Float32Array(2 * n);
        for (let i = 0; i < n; i++) { p[2 * i] = 0.9; p[2 * i + 1] = 0.1; }
        return { label: new Tensor("int64", new BigInt64Array(n), [n]), probabilities: new Tensor("float32", p, [n, 2]) };
      },
      release: async () => { globalThis.__ort.released++; } };
  } };`;
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};
export { modelsFor } from ${JSON.stringify(resolve(WEB, "src/lib/config"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/","VITE_CLIENT_ONLY":"1"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /^onnxruntime-web\/(wasm|webgpu)$/ }, () => ({ path: "ort", namespace: "fakeort" }));
    b.onLoad({ filter: /.*/, namespace: "fakeort" }, () => ({ contents: FAKE_ORT, loader: "js" }));
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
globalThis.__ort = { created: 0, released: 0, runs: 0, severity: [] };
const ok = (text, type) => ({ ok: true, status: 200, headers: { get: () => type },
  json: async () => JSON.parse(text), text: async () => text,
  arrayBuffer: async () => new TextEncoder().encode(text).buffer });
const miss = { ok: false, status: 404, statusText: "Not Found", headers: { get: () => null },
               json: async () => { throw new Error("404"); }, text: async () => "" };
globalThis.fetch = async (url) => {
  const p = String(url);
  const hub = p.match(/^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/[^/]+\/(.+)$/);
  if (hub) {
    if (hub[2] === "eval_sample.json" && SAMPLES[hub[1]]) return ok(readFileSync(join(REPO, SAMPLES[hub[1]]), "utf8"), "application/json");
    if (hub[2].startsWith("onnx/")) return ok("onnx", "application/octet-stream");
    return miss;
  }
  if (/^[a-z]+:/i.test(p)) return miss;
  try { return ok(readFileSync(join(WEB, "public", p.replace(/^\/+/, "")), "utf8"), "application/json"); } catch { return miss; }
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
// Since Round 86 the browser checks each Hub file against a recorded digest. These bytes stand in for the real graphs, so the
// record they are checked against has to stand in too: the digest of what this harness serves.
const digest = (text) => createHash("sha256").update(text).digest("hex");
for (const spec of m.modelsFor("finance_aml")) {
  if (!spec.file_sha256) continue;
  for (const path of Object.keys(spec.file_sha256)) {
    if (path.startsWith("onnx/")) spec.file_sha256[path] = digest("onnx");
    else if (path === "eval_sample.json" && SAMPLES[spec.hf_repo_id]) spec.file_sha256[path] = digest(readFileSync(join(REPO, SAMPLES[spec.hf_repo_id]), "utf8"));
  }
}
const run = (config) => new Promise((done) => {
  const verified = [];
  const timer = setTimeout(() => done({ verified }), 600_000);
  m.runClientPipeline("finance_aml", { seed: 0, ...config }, {
    onEvent: (e) => { if (e.type === "log" && /^verification (PASS|FAIL) /.test(e.message)) verified.push(e.message.split(" ")[2].replace(/:$/, "")); },
    onEnd: () => { clearTimeout(timer); done({ verified }); },
  });
});
const steps = [];
for (const [label, config] of [["default", {}], ["default", {}], ["default", {}],
                               ["student", { published_model: "finance_aml.student" }],
                               ["student webgpu", { published_model: "finance_aml.student", onnx_backend: "webgpu" }],
                               ["default", {}], ["student webgpu", { published_model: "finance_aml.student", onnx_backend: "webgpu" }]]) {
  const r = await run(config);
  steps.push({ label, created: globalThis.__ort.created, verified: r.verified });
}
console.log(JSON.stringify({ steps, severity: globalThis.__ort.severity }));
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
def traced(tmp_path_factory) -> dict:
    _require_node_or_skip()
    script = tmp_path_factory.mktemp("r85ort") / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(REPO), json.dumps(SAMPLES)], cwd=WEB,
                          capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"the browser pipeline did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def steps(traced) -> list:
    return traced["steps"]


def test_three_runs_open_one_session_per_model(steps) -> None:
    assert [s["created"] for s in steps[:3]] == [2, 2, 2], steps[:3]


def test_a_new_backend_opens_each_model_once_more_and_nothing_reopens(steps) -> None:
    assert [s["created"] for s in steps[3:]] == [2, 4, 4, 4], steps


def test_every_run_still_rescores_both_published_models(steps) -> None:
    for s in steps:
        assert sorted(set(s["verified"])) == ["finance_aml.gbm", "finance_aml.student"], s


def test_every_session_logs_errors_only(traced) -> None:
    """ORT logs node placement as warnings through console.error, red in a visitor's console."""
    assert traced["severity"] and all(level == 3 for level in traced["severity"]), traced["severity"]
