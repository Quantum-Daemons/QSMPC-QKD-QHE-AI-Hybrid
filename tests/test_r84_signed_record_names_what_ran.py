"""The signed run record names the backends the run executed, not every backend the tab has.

WHY THIS EXISTS (Round 84, deferred from Round 83)

The browser signs a record of each run with ML-DSA, and its `config.fhe` and `config.mpc` fields
named "node-seal-ckks" and "polytune-garbled-circuit-2pc-wasm" on every use case of the client-only
demo: on satellite, whose flow runs neither, and on telecom, which runs a CKKS check of its student
and no MPC. A tamper-evident record of a computation that did not happen invites exactly the trust
it should not get. The fields now come from the run's own measurements: CKKS when a CKKS evaluation
reported (`fhe_model`, `encrypted_rows_scored`), the garbled-circuit argmax when it was checked
against the clear one (`mpc_matches_clear`), else null.

WHAT IS ASSERTED, on runs of the browser pipeline in node built as the client-only demo:

- satellite: both null; telecom: CKKS and no MPC; finance: both.
- Each named backend is backed by the measurement that names it, in the same run.

The garbled-circuit argmax is a gitignored wasm build, so a plaintext argmax stands in for it here,
as in tests/test_r84_finance_browser_run.py; what is checked is which backends the record names.

On the base commit (de1111c) satellite and telecom fail: their records name both backends.
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

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-sig-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  // The client-only demo's build flag: the branch of the record this test is about.
  define: { "import.meta.env": '{"BASE_URL":"/","VITE_CLIENT_ONLY":"1"}', "process.env.NODE_ENV": '"production"' },
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
const result = {};
for (const usecase of ["satellite_comms", "telecom_ids", "finance_aml"]) {
  const metrics = {};
  const artifacts = await new Promise((done) => {
    const timer = setTimeout(() => done(null), 600_000);
    m.runClientPipeline(usecase, { seed: 0 }, {
      onEvent: (e) => { if (e.type === "metric") metrics[e.data.name] = e.data.value; },
      onEnd: (s) => { clearTimeout(timer); done(s.artifacts); },
    });
  });
  result[usecase] = { record: artifacts?.result_signature?.signed_payload?.config ?? null,
                      fhe_model: metrics.fhe_model ?? null,
                      encrypted_rows_scored: metrics.encrypted_rows_scored ?? null,
                      mpc_matches_clear: metrics.mpc_matches_clear ?? null };
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
def runs(tmp_path_factory) -> dict:
    _require_node_or_skip()
    script = tmp_path_factory.mktemp("r84sig") / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB)], cwd=WEB, capture_output=True, text=True,
                          timeout=1500)
    assert proc.returncode == 0, f"the browser pipeline did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("usecase,fhe,mpc", [
    ("satellite_comms", None, None),
    ("telecom_ids", "node-seal-ckks", None),
    ("finance_aml", "node-seal-ckks", "polytune-garbled-circuit-2pc-wasm"),
])
def test_the_record_names_the_backends_the_run_executed(runs, usecase, fhe, mpc) -> None:
    r = runs[usecase]
    assert r["record"] is not None, f"{usecase}: the run signed no record"
    assert (r["record"]["fhe"], r["record"]["mpc"]) == (fhe, mpc), r
    # A named backend is backed by its own measurement in the same run, and an absent one by none.
    assert (r["fhe_model"] is not None or r["encrypted_rows_scored"] is not None) == (fhe is not None), r
    assert (r["mpc_matches_clear"] is not None) == (mpc is not None), r
