"""Repeated runs in one tab do not grow the CKKS WebAssembly heap.

WHY THIS EXISTS (Round 85)

node-seal's plaintexts and ciphertexts live in the WebAssembly heap, which the JavaScript collector
neither sees nor frees: an object nobody deletes stays allocated for the life of the tab. Only the
dense layer and the masked decryption deleted theirs, so every other CKKS path (the held-out batch,
the single-row linear layer, the convolution) left its inputs, weights, products and rotations
behind. A finance run scores 8,192 held-out rows in 64 ciphertexts, and each run left about 330 MB:
the heap grew from 431 MB after the first run to 2,097 MB after the sixth, measured here in node with
the collector forced between runs. In headless Chrome the finance page stopped responding at its
fourteenth run with nothing changed between runs but the Start button, its renderer past 5 GB (a
desktop Chrome walking the page's selections stopped at its twelfth); a 32-bit WebAssembly heap
cannot grow past 4 GiB.

WHAT IS ASSERTED, on runs of the browser pipeline in node built as the client-only demo:

- Finance, telecom and genomics each run four times, each use case in a node process of its own, the
  collector forced after each run, and the external (WebAssembly) memory after the fourth run is
  within a few megabytes of the second. Run one is excluded: it generates the CKKS keys and compiles
  the backends, which stay. One process per use case, because a heap is grown in steps and does not
  shrink: run after finance in the same process, telecom's and genomics' leaks fitted into the room
  finance's had already reserved, and both passed on the base commit.
- Every run still encrypts the rows it encrypted before, so the check is not passing because the
  encrypted tier stopped running.

On the base commit (6bb8657) finance grows by 718 MB between runs two and four, telecom by 151 MB
and genomics by 44 MB, and the test fails on all three.
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

USECASES = ("finance_aml", "telecom_ids", "genomics")
RUNS = 4
# Megabytes the heap may move between the second run and the last: allocator slack, not a leak.
# The leak this guards against was 20 to 330 MB per run.
SLACK_MB = 8

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const RUNS = Number(process.argv[3]);
const usecase = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r85-heap-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/","VITE_CLIENT_ONLY":"1"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    // The garbled-circuit argmax is a gitignored wasm build: a plaintext argmax stands in, as in
    // tests/test_r84_finance_browser_run.py. Its heap is not the one under test.
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
const runs = [];
for (let r = 0; r < RUNS; r++) {
  const metrics = {};
  await new Promise((done) => {
    const timer = setTimeout(done, 600_000);
    m.runClientPipeline(usecase, { seed: r }, {
      onEvent: (e) => { if (e.type === "metric") metrics[e.data.name] = e.data.value; },
      onEnd: () => { clearTimeout(timer); done(); },
    });
  });
  globalThis.gc();
  runs.push({ externalMB: process.memoryUsage().external / 1048576,
              encrypted: metrics.encrypted_rows_scored ?? null });
}
console.log(JSON.stringify(runs));
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
    script = tmp_path_factory.mktemp("r85heap") / "harness.mjs"
    script.write_text(HARNESS)
    out = {}
    for usecase in USECASES:
        proc = subprocess.run(["node", "--expose-gc", str(script), str(WEB), str(RUNS), usecase], cwd=WEB,
                              capture_output=True, text=True, timeout=1500)
        assert proc.returncode == 0, f"{usecase}: the browser pipeline did not run:\n{proc.stderr[-3000:]}"
        out[usecase] = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


@pytest.mark.parametrize("usecase", USECASES)
def test_repeated_runs_leave_the_ckks_heap_where_it_was(runs, usecase) -> None:
    r = runs[usecase]
    grew = r[-1]["externalMB"] - r[1]["externalMB"]
    assert grew <= SLACK_MB, (
        f"{usecase}: the WebAssembly heap grew {grew:.0f} MB between run 2 and run {RUNS} "
        f"({[round(x['externalMB']) for x in r]} MB after each run): a CKKS object is not being deleted")


@pytest.mark.parametrize("usecase", USECASES)
def test_every_run_still_encrypts_its_rows(runs, usecase) -> None:
    counts = [x["encrypted"] for x in runs[usecase]]
    assert counts[0] and all(c == counts[0] for c in counts), (usecase, counts)
