"""The page's own re-scoring, run in node on every published graph and sample, agrees with each row's record (Round 86).

WHY THIS EXISTS

The page re-scores each published graph on its evaluation sample and, until Round 86, compared one aggregate figure
with the record within 0.05, so a graph four points worse passed. No test ran the real runtime on a published graph
(`tests/test_r85_onnx_sessions_open_once.py` says so: its runtime is a stand-in). Each sample now carries a per-row
record (`qsh.models.sample_per_row`), and the page compares every row.

WHAT IS ASSERTED

`evaluatePublishedModel` from `web/src/lib/onnxInference.ts`, bundled with esbuild and run in node with the REAL
ONNX Runtime Web (its WASM build), on every published model whose sample carries a per-row record. The fetches are
answered from the local copies of the pinned files, and each body is checked against the SHA-256 the shipped config
records for its pin, as the page checks it, so a local file that is not the pinned one fails here. For every model:
the run completes, the per-row comparison is not refused, no row decides differently beyond a near tie, the largest
margin difference is within the record's tolerance, and the verification passes. It runs ONE thread: node cannot
start the runtime's threaded build, and the deployed page, which is cross-origin isolated and runs up to four, compares
against the same record on every visit (checklist PI25).

The graphs are not in the repository (only their digests are), so on a machine without them this skips and names
what is missing; `scripts/ci_local.sh` runs it where they are. `QSH_RESCORE_TRUST_LOCAL=1` replaces the digest check
with a stand-in, for measuring a sample before it is published (the run prints the largest differences).

A worktree has none of the gitignored graphs, so until 2026-11-28 this skipped in every worktree gate as it does in CI,
and nothing said so but a skip line. `QSH_PUBLISHED_GRAPHS` names a checkout that holds them: a graph absent here is
served from there, under the same digest check, so only the pinned bytes pass. `QSH_REQUIRE_PUBLISHED_GRAPHS=1` turns
the skip into a failure. And a sample that names a graph but carries no per-row record was left out in silence; the
last test below, which needs no graph and runs in CI, now requires every one to carry it.
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
import { mkdtempSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
const [WEB, PLAN] = process.argv.slice(2);
const plan = JSON.parse(readFileSync(PLAN, "utf8"));
const require = createRequire(join(WEB, "package.json"));
const { build } = require("esbuild");
// The ESM build the page loads (the package's "import" condition), not the CommonJS one `require` finds.
const ortPath = join(WEB, "node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs");
const sealWasm = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const trust = process.env.QSH_RESCORE_TRUST_LOCAL === "1";
const dir = mkdtempSync(join(tmpdir(), "qsh-rescore-"));
const out = join(dir, "bundle.mjs");
await build({
  stdin: { contents: `export { evaluatePublishedModel } from ${JSON.stringify(join(WEB, "src/lib/onnxInference"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    // The real runtime, loaded by node from its own directory so it finds its .wasm beside it.
    b.onResolve({ filter: /^onnxruntime-web\/(wasm|webgpu)$/ }, () => ({ path: ortPath, external: true }));
    // ortSameOrigin.ts (2026-11-20) points the WebGPU build's files at the site's copies by `?url` imports, which only
    // Vite resolves; without this the bundle failed to build, which the skip below hid wherever a graph was absent.
    b.onResolve({ filter: /\.(wasm|mjs)\?url$/ }, (a) => ({ path: a.path, namespace: "url-stub" }));
    b.onLoad({ filter: /.*/, namespace: "url-stub" }, () => ({ contents: "export default '';", loader: "js" }));
    if (trust) {
      b.onResolve({ filter: /(^|\/)pinnedFetch$/ }, () => ({ path: "pinned", namespace: "trust" }));
      b.onLoad({ filter: /.*/, namespace: "trust" }, () => ({ loader: "js", contents: `
        export class HubFileRefused extends Error {}
        export const asJson = (bytes) => JSON.parse(new TextDecoder().decode(bytes));
        export async function checkPinned(buf) {
          const d = await crypto.subtle.digest("SHA-256", buf);
          return Array.from(new Uint8Array(d), (x) => x.toString(16).padStart(2, "0")).join("");
        }` }));
    }
  } }],
});
// Every Hub request is answered from the local copy of the file at that path; anything else is a 404.
globalThis.fetch = async (url) => {
  const m = String(url).match(/^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/[0-9a-f]{40}\/(.+)$/);
  let local = m && plan.dirs[m[1]] ? join(plan.dirs[m[1]], decodeURIComponent(m[2])) : null;
  // A graph this checkout lacks, from the checkout QSH_PUBLISHED_GRAPHS names; the page's digest check still applies.
  if (local && !existsSync(local) && (plan.graphDirs || {})[m[1]] && decodeURIComponent(m[2]).startsWith("onnx/")) {
    local = join(plan.graphDirs[m[1]], decodeURIComponent(m[2]));
  }
  if (!local || !existsSync(local)) return new Response("Entry not found", { status: 404 });
  const seed = plan.seed;
  if (seed && m[1] === seed.repo && m[2] === "eval_sample.json") {
    // A record altered on purpose, to show the comparison catches it: one decision flipped, or one margin moved.
    const j = JSON.parse(readFileSync(local, "utf8"));
    const pr = j.expected.per_row;
    if (seed.flip != null) pr.predicted[seed.flip] = pr.predicted[seed.flip] === 0 ? 1 : 0;
    if (seed.nudge != null) pr.margin[seed.nudge] += seed.by;
    return new Response(JSON.stringify(j), { status: 200 });
  }
  return new Response(readFileSync(local), { status: 200 });
};
// One thread: node cannot start ONNX Runtime Web's threaded build, whose worker loads from a blob: URL. The deployed
// page is cross-origin isolated and runs up to four threads, and re-scores against the same record on every visit.
Object.defineProperty(globalThis.navigator, "hardwareConcurrency", { value: 1, configurable: true });
const mod = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const results = [];
for (const { usecase, key } of plan.models) {
  const t0 = performance.now();
  const v = await mod.evaluatePublishedModel(usecase, 0.05, key, "wasm");
  results.push({ key, ms: Math.round(performance.now() - t0), ran: v?.ran ?? false, error: v?.error ?? null,
                 matchesExpected: v?.matchesExpected ?? false, metric: v?.metricName, score: v?.score,
                 expected: v?.expected, perRow: v?.perRow ?? null });
}
console.log(JSON.stringify(results));
process.exit(0);
"""


def _plan() -> tuple[list[dict], dict[str, str], dict[str, str], list[str], list[str]]:
    """The models to re-score, their directories, where an absent graph is served from, what is absent, and the keys
    whose sample names a graph without a per-row record."""
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import registry

    pins = json.loads((REPO / "models" / "hub_file_hashes.json").read_text())
    other = os.environ.get("QSH_PUBLISHED_GRAPHS")
    models, dirs, graph_dirs, missing, unrecorded = [], {}, {}, [], []
    for usecase, specs in registry.frontend_models().items():
        for spec in specs:
            key = spec["key"]
            if key not in registry.ARTIFACT_DIRS:          # a reference architecture with no artifact
                continue
            directory = REPO / registry.ARTIFACT_DIRS[key]
            sample = directory / "eval_sample.json"
            if key not in pins or not sample.exists():
                continue
            expected = json.loads(sample.read_text()).get("expected", {})
            if "per_row" not in expected:
                if "graph" in expected:
                    unrecorded.append(key)
                continue
            graph = directory / "onnx" / expected["graph"]
            if not graph.exists() and other:
                there = Path(other) / registry.ARTIFACT_DIRS[key]
                if (there / "onnx" / expected["graph"]).exists():
                    graph_dirs[spec["hf_repo_id"]] = str(there)
                    graph = there / "onnx" / expected["graph"]
            if not graph.exists():
                missing.append(str((directory / "onnx" / expected["graph"]).relative_to(REPO)))
                continue
            models.append({"usecase": usecase, "key": key})
            dirs[spec["hf_repo_id"]] = str(directory)
    return models, dirs, graph_dirs, missing, unrecorded


def _run(tmp_path, models: list[dict], dirs: dict[str, str], graph_dirs: dict[str, str], seed: dict | None = None,
         trust: bool = False) -> list[dict]:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"models": models, "dirs": dirs, "graphDirs": graph_dirs, "seed": seed}))
    script = tmp_path / "harness.mjs"
    script.write_text(HARNESS)
    env = {**os.environ, **({"QSH_RESCORE_TRUST_LOCAL": "1"} if trust else {})}
    proc = subprocess.run(["node", str(script), str(WEB), str(plan)], cwd=WEB, capture_output=True, text=True,
                          timeout=1800, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _ready() -> tuple[list[dict], dict[str, str], dict[str, str]]:
    if not ESBUILD.exists() or shutil.which("node") is None:
        pytest.skip("node or web/node_modules is absent, so the page's re-scoring cannot run here")
    models, dirs, graph_dirs, missing, _ = _plan()
    if missing:
        msg = ("the published graphs are not in the repository; absent here"
               + (" and in QSH_PUBLISHED_GRAPHS" if os.environ.get("QSH_PUBLISHED_GRAPHS") else "") + ": "
               + ", ".join(missing))
        if os.environ.get("QSH_REQUIRE_PUBLISHED_GRAPHS") == "1":
            pytest.fail(msg + ". QSH_REQUIRE_PUBLISHED_GRAPHS=1 is set, so this is a failure, not a skip.")
        pytest.skip(msg)
    if not models:
        pytest.skip("no sample carries a per-row record yet")
    return models, dirs, graph_dirs


def test_every_published_graph_reproduces_each_row_of_its_record(tmp_path) -> None:
    models, dirs, graph_dirs = _ready()
    results = _run(tmp_path, models, dirs, graph_dirs)
    report = []
    for r in results:
        pr = r["perRow"] or {}
        report.append(f"{r['key']}: ran {r['ran']}, {r['metric']} {r['score']} (recorded {r['expected']}), rows "
                      f"{pr.get('rows')}, different {pr.get('mismatches')}, near ties {pr.get('nearTies')}, largest "
                      f"difference {pr.get('maxDiff')}, {r['ms']} ms" + (f", refused: {pr['refused']}" if pr.get("refused") else "")
                      + (f", error: {r['error']}" if r["error"] else ""))
    print("\n".join(report))
    for r in results:
        pr = r["perRow"]
        assert r["ran"] and not r["error"], f"{r['key']}: {r['error']}"
        assert pr and not pr.get("refused"), f"{r['key']}: {pr}"
        assert pr["mismatches"] == 0 and pr["maxDiff"] <= pr["tolerance"], f"{r['key']}: {pr}"
        assert r["matchesExpected"], f"{r['key']}: {r}"


def test_a_record_altered_by_one_row_fails_the_verification(tmp_path) -> None:
    """Seeded: one decision flipped, then one margin moved by 0.01, in the telecom QML sample as served."""
    models, dirs, graph_dirs = _ready()
    qml = [m for m in models if m["key"] == "telecom_ids.qml"]
    if not qml:
        pytest.skip("the telecom QML graph is not here")
    repo = next(r for r, d in dirs.items() if d.endswith("telecom_qml_ids"))
    flipped = _run(tmp_path, qml, dirs, graph_dirs, seed={"repo": repo, "flip": 0}, trust=True)[0]
    assert flipped["perRow"]["mismatches"] == 1 and not flipped["matchesExpected"], flipped
    nudged = _run(tmp_path, qml, dirs, graph_dirs, seed={"repo": repo, "nudge": 1, "by": 0.01}, trust=True)[0]
    assert nudged["perRow"]["mismatches"] == 0 and nudged["perRow"]["maxDiff"] > nudged["perRow"]["tolerance"], nudged
    assert not nudged["matchesExpected"], nudged


def test_every_published_sample_that_names_a_graph_carries_its_per_row_record() -> None:
    """Needs no graph, so it runs in CI: a sample without the record would be re-scored by one aggregate within 0.05."""
    *_, unrecorded = _plan()
    assert not unrecorded, f"these published samples name a graph and carry no per-row record: {unrecorded}"

