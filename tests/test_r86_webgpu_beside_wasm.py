"""Each page's published fp32 graphs on WebGPU beside WASM (Round 86, batch 8; the owner's choice of 2026-11-27, CS23).

WHY THIS EXISTS

Every page with a published ONNX graph lets a visitor pick WebGPU, and until batch 8 only the medical page had that
measured. `BackendCompareCard` runs a model's fp32 graph on WASM and on WebGPU in the tab over every row of its
published sample and compares every value and decision (`web/src/lib/backendCompare.ts`, `compareBackends` in
`web/src/lib/onnxInference.ts`). WebGPU itself cannot run in node, so what is held here is everything around it:

  - which graph is compared: the page's own graph when it is fp32, else its fp32 twin (an int8 graph has no WebGPU
    kernel for its quantised nodes and stays on WASM), and that twin is pinned, so the page's digest check accepts it;
  - which models are not compared, and why (an `ai.onnx.ml` graph, which the WebGPU build does not register);
  - the comparison's arithmetic: differences, decisions by the sample's rule, near ties, the tolerance;
  - the whole run with the REAL ONNX Runtime Web (its WASM build) on two sessions of one graph, for a row sample, a
    two-class sample with a threshold, a forecast and a graph sample, built here so CI has them, and on every pinned
    graph found locally: two runs of one graph on one backend agree exactly;
  - that the WebGPU session is opened with WebGPU alone, so a backend that cannot start fails rather than quietly
    running on WASM and being timed as WebGPU; and that every page with a published graph mounts the card.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
PY = sys.executable

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
const [WEB, PLAN] = process.argv.slice(2);
const plan = JSON.parse(readFileSync(PLAN, "utf8"));
const require = createRequire(join(WEB, "package.json"));
const { build } = require("esbuild");
const ortPath = join(WEB, "node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs");
const sealWasm = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const dir = mkdtempSync(join(tmpdir(), "qsh-backends-"));
const out = join(dir, "bundle.mjs");
await build({
  stdin: { contents: `export * from ${JSON.stringify(join(WEB, "src/lib/backendCompare"))};
                      export { scoreAndTime, compareScored, adapterInfoOf } from ${JSON.stringify(join(WEB, "src/lib/onnxInference"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    b.onResolve({ filter: /^onnxruntime-web\/(wasm|webgpu)$/ }, () => ({ path: ortPath, external: true }));
    // ortSameOrigin.ts points the WebGPU build's files at this site's copies by `?url` imports, which only Vite resolves.
    b.onResolve({ filter: /\.(wasm|mjs)\?url$/ }, (a) => ({ path: a.path, namespace: "url-stub" }));
    b.onLoad({ filter: /.*/, namespace: "url-stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
const mod = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const results = {};
if (plan.artifacts) {
  results.choice = plan.artifacts.map(({ key, artifacts, domains }) => {
    const graph = mod.fp32GraphFor(artifacts);
    return { key, graph, page: mod.pageGraphOf(artifacts)?.path ?? null, why: mod.webgpuInapplicable(domains, graph) };
  });
}
if (plan.arithmetic) {
  const d = (cls, margin) => ({ cls, margin });
  const same = mod.compareOutputs([1, 2, 3, 4], [1, 2, 3, 4], 2, null);
  const over = mod.compareOutputs([1, 2, 3, 4], [1, 2, 3.002, 4], 2, null);
  const flip = mod.compareOutputs([0, 1], [0, 1], 2, { wasm: () => d(1, 0.5), other: () => d(0, -0.5) });
  const tie = mod.compareOutputs([0, 1], [0, 1], 2, { wasm: () => d(1, 4e-4), other: () => d(0, -4e-4) });
  // The sample's rules: a threshold on p(class 1); the first maximum; one score against 0.5.
  const thr = mod.decider([0.7, 0.3, 0.2, 0.8], 2, 0.4, [0.7, 0.3, 0.2, 0.8]);
  const arg = mod.decider([1, 5, 5, 0, 9, 2], 3, null, null);
  const one = mod.decider([0.2, 0.9], 1, null, null);
  let refused = null;
  try { mod.decider([0, 1], 2, 0.5, null); } catch (e) { refused = String(e.message); }
  let unequal = null;
  try { mod.compareOutputs([1, 2], [1], 1, null); } catch (e) { unequal = String(e.message); }
  results.arithmetic = { same, over, flip, tie, thr: [thr(0), thr(1)], arg: [arg(0), arg(1)], one: [one(0), one(1)],
                         refused, unequal, median: [mod.median([3, 1, 2]), mod.median([4, 1, 2, 3])], tolerance: mod.BACKEND_TOLERANCE };
}
if (plan.adapter) {
  // As Chrome's GPUAdapterInfo: every field an accessor on the prototype, none an own property.
  class Info { get vendor() { return "nvidia"; } get architecture() { return "blackwell"; }
               get description() { return "a stand-in"; } get isFallbackAdapter() { return false; } }
  const info = new Info();
  results.adapter = { spread: { ...info }, read: mod.adapterInfoOf(info), none: mod.adapterInfoOf(null) ?? null };
}
if (plan.runs) {
  const ort = await import(pathToFileURL(ortPath).href);
  ort.env.wasm.numThreads = 1;      // node cannot start the threaded build
  results.runs = [];
  for (const r of plan.runs) {
    const bytes = readFileSync(r.graph);
    const sample = JSON.parse(readFileSync(r.sample, "utf8"));
    const open = async () => ({ ort, session: await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ["wasm"] }) });
    const s1 = await open(), s2 = await open();
    const a = await mod.scoreAndTime(s1, sample, 0), b = await mod.scoreAndTime(s2, sample, 0);
    const c = mod.compareScored(a, b, sample);
    results.runs.push({ name: r.name, rows: a.rows, width: a.width, output: a.name, out: c, times: [a.times, b.times] });
    await s1.session.release(); await s2.session.release();
  }
}
console.log(JSON.stringify(results));
process.exit(0);
"""


def _node(tmp_path: Path, plan: dict) -> dict:
    if not ESBUILD.exists() or shutil.which("node") is None:
        pytest.skip("node or web/node_modules is absent, so the comparison cannot run here")
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "harness.mjs").write_text(HARNESS)
    proc = subprocess.run(["node", str(tmp_path / "harness.mjs"), str(WEB), str(tmp_path / "plan.json")], cwd=WEB,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _published() -> list[dict]:
    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import registry

    pins = json.loads((REPO / "models" / "hub_file_hashes.json").read_text())
    out = []
    for specs in registry.frontend_models().values():
        for spec in specs:
            arts = [a for a in spec.get("artifacts", []) if a.get("fmt") == "onnx"]
            if not spec.get("hf_repo_id") or not spec.get("hf_revision") or not arts:
                continue
            if not any(a.get("serves") in ("browser", "both") for a in arts):
                continue
            out.append({"key": spec["key"], "artifacts": spec["artifacts"], "domains": spec.get("onnx_domains"),
                        "pins": pins.get(spec["key"], {}).get("files", {})})
    return out


def _expected_choice(artifacts: list[dict]) -> tuple[str | None, str | None]:
    """The rule written out again: the page's graph (int8 first), and its fp32 twin."""
    onnx = [a for a in artifacts if a.get("fmt") == "onnx" and a.get("serves") in ("browser", "both")]
    page = next((a for a in onnx if a.get("precision") == "int8"), None) \
        or next((a for a in onnx if a.get("precision") == "fp32"), None) or (onnx[0] if onnx else None)
    if page is None:
        return None, None
    if page.get("precision") == "fp32":
        return page["path"], page["path"]
    twin = page["path"].replace("_int8.onnx", ".onnx")
    fp32 = next((a["path"] for a in artifacts if a.get("fmt") == "onnx" and a.get("precision") == "fp32"
                 and a["path"] == twin), None)
    return page["path"], fp32


def test_each_published_model_compares_its_fp32_graph_or_says_why_not(tmp_path) -> None:
    models = _published()
    assert models, "no published model with a browser graph"
    got = {c["key"]: c for c in _node(tmp_path, {"artifacts": [{k: m[k] for k in ("key", "artifacts", "domains")}
                                                                for m in models]})["choice"]}
    comparable = []
    for m in models:
        page, fp32 = _expected_choice(m["artifacts"])
        c = got[m["key"]]
        assert c["graph"] == fp32 and c["page"] == page, (m["key"], c, page, fp32)
        ml = "ai.onnx.ml" in (m["domains"] or [])
        assert (c["why"] is not None) == (ml or fp32 is None), (m["key"], c)
        if c["why"] is None:
            comparable.append(m["key"])
            # The page checks the fetched graph against the digest shipped for this pin and refuses one with none.
            assert fp32 in m["pins"], f"{m['key']}: {fp32} has no pinned SHA-256, so the page would refuse it"
        if page and page.endswith("_int8.onnx"):
            assert fp32 and not fp32.endswith("_int8.onnx"), (m["key"], fp32)
    # Every page with a published graph has at least one the card can compare (genomics and DeFi publish none).
    pages = {k.split(".")[0] for k in comparable}
    assert {"finance_aml", "medical_fl", "satellite_comms", "telecom_ids"} <= pages, comparable


def test_the_comparison_counts_differences_decisions_and_ties(tmp_path) -> None:
    a = _node(tmp_path, {"arithmetic": True})["arithmetic"]
    assert a["tolerance"] == 1e-3
    assert a["same"]["maxAbsDiff"] == 0 and a["same"]["withinTolerance"] and a["same"]["decisions"] == 0
    assert a["over"]["maxAbsDiff"] == pytest.approx(0.002) and not a["over"]["withinTolerance"]
    assert a["over"]["maxAbs"] == 4
    assert a["flip"]["decisionsDiffer"] == 2 and not a["flip"]["withinTolerance"]
    assert a["tie"]["decisionsDiffer"] == 0 and a["tie"]["nearTies"] == 2 and a["tie"]["withinTolerance"]
    assert [t["cls"] for t in a["thr"]] == [0, 1] and a["thr"][0]["margin"] == pytest.approx(-0.1)
    assert [t["cls"] for t in a["arg"]] == [1, 1] and a["arg"][0]["margin"] == 0     # the first maximum wins a tie
    assert a["arg"][1]["margin"] == 7
    assert [t["cls"] for t in a["one"]] == [0, 1]
    assert "needs probabilities" in a["refused"] and "returned 2 and 1 values" in a["unequal"]
    assert a["median"] == [2, 2.5]


def _tiny_graphs(tmp: Path) -> list[dict]:
    """Four graphs and samples of the four shapes the card meets, so CI, which has no published graph, runs them."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)

    def save(name: str, nodes, inputs, outputs, inits) -> Path:
        g = helper.make_graph(nodes, name, inputs, outputs, [numpy_helper.from_array(v, k) for k, v in inits.items()])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
        m.ir_version = 8
        onnx.checker.check_model(m)
        path = tmp / f"{name}.onnx"
        path.write_bytes(m.SerializeToString())
        return path

    f32 = TensorProto.FLOAT
    runs = []
    # Rows to three classes' logits, read by the first maximum.
    W = rng.normal(size=(4, 3)).astype(np.float32)
    g = save("rows", [helper.make_node("MatMul", ["x", "W"], ["logits"])],
             [helper.make_tensor_value_info("x", f32, ["n", 4])], [helper.make_tensor_value_info("logits", f32, ["n", 3])],
             {"W": W})
    s = {"provenance": "made by the test", "input": {"shape": [24, 4], "dtype": "float32", "preprocessing": [], "layout": "n,f"},
         "samples": rng.normal(size=(24, 4)).round(4).tolist(), "labels": [0] * 24, "expected": {"graph": "rows.onnx", "note": ""}}
    (tmp / "rows.json").write_text(json.dumps(s))
    runs.append({"name": "rows", "graph": str(g), "sample": str(tmp / "rows.json")})
    # Two classes' probabilities, read at a threshold, with the image preprocessing the samples state.
    W2 = rng.normal(size=(6, 2)).astype(np.float32)
    g = save("threshold", [helper.make_node("MatMul", ["x", "W"], ["z"]), helper.make_node("Softmax", ["z"], ["probabilities"], axis=1)],
             [helper.make_tensor_value_info("x", f32, ["n", 6])], [helper.make_tensor_value_info("probabilities", f32, ["n", 2])],
             {"W": W2})
    s = {"provenance": "made by the test", "input": {"shape": [16, 6], "dtype": "float32", "layout": "n,f",
                                                      "preprocessing": ["divide by 255", "subtract 0.5", "divide by 0.5"]},
         "samples": rng.integers(0, 256, size=(16, 6)).tolist(), "labels": [1] * 16,
         "expected": {"graph": "threshold.onnx", "note": "", "decision_threshold": 0.37}}
    (tmp / "threshold.json").write_text(json.dumps(s))
    runs.append({"name": "threshold", "graph": str(g), "sample": str(tmp / "threshold.json")})
    # A forecast: windows of 5 steps of 3 channels to the next step, compared value by value.
    g = save("forecast", [helper.make_node("ReduceMean", ["x", "axes"], ["forecast"], keepdims=0)],
             [helper.make_tensor_value_info("x", f32, ["n", 5, 3])], [helper.make_tensor_value_info("forecast", f32, ["n", 3])],
             {"axes": np.array([1], dtype=np.int64)})
    s = {"task": "forecast", "provenance": "made by the test",
         "input": {"shape": [8, 5, 3], "dtype": "float32", "preprocessing": [], "layout": "n,t,c"},
         "samples": rng.normal(size=(8, 15)).round(4).tolist(), "targets": rng.normal(size=(8, 3)).round(4).tolist(),
         "expected": {"graph": "forecast.onnx", "note": ""}}
    (tmp / "forecast.json").write_text(json.dumps(s))
    runs.append({"name": "forecast", "graph": str(g), "sample": str(tmp / "forecast.json")})
    # A graph sample, as the finance GNNs' (the five named inputs); the logits read at the seeds' positions.
    Wg = rng.normal(size=(3, 2)).astype(np.float32)
    ins = [helper.make_tensor_value_info("x", f32, ["v", 3]),
           helper.make_tensor_value_info("edge_index", TensorProto.INT64, [2, "e"]),
           helper.make_tensor_value_info("edge_attr", f32, ["e", 2]),
           helper.make_tensor_value_info("rev_edge_index", TensorProto.INT64, [2, "e"]),
           helper.make_tensor_value_info("rev_edge_attr", f32, ["e", 2])]
    g = save("graph", [helper.make_node("MatMul", ["x", "W"], ["xw"]),
                       helper.make_node("ReduceSum", ["edge_attr", "ax0"], ["s"], keepdims=1),
                       helper.make_node("ReduceSum", ["rev_edge_attr", "ax0"], ["r"], keepdims=1),
                       helper.make_node("Add", ["s", "r"], ["sr"]),
                       helper.make_node("Add", ["xw", "sr"], ["logits"])],
             ins, [helper.make_tensor_value_info("logits", f32, ["v", 2])], {"W": Wg, "ax0": np.array([0], dtype=np.int64)})
    v, e = 10, 6
    s = {"kind": "graph", "protocol": "test", "protocol_note": "made by the test",
         "inputs": {"x": rng.normal(size=v * 3).round(4).tolist(), "x_shape": [v, 3],
                    "edge_index": rng.integers(0, v, size=2 * e).tolist(), "edge_attr": rng.normal(size=e * 2).round(4).tolist(),
                    "edge_attr_dim": 2, "rev_edge_index": rng.integers(0, v, size=2 * e).tolist(),
                    "rev_edge_attr": rng.normal(size=e * 2).round(4).tolist(), "n_edges": e},
         "seeds": {"positions_in_batch": [1, 4, 7], "row_index_in_test_window": [0, 1, 2], "labels": [0, 1, 0]},
         "expected": {"decision_threshold": 0.5, "accuracy": 0.0, "graph": "graph.onnx"}}
    (tmp / "graph.json").write_text(json.dumps(s))
    runs.append({"name": "graph", "graph": str(g), "sample": str(tmp / "graph.json")})
    return runs


def _local_published() -> list[dict]:
    """Every pinned fp32 graph the card compares that is present here, with its sample (none in CI)."""
    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import registry

    runs = []
    for m in _published():
        page, fp32 = _expected_choice(m["artifacts"])
        if not fp32 or "ai.onnx.ml" in (m["domains"] or []) or m["key"] not in registry.ARTIFACT_DIRS:
            continue
        d = REPO / registry.ARTIFACT_DIRS[m["key"]]
        graph, sample = d / fp32, d / "eval_sample.json"
        if not graph.exists() or not sample.exists():
            continue
        if hashlib.sha256(graph.read_bytes()).hexdigest() != m["pins"].get(fp32):
            continue
        if json.loads(sample.read_text()).get("task") == "forecast_events":
            continue
        runs.append({"name": m["key"], "graph": str(graph), "sample": str(sample)})
    return runs


def test_two_runs_of_one_graph_on_one_backend_agree_exactly(tmp_path) -> None:
    pytest.importorskip("onnx")
    runs = _tiny_graphs(tmp_path) + _local_published()
    got = {r["name"]: r for r in _node(tmp_path, {"runs": runs})["runs"]}
    print("\n".join(f"{k}: {r['rows']} rows x {r['width']} ({r['output']}), largest difference {r['out']['maxAbsDiff']}, "
                    f"decisions {r['out']['decisions']}, times {r['times'][0]}" for k, r in got.items()))
    for name, r in got.items():
        o = r["out"]
        assert o["maxAbsDiff"] == 0 and o["decisionsDiffer"] == 0 and o["nearTies"] == 0 and o["withinTolerance"], (name, o)
        for t in r["times"]:
            assert all(t[k] is None or t[k] >= 0 for k in ("createMs", "firstMs", "batchMs", "rowMs")), (name, t)
    assert got["rows"]["rows"] == 24 and got["rows"]["width"] == 3 and got["rows"]["out"]["decisions"] == 24
    assert got["threshold"]["output"] == "probabilities" and got["threshold"]["out"]["decisions"] == 16
    assert got["forecast"]["out"]["decisions"] == 0 and got["forecast"]["out"]["values"] == 24
    assert got["graph"]["rows"] == 3 and got["graph"]["times"][0]["rowMs"] is None


def test_the_adapter_is_read_field_by_field(tmp_path) -> None:
    """`{ ...adapter.info }` copies nothing from Chrome's GPUAdapterInfo, whose fields are prototype accessors."""
    a = _node(tmp_path, {"adapter": True})["adapter"]
    assert a["spread"] == {}, "the stand-in does not behave as Chrome's GPUAdapterInfo"
    assert a["read"] == {"vendor": "nvidia", "architecture": "blackwell", "description": "a stand-in",
                         "isFallbackAdapter": False}
    assert a["none"] is None
    src = (WEB / "src" / "lib" / "onnxInference.ts").read_text()
    assert "{ ...adapter.info }" not in src.replace("`{ ...adapter.info }`", ""), "an adapter's info is spread again"


def test_the_webgpu_session_cannot_fall_back_to_wasm_unseen_and_every_page_mounts_the_card() -> None:
    src = (WEB / "src" / "lib" / "onnxInference.ts").read_text()
    body = src[src.index("export async function compareBackends"):]
    assert 'executionProviders: [gpu ? "webgpu" : "wasm"]' in body, "the WebGPU session must be opened with WebGPU alone"
    assert "checkPinned(buf, spec.file_sha256?.[graph!]" in body, "the compared graph must be checked against its pin"
    assert "s.session.release()" in body, "the comparison's own sessions must be released"
    page = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert "<BackendCompareCard usecase={usecase} />" in page
    card = (WEB / "src" / "components" / "BackendCompareCard.tsx").read_text()
    assert "{BACKEND_TOLERANCE}" in card and "1e-3" not in card and "0.001" not in card, "the card states the module's tolerance"
    assert "{BATCH_REPEATS}" in card and "{ROW_REPEATS}" in card
