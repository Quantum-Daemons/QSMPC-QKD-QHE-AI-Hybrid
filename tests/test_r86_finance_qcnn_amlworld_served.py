"""Round 86, batch 6: the finance QCNN is the AMLworld circuit of the pre-registered comparison, served as trained.

WHY THIS EXISTS (the owner's choices of 2026-11-26; references/finance-qcnn-amlworld-comparison-preregistration.md)

From Round 84 the finance page ran a PaySim circuit (four wires, PaySim's four columns), and this file, then
`test_r84_qcnnfin_four_columns.py`, held it to its trainer, its payload and the page. The comparison's arm 1, seed 6,
replaced it: eight wires reading eight of AMLworld's 114 inputs, written by `scripts/export_finance_qcnn_amlworld.py`
with the fit window's standardisation folded into its moments and the validation-best threshold into its bias. The same
chain is held here to the new circuit.

WHAT IS ASSERTED

- The trained and served copies are one file: the pre-registered eight inputs at eight wires, the tree at c = 0.5 with
  77 angles, raw inputs, and arm 1's training command at seed 6, the seed the record ships.
- What the export changed is recorded, and changed no decision: the clip it does not carry moves no test decision, and
  the threshold folded into the bias is the record's.
- No trainer run can write over it: a PaySim run writes `models/qcnn_finance_paysim`, an AMLworld run
  `models/qcnn_finance_amlworld`, and every other artifact's recorded command still writes the directory it ships from.
- No parameter of the shipped circuit is dead, and the page declares the width it has.
- The Python tier's implementation gives the trainer's class on every row of the page's sample.
- The page feeds it the scored transaction's own eight inputs, in the circuit's order, and the order is load-bearing.
- The page's pipeline runs the eight-wire circuit, and a transaction that lacks one of its inputs is refused, not
  filled or sliced.
- The card states the figures the weights file records.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
PUBLIC = WEB / "public" / "data"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
TRAINED = REPO / "models" / "qcnn_finance" / "weights.json"
SERVED = PUBLIC / "finance" / "qcnn_weights.json"
SAMPLE = PUBLIC / "finance" / "qcnn_heldout.json"
STUDENT = PUBLIC / "finance" / "student_heldout.json"
CARD = REPO / "models" / "qcnn_finance" / "README.md"
FIRST_RECORD = REPO / "docs" / "measurements" / "finance-qcnn-amlworld-2026-11-26.json"
COMPARISON = REPO / "docs" / "measurements" / "finance-qcnn-amlworld-comparison-2026-11-26.json"
EXPORT = REPO / "scripts" / "export_finance_qcnn_amlworld.py"
sys.path.insert(0, str(REPO / "scripts" / "train"))
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")

#: Written out, so each check reads the ARTIFACT against a fixed list as well as against the record.
EIGHT = ["format_risk", "pair_log_same_direction", "pair_first_contact", "dst_in_uniq_4320", "dst_bank_diversity",
         "src_out_uniq_60", "src_out_mean_60", "src_out_mean_360"]


def _q():
    import qcnn_variational as q

    return q


def _w(path: Path = TRAINED) -> dict:
    return json.loads(path.read_text())


def _sample() -> tuple[np.ndarray, np.ndarray]:
    held = json.loads(SAMPLE.read_text())
    return np.asarray(held["rows"], dtype=float), np.asarray(held["labels"])


def _trainer_readout(w: dict, raw: np.ndarray) -> np.ndarray:
    """`<Z0>` of the trainer's own circuit on raw rows: standardise, tanh, scale, simulate."""
    q = _q()
    z = (np.asarray(raw, dtype=float) - np.asarray(w["feature_mean"])) / np.asarray(w["feature_std"])
    x = float(w.get("encoding_scale", 1.0)) * np.tanh(z)
    return q.circuit(torch.tensor(x, dtype=torch.float32), torch.tensor(w["theta"], dtype=torch.float32),
                     int(w["n_qubits"]), readout_qubits=0, layers=int(w["layers"]),
                     arch=str((w.get("circuit") or {}).get("name", "tree"))).detach().numpy()


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the browser code cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


# --------------------------------------------------------------------------- the artifact
def test_the_trained_and_served_copies_are_one_file_at_eight_wires() -> None:
    assert TRAINED.read_bytes() == SERVED.read_bytes(), "the page serves another file than the trained artifact"
    q, w = _q(), _w()
    assert w["feature_names"] == EIGHT == json.loads(FIRST_RECORD.read_text())["inputs"]["features"]
    assert w["n_features"] == w["n_qubits"] == 8 and w["usecase"] == "finance" and w["trained_as"] == "finance_amlworld"
    assert w["input_space"] == "raw" and w["circuit"]["name"] == "tree" and float(w["encoding_scale"]) == 0.5
    assert len(w["theta"]) == q.param_count(8, 0, int(w["layers"]), "tree") == 77
    assert "AMLworld" in w["provenance"] and "https://cdla.dev/sharing-1-0/" in w["provenance"]
    cmd = w["training"]["command"]
    for flag in ("--usecase finance_amlworld", "--qubits 8", "--arch tree", "--layers 3", "--encoding-scale 0.5",
                 "--select-on ap", "--seed 6"):
        assert flag in cmd, cmd
    ship = json.loads(COMPARISON.read_text())["arms"]["qcnn1"]["shipped_seed"]
    assert ship["seed"] == w["selection"]["seed"] == w["training"]["seed"] == 6
    assert w["selection"]["of_seeds"] == len(json.loads(COMPARISON.read_text())["arms"]["qcnn1"]["seeds"]) == 20


def test_what_the_export_changed_is_recorded_and_moved_no_decision() -> None:
    w = _w()
    assert w["standardisation"]["clip"] == 10.0 and "not carried" in w["standardisation"]["folded"]
    assert w["clip_effect"]["decisions_that_could_change"] == 0
    assert w["clip_effect"]["largest_readout_change_up_to_the_limit"] < 1e-6
    t = w["decision_threshold"]
    assert t["folded_into_bias"] and "validation window" in t["rule"] and "never chosen on the test window" in t["rule"]
    assert abs(float(w["bias"]) - (t["trainer_bias"] - t["logit"])) < 1e-9
    check = json.loads(COMPARISON.read_text())["criteria"]["1_export_check"]
    assert check["criterion_met"] and check["page_vs_state_vector"]["decisions_differing"] == 0
    assert w["comparison"]["export_check"]["test_rows"] == check["test_rows"] == w["n_test"]


def test_no_trainer_run_writes_over_the_served_circuit() -> None:
    """The served circuit is written by the export from the comparison's work folder, so no command the trainer runs
    without `--out` may land in `models/qcnn_finance`, and every other artifact's recorded command still writes the
    directory it ships from."""
    q = _q()
    assert q.artifact_dir("finance", 4).name == "qcnn_finance_paysim"
    assert q.artifact_dir("finance", 8).name == "qcnn_finance_paysim_q8"
    assert q.artifact_dir("finance_amlworld", 8).name == "qcnn_finance_amlworld"
    assert q.artifact_dir("telecom", 7).name == "qcnn_telecom_q7"
    assert 'WEIGHTS = REPO / "models" / "qcnn_finance" / "weights.json"' in EXPORT.read_text(), "the export writes the served file"
    seen = 0
    for path in sorted((REPO / "models").glob("qcnn_*/weights.json")):
        cmd = _w(path)["training"]["command"]
        usecase = re.search(r"--usecase (\w+)", cmd).group(1)
        width = int(re.search(r"--qubits (\d+)", cmd).group(1))
        assert width == _w(path)["n_qubits"], f"{path.parent.name}: the command trains {width} wires"
        if path == TRAINED:
            assert q.artifact_dir(usecase, width) != path.parent, "a plain training run would overwrite the served file"
        else:
            assert q.artifact_dir(usecase, width) == path.parent, (
                f"{path.parent.name}: its recorded command writes {q.artifact_dir(usecase, width)}")
        seen += 1
    assert seen >= 4


def test_no_parameter_of_the_shipped_circuit_is_dead() -> None:
    """The structure test's perturbation at the layout finance ships (8 wires, binary, 3 rounds), on random angles and on
    the shipped ones."""
    q, w = _q(), _w()
    nq, layers = int(w["n_qubits"]), int(w["layers"])
    assert (nq, layers) == (8, 3)
    g = torch.Generator().manual_seed(0)
    x = 0.5 * torch.tanh(torch.randn(8, nq, generator=g) * 0.7)
    for theta in (torch.randn(q.param_count(nq, 0, layers), generator=g) * 0.6,
                  torch.tensor(w["theta"], dtype=torch.float32)):
        base = q.circuit(x, theta, nq=nq, readout_qubits=0, layers=layers).detach()
        dead = []
        for i in range(len(theta)):
            bumped = theta.clone()
            bumped[i] += 1.234
            out = q.circuit(x, bumped, nq=nq, readout_qubits=0, layers=layers).detach()
            if float((base - out).abs().max()) < 1e-5:
                dead.append(i)
        assert not dead, f"these angles cannot move the readout: {dead}"


def test_the_page_declares_the_width_the_artifact_has() -> None:
    src = (WEB / "src" / "lib" / "usePipelineRun.ts").read_text()
    m = re.search(r"QCNN_WIDTHS: Record<string, number\[\]> = \{.*?finance_aml: \[([^\]]*)\]", src, re.S)
    assert m and [int(v) for v in re.findall(r"\d+", m.group(1))] == [_w(SERVED)["n_qubits"]] == [8]


def test_the_python_tier_gives_the_trainers_class_on_every_row_of_the_sample() -> None:
    """`qcnn_trained.infer`, the Python pipeline's implementation, fed the page's sample raw as `usecases/finance_aml`
    feeds it, against the trainer's circuit."""
    from qsh.qhe.qcnn_trained import infer

    w = _w()
    X, _ = _sample()
    want = _trainer_readout(w, X)
    got = [infer(list(r), w) for r in X]
    assert float(np.abs(np.asarray([g[0] for g in got]) - want).max()) < 1e-4
    t_cls = (want * float(w["scale"]) + float(w["bias"]) > 0).astype(int)
    assert [g[1] for g in got] == t_cls.tolist()


# --------------------------------------------------------------------------- the page
_INPUT_HARNESS = r"""
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2], STEP = Number(process.argv[3]);
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r86-qcnnfin-"));
const out = join(dir, "bundle.mjs");
await build({
  stdin: { contents: `export { realInput } from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};
export { qcnnInferTrained } from ${JSON.stringify(resolve(WEB, "src/lib/quantumSim"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
});
globalThis.fetch = async (url) => {
  const text = readFileSync(join(WEB, "public", String(url).replace(/^\/+/, "")), "utf8");
  return { ok: true, status: 200, headers: { get: () => "application/json" },
           json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const w = JSON.parse(readFileSync(join(WEB, "public/data/finance/qcnn_weights.json"), "utf8"));
const n = JSON.parse(readFileSync(join(WEB, "public/data/finance/student_heldout.json"), "utf8")).samples.length;
const res = [];
for (let i = 0; i < n; i += STEP) {
  // The page's own selection: `u` lands on row i exactly, floor(((i + 0.5) / n) * n) = i.
  const one = await m.realInput("finance_aml", () => (i + 0.5) / n);
  const fed = one.bySpace[w.input_space];
  const r = m.qcnnInferTrained(fed.slice(0, w.n_qubits), w);
  res.push({ i, raw: one.raw, fed, expectation: r.expectation, cls: r.cls, note: one.qcnnRowNote });
}
console.log(JSON.stringify(res));
"""


def test_the_page_feeds_the_qcnn_the_transactions_own_eight_inputs_in_its_order() -> None:
    _require_node_or_skip()
    w, student = _w(SERVED), json.loads(STUDENT.read_text())
    cols = [student["feature_names"].index(n) for n in w["feature_names"]]
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "input.mjs"
        script.write_text(_INPUT_HARNESS)
        proc = subprocess.run(["node", str(script), str(WEB), "16"], cwd=WEB, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(got) == len(student["samples"]) // 16
    raw = np.asarray([student["samples"][g["i"]] for g in got], dtype=float)
    fed = np.asarray([g["fed"] for g in got], dtype=float)
    np.testing.assert_array_equal(fed, raw[:, cols])          # the transaction's own inputs, all eight, in order
    assert all("the same transaction at its eight inputs" in g["note"] for g in got)
    want = _trainer_readout(w, fed)
    browser = np.asarray([g["expectation"] for g in got])
    assert float(np.abs(browser - want).max()) <= 1e-5
    t_cls = (want * float(w["scale"]) + float(w["bias"]) > 0).astype(int)
    assert [g["cls"] for g in got] == t_cls.tolist()
    # The order is load-bearing, so a feed in another order is something this check would see.
    for perm in ([1, 0, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 7, 6], list(range(7, -1, -1))):
        moved = _trainer_readout(w, fed[:, perm])
        assert float(np.abs(moved - want).max()) > 1e-2, perm


_PIPELINE_HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const NARROW = process.argv[3] === "narrow";
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r86-qcnnfin-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};`,
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
  if (NARROW && p.replace(/^\/+/, "") === "data/finance/student_heldout.json") {
    // One of the circuit's eight inputs renamed away: the row the page builds would lack it.
    const j = JSON.parse(text);
    j.feature_names = j.feature_names.map((n) => (n === "src_out_mean_360" ? "src_out_mean_360_absent" : n));
    text = JSON.stringify(j);
  }
  return { ok: true, status: 200, headers: { get: () => "application/json" },
           json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const logs = [], metrics = {}, seen = [];
await new Promise((done) => {
  let ctl = null, finished = false;
  const finish = () => { if (!finished) { finished = true; clearTimeout(timer); ctl?.stop?.(); done(); } };
  const timer = setTimeout(finish, 300_000);
  ctl = m.runClientPipeline("finance_aml", { seed: 0, inference: "qcnn" }, {
    onEvent: (e) => {
      if (finished) return;
      if (e.stage && !seen.includes(e.stage)) {
        if (seen.length && seen[seen.length - 1] === "private_inference") { finish(); return; }
        seen.push(e.stage);
      }
      if (e.type === "log") logs.push([e.stage, String(e.message ?? "")]);
      if (e.type === "metric") (metrics[e.stage] ??= {})[e.data.name] = e.data.value;
    },
    onEnd: finish,
  });
});
console.log(JSON.stringify({ seen, logs, metrics }));
process.exit(0);
"""


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    """The finance page through `private_inference` with the QCNN selected, on the shipped files and on a student file
    that lacks one of the circuit's inputs. One node process each: `realData` caches a payload for the tab's lifetime."""
    _require_node_or_skip()
    script = tmp_path_factory.mktemp("r86qcnnfin") / "pipeline.mjs"
    script.write_text(_PIPELINE_HARNESS)
    out = {}
    for mode in ("shipped", "narrow"):
        proc = subprocess.run(["node", str(script), str(WEB), mode], cwd=WEB, capture_output=True,
                              text=True, timeout=900)
        assert proc.returncode == 0, f"the browser pipeline did not run ({mode}):\n{proc.stderr[-3000:]}"
        out[mode] = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


def test_the_page_runs_the_eight_wire_circuit_and_says_which_figures_are_whose(pipeline) -> None:
    run = pipeline["shipped"]
    assert "private_inference" in run["seen"], run["seen"]
    m = run["metrics"]["private_inference"]
    assert m["primary_inference_backend"] == "qcnn-trained" and m["qcnn_qubits"] == 8, m
    held = json.loads(SAMPLE.read_text())
    assert m["qcnn_rescored_rows"] == len(held["rows"]) and m["qcnn_rescore_matches_record"] == 1
    line = next(x for s, x in run["logs"] if x.startswith("inference backend = QCNN"))
    w = _w(SERVED)
    assert f"{len(w['theta'])} trained angles" in line and "over 8 wires" in line, line
    # The figure beside the comparators is the whole test window's; the sample's re-score is held to its own record.
    assert f"Held-out minority-class F1 {w['test_minority_f1']:.4f}" in line, line
    assert f"logistic regression at {w['classical_lr_minority_f1']:.4f}" in line, line
    assert f"the page's sample of {len(held['rows'])} of that split's rows" in line, line
    assert f"{held['expected']['test_minority_f1']:.4f}, the value recorded for these rows" in line, line
    assert "for seed 6, the best of 20 on validation average precision" in line, line


def test_a_transaction_that_lacks_one_of_its_inputs_is_refused_not_filled(pipeline) -> None:
    run = pipeline["narrow"]
    assert "private_inference" in run["seen"], run["seen"]
    m = run["metrics"]["private_inference"]
    assert m["primary_inference_backend"] == "ckks", "the QCNN ran on a row missing one of the inputs it names"
    assert "qcnn_qubits" not in m
    logs = [x for s, x in run["logs"] if s == "private_inference"]
    assert any("ships no rows in the space the trained QCNN declares (raw)" in x for x in logs), logs


# --------------------------------------------------------------------------- the card
def test_the_card_states_the_figures_the_weights_file_records() -> None:
    w = _w()
    text = " ".join(CARD.read_text().split())
    for key in ("test_minority_f1", "test_auprc", "test_accuracy", "classical_lr_minority_f1", "classical_lr_auprc",
                "majority_baseline", "all_positive_f1"):
        assert f"| {key} | {w[key]:.6f} |" in CARD.read_text(), key
    sur = w["classical_surrogate"]
    assert sur["converged"] and sur["threads"] == 1
    assert f"minority-class F1 {sur['test_minority_f1']:.6f} and average precision {sur['test_auprc']:.6f}" in text
    assert "**Seed position:**" in CARD.read_text() and "NOT PUBLISHED TO HUGGING FACE" in text
    assert f"{len(w['theta'])} trained angles" in text and "Nothing here is a claim of quantum advantage" in text
