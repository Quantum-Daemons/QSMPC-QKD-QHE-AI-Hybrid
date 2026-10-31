"""Round 84 (qcnnfin): the finance QCNN reads the four columns the finance linear model reads.

WHY THIS EXISTS

Round 84 refit the bundled PaySim linear model on four of the sample's eight columns
(`registry.PAYSIM_MODEL_FEATURES`), because two of the eight, `device_risk` and `geo_mismatch`, are
computed from PaySim's balance columns, which the dataset's description says must not be used for
fraud detection: fraudulent transactions are cancelled, so the balances record the label. The finance
QCNN artifact still learned from all eight, and its published figures (minority-class F1 0.735294
against logistic regression's 0.865672) were measured with those two. DATA_LICENSES.md states that
since its Round 84 retrain the artifact reads the four columns; these tests make that true and keep
it true, in the trainer, the artifact, the served copy, the page's payload and the page itself.

WHAT IS ASSERTED, AND WHAT EACH FAILS ON AT THE BASE COMMIT (aa293ab)

- The trainer's finance loader reads exactly the registry's four columns. Base: all eight.
- The trained artifact and the copy the page serves read those four at four wires, with that
  circuit's parameter count and the four columns' own training moments. Base: eight at eight wires.
- Every artifact's recorded command writes the directory it ships from (`artifact_dir`), so the
  four-wire finance run lands in `models/qcnn_finance`. Base: no `artifact_dir`; the expression it
  replaced would have written `models/qcnn_finance_q4`.
- The page's payload ships the artifact's four columns, in its order, as the TSV's own rows. Base:
  an eight-wide raw block beside an eight-feature artifact.
- The browser, handed each of the 400 bundled transactions the way the page selects it, computes the
  trainer's readout and class on every one, and permuting the columns would change classes, so the
  check sees order. Base: the artifact reads eight columns.
- A bundled raw row of another width is refused rather than sliced to its first columns. Base: the
  page sliced it and ran the circuit.
- No parameter of the shipped four-wire circuit is dead. Base: the shipped circuit is eight wires.
- The numpy path gives the trainer's class on every held-out row: a guard for the card's sentence,
  which passes at the base on the eight-column artifact as well.
- The card's paired comparison against logistic regression is what the shipped weights give on the
  trainer's own split. Base: the card describes the eight-column circuit.
- (slow) The recorded command regenerates the artifact, and the five-seed figures on the card are
  what the same command gives at seeds 0 to 4.
"""

from __future__ import annotations

import json
import math
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
PAYLOAD = PUBLIC / "finance" / "paysim_sample.json"
TSV = REPO / "data" / "finance" / "paysim_sample.tsv"
CARD = REPO / "models" / "qcnn_finance" / "README.md"
sys.path.insert(0, str(REPO / "scripts" / "train"))
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")

#: Written out, so each check reads the ARTIFACT against a fixed list as well as against the registry.
FOUR = ["amount", "cross_border", "night_tx", "new_payee"]
LEAKY = {"device_risk", "geo_mismatch"}


def _q():
    import qcnn_variational as q

    return q


def _w(path: Path = TRAINED) -> dict:
    return json.loads(path.read_text())


def _tsv() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [ln for ln in TSV.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    header = rows[0].split("\t")
    data = np.asarray([[float(v) for v in ln.split("\t")] for ln in rows[1:]])
    return data, data[:, header.index("isFraud")].astype(int), header


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


# --------------------------------------------------------------------------- the trainer
def test_the_trainer_reads_the_four_registry_columns() -> None:
    from qsh.models.registry import PAYSIM_MODEL_FEATURES

    X, y, names, prov = _q().load("finance")
    assert names == FOUR == list(PAYSIM_MODEL_FEATURES), names
    assert not LEAKY & set(names)
    data, label, header = _tsv()
    assert X.shape == (len(data), 4) == (4000, 4)
    np.testing.assert_array_equal(X, data[:, [header.index(n) for n in FOUR]])
    np.testing.assert_array_equal(y, label)
    assert "four columns" in prov and "8 AML features" not in prov, prov


def test_every_recorded_command_writes_the_directory_it_ships_from() -> None:
    """The four-wire finance run must land in `models/qcnn_finance`, where the page's copy comes
    from, and no other artifact's command may move: medical and telecom at 8, telecom also at 7."""
    q = _q()
    assert q.SHIPPED_WIDTH["finance"] == 4
    seen = 0
    for path in sorted((REPO / "models").glob("qcnn_*/weights.json")):
        cmd = _w(path)["training"]["command"]
        usecase = re.search(r"--usecase (\w+)", cmd).group(1)
        width = int(re.search(r"--qubits (\d+)", cmd).group(1))
        assert width == _w(path)["n_qubits"], f"{path.parent.name}: the command trains {width} wires"
        assert q.artifact_dir(usecase, width) == path.parent, (
            f"{path.parent.name}: its recorded command writes {q.artifact_dir(usecase, width)}")
        seen += 1
    assert seen >= 4
    # A run at another width cannot overwrite the shipped one.
    assert q.artifact_dir("finance", 8).name == "qcnn_finance_q8"
    assert q.artifact_dir("telecom", 7).name == "qcnn_telecom_q7"


def test_the_trainer_defaults_finance_to_its_shipped_width(tmp_path: Path) -> None:
    """Without `--qubits`, finance trains its four features on four wires, not on N_QUBITS = 8 with
    four unencoded. One epoch, written under tmp_path, never into models/."""
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "train" / "qcnn_variational.py"),
                           "--usecase", "finance", "--epochs", "1", "--out", str(tmp_path)],
                          cwd=REPO, capture_output=True, text=True, timeout=900,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    w = json.loads((tmp_path / "weights.json").read_text())
    assert w["n_qubits"] == w["n_features"] == 4 and "--qubits 4" in w["training"]["command"]
    assert "unencoded" not in proc.stdout, "the default run padded wires the features do not fill"


# --------------------------------------------------------------------------- the artifact
@pytest.mark.parametrize("path", [TRAINED, SERVED], ids=["trained", "served"])
def test_the_artifact_reads_the_four_columns_at_four_wires(path: Path) -> None:
    q, w = _q(), _w(path)
    assert w["feature_names"] == FOUR, w["feature_names"]
    assert w["n_features"] == w["n_qubits"] == len(FOUR) == q.SHIPPED_WIDTH["finance"]
    assert w["input_space"] == "raw" and w["circuit"]["name"] == "tree"
    assert float(w["encoding_scale"]) == 1.0, "the finance scale was decided at c = 1 (the card)"
    assert len(w["theta"]) == q.param_count(4, 0, int(w["layers"]), "tree") == 52
    assert "four columns" in w["provenance"] and not any(n in w["provenance"] for n in LEAKY)
    cmd = w["training"]["command"]
    for flag in ("--usecase finance", "--qubits 4", "--arch tree", "--encoding-scale 1"):
        assert flag in cmd, cmd
    # The moments are the four columns' own, over the trainer's own training split.
    X, y, _, _ = q.load("finance")
    perm, cut, _ = q.shuffled_split(len(y), int(w["training"]["seed"]))
    np.testing.assert_allclose(w["feature_mean"], X[perm][:cut].mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(w["feature_std"], X[perm][:cut].std(0), rtol=0, atol=1e-12)


def test_the_page_declares_the_width_the_artifact_has() -> None:
    src = (WEB / "src" / "lib" / "usePipelineRun.ts").read_text()
    m = re.search(r"QCNN_WIDTHS: Record<string, number\[\]> = \{.*?finance_aml: \[([^\]]*)\]", src, re.S)
    assert m and [int(v) for v in re.findall(r"\d+", m.group(1))] == [_w(SERVED)["n_qubits"]] == [4]


def test_no_parameter_of_the_shipped_four_wire_circuit_is_dead() -> None:
    """The structure test's perturbation, at the layout finance now ships (4 wires, binary, 3 rounds),
    which is not among the five it covers, on random angles and on the shipped ones."""
    q, w = _q(), _w()
    nq, layers = int(w["n_qubits"]), int(w["layers"])
    assert (nq, layers) == (4, 3)
    g = torch.Generator().manual_seed(0)
    x = torch.tanh(torch.randn(8, nq, generator=g) * 0.7)
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


def test_the_numpy_path_gives_the_trainers_class_on_every_held_out_row() -> None:
    """The Python pipeline's implementation (`qcnn_trained.infer`), fed the held-out rows raw as
    `usecases/finance_aml` feeds it, against the trainer's circuit: the card says every class agrees.
    A guard rather than a behaviour test: it holds at the base commit on the eight-column artifact
    too, and is here so the card's sentence about this artifact is checked by code."""
    from qsh.qhe.qcnn_trained import infer

    q, w = _q(), _w()
    X, _, _, _ = q.load("finance")
    perm, _, vcut = q.shuffled_split(len(X), int(w["training"]["seed"]))
    held = X[perm][vcut:]
    want = _trainer_readout(w, held)
    got = np.asarray([infer(list(r), w)[0] for r in held])
    assert float(np.abs(got - want).max()) < 1e-4
    t_cls = (want * float(w["scale"]) + float(w["bias"]) > 0).astype(int)
    assert [infer(list(r), w)[1] for r in held] == t_cls.tolist()


# --------------------------------------------------------------------------- the payload
def test_the_payload_ships_the_artifacts_four_columns_as_the_tsvs_own_rows() -> None:
    d, w = json.loads(PAYLOAD.read_text()), _w(SERVED)
    raw = np.asarray(d["samples_by_space"]["raw"], dtype=float)
    assert d["names"] == w["feature_names"] == FOUR
    assert raw.shape == (len(d["labels"]), w["n_features"]) == (400, 4)
    data, label, header = _tsv()
    tsv4 = np.round(data[:, [header.index(n) for n in FOUR]], 5)
    labels_of: dict[tuple, set[int]] = {}
    for r, lab in zip(tsv4.tolist(), label.tolist(), strict=True):
        labels_of.setdefault(tuple(r), set()).add(int(lab))
    missing = [i for i, r in enumerate(raw.tolist()) if tuple(r) not in labels_of]
    assert not missing, f"{len(missing)} raw rows are not rows of the TSV's four columns"
    wrong = [i for i, r in enumerate(raw.tolist()) if d["labels"][i] not in labels_of[tuple(r)]]
    assert not wrong, f"{len(wrong)} raw rows carry a label no TSV row with those values has"


# --------------------------------------------------------------------------- the page
_INPUT_HARNESS = r"""
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-qcnnfin-"));
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
const res = [];
for (let i = 0; i < 400; i++) {
  // The page's own selection (clientCompute, private_inference): the row in the space the artifact
  // declares, sliced to its wires. `u` lands on row i exactly: floor(((i + 0.5) / 400) * 400) = i.
  const one = await m.realInput("finance_aml", () => (i + 0.5) / 400);
  const fed = one.bySpace[w.input_space];
  const r = m.qcnnInferTrained(fed.slice(0, w.n_qubits), w);
  res.push({ fed, expectation: r.expectation, cls: r.cls });
}
console.log(JSON.stringify(res));
"""


def test_the_page_feeds_the_qcnn_its_four_columns_in_its_order() -> None:
    _require_node_or_skip()
    d, w = json.loads(PAYLOAD.read_text()), _w(SERVED)
    # The contract the positional feed rests on: the payload's columns are the artifact's, in order.
    assert d["names"] == w["feature_names"] == FOUR, (d["names"], w["feature_names"])
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "input.mjs"
        script.write_text(_INPUT_HARNESS)
        proc = subprocess.run(["node", str(script), str(WEB)], cwd=WEB, capture_output=True,
                              text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    raw = np.asarray(d["samples_by_space"]["raw"], dtype=float)
    fed = np.asarray([g["fed"] for g in got], dtype=float)
    np.testing.assert_array_equal(fed, raw)          # row i is row i, all four columns, in order
    want = _trainer_readout(w, raw)
    browser = np.asarray([g["expectation"] for g in got])
    assert float(np.abs(browser - want).max()) <= 1e-5
    t_cls = (want * float(w["scale"]) + float(w["bias"]) > 0).astype(int)
    assert [g["cls"] for g in got] == t_cls.tolist()
    # The order is load-bearing, so a feed in another order is something this check would see.
    for perm in ([1, 0, 2, 3], [0, 1, 3, 2], [3, 2, 1, 0]):
        moved = _trainer_readout(w, raw[:, perm])
        assert float(np.abs(moved - want).max()) > 1e-2, perm


_PIPELINE_HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const WIDEN = process.argv[3] === "widen";
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-qcnnfin-"));
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
  if (WIDEN && p.replace(/^\/+/, "") === "data/finance/paysim_sample.json") {
    // The eight-column layout the payload shipped while the artifact read eight: amount,
    // velocity_24h, cross_border, night_tx, new_payee, amount_zscore, device_risk, geo_mismatch.
    const j = JSON.parse(text);
    j.samples_by_space.raw = j.samples_by_space.raw.map(([a, cb, nt, np]) => [a, 0, cb, nt, np, 0, 0, 0]);
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
    """The finance page through `private_inference` with the QCNN selected, on the shipped payload and
    on one whose raw block is widened to the old eight-column layout. One node process each:
    `realData` caches a payload for the tab's lifetime."""
    _require_node_or_skip()
    script = tmp_path_factory.mktemp("r84qcnnfin") / "pipeline.mjs"
    script.write_text(_PIPELINE_HARNESS)
    out = {}
    for mode in ("shipped", "widen"):
        proc = subprocess.run(["node", str(script), str(WEB), mode], cwd=WEB, capture_output=True,
                              text=True, timeout=900)
        assert proc.returncode == 0, f"the browser pipeline did not run ({mode}):\n{proc.stderr[-3000:]}"
        out[mode] = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


def test_the_page_runs_the_four_wire_circuit_on_the_shipped_payload(pipeline) -> None:
    run = pipeline["shipped"]
    assert "private_inference" in run["seen"], run["seen"]
    m = run["metrics"]["private_inference"]
    assert m["primary_inference_backend"] == "qcnn-trained" and m["qcnn_qubits"] == 4, m
    line = next(x for s, x in run["logs"] if x.startswith("inference backend = QCNN"))
    w = _w(SERVED)
    assert f"{len(w['theta'])} trained angles" in line and "over 4 wires" in line, line
    assert f"Held-out minority-class F1 {w['test_minority_f1']:.4f}" in line, line
    assert "the classical model is ahead" in line, line


def test_a_raw_row_of_another_width_is_refused_not_sliced(pipeline) -> None:
    run = pipeline["widen"]
    assert "private_inference" in run["seen"], run["seen"]
    m = run["metrics"]["private_inference"]
    assert m["primary_inference_backend"] == "ckks", (
        "the QCNN ran on an eight-value raw row: sliced to its four wires, that feeds it amount, "
        "velocity_24h, cross_border and night_tx, not the four columns it names")
    assert "qcnn_qubits" not in m
    logs = [x for s, x in run["logs"] if s == "private_inference"]
    assert any("raw row has 8 values and the trained QCNN reads 4" in x for x in logs), logs


# --------------------------------------------------------------------------- the card
def _paired(w: dict) -> dict:
    """The trainer's own split and comparator, and the shipped weights' predictions on its test rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    q = _q()
    X, y, _, _ = q.load("finance")
    perm, cut, vcut = q.shuffled_split(len(y), int(w["training"]["seed"]))
    X, y = X[perm], y[perm]
    mu, sd = X[:cut].mean(0), X[:cut].std(0)
    Xs = np.tanh((X - mu) / np.where(sd == 0, 1.0, sd))
    lr = LogisticRegression(max_iter=2000).fit(Xs[:cut], y[:cut])
    yt, lr_pred = y[vcut:], lr.predict(Xs[vcut:])
    qc_pred = (_trainer_readout(w, X[vcut:]) * float(w["scale"]) + float(w["bias"]) > 0).astype(int)
    # The recomputation is the recorded one, or the counts below describe some other model.
    assert round(float(f1_score(yt, qc_pred)), 6) == w["test_minority_f1"]
    assert round(float(f1_score(yt, lr_pred)), 6) == w["classical_lr_minority_f1"]
    b = int(((lr_pred == yt) & (qc_pred != yt)).sum())
    c = int(((lr_pred != yt) & (qc_pred == yt)).sum())
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return {"rows": len(yt), "fraud": int(yt.sum()), "qc_tp": int((qc_pred & yt).sum()),
            "qc_fp": int((qc_pred & (1 - yt)).sum()), "lr_tp": int((lr_pred & yt).sum()),
            "lr_fp": int((lr_pred & (1 - yt)).sum()), "b": b, "c": c, "p": p}


def test_the_card_states_the_paired_comparison_the_shipped_weights_give() -> None:
    pytest.importorskip("sklearn")
    r = _paired(_w())
    text = " ".join(CARD.read_text().split())
    qc_fp = "no legitimate row" if r["qc_fp"] == 0 else f"{r['qc_fp']} legitimate rows"
    for phrase in (
        f"Of the {r['fraud']} fraud rows among the {r['rows']}, the circuit flags {r['qc_tp']} and "
        f"logistic regression {r['lr_tp']}; the circuit flags {qc_fp} and logistic regression "
        f"{r['lr_fp']}.",
        f"The two disagree on {r['b'] + r['c']} of the {r['rows']} rows, logistic regression right on "
        f"{r['b']} of them and the circuit on {r['c']}, so the exact McNemar test on the discordant "
        f"rows gives p = {r['p']:.5f}",
    ):
        assert phrase in text, f"the card does not say: {phrase}"
    assert "**At this seed logistic regression is ahead" in text
    assert r["b"] > r["c"] and r["p"] > 0.05, "the card says the split does not separate them"


# --------------------------------------------------------------------------- slow: regeneration
def _train(tmp: Path, seed: int) -> dict:
    args = _w()["training"]["command"].split()
    assert args[:2] == ["python", "scripts/train/qcnn_variational.py"], args
    args = args[2:]
    args[args.index("--seed") + 1] = str(seed)
    out = tmp / f"seed{seed}"
    subprocess.run([sys.executable, str(REPO / "scripts" / "train" / "qcnn_variational.py"), *args,
                    "--out", str(out)], cwd=REPO, check=True, capture_output=True, text=True,
                   env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=3600)
    return json.loads((out / "weights.json").read_text())


@pytest.mark.slow
def test_the_recorded_command_regenerates_the_artifact_and_the_cards_five_seeds(tmp_path) -> None:
    from scipy import stats

    shipped = _w()
    runs = [_train(tmp_path, s) for s in range(5)]
    on_cuda = runs[0]["training"]["device"] == shipped["training"]["device"] == "cuda"
    if on_cuda:
        assert (tmp_path / "seed0" / "weights.json").read_bytes() == TRAINED.read_bytes()
    for k in ("test_minority_f1", "classical_lr_minority_f1", "test_accuracy", "classical_lr_accuracy"):
        assert runs[0][k] == shipped[k], k
    qc = [r["test_minority_f1"] for r in runs]
    lr = [r["classical_lr_minority_f1"] for r in runs]
    text = CARD.read_text()
    row = next(ln for ln in text.splitlines() if ln.startswith("| **tree, four columns"))
    lr_row = next(ln for ln in text.splitlines() if ln.startswith("| logistic regression, the same split"))
    num = lambda line: [float(v) for v in re.findall(r"\d\.\d{4}", line)]   # noqa: E731
    assert num(row) == [round(v, 4) for v in qc] + [round(float(np.mean(qc)), 4)], row
    assert num(lr_row) == [round(v, 4) for v in lr] + [round(float(np.mean(lr)), 4)], lr_row
    flat = " ".join(text.split())
    diffs = np.asarray(qc) - np.asarray(lr)
    t = stats.ttest_rel(qc, lr)
    for phrase in (
        f"sample sd {np.std(qc, ddof=1):.4f}",
        f"+{diffs.mean():.4f} minority-class F1 on average",
        f"spread by {np.std(diffs, ddof=1):.4f}",
        f"paired t(4) = {t.statistic:.2f}, p = {t.pvalue:.2f}",
        f"Wilcoxon signed-rank p is {stats.wilcoxon(qc, lr).pvalue:.2f}",
        f"leads logistic regression on {int((diffs > 0).sum())} of the 5 and trails it on "
        f"{int((diffs < 0).sum())}",
    ):
        assert phrase in flat, f"the card does not say: {phrase}"
    assert qc[0] == min(qc), "the card calls seed 0 the lowest of the five"
