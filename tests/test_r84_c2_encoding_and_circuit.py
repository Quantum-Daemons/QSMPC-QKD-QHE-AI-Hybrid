"""Round 84: the encoding scale and the SU(4) telecom circuit, in every language that runs them.

WHY THIS EXISTS

Two pre-registered experiments (2026-10-30) changed what the QCNN artifacts are. The half-angle one
moved medical and both telecom widths to RY(c * pi * tanh(z)) with c = 0.5, because at c = 1 the two
tails of every feature fold onto one state; the overlap one rebuilt telecom with Hur, Kim and Park's
SU(4) block on every nearest-neighbour bond, because on disjoint pairs each survivor saw only its own
feature pair. Each change lives in three places - the trainer, `quantumSim.ts` and the artifact - and
"one change, a subset of the places" is the defect this project has shipped most often. So:

  * the artifacts DECLARE `encoding_scale` and `circuit`, and the retrained ones carry the values the
    experiments chose (finance stays the tree at c = 1, unchanged);
  * the browser reproduces the trainer to 1e-5 at c != 1 and on the SU(4) circuit, on synthetic
    weights (so the check does not depend on what ships) and on every shipped artifact's held-out
    rows (so it does);
  * the numpy path applies the scale too;
  * the shipped telecom circuits are NOT trees: their survivors are entangled, where a tree's are a
    product state - the property that made the old circuit classically simulable in linear time, and
    the reason every label calling this one "a tree" had to change.

Every assertion here fails on the Round 82 tree: no artifact declared a scale or a circuit, the
browser ignored both, and the trainer had no `arch`.
"""
from __future__ import annotations

import json
import os
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
sys.path.insert(0, str(REPO / "scripts" / "train"))
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")

#: What the two experiments chose, per shipped artifact: (circuit, encoding scale). Finance is the
#: control - its c = 0.75 arm failed a confirmatory run on fresh seeds, so it keeps the tree at 1.
CHOSEN = {
    "finance/qcnn_weights.json": ("tree", 1.0),
    "medical/qcnn_weights.json": ("tree", 0.5),
    "telecom/qcnn_weights.json": ("hur-su4", 0.5),
    "telecom/qcnn_weights_q7.json": ("hur-su4", 0.5),
}


def _q():
    import qcnn_variational as q

    return q


def _arch(w: dict) -> str:
    return str((w.get("circuit") or {}).get("name", "tree"))


def _browser(cases: list[dict]) -> list[dict]:
    """Run the browser's own `qcnnInferTrained` (esbuild + node) on each {w, rows} case."""
    if not ESBUILD.exists():
        msg = "web/node_modules/.bin/esbuild is absent; run `npm ci` in web/. A SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 makes it a failure.")
        pytest.skip(msg)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, data, runner = d / "qsim.mjs", d / "cases.json", d / "run.mjs"
        subprocess.run([str(ESBUILD), str(WEB / "src" / "lib" / "quantumSim.ts"), "--bundle",
                        "--format=esm", "--platform=node", f"--outfile={bundle}", "--log-level=error"],
                       cwd=WEB, check=True, capture_output=True)
        data.write_text(json.dumps(cases))
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            "import { readFileSync } from 'node:fs';\n"
            f"const cases = JSON.parse(readFileSync({json.dumps(str(data))}, 'utf8'));\n"
            "console.log(JSON.stringify(cases.map(({ w, rows }) => {\n"
            "  try { return { ok: rows.map((r) => { const o = m.qcnnInferTrained(r, w);\n"
            "    return { cls: o.cls, probs: o.probs ?? [o.expectation] }; }) }; }\n"
            "  catch (e) { return { error: String((e && e.message) || e) }; }\n"
            "})));\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def _synthetic(nq: int, ro: int, layers: int, arch: str, c: float, seed: int) -> dict:
    """A well-formed artifact with random angles, so the cross-language check covers a circuit and a
    scale whether or not anything ships with them. Raw input with unit moments: z = x."""
    q = _q()
    g = torch.Generator().manual_seed(seed)
    n = q.param_count(nq, ro, layers, arch)
    k = 1 << ro if ro else 1
    w = {"n_qubits": nq, "layers": layers, "n_features": nq, "input_space": "raw",
         "encoding": "tanh", "encoding_scale": c, "circuit": {"name": arch},
         "feature_mean": [0.0] * nq, "feature_std": [1.0] * nq,
         "theta": (torch.randn(n, generator=g) * 0.6).tolist(),
         "test_accuracy": 0, "classical_lr_accuracy": 0, "majority_baseline": 0}
    if ro:
        w["readout"] = {"kind": "basis_marginal", "n_classes": k, "readout_qubits": ro,
                        "wires": q.readout_wires(nq, ro, layers), "bit_order": "lsb_first"}
        w["scale"], w["bias"] = [1.0] * k, [0.0] * k
    else:
        w["readout"] = {"kind": "expval_z0", "n_classes": 2}
        w["scale"], w["bias"] = 3.0, 0.1
    return w


def _trainer_out(w: dict, rows: np.ndarray) -> np.ndarray:
    """The trainer's circuit on rows in the artifact's input space: standardise if raw, tanh, scale."""
    q = _q()
    nq, nf = int(w["n_qubits"]), int(w.get("n_features", w["n_qubits"]))
    x = np.asarray(rows, dtype=float)[:, :nf]
    if w.get("input_space") == "raw":
        x = (x - np.asarray(w["feature_mean"])[:nf]) / np.asarray(w["feature_std"])[:nf]
    x = float(w.get("encoding_scale", 1.0)) * np.tanh(x)
    if nf < nq:
        x = np.hstack([x, np.zeros((len(x), nq - nf))])
    ro = int((w.get("readout") or {}).get("readout_qubits", 0) or 0)
    out = q.circuit(torch.tensor(x, dtype=torch.float32), torch.tensor(w["theta"], dtype=torch.float32),
                    nq, readout_qubits=ro, layers=int(w["layers"]), arch=_arch(w)).detach().numpy()
    return out if ro else out[:, None]


def _held_out(w: dict) -> tuple[np.ndarray, np.ndarray]:
    """The artifact's own held-out rows, in raw units, through the trainer's own split."""
    q = _q()
    X, y, _, _ = q.load(w["usecase"])
    perm, _, vcut = q.shuffled_split(len(y), int(w["training"]["seed"]))
    return X[perm][vcut:], y[perm][vcut:]


# --------------------------------------------------------------------------- the declarations
@pytest.mark.parametrize("rel", sorted(CHOSEN))
def test_each_artifact_declares_the_circuit_and_scale_the_experiments_chose(rel: str) -> None:
    w = json.loads((PUBLIC / rel).read_text())
    want_arch, want_c = CHOSEN[rel]
    assert _arch(w) == want_arch, f"{rel} runs {_arch(w)!r}; the overlap experiment chose {want_arch!r}"
    assert float(w.get("encoding_scale", 1.0)) == want_c, (
        f"{rel} encodes at c = {w.get('encoding_scale', 1.0)}; the half-angle experiment chose {want_c}")
    if want_c != 1.0 or want_arch != "tree":
        # A retrained artifact says so explicitly rather than leaning on a default.
        assert "encoding_scale" in w and "circuit" in w, f"{rel} relies on defaults it does not state"
    q = _q()
    ro = int((w.get("readout") or {}).get("readout_qubits", 0) or 0)
    assert len(w["theta"]) == q.param_count(int(w["n_qubits"]), ro, int(w["layers"]), want_arch)


# --------------------------------------------------------------------------- cross-language
SYNTHETIC = [  # (nq, readout, layers, circuit, c): both shipped telecom layouts, a binary SU(4), a tree
    (8, 3, 6, "hur-su4", 0.5), (7, 3, 6, "hur-su4", 0.5), (8, 0, 2, "hur-su4", 0.3),
    (8, 0, 3, "tree", 0.3), (7, 3, 2, "tree", 0.7),
]


def test_the_browser_reproduces_the_trainer_at_every_scale_and_on_both_circuits() -> None:
    g = np.random.default_rng(0)
    cases, expected = [], []
    for i, (nq, ro, layers, arch, c) in enumerate(SYNTHETIC):
        w = _synthetic(nq, ro, layers, arch, c, seed=i)
        rows = g.normal(0, 1.2, (16, nq))
        cases.append({"w": w, "rows": rows.tolist()})
        expected.append(_trainer_out(w, rows))
    got = _browser(cases)
    for spec, want, res in zip(SYNTHETIC, expected, got, strict=True):
        assert "ok" in res, f"{spec}: the browser refused a well-formed artifact: {res.get('error')}"
        probs = np.array([r["probs"] for r in res["ok"]])
        delta = float(np.abs(probs - want).max())
        assert delta <= 1e-5, (
            f"{spec}: browser and trainer differ by {delta:.2e}. Both read `circuit.name` and "
            "`encoding_scale`; a difference means one of them builds another circuit or another angle.")


def test_that_check_would_notice_a_browser_that_ignored_the_scale_or_the_circuit() -> None:
    """Anti-tautology: the same comparison must reject the two omissions it exists for."""
    nq, ro, layers = 8, 3, 6
    w = _synthetic(nq, ro, layers, "hur-su4", 0.5, seed=0)
    rows = np.random.default_rng(1).normal(0, 1.2, (16, nq))
    right = _trainer_out(w, rows)
    at_c1 = _trainer_out({**w, "encoding_scale": 1.0}, rows)
    assert float(np.abs(right - at_c1).max()) > 1e-2, "c = 0.5 and c = 1 are indistinguishable here"
    q = _q()
    as_tree = {**_synthetic(nq, ro, layers, "tree", 0.5, seed=0), "circuit": {"name": "tree"}}
    as_tree["theta"] = w["theta"][: q.param_count(nq, ro, layers, "tree")]
    assert float(np.abs(right - _trainer_out(as_tree, rows)).max()) > 1e-2


@pytest.mark.parametrize("rel", sorted(CHOSEN))
def test_the_browser_classifies_every_held_out_row_as_the_trainer_does(rel: str) -> None:
    w = json.loads((PUBLIC / rel).read_text())
    if w.get("input_space") != "raw":
        pytest.fail(f"{rel} declares input_space {w.get('input_space')!r}; every artifact is raw "
                    "since Round 84, so the held-out rows can be fed in the units it declares")
    X, _ = _held_out(w)
    want = _trainer_out(w, X)
    res = _browser([{"w": w, "rows": X.tolist()}])[0]
    assert "ok" in res, f"{rel}: the browser refused the shipped artifact: {res.get('error')}"
    probs = np.array([r["probs"] for r in res["ok"]])
    assert float(np.abs(probs - want).max()) <= 1e-5
    sc, bi = np.asarray(w["scale"], dtype=float), np.asarray(w["bias"], dtype=float)
    ro = int((w.get("readout") or {}).get("readout_qubits", 0) or 0)
    t_cls = (want * sc + bi).argmax(1) if ro else ((want[:, 0] * sc + bi) > 0).astype(int)
    b_cls = np.array([r["cls"] for r in res["ok"]])
    assert int((t_cls == b_cls).sum()) == len(X), (
        f"{rel}: {int((t_cls != b_cls).sum())} of {len(X)} held-out rows get a different class in "
        "the browser than in the trainer")


def test_the_numpy_path_applies_the_scale() -> None:
    from qsh.qhe.qcnn_trained import infer

    w = _synthetic(8, 0, 3, "tree", 0.3, seed=7)
    rows = np.random.default_rng(2).normal(0, 1.2, (12, 8))
    want = _trainer_out(w, rows)[:, 0]
    got = np.array([infer(r, w)[0] for r in rows])
    assert float(np.abs(got - want).max()) <= 1e-5, (
        "qcnn_trained.py does not compute the trainer's circuit at c = 0.3; it must multiply the "
        "angle by `encoding_scale` as the trainer and the browser do")


# --------------------------------------------------------------------------- not a tree
def _survivor_trace_distance(w: dict, rows: np.ndarray) -> float:
    """Max over rows of || rho_S - (x)_j rho_j ||_1 / 2 after pooling: zero for a tree, whose
    survivors are a product of one-qubit states, and not zero once neighbouring pairs are coupled."""
    q = _q()
    nq, nf = int(w["n_qubits"]), int(w.get("n_features", w["n_qubits"]))
    x = (np.asarray(rows, dtype=float)[:, :nf] - np.asarray(w["feature_mean"])[:nf]) / np.asarray(
        w["feature_std"])[:nf]
    x = float(w.get("encoding_scale", 1.0)) * np.tanh(x)
    if nf < nq:
        x = np.hstack([x, np.zeros((len(x), nq - nf))])
    ro = int((w.get("readout") or {}).get("readout_qubits", 0) or 0)
    trace: list = []
    q.circuit(torch.tensor(x, dtype=torch.float32), torch.tensor(w["theta"], dtype=torch.float32), nq,
              readout_qubits=ro, layers=int(w["layers"]), arch=_arch(w), trace=trace)
    _, surv, state = trace[-1]
    worst = 0.0
    for psi in state.numpy().astype(np.complex128):
        t = psi.reshape([2] * nq)                      # axis k is wire nq-1-k (wire 0 is the LSB)
        axes = [nq - 1 - s for s in surv]
        rest = [a for a in range(nq) if a not in axes]
        m = np.transpose(t, axes + rest).reshape(2 ** len(surv), -1)
        rho = m @ m.conj().T                           # survivors' reduced state, surv[0] first
        s = len(surv)
        r4 = rho.reshape([2] * (2 * s))                # kets on axes 0..s-1, bras on s..2s-1
        prod = np.ones((1, 1), dtype=complex)
        for j in range(s):
            # survivor j's one-qubit marginal: trace every other survivor out
            idx = list(range(2 * s))
            for k in range(s):
                if k != j:
                    idx[s + k] = idx[k]
            prod = np.kron(prod, np.einsum(r4, idx, [j, s + j]))
        ev = np.linalg.eigvalsh(rho - prod)
        worst = max(worst, 0.5 * float(np.abs(ev).sum()))
    return worst


@pytest.mark.parametrize("rel", ["telecom/qcnn_weights.json", "telecom/qcnn_weights_q7.json"])
def test_the_shipped_telecom_circuits_are_not_trees(rel: str) -> None:
    w = json.loads((PUBLIC / rel).read_text())
    X, _ = _held_out(w)
    d = _survivor_trace_distance(w, X[:24])
    assert d > 0.05, (
        f"{rel}: after pooling the survivors are a product state to {d:.2e} (trace distance to the "
        "product of their marginals), which is what a tree of two-qubit blocks gives. The SU(4) "
        "circuit couples neighbouring pairs before pooling and must not reduce to a tree.")


def test_the_tree_check_does_reduce_the_tree_to_a_product() -> None:
    """The negative control: the same measure on the tree layout is a product state, so the check
    above separates the circuits rather than reporting something every circuit has."""
    w = {**_synthetic(8, 3, 6, "tree", 0.5, seed=3), "usecase": "telecom"}
    rows = np.random.default_rng(4).normal(0, 1.0, (8, 8))
    assert _survivor_trace_distance(w, rows) < 1e-5


def test_no_shipped_parameter_of_the_su4_circuit_is_dead() -> None:
    """The dead-parameter check of the structure test, at the two layouts telecom ships.

    Measured while writing this: both are fully live; the binary SU(4) layout at 8 wires has one
    dead angle (the final U3's RZ on the discarded wire of the last stage, theta[133]), and no
    artifact ships it.
    """
    q = _q()
    for nq in (8, 7):
        n = q.param_count(nq, 3, 6, "hur-su4")
        g = torch.Generator().manual_seed(0)
        theta = torch.randn(n, generator=g) * 0.6
        x = torch.tanh(torch.randn(8, nq, generator=g) * 0.7) * 0.5
        base = q.circuit(x, theta, nq, readout_qubits=3, layers=6, arch="hur-su4").detach()
        dead = []
        for i in range(n):
            b = theta.clone()
            b[i] += 1.234
            out = q.circuit(x, b, nq, readout_qubits=3, layers=6, arch="hur-su4").detach()
            if float((base - out).abs().max()) < 1e-5:
                dead.append(i)
        assert not dead, f"{nq} wires: these SU(4) angles cannot move the readout: {dead}"


def test_no_label_calls_the_telecom_circuit_a_tree_or_disjoint() -> None:
    """Decision 6: labels stay true for the new circuit. The width control's tooltip described the
    telecom circuit as "two-qubit blocks on disjoint pairs"; the artifact describes itself."""
    panel = (WEB / "src" / "components" / "ParameterPanel.tsx").read_text()
    start = panel.index('htmlFor="pp-qubits"')
    tooltip = panel[start:panel.index("QCNN qubits</label>", start)]
    assert "disjoint" not in tooltip and "tree" not in tooltip, (
        "the multi-width tooltip (telecom's) still calls the circuit a tree on disjoint pairs")
    assert "classically simulable at this size" in tooltip
    for rel in ("telecom/qcnn_weights.json", "telecom/qcnn_weights_q7.json"):
        sim = json.loads((PUBLIC / rel).read_text())["circuit"]["simulability"]
        assert sim.startswith("not a tree") and "classically simulable at this size" in sim, rel
