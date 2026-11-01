"""Round 84: a truncated or malformed QCNN artifact fails loudly, on the binary path too.

WHY THIS EXISTS

Checklist rows QC28 and QC35. `qcnnInferTrained` checked the parameter count inside its MULTICLASS
branch only. A binary artifact (finance, medical) with a short `theta` indexed past the end of it,
`undefined` turned into NaN through cos and sin, `NaN > 0` is false, and the page reported class 0
with `<Z0>` = NaN - measured on the deploy bundle with 75 of finance's 77 angles. Class 0 is "not
fraud" on finance and "benign" on medical, so a stale file labelled every row the reassuring answer
without a word.

Every malformation below used to reach a readout in at least one language. Each must now be
refused before anything is simulated, with a message that names what is wrong: in the browser by a
thrown Error, in `qcnn_trained.py` by ValueError, and in the trainer's own unpacking.

Also here: the retrained artifacts record what reproduces them (decision of 2026-10-30): device,
seed, the command with every weight-changing flag, the encoding scale and the metrics.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
PUBLIC = WEB / "public" / "data"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
sys.path.insert(0, str(REPO / "scripts" / "train"))
sys.path.insert(0, str(REPO / "src"))

BINARY = ["finance/qcnn_weights.json", "medical/qcnn_weights.json"]


def _load(rel: str) -> dict:
    return json.loads((PUBLIC / rel).read_text())


def _malformed(w: dict) -> dict[str, dict]:
    """Named corruptions of a well-formed binary artifact."""
    out = {}
    short = copy.deepcopy(w)
    short["theta"] = w["theta"][:-2]
    out["theta two short"] = short
    long_ = copy.deepcopy(w)
    long_["theta"] = w["theta"] + [0.1]
    out["theta one long"] = long_
    nan = copy.deepcopy(w)
    nan["theta"][5] = None                  # what a NaN becomes in JSON
    out["theta with a null"] = nan
    arr = copy.deepcopy(w)
    arr["scale"] = [arr["scale"]]
    out["binary scale as an array"] = arr
    circ = copy.deepcopy(w)
    circ["circuit"] = {"name": "mera"}
    out["unknown circuit"] = circ
    enc = copy.deepcopy(w)
    enc["encoding_scale"] = 2.0
    out["encoding scale above 1"] = enc
    mom = copy.deepcopy(w)
    mom["feature_std"] = mom["feature_std"][:3]
    out["raw input with short moments"] = mom
    return out


def _browser(cases: list[dict]) -> list[dict]:
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
            "console.log(JSON.stringify(cases.map(({ w, row }) => {\n"
            "  try { const o = m.qcnnInferTrained(row, w);\n"
            "        return { cls: o.cls, z: Number.isFinite(o.expectation) ? o.expectation : String(o.expectation) }; }\n"
            "  catch (e) { return { error: String((e && e.message) || e) }; }\n"
            "})));\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


@pytest.mark.parametrize("rel", BINARY)
def test_the_browser_refuses_every_malformed_binary_artifact(rel: str) -> None:
    w = _load(rel)
    row = [float(v) for v in w["feature_mean"]]
    bad = _malformed(w)
    res = _browser([{"w": w, "row": row}] + [{"w": b, "row": row} for b in bad.values()])
    assert "error" not in res[0], f"{rel}: the well-formed artifact was refused: {res[0]['error']}"
    silent = {name: r for name, r in zip(bad, res[1:], strict=True) if "error" not in r}
    assert not silent, (
        f"{rel}: these malformed artifacts produced a class instead of an error: {silent}. A short "
        "theta used to come back as class 0 with <Z0> = NaN, which on this page is the reassuring "
        "answer.")
    short = res[1]["error"]
    assert "malformed" in short and f"theta carries {len(w['theta']) - 2} angles" in short, short
    assert f"needs {len(w['theta'])}" in short, short


def test_the_browser_refuses_a_multiclass_artifact_with_a_scalar_head() -> None:
    """The other half of the old branch test: m > 0 with scalar scale/bias used to fall through to
    the binary readout and answer an eight-class question with two classes."""
    w = _load("telecom/qcnn_weights.json")
    bad = copy.deepcopy(w)
    bad["scale"], bad["bias"] = 4.0, 0.0
    short = copy.deepcopy(w)
    short["theta"] = w["theta"][:-1]
    row = [float(v) for v in w["feature_mean"]]
    res = _browser([{"w": w, "row": row}, {"w": bad, "row": row}, {"w": short, "row": row}])
    assert "error" not in res[0], res[0]
    assert "error" in res[1] and "readout needs 8" in res[1]["error"], res[1]
    assert "error" in res[2] and "theta carries 99 angles" in res[2]["error"], res[2]


@pytest.mark.parametrize("rel", BINARY)
def test_the_numpy_path_refuses_every_malformed_binary_artifact(rel: str) -> None:
    from qsh.qhe.qcnn_trained import infer

    w = _load(rel)
    row = [float(v) for v in w["feature_mean"]]
    infer(row, w)                                        # the well-formed one runs
    for name, b in _malformed(w).items():
        with pytest.raises(ValueError, match="QCNN artifact is malformed") as e:
            infer(row, b)
        if name == "theta two short":
            assert f"theta carries {len(w['theta']) - 2} angles" in str(e.value)


def test_the_numpy_path_refuses_a_circuit_it_does_not_implement() -> None:
    """`qcnn_trained.py` implements the tree only; an SU(4) artifact must not run through it."""
    from qsh.qhe.qcnn_trained import infer

    w = _load("telecom/qcnn_weights.json")
    with pytest.raises(ValueError, match="not implemented here"):
        infer([float(v) for v in w["feature_mean"]], w)


def test_the_trainer_refuses_a_theta_of_the_wrong_length() -> None:
    torch = pytest.importorskip("torch")
    import qcnn_variational as q

    for arch, nq, ro, layers in (("tree", 8, 0, 3), ("hur-su4", 8, 3, 6)):
        n = q.param_count(nq, ro, layers, arch)
        x = torch.zeros(2, nq)
        q.circuit(x, torch.zeros(n), nq, readout_qubits=ro, layers=layers, arch=arch)
        for wrong in (n - 2, n + 1):
            with pytest.raises(ValueError, match=f"theta carries {wrong} angles"):
                q.circuit(x, torch.zeros(wrong), nq, readout_qubits=ro, layers=layers, arch=arch)


# --------------------------------------------------------------------------- what reproduces them
RETRAINED = ["qcnn_medical", "qcnn_telecom", "qcnn_telecom_q7"]


@pytest.mark.parametrize("name", RETRAINED)
def test_a_retrained_artifact_records_what_reproduces_it(name: str) -> None:
    w = json.loads((REPO / "models" / name / "weights.json").read_text())
    t = w["training"]
    for key in ("device", "device_name", "torch", "seed", "command", "encoding_scale", "best_epoch",
                "best_val_score"):
        assert t.get(key) is not None, f"{name}: training.{key} is not recorded"
    assert t["device"] == "cuda", (
        f"{name} was produced on {t['device']}; the reproduction claim in training.note is made for "
        "CUDA, where the multiclass artifacts regenerate bit for bit and a CPU run does not")
    assert t["encoding_scale"] == w["encoding_scale"]
    cmd = t["command"]
    for flag in (f"--seed {t['seed']}", f"--layers {w['layers']}", f"--epochs {t['epochs']}",
                 f"--qubits {w['n_qubits']}", f"--arch {w['circuit']['name']}",
                 f"--encoding-scale {w['encoding_scale']:g}"):
        assert flag in cmd, f"{name}: the recorded command lacks `{flag}`: {cmd}"
    assert "--out" not in cmd, f"{name}: a recorded command must write the artifact where it ships"
    head = {"accuracy": "test_accuracy", "macro_f1": "test_macro_f1",
            "minority_class_f1": "test_minority_f1"}[w["headline_metric"]]
    assert isinstance(w[head], float) and isinstance(w["classical_lr_accuracy"], float)
