"""Round 84: the medical QCNN standardises with its OWN constants, so it agrees with its trainer.

WHY THIS EXISTS

Checklist rows QC4 and QC38. The medical artifact declared `input_space: "zscore"` and was fed the
bundled sample's z-scores unchanged. Same units as the trainer's - different CONSTANTS: the bundle
standardised with a StandardScaler on the unshuffled first 70 percent of the 569 rows
(`qsh.data.loaders.load_breast_cancer`), the trainer with the moments of its own shuffled training
split. Measured on the Round 82 artifact: 84 of the trainer's 86 held-out rows got the trainer's
class in the browser, 2 did not.

The user's decision (2026-10-30): the browser standardises with the artifact's own feature_mean /
feature_std. The artifact therefore declares `raw`, every caller hands it raw WDBC units, and every
inference path standardises them with the shipped moments - which are the trainer's, so the two
cannot drift whoever rescales their copy of the data.

These tests feed the 86 held-out rows the way the page fed them for the space the artifact declares:
raw units for `raw`, and for `zscore` the bundle's z-scores (the load_breast_cancer scaler, rounded
to 5 decimals as the sample stores them). On the Round 82 tree they fail at 84 / 86; on this one they
pass at 86 / 86 and to 1e-5.

What they do NOT cover is that the bundled sample SHIPS raw rows for the page to feed:
`tests/test_the_encoder_receives_the_space_the_artifact_declares.py` refuses an artifact whose
declared space the sample does not carry, and that is the gate for the data half.
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
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
SHIPPED = WEB / "public" / "data" / "medical" / "qcnn_weights.json"
TRAINED = REPO / "models" / "qcnn_medical" / "weights.json"
sys.path.insert(0, str(REPO / "scripts" / "train"))
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")


def _q():
    import qcnn_variational as q

    return q


def _held_out(w: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train rows, held-out rows, held-out labels) in raw units, through the trainer's split.

    `shuffled_split` is the trainer's own; the inline form is what `main()` did before Round 84
    exposed it, kept so this file runs - and fails for the right reason - on the older tree too.
    """
    q = _q()
    X, y, _, _ = q.load("medical")
    seed = int(w["training"]["seed"])
    if hasattr(q, "shuffled_split"):
        perm, cut, vcut = q.shuffled_split(len(y), seed)
    else:
        perm = np.random.default_rng(seed).permutation(len(y))
        cut, vcut = int(0.7 * len(y)), int(0.85 * len(y))
    return X[perm][:cut], X[perm][vcut:], y[perm][vcut:]


def _as_the_page_supplies(w: dict, X_raw: np.ndarray) -> np.ndarray:
    """The rows in the space the artifact declares, made the way the page made that space."""
    if w["input_space"] == "raw":
        return X_raw
    assert w["input_space"] == "zscore", w["input_space"]
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    d = load_breast_cancer()
    names = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
             "worst radius", "worst concavity", "worst symmetry"]
    idx = [list(d.feature_names).index(c) for c in names]
    scaler = StandardScaler().fit(d.data[: int(len(d.data) * 0.7)])
    return np.round((X_raw - scaler.mean_[idx]) / scaler.scale_[idx], 5)


def _trainer_reference(w: dict, X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(<Z0>, class) from the trainer's circuit on its OWN standardisation of the raw rows."""
    q = _q()
    x = np.tanh((X_raw - np.asarray(w["feature_mean"])) / np.asarray(w["feature_std"]))
    x = float(w.get("encoding_scale", 1.0)) * x
    kw = {} if (w.get("circuit") or {}).get("name", "tree") == "tree" else {"arch": w["circuit"]["name"]}
    z = q.circuit(torch.tensor(x, dtype=torch.float32), torch.tensor(w["theta"], dtype=torch.float32),
                  int(w["n_qubits"]), layers=int(w["layers"]), **kw).detach().numpy()
    return z, (z * float(w["scale"]) + float(w["bias"]) > 0).astype(int)


def _browser(w: dict, rows: np.ndarray) -> list[dict]:
    if not ESBUILD.exists():
        msg = "web/node_modules/.bin/esbuild is absent; run `npm ci` in web/. A SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 makes it a failure.")
        pytest.skip(msg)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, data, runner = d / "qsim.mjs", d / "case.json", d / "run.mjs"
        subprocess.run([str(ESBUILD), str(WEB / "src" / "lib" / "quantumSim.ts"), "--bundle",
                        "--format=esm", "--platform=node", f"--outfile={bundle}", "--log-level=error"],
                       cwd=WEB, check=True, capture_output=True)
        data.write_text(json.dumps({"w": w, "rows": rows.tolist()}))
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            "import { readFileSync } from 'node:fs';\n"
            f"const c = JSON.parse(readFileSync({json.dumps(str(data))}, 'utf8'));\n"
            "console.log(JSON.stringify(c.rows.map((r) => { const o = m.qcnnInferTrained(r, c.w);\n"
            "  return { cls: o.cls, z: o.expectation }; })));\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def test_the_medical_artifact_declares_raw_input_and_the_trainers_own_moments() -> None:
    w = json.loads(SHIPPED.read_text())
    assert w["input_space"] == "raw", (
        f"the medical QCNN declares input_space {w['input_space']!r}. Fed z-scores, it would be "
        "standardised with whatever constants made them; declared raw, it is standardised with its "
        "own, which is the decision recorded for QC4 / QC38")
    train, _, _ = _held_out(w)
    sd = train.std(0)
    assert np.allclose(w["feature_mean"], train.mean(0), rtol=0, atol=1e-9)
    assert np.allclose(w["feature_std"], np.where(sd == 0, 1.0, sd), rtol=0, atol=1e-9), (
        "feature_mean / feature_std are not the trainer's training-split moments")
    assert json.loads(TRAINED.read_text())["input_space"] == w["input_space"]


def test_the_browser_gives_the_trainers_class_on_all_86_held_out_rows() -> None:
    w = json.loads(SHIPPED.read_text())
    _, X, _ = _held_out(w)
    z_ref, cls_ref = _trainer_reference(w, X)
    got = _browser(w, _as_the_page_supplies(w, X))
    cls = np.array([g["cls"] for g in got])
    agree = int((cls == cls_ref).sum())
    assert agree == len(X) == 86, (
        f"the browser gives the trainer's class on {agree} of {len(X)} held-out rows. Round 82 gave "
        "84 of 86: the rows reached the circuit standardised with the bundle's constants rather than "
        "the artifact's.")
    assert float(np.abs(np.array([g["z"] for g in got]) - z_ref).max()) <= 1e-5


def test_the_python_path_gives_the_trainers_class_on_all_86_held_out_rows() -> None:
    from qsh.qhe.qcnn_trained import infer

    w = json.loads(TRAINED.read_text())
    _, X, _ = _held_out(w)
    z_ref, cls_ref = _trainer_reference(w, X)
    out = [infer(r, w) for r in _as_the_page_supplies(w, X)]
    cls = np.array([o[1] for o in out])
    assert int((cls == cls_ref).sum()) == 86, (
        f"qcnn_trained.py gives the trainer's class on {int((cls == cls_ref).sum())} of 86 rows")
    assert float(np.abs(np.array([o[0] for o in out]) - z_ref).max()) <= 1e-5


def test_the_recorded_accuracy_is_what_these_weights_give() -> None:
    """The headline the card and the page quote is recomputed from the shipped weights."""
    w = json.loads(SHIPPED.read_text())
    _, X, y = _held_out(w)
    _, cls = _trainer_reference(w, X)
    assert round(float((cls == y).mean()), 6) == w["test_accuracy"]
