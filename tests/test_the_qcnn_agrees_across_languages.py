"""The trainer and the browser must compute the same QCNN, on the shipped weights.

WHY THIS EXISTS

Round 46 implemented the QCNN in three places - `scripts/train/qcnn_variational.py`, the numpy
inference path, and `web/src/lib/quantumSim.ts` - and asserted only that they had the same
STRUCTURE. Nothing put a number from one beside a number from another. They happened to agree, so
the omission cost nothing and was invisible.

Round 47 changed the fully-connected block from one shared scalar to a per-survivor unitary, in the
trainer, and did not change the browser. That is the fourth instance of this project's most-repeated
defect, committed inside the round named for it. Measured before it was caught: maximum readout
probability difference **0.45**, and the argmax class disagreed on **39 of 64 telecom rows** - the
deployed page would have served confident, wrong labels from weights it was not able to interpret,
while every structural test stayed green and both cards printed the trained metrics.

So this is the numeric check that was missing. It executes the browser's own `qcnnInfer` through
esbuild + node on the artifact the browser actually loads, and compares class probabilities against
the trainer's `circuit()` on the same inputs.

WHY IT MUST BE SEEDED TO BE EVIDENCE

The three implementations agree today, so this file is green on arrival, and a gate that has never
been observed to fail is a gate nobody has checked. `test_the_comparison_would_notice_a_divergence`
perturbs one implementation's parameter layout and asserts the comparison rejects it.

THE PREPROCESSING CONTRACT IS PINNED HERE TOO

Not decoration. Building this comparison, the contract was got wrong twice in ten minutes: once by
feeding raw features to `circuit()` when `infer()` standardises internally (apparent delta 6.3e-01,
with the sign inverted), and once by passing 8 features to a 7-feature model (apparent delta
4.7e-03). Both looked exactly like real divergences. Nothing in the repository stated the contract,
so it is stated below and asserted.
"""

from __future__ import annotations

import json
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

torch = pytest.importorskip("torch")

#: use case directory -> shipped weights file, for every multiclass artifact. The binary artifacts
#: read out a single <Z0> and are covered by the same comparison at `readout_qubits = 0`.
SHIPPED = [
    ("telecom", "qcnn_weights.json"),
    ("telecom", "qcnn_weights_q7.json"),
    ("finance", "qcnn_weights.json"),
    ("medical", "qcnn_weights.json"),
]


def _trainer():
    import qcnn_variational as q

    return q


def _standardise(x, w) -> "torch.Tensor":
    """THE PREPROCESSING CONTRACT, in one place, asserted rather than assumed.

    A caller supplies features in the model's INPUT SPACE, which `w["input_space"]` names. The two
    implementations divide the work differently, and that division is the contract:

      * `input_space == "raw"` - the caller's columns are in original units. The BROWSER
        standardises internally with the shipped `feature_mean` / `feature_std`; the trainer's
        `circuit()` does NOT, so the same columns must be standardised here before they reach it.
      * anything else (medical arrives already StandardScaler-ed) - neither side standardises, and
        applying the shipped moments a second time would be wrong on both.

    THE TANH IS NOT SYMMETRIC BETWEEN THE TWO EITHER, and this is the third form the same mistake
    takes. The browser encodes `RY(tanh(v) * pi)` from the standardised value. The trainer's
    `circuit()` encodes `RY(x * pi)` with NO tanh - it expects input that has already been squashed,
    because `load()` hands it `np.tanh((X - mu) / sd)`. So the trainer's input is
    `tanh(standardise(u))` and the browser's is `u`. `n_features` may also be FEWER than `n_qubits`
    - telecom ships a 7-feature model on 8 wires - and the remaining wires are fed zero, which after
    either encoding leaves them in |0>.

    Getting this wrong does not look like a contract error, it looks like a physics disagreement:
    standardising twice gave an apparent delta of 6.3e-01 with the sign inverted, and supplying 8
    columns to a 7-feature model gave 4.7e-03. Both were mistaken for real divergences.
    """
    if w.get("input_space") != "raw":
        return torch.tanh(x)
    mean = torch.tensor(w["feature_mean"], dtype=torch.float32)
    std = torch.tensor(w["feature_std"], dtype=torch.float32)
    assert mean.shape == std.shape, "feature_mean and feature_std disagree in length"
    assert len(mean) >= int(w["n_features"]), (
        f"the artifact standardises {len(mean)} columns but declares n_features={w['n_features']}"
    )
    n = x.shape[1]
    return torch.tanh((x - mean[:n]) / torch.clamp(std[:n], min=1e-9))


def _browser_probs(rel_dir: str, fname: str, rows: list[list[float]]) -> list[list[float]]:
    """Run the browser's own QCNN on the shipped artifact, through the module graph it ships."""
    if not ESBUILD.exists():
        pytest.skip(
            "web/node_modules/.bin/esbuild is absent, so the browser model cannot be executed. "
            "Run `npm ci` in web/. This is a SKIP rather than a pass: the identity below is "
            "unchecked, not satisfied."
        )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            "export { qcnnInferTrained } from "
            f"{json.dumps(str(WEB / 'src/lib/quantumSim'))};\n")
        proc = subprocess.run(
            [str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
             f"--outfile={bundle}", "--log-level=error"],
            cwd=WEB, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"esbuild could not bundle the browser model: {proc.stderr[:400]}")
        weights = json.loads((PUBLIC / rel_dir / fname).read_text())
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            f"const w = {json.dumps(weights)};\n"
            f"const rows = {json.dumps(rows)};\n"
            "const out = rows.map((r) => {\n"
            "  const res = m.qcnnInferTrained(r, w);\n"
            "  return res.probs ?? [res.expectation];\n"
            "});\n"
            "console.log(JSON.stringify(out));\n")
        res = subprocess.run(["node", str(runner)], capture_output=True, text=True)
        if res.returncode != 0:
            pytest.fail(f"the browser QCNN threw on the shipped weights:\n{res.stderr[:800]}")
        return json.loads(res.stdout)


def _rows_for(w: dict, n: int = 8) -> list[list[float]]:
    """Deterministic inputs in the artifact's own feature space."""
    g = torch.Generator().manual_seed(0)
    return (torch.randn(n, int(w["n_features"]), generator=g) * 0.7).tolist()


@pytest.mark.parametrize("rel_dir,fname", SHIPPED, ids=[f"{d}/{f}" for d, f in SHIPPED])
def test_the_browser_and_the_trainer_compute_the_same_circuit(rel_dir: str, fname: str) -> None:
    """The identity, numerically, on the weights the visitor's browser downloads."""
    path = PUBLIC / rel_dir / fname
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} is not shipped")
    w = json.loads(path.read_text())
    q = _trainer()

    rows = _rows_for(w)
    x = _standardise(torch.tensor(rows, dtype=torch.float32), w)
    nq, layers = int(w["n_qubits"]), int(w["layers"])
    ro = int(w.get("readout", {}).get("readout_qubits", 0) or 0)

    # The pad wires. `n_features` may be fewer than `n_qubits`; the remainder are fed zero, which
    # after the tanh encoding leaves them in |0>.
    if x.shape[1] < nq:
        x = torch.cat([x, torch.zeros(x.shape[0], nq - x.shape[1])], dim=1)

    theta = torch.tensor(w["theta"], dtype=torch.float32)
    assert len(theta) == q.param_count(nq, readout_qubits=ro, layers=layers), (
        f"{path.relative_to(REPO)} ships {len(theta)} angles; the circuit at nq={nq}, "
        f"layers={layers}, readout={ro} needs "
        f"{q.param_count(nq, readout_qubits=ro, layers=layers)}"
    )

    mine = q.circuit(x, theta, nq=nq, readout_qubits=ro, layers=layers).detach()
    theirs = torch.tensor(_browser_probs(rel_dir, fname, rows), dtype=torch.float32)
    if mine.dim() == 1:
        mine = mine.unsqueeze(1)
    assert theirs.shape == mine.shape, (
        f"shape mismatch: trainer {tuple(mine.shape)}, browser {tuple(theirs.shape)}"
    )
    delta = float((mine - theirs).abs().max())
    assert delta < 1e-5, (
        f"{path.relative_to(REPO)}: the browser and the trainer compute DIFFERENT circuits on the "
        f"same weights - maximum readout difference {delta:.6f}.\n\n"
        "Both implement the same architecture on paper. When they diverge it is because one was "
        "changed and the other was not, which is this project's most-repeated defect. Compare the "
        "parameter LAYOUT first: conv (stages, layers, 8), then pool (stages,), then fc "
        "(survivors, 2), flat in that order."
    )
    if ro:
        argmax_mine = mine.argmax(1)
        argmax_theirs = theirs.argmax(1)
        disagree = int((argmax_mine != argmax_theirs).sum())
        assert disagree == 0, (
            f"{disagree} of {len(rows)} rows are given DIFFERENT CLASSES by the two "
            "implementations. A probability difference can be rounding; a class difference is what "
            "the visitor is shown."
        )


def test_the_comparison_would_notice_a_divergence() -> None:
    """Anti-tautology: perturb one implementation and require the comparison to reject it.

    The three implementations agree, so every assertion above is green on arrival. This makes the
    green mean something by showing the same comparison is not satisfied by an altered circuit -
    here by rotating the fully-connected block's parameter layout, which is precisely the change
    that was made in the trainer and missed in the browser.
    """
    q = _trainer()
    path = PUBLIC / "telecom" / "qcnn_weights.json"
    if not path.exists():
        pytest.skip("the telecom artifact is not shipped")
    w = json.loads(path.read_text())
    nq, layers = int(w["n_qubits"]), int(w["layers"])
    ro = int(w.get("readout", {}).get("readout_qubits", 0) or 0)
    rows = _rows_for(w)
    x = _standardise(torch.tensor(rows, dtype=torch.float32), w)
    if x.shape[1] < nq:
        x = torch.cat([x, torch.zeros(x.shape[0], nq - x.shape[1])], dim=1)

    theta = torch.tensor(w["theta"], dtype=torch.float32)
    good = q.circuit(x, theta, nq=nq, readout_qubits=ro, layers=layers).detach()

    shapes = q.param_shapes(nq, readout_qubits=ro, layers=layers)
    n_conv = shapes["conv"][0] * shapes["conv"][1] * shapes["conv"][2]
    n_pool = shapes["pool"][0]
    bad = theta.clone()
    # Collapse the per-survivor block back to a single shared angle: the pre-Round-47 browser.
    bad[n_conv + n_pool:] = theta[n_conv + n_pool]
    altered = q.circuit(x, bad, nq=nq, readout_qubits=ro, layers=layers).detach()

    delta = float((good - altered).abs().max())
    assert delta > 1e-3, (
        f"collapsing the fully-connected block to one shared angle moved the readout by only "
        f"{delta:.2e}. If that is genuinely below the comparison's 1e-5 tolerance, this test "
        "cannot distinguish the two implementations and the identity above proves nothing."
    )
