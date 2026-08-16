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

def _shipped() -> list[tuple[str, str]]:
    """Every shipped QCNN artifact, from the filesystem rather than from a list kept by hand.

    THIS WAS A HAND-WRITTEN LIST OF FOUR, on a branch named "the gate was a file list". Six
    artifacts ship; the two it omitted were `finance/qcnn_weights_q9.json` and
    `medical/qcnn_weights_q9.json`, and BOTH widths are selectable by the visitor from the QCNN
    qubits control - so the two implementations could have diverged on a configuration a reader can
    reach, with nothing comparing them.
    """
    out = [(p.parent.name, p.name) for p in sorted(PUBLIC.glob("*/qcnn_weights*.json"))]
    assert len(out) >= 4, (
        f"found {len(out)} shipped QCNN artifacts under {PUBLIC}; the glob is not seeing files it "
        "should, and every comparison below would then be vacuous")
    return out


SHIPPED = _shipped()


def _require_browser_or_skip() -> None:
    """Skip when the browser toolchain is absent - unless the environment forbids skipping.

    THE SKIP WAS THE HOLE. These gates are the only ones that can see a change reaching one
    language and not the other, and they guarded on `if not ESBUILD.exists(): pytest.skip(...)`.
    `web/node_modules` is gitignored and the CI python job installed no Node, so on every GitHub
    run they reported SKIP and the workflow went green. Measured on a clone of the tip with no
    `web/node_modules`: **21 collected, 11 passed, 10 skipped** - 6 in
    `test_the_browser_computes_the_bound_the_page_names.py` and 4 here. An earlier revision said
    20 / 11 / 9, which was the count one commit before this docstring was written.

    Seeding the pre-Round-47 shared-scalar fully-connected block into `quantumSim.ts` fails FOUR
    of these gates locally - maximum readout differences 0.399170 (telecom-q7), 0.331032 (telecom),
    0.016233 (medical), 0.008699 (finance) - and passes the entire CI workflow. The single "0.585"
    an earlier revision quoted belongs to the PRE-RETRAIN weights of f5861c9, where the same seeding
    gives 0.585404 on telecom; the commit that published it retrained all six artifacts.

    `QSH_REQUIRE_BROWSER_GATES=1` turns the skip into a failure. CI sets it; a developer without
    node still gets the skip, with the message saying what is unchecked.
    """
    import os

    if ESBUILD.exists():
        return
    msg = (
        "web/node_modules/.bin/esbuild is absent, so the browser model cannot be executed. "
        "Run `npm ci` in web/. This is a SKIP rather than a pass: the identity is unchecked, "
        "not satisfied."
    )
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(
            msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE: the cross-language "
            "checks are mandatory in this environment and skipping them is what let a "
            "one-language fix ship green."
        )
    pytest.skip(msg)


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
    _require_browser_or_skip()
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


@pytest.mark.parametrize("usecase", ["finance", "medical"])
def test_the_numpy_inference_path_computes_the_same_circuit(usecase: str) -> None:
    """THE THIRD IMPLEMENTATION, which the two checks above do not cover.

    `src/qsh/qhe/qcnn_trained.py` is a numpy re-implementation of the same circuit, reached from
    `orchestration/stages.py` - a production path, not a demo one. It unpacks the SAME flat theta,
    and until Round 47 it unpacked the fully-connected block as a single scalar: the pre-Round-47
    form, left behind when the trainer and the browser were both moved to a per-survivor RY+RZ.

    That is the sixth instance of this project's most-repeated defect and the third in this one
    circuit. It is also the most instructive, because it was NOT producing a wrong number. Both
    artifacts it serves are binary, so they pool to a single survivor; `theta[n_conv + stages]` is
    then exactly that survivor's RY angle, and the dropped RZ commutes with the `<Z0>` readout.
    Measured on the shipped weights before the fix: 6.9e-07 on finance, 1.4e-06 on medical.

    A defect that is correct by coincidence is worse than one that is wrong, because nothing can
    see it. `test_the_old_fc_block_diverges_once_there_are_two_survivors` below is the seed that
    makes this assertion evidence rather than a restatement of that coincidence.

    Only the binary use cases are covered here because this module computes `<Z0>` and has no
    multiclass readout at all - it cannot consume telecom. That asymmetry is itself worth knowing,
    and `test_every_qcnn_implementation_is_pinned_to_another` records which pairs are pinned.
    """
    import numpy as np

    from qsh.qhe.qcnn_trained import infer

    path = REPO / "models" / f"qcnn_{usecase}" / "weights.json"
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} is not present")
    w = json.loads(path.read_text())
    q = _trainer()
    nq, layers = int(w["n_qubits"]), int(w["layers"])
    ro = int(w.get("readout", {}).get("readout_qubits", 0) or 0)

    rows = _rows_for(w, n=24)
    x = _standardise(torch.tensor(rows, dtype=torch.float32), w)
    if x.shape[1] < nq:
        x = torch.cat([x, torch.zeros(x.shape[0], nq - x.shape[1])], dim=1)
    theta = torch.tensor(w["theta"], dtype=torch.float32)

    mine = q.circuit(x, theta, nq=nq, readout_qubits=ro, layers=layers).detach().numpy()
    theirs = np.array([infer([float(v) for v in r], w)[0] for r in rows])
    delta = float(np.max(np.abs(mine - theirs)))
    assert delta < 1e-4, (
        f"qcnn_trained.py and the trainer disagree by {delta:.6f} on models/qcnn_{usecase}. Both "
        "unpack the same flat theta; check the fully-connected block first, which is (survivors, 2) "
        "and was a single scalar until Round 47."
    )


def test_the_old_fc_block_diverges_once_there_are_two_survivors() -> None:
    """The seed for the check above, because that one passes on a coincidence.

    At `readout_qubits = 2` on 8 wires the register pools to TWO survivors, and the shared-scalar
    block that `qcnn_trained.py` used to implement stops agreeing with the trainer. Without this,
    a green numpy-vs-trainer comparison would be consistent with the defect still being present -
    which is exactly the state the repository was in until this round.
    """
    q = _trainer()
    nq, layers, ro = 8, 3, 2
    survivors = q.surviving_wires(nq, ro, layers)
    assert len(survivors) == 2, f"this seed needs two survivors, got {survivors}"

    n = q.param_count(nq, readout_qubits=ro, layers=layers)
    g = torch.Generator().manual_seed(1)
    theta = torch.randn(n, generator=g) * 0.5
    x = torch.tanh(torch.randn(16, nq, generator=g) * 0.7)

    good = q.circuit(x, theta, nq=nq, readout_qubits=ro, layers=layers).detach()
    shapes = q.param_shapes(nq, readout_qubits=ro, layers=layers)
    n_conv = shapes["conv"][0] * shapes["conv"][1] * shapes["conv"][2]
    collapsed = theta.clone()
    collapsed[n_conv + shapes["pool"][0]:] = theta[n_conv + shapes["pool"][0]]
    altered = q.circuit(x, collapsed, nq=nq, readout_qubits=ro, layers=layers).detach()

    delta = float((good - altered).abs().max())
    assert delta > 1e-2, (
        f"collapsing the fully-connected block to one shared angle moved a two-survivor readout by "
        f"only {delta:.2e}. If that is genuinely negligible, the comparison above cannot "
        "distinguish the old block from the new one and proves nothing."
    )


def test_every_qcnn_implementation_is_pinned_to_another() -> None:
    """Scope, so a FOURTH re-implementation cannot appear unpinned.

    This circuit exists in more than one place by design - a torch trainer, a numpy inference path
    and a TypeScript browser mirror - and every round that changed it changed a subset. Rather than
    trusting that the three checked above are all of them, this discovers files that unpack a flat
    `theta` and requires each to be named here with the check that pins it.
    """
    import re
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "--", "*.py", "*.ts"], cwd=REPO,
                             check=True, capture_output=True, text=True).stdout.split()
    #: file -> the test that numerically pins it to another implementation.
    PINNED = {
        "scripts/train/qcnn_variational.py": "the reference; every other check compares against it",
        "web/src/lib/quantumSim.ts": "test_the_browser_and_the_trainer_compute_the_same_circuit",
        "src/qsh/qhe/qcnn_trained.py": "test_the_numpy_inference_path_computes_the_same_circuit",
    }
    unpinned = []
    for rel in tracked:
        if rel.startswith("tests/"):
            continue
        src = (REPO / rel).read_text(errors="ignore")
        # Unpacks a flat parameter vector AND builds a pooling register: that is this circuit.
        if re.search(r"theta\[|w\.theta\[", src) and re.search(r"survivor|active\[0::2\]|pool", src):
            if rel not in PINNED:
                unpinned.append(rel)
    assert not unpinned, (
        "these files unpack the QCNN's flat parameter vector and no numeric check pins them to "
        "another implementation:\n  " + "\n  ".join(unpinned)
        + "\n\nAdd a comparison, or record here which existing check covers it. Three separate "
        "rounds have now shipped a change that reached some implementations and not others; the "
        "count of implementations is exactly what nobody was tracking."
    )
    assert len(PINNED) >= 3, "the pinned set has shrunk; a re-implementation was removed silently"
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
