"""A circuit called a QCNN must have the structure that word denotes.

WHY THIS EXISTS

Everything in this repository named `qcnn_*` was, until Round 46, a hardware-efficient variational
ansatz: RY on every wire, a CNOT ring, repeat. That is exactly PennyLane's `BasicEntanglerLayers`,
and it scores **zero of the five** structural elements of Cong, Choi & Lukin, "Quantum Convolutional
Neural Networks", *Nat. Phys.* **15**, 1273-1278 (2019):

  1. translationally-invariant SHARED weights within a layer  - it had per-wire independent angles
  2. POOLING by measurement with a conditioned rotation       - it had none
  3. register REDUCTION, so later layers act on fewer qubits  - every layer acted on all 8
  4. a fully-connected unitary on the reduced register        - there was no reduced register
  5. O(log N) parameters                                      - it had Theta(L * n)

The model cards already described the circuit CORRECTLY - "3 layers of trainable RY rotations plus a
CNOT ring on 8 qubits". Only the name was false, which is the harder version of this defect: the
honest description sat one line below the dishonest label, so anyone checking the description found
it accurate and moved on.

WHY A STRUCTURAL TEST AND NOT AN ACCURACY TEST

Accuracy cannot distinguish these architectures - that is the whole point of Bowles, Ahmed & Schuld
(arXiv:2403.07059), which this repository already cites for the entanglement ablation. A QCNN and a
generic ansatz can score identically on a small tabular task. So the claim "this is a QCNN" is a
claim about STRUCTURE, and structure is what gets asserted here.

WHAT IS DELIBERATELY NOT ASSERTED

Not accuracy, not a published number, and not that this reproduces Hur, Kim & Park (*Quantum Machine
Intelligence* **4**, 3, 2022) to the digit. Their 98.5 +/- 0.2% is on their preprocessing of their
dataset. What is asserted is that the circuit has the shape the literature means, so the comparison
in the model cards is between comparable things.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "train"))

torch = pytest.importorskip("torch")


def _mod():
    import qcnn_variational as q

    return q


def test_parameter_count_scales_logarithmically_not_linearly() -> None:
    """O(log N), the property that separates a QCNN from a generic layered ansatz.

    `BasicEntanglerLayers` needs `layers * n_qubits` angles, so doubling the register doubles the
    parameter count. A QCNN's depth IS log2(N) - each pooling layer halves the register - so the
    count grows by a constant per doubling. Asserted as a RATIO across two widths, because an
    absolute count is satisfiable by choosing a small constant.
    """
    q = _mod()
    p4 = q.param_count(4, readout_qubits=1)
    p8 = q.param_count(8, readout_qubits=1)
    p16 = q.param_count(16, readout_qubits=1)
    assert p4 < p8 < p16, f"parameter count is not monotone in width: {p4}, {p8}, {p16}"
    # Doubling the register adds ONE conv+pool layer, so the increment is constant. A linear model
    # would double the count each time instead.
    assert (p8 - p4) == (p16 - p8), (
        f"parameter count grows by {p8 - p4} then {p16 - p8} across doublings - not a constant "
        f"increment, so this is not O(log N). Counts: 4->{p4}, 8->{p8}, 16->{p16}. A "
        "`BasicEntanglerLayers`-shaped model gives layers*n and doubles."
    )
    assert p16 < 4 * p4, (
        f"16 qubits needs {p16} parameters against {p4} for 4 - a linear ansatz would need ~4x"
    )


def test_weights_are_shared_within_a_layer() -> None:
    """Translational invariance: one filter, applied at every position.

    This is what makes it a CONVOLUTION. The old circuit had `theta[layer, q]` - an independent
    angle per wire - which is the definition of NOT sharing.
    """
    q = _mod()
    s8 = q.param_shapes(8, readout_qubits=1)
    s16 = q.param_shapes(16, readout_qubits=1)
    assert s8, "param_shapes returned nothing; the circuit exposes no parameter structure"
    assert set(s8) == set(s16), f"parameter blocks differ between widths: {set(s8)} vs {set(s16)}"
    # THE PROPERTY, not a proxy for it. A first version of this check asserted "no axis equals 8"
    # and fired on the filter's own 8 angles - a shape that has nothing to do with the register.
    # What sharing actually means is that a stage's block does not GROW with the register: only the
    # NUMBER of stages may differ between widths, never the size of a stage's filter.
    for name in s8:
        a, b = s8[name], s16[name]
        assert a[1:] == b[1:], (
            f"parameter block {name!r} is {a} at 8 qubits and {b} at 16 - the per-stage filter grows "
            "with the register, so weights are per-position rather than shared. A convolution "
            "applies ONE filter at every position."
        )


def test_pooling_reduces_the_register_by_measurement_conditioned_rotation() -> None:
    """Elements 2, 3 and 4: pooling, reduction, and a final block on what survives."""
    q = _mod()
    plan = q.layer_plan(8, readout_qubits=1)
    widths = [p["active"] for p in plan]
    assert widths[0] == 8, f"the first layer must act on all 8 wires, got {widths[0]}"
    assert widths == sorted(widths, reverse=True) and widths[-1] < widths[0], (
        f"the register never shrinks: {widths}. Without reduction there is no pooling, and every "
        "layer acting on all 8 wires is the generic ansatz this replaced."
    )
    assert any(p["kind"] == "pool" for p in plan), (
        "no pooling layer in the plan. A QCNN pools by measuring one qubit of each pair and "
        "applying a rotation on the survivor CONDITIONED on that outcome."
    )
    halvings = math.log2(widths[0] / max(widths[-1], 1))
    assert halvings >= 2, (
        f"the register halves only {halvings:.0f} time(s) from 8 wires; a QCNN's depth is log2(N)"
    )


def test_the_circuit_still_runs_and_returns_a_usable_readout() -> None:
    """Structure is necessary, not sufficient - it has to produce a class distribution."""
    q = _mod()
    x = torch.rand(5, 8)
    theta = q.init_params(8, readout_qubits=3, seed=0)
    out = q.circuit(x, theta, nq=8, readout_qubits=3)
    assert out.shape == (5, 8), f"expected (batch, 2**3) class probabilities, got {tuple(out.shape)}"
    assert torch.all(out >= -1e-6), "negative probabilities"
    assert torch.allclose(out.sum(1), torch.ones(5), atol=1e-4), (
        f"class probabilities do not sum to 1: {out.sum(1)}"
    )


def test_no_shipped_artifact_claims_an_architecture_it_does_not_have() -> None:
    """The label, checked against the weights that ship.

    Every `models/qcnn_*/weights.json` must record which architecture produced it, so an artifact
    trained by the old circuit cannot sit under a card describing the new one. That is the failure
    the round's own plan named: "renaming the label and shipping weights from the old circuit".
    """
    import json

    dirs = sorted((REPO / "models").glob("qcnn_*"))
    assert dirs, "no models/qcnn_* directories; this check has no subject"
    stale = []
    for d in dirs:
        w = json.loads((d / "weights.json").read_text())
        arch = str(w.get("training", {}).get("architecture", ""))
        if "qcnn" not in arch.lower():
            stale.append(f"{d.name}: architecture={arch!r}")
    assert not stale, (
        "these shipped artifacts do not record a QCNN architecture:\n  " + "\n  ".join(stale)
        + "\n\nRetrain them. A card naming an architecture over weights from a different circuit is "
        "the same defect as the label this round removed, one layer down."
    )
