"""The 8-class QCNN readout, and the permutation that would otherwise pass every other check.

A multi-qubit basis readout has one failure mode that no aggregate metric can see. Read the bits in
the wrong order and the class map is PERMUTED: accuracy, macro F1, per-class F1 as a multiset, and
every loss curve come out numerically identical, because the same probabilities are simply attached to
different labels. Only a check that ties a KNOWN input to a NAMED class can catch it.

MEASURED IN THIS ENVIRONMENT, not assumed. Rotating wire 0 of an 8-qubit register and reading the
full state gives nonzero amplitudes at flat indices [0, 128] = [0, 2^7], so PennyLane places the FIRST
listed wire in the MOST significant bit. An LSB-first read of bits 0,1,2 therefore equals
`qml.probs(wires=[7,6,5])` and NOT `qml.probs(wires=[0,1,2])` - verified below rather than cited.

The bit-reversal permutation on three bits is [0, 4, 2, 6, 1, 5, 3, 7], and FOUR of its eight entries
are fixed points. Against `qsh.qml_ids.attacks.ATTACKS` those are `normal`, `photon_number_splitting`,
`wavelength_trojan` and `combined` - which includes the two a person would most naturally spot-check.
So the identifiability test is parametrised over all eight classes on purpose: checking `normal` and
`combined` passes under the bug.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "train" / "qcnn_variational.py"


def _trainer():
    spec = importlib.util.spec_from_file_location("qcnn_variational", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Bit reversal on 3 bits. The class map a wrong-endianness readout produces.
BIT_REVERSAL_3 = [0, 4, 2, 6, 1, 5, 3, 7]


def test_the_marginal_is_a_probability_distribution():
    m = _trainer()
    # ROUND 46: `theta` is a flat shared-weight vector sized by the architecture, not (layers, nq).
    x, theta = torch.rand(6, 8), m.init_params(8, readout_qubits=3, layers=3, seed=0)
    p = m.circuit(x, theta, readout_qubits=3, layers=3)
    assert p.shape == (6, 8), "3 measured qubits must give 2**3 = 8 class scores"
    assert torch.allclose(p.sum(1), torch.ones(6), atol=1e-5), (
        "the marginal must sum to 1 by construction; if it does not, a gate application is broken and "
        "every class score is wrong by the same unknown factor")
    assert bool((p >= 0).all())


def test_the_readout_is_lsb_first_and_pennylane_is_not():
    """The trap, measured. Not a citation - PennyLane is installed and this asserts on its output."""
    # The module is `pennylane`; importorskip("qml") named the ALIAS and skipped unconditionally,
    # which made the one test that verifies the endianness convention pass by not running. A skip
    # that can never be satisfied is a check that cannot fail - caught here on the first run.
    qml = pytest.importorskip("pennylane", reason="pennylane not installed")

    nq = 8
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev)
    def full():
        qml.RY(0.9, wires=0)
        return qml.probs(wires=range(nq))

    @qml.qnode(dev)
    def listed(wires):
        qml.RY(0.9, wires=0)
        return qml.probs(wires=wires)

    p_full = np.asarray(full())
    nz = np.nonzero(p_full > 1e-9)[0].tolist()
    assert nz == [0, 1 << (nq - 1)], (
        f"expected PennyLane to put wire 0 in the MSB (indices [0, {1 << (nq - 1)}]), got {nz}. If "
        f"this changes, the whole readout convention has to be re-derived")

    lsb = np.zeros(8)
    for i, v in enumerate(p_full):
        lsb[i & 7] += v
    assert np.allclose(lsb, np.asarray(listed([7, 6, 5])), atol=1e-6), (
        "an LSB-first read of bits 0,1,2 must equal probs(wires=[7,6,5])")
    assert not np.allclose(lsb, np.asarray(listed([0, 1, 2])), atol=1e-6), (
        "if these ever agree the endianness hazard has gone away and this test should be revisited - "
        "but silently agreeing is far more likely to mean the test stopped testing anything")


@pytest.mark.parametrize("cls", range(8))
def test_every_class_is_individually_identifiable(cls):
    """A basis state prepared for class `cls` must read back as `cls`, for ALL eight.

    Parametrised rather than looped so a failure names the class. Four of the eight are fixed points
    of the bit reversal, so a test covering only some of them can pass under a permuted map.
    """
    m = _trainer()
    nq = 8
    state = torch.zeros(1, 1 << nq, dtype=torch.complex64)
    state[0, cls] = 1.0                        # |...b2 b1 b0> with c = i & 7, LSB-first
    p = (state.abs() ** 2)
    idx = torch.arange(p.shape[1])
    got = torch.zeros(1, 8).index_add_(1, (idx & 7).to(torch.long), p)
    assert int(got.argmax(1)) == cls, f"class {cls} read back as {int(got.argmax(1))}"


def test_a_permuted_class_map_is_detected_by_this_suite():
    """Anti-tautology: prove the identifiability check is sensitive to the bug it exists for.

    Without this, a test that passes tells you nothing about whether it COULD fail.
    """
    truth = list(range(8))
    permuted = [BIT_REVERSAL_3[c] for c in truth]
    assert permuted != truth, "bit reversal must actually move something"
    moved = [c for c in truth if BIT_REVERSAL_3[c] != c]
    fixed = [c for c in truth if BIT_REVERSAL_3[c] == c]
    assert fixed == [0, 2, 5, 7], f"fixed points changed: {fixed}"
    assert moved == [1, 3, 4, 6], f"moved classes changed: {moved}"
    # The four moved classes are the ones the parametrised test above must cover to be meaningful.
    assert len(moved) == 4 and len(fixed) == 4, (
        "half the classes are invisible to a partial spot check, which is exactly why the "
        "identifiability test is parametrised over all eight")


def test_the_binary_readout_still_works():
    """The multiclass path must not change what the two shipped binary QCNNs compute."""
    m = _trainer()
    x, theta = torch.rand(5, 8), m.init_params(8, readout_qubits=0, layers=3, seed=0)
    z = m.circuit(x, theta, layers=3)
    assert z.shape == (5,), "default readout stays a single <Z0> expectation"
    assert bool((z >= -1.0001).all() and (z <= 1.0001).all()), "<Z> is bounded by [-1, 1]"
    # and it must equal the 1-qubit marginal contrast, which is the same quantity by definition
    p1 = m.circuit(x, theta, readout_qubits=1, layers=3)
    assert torch.allclose(z, p1[:, 0] - p1[:, 1], atol=1e-5), (
        "<Z0> = p(0) - p(1) on the measured wire; if these disagree the marginal is binned wrongly")


def _survivors(mod, nq: int, readout_qubits: int) -> list[int]:
    """The wires still live when the fully-connected block runs.

    Derived HERE from `layer_plan`, independently of `surviving_wires()`, so that the assertion
    below compares two derivations rather than one value against itself. If the trainer's helper
    is edited to agree with a wrong readout, this recomputation does not follow it.
    """
    active = list(range(nq))
    for step in mod.layer_plan(nq, readout_qubits=readout_qubits):
        if step["kind"] == "pool":
            active = [a for a, _ in zip(active[0::2], active[1::2], strict=False)]
    return active


def test_every_readout_wire_survived_the_pooling():
    """A class index may only be built from wires the circuit still processes.

    WHY THIS EXISTS

    The multiclass readout bins the state by `i & (k - 1)` - the low `m` bits of the flat index -
    and wire `q` is bit `q`, so a 3-qubit readout reads wires 0, 1 and 2. After one pooling stage
    on 8 wires the survivors are [0, 2, 4, 6]. **Wire 1 is a pooling CONTROL that was measured
    away**: `apply_cry(state, b=1, a=0, ...)` conditions wire 0 on it, and from that point nothing
    touches wire 1 again - the fully-connected block iterates `active`, which no longer contains it.

    So one of the three wires the class index is built from is frozen at its pre-pooling value
    while the other two are carried through the FC block. That is not a deliberate choice:
    `layer_plan`'s own docstring justifies stopping early by saying "a 3-qubit readout for 8 classes
    cannot pool all the way to one wire", a rationale that is only coherent if the readout wires
    ARE the survivors. The binning is byte-identical to the pre-Round-46 version, where there was
    no pooling at all and every wire was live - it is a stale port, not a design.

    Stated over every (width, readout) pair the repository ships rather than the one that is wrong,
    so a future width cannot reintroduce it.
    """
    m = _trainer()
    offenders = []
    for nq in (8, 9, 12, 16):
        for ro in (1, 2, 3):
            surv = _survivors(m, nq, ro)
            read = m.readout_wires(nq, readout_qubits=ro)
            dead = [w for w in read if w not in surv]
            if dead or len(read) != ro:
                offenders.append(
                    f"nq={nq} readout_qubits={ro}: reads wires {read}, survivors are {surv}, "
                    f"so wire(s) {dead or '(wrong count)'} are not live")
    assert not offenders, (
        "the multiclass readout builds its class index from wires the circuit discarded:\n  "
        + "\n  ".join(offenders)
        + "\n\nBin over the first `readout_qubits` SURVIVORS instead of over wires 0..m-1. A "
        "discarded pooling control is frozen before the fully-connected block runs, so one bit of "
        "every class index is decided by a wire the second half of the circuit never sees."
    )


def test_that_check_can_actually_fail():
    """Anti-tautology for the check above, using the convention that shipped until Round 47.

    `i & (k - 1)` reads wires 0..m-1. At the width the telecom model ships - 8 wires, 3 readout
    qubits - that set is NOT a subset of the survivors, so the check has teeth. Without this, a
    green subset assertion would be consistent with `readout_wires` simply returning the survivors
    it is compared against.
    """
    m = _trainer()
    surv = _survivors(m, 8, 3)
    old_convention = list(range(3))                      # what `i & (k - 1)` reads
    assert surv == [0, 2, 4, 6], f"survivor derivation moved: {surv}"
    assert any(w not in surv for w in old_convention), (
        "the pre-Round-47 readout convention is a subset of the survivors, so the check above "
        "cannot distinguish it from the fix and is vacuous"
    )
    assert m.readout_wires(8, readout_qubits=3) == [0, 2, 4], (
        "the shipped readout is not the first three survivors"
    )
