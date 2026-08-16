#!/usr/bin/env python3
"""Train a variational QCNN whose readout actually depends on its input.

WHY THIS EXISTS. The browser's `qotpQcnnInfer` is a genuine 8-qubit state-vector simulation and a
genuine Clifford+T circuit, and its class output carries no information about the data. Measured over
24 feature vectors x 8 seeds: 100% of vectors flip class with the seed, mean |<Z0>| is 0.076, and the
correlation between the unmasked class and mean(feature) is -0.070. Seven FIXED rounds of H, a CNOT
ring and T on every qubit scramble the state toward uniform, so the sign of <Z0>' is numerical noise.
Full workings: references/qcnn-inference-investigation.md.

WHAT CHANGES. Two things, and the second matters more than the obvious one:

  1. The convolution rounds get TRAINABLE RY angles. A circuit with no parameters cannot learn, and
     no amount of QOTP correctness fixes that.
  2. The prediction path does NOT apply the QOTP mask. Undoing a Pauli frame through a Clifford+T
     circuit is not a detail - `T X T@` is not Pauli, which is exactly the difficulty Ortega &
     Martin-Delgado's scheme exists to solve, and this repository does not implement it. So hiding
     and prediction are separated: `qotpQcnnInfer` stays as the hiding + T-count demonstration, and
     this trains the predictor. Claiming a correction that is not implemented would be worse than
     having no quantum predictor at all.

HONEST SCOPE. This is a simulated variational circuit trained by backpropagation through a dense
state vector. It is not quantum hardware and not a claim of quantum advantage - the classical
baseline is reported beside it every time, and if the QCNN loses, the number that ships says so.

Usage:  .venv/bin/python scripts/train/qcnn_variational.py --usecase medical
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

N_QUBITS = 8


def _pair_index(nq: int, q: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat indices (i with bit q clear, i with bit q set).

    Flat index arithmetic rather than movedim: the first version of this file permuted axes with
    two chained movedim calls and used the ORIGINAL axis number for the second, so after the first
    move the control and target of every CNOT were crossed. It trained fine - the circuit was
    self-consistent, just not the circuit documented - and only a parity check against an
    independent reimplementation caught it, at differences up to 0.73 on a value bounded in [-1, 1].
    """
    N = 1 << nq
    i = torch.arange(N, device=device)
    bit = 1 << q
    i0 = i[(i & bit) == 0]
    return i0, i0 | bit


def apply_1q(state: torch.Tensor, q: int, m: torch.Tensor, nq: int) -> torch.Tensor:
    """Apply a 2x2 gate to qubit `q`; qubit 0 is the LEAST significant bit of the flat index, which
    is what makes `i & 1` the correct <Z0> readout in both this file and the browser."""
    i0, i1 = _pair_index(nq, q, state.device)
    a, b = state[:, i0], state[:, i1]
    out = state.clone()
    out[:, i0] = m[0, 0] * a + m[0, 1] * b
    out[:, i1] = m[1, 0] * a + m[1, 1] * b
    return out


def apply_cnot(state: torch.Tensor, ctl: int, tgt: int, nq: int) -> torch.Tensor:
    N = 1 << nq
    i = torch.arange(N, device=state.device)
    cb, tb = 1 << ctl, 1 << tgt
    sel = ((i & cb) != 0) & ((i & tb) == 0)
    src, dst = i[sel], i[sel] | tb
    out = state.clone()
    out[:, src] = state[:, dst]
    out[:, dst] = state[:, src]
    return out


def _ry(th: torch.Tensor, cdt) -> torch.Tensor:
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(cdt)


def _rz(th: torch.Tensor, cdt) -> torch.Tensor:
    """diag(e^{-i th/2}, e^{+i th/2}). RY alone spans only real rotations, so a filter built from it
    cannot reach a general two-qubit unitary however many times it is applied."""
    z = torch.zeros((), dtype=th.dtype, device=th.device)
    p = torch.complex(torch.cos(th / 2), -torch.sin(th / 2))
    q = torch.complex(torch.cos(th / 2), torch.sin(th / 2))
    zc = torch.complex(z, z)
    return torch.stack([torch.stack([p, zc]), torch.stack([zc, q])]).to(cdt)


def apply_cry(state: torch.Tensor, ctl: int, tgt: int, th: torch.Tensor,
              nq: int) -> torch.Tensor:
    """Controlled-RY: rotate `tgt` only on the branches where `ctl` is |1>.

    THIS IS THE POOLING OPERATION, and it is what makes the circuit a QCNN rather than a layered
    ansatz. Cong, Choi & Lukin pool by MEASURING one qubit of each pair and applying a rotation on
    the survivor conditioned on that outcome. A state-vector simulator has no mid-circuit
    measurement, so the standard equivalent - and the one PennyLane's own QCNN demo uses - is a
    controlled rotation followed by simply never touching the control qubit again. The conditioning
    is unitary instead of classical; the register reduction and the information flow are the same.
    """
    N = 1 << nq
    i = torch.arange(N, device=state.device)
    cb, tb = 1 << ctl, 1 << tgt
    sel0 = ((i & cb) != 0) & ((i & tb) == 0)     # control set, target clear
    src, dst = i[sel0], i[sel0] | tb
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    a, b = state[:, src], state[:, dst]
    out = state.clone()
    out[:, src] = c.to(state.dtype) * a - s.to(state.dtype) * b
    out[:, dst] = s.to(state.dtype) * a + c.to(state.dtype) * b
    return out


def layer_plan(nq: int = N_QUBITS, readout_qubits: int = 0, layers: int = 1) -> list[dict]:
    """The architecture, as data, so a test can assert its SHAPE without running it.

    One entry per stage. `active` is how many wires the stage acts on; `kind` is `conv` or `pool`.
    The register halves at each pool and stops when a further halving would leave fewer wires than
    the readout needs - a 3-qubit readout for 8 classes cannot pool all the way to one wire.
    """
    keep = max(1, readout_qubits)
    plan, active = [], nq
    while active // 2 >= keep and active > 1:
        for r in range(layers):
            plan.append({"kind": "conv", "active": active, "_round": r})
        plan.append({"kind": "pool", "active": active})
        active //= 2
    plan.append({"kind": "fc", "active": active})
    return plan


def surviving_wires(nq: int = N_QUBITS, readout_qubits: int = 0, layers: int = 1) -> list[int]:
    """WHICH wires are still live when the fully-connected block runs.

    `layer_plan` says how MANY wires each stage acts on; this says which ones. Computed the way
    `circuit()` computes it - pair up the active list, keep the `a` of each pair - so the two agree
    by construction rather than by a comment promising they do.
    """
    active = list(range(nq))
    for step in layer_plan(nq, readout_qubits, layers):
        if step["kind"] == "pool":
            active = [a for a, _ in zip(active[0::2], active[1::2], strict=False)]
    return active


def readout_wires(nq: int = N_QUBITS, readout_qubits: int = 0, layers: int = 1) -> list[int]:
    """The wires the class index is built from, least-significant bit first.

    THE SURVIVORS, not wires 0..m-1. Until Round 47 the readout binned by `i & (k - 1)`, which
    reads bits 0..m-1, and wire q is bit q - so a 3-qubit readout read wires 0, 1, 2 while the
    survivors were [0, 2, 4, 6]. Wire 1 is the pooling control of the pair (0, 1): conditioned
    wire 0 on itself and was then never touched again, so one bit of every class index came from a
    wire frozen before the fully-connected block ran.

    Exposed as a function rather than inlined because it has to reach three places - the circuit,
    the artifact that ships to the browser, and the gate that checks it - and a convention that
    lives in only one of them is how this project's last four rounds each shipped a defect.
    """
    surv = surviving_wires(nq, readout_qubits, layers)
    if readout_qubits > len(surv):
        raise ValueError(
            f"a {readout_qubits}-qubit readout needs {readout_qubits} surviving wires but pooling "
            f"leaves {len(surv)} ({surv}) at nq={nq}. `layer_plan` must stop earlier."
        )
    return surv[:readout_qubits]


def param_shapes(nq: int = N_QUBITS, readout_qubits: int = 0, layers: int = 1) -> dict:
    """Named parameter blocks and their shapes.

    NO AXIS IS THE QUBIT COUNT. That is the whole difference from what this file used to do: the
    old `theta` was `(layers, nq)`, one independent angle per wire, which is per-position weights -
    the opposite of a convolution. Here a stage carries `(2,)` conv angles and `(1,)` pool angle
    SHARED across every pair in that stage, so the filter is translationally invariant.
    """
    stages = sum(1 for p in layer_plan(nq, readout_qubits, layers) if p["kind"] == "pool")
    # EIGHT angles per filter, not two. The first version of this architecture used a single
    # RY per wire plus a CNOT - 2 shared angles - and it underfits badly where the readout needs
    # several wires: telecom (8 classes, 3 readout qubits) gets ONE pool stage, so 2-angle filters
    # gave 14 parameters for an 8-class problem against the old ansatz's 48, and macro F1 fell from
    # 0.81 to 0.48. Hur et al. use an expressive two-qubit block for exactly this reason.
    #
    # The filter is still SHARED across every pair in a stage, so this adds capacity without
    # reintroducing per-position weights, and the count is still O(log N) in the register width -
    # each doubling adds one stage.
    #
    # THE `fc` BLOCK IS NOT SHARED, AND MUST NOT BE. Cong, Choi & Lukin: "a fully connected layer
    # is applied as a unitary F on the remaining qubits". Translational invariance is a property of
    # the CONVOLUTION layers - it is what makes them convolutions - and F is a general unitary on
    # the reduced register, not a filter applied at every position. Until Round 47 `fc` was a
    # single scalar RY angle applied identically to every survivor, which is the weakest of the
    # five structural elements this architecture claims and is not what the paper specifies.
    #
    # It stays O(log N) because the SURVIVOR COUNT IS BOUNDED, not because the block is shared:
    # `layer_plan` stops pooling while `active // 2 >= readout_qubits`, so the register that
    # reaches F has between `readout_qubits` and `2*readout_qubits - 1` wires whatever nq is - 4
    # survivors at both 8 and 16 qubits. A block of `(survivors, 2)` is therefore constant in the
    # register width, and the per-doubling increment stays constant.
    surv = len(surviving_wires(nq, readout_qubits, layers))
    return {"conv": (stages, layers, 8), "pool": (stages,), "fc": (surv, 2)}


def param_count(nq: int = N_QUBITS, readout_qubits: int = 0, layers: int = 1) -> int:
    """Total trainable scalars. O(log N) in the register width, by construction.

    Each doubling of `nq` adds exactly one pool stage, hence `2 * layers + 1` parameters - a
    CONSTANT increment, where a per-wire ansatz would double. `tests/test_the_qcnn_has_the_structure_the_name_claims.py`
    asserts the increment rather than the value, because any single count is satisfiable by picking
    a small constant.
    """
    import math as _m

    return int(sum(_m.prod(s) if s else 1 for s in param_shapes(nq, readout_qubits, layers).values()))


def init_params(nq: int = N_QUBITS, readout_qubits: int = 0, layers: int = 1,
                seed: int = 0) -> torch.Tensor:
    """Flat parameter vector, small-angle initialised."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(param_count(nq, readout_qubits, layers), generator=g) * 0.1).requires_grad_(True)


def _unpack(theta: torch.Tensor, nq: int, readout_qubits: int, layers: int):
    shapes = param_shapes(nq, readout_qubits, layers)
    n_conv = shapes["conv"][0] * shapes["conv"][1] * shapes["conv"][2]
    n_pool = shapes["pool"][0]
    conv = theta[:n_conv].reshape(shapes["conv"])
    pool = theta[n_conv:n_conv + n_pool]
    fc = theta[n_conv + n_pool:].reshape(shapes["fc"])
    return conv, pool, fc


def circuit(x: torch.Tensor, theta: torch.Tensor, nq: int = N_QUBITS,
            readout_qubits: int = 0, entangle: bool = True, layers: int = 1) -> torch.Tensor:
    """(batch, nq) features + a FLAT shared-weight vector -> a readout. State FLAT as (batch, 2**nq).

    `theta` is no longer `(layers, nq)`. It is the flat concatenation of the conv, pool and
    fully-connected blocks that `param_shapes()` describes, none of whose axes is the qubit count -
    see the note there for why per-wire angles are the thing a convolution replaces.

    ``readout_qubits = 0`` gives the original single ``<Z0>`` expectation, shape (batch,), which is
    BINARY BY CONSTRUCTION: one number through one scalar scale and bias is one logit. That is what
    this circuit produced until Round 34, and it is why an 8-class macro F1 could not honestly be
    reported for it - six of eight classes had a floor of zero.

    ``readout_qubits = m`` gives the marginal probabilities of the first m qubits, shape
    (batch, 2**m), which is a genuine K-class readout for K = 2**m. Qiskit's VQC standardises on
    exactly this (a computational-basis readout mapped to classes), and at m=3 for 8 classes the map
    is the identity: nothing is discarded and the probabilities sum to 1 by construction.

    ENDIANNESS. Qubit q is bit q of the flat index, qubit 0 the LEAST significant - the same
    convention the ``idx & 1`` below relies on for ``<Z0>``, and the one ``quantumSim.ts`` uses. So
    class ``c = i & (2**m - 1)``. PennyLane puts the FIRST LISTED wire in the MOST significant bit, so
    an LSB-first read of bits 0,1,2 equals ``qml.probs(wires=[7,6,5])`` and NOT ``wires=[0,1,2]``.
    Getting that backwards permutes the class map by ``[0,4,2,6,1,5,3,7]`` while leaving accuracy,
    macro F1 and every loss curve identical - and four of the eight classes are FIXED POINTS of that
    permutation, so a spot check on the obvious ones passes. ``tests/test_qcnn_multiclass_readout.py``
    covers all eight individually for that reason.
    """
    b = x.shape[0]
    dev, cdt = x.device, torch.complex64
    state = torch.zeros((b, 1 << nq), dtype=cdt, device=dev)
    state[:, 0] = 1.0                                     # |0...0>

    for q in range(nq):                                   # data encode, per-row angle
        i0, i1 = _pair_index(nq, q, dev)
        th = (x[:, q] * torch.pi).reshape(-1, 1)
        c, s = torch.cos(th / 2).to(cdt), torch.sin(th / 2).to(cdt)
        a, bb = state[:, i0], state[:, i1]
        nxt = state.clone()
        nxt[:, i0] = c * a - s * bb
        nxt[:, i1] = s * a + c * bb
        state = nxt

    # ---- QCNN proper: conv (shared filter) -> pool (measurement-conditioned) -> reduce ----
    #
    # ROUND 46 REPLACED WHAT WAS HERE. It was `for layer: for q: RY(theta[layer,q])` plus a CNOT
    # ring - PennyLane's `BasicEntanglerLayers`, exactly - under the name QCNN. Zero of the five
    # structural elements of Cong, Choi & Lukin (Nat. Phys. 15, 1273, 2019) were present: no shared
    # weights, no pooling, no register reduction, no fully-connected block on a reduced register,
    # and Theta(L*n) parameters instead of O(log N).
    #
    # Reference architecture: Hur, Kim & Park, "Quantum convolutional neural network for classical
    # data classification", Quantum Machine Intelligence 4, 3 (2022), arXiv:2108.00661.
    conv, pool, fc = _unpack(theta, nq, readout_qubits, layers)
    cdt = torch.complex64
    active = list(range(nq))
    stage = 0
    for step in layer_plan(nq, readout_qubits, layers):
        if step["kind"] == "conv":
            # ONE FILTER, APPLIED AT EVERY POSITION. `conv[stage, round]` carries two angles shared
            # by every pair in this stage - that sharing is what the word "convolution" denotes.
            r = step.get("_round", 0)
            w = conv[stage, r]
            # strict=False on purpose: an ODD register leaves an orphan, which is
            # dropped here and pooled into the first survivor below.
            for a, b in zip(active[0::2], active[1::2], strict=False):
                # RY,RZ on each wire -> CNOT -> RY,RZ on each. One filter, every position.
                for wire, iy, iz in ((a, 0, 1), (b, 2, 3)):
                    state = apply_1q(state, wire, _ry(w[iy], cdt), nq)
                    state = apply_1q(state, wire, _rz(w[iz], cdt), nq)
                if entangle:
                    state = apply_cnot(state, a, b, nq)
                # RZ BEFORE RY in the second half of the filter, for the same reason as the
                # fully-connected block. On the wire pooling is about to discard, the only later
                # interaction is as the CONTROL of a CRY - diagonal in that wire's basis - and then
                # marginalisation at readout. A diagonal gate followed only by diagonal use is
                # unobservable, so `rz(b, w8[7])` was dead: moving it by 1.234 rad changed the
                # readout by less than 1e-5 at every width. That was 3 of 72 conv angles on the
                # binary models and 1 of 48 on telecom, carried since Round 46.
                for wire, iy, iz in ((a, 4, 5), (b, 6, 7)):
                    state = apply_1q(state, wire, _rz(w[iz], cdt), nq)
                    state = apply_1q(state, wire, _ry(w[iy], cdt), nq)
        elif step["kind"] == "pool":
            pairs = list(zip(active[0::2], active[1::2], strict=False))
            for a, b in pairs:
                state = apply_cry(state, b, a, pool[stage], nq)   # condition survivor a on b
            # AN ODD REGISTER HAS AN ORPHAN, and it must not simply survive. `active[0::2]` kept it
            # in the active set, so on a 9-wire register the widths went 9->5->3->2 while
            # `layer_plan` said 9->4->2->1, and one wire carried an encoded feature to the readout
            # having never been convolved or pooled. Measured cost: finance at 9 qubits collapsed to
            # predicting every row positive - accuracy 0.0517 on a 5% positive set - and medical at 9
            # qubits to minority-F1 0.0000.
            #
            # The orphan is now pooled into the FIRST survivor, so its information reaches the
            # readout, and the survivors are the `a` of each pair - which makes the circuit's widths
            # equal `layer_plan`'s by construction rather than by coincidence.
            if len(active) % 2 == 1:
                state = apply_cry(state, active[-1], active[0], pool[stage], nq)
            active = [a for a, _ in pairs]
            stage += 1
        else:                                                      # fully connected on survivors
            # A UNITARY ON THE REDUCED REGISTER, so each survivor carries its OWN angles. Sharing
            # one scalar across every survivor - what this did until Round 47 - makes F a single
            # global rotation, which cannot distinguish the wires the class index is read from.
            # Cong, Choi & Lukin call for "a unitary F on the remaining qubits"; translational
            # invariance belongs to the convolution layers, not to F.
            # RZ BEFORE RY, and the order is the whole point. A diagonal gate applied LAST cannot
            # be observed: the only operations after this block are a CNOT ring, which permutes
            # basis states, and |amplitude|^2, which discards phase. Round 47 added these RZ angles
            # for faithfulness to Cong et al.'s "unitary F on the remaining qubits" and applied them
            # after the RY, so all of them were inert.
            #
            # RE-MEASURED ON THE SHIPPED WEIGHTS, WITH THE RECIPE STATED - an earlier revision of
            # this comment gave four figures with no seed, no inputs and no command, and the weights
            # they were taken on were replaced by a later retrain. Reconstruct the pre-fix order
            # (`git show 8017538^:scripts/train/qcnn_variational.py`), take each artifact's theta,
            # feed x = tanh(randn(8, n_qubits) * 0.7) at torch.Generator().manual_seed(0), add
            # 1.234 rad to every fc RZ angle, and take max |delta| of circuit():
            #
            #                  fc RZ, pre-fix   fc RY, pre-fix   fc RZ, shipped order
            #   telecom          5.96e-08         1.77e-01            2.28e-01
            #   telecom_q7       1.19e-07         3.64e-01            1.59e-01
            #   finance          5.96e-08         9.89e-01            4.82e-02
            #   medical          8.94e-08         5.12e-01            7.16e-02
            #
            # The three pre-fix RZ figures are float32 ULPs - 2^-24, 2^-23 and 1.5 * 2^-24 - so the
            # angles were inert, not merely small. Half the block was decoration.
            #
            # With RZ first, the phases it writes are mixed back into amplitudes by the RY, so both
            # parameters of the single-qubit rotation reach the readout and F is a genuine
            # two-parameter unitary rather than a one-parameter one carrying a spare number.
            for j, wire in enumerate(active):
                state = apply_1q(state, wire, _rz(fc[j, 1], cdt), nq)
                state = apply_1q(state, wire, _ry(fc[j, 0], cdt), nq)
            if entangle and len(active) > 1:
                for j in range(len(active)):
                    state = apply_cnot(state, active[j], active[(j + 1) % len(active)], nq)

    p = state.abs() ** 2
    idx = torch.arange(p.shape[1], device=dev)
    if readout_qubits:
        # Marginal over the first `readout_qubits` SURVIVORS - the wires still live when the
        # fully-connected block runs - and NOT over wires 0..m-1.
        #
        # ROUND 47 FIXED THIS. `i & (k - 1)` reads bits 0..m-1, and wire q is bit q, so a 3-qubit
        # readout read wires 0, 1 and 2. After one pooling stage the survivors are [0, 2, 4, 6]:
        # wire 1 is the CRY CONTROL of the pair (0, 1), measured away, never touched again. One bit
        # of every class index was therefore decided by a wire the second half of the circuit does
        # not see, frozen at its pre-pooling value.
        #
        # It was not a choice. `layer_plan` justifies stopping early with "a 3-qubit readout for 8
        # classes cannot pool all the way to one wire" - a rationale that only makes sense if the
        # readout wires ARE the survivors - and this binning is byte-identical to the pre-Round-46
        # version, where nothing pooled and every wire was live. A stale port, carried across the
        # rewrite that introduced pooling.
        k = 1 << readout_qubits
        wires = readout_wires(nq, readout_qubits, layers)
        assert wires == active[:readout_qubits], (
            f"the declared readout wires {wires} are not the ones this run pooled down to "
            f"({active[:readout_qubits]}); the artifact would ship a convention the circuit does "
            "not use"
        )
        cls = torch.zeros(p.shape[1], dtype=torch.long, device=dev)
        for j, wire in enumerate(wires):
            cls |= ((idx >> wire) & 1) << j
        # A GATHER AND A VIEW, not `index_add_`. Every class owns exactly 2**(nq-m) flat indices, so
        # a stable sort by class makes the groups contiguous and equal-sized and the marginal is a
        # plain reduction over the last axis.
        #
        # It has to avoid `index_add_` because that accumulates with atomicAdd on CUDA, which
        # PyTorch documents as nondeterministic: the summation order changes between runs, so the
        # same seeded command produced macro F1 0.809846 on some runs and 0.807781 on others. That
        # is why only the MULTICLASS artifact was irreproducible - the binary readout below never
        # touches this path. Chasing it through thread pinning and PYTHONHASHSEED found nothing,
        # because the training does not run on the CPU at all. `torch.argsort(stable=True)` and a
        # fixed-shape sum are both deterministic.
        perm = torch.argsort(cls, stable=True)
        return p[:, perm].view(p.shape[0], k, -1).sum(2)
    sign = torch.where((idx & 1) == 0, 1.0, -1.0)         # bit 0 set -> -1
    return (p * sign).sum(1)


def load(usecase: str):
    if usecase == "medical":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        cols = ["mean radius", "mean texture", "mean perimeter", "mean area",
                "mean smoothness", "worst radius", "worst concavity", "worst symmetry"]
        idx = [list(d.feature_names).index(c) for c in cols]
        X, y = d.data[:, idx], (d.target == 0).astype(int)     # 1 = malignant
        return X, y, cols, "Wisconsin Diagnostic Breast Cancer (UCI, CC-BY-4.0), 8 of 30 features"
    if usecase == "telecom":
        # 7 QKD telemetry features, 8 attack classes. This is the MULTICLASS path, and it is the
        # reason `circuit(readout_qubits=3)` exists: a single <Z0> is one logit and cannot represent
        # eight classes, so scoring the binary QCNN against 8-class labels gave six of them a floor
        # of zero by construction.
        #
        # Only 7 of the 8 wires carry data. Wire 7 is left unencoded rather than padded with a
        # constant: a constant would be a feature with zero variance that the entangling layers still
        # rotate, which is indistinguishable from a real input at training time and meaningless at
        # inference. Unencoded, it is extra Hilbert space the CNOT ring can use and nothing more.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from qsh.qml_ids.attacks import ATTACKS, FEATURES, generate
        X, y, class_names = generate(n_per_class=400, seed=0)
        assert list(class_names) == list(ATTACKS)
        assert X.shape[1] == len(FEATURES) == 7, (
            f"expected 7 telemetry features, got {X.shape[1]} - a silent mislabel of every per-feature "
            f"number downstream, which this corpus has already produced once")
        return X, y, list(FEATURES), (
            "SYNTHETIC QKD-attack telemetry generated by qsh.qml_ids.attacks; class structure per "
            "Al-Kuwari et al. (arXiv:2509.14282). 8 classes = 7 attack types + normal traffic. No "
            "commercial-OK real attack corpus exists, and the synthetic origin is why no result here "
            "is a performance claim about real links.")

    if usecase == "finance":
        import pandas as pd
        df = pd.read_csv(REPO / "data" / "finance" / "paysim_sample.tsv", sep="\t", comment="#")
        cols = [c for c in df.columns if c != "isFraud"]
        X, y = df[cols].to_numpy(float), df["isFraud"].to_numpy(int)
        return X, y, cols, ("PaySim simulator output (Kaggle ealaxi/paysim1, CC-BY-SA-4.0; "
                            "Lopez-Rojas et al. 2016) - SYNTHETIC transactions from aggregated real "
                            "mobile-money logs, 8 AML features, the same 8 the browser pipeline feeds")
    raise SystemExit(f"no loader for {usecase!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usecase", default="medical")
    ap.add_argument("--layers", type=int, default=3)
    # WIDTH. Was the module constant N_QUBITS=8 for every use case, which made the browser's
    # "QCNN qubits" control inert on every page that ships trained weights: the control fed the
    # benchmark row and the untrained fallback, while the trained branch read w.n_qubits and got 8
    # whatever the visitor chose. Telecom carries 7 features on 8 wires, so 7 is the width its data
    # actually asks for and the pad wire is the thing worth measuring.
    ap.add_argument("--qubits", type=int, default=N_QUBITS,
                    help="circuit width; must be >= the feature count")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    # 1 by default: reproducibility of a published artifact outweighs training speed for a model
    # this size. Raise it only for exploration, where the numbers are not going to be shipped.
    ap.add_argument("--threads", type=int, default=1)
    #: Ablation only. `training.entangle` in the artifact records which circuit produced it, so a
    #: file from an ablation run cannot be mistaken for a shipped one.
    ap.add_argument("--no-entangle", action="store_true",
                    help="drop the CNOT ring: identical parameters, no entanglement (ablation)")
    a = ap.parse_args()

    X, y, names, prov = load(a.usecase)
    # SEED TORCH TOO. `--seed` drove only the numpy split permutation, while the parameter init came
    # from an unseeded `torch.randn`, so the same command produced different weights every run -
    # medical moved across 0.9181 / 0.9064 / 0.8947 / 0.8480 accuracy on identical inputs. Published
    # weights that cannot be regenerated are exactly what this project refuses to ship, and the swing
    # was large enough to be mistaken for a real effect of the changes being made alongside it.
    entangle = not a.no_entangle
    torch.manual_seed(a.seed)
    # Only bites on the CPU fallback: with a capable GPU present this trains on CUDA, so the thread
    # count governs nothing here. It is set anyway so a CPU host gets the same guarantee.
    #
    # Kept as a record of a wrong turn, because the wrong answer looked right twice. The 8-class
    # artifact would not regenerate - the same seeded command gave macro F1 0.809846 on some runs
    # and 0.807781 on others - and thread pinning was the first hypothesis. Then PYTHONHASHSEED=0
    # produced three identical runs in a row, which read as a confirmed fix until a fourth run at
    # the same hash seed broke it. Neither was the cause. The cause was `index_add_` in the
    # multiclass readout, whose CUDA implementation accumulates with atomicAdd; see the note there.
    # Both false leads shared a premise worth naming: they assumed the training ran on the CPU,
    # which nothing had checked.
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    # THREE-WAY SPLIT. An earlier version selected the best epoch on the same rows it then reported,
    # which is test-set leakage and makes every published figure optimistic by construction. Epoch
    # selection now happens on a validation split the reported number never sees.
    n = len(y)
    cut, vcut = int(0.7 * n), int(0.85 * n)
    # TRAIN IN THE SPACE THE BROWSER ACTUALLY SUPPLIES. The bundled medical sample is
    # StandardScaler over the full corpus - verified, L1 distance 0.0002 against a recomputed
    # z-score - so it is z-scores that arrive at inference time, not raw units. A first version of
    # this script fitted a min-max scaler on raw WDBC (area 143..2501); feeding it z-scores pinned
    # seven of eight features to -1 and produced <Z0> = 0.0098, a degenerate state that made the
    # TRAINED path exactly as meaningless as the untrained one it replaced. Same class of mistake as
    # the ResNet preprocessing bug already recorded in prep_model_inputs.py.
    #
    # z-scores are unbounded - WDBC spans [-3.11, 6.05] - and RY(z*pi) would wrap the Bloch sphere
    # several times, so the encoding is RY(tanh(z)*pi): bounded, smooth, and order-preserving.
    # WHICH SPACE DOES THE BROWSER SUPPLY? It differs per use case, and getting it wrong is how the
    # first trained model became as meaningless as the untrained one it replaced.
    #
    #   medical  the bundled sample is StandardScaler output (verified, L1 0.0002), so z-scores
    #            arrive at inference time and the browser must NOT rescale.
    #   finance  the bundled sample is RAW and mixed-scale - `amount` is log 0..17.7, `amount_zscore`
    #            is already a z-score, five columns are binary 0/1 - and all 16 bundled rows are found
    #            in this tsv within 1e-3, so the browser supplies exactly these units. It therefore
    #            MUST standardise, with the moments fitted here and shipped alongside.
    #
    # Either way the encoding is RY(tanh(z)*pi): bounded, smooth, order-preserving. tanh on a raw
    # log-amount of 14 would saturate at 1.0 and throw the column away, which is why standardising
    # first is not optional for finance.
    INPUT_SPACE = {"medical": "zscore", "finance": "raw", "telecom": "raw"}[a.usecase]
    mu, sd = X[:cut].mean(0), X[:cut].std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = np.tanh((X - mu) / sd)
    # Pad to the wire count. telecom has 7 features on 8 wires, and the pad is ZERO because
    # RY(tanh(0)*pi) = RY(0) = identity - the wire is genuinely unencoded rather than carrying a
    # constant. A constant would be a zero-variance feature the entangling layers still rotate, which
    # looks like an input at training time and means nothing at inference.
    nq = int(a.qubits)
    if Xs.shape[1] < nq:
        pad = nq - Xs.shape[1]
        print(f"  {Xs.shape[1]} features on {nq} wires: {pad} wire(s) left unencoded (RY(0))")
        Xs = np.hstack([Xs, np.zeros((len(Xs), pad), dtype=Xs.dtype)])
    elif Xs.shape[1] > nq:
        raise SystemExit(f"{Xs.shape[1]} features exceeds {nq} wires")
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7 else "cpu"
    xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    tr, va, te = slice(0, cut), slice(cut, vcut), slice(vcut, n)
    print(f"{a.usecase}: {n} rows, {int(y.sum())} positive; train {cut} / val {vcut-cut} / "
          f"test {n-vcut}; device {dev}")

    # MULTICLASS OR BINARY, decided by the labels rather than by a flag, so a corpus cannot be
    # trained under the wrong head by passing the wrong argument.
    n_classes = int(len(torch.unique(yt)))
    # Class NAMES, not just a count. A per-class F1 keyed by integer is unreadable and, worse, an
    # integer key survives a permuted class map without looking wrong.
    if a.usecase == "telecom":
        from qsh.qml_ids.attacks import ATTACKS
        class_labels = list(ATTACKS)
    else:
        class_labels = None
    readout_q = 0 if n_classes <= 2 else max(1, (n_classes - 1).bit_length())
    n_out = 1 if readout_q == 0 else (1 << readout_q)
    if readout_q and n_out != n_classes:
        raise SystemExit(
            f"{n_classes} classes needs {readout_q} measured qubits giving {n_out} basis states. "
            f"This readout maps basis states to classes one-to-one, so a non-power-of-two class count "
            f"would need a discard rule and there is no defensible one - the alternative in the "
            f"literature discards the surplus states without renormalising.")

    # FLAT and SHARED. Was `(a.layers, nq)` - one independent angle per wire, which is precisely
    # the per-position weighting a convolution replaces with one filter. `param_count` is O(log nq).
    theta = torch.nn.Parameter(
        init_params(nq, readout_qubits=readout_q or 0, layers=a.layers, seed=a.seed).to(dev))
    # Per-class scale and bias. They were SCALARS, which is the other half of why the old readout was
    # binary: one expectation through one affine map is one logit however many classes the labels have.
    scale = torch.nn.Parameter(torch.full((n_out,), 4.0, device=dev) if readout_q
                               else torch.tensor(4.0, device=dev))
    bias = torch.nn.Parameter(torch.zeros(n_out, device=dev) if readout_q
                              else torch.tensor(0.0, device=dev))
    opt = torch.optim.Adam([theta, scale, bias], lr=0.08)
    lossf = torch.nn.CrossEntropyLoss() if readout_q else torch.nn.BCEWithLogitsLoss()
    print(f"  head: {'multiclass' if readout_q else 'binary'}, {n_classes} classes, "
          f"{readout_q or 0} measured qubit(s), {int(theta.numel()) + 2 * n_out} parameters")
    best, best_state = 0.0, None
    for ep in range(a.epochs):
        opt.zero_grad()
        z = circuit(xt[tr], theta, nq, readout_qubits=readout_q, entangle=entangle, layers=a.layers)
        loss = (lossf(z * scale + bias, yt[tr].long()) if readout_q
                else lossf(z * scale + bias, yt[tr]))
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                zt = circuit(xt[va], theta, nq, readout_qubits=readout_q, entangle=entangle, layers=a.layers)
                logits = zt * scale + bias
                pred = (logits.argmax(1).float() if readout_q else (logits > 0).float())
                acc = (pred == yt[va]).float().mean().item()
                # SELECT ON THE METRIC THAT WILL BE REPORTED. Selecting the best epoch by accuracy on
                # an imbalanced problem picks a checkpoint that can afford to ignore the minority
                # class: on finance it cost 0.14 of minority-class F1 (0.6992 at the F1-best epoch
                # against 0.5641 at the accuracy-best one) while accuracy moved by 0.01. The
                # selection criterion has to match the headline or it optimises the wrong thing.
                if readout_q:
                    # MACRO F1, and selecting on it is the point: a checkpoint chosen on accuracy can
                    # afford to drop a class entirely, which is the failure this whole readout exists
                    # to stop being invisible.
                    f1s = []
                    for c in range(n_classes):
                        pc, yc = (pred == c).float(), (yt[va] == c).float()
                        tp = float((pc * yc).sum())
                        fp = float((pc * (1 - yc)).sum())
                        fn = float(((1 - pc) * yc).sum())
                        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
                    f1 = float(np.mean(f1s))
                    score = f1
                else:
                    tp = float((pred * yt[va]).sum())
                    fp = float((pred * (1 - yt[va])).sum())
                    fn = float(((1 - pred) * yt[va]).sum())
                    f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
                    prev = float(yt[va].mean())
                    score = f1 if min(prev, 1 - prev) < 0.25 else acc
                if score > best:
                    best = score
                    best_state = (theta.detach().cpu().numpy().copy(),
                                  scale.detach().cpu().numpy().copy(),
                                  bias.detach().cpu().numpy().copy())
            print(f"  epoch {ep:3d} loss {loss.item():.4f} VAL acc {acc:.4f} VAL minority-F1 {f1:.4f}")

    # Classical baselines on the identical split - the number that decides whether this ships.
    #
    # AND THE RIGHT METRIC, which is not always accuracy. This project treats accuracy on an
    # imbalanced problem as a defect: at finance's 6.33% fraud rate the QCNN scores 0.9692 accuracy
    # against a 0.9367 majority baseline, which reads as a good model and says almost nothing. Its
    # minority-class F1 is 0.6992 against an all-positive baseline of 0.1191 - still a real 5.9x
    # lift, but a very different sentence. Both are recorded; the imbalanced case leads with F1.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    lr = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
    lr_acc = lr.score(Xs[te], y[te])
    # Majority-class ACCURACY. For multiclass that is the largest class share, not
    # max(mean, 1-mean) - which on integer labels is the mean class index and printed
    # "majority-class accuracy 3.6458" on the first multiclass run.
    major = (float(np.bincount(y[te].astype(int)).max() / len(y[te]))
             if len(np.unique(y)) > 2 else float(max(y[te].mean(), 1 - y[te].mean())))
    # PREVALENCE IS A BINARY NOTION. `y.mean()` over 8-class integer labels is the mean CLASS INDEX,
    # not a positive rate - it printed "test prevalence 3.6458" on the first multiclass run, a number
    # with no meaning that the surrounding text then used to decide which metric to lead with. The
    # multiclass corpus here is balanced by construction (equal n_per_class), so the imbalance branch
    # does not apply and macro F1 leads for a different reason: it is the metric a per-class collapse
    # shows up in.
    prevalence = None if readout_q else float(y[te].mean())
    imbalanced = False if readout_q else (min(prevalence, 1 - prevalence) < 0.25)

    with torch.no_grad():
        th_t = torch.tensor(best_state[0], dtype=torch.float32, device=dev)
        sc_t = torch.tensor(best_state[1], dtype=torch.float32, device=dev)
        bi_t = torch.tensor(best_state[2], dtype=torch.float32, device=dev)
        z_all = circuit(xt[te], th_t, nq, readout_qubits=readout_q, entangle=entangle, layers=a.layers)
        logits_all = z_all * sc_t + bi_t
        qcnn_pred = ((logits_all.argmax(1) if readout_q else (logits_all > 0))
                     .int().cpu().numpy())
    # Macro F1 for multiclass - and n_classes_predicted beside it, because a model that stops
    # emitting a class scores 0 on it by construction and macro F1 then averages in a zero that reads
    # as poor discrimination rather than an absent output.
    if readout_q:
        qcnn_f1 = float(f1_score(y[te], qcnn_pred, average="macro", zero_division=0))
        per_class = f1_score(y[te], qcnn_pred, average=None, labels=list(range(n_classes)),
                             zero_division=0)
        n_predicted = int(len(set(qcnn_pred.tolist())))
        print(f"\n  QCNN multiclass: macro F1 {qcnn_f1:.4f}, {n_predicted}/{n_classes} classes "
              f"predicted" + ("" if n_predicted == n_classes else "   <- COLLAPSED"))
        for name, v in sorted(zip(class_labels or range(n_classes), per_class, strict=True),
                              key=lambda kv: kv[1]):
            print(f"    {str(name):24s} {v:.4f}")
    else:
        qcnn_f1 = float(f1_score(y[te], qcnn_pred, zero_division=0))
        per_class, n_predicted = None, None
    # `best` is the SELECTION score (F1 when imbalanced), not accuracy. Recompute accuracy from the
    # selected checkpoint's own predictions - an earlier version printed `best` under an "accuracy"
    # label, so finance reported accuracy 0.8027 when that was its F1.
    qcnn_acc = float((qcnn_pred == y[te]).mean())
    lr_f1 = float(f1_score(y[te], lr.predict(Xs[te]),
                           average=("macro" if readout_q else "binary"), zero_division=0))
    all_pos_f1 = float(2 * prevalence / (1 + prevalence)) if prevalence else 0.0
    if readout_q:
        print(f"\n  {n_classes}-class, balanced by construction -> MACRO F1 leads, because a class "
              f"the model stops predicting scores 0 there and nowhere else")
    else:
        print(f"\n  test prevalence {prevalence:.4f} -> {'IMBALANCED, F1 leads' if imbalanced else 'balanced, accuracy leads'}")
    _flabel = "macro F1        " if readout_q else "minority-class F1"
    print(f"  QCNN  {_flabel} {qcnn_f1:.4f}   accuracy {qcnn_acc:.4f}")
    print(f"  LR    {_flabel} {lr_f1:.4f}   accuracy {lr_acc:.4f}")
    if readout_q:
        print(f"  floors: uniform-random macro F1 {1.0/n_classes:.4f}, constant-predictor macro F1 "
              f"{2.0/(n_classes*(n_classes+1)):.4f}, majority-class accuracy {major:.4f}"
              "   <- without these a macro F1 has no scale")
    else:
        print(f"  baselines: all-positive F1 {all_pos_f1:.4f}   majority-class accuracy {major:.4f}")
    if imbalanced:
        print(f"  headline: F1 {qcnn_f1:.4f} vs all-positive {all_pos_f1:.4f} "
              f"({qcnn_f1/max(all_pos_f1,1e-9):.1f}x). Accuracy {qcnn_acc:.4f} against a {major:.4f} "
              f"majority baseline would flatter it and is NOT the headline.")
    verdict = ("BEATS" if (qcnn_f1 if imbalanced else qcnn_acc) > (all_pos_f1 if imbalanced else major)
               else "DOES NOT beat")
    print(f"  verdict: QCNN {verdict} its baseline; classical LR is "
          f"{'ahead' if (lr_f1 if imbalanced else lr_acc) > (qcnn_f1 if imbalanced else qcnn_acc) else 'behind'}")

    th, sc, bi = best_state
    # THE DIRECTORY IS A FUNCTION OF THE WIDTH, because the artifact is.
    #
    # This wrote `models/qcnn_{usecase}` for every width. `models/qcnn_telecom_q7` exists anyway,
    # because Round 40 trained at 7 wires and MOVED the result by hand - an unrecorded step, and the
    # cost of it is that the reproduce command stored inside `qcnn_telecom_q7/weights.json` does not
    # reproduce `qcnn_telecom_q7`. Run it and it writes `models/qcnn_telecom`, DESTROYING the 8-wire
    # weights the demo serves and that card's headline number describes.
    #
    # The comment forty lines below already states the rule this broke - "a reproduction line that
    # does not reproduce is worse than none, since it gets checked once and trusted after" - and it
    # was written about the `--qubits` FLAG being missing from the recorded string. The flag went in;
    # the location it writes to stayed keyed to the use case alone. The command reproduced the
    # weights and not the artifact.
    #
    # `_q{nq}` is the directory Round 40 created by hand, so nothing moves: the naming that already
    # exists becomes the naming the trainer produces.
    suffix = "" if nq == N_QUBITS else f"_q{nq}"
    out = REPO / "models" / f"qcnn_{a.usecase}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.json").write_text(json.dumps({
        "usecase": a.usecase, "n_qubits": nq, "layers": a.layers,
        # How many wires actually carry data. It equalled n_qubits until telecom brought 7
        # features to 8 wires, and the spare wire is left UNENCODED - RY(tanh(0)*pi) is the
        # identity. The browser reaches the same state by `features[q] ?? 0` with a missing
        # mean defaulting to 0 and a missing std to 1, so the two agree by construction; this
        # field is what lets a test assert that rather than infer it.
        "n_features": len(names),
        "theta": th.tolist(),
        "scale": (sc.tolist() if hasattr(sc, "tolist") else sc),
        "bias": (bi.tolist() if hasattr(bi, "tolist") else bi),
        # The readout convention, shipped AS DATA rather than left implicit. A consumer that reads
        # the bits in a different order gets a permuted class map with identical accuracy, identical
        # macro F1 and identical loss curves, so nothing downstream can detect it - which is why the
        # convention travels with the weights and `tests/test_qcnn_multiclass_readout.py` asserts it.
        "readout": ({"kind": "expval_z0", "n_classes": 2,
                     "note": "single <Z0> through a scalar scale and bias - one logit, binary by "
                             "construction"}
                    if readout_q == 0 else
                    {"kind": "basis_marginal", "n_classes": n_out, "readout_qubits": readout_q,
                     "bit_order": "lsb_first",
                     # THE WIRES, not a formula over wire indices. `c = i & 7` described a readout
                     # over wires 0..m-1, which Round 47 found includes wires pooling discarded.
                     "wires": readout_wires(nq, readout_q, a.layers),
                     "class_index": "c = sum_j bit(wires[j]) << j",
                     "note": f"marginal probabilities of the first {readout_q} SURVIVING wires "
                             f"{readout_wires(nq, readout_q, a.layers)} - the ones the "
                             "fully-connected block acts on - with survivor j as bit j, least "
                             "significant first. NOT the first m wires: pooling keeps every other "
                             "wire, so wires 0..m-1 include discarded pooling controls. NOT "
                             "PennyLane's order either: qml.probs lists the first wire as MOST "
                             "significant. Reading the bits backwards permutes the classes by "
                             "[0,4,2,6,1,5,3,7], of which four are fixed points."}),
        "class_names": (class_labels if readout_q else None),
        "feature_names": names, "input_space": INPUT_SPACE, "encoding": "tanh",
        "feature_mean": mu.tolist(), "feature_std": sd.tolist(),
        "provenance": prov,
        # THE INVOCATION, recorded because it was not. `layers: 6` was in the file, so the
        # architecture was recoverable, but `epochs` and `seed` were not - and the defaults do not
        # reproduce this artifact. Re-running `--usecase telecom` gave macro F1 0.4998 against a
        # published 0.8098, and `--layers 6` alone gave 0.6443; only `--layers 6 --epochs 400`
        # returns 0.809846. That took an epoch sweep to recover a number the demo was already
        # displaying as measured, which is the whole argument for writing it down here.
        # `--qubits` belongs in this string because it changes the weights. The first 7-wire run
        # omitted it, so the recorded command reproduced an EIGHT-wire model - a reproduction line
        # that does not reproduce is worse than none, since it gets checked once and trusted after.
        # THE ARTIFACT RECORDS ITS OWN ARCHITECTURE. Without it a retrained card and a stale
        # weights file are indistinguishable, which is the failure this round's plan named:
        # "renaming the label and shipping weights from the old circuit".
        "training": {"architecture": "qcnn-cong-hur",
                     "layers": a.layers, "epochs": a.epochs, "seed": a.seed, "qubits": nq,
                     "threads": a.threads, "entangle": entangle,
                     "command": f"python scripts/train/qcnn_variational.py --usecase {a.usecase} "
                                f"--layers {a.layers} --epochs {a.epochs} --seed {a.seed} "
                                f"--threads {a.threads} --qubits {nq}",
                     "note": ("--seed drives the numpy split permutation and torch parameter "
                              "init. Re-running this command regenerates this file: five "
                              "consecutive runs gave identical metrics and identical theta to nine "
                              "decimal places. That is true only since the multiclass readout "
                              "stopped using index_add_, whose CUDA atomicAdd made the same seeded "
                              "command land on macro F1 0.809846 or 0.807781 depending on the "
                              "run.")},
        "test_accuracy": round(qcnn_acc, 6),
        "classical_lr_accuracy": round(float(lr_acc), 6),
        "majority_baseline": round(float(major), 6),
        # Named for what was MEASURED, which is not the same metric in both regimes. On two
        # classes `f1_score(average="binary")` is the minority-class F1 and an all-positive
        # predictor is a meaningful floor for it. On K > 2 the same variables hold a MACRO F1,
        # for which "minority class" names nothing and an all-positive predictor does not exist.
        #
        # This is the identical reasoning the `test_prevalence` comment below already applies,
        # and it was applied to one field of three: the 8-class telecom artifact shipped a macro
        # F1 of 0.809846 under `test_minority_f1`, carried `macro_f1_floors` for a metric the
        # file never named, and reported `all_positive_f1: 0.0` - a floor of zero for a baseline
        # that cannot be constructed. Null in the inapplicable regime, so a consumer that reaches
        # for the wrong one gets nothing rather than a plausible number.
        "test_minority_f1": (None if readout_q else round(qcnn_f1, 6)),
        "classical_lr_minority_f1": (None if readout_q else round(lr_f1, 6)),
        "all_positive_f1": (None if readout_q else round(all_pos_f1, 6)),
        "test_macro_f1": (round(qcnn_f1, 6) if readout_q else None),
        "classical_lr_macro_f1": (round(lr_f1, 6) if readout_q else None),
        # Null for multiclass rather than a number: prevalence is a binary notion and the
        # mean of integer class labels is not one. Recording null is what stops a consumer
        # reading a mean class index as a positive rate.
        "test_prevalence": (None if prevalence is None else round(prevalence, 6)),
        "n_classes_predicted": n_predicted,
        "per_class_f1": (None if per_class is None else
                         {str(k): round(float(v), 6)
                          for k, v in zip(class_labels or range(n_classes), per_class,
                                              strict=True)}),
        "macro_f1_floors": (None if not readout_q else {
            "uniform_random": round(1.0 / n_classes, 6),
            "majority_class": round(2.0 / (n_classes * (n_classes + 1)), 6),
            "note": ("Without a floor a macro F1 has no scale. Uniform-random is about 1/K; a "
                     "constant predictor scores 2/(K(K+1)), earning F1 on one class and zero on "
                     "the rest."),
        }),
        # THREE regimes, because the two-way version disagreed with this script's own output. It
        # asked only "is the binary problem imbalanced", so a balanced 8-class run fell through to
        # "accuracy" while the console printed "MACRO F1 leads" a hundred lines earlier. The artifact
        # therefore shipped floors for macro F1 under `macro_f1_floors` and then told every consumer
        # to headline something else, so the metric those floors exist to scale was never surfaced.
        "headline_metric": ("macro_f1" if readout_q else
                            ("minority_class_f1" if imbalanced else "accuracy")),
        # `n - vcut`, NOT `len(y) - cut`. The split is 70/15/15 and the reported metrics are
        # computed on `te = slice(vcut, n)` alone, but this recorded validation PLUS test - so every
        # model card stated a held-out size exactly twice the one the number was measured on
        # (medical "171 test rows" for 86, finance "1200" for 600, telecom "960" for 480). The
        # validation split exists precisely so checkpoint selection never touches the reported rows;
        # counting it as test undoes that separation in the record even though the code respected it.
        "n_train": int(cut), "n_val": int(vcut - cut), "n_test": int(n - vcut),
        "note": ("Variational QCNN trained by backpropagation through a dense state vector. The "
                 "prediction path applies NO QOTP mask: undoing a Pauli frame through Clifford+T is "
                 "not implemented here (T X T-dagger is not Pauli), so hiding and prediction are "
                 "kept separate rather than claiming a correction that does not exist. Simulated, "
                 "not hardware; the classical baseline above is reported so no quantum advantage is "
                 "implied."),
    }, indent=1) + "\n")
    print(f"  wrote {(out / 'weights.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
