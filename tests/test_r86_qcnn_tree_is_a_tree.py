"""The finance and medical QCNNs are what their cards call them: a tree, simulable in linear time (Round 86).

WHY THIS EXISTS. Both cards say "a tree of two-qubit blocks, and therefore classically simulable in linear time at any
width". The check behind that sentence lived in a script outside the repository (audit item O9, checklist QC36), and what
the repository tested was part of the property: that the survivors of each pooling stage are a product state.

WHAT IS ASSERTED. A second evaluator follows the structure the card names and nothing else: one 2x2 density matrix per
wire, each pair of a stage merged through one 4x4 block and traced down to one wire. It reproduces the trainer's full
state vector on every artifact declared a tree, on rows it has never seen, and the number of merges it needs is linear
in the register. It does not reproduce the same circuit with its angles shuffled, so it is not agreeing by accident.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "train"))
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")
qcnn = pytest.importorskip("qcnn_variational")

#: Globbed, and selected by what the artifact declares: a new width or a new page is in scope the moment it ships.
TREES = [p for p in sorted((REPO / "web/public/data").glob("*/qcnn_weights*.json"))
         if json.loads(p.read_text())["circuit"]["name"] == "tree"]
ROWS = 64

I2 = np.eye(2, dtype=complex)
P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
CNOT_FIRST_TO_SECOND = np.kron(P0, I2) + np.kron(P1, X)


def ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t: float) -> np.ndarray:
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


def cry_second_controls_first(t: float) -> np.ndarray:
    return np.kron(I2, P0) + np.kron(ry(t), P1)


def block(w: np.ndarray) -> np.ndarray:
    """The shared 8-angle filter on (first, second), in the trainer's gate order."""
    first, second = lambda g: np.kron(g, I2), lambda g: np.kron(I2, g)
    u = first(ry(w[0]))
    u = first(rz(w[1])) @ u
    u = second(ry(w[2])) @ u
    u = second(rz(w[3])) @ u
    u = CNOT_FIRST_TO_SECOND @ u
    u = first(rz(w[5])) @ u
    u = first(ry(w[4])) @ u
    u = second(rz(w[7])) @ u
    return second(ry(w[6])) @ u


def keep_first(rho4: np.ndarray) -> np.ndarray:
    return rho4.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)


def tree_z0(encoded: np.ndarray, artifact: dict, theta: np.ndarray) -> tuple[float, int]:
    """<Z> on the last survivor, from one-wire states only, and how many merges that took."""
    nq, layers = int(artifact["n_qubits"]), int(artifact["layers"])
    shapes = qcnn.param_shapes(nq, 0, layers)
    n_conv = int(np.prod(shapes["conv"]))
    stages = shapes["pool"][0]
    conv, pool, fc = theta[:n_conv].reshape(shapes["conv"]), theta[n_conv:n_conv + stages], theta[n_conv + stages:].reshape(-1, 2)
    rho: dict[int, np.ndarray] = {}
    for wire in range(nq):
        v = ry(float(encoded[wire]) * np.pi) @ np.array([1, 0], dtype=complex)
        rho[wire] = np.outer(v, v.conj())
    active, merges = list(range(nq)), 0
    for stage in range(stages):
        u = np.eye(4, dtype=complex)
        for layer in range(layers):
            u = block(conv[stage, layer]) @ u
        u = cry_second_controls_first(float(pool[stage])) @ u
        pairs = list(zip(active[0::2], active[1::2], strict=False))
        for a, b in pairs:
            rho[a] = keep_first(u @ np.kron(rho[a], rho[b]) @ u.conj().T)
            merges += 1
        if len(active) % 2 == 1:      # an odd register's last wire is pooled into the first, never convolved
            v = cry_second_controls_first(float(pool[stage]))
            rho[active[0]] = keep_first(v @ np.kron(rho[active[0]], rho[active[-1]]) @ v.conj().T)
            merges += 1
        active = [a for a, _ in pairs]
    assert len(active) == 1
    g = ry(float(fc[0, 0])) @ rz(float(fc[0, 1]))
    final = g @ rho[active[0]] @ g.conj().T
    return float(np.real(np.trace(np.diag([1, -1]) @ final))), merges


def _rows(artifact: dict) -> np.ndarray:
    z = np.random.default_rng(3).normal(0, 1, (ROWS, int(artifact["n_features"])))
    encoded = float(artifact.get("encoding_scale", 1.0)) * np.tanh(z)
    return np.hstack([encoded, np.zeros((ROWS, int(artifact["n_qubits"]) - encoded.shape[1]))])


def _state_vector(rows: np.ndarray, artifact: dict, theta: np.ndarray) -> np.ndarray:
    return qcnn.circuit(torch.tensor(rows, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32),
                        nq=int(artifact["n_qubits"]), readout_qubits=0, layers=int(artifact["layers"])).detach().numpy()


def test_the_scan_found_the_tree_artifacts() -> None:
    assert {p.parent.name for p in TREES} == {"finance", "medical"}


@pytest.mark.parametrize("path", TREES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_one_wire_states_reproduce_the_full_state_vector(path: Path) -> None:
    artifact = json.loads(path.read_text())
    theta = np.array(artifact["theta"], dtype=float)
    rows = _rows(artifact)
    tree = np.array([tree_z0(r, artifact, theta)[0] for r in rows])
    assert np.max(np.abs(tree - _state_vector(rows, artifact, theta))) < 1e-5


@pytest.mark.parametrize("path", TREES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_agreement_is_about_this_circuit(path: Path) -> None:
    artifact = json.loads(path.read_text())
    theta = np.array(artifact["theta"], dtype=float)
    shuffled = np.random.default_rng(7).permutation(theta)
    rows = _rows(artifact)
    tree = np.array([tree_z0(r, artifact, shuffled)[0] for r in rows])
    assert np.max(np.abs(tree - _state_vector(rows, artifact, theta))) > 0.01


@pytest.mark.parametrize("path", TREES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_work_is_linear_in_the_register(path: Path) -> None:
    artifact = json.loads(path.read_text())
    _, merges = tree_z0(_rows(artifact)[0], artifact, np.array(artifact["theta"], dtype=float))
    assert merges == int(artifact["n_qubits"]) - 1
