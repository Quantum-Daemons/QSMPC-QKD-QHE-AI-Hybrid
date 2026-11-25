"""A binary tree QCNN as its classical contraction: one bilinear map a pooling stage on single-qubit Bloch vectors.

references/satellite-qcnn-tn-preregistration.md, sections C and D and addendum 3. Every shipped binary circuit encodes
feature q on wire q as RY(theta_q)|0>, so wire q starts as the Bloch vector r_q = (1, sin theta_q, 0, cos theta_q).
A stage's filter rounds and pool act on the disjoint pairs (a, b) of its active wires, and the pooled wire b is never
touched again, so the survivor's state depends only on the pair's: r'_k = sum_ij M_kij r_a,i r_b,j with

    M_kij = 1/4 Tr[(sigma_k on a) V (sigma_i on a, sigma_j on b) V^dagger],

V the pair's filter rounds and pool, built here gate for gate as ``qcnn_trained._run_circuit`` applies them. The last
survivor's RZ, RY block and <Z0> are linear in its Bloch vector. The pairs of a stage share one M, since the filter
and the pool angle are shared. This is the tree tensor network the circuit is (Grant et al. 2018): it runs in time
linear in the width, and it is what `scripts/export_qcnn_tree_onnx.py` writes to ONNX and what the CKKS prototype
evaluates one row a slot.
"""

from __future__ import annotations

import numpy as np

from .qcnn_trained import validate

_I = np.eye(2, dtype=complex)
_PAULI = (_I, np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0]).astype(complex))


def _ry(th: float) -> np.ndarray:
    c, s = np.cos(th / 2.0), np.sin(th / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(th: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * th), np.exp(0.5j * th)])


def _on_a(g: np.ndarray) -> np.ndarray:
    """A one-qubit gate on wire a, the low bit of a two-wire index (index = 2 b + a)."""
    return np.kron(_I, g)


def _on_b(g: np.ndarray) -> np.ndarray:
    return np.kron(g, _I)


def _cnot_ab() -> np.ndarray:
    """Control a, target b."""
    m = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        a, b = i & 1, (i >> 1) & 1
        m[(a ^ 0) | ((b ^ a) << 1), i] = 1.0
    return m


def _cry_ba(th: float) -> np.ndarray:
    """Control b, target a: RY(th) on a when b is 1."""
    m = np.eye(4, dtype=complex)
    r = _ry(th)
    idx = (2, 3)                                   # b = 1: a = 0 -> index 2, a = 1 -> index 3
    for i_out, o in enumerate(idx):
        for i_in, j in enumerate(idx):
            m[o, j] = r[i_out, i_in]
    return m


def _layout(weights: dict) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    layout = validate(weights)
    if int((weights.get("readout") or {}).get("readout_qubits", 0) or 0):
        raise ValueError("the tree contraction here is for binary artifacts (a single <Z0>)")
    nq, layers, stages = layout["nq"], int(weights["layers"]), int(layout["stages"])
    if nq & (nq - 1):
        raise ValueError(f"{nq} wires: the contraction here needs a power of two, so that no stage has an odd wire")
    theta = np.asarray(weights["theta"], dtype=float).ravel()
    n_conv = stages * layers * 8
    conv = theta[:n_conv].reshape(stages, layers, 8)
    pool = theta[n_conv:n_conv + stages]
    fc = theta[n_conv + stages:].reshape(-1, 2)
    return nq, stages, conv, pool, fc


def stage_unitary(conv_stage: np.ndarray, pool_angle: float) -> np.ndarray:
    """V for one pair: every filter round, then the pool, in the order `_run_circuit` applies them."""
    v = np.eye(4, dtype=complex)
    for w8 in conv_stage:
        for g in (_on_a(_ry(w8[0])), _on_a(_rz(w8[1])), _on_b(_ry(w8[2])), _on_b(_rz(w8[3])), _cnot_ab(),
                  _on_a(_rz(w8[5])), _on_a(_ry(w8[4])), _on_b(_rz(w8[7])), _on_b(_ry(w8[6]))):
            v = g @ v
    return _cry_ba(pool_angle) @ v


def bilinear_map(v: np.ndarray) -> np.ndarray:
    """M (4, 4, 4), real: M[k, i, j] = 1/4 Tr[(sigma_k on a) V (sigma_i on a, sigma_j on b) V^dagger]."""
    m = np.zeros((4, 4, 4))
    for k in range(4):
        sk = _on_a(_PAULI[k])
        for i in range(4):
            for j in range(4):
                m[k, i, j] = 0.25 * np.trace(sk @ v @ np.kron(_PAULI[j], _PAULI[i]) @ v.conj().T).real
    return m


def network(weights: dict) -> dict:
    """The stage maps and the readout vector: everything the contraction needs, as plain arrays."""
    nq, stages, conv, pool, fc = _layout(weights)
    maps = [bilinear_map(stage_unitary(conv[s], float(pool[s]))) for s in range(stages)]
    r = _ry(float(fc[0, 0])) @ _rz(float(fc[0, 1]))
    readout = np.array([0.5 * np.trace(_PAULI[3] @ r @ _PAULI[i] @ r.conj().T).real for i in range(4)])
    return {"n_qubits": nq, "maps": maps, "readout": readout}


def encoded_bloch(weights: dict, X: np.ndarray) -> np.ndarray:
    """(rows, wires, 4) extended Bloch vectors of the encoded rows; X in the artifact's input space."""
    layout = validate(weights)
    nq, nf, c = layout["nq"], layout["nf"], layout["c"]
    x = np.asarray(X, dtype=float)[:, :nf]
    if weights.get("input_space") == "raw":
        mu = np.asarray(weights["feature_mean"], dtype=float)[:nf]
        sd = np.asarray(weights["feature_std"], dtype=float)[:nf]
        x = (x - mu) / np.where(sd == 0, 1.0, sd)
    th = np.zeros((len(x), nq))
    th[:, :nf] = c * np.tanh(x) * np.pi
    out = np.zeros((len(x), nq, 4))
    out[:, :, 0] = 1.0
    out[:, :, 1] = np.sin(th)
    out[:, :, 3] = np.cos(th)
    return out


def contract(net: dict, bloch: np.ndarray) -> np.ndarray:
    """<Z0> for each row, from its wires' Bloch vectors: the stages, then the readout vector."""
    r = bloch
    for m in net["maps"]:
        r = np.einsum("kij,nwi,nwj->nwk", m, r[:, 0::2], r[:, 1::2])
    return r[:, 0] @ net["readout"]


def readout(weights: dict, X: np.ndarray) -> np.ndarray:
    return contract(network(weights), encoded_bloch(weights, X))
