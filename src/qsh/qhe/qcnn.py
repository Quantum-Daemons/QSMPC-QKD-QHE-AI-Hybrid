r"""Quantum Convolutional Neural Network (QCNN) + QHE private inference (WS4).

The model is the standard MERA-style QCNN (Cong et al.) built in PennyLane:
alternating two-qubit "convolution" unitaries and "pooling" that halves the qubit
count, ending in a single measured qubit -> binary classifier. This mirrors the QCNN
used by Ortega & Martín-Delgado (arXiv:2602.12712, 2026) for perfectly-secure
delegated training / private inference.

Two things are provided:
  * :class:`QCNN` -- a trainable PennyLane classifier (the application model).
  * :func:`qotp_clifford_homomorphic_demo` -- a *correct* QHE demonstration on
    Clifford circuits: the server evaluates on the QOTP-encrypted state and the
    client decrypts by tracking Pauli-key updates, recovering the exact result
    while the server only ever sees the maximally-mixed (hidden) state.

Non-Clifford layers (the trainable rotations) require T-gate magic-state
corrections in a full deployment; their cost is reported by :meth:`QCNN.t_count`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qotp import decrypt, encrypt, encrypted_density_matrix, pauli_key


@dataclass
class QCNNConfig:
    n_qubits: int = 8


class QCNN:
    """A small MERA-style QCNN binary classifier (PennyLane)."""

    def __init__(self, n_qubits: int = 8, seed: int | None = None) -> None:
        import pennylane as qml

        self.qml = qml
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build()
        rng = np.random.default_rng(seed)
        self.n_params = self._param_count()
        self.params = rng.normal(0, 0.1, self.n_params)

    def _layers(self) -> list[list[int]]:
        """Pairs of wires for each conv/pool layer, halving each round."""
        wires = list(range(self.n_qubits))
        layers = []
        while len(wires) > 1:
            pairs = [[wires[i], wires[i + 1]] for i in range(0, len(wires) - 1, 2)]
            layers.append(pairs)
            wires = wires[::2]  # pooling keeps every other wire
        return layers

    def _param_count(self) -> int:
        return sum(len(pairs) * 3 for pairs in self._layers())

    def _build(self) -> None:
        qml = self.qml
        layers = self._layers()

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, params):
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
            k = 0
            for pairs in layers:
                for (a, b) in pairs:
                    qml.CNOT(wires=[a, b])
                    qml.RY(params[k], wires=a); k += 1
                    qml.RY(params[k], wires=b); k += 1
                    qml.RZ(params[k], wires=b); k += 1
            last = layers[-1][0][0] if layers else 0
            return qml.expval(qml.PauliZ(last))

        self._circuit = circuit

    def forward(self, x, params=None) -> float:
        """Expectation in [-1, 1]; map to a probability with (1 - z) / 2."""
        p = self.params if params is None else params
        x = np.asarray(x, dtype=float)[: self.n_qubits]
        if x.size < self.n_qubits:
            x = np.pad(x, (0, self.n_qubits - x.size))
        return float(self._circuit(x, p))

    def predict_proba(self, x, params=None) -> float:
        return (1.0 - self.forward(x, params)) / 2.0

    def t_count(self, eps: float | None = None) -> int:
        """Rough fault-tolerant T-gate budget (each rotation ~ O(log(1/eps)) T gates)."""
        if eps is None:
            from ..config import get_config

            eps = get_config().qhe.sk_eps
        per_rotation = int(np.ceil(3 * np.log2(1 / eps)))  # Solovay-Kitaev-ish estimate
        n_rotations = self._param_count()
        return per_rotation * n_rotations


def qotp_clifford_homomorphic_demo(n: int = 3, seed: int | None = None) -> dict:
    """Correct QHE demo: evaluate a Clifford circuit on a QOTP-encrypted state.

    Returns a report dict including the recovered-vs-plaintext fidelity (==1) and,
    for small n, proof that the encrypted state is maximally mixed (security).
    """
    rng = np.random.default_rng(seed)
    # random input state
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)

    # plaintext reference: apply H on qubit 0 then CNOT(0->1) (a Clifford circuit)
    from .qotp import _apply_1q, update_key_cnot, update_key_h

    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def clifford(state):
        s = _apply_1q(state, H, 0, n)
        # CNOT(0,1)
        sr = s.reshape([2] * n)
        sr = np.moveaxis(sr, [0, 1], [0, 1])
        flat = sr.reshape(2, 2, -1).copy()
        flat[1] = flat[1][::-1]  # flip target when control=1
        return flat.reshape(-1)

    ref = clifford(psi)

    # QHE path: client encrypts, server applies the same Clifford, client tracks keys
    a, b = pauli_key(n, rng)
    enc = encrypt(psi, a, b)
    server_out = clifford(enc)
    # track Pauli key updates through H(0) and CNOT(0,1)
    a0, b0 = update_key_h(int(a[0]), int(b[0]))
    a[0], b[0] = a0, b0
    ac, bc, at, bt = update_key_cnot(int(a[0]), int(b[0]), int(a[1]), int(b[1]))
    a[0], b[0], a[1], b[1] = ac, bc, at, bt
    recovered = decrypt(server_out, a, b)

    fidelity = float(abs(np.vdot(ref, recovered)) ** 2)

    report = {
        "n_qubits": n,
        "fidelity_recovered_vs_plaintext": round(fidelity, 12),
        "correct": fidelity > 1 - 1e-9,
    }
    if n <= 3:
        rho = encrypted_density_matrix(psi)
        maximally_mixed = np.eye(2 ** n) / 2 ** n
        report["encrypted_is_maximally_mixed"] = bool(
            np.allclose(rho, maximally_mixed, atol=1e-9)
        )
    return report
