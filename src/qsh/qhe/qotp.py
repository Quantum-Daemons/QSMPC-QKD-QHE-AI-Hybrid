r"""Quantum One-Time Pad (QOTP) — the encryption layer of perfectly-secure QHE.

A quantum state is encrypted by applying a random Pauli  X^a Z^b  to each qubit.
Averaged over the uniformly-random key (a, b), any state becomes the maximally
mixed state I/2^n -> information-theoretic ("perfect") security, independent of the
server's compute power (Ortega & Martín-Delgado, arXiv:2602.12712, 2026 --
``references/Reverse_Delegated_Training_and_Private_Inference_via_Perfectly-Secure_
Quantum_Homomorphic_Encryption.pdf``).

Clifford gates can be evaluated *homomorphically* by updating the Pauli keys (the
Clifford group normalises the Pauli group); only T gates need a magic-state /
teleportation correction. This module implements the statevector QOTP and the
Clifford key-update rules used by :mod:`~qsh.qhe.qcnn`.
"""

from __future__ import annotations

import numpy as np

_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _apply_1q(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply a single-qubit ``gate`` to qubit ``q`` of an n-qubit statevector."""
    st = state.reshape([2] * n)
    st = np.tensordot(gate, st, axes=([1], [q]))
    st = np.moveaxis(st, 0, q)
    return st.reshape(-1)


def pauli_key(n: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fresh QOTP key (a, b) in {0,1}^n x {0,1}^n."""
    rng = rng or np.random.default_rng()
    return rng.integers(0, 2, n), rng.integers(0, 2, n)


def encrypt(state: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Encrypt: apply X^{a_i} Z^{b_i} to each qubit i."""
    n = int(np.log2(state.size))
    out = state.copy()
    for q in range(n):
        if b[q]:
            out = _apply_1q(out, _Z, q, n)
        if a[q]:
            out = _apply_1q(out, _X, q, n)
    return out


def decrypt(state: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Decrypt: apply the inverse Pauli (X then Z) to each qubit."""
    n = int(np.log2(state.size))
    out = state.copy()
    for q in range(n):
        if a[q]:
            out = _apply_1q(out, _X, q, n)
        if b[q]:
            out = _apply_1q(out, _Z, q, n)
    return out


def encrypted_density_matrix(state: np.ndarray) -> np.ndarray:
    """Average encryption over ALL Pauli keys -> proves it equals I / 2^n.

    Returns rho = (1/4^n) sum_{a,b} E_{a,b}|psi><psi| E_{a,b}^dagger, which is the
    maximally mixed state for any input -> the QOTP leaks nothing.
    """
    n = int(np.log2(state.size))
    dim = state.size
    rho = np.zeros((dim, dim), dtype=complex)
    for a in range(dim):
        abits = np.array([(a >> i) & 1 for i in range(n)])
        for b in range(dim):
            bbits = np.array([(b >> i) & 1 for i in range(n)])
            enc = encrypt(state, abits, bbits)
            rho += np.outer(enc, enc.conj())
    return rho / (4 ** n)


# -- Clifford key-update rules (homomorphic evaluation without decrypting) ----
def update_key_h(a: int, b: int) -> tuple[int, int]:
    """H X^a Z^b = X^b Z^a H  -> keys (a,b) -> (b,a)."""
    return b, a


def update_key_cnot(a_c: int, b_c: int, a_t: int, b_t: int) -> tuple[int, int, int, int]:
    """CNOT propagation: (a_c,b_c,a_t,b_t) -> (a_c, b_c ^ b_t, a_c ^ a_t, b_t)."""
    return a_c, b_c ^ b_t, a_c ^ a_t, b_t


def pauli_concealment_demo(n: int = 3, seed: int | None = None) -> dict:
    """Demonstrate Pauli-gate concealment = circuit/output privacy (report WS4(c)/(f)).

    The server returns its computation result under a random Pauli frame. A client that
    knows the frame decrypts it exactly; a malicious client guessing the wrong key
    recovers essentially random data. Returns fidelities proving the result (and hence
    the server's circuit effect) is hidden without the correct key.
    """
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)

    a, b = pauli_key(n, rng)                       # the true frame (known only to the client)
    server_out = encrypt(psi, a, b)               # server result under the random Pauli frame

    recovered = decrypt(server_out, a, b)          # legitimate client
    wa, wb = pauli_key(n, rng)                      # malicious client's wrong guess
    while np.array_equal(wa, a) and np.array_equal(wb, b):
        wa, wb = pauli_key(n, rng)
    wrong = decrypt(server_out, wa, wb)

    fid_correct = float(abs(np.vdot(psi, recovered)) ** 2)
    fid_wrong = float(abs(np.vdot(psi, wrong)) ** 2)
    return {
        "n_qubits": n,
        "fidelity_correct_key": round(fid_correct, 6),
        "fidelity_wrong_key": round(fid_wrong, 6),
        "circuit_private": fid_correct > 0.99 and fid_wrong < 0.9,
    }
