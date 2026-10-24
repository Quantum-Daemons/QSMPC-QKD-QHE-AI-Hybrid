r"""Quantum One-Time Pad (QOTP) — the encryption layer of perfectly-secure QHE.

A quantum state is encrypted by applying a random Pauli  X^a Z^b  to each qubit.
Averaged over the uniformly-random key (a, b), any state becomes the maximally
mixed state I/2^n -> information-theoretic ("perfect") security, independent of the
server's compute power (Ortega & Martín-Delgado, arXiv:2602.12712, 2026 --
https://arxiv.org/abs/2602.12712; the PDF is not redistributed here - its arXiv licence grants
third parties nothing, see ``DATA_LICENSES.md``).

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


def update_key_s(a: int, b: int) -> tuple[int, int]:
    """S X^a Z^b = X^a Z^(a^b) S, up to a global phase -> keys (a,b) -> (a, a ^ b).

    The third generator, and the one that was missing. {H, S, CNOT} generates the Clifford group, so
    without S the homomorphic demo could only evaluate the subgroup reachable from H and CNOT - which
    is why the existing demonstration was a fixed two-gate circuit rather than an arbitrary one.

    Derivation, since a sign error here is invisible in the fidelity of a single fixed circuit:
    ``S X S-dagger = Y = i X Z``, so conjugating ``X^a Z^b`` by S sends it to ``X^a Z^(a xor b)`` with a
    phase that is global and therefore unobservable. The ``a`` in the new ``b`` is exactly the X part
    leaking into the Z part, which is what makes S non-trivial for frame tracking while H merely swaps.
    """
    return a, a ^ b


def pauli_output_concealment_demo(n: int = 3, seed: int | None = None) -> dict:
    """Show that a state under a random Pauli frame is unreadable without the key.

    SCOPE, and a retraction. This function was called ``pauli_concealment_demo`` and its docstring
    claimed to demonstrate "circuit/output privacy" for report items WS4(c) and WS4(f) - circuit
    privacy against a malicious server, and hiding the server's circuit from a malicious client.
    **It does not, because it applies no circuit.** ``server_out`` below is ``encrypt(psi, ...)``:
    the input state under a mask. There is no gate, no unitary, and therefore no "circuit effect"
    to conceal. What is actually demonstrated is that QOTP hides a STATE, which is real, standard,
    and strictly weaker than circuit privacy.

    A second defect made the overclaim hard to see. ``fidelity_correct_key`` is
    ``decrypt(encrypt(psi, a, b), a, b)`` compared against ``psi``, which is 1.0 by construction for
    any input - and it used to be half of the ``circuit_private`` verdict, so the verdict was carried
    by a term that could not fail. ``tests/test_round2.py`` asserted both the 1.0 and the verdict, so
    a green test read as confirmation of a claim nothing had tested. That is the same tautology this
    project found and fixed in ``qhe/private_inference.py``; this was the last instance.

    The correct-key round trip is still returned, because it is a useful *self-check* that
    ``encrypt``/``decrypt`` are inverses - but it is named for what it is and excluded from the
    verdict, which now rests only on the falsifiable half: a wrong key must destroy the state.

    WHAT A REAL CIRCUIT-PRIVACY DEMONSTRATION WOULD NEED. Apply an actual circuit on the server and
    propagate the Pauli frame through it. That is well defined for Clifford circuits, where a Pauli
    conjugated by a Clifford is again a Pauli, so the client can update its frame and decrypt. It
    breaks at T, since ``T X T-dagger`` is not Pauli - the difficulty Ortega and Martin-Delgado's
    scheme exists to address and which this repository does not implement. Until that exists, no
    document here may cite this function as evidence of circuit privacy.
    """
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)

    a, b = pauli_key(n, rng)                       # the true frame (known only to the client)
    masked = encrypt(psi, a, b)                    # the STATE under a random Pauli frame, not a result

    roundtrip = decrypt(masked, a, b)              # legitimate client
    wa, wb = pauli_key(n, rng)                      # malicious client's wrong guess
    while np.array_equal(wa, a) and np.array_equal(wb, b):
        wa, wb = pauli_key(n, rng)
    wrong = decrypt(masked, wa, wb)

    fid_roundtrip = float(abs(np.vdot(psi, roundtrip)) ** 2)
    fid_wrong = float(abs(np.vdot(psi, wrong)) ** 2)
    return {
        "n_qubits": n,
        # Named for what it is: an encrypt/decrypt inverse check, 1.0 by construction. NOT evidence.
        "roundtrip_fidelity_selfcheck": round(fid_roundtrip, 6),
        "fidelity_wrong_key": round(fid_wrong, 6),
        # The only falsifiable claim here, and deliberately not called `circuit_private`.
        "output_concealed": fid_wrong < 0.9,
        "applies_a_circuit": False,
    }
