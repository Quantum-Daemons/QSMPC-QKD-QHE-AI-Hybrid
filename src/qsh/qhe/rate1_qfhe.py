r"""Rate-1 Quantum Fully Homomorphic Encryption (hybrid QOTP + classical FHE).

Construction of Chardouvelis-Döttling-Malavolta (IACR ePrint 2020/1454, TCC 2021 --
``references/Rate-1_Quantum_Fully_Homomorphic_Encryption.pdf``): a quantum state is
encrypted with a Quantum One-Time Pad, and the (classical) QOTP keys are encrypted
with a classical FHE scheme:

    ct = ( QOTP.Enc(otk, |psi>) ,  FHE.Enc(pk, otk) )

The quantum part has the same dimension as the plaintext, so the *ciphertext/plaintext
ratio approaches 1* ("rate-1") -- decisive for transferring large encrypted states
(e.g. medical images). This module wires QOTP (:mod:`~qsh.qhe.qotp`) to a classical
key-encapsulation stand-in and reports the achieved rate, the metric the benchmark
tracks (target <= 1.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pqc.hybrid_kem import HybridKEMRecipient, encapsulate
from .qotp import decrypt, encrypt, pauli_key


@dataclass
class Rate1Ciphertext:
    """A rate-1 QFHE ciphertext: encrypted state + encapsulated QOTP key."""

    enc_state: np.ndarray          # QOTP-encrypted statevector (quantum part)
    key_ciphertext: bytes          # classically-encrypted QOTP key material
    n_qubits: int


def encrypt_state(state: np.ndarray, recipient_pub) -> tuple[Rate1Ciphertext, np.ndarray, np.ndarray]:
    """Encrypt ``state`` (QOTP) and encapsulate the key to ``recipient_pub``.

    Returns (ciphertext, a, b) -- the keys are also returned for local verification;
    in deployment only the recipient (who holds the matching secret key) recovers them.
    """
    n = int(np.log2(state.size))
    a, b = pauli_key(n)
    enc = encrypt(state, a, b)
    # classically protect the QOTP key (here: derive a one-time pad from a hybrid-KEM
    # shared secret and mask the 2n key bits -- stands in for FHE.Enc on the keys).
    ct, shared = encapsulate(recipient_pub, length=max(32, (2 * n + 7) // 8))
    keybits = np.concatenate([a, b]).astype(np.uint8)
    keybytes = np.packbits(keybits).tobytes()
    import hashlib

    pad = hashlib.shake_256(shared).digest(len(keybytes))
    masked = bytes(k ^ p for k, p in zip(keybytes, pad))
    key_ct = _frame(ct.mlkem_ciphertext) + _frame(ct.x25519_ephemeral_public) + _frame(masked)
    return Rate1Ciphertext(enc, key_ct, n), a, b


def _frame(b: bytes) -> bytes:
    """Length-prefix a byte field (4-byte big-endian length) for robust parsing."""
    return len(b).to_bytes(4, "big") + b


def _unframe(buf: bytes, off: int) -> tuple[bytes, int]:
    n = int.from_bytes(buf[off:off + 4], "big")
    off += 4
    return buf[off:off + n], off + n


def ciphertext_to_plaintext_ratio(ct: Rate1Ciphertext) -> float:
    """Achieved rate = total ciphertext bytes / plaintext (quantum state) bytes."""
    quantum_bytes = ct.enc_state.nbytes
    total = quantum_bytes + len(ct.key_ciphertext)
    return total / quantum_bytes


def decrypt_state(ct: Rate1Ciphertext, recipient: HybridKEMRecipient) -> np.ndarray:
    """Recover |psi> using the recipient's hybrid secret key."""
    from ..pqc.hybrid_kem import HybridCiphertext
    import hashlib

    mlkem_ct, off = _unframe(ct.key_ciphertext, 0)
    eph_pub, off = _unframe(ct.key_ciphertext, off)
    masked, _ = _unframe(ct.key_ciphertext, off)
    shared = recipient.decapsulate(HybridCiphertext(mlkem_ct, eph_pub),
                                   length=max(32, (2 * ct.n_qubits + 7) // 8))
    pad = hashlib.shake_256(shared).digest(len(masked))
    keybytes = bytes(m ^ p for m, p in zip(masked, pad))
    keybits = np.unpackbits(np.frombuffer(keybytes, dtype=np.uint8))[: 2 * ct.n_qubits]
    a, b = keybits[: ct.n_qubits], keybits[ct.n_qubits:]
    return decrypt(ct.enc_state, a, b)
