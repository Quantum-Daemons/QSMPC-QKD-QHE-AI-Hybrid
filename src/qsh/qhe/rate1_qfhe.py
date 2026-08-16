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

WHAT THIS MODULE IS NOT, stated because the name promises more than the code delivers.
There is **no ``Eval``**. The public surface is ``encrypt_state``,
``ciphertext_to_plaintext_ratio`` and ``decrypt_state`` - encrypt, ship, decrypt. Homomorphic
evaluation is the entire point of the Chardouvelis-Doettling-Malavolta construction, and it
requires homomorphically UPDATING the encrypted Pauli keys as the circuit is applied; none of that
is here. What is implemented is **rate-1 quantum transport**: a QOTP-encrypted state whose keys
travel under a hybrid KEM, at a measured ct/pt ratio.

The FHE half is a self-declared stand-in, not an FHE scheme - the QOTP key is masked with a pad
derived from a hybrid-KEM shared secret. That is stated in ``encrypt_state`` and repeated here
because a reader who only skims the module header would otherwise take "Rate-1 QFHE" at face value.

The measured ratio is size-dependent and approaches 1 from above, because the encapsulated key is a
fixed ~1135 bytes while the state grows: **1.2769 at 8 qubits, 1.0693 at 10, 1.0173 at 12**. The
1.32 figure in ``config/defaults.py`` is the LITERATURE value feeding a static Overview KPI, not a
measurement of this code.
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
    import hashlib

    from ..pqc.hybrid_kem import HybridCiphertext

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
