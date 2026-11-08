"""Quantum Oblivious Transfer (BB84 + bit commitment), as a sketch.

Follows Zhang et al., *Experimental SMPC from Quantum Oblivious Transfer with Bit
Commitment*, npj Quantum Information 12, 76 (2026), https://www.nature.com/articles/s41534-026-01219-w
(its CC BY-NC-ND 4.0 PDF is not bundled; see ``references/REFERENCES.md``):
a 1-out-of-2 OT is built from decoy-state BB84 plus a hash-based bit commitment,
and OT yields Private Set Intersection (PSI). PSI on bank blacklists is the paper's
headline application and our Finance/AML use case.

:func:`quantum_oblivious_transfer` is a sketch of the 1-of-2 QOT information structure (BB84 basis
mismatch). It is not a secure OT: the SHA-256 commitment is computed and never opened or checked,
and the unchosen message is never transmitted at all. Nothing is built on it: the finance use
case's PSI is the classical OPRF PSI of :mod:`qsh.qsmpc.psi` (RFC 9497 over ristretto255), not a
PSI from this QOT.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
#  1-of-2 Quantum Oblivious Transfer (faithful primitive)
# --------------------------------------------------------------------------- #
def _commit(value: bytes, nonce: bytes) -> bytes:
    return hashlib.sha256(nonce + value).digest()


@dataclass
class QOTTranscript:
    """Diagnostic transcript of a 1-of-2 QOT run."""

    choice: int
    received: bytes
    test_passed: bool
    n_raw: int
    n_good: int


def quantum_oblivious_transfer(
    m0: bytes,
    m1: bytes,
    choice: int,
    *,
    n: int = 512,
    test_fraction: float = 0.25,
    seed: int | None = None,
) -> QOTTranscript:
    """1-out-of-2 OT, as a sketch: the receiver gets ``m_choice``.

    The protocol it sketches (semi-honest): Alice sends ``n`` BB84 states; Bob measures in
    random bases and *commits* to them (SHA-256). Alice tests a random subset,
    then reveals her bases; positions where bases matched form Bob's "good" set for
    the chosen branch, the rest mask the unchosen branch. A real deployment uses
    decoy states + error-correcting codes (BCH) + privacy amplification.

    What this code does is narrower, on equal-length byte messages: it computes the commitment and
    discards it (nothing opens or checks it), the test passes by construction (no noise and no
    adversary are simulated), and it pads and returns only the chosen message. The unchosen one is
    never sent, so "the receiver learns nothing about the other" holds trivially, not by the
    protocol's security.
    """
    if choice not in (0, 1):
        raise ValueError("choice must be 0 or 1")
    if len(m0) != len(m1):
        raise ValueError("messages must be equal length")
    rng = np.random.default_rng(seed)

    a_bits = rng.integers(0, 2, n)
    a_bases = rng.integers(0, 2, n)
    b_bases = rng.integers(0, 2, n)
    b_bits = np.where(a_bases == b_bases, a_bits, rng.integers(0, 2, n))  # mismatched = random

    # Bob commits to his bases+bits (hash-based bit commitment)
    nonce = rng.integers(0, 256, n).astype(np.uint8).tobytes()
    _ = _commit(b_bits.astype(np.uint8).tobytes(), nonce)  # commitment (kept by Alice)

    # Alice tests a random subset, then reveals all bases
    n_test = int(test_fraction * n)
    test_idx = rng.choice(n, n_test, replace=False)
    test_passed = bool(np.all(b_bits[test_idx][a_bases[test_idx] == b_bases[test_idx]]
                              == a_bits[test_idx][a_bases[test_idx] == b_bases[test_idx]]))

    # "good" positions for Bob's chosen branch = matching bases (outside the test set)
    mask = np.ones(n, bool); mask[test_idx] = False
    good = np.flatnonzero(mask & (a_bases == b_bases))

    # one-time-pad the chosen message with a key derived from the good positions
    key_material = a_bits[good].astype(np.uint8).tobytes()
    pad = hashlib.shake_256(key_material + bytes([choice])).digest(len(m0))
    chosen = m1 if choice == 1 else m0
    received = bytes(c ^ p for c, p in zip(chosen, pad))
    recovered = bytes(c ^ p for c, p in zip(received, pad))  # Bob removes the pad

    return QOTTranscript(
        choice=choice, received=recovered, test_passed=test_passed,
        n_raw=n, n_good=int(good.size),
    )
