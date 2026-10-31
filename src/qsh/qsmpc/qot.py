"""Quantum Oblivious Transfer (BB84 + bit commitment) and quantum-safe PSI.

Follows Zhang et al., *Experimental SMPC from Quantum Oblivious Transfer with Bit
Commitment*, npj Quantum Information 2026 (``references/Experimental_secure_
multiparty_computation_from_quantum_oblivious_transfer_with_bit_commitment.pdf``):
a 1-out-of-2 OT is built from decoy-state BB84 plus a hash-based bit commitment,
and OT yields Private Set Intersection (PSI). PSI on bank blacklists is the paper's
headline application and our Finance/AML use case.

Two layers are provided, and they are NOT connected: the PSI below is not built from the QOT.
  * :func:`quantum_oblivious_transfer` -- a sketch of the 1-of-2 QOT information structure
    (BB84 basis mismatch). It is not a secure OT: the SHA-256 commitment is computed and
    never opened or checked, and the unchosen message is never transmitted at all.
  * :func:`private_set_intersection` -- a classical keyed-hash PSI: both sets are HMAC-SHA256'd
    under one session key that BOTH parties hold. What holds: the hashes travel inside the
    PQC-protected session, so no third party on the channel sees either list. What does NOT
    hold: privacy from the other party. Over a small ID space either party can hash every
    candidate ID and recover the other's list. A production PSI would use an OPRF-based
    protocol. Used by the finance demo.
"""

from __future__ import annotations

import hashlib
import hmac
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


# --------------------------------------------------------------------------- #
#  Keyed-hash Private Set Intersection (classical; run over the PQC session)
# --------------------------------------------------------------------------- #
@dataclass
class PSIResult:
    """Result of a private set intersection."""

    intersection: list
    size_a: int
    size_b: int
    intersection_size: int


def _keyed_hash(key: bytes, element) -> bytes:
    return hmac.new(key, str(element).encode(), hashlib.sha256).digest()


def private_set_intersection(set_a, set_b, shared_key: bytes) -> PSIResult:
    """Compute |A ∩ B| and the common elements using a keyed-hash PSI.

    Both parties derive ``shared_key`` from the session (PQC, with simulated QKD bits folded in by
    HKDF when the full deployment has them), so the keyed hashes travel confidentially: no third
    party on the channel sees either set or its hashes.

    WHAT THIS DOES NOT PROTECT: the non-matching elements from the OTHER party. Whoever receives
    the other side's hashes holds the same key, so over an enumerable identifier space it can
    hash every candidate offline and read the other's set, without deviating from the protocol.
    The Python demo IDs are a short prefix plus a 7-digit number (about 9e6 candidates per
    prefix). This is the naive-hashing failure Pinkas, Schneider and Zohner describe
    (USENIX Security 2014). A production PSI would use an OPRF-based protocol.

    For the finance demo this finds accounts flagged by both banks; it does not keep either bank's
    other flagged accounts from the other bank.
    """
    set_a = list(set_a)
    set_b = list(set_b)
    hb = {_keyed_hash(shared_key, e): e for e in set_b}
    inter = []
    for e in set_a:
        h = _keyed_hash(shared_key, e)
        if h in hb:
            inter.append(e)
    return PSIResult(
        intersection=inter,
        size_a=len(set_a),
        size_b=len(set_b),
        intersection_size=len(inter),
    )


def multiparty_psi(sets: list, shared_key: bytes) -> PSIResult:
    """PSI across >=2 parties: elements present in *every* party's set.

    The same keyed hash as :func:`private_set_intersection`, with the same limit: every party holds
    the key, so none of the sets is private from the others over a small ID space.
    """
    if len(sets) < 2:
        raise ValueError("need at least two sets")
    hashed = [set(_keyed_hash(shared_key, e) for e in s) for s in sets]
    common_hashes = set.intersection(*hashed)
    # map back via the first set
    inter = [e for e in sets[0] if _keyed_hash(shared_key, e) in common_hashes]
    return PSIResult(
        intersection=inter,
        size_a=len(sets[0]),
        size_b=min(len(s) for s in sets[1:]),
        intersection_size=len(inter),
    )
