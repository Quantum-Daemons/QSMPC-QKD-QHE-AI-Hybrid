"""Quantum-safe transport: hybrid-KEM session + AES-256-GCM AEAD.

A minimal authenticated-encryption channel built on the hybrid KEM
(:mod:`~qsh.pqc.hybrid_kem`). This is the report's "ML-KEM hybrid layer over MCP"
(Gopher 2026) — the secure pipe over which orchestration stages and AI agents
exchange data. A QKD-derived key can be mixed in for a defence-in-depth
(PQC + QKD) session. NSA publishes no recommendation for this; CNSA 2.0 does not require hybrid
products at all (see :mod:`~qsh.pqc.hybrid_kem`). This line said "matching the NSA hybrid
recommendation" until Round 75.

Key agreement (PQC, or PQC+QKD) -> AEAD record layer (AES-256-GCM).
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .hybrid_kem import (
    HybridCiphertext,
    HybridKEMRecipient,
    HybridPublicKey,
    encapsulate,
)


def _mix_qkd_its(pqc_key: bytes, qkd_key: bytes) -> bytes:
    """Toeplitz + QLHL hybrid combiner after Giestinhas & Spiller (arXiv:2603.26907), computationally
    seeded, so NOT information-theoretic as implemented.

    WHY IT IS COMPUTATIONAL HERE. The Toeplitz seed below is SHAKE-256 of a label and of the same two
    keys it extracts from, so seed and input are fully correlated. The paper names a seed
    independent of the input, conditioned on any adversary, as a necessary condition for its quantum
    leftover hash lemma, so its leakage-resilience property is not established for this code, and
    the security of the output rests on SHAKE-256 and the inputs, like HKDF's. In the browser and in
    the full-deployment MCP endpoints the QKD input is a fixed 32-byte constant, which carries no
    entropy at all.

    NOT "secure-if-either": plain XOR already gives that, and the paper never uses the phrase. It
    explicitly declines to claim superiority over XOR, saying strong seeded extractors "present
    different characteristics". The property it does claim is narrower — if the combined output AND
    one input key are compromised, the remaining key material retains quantifiable min-entropy —
    bought at the cost of longer keys and worse computational complexity.

    A modified-Toeplitz + Leftover-Hash-Lemma extractor over ``pqc_key || qkd_key``. WITH AN
    INDEPENDENT SEED, the paper's construction lets the 256-bit output retain the min-entropy of
    whichever input stays uncompromised; this code does not have one. The Toeplitz matrix is derived
    from both keys via SHAKE-256 (a computationally-seeded extractor; the information-theoretic
    variant would use an independent public seed, paper Sec. 6). Deterministic, so initiator and
    responder derive the same key. Returns 32 bytes.
    """
    import hashlib

    import numpy as np

    from ..qrng.simulator import toeplitz_extract

    combined = np.unpackbits(np.frombuffer(pqc_key + qkd_key, dtype=np.uint8))   # the extractor's input bits
    out_bits = 256                                                              # 32-byte AES-256 session key
    need_t = out_bits + combined.size - 1
    tseed = hashlib.shake_256(b"qsh/its-combiner/v1\x00" + pqc_key + qkd_key).digest((need_t + 7) // 8)
    tbits = np.unpackbits(np.frombuffer(tseed, dtype=np.uint8))[:need_t]
    out = toeplitz_extract(combined, out_len=out_bits, toeplitz_bits=tbits)
    return np.packbits(out.astype(np.uint8)).tobytes()[:32]


def _mix_qkd(session_key: bytes, qkd_key: bytes | None, *, mode: str | None = None) -> bytes:
    """Fold a QKD-derived key into the session key (PQC+QKD hybrid).

    ``mode`` selects the combiner (``None`` -> from config): ``"hkdf"`` = HKDF-SHA256, a dual-PRF KDF
    (secure if EITHER input is a secret high-entropy value, under the HMAC dual-PRF assumption);
    ``"its_toeplitz"`` = the Toeplitz+QLHL combiner (:func:`_mix_qkd_its`), seeded with SHAKE-256 of
    its own inputs, so computational as implemented, not information-theoretic.
    """
    if not qkd_key:
        return session_key
    if mode is None:
        from ..config import get_config
        mode = getattr(get_config().crypto, "hybrid_combiner_mode", "hkdf")
    if mode == "its_toeplitz":
        return _mix_qkd_its(session_key, qkd_key)
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=qkd_key,
        info=b"qsh/transport/pqc+qkd",
    ).derive(session_key)


@dataclass
class SecureChannel:
    """An established AEAD channel with a derived 256-bit session key."""

    session_key: bytes
    _counter: int = 0
    #: First counter value this channel may NOT use. Two channels that share a key are kept apart by
    #: disjoint counter ranges (`mcp_bridge.py` starts the responder at 2**40), and a range is only
    #: disjoint if it is enforced: GCM with a repeated (key, nonce) leaks the authentication key.
    _limit: int = 1 << 96

    def seal(self, plaintext: bytes, associated_data: bytes = b"") -> bytes:
        """Encrypt+authenticate a record. Nonce = 12-byte big-endian counter."""
        if self._counter >= self._limit:
            raise RuntimeError(f"nonce range exhausted at {self._counter}: re-key before sealing again")
        nonce = self._counter.to_bytes(12, "big")
        self._counter += 1
        return nonce + AESGCM(self.session_key).encrypt(nonce, plaintext, associated_data)

    def open(self, record: bytes, associated_data: bytes = b"") -> bytes:
        """Decrypt+verify a record produced by :meth:`seal`."""
        nonce, ct = record[:12], record[12:]
        return AESGCM(self.session_key).decrypt(nonce, ct, associated_data)


def establish_initiator(
    pub: HybridPublicKey, *, qkd_key: bytes | None = None, combiner_mode: str | None = None
) -> tuple[HybridCiphertext, SecureChannel]:
    """Initiator side: returns (ciphertext to send, established channel)."""
    ct, shared = encapsulate(pub, length=32)
    return ct, SecureChannel(_mix_qkd(shared, qkd_key, mode=combiner_mode))


def establish_responder(
    recipient: HybridKEMRecipient, ct: HybridCiphertext, *, qkd_key: bytes | None = None,
    combiner_mode: str | None = None,
) -> SecureChannel:
    """Responder side: derive the same channel from the received ciphertext."""
    shared = recipient.decapsulate(ct, length=32)
    return SecureChannel(_mix_qkd(shared, qkd_key, mode=combiner_mode))
