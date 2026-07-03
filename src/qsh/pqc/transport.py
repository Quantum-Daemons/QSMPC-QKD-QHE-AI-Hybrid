"""Quantum-safe transport: hybrid-KEM session + AES-256-GCM AEAD.

A minimal authenticated-encryption channel built on the hybrid KEM
(:mod:`~qsh.pqc.hybrid_kem`). This is the report's "ML-KEM hybrid layer over MCP"
(Gopher 2026) — the secure pipe over which orchestration stages and AI agents
exchange data. A QKD-derived key can be mixed in for a defence-in-depth
(PQC + QKD) session, matching the NSA hybrid recommendation.

Key agreement (PQC, or PQC+QKD) -> AEAD record layer (AES-256-GCM).
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .hybrid_kem import (
    HybridCiphertext,
    HybridKEMRecipient,
    HybridPublicKey,
    encapsulate,
)


def _mix_qkd_its(pqc_key: bytes, qkd_key: bytes) -> bytes:
    """Information-theoretic "secure-if-either" hybrid combiner (Giestinhas & Spiller 2026).

    A modified-Toeplitz + Leftover-Hash-Lemma strong extractor over ``pqc_key || qkd_key``: the 256-bit
    output retains the min-entropy of WHICHEVER input stays uncompromised (the QKD side is information-
    theoretic, the PQC side computational) -- unlike a naive XOR/concatenation, which leaks linearly. The
    Toeplitz matrix is derived from both keys via SHAKE-256 (a computationally-seeded strong extractor; a
    fully information-theoretic variant would use an independent public seed, paper Sec. 6). Deterministic,
    so initiator and responder derive the same key. Returns 32 bytes.
    """
    import hashlib

    import numpy as np

    from ..qrng.simulator import toeplitz_extract

    combined = np.unpackbits(np.frombuffer(pqc_key + qkd_key, dtype=np.uint8))   # info-theoretic input material
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
    ``"its_toeplitz"`` = the information-theoretic secure-if-either Toeplitz+LHL combiner (:func:`_mix_qkd_its`).
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

    def seal(self, plaintext: bytes, associated_data: bytes = b"") -> bytes:
        """Encrypt+authenticate a record. Nonce = 12-byte big-endian counter."""
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
