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


def _mix_qkd(session_key: bytes, qkd_key: bytes | None) -> bytes:
    """Optionally fold a QKD-derived key into the session key (PQC+QKD hybrid)."""
    if not qkd_key:
        return session_key
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
    pub: HybridPublicKey, *, qkd_key: bytes | None = None
) -> tuple[HybridCiphertext, SecureChannel]:
    """Initiator side: returns (ciphertext to send, established channel)."""
    ct, shared = encapsulate(pub, length=32)
    return ct, SecureChannel(_mix_qkd(shared, qkd_key))


def establish_responder(
    recipient: HybridKEMRecipient, ct: HybridCiphertext, *, qkd_key: bytes | None = None
) -> SecureChannel:
    """Responder side: derive the same channel from the received ciphertext."""
    shared = recipient.decapsulate(ct, length=32)
    return SecureChannel(_mix_qkd(shared, qkd_key))
