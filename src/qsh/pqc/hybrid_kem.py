"""Hybrid post-quantum key encapsulation: ML-KEM + X25519.

Implements the NSA/NIST/IETF-recommended *hybrid* construction (a classical ECDH
combined with a NIST PQC KEM) so the session key is secure as long as *either*
primitive holds. This is the report's "PQC-first hybrid" mandate (CNSA 2.0, FIPS 203)
and defends against Harvest-Now-Decrypt-Later: the X25519 half gives today's
assurance, the ML-KEM half gives quantum resistance.

Backends: ML-KEM via Open Quantum Safe ``liboqs`` (``oqs``); X25519 + HKDF via
``cryptography``. The two shared secrets are bound together with HKDF-SHA256 over a
domain-separated transcript, per the hybrid-KEM combiner pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import oqs
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

DEFAULT_KEM = "ML-KEM-768"   # NIST FIPS 203, ~AES-192 security category 3
_COMBINER_INFO = b"qsh/hybrid-kem/v1 ML-KEM-768+X25519"


@dataclass
class HybridPublicKey:
    """Recipient's public material (safe to publish)."""

    kem_name: str
    mlkem_public: bytes
    x25519_public: bytes


@dataclass
class HybridCiphertext:
    """What the sender transmits to the recipient."""

    mlkem_ciphertext: bytes
    x25519_ephemeral_public: bytes


def _combine(ss_pq: bytes, ss_classical: bytes, transcript: bytes, length: int) -> bytes:
    """HKDF-SHA256 combiner binding both shared secrets + a transcript."""
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=_COMBINER_INFO + b"|" + transcript,
    ).derive(ss_pq + ss_classical)


class HybridKEMRecipient:
    """Holds the long-term hybrid private key and decapsulates."""

    def __init__(self, kem_name: str = DEFAULT_KEM) -> None:
        self.kem_name = kem_name
        self._kem = oqs.KeyEncapsulation(kem_name)
        self._mlkem_public = self._kem.generate_keypair()
        self._x_priv = X25519PrivateKey.generate()
        self._x_pub = self._x_priv.public_key().public_bytes_raw()

    @property
    def public_key(self) -> HybridPublicKey:
        return HybridPublicKey(self.kem_name, self._mlkem_public, self._x_pub)

    def decapsulate(self, ct: HybridCiphertext, *, length: int = 32) -> bytes:
        """Recover the shared secret from a :class:`HybridCiphertext`."""
        ss_pq = self._kem.decap_secret(ct.mlkem_ciphertext)
        eph_pub = X25519PublicKey.from_public_bytes(ct.x25519_ephemeral_public)
        ss_classical = self._x_priv.exchange(eph_pub)
        transcript = self._mlkem_public + self._x_pub + ct.x25519_ephemeral_public
        return _combine(ss_pq, ss_classical, transcript, length)

    def free(self) -> None:
        self._kem.free()

    def __enter__(self) -> HybridKEMRecipient:
        return self

    def __exit__(self, *exc) -> None:
        self.free()


def encapsulate(pub: HybridPublicKey, *, length: int = 32) -> tuple[HybridCiphertext, bytes]:
    """Sender side: produce (ciphertext, shared_secret) for recipient ``pub``."""
    with oqs.KeyEncapsulation(pub.kem_name) as kem:
        mlkem_ct, ss_pq = kem.encap_secret(pub.mlkem_public)
    eph = X25519PrivateKey.generate()
    eph_pub = eph.public_key().public_bytes_raw()
    recipient_x = X25519PublicKey.from_public_bytes(pub.x25519_public)
    ss_classical = eph.exchange(recipient_x)
    transcript = pub.mlkem_public + pub.x25519_public + eph_pub
    shared = _combine(ss_pq, ss_classical, transcript, length)
    return HybridCiphertext(mlkem_ct, eph_pub), shared
