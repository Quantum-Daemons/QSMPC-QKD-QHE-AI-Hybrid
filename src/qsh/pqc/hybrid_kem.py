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
_COMBINER_INFO = b"qsh/hybrid-kem/v2 ML-KEM-768+X25519"


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
        # c1 || c2 || ek1 || ek2 - see the note on `encapsulate`.
        transcript = (ct.mlkem_ciphertext + ct.x25519_ephemeral_public
                      + self._mlkem_public + self._x_pub)
        return _combine(ss_pq, ss_classical, transcript, length)

    def free(self) -> None:
        self._kem.free()

    def __enter__(self) -> HybridKEMRecipient:
        return self

    def __exit__(self, *exc) -> None:
        self.free()


def encapsulate(pub: HybridPublicKey, *, length: int = 32) -> tuple[HybridCiphertext, bytes]:
    """Sender side: produce (ciphertext, shared_secret) for recipient ``pub``.

    Transcript order and contents follow NIST SP 800-227 (September 2025) section 4.6.3, whose
    IND-CCA-preserving combiner is

        KeyCombineCCA_H:  H(K1, K2, c1, c2, ek1, ek2, domain_sep)

    with inputs ek1, c1, K1 from the first KEM and ek2, c2, K2 from the second. Until Round 38 the
    transcript here was ek1 || ek2 || c2 - the two encapsulation keys and the X25519 ephemeral
    public - and OMITTED c1, the ML-KEM ciphertext. That is the wrong half to drop: the same section
    states that reference [25] "does not incorporate encapsulation keys into the combiner, as this
    is not needed to achieve the IND-CCA-preserving property", so the ciphertexts are the load-
    bearing inputs and the encapsulation keys are the optional ones, useful for binding the shared
    secret to party identities.

    The failure direction is the one a hybrid exists to hedge. If ML-KEM is the broken component and
    an adversary mauls c1 into a c1' that decapsulates to the same K1, nothing in the old transcript
    observed c1, so the composite derived a byte-identical session key - X25519 remaining sound did
    not rescue it. That is precisely the non-preservation the section warns about.

    Scoping it honestly: this demo runs in one process with no network transport and no exposed
    decapsulation oracle, so this was a conformance and claim-accuracy defect rather than an
    exploitable break in the shipped artefact. It is fixed because the project's claim is
    standards-conformance.

    The domain separator moves to v2 with the change. Deriving a different key under an unchanged
    label is exactly what domain separation exists to prevent.
    """
    with oqs.KeyEncapsulation(pub.kem_name) as kem:
        mlkem_ct, ss_pq = kem.encap_secret(pub.mlkem_public)
    eph = X25519PrivateKey.generate()
    eph_pub = eph.public_key().public_bytes_raw()
    recipient_x = X25519PublicKey.from_public_bytes(pub.x25519_public)
    ss_classical = eph.exchange(recipient_x)
    # c1 || c2 || ek1 || ek2, matching SP 800-227 section 4.6.3's argument order after K1 and K2.
    transcript = mlkem_ct + eph_pub + pub.mlkem_public + pub.x25519_public
    shared = _combine(ss_pq, ss_classical, transcript, length)
    return HybridCiphertext(mlkem_ct, eph_pub), shared
