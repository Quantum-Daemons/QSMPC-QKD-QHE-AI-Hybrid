"""Post-quantum digital signatures (ML-DSA / FIPS 204) via liboqs.

Used to authenticate orchestration messages and pipeline artifacts so the hybrid
transport is authenticated, not just confidential. ML-DSA (Dilithium) is the NIST
primary lattice signature; an optional classical Ed25519 co-signature can be layered
for a hybrid signature if required by policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import oqs

DEFAULT_SIG = "ML-DSA-65"   # NIST FIPS 204, security category 3


@dataclass
class SignatureKeypair:
    """A public/secret ML-DSA keypair (secret kept by the signer)."""

    sig_name: str
    public_key: bytes
    _signer: "oqs.Signature"

    def sign(self, message: bytes) -> bytes:
        return self._signer.sign(message)

    def free(self) -> None:
        self._signer.free()


def generate_keypair(sig_name: str = DEFAULT_SIG) -> SignatureKeypair:
    signer = oqs.Signature(sig_name)
    public = signer.generate_keypair()
    return SignatureKeypair(sig_name, public, signer)


def verify(message: bytes, signature: bytes, public_key: bytes, sig_name: str = DEFAULT_SIG) -> bool:
    """Verify ``signature`` over ``message`` under ``public_key``."""
    with oqs.Signature(sig_name) as verifier:
        return bool(verifier.verify(message, signature, public_key))
