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
    _signer: oqs.Signature

    def sign(self, message: bytes) -> bytes:
        return self._signer.sign(message)

    def free(self) -> None:
        self._signer.free()


def generate_keypair(sig_name: str = DEFAULT_SIG) -> SignatureKeypair:
    signer = oqs.Signature(sig_name)
    public = signer.generate_keypair()
    return SignatureKeypair(sig_name, public, signer)


def verify(message: bytes, signature: bytes, public_key: bytes, sig_name: str = DEFAULT_SIG) -> bool:
    """Verify ``signature`` over ``message`` under ``public_key``.

    A public key of the wrong length is rejected before liboqs sees it, and so is a signature longer
    than the parameter set allows. liboqs-python up to 0.16.0.1 copied a short public key into a
    buffer of the expected size, zero-padded, and verified against that: advisory GHSA-w696-4vrr-x6x3
    (2026-10-28) shows an ML-DSA key of 32 bytes or less accepting forged signatures. 0.16.0.2 fixes
    the wrapper; this check holds whichever version is installed, as `hybrid_kem._require_length`
    does for the KEM's inputs.
    """
    with oqs.Signature(sig_name) as verifier:
        if len(public_key) != verifier.details["length_public_key"]:
            return False
        if not signature or len(signature) > verifier.details["length_signature"]:
            return False
        return bool(verifier.verify(message, signature, public_key))
