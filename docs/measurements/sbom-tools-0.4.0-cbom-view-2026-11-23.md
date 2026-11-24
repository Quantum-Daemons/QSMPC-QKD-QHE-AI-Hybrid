> **Dated tool output, not a compliance statement.** This is the CBOM part of what sbom-tools 0.4.0 (MIT; release asset
> `sbom-tools-linux-x86_64.tar.gz`, SHA-256 `94e2447f7048059b26f23dbbca87537f3711ddf2871abf668c27829b6f3b86f8`) printed on
> 2026-11-23 for `web/public/cbom/qsh.cdx.json` (SHA-256 `d708c1ef0a9c6c66c09064208264b22568f47a24c45813834ee56b08bf21b2af`), offline:
> `sbom-tools view qsh.cdx.json --bom-type cbom --as-of 2026-11-23 --offline -o markdown`. Below the line the text is the
> tool's, unaltered, up to its CRA section, which is left out here: it scores product metadata (a PSIRT contact, a supplier,
> a CE marking, a support period) that a cryptography inventory does not carry. CI keeps the whole view, unaltered, as an
> artefact of every run. "Quantum Readiness" and its percentage are the tool's own measure. This repository's reading of the
> same document is on the Validations page; where the two differ is in `sbom-tools-comparison-2026-11-23.json` beside this file.

---

# SBOM Report

**Format:** CycloneDX
**Version:** 1.7
**Name:** qsh

## Summary

| Metric | Value |
|--------|-------|
| Total Components | 26 |
| Total Dependencies | 29 |
| Total Vulnerabilities | 0 |
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |

## Components

| Name | Version | Ecosystem | License | Vulnerabilities |
|------|---------|-----------|---------|-----------------|
| qsh | 0.86.0 | - | - | 0 |
| ML-KEM-768 | - | - | - | 0 |
| ML-KEM-1024 | - | - | - | 0 |
| X25519 | - | - | - | 0 |
| ECDH P-256 | - | - | - | 0 |
| HKDF-SHA256 | - | - | - | 0 |
| ML-KEM + X25519 two-step HKDF combiner | - | - | - | 0 |
| AES-256-GCM | - | - | - | 0 |
| ChaCha20-Poly1305 | - | - | - | 0 |
| ML-DSA-65 | - | - | - | 0 |
| ML-DSA-87 | - | - | - | 0 |
| SHA-256 | - | - | - | 0 |
| SHA-384 | - | - | - | 0 |
| SHA-512 | - | - | - | 0 |
| SHAKE256 | - | - | - | 0 |
| HMAC-SHA256 | - | - | - | 0 |
| BLAKE2s-256 | - | - | - | 0 |
| OPRF(ristretto255, SHA-512) | - | - | - | 0 |
| CKKS | - | - | - | 0 |
| Threshold CKKS | - | - | - | 0 |
| Threshold BGV | - | - | - | 0 |
| Threshold BGV, 2 of 3 | - | - | - | 0 |
| BFV | - | - | - | 0 |
| WRK17b authenticated garbling | - | - | - | 0 |
| Shamir secret sharing | - | - | - | 0 |
| Platform CSPRNG | - | - | - | 0 |

## Cryptographic Inventory

| Name | Asset Type | Family | Primitive | Security Level | Quantum Level |
|------|-----------|--------|-----------|---------------|--------------|
| ML-KEM-768 | algorithm | ML-KEM | kem | - | 3 |
| ML-KEM-1024 | algorithm | ML-KEM | kem | - | 5 |
| X25519 | algorithm | ECDH | key-agree | 128 | 0 |
| ECDH P-256 | algorithm | ECDH | key-agree | 128 | 0 |
| HKDF-SHA256 | algorithm | HKDF | kdf | - | - |
| ML-KEM + X25519 two-step HKDF combiner | algorithm | - | combiner | - | - |
| AES-256-GCM | algorithm | AES | ae | 256 | 5 |
| ChaCha20-Poly1305 | algorithm | ChaCha20 | ae | 256 | - |
| ML-DSA-65 | algorithm | ML-DSA | signature | - | 3 |
| ML-DSA-87 | algorithm | ML-DSA | signature | - | 5 |
| SHA-256 | algorithm | SHA-2 | hash | - | - |
| SHA-384 | algorithm | SHA-2 | hash | - | - |
| SHA-512 | algorithm | SHA-2 | hash | - | - |
| SHAKE256 | algorithm | SHA-3 | xof | - | - |
| HMAC-SHA256 | algorithm | HMAC | mac | - | - |
| BLAKE2s-256 | algorithm | - | hash | - | - |
| OPRF(ristretto255, SHA-512) | algorithm | - | other | 128 | 0 |
| CKKS | algorithm | - | other | 128 | - |
| Threshold CKKS | algorithm | - | other | 128 | - |
| Threshold BGV | algorithm | - | other | 128 | - |
| Threshold BGV, 2 of 3 | algorithm | - | other | 128 | - |
| BFV | algorithm | - | other | 128 | - |
| WRK17b authenticated garbling | algorithm | - | other | - | 0 |
| Shamir secret sharing | algorithm | - | other | - | - |
| Platform CSPRNG | algorithm | - | drbg | - | - |

**Quantum Readiness:** 20% (5 safe / 25 total algorithms)
