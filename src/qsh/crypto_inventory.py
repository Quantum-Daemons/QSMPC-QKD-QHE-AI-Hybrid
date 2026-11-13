"""What cryptography this repository's code calls, as data (Round 86).

`scripts/generate_cbom.py` turns this into a CycloneDX 1.7 document, `web/public/cbom/qsh.cdx.json`, which the Validations page renders. It is hand-written and
checked rather than scanned: each entry names files that must still contain the call, and `tests/test_r86_cbom.py` fails on any cryptographic import that no entry
accounts for. A scanner was not used because the one built for this job, CBOMkit's sonar-cryptography, carries SonarSource's source-available licence.

It covers the primitives this repository's own code selects or calls. It does not describe the internals of the libraries that implement them, and it says nothing about
whether any of this is validated: nothing here is FIPS 140-3 validated.

`nist_quantum_level` is the NIST security category of the entry, 0 where a cryptographically relevant quantum computer breaks it, and None where this project does not rate it.
"""

from __future__ import annotations

from dataclasses import dataclass

BROWSER = "browser"
FULL = "full deployment"


@dataclass(frozen=True)
class CryptoAsset:
    ref: str
    name: str
    primitive: str                  # CycloneDX algorithmProperties.primitive
    family: str | None              # CycloneDX algorithm family registry, where it has one
    functions: tuple[str, ...]      # CycloneDX cryptoFunctions
    role: str
    standard: str
    scope: tuple[str, ...]
    evidence: tuple[tuple[str, str], ...]        # (file, text it must contain)
    parameter_set: str | None = None
    nist_quantum_level: int | None = None
    classical_bits: int | None = None
    mode: str | None = None
    curve: str | None = None
    oid: str | None = None


BOTH = (BROWSER, FULL)

ASSETS: tuple[CryptoAsset, ...] = (
    CryptoAsset("ml-kem-768", "ML-KEM-768", "kem", "ML-KEM", ("keygen", "encapsulate", "decapsulate"),
                "The post-quantum half of the session key, by default; the MCP channel", "FIPS 203", BOTH,
                (("web/src/lib/clientCompute.ts", "ml_kem768"), ("src/qsh/pqc/hybrid_kem.py", 'DEFAULT_KEM = "ML-KEM-768"')),
                parameter_set="768", nist_quantum_level=3, oid="2.16.840.1.101.3.4.4.2"),
    CryptoAsset("ml-kem-1024", "ML-KEM-1024", "kem", "ML-KEM", ("keygen", "encapsulate", "decapsulate"),
                "The post-quantum half of the session key, when selected", "FIPS 203", BOTH,
                (("web/src/lib/clientCompute.ts", "ml_kem1024"),), parameter_set="1024", nist_quantum_level=5, oid="2.16.840.1.101.3.4.4.3"),
    CryptoAsset("x25519", "X25519", "key-agree", "ECDH", ("keygen", "keyderive"),
                "The classical half of the session key; federated mask seeds; the WireGuard handshake test", "RFC 7748", BOTH,
                (("web/src/lib/clientCompute.ts", '"@noble/curves/ed25519.js"'), ("web/src/lib/clientQFL.ts", "x25519"), ("src/qsh/pqc/hybrid_kem.py", "X25519PrivateKey")),
                nist_quantum_level=0, classical_bits=128, curve="other/Curve25519", oid="1.3.101.110"),
    CryptoAsset("ecdh-p256", "ECDH P-256", "key-agree", "ECDH", ("keygen", "keyderive"),
                "The IKEv2 key schedule test on the Validations page", "NIST SP 800-56A", (BROWSER,),
                (("web/src/lib/netkeys/ikev2.ts", '"@noble/curves/nist.js"'),), nist_quantum_level=0, classical_bits=128, curve="nist/P-256"),
    CryptoAsset("hkdf-sha256", "HKDF-SHA256", "kdf", "HKDF", ("keyderive",),
                "Session key derivation; federated mask seeds; the MCP channel", "RFC 5869", BOTH,
                (("web/src/lib/hybridCombiner.ts", "@noble/hashes/hkdf.js"), ("src/qsh/pqc/hybrid_kem.py", "HKDF(")),
                parameter_set="SHA-256", oid="1.2.840.113549.1.9.16.3.29"),
    CryptoAsset("hybrid-combiner", "ML-KEM + X25519 two-step HKDF combiner", "combiner", None, ("keyderive",),
                "Combines the two shared secrets with the transcript", "NIST SP 800-227 section 4.6", BOTH,
                (("web/src/lib/hybridCombiner.ts", "qsh/hybrid-kem/v2"), ("src/qsh/pqc/hybrid_kem.py", "qsh/hybrid-kem/v2"))),
    CryptoAsset("aes-256-gcm", "AES-256-GCM", "ae", "AES", ("encrypt", "decrypt", "tag"),
                "Seals the PSI's messages, the MCP channel and the full deployment's records", "NIST SP 800-38D", BOTH,
                (("web/src/lib/clientMcp.ts", '"AES-GCM"'), ("src/qsh/pqc/transport.py", "AESGCM(")),
                parameter_set="256", nist_quantum_level=5, classical_bits=256, mode="gcm", oid="2.16.840.1.101.3.4.1.46"),
    CryptoAsset("chacha20-poly1305", "ChaCha20-Poly1305", "ae", "ChaCha20", ("encrypt", "decrypt", "tag"),
                "The WireGuard handshake test on the Validations page", "RFC 8439", (BROWSER,),
                (("web/src/lib/netkeys/noiseIKpsk2.ts", "chacha20poly1305"),), classical_bits=256),
    CryptoAsset("ml-dsa-65", "ML-DSA-65", "signature", "ML-DSA", ("keygen", "sign", "verify"),
                "Signs each run's result, by default; the MCP channel", "FIPS 204", BOTH,
                (("web/src/lib/clientCompute.ts", "ml_dsa65"), ("src/qsh/pqc/signatures.py", "oqs.Signature")),
                parameter_set="65", nist_quantum_level=3, oid="2.16.840.1.101.3.4.3.18"),
    CryptoAsset("ml-dsa-87", "ML-DSA-87", "signature", "ML-DSA", ("keygen", "sign", "verify"),
                "Signs each run's result, when selected", "FIPS 204", BOTH,
                (("web/src/lib/clientCompute.ts", "ml_dsa87"),), parameter_set="87", nist_quantum_level=5, oid="2.16.840.1.101.3.4.3.19"),
    CryptoAsset("sha-256", "SHA-256", "hash", "SHA-2", ("digest",),
                "Inside HKDF and HMAC; digests of published files, including the check of every model file fetched from the Hub against its recorded digest; the account hashes of the PSI among banks", "FIPS 180-4", BOTH,
                (("web/src/lib/clientMultiPsi.ts", '"SHA-256"'), ("web/src/lib/hybridCombiner.ts", "@noble/hashes/sha2.js"), ("web/src/lib/pinnedFetch.ts", 'crypto.subtle.digest("SHA-256"')), oid="2.16.840.1.101.3.4.2.1"),
    CryptoAsset("sha-384", "SHA-384", "hash", "SHA-2", ("digest",),
                "The IKEv2 key schedule test", "FIPS 180-4", (BROWSER,), (("web/src/lib/netkeys/ikev2.ts", "sha384"),), oid="2.16.840.1.101.3.4.2.2"),
    CryptoAsset("sha-512", "SHA-512", "hash", "SHA-2", ("digest",),
                "The IKEv2 key schedule test; the OPRF's suite", "FIPS 180-4", BOTH,
                (("web/src/lib/netkeys/ikev2.ts", "sha512"), ("src/qsh/qsmpc/oprf.py", "sha512")), oid="2.16.840.1.101.3.4.2.3"),
    CryptoAsset("shake256", "SHAKE256", "xof", "SHA-3", ("digest",),
                "Expands federated mask seeds; seeds the Toeplitz combiner", "FIPS 202", BOTH,
                (("web/src/lib/clientQFL.ts", "shake256"), ("web/src/lib/clientMcp.ts", "shake256")), oid="2.16.840.1.101.3.4.2.12"),
    CryptoAsset("hmac-sha256", "HMAC-SHA256", "mac", "HMAC", ("tag",),
                "The IKEv2 and WireGuard key schedule tests", "RFC 2104", (BROWSER,),
                (("web/src/lib/netkeys/ikev2.ts", "hmac"),), oid="1.2.840.113549.2.9"),
    CryptoAsset("blake2s-256", "BLAKE2s-256", "hash", None, ("digest",),
                "The WireGuard handshake test on the Validations page", "RFC 7693", (BROWSER,),
                (("web/src/lib/netkeys/wireguard.ts", "blake2s"),)),
    CryptoAsset("oprf-ristretto255", "OPRF(ristretto255, SHA-512)", "other", None, ("other",),
                "The private set intersection between two banks", "RFC 9497, mode 0x00", BOTH,
                (("web/src/lib/oprfPsi.ts", "ristretto255_oprf"), ("src/qsh/qsmpc/oprf.py", "ristretto255")),
                nist_quantum_level=0, classical_bits=128),
    CryptoAsset("ckks", "CKKS", "other", None, ("encrypt", "decrypt"),
                "Homomorphic evaluation of linear layers, and of the trained QCNN's readout as a linear form, on Microsoft SEAL (node-seal in the browser; TenSEAL, OpenFHE or HEonGPU in the full deployment)",
                "HomomorphicEncryption.org security standard (2018)", BOTH,
                (("web/src/lib/clientFHE.ts", "seal.SchemeType.ckks"), ("src/qsh/fhe/backends.py", "CKKS"), ("web/src/lib/clientQcnnCkks.ts", "evaluator.multiplyPlain"), ("src/qsh/qhe/qcnn_pauli.py", "enc.dot("),
                 ("web/src/lib/clientFedAgg.ts", "seal.SchemeType.ckks")),
                parameter_set="browser: N = 8192, coefficient moduli {60, 40, 40, 60} bits, scale 2^40; the medical page's shared-key aggregation {60, 40, 40} bits, NVIDIA FLARE's", classical_bits=128),
    CryptoAsset("ckks-threshold", "Threshold CKKS", "other", None, ("encrypt", "decrypt"),
                "The medical page's encrypted-aggregation comparison: four hospitals' collective key, their FedAvg updates summed under it, and each sum decrypted once by all four noise-flooded shares, on Lattigo v6.2.0 compiled to WebAssembly",
                "Mouchet, Troncoso-Pastoriza, Bossuat and Hubaux, PETS 2021 (multiparty RLWE); Cheon, Kim, Kim and Song, ASIACRYPT 2017", (BROWSER,),
                (("wasm/qsh-fedthresh/agg/agg.go", "ckks.NewParametersFromLiteral"), ("wasm/qsh-fedthresh/agg/agg.go", "multiparty.NewKeySwitchProtocol")),
                parameter_set="N = 8192, Q 2 x 60 bits; each share flooded with noise uniform on [-(2^f - 1), 2^f - 1] for the analytic worst-case bound 2^22.26: scale 2^80 with f = 51 (16 bits a share), or scale 2^100 with f = 75 (40 bits)", classical_bits=128),
    CryptoAsset("bgv-threshold", "Threshold BGV", "other", None, ("encrypt", "decrypt"),
                "The same comparison's exact row: the masks' own fixed-point integers encrypted under the four hospitals' collective key and each sum decrypted once by their shares, on Lattigo v6.2.0",
                "Brakerski, Gentry and Vaikuntanathan, ITCS 2012; Mouchet et al., PETS 2021", (BROWSER,),
                (("wasm/qsh-fedthresh/agg/agg.go", "bgv.NewParametersFromLiteral"),),
                parameter_set="N = 8192, Q 60+60+60+20 bits (inside the HE standard's post-quantum 202 at N = 8192), a 33-bit plaintext modulus, each share flooded uniformly with f = 163 (128 bits a share)", classical_bits=128),
    CryptoAsset("bgv-threshold-committee", "Threshold BGV, 2 of 3", "other", None, ("encrypt", "decrypt"),
                "The DeFi page's committee: three members' collective key, each holding a Shamir share of its secret, the borrower's balances encrypted under it, and each masked margin re-encrypted to the circuit's party by two members' flooded public-key-switch shares, once per ciphertext, on Lattigo v6.2.0",
                "Brakerski, Gentry and Vaikuntanathan, ITCS 2012; Mouchet et al., PETS 2021 (public-key switch); Mouchet, Bertrand and Hubaux, ePrint 2022/780 (t-out-of-N)", (BROWSER,),
                (("wasm/qsh-fedthresh/agg/committee.go", "multiparty.NewPublicKeySwitchProtocol"), ("wasm/qsh-fedthresh/agg/committee.go", "multiparty.NewThresholdizer")),
                parameter_set="N = 16384, Q 60+60+60+60+20 bits (inside the HE standard's post-quantum 411 at N = 16384), a 50-bit plaintext modulus, 2 of 3 members, each share flooded uniformly with f = 189 (128 bits a share)", classical_bits=128),
    CryptoAsset("wrk17b", "WRK17b authenticated garbling", "other", None, ("other",),
                "The argmax, the CNN's ReLUs, the PSI among banks and the sign of the lending committee's margin modulo t, on Polytune. Its oblivious transfers rest on a discrete-log assumption", "Wang, Ranellucci and Katz, CCS 2017", (BROWSER,),
                (("web/src/lib/clientMPC.ts", "WRK17b"), ("wasm/qsh-mpc/Cargo.toml", "polytune")), nist_quantum_level=0),
    CryptoAsset("shamir", "Shamir secret sharing", "other", None, ("other",),
                "The non-linear layers in the full deployment, on MPyC", "Shamir, CACM 1979", (FULL,),
                (("src/qsh/benchmark/classical_mpc_fhe.py", "mpyc-shamir"),)),
    CryptoAsset("csprng", "Platform CSPRNG", "drbg", None, ("generate",),
                "Masks, shuffles and nonces: Web Crypto getRandomValues in the browser", "W3C Web Cryptography API", (BROWSER,),
                (("web/src/lib/csprng.ts", "getRandomValues"),)),
)

#: Names a crypto library exports that are not algorithms. The test treats any other import from these libraries as a primitive that needs an entry.
TS_UTILITY_IMPORTS = frozenset({"bytesToHex", "concatBytes", "equalBytes", "hexToBytes", "utf8ToBytes", "randomBytes", "numberToBytesBE", "bytesToNumberBE"})

#: Each algorithm imported from a crypto library, by the name it is imported under, and the entry that accounts for it.
TS_IMPORTS = {
    "ml_kem768": "ml-kem-768", "ml_kem1024": "ml-kem-1024", "ml_dsa65": "ml-dsa-65", "ml_dsa87": "ml-dsa-87", "x25519": "x25519", "p256": "ecdh-p256",
    "hkdf": "hkdf-sha256", "hmac": "hmac-sha256", "sha256": "sha-256", "sha384": "sha-384", "sha512": "sha-512", "shake256": "shake256",
    "blake2s": "blake2s-256", "chacha20poly1305": "chacha20-poly1305", "ristretto255": "oprf-ristretto255", "ristretto255_hasher": "oprf-ristretto255",
    "ristretto255_oprf": "oprf-ristretto255",
}
PY_IMPORTS = {"X25519PrivateKey": "x25519", "X25519PublicKey": "x25519", "AESGCM": "aes-256-gcm", "HKDF": "hkdf-sha256"}
PY_HASHLIB = {"sha256": "sha-256", "sha512": "sha-512", "shake_256": "shake256"}
