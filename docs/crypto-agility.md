# Crypto-Agility (the library strategy)

## Definition we follow

We use the definition from the EU NIS Cooperation Group,
[*A Coordinated Implementation Roadmap for the Transition to Post-Quantum Cryptography*](https://digital-strategy.ec.europa.eu/en/library/coordinated-implementation-roadmap-transition-post-quantum-cryptography)
(Part 1, Version 1.1, 11 June 2025; published via the EC digital-strategy library 23 June 2025), section 3.2:

> **cryptographic agility (or crypto-agility)**
> The design of cryptographic protocols and systems in a modular way that enables replacing
> the cryptographic components. This concept must not be confused with a requirement to
> negotiate the cipher suite during protocol execution; the latter often leads to downgrade
> attacks if not done very carefully.

**QSH implements the first half of that definition and deliberately does not implement the
second.** Algorithms are never hard-coded; they are resolved from a registry so the system
survives a future PQC migration — or the withdrawal of a broken primitive — as a
*configuration change*, not a rewrite. That is the modular-replaceability property. It is
**not** cipher-suite negotiation.

### What "selectable" means here (and what it does not)

| | QSH |
|---|---|
| Who chooses the provider | **The operator, via configuration / `GET /api/crypto-options`, pinned into the run config before the run starts** |
| In-band cipher-suite negotiation between parties | **None anywhere in the codebase** — so there is no negotiated-downgrade surface |
| Silent fallback to a weaker provider | **None.** A missing/unavailable provider fails the run loudly (repository *no-easy-fallback* policy) |
| `qkd: pqc-only` | A **deployment profile, not a fallback.** Both `bb84-decoy` and `pqc-only` leave the PQC layer (hybrid ML-KEM + X25519, ML-DSA) unchanged, so the choice governs whether QKD material is mixed in — it does not weaken the post-quantum guarantee, and a peer cannot induce it |

> **Terminology note.** `qkd_negotiation_ms` in the timing output is a label for the BB84
> *key-agreement* step. It is unrelated to cipher-suite negotiation.

The EU roadmap also **recommends beginning with hybrid classical-plus-post-quantum
constructions wherever feasible**, which is what the `kem` family does by default
(`ml-kem-768+x25519`). Alongside this we track CNSA 2.0 and FIPS 203/204/205.

## The registry
`src/qsh/orchestration/crypto_agility.py` defines `CryptoRegistry` with five families:

| Family | Providers (June 2026) | Default |
|--------|-----------------------|---------|
| `kem` | `ml-kem-768+x25519`, `ml-kem-1024+x25519` | ml-kem-768+x25519 |
| `qkd` | `bb84-decoy`, `pqc-only` | bb84-decoy |
| `fhe` | `tenseal-cpu`, `openfhe-cpu` | tenseal-cpu |
| `mpc` | `mpyc-shamir`, `mp-spdz`, `hpmpc-gpu` | mpyc-shamir |
| `sig` | `ml-dsa-65`, `ml-dsa-87` | ml-dsa-65 |

`fhe` and `mpc` are the two halves of the **hybrid private-inference** path: linear layers run under
FHE/CKKS (`fhe`), non-linear layers (ReLU/argmax) under secret-sharing MPC (`mpc`). `mpyc-shamir`
(MIT, vendored) is live; `mp-spdz` / `hpmpc-gpu` are heavyweight production references.

## Config sub-strategies (not registry providers, but runtime-selectable + security-relevant)
Three knobs in `config/defaults.py` are also runtime-selectable and shift the observed security, so they belong
to the crypto-agility story even though they are not `CryptoRegistry` families. `qkd.protocol` and
`qrng.post_processing` are edited in the **Parameters panel**; `crypto.hybrid_combiner_mode` is selected in the
**Agentic-AI · MCP panel** (it governs that panel's PQC+QKD channel-key derivation). Each visibly changes the run:

| Knob | Options | Effect (demo-observable) |
|------|---------|--------|
| `qkd.protocol` | `bb84` (two-intensity decoy-state) · `e91` (entanglement / CHSH) | picks the physical QKD model that actually runs — selecting `e91` moves the surfaced QBER / secure-key-bits and adds a CHSH `S`-parameter metric (independent of OptiQKD, which then tunes whichever is chosen) |
| `qrng.post_processing` | `von_neumann` · `toeplitz` (Leftover-Hash-Lemma, ε-parameterised) · `seedless` (two-source) | the entropy extractor + min-entropy floor that seeds BB84/E91 (the LHL `ε` knob applies only to `toeplitz`/`seedless`, so it is greyed out on the `von_neumann` default) |
| `crypto.hybrid_combiner_mode` | `hkdf` (default, dual-PRF KDF) · `its_toeplitz` (Toeplitz+QLHL) | the PQC+QKD key combiner. Both keep the session key if either input holds; `its_toeplitz` adds information-theoretic leakage resilience (output + one key compromised leaves the other with quantifiable min-entropy) at the cost of longer keys — arXiv:2603.26907, which explicitly does *not* claim to beat XOR. The MCP envelope reports the combiner used + the derived-key fingerprint |

Each provider is a factory; `register(family, name, factory, default=…)`, `get(family, name)`,
`options(family)`, `default(family)`. Adding a new algorithm (e.g. SLH-DSA, FN-DSA/Falcon,
a new FHE backend) is **one `register(...)` call** — no stage code changes.

## Runtime selection (observable)
- **API**: `GET /api/crypto-options` returns `{families, defaults}`.
- **SPA**: the E2E Orchestration page (and every per-use-case page) renders a selector per family
  (KEM / QKD / FHE (linear) / MPC (non-linear) / Signature); the choice is sent in the run config and
  threaded to the stages via `ctx.config`. In the **public demo** the FHE/MPC *backend* selectors are
  replaced by a "real, in-browser" badge — the demo runs genuine client-side **node-seal CKKS** +
  **emp-wasm garbled-circuit 2PC** rather than the server-side TenSEAL/MPyC backends, so a backend
  choice would be inert; KEM/QKD/Signature stay selectable (they run client-side via `@noble`). The
  **Parameters panel** additionally exposes `qkd.protocol` (BB84 ↔ E91), the QRNG extractor + bits + ε, the
  OptiQKD budget, the QCNN qubit count, and the per-use-case data knobs; the **Agentic-AI · MCP panel**
  exposes the `hybrid_combiner_mode` combiner (HKDF ↔ Toeplitz+LHL). Every one of these genuinely changes the
  demo run's metrics/log/result client-side (zero `/api`), so crypto-agility is *observable*, not asserted.
- **Stages consume the choice**:
  - `PQCHandshakeStage` → `kem` (hybrid ML-KEM + X25519 session).
  - `QKDKeyAgreementStage` → `qkd` (BB84 with QBER, or `pqc-only` to drop QKD).
  - `PrivateInferenceStage` → `fhe` for the linear layers (TenSEAL CPU ↔ OpenFHE CPU; both
    commercial-OK) and records the `mpc` provider for the non-linear layers — the full
    FHE-linear + MPC-non-linear hybrid is exercised by the benchmark
    (`qsh.mpc.secret_inference` + `qsh.benchmark.classical_mpc_fhe`), so the chosen MPC backend
    is named in the signed run summary.
  - `SignatureStage` → `sig` (ML-DSA signs + verifies the run summary → tamper-evident,
    quantum-safe audit record; emits `signature_verified`).

## Hybrid + defence-in-depth
- **KEM is always hybrid** (`ML-KEM` ⊕ `X25519`): secure if *either* primitive holds — the
  HNDL hedge. QKD key material can be mixed in (`PQC + QKD`) per **BSI TR-02102-1 v2026-01 §2.2 "Key Derivation and
Hybridisation"** (p.29; §2.1 is "Use of Quantum-Safe Mechanisms"),
  which recommends *"a hybrid key agreement mechanism that combines a quantum-safe and a classical
  mechanism"* that *"should remain secure as long as one of the methods used is secure"*. This was
  previously attributed to an "NSA hybrid recommendation" — **NSA publishes no such PQC+QKD hybrid
  recommendation**; its published position is against QKD for National Security Systems, and the same
  BSI guideline states QKD *"is currently not ready for use from a security point of view"*;
  QKD is never used standalone.
- **Transport** (`pqc/transport.py`) wraps the agreed key in AES-256-GCM; the **MCP agent
  bridge** (`pqc/mcp_bridge.py`) adds ML-DSA message signatures.

## Security score reflects the chosen primitives
The benchmark's numeric **security score** (`src/qsh/benchmark/security.py`,
`compute_security_score(...) → 0–100`) is *derived from the crypto-agility selection*, so
swapping a primitive visibly moves the score — agility you can measure, not just assert:

- **KEM category** → key-strength component: `ml-kem-768+x25519` (NIST cat 3) scores 16,
  `ml-kem-1024+x25519` (cat 5) scores 20. Picking the cat-5 KEM in the SPA raises the score.
- **QKD on/off** → HNDL component: a ≥256-bit BB84 key mixed in (`bb84-decoy`) adds a
  defence-in-depth bonus; `pqc-only` drops it.
- **QBER** → eavesdropper-detection component: below the ~11% BB84 threshold scores full marks;
  an active eavesdropper pushes QBER up and the score down (and, past threshold, the key is
  rejected entirely — see the satellite QSDC use case).
- **Security model** → information-theoretic (QOTP perfect hiding, the QSMPC-FHE path) beats
  computational (RLWE, the classical MPC-FHE path): 96 vs 68 in the default benchmark.

The same formula is mirrored client-side (`web/src/lib/clientCompute.ts`) so the public demo
shows identical scores with no backend. The `sig` family is likewise runtime-selected and the
chosen ML-DSA level signs the run summary (`signature_verified`), so the audit record names the
exact algorithms used.

## Reference: hybrid TLS
For protocol-level agility beyond this PoC, the `oqs-provider` submodule (OpenSSL 3 PQC
provider, Apache-2.0) enables hybrid TLS 1.3 (e.g. `X25519MLKEM768`) and composite PQC
certificates (IETF LAMPS), the production counterpart to this registry.

## Migration drill
1. New algorithm lands → `register("kem", "ml-kem-1024+x25519", …)`.
2. Flip the default in `config.yaml` (`crypto.kem`) or pick it in the SPA.
3. Re-run — the signed run summary records exactly which algorithms were used.
