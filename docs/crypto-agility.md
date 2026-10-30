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
survives a future PQC migration - or the withdrawal of a broken primitive - as a
*configuration change*, not a rewrite. That is the modular-replaceability property. It is
**not** cipher-suite negotiation.

### What "selectable" means here (and what it does not)

| | QSH |
|---|---|
| Who chooses the provider | **The operator, via configuration / `GET /api/crypto-options`, pinned into the run config before the run starts** |
| In-band cipher-suite negotiation between parties | **None anywhere in the codebase** - so there is no negotiated-downgrade surface |
| Silent fallback to a weaker provider | **None.** A missing/unavailable provider fails the run loudly (repository *no-easy-fallback* policy) |
| `qkd: pqc-only` | A **deployment profile, not a fallback.** Both `bb84-decoy` and `pqc-only` leave the PQC layer (hybrid ML-KEM + X25519, ML-DSA) unchanged, so the choice governs whether simulated QKD material is mixed in, which only the full deployment does - it does not weaken the post-quantum guarantee, and a peer cannot induce it |

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
(MIT, a normal PyPI dependency - nothing in this repository is vendored) is live; `mp-spdz` / `hpmpc-gpu` are heavyweight production references.

## Config sub-strategies (not registry providers, but runtime-selectable + security-relevant)
Three knobs in `config/defaults.py` are also runtime-selectable and shift the observed security, so they belong
to the crypto-agility story even though they are not `CryptoRegistry` families. `qkd.protocol` and
`qrng.post_processing` are edited in the **Parameters panel**; `crypto.hybrid_combiner_mode` is selected in the
**Agentic-AI, MCP panel** (it governs that panel's PQC+QKD channel-key derivation). Each visibly changes the run:

| Knob | Options | Effect (demo-observable) |
|------|---------|--------|
| `qkd.protocol` | `bb84` (two-intensity decoy-state), `e91` (entanglement / CHSH) | picks the simulated QKD model the run uses - selecting `e91` moves the surfaced QBER / secure-key-bits and adds a CHSH `S`-parameter metric (independent of OptiQKD, which then tunes whichever is chosen) |
| `qrng.post_processing` | `von_neumann`, `toeplitz` (Leftover-Hash-Lemma, epsilon-parameterised), `seedless` (two-source) | the entropy extractor + min-entropy floor of the simulated QRNG, which can seed BB84/E91 in Python when `qrng.enabled` is set (off by default; in the browser its bits feed only the health metrics). The LHL `epsilon` knob applies only to `toeplitz`/`seedless`, so it is greyed out on the `von_neumann` default |
| `crypto.hybrid_combiner_mode` | `hkdf` (default, dual-PRF KDF), `its_toeplitz` (Toeplitz+QLHL) | the PQC+QKD key combiner. Both keep the session key if either input holds. `its_toeplitz` is the Toeplitz+QLHL extractor of arXiv:2603.26907 (which explicitly does *not* claim to beat XOR), seeded here with SHAKE-256 of its own inputs. The paper's leakage-resilience property needs an independent seed, so as implemented the combiner is computational, not information-theoretic, and the demo's QKD input to it is a fixed constant. The MCP envelope reports the combiner used + the derived-key fingerprint |

Each provider is a factory; `register(family, name, factory, default=...)`, `get(family, name)`,
`options(family)`, `default(family)`. Adding a new algorithm (e.g. SLH-DSA, FN-DSA/Falcon,
a new FHE backend) is **one `register(...)` call** - no stage code changes.

## Runtime selection (observable)
- **API**: `GET /api/crypto-options` returns `{families, defaults}`.
- **SPA**: the E2E Orchestration page (and every per-use-case page) renders a selector per family
  (KEM / QKD / FHE (linear) / MPC (non-linear) / Signature); the choice is sent in the run config and
  threaded to the stages via `ctx.config`. In the **public demo** the FHE/MPC *backend* selectors are
  replaced by a "real, in-browser" badge - the demo runs genuine client-side **node-seal CKKS** +
  **Polytune garbled-circuit 2PC** rather than the server-side TenSEAL/MPyC backends, so a backend
  choice would be inert; KEM/QKD/Signature stay selectable (they run client-side via `@noble`). The
  **Parameters panel** additionally exposes `qkd.protocol` (BB84 <-> E91), the QRNG extractor + bits + epsilon, the
  OptiQKD budget, the QCNN qubit count, and the per-use-case data knobs; the **Agentic-AI, MCP panel**
  exposes the `hybrid_combiner_mode` combiner (HKDF <-> Toeplitz+LHL). Every one of these genuinely changes the
  demo run's metrics/log/result client-side (zero `/api`), so crypto-agility is *observable*, not asserted.
- **Stages consume the choice**:
  - `PQCHandshakeStage` -> `kem` (hybrid ML-KEM + X25519 session).
  - `QKDKeyAgreementStage` -> `qkd` (BB84 with QBER, or `pqc-only` to drop QKD).
  - `PrivateInferenceStage` -> `fhe` for the linear layers (TenSEAL CPU <-> OpenFHE CPU; both
    commercial-OK) and records the `mpc` provider for the non-linear layers - the full
    FHE-linear + MPC-non-linear hybrid is exercised by the benchmark
    (`qsh.mpc.secret_inference` + `qsh.benchmark.classical_mpc_fhe`), so the chosen MPC backend
    is named in the signed run summary.
  - `SignatureStage` -> `sig` (ML-DSA signs + verifies the run summary -> tamper-evident,
    quantum-safe audit record; emits `signature_verified`).

## Hybrid + defence-in-depth
- **KEM is always hybrid** (`ML-KEM` + `X25519`, combined with HKDF-SHA256 over both shared
  secrets under a domain-separated transcript, not XOR): secure if *either* primitive holds - the
  HNDL hedge. The construction is `src/qsh/pqc/hybrid_kem.py` and
  `web/src/lib/hybridCombiner.ts`, which `npm run check:combiner` holds byte-identical;
  described there rather than restated here, so this file cannot drift from it again. The PQC +
  classical hybrid follows **BSI TR-02102-1 v2026-01 section 2.2 "Key Derivation and Hybridisation"**
  (p.29), which recommends *"a hybrid key agreement mechanism that combines a quantum-safe and a
  classical mechanism"* that *"should remain secure as long as one of the methods used is secure"*.
  Simulated QKD key material can additionally be mixed in (`PQC + QKD`) in the full deployment, and
  that is this project's own choice: section 2.2 does not mention QKD, and section 2.1 ("Use of
  Quantum-Safe Mechanisms") states QKD *"is currently not ready for use from a security point of
  view"* and that BSI does not recommend QKD protocols. **NSA publishes no PQC+QKD hybrid
  recommendation either**, and its published position is against QKD for National Security Systems.
  QKD is never used standalone.
- **Transport** (`pqc/transport.py`) wraps the agreed key in AES-256-GCM; the **MCP agent
  bridge** (`pqc/mcp_bridge.py`) adds ML-DSA message signatures.

## Agility you can measure: the posture facts move with the selection

Round 22 retired the 0-100 security score this section used to document. It was not a defensible
measure: of 17 fields in its config exactly one was traceable to a citation, its maxima summed to
exactly 100, and it awarded a perfect score to a run whose quantum channel produced ZERO secure key
bits. Five candidate published replacements were read in full and all five declined; see
`../references/security-scoring-fact-check.md`.

What replaced it is stronger for this purpose, because agility is now demonstrated against a
published standard rather than against a weighted sum this project invented. Each row is either
checkable against a specification or produced by the run, and says which.

**Swapping the KEM moves standard-checkable facts.** Measured in Chrome on the public demo,
2026-08-08, one run before and one after:

| Fact | ml-kem-768+x25519 | ml-kem-1024+x25519 | Checkable against |
|---|---|---|---|
| NIST security category | 3 | 5 | FIPS 203 |
| KEM ciphertext | 1088 bytes | 1568 bytes | FIPS 203 |

Both ciphertext sizes are the values FIPS 203 specifies, so a reader can verify the claim without
trusting this project's arithmetic. That is the difference from the retired score, whose "16 vs 20"
could only be checked against itself.

**Swapping the QKD profile** changes whether simulated QKD key material is mixed at all in the full
deployment (the browser mixes none either way): `pqc-only` reports 0 secure bits and the posture
table says so, where `bb84-decoy` reports the simulated channel's `secure_key_bits` and secret-key
rate.

**Driving the channel** changes the QBER row and its verdict. The verdict is taken from what the run
produced (`secure_key_bits > 0`), not from a threshold comparison, because the threshold cannot
answer it: this repository's zero-key point is well BELOW the textbook BB84 bound of 0.11, so a QBER
between the two passes the textbook test while the model returns nothing.

**The zero-key point is channel-dependent, and it is below 0.11.** Under the two-intensity decoy
bound and the finite-key block, the QBER at which `secure_key_bits` reaches zero depends on the
distance, the detector set (Rusca et al. 2018: efficiency 0.5, 1e-8 dark counts per gate, a 1 percent
error floor) and the intensities ($`\mu = 0.3416`$ / $`\nu = 0.1417`$), so no single threshold is quoted
here; on this detector set the asymptotic bound reaches zero at 301.8 km. Two quantities must not
share a name: `e_det` is the error on DETECTED SIGNAL photons and is what the bound takes as input,
while the QBER a panel renders adds the dark-count background back as `e0 * y0`, and
`e_det_for_bound` exists to keep them apart. In the browser, eavesdropper 0 gives QBER 0.01 and a
positive key, and eavesdropper 0.35 gives QBER about 0.1, which passes the textbook test, and 0
secure bits; the posture table reports the failure.

**The signature family** changes the recorded algorithm name and the signature size (ML-DSA-65 =
3309 bytes, FIPS 204), and the chosen level signs the run summary (`signature_verified`), so the
audit record names the exact algorithms used.

**What is reported but not scored**, so the list is not read as a ranking: the FHE backend selection
changes measured latency; the PQC+QKD combiner (`hkdf` vs `its_toeplitz`) changes the derived key and
its fingerprint. These are shown as measurements, and nothing sums them into a single number.

The same facts are produced client-side (`web/src/lib/clientCompute.ts`) so the public demo shows
them with no backend. `src/qsh/benchmark/security.py` still computes the old composite, and the full
deployment's `/api/benchmark` returns it in each row, but no page renders it; the benchmark panel
carries only a note on why it was retired.

NIST's own crypto-agility reference is CSWP 39upd1, *Considerations for Achieving Crypto Agility*
(final, 2026-06-29); this document keeps the EU definition above, which rules out in-band
negotiation in so many words.

## Reference: hybrid TLS
For protocol-level agility beyond this PoC, `oqs-provider` (OpenSSL 3 PQC provider, Apache-2.0)
enables hybrid TLS 1.3 (e.g. `X25519MLKEM768`) and composite PQC
certificates (IETF LAMPS), the production counterpart to this registry.

## Migration drill
1. New algorithm lands -> `register("kem", "ml-kem-1024+x25519",...)`.
2. Flip the default in `config.yaml` (`crypto.kem`) or pick it in the SPA.
3. Re-run - the signed run summary records exactly which algorithms were used.
