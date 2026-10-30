# Threat model

## Adversaries
- **Harvest-Now-Decrypt-Later (HNDL)**: records ciphertext today to decrypt once a CRQC
  exists. Mitigation: PQC-first hybrid (ML-KEM+X25519) on every channel - confidential
  as long as *either* primitive holds; the X25519 half covers today, ML-KEM the quantum future.
- **Quantum adversary at Q-Day**: breaks RSA/ECC. Mitigation: ML-KEM/ML-DSA (FIPS 203/204);
  QKD is an optional *add-on* to the PQC hybrid, never standalone - this project's choice, which
  neither BSI nor NSA recommends. Real QKD would add key material whose secrecy rests on physics
  rather than on a hardness assumption; here the quantum channel is simulated, so the add-on
  contributes none of that security. **BSI TR-02102-1 v2026-01** section 2.1 says QKD *"is currently
  not ready for use from a security point of view"* and does not recommend it; the hybrid BSI does
  recommend (section 2.2) is quantum-safe plus classical, the ML-KEM + X25519 layer. NSA's CNSA 2.0
  and Quantum Computing FAQ (Ver. 2.2, October 2026) answers "No" to using QKD to protect a national
  security system.
- **Honest-but-curious compute server** (delegated AI): would read inputs/model. Mitigation:
  QHE (QOTP, perfectly hiding as a scheme; the demo's pad is seeded) for the quantum path; CKKS for
  the classical path, whose masked HE-to-MPC conversion protects slot 0 only (see Trust assumptions);
  secure aggregation is meant to hide per-client updates in federated learning -- **but it hides
  nothing here, so this mitigation is not in force:** in the browser every pairwise mask is an HKDF
  output of the one PQC session key, which the aggregator also holds, and in Python the default pair
  keys are a public constant (residual risk below).
- **Malicious MPC participant**: injects bad inputs / aborts. Mitigation: identifiable-abort
  pattern (scaffold) and QOT test rounds. The keyed-hash PSI does **not** reveal only the
  intersection: both banks hold the HMAC key, so over a small ID space either one can hash every
  candidate ID and recover the other's list (residual risk below).
- **QKD link attacker**: intercept-resend, PNS, Trojan-horse, detector-blinding, RNG
  manipulation. Mitigation: QBER monitoring + two-intensity decoy-state single-photon bounds; QRNG
  health + **SP 800-90B min-entropy** and section 4.4 continuous health checks guard the entropy source; the
  config-gated **Toeplitz+QLHL PQC+QKD combiner** (`its_toeplitz`). Note the property carefully: `src/qsh/pqc/transport.py`
  explicitly declines the "secure-if-either" framing, because plain XOR already gives
  that and the cited paper never claims superiority over XOR. What the paper claims is narrower -- if the
  combined output AND one input key are compromised, the remaining key material retains quantifiable
  min-entropy -- and that claim needs a seed independent of the input. Here the extractor's seed is
  `SHAKE-256` over the same two keys it extracts from (`transport.py:64`), so as implemented it is a
  **computationally-seeded** extractor, not an information-theoretic or leakage-resilient one, and its
  QKD input is a fixed constant on the MCP channel. The docstring says so; the label "ITS" does not; + QML-IDS (WS2) classifying 7 attack types against normal traffic (8 classes).
- **Classical control-plane adversary against the QKD stack** - recorded 2026-08-12 as a **GAP**,
  because until then this document had no such adversary at all. Every QKD adversary above attacks the
  quantum channel or the network crypto; none attacks the **classical procedure text** that drives the
  quantum layer. Malik, Anwar and Raza, "Beyond the Quantum Promise: A Security Analysis of Classical
  Control in Quantum Key Distribution" (arXiv:2608.07626, 2026-08-07), build the first hybrid Tamarin
  model of the ETSI and ITU-T QKD specifications and find **three specification-level
  vulnerabilities** - subverted entanglement injection, basis-deferred measurement, and message
  reflection - each arising from *"a classical control-plane omission in the procedure text"*, with
  verified countermeasures (**measurement commitment**, **identity-bound MACs**). This lands on
  `src/qsh/qkd/sdn_controller.py`, which models ETSI GS QKD 015 `PHYS`/`VIRT` links, and on
  `qkd/kme.py` (GS QKD 014). **Mitigation: none implemented**, and the uncomfortable part is why -
  the findings are in the specifications this project implements faithfully, so implementing them
  faithfully reproduces them. **Resolving condition**, so this is falsifiable rather than a standing
  worry: it closes when the controller and KME either (a) bind SAE identity into the authentication of
  every control-plane message instead of accepting it as a parameter - the same missing identity
  binding the GS QKD 014 gap below already records, so one fix serves both - or (b) declare in-band
  which of the three findings a given payload is exposed to, the way `trust_note()` already declares
  trusted-relay exposure. Until one of those exists, the control plane here is **modelled, not
  defended**.
- **Network MITM / tampering**: Mitigation: AEAD (AES-256-GCM) record layer + ML-DSA
  message authentication over the hybrid session.

## Trust assumptions
- Endpoints (clients) are honest and hold their own secret keys.
- The QKD hardware is modelled by simulation; real deployments must address device
  imperfections (the WS2 IDS targets exactly these).
- The crypto/compute libraries are trusted as implemented (pin versions + review): server-side
  TenSEAL / OpenFHE / liboqs / MPyC (and HEonGPU, if its build-gated adapter is ever wired); client-side **node-seal** (CKKS) + **Polytune /
  Garble** (garbled-circuit 2PC, Rust compiled to wasm from `wasm/qsh-mpc`, Round 65) for the in-browser
  demo; the **MCP Python SDK** (`MCPServer`, named `FastMCP` before v2.0.0) for
  the full-deployment MCP server.
- **Server-tier MPC** (full deployment, `src/qsh/mpc/parties.py`): three MPyC parties, the client
  who holds the CKKS key, the server who holds the model, and a helper; Shamir with t = 1, so
  **semi-honest with an honest majority**. The conversions are masked (`src/qsh/mpc/secret_inference.py`),
  but the mask is a scalar added to every CKKS slot, so it protects slot 0 only: a key holder who
  decodes every slot of one masked dot product recovers the unmasked output and the layer's weights
  (confirmed with the TenSEAL backend). Until that is fixed the conversion is **not model-private**;
  slot 0 alone is hidden statistically rather than perfectly, with the bound reported per run. A
  client and helper who pool their shares recover the server's masks and with them every
  intermediate. Here the three run on one host over loopback; a deployment that relies on the honest
  majority needs them on three hosts under three operators.
- **MCP tool dispatch** (full deployment): the quantum-safe channel authenticates + seals the
  `tools/call` and its result (ML-DSA + AES-GCM), but the tool *logic* runs server-side and is trusted;
  the official filesystem MCP server is sandboxed to a use-case data dir. The public demo runs no
  server-side MCP (client mirror only), so it adds no server trust surface.

## Key management (what each key is, and what it is not)

Enumerated because "key management is out of scope" reads as *not yet built*, when in fact several
keys exist and some of them are recomputable by a party they were meant to hide from.

| Key | Lifetime | Where it lives | Caveat |
|---|---|---|---|
| Hybrid KEM (ML-KEM-768 + X25519) | Per session, ephemeral | Process / tab memory | Both halves genuinely computed on both sides. Never persisted, never escrowed. |
| QKD-derived key material | Per run, full deployment only | Memory | Simulated channel - no QKD hardware is involved anywhere in this project. The full deployment folds 256 or more simulated sifted bits into the session key by HKDF-SHA256; the browser derives no key from QKD, and the MCP "QKD-domain" input is a fixed 32-byte constant. The Toeplitz seed is SHAKE-256 of the inputs, so no key here is information-theoretically secure. |
| ML-DSA signing keypair | Per run, discarded | Generated inside the signature stage | The demo now records the public half and the exact signed payload; the **backend** path still discards the public key and keeps only a 12-byte signature prefix, so a full-mode run's signature is not third-party verifiable. |
| Secure-aggregation pairwise masks (browser, `clientQFL.ts`) | Per round | HKDF-SHA256 of the run's one PQC session key, with the pair and round in the info string | **The aggregator holds that key**, so it can derive every mask and unmask each update. The masks cancel exactly and hide nothing from a session-key holder. Bonawitz et al. derive each pair's seed by a pairwise key agreement, which is what would make this hiding. |
| Secure-aggregation pairwise masks (Python, `federated.py`) | Per round | Derived from `sha256(f"pair-{i}-{j}")` by default | **A public constant.** Anyone can recompute every mask. Nothing passes other keys to `SecureAggregator`. |
| Browser secure-argmax share masks (`clientMPC.ts`) | Per run | `crypto.getRandomValues`, 32 bits per logit; shares modulo $`2^{32}`$ (since Round 66) | Each share alone is uniform whatever the logit. Until Round 66 the mask came from the run's seeded generator (mulberry32 on the seed the page shows) and was bounded to $`2^{16}`$, so it was recomputable and only statistically hiding (distance up to 1/64). Both parties are in one tab, so this is a property of the sharing, not of two hosts. |
| Browser secure-sum masks (`clientCompute.ts`) | Per run | `crypto.getRandomValues`, uniform over $`\mathbb{Z}_p`$ by rejection (since Round 66) | Until Round 66 `floor(u() * p)` from the same seeded generator: recomputable, and biased because no odd prime divides $`2^{32}`$. |
| QOTP demo key (`quantumSim.ts`) | Per run | The run's seeded generator | **Disclosed in place since Round 40, not changed.** The pad is perfectly hiding as a scheme, but this key decides the circuit's output, so it stays reproducible and anyone with the seed can recompute it. |
| ETSI KME key IDs (`qkd/kme.py`) | Per issued key, in memory | The KME's own store | Bound to one `(master_SAE_ID, slave_SAE_ID)` pair, so a `key_ID` observed in transit is not a bearer token. That binding is the only thing protecting it, because **mutual TLS is absent** - see below. |

**The GS QKD 014 authentication gap, stated rather than implied.** The spec mandates mutual TLS with
the SAE identity taken from the client certificate. This project's KME runs in-process and has no
transport to attach a certificate to, so **SAE identity here is a parameter, not an authenticated
fact**: any caller can name itself `sae-master`. The key-ID binding above is therefore meaningful
only against a caller who does not lie about its own identity, which is not a threat model - it is a
convention. A deployment must add mTLS before that binding means anything, and
`KeyManagementEntity.authentication_note()` says so at the call site so the gap cannot be read as an
oversight.

**Trusted relays are an assumption, not a mechanism.** When the configured distance exceeds a single
quantum span, `qkd/sdn_controller.py` creates a VIRT link through a relay node, and that relay holds
the key **in plaintext**. End-to-end secrecy then rests entirely on the relay's physical security,
which no cryptography in this repository provides. The controller is built to surface this rather
than smooth it over: `trust_note()` names every intermediate node, and the browser panel prints it.
ANSSI and BSI both press on exactly this point, and it is the honest limit of trusted-node QKD.

## What CKKS does and does not promise here

The product's headline is privacy-preserving inference under CKKS, so the security notion CKKS
actually satisfies belongs in this document. It was absent until Round 23.

**CKKS is IND-CPA, not IND-CPA-D.** The distinction is not academic for a system that returns
decrypted results:

- Li and Micciancio (EUROCRYPT 2021, ePrint 2020/1533) showed approximate FHE fails IND-CPA-D -
  security when the adversary sees DECRYPTION RESULTS. It is a **key-recovery** attack, demonstrated
  against the CKKS implementations in HEAAN, SEAL, HElib and PALISADE, with **"very modest running
  times"**.
  Two claims this document used to make were checked at primary source on 2026-08-12 and did not
  survive, so they are corrected rather than left: the abstract says "very modest running times" and
  **not "within several seconds"**, which was quoted here as if verbatim; and the key-recovery
  expression is `[UNVERIFIED]` - this document rendered it $`s = a^{-1}(m'' - b)`$ while secondary
  sources render it $`s = a^{-1} b'`$, and the PDF is unreachable (eprint returns HTTP 403, the UCSD
  mirror 404), so **neither form is asserted here**. Nothing below depends on which it is: the
  conclusion that decryption results leak the key is the abstract's own.
- **The "needs one decryption query" figure is verifiable, but from a different source.** It is
  attributed here to IBM **HElib's `CKKS-security.md`**, which states the attack can succeed with
  **"sometimes as few as a single decryption"**. Li and Micciancio is the result; HElib is the
  implementation that writes the query count down in its own security documentation.
- Li, Micciancio, Schultz and Sorrell (CRYPTO 2022, ePrint 2022/816) give the noise-flooding
  countermeasure and its precision cost.
- Cheon, Choe, Passelegue, Stehle and Suvanto (ACM CCS 2024, ePrint 2024/127) extended the attack
  class to EXACT FHE schemes.
- Checri, Clet, Renard and Sirdey (ePrint 2026/203) prove an **impossibility**: no "HELLHO" scheme -
  one whose homomorphic addition is linear over the ciphertext domain, which covers the basic
  variants of BFV, BGV, **CKKS** and TFHE - can achieve CPAD security at all. They also demonstrate
  a practical key-recovery attack on DE-CKKS, a scheme OpenFHE supports, achieving full key recovery
  "in a matter of hours on an average laptop PC".
- Ogilvie (ASIACRYPT 2025, ePrint 2025/1618) is the constructive counterpoint: in regimes where
  rescaling noise dominates, "an additional precision loss of as little as two bits is sufficient to
  restore a high level of security against passive key-recovery attacks for standard parameters".

**The implementations say this in their own security documentation, and node-seal IS SEAL.** The demo
runs Microsoft SEAL compiled to WASM, so SEAL's `SECURITY.md` is not background reading, it is the
upstream security statement for the cipher on the page. Verbatim: *"The CKKS scheme requires
particular care because it is approximate: decryption returns the message plus a secret-key-dependent
error term rather than the exact message"*; *"Releasing such decryptions to any party other than the
secret key owner can therefore leak the secret key, even when only ordinary computation results are
shared"*; and *"Formally, the CKKS implementation in Microsoft SEAL does not provide IND-CPA-D
security ...; it targets only IND-CPA security"*. IBM HElib's `CKKS-security.md` states the mechanism
in one sentence: *"since the scheme itself adds some error, the attacker could learn something from
the decryption result even if it knows what the decrypted result was supposed to be. Specifically, it
learns the error, which may leak information about the secret key."*

**How much of this reaches this project.** The public demo is the benign case and should be stated
as such rather than claimed as a defence: the CKKS keys are generated in the browser tab, used in
that tab, and the decrypted class is shown to the same person who holds the secret key. There is no
decryption oracle exposed to anyone else, so an IND-CPA-D adversary would have to be the key holder
attacking themselves.

**The displayed approximation error is in scope of this section, and it is safe. Both halves of that
belong in writing.** The demo does not merely decrypt a result, it reports the per-sample CKKS
approximation error itself: `web/src/lib/clientFHE.ts` computes `maxAbsError` against a plaintext
reference and `clientCompute.ts` emits it as the run metrics `fhe_max_abs_error` and
`fhe_conv_max_abs_error` (about 1e-9), which the run panels render. By HElib's sentence above that
quantity - the error - IS
the leaking object, so a reader who knows this literature is right to stop on it, and a document that
listed only "decrypted results" would have skipped the sharpest case. It is nonetheless safe here, and
the reason is the predicate SEAL states: *"Any mitigation, such as adding noise to decrypted values
before they are shared (noise flooding), must be implemented by the application before a decrypted
CKKS value crosses a **trust boundary**."* The trigger is **crossing a trust boundary**, not being
displayed. A user reading their own approximation error, produced from a key generated in their own
tab, crosses nothing: the key holder, the decryptor and the viewer are one party. The moment any of
those three separate - a server-side key, a shared session, a value posted anywhere - the same number
becomes a key-recovery oracle and the mitigation below is owed. That is the falsifiable condition,
and it is a property of the architecture rather than of the cipher.

**The server-side deployment is the case that would need work.** If a model owner ran CKKS on a
server and returned decrypted results to a client, that is precisely the CPAD setting, and the
countermeasure - noise flooding, with its documented precision cost - would be required before any
production claim. This project does not use DE-CKKS and does not perform CKKS-to-FHEW scheme
switching (ePrint 2026/285 breaks PEGASUS-style switching in OpenFHE, so this is worth stating
explicitly rather than leaving to inference).

**"This project does not implement noise flooding" was an understatement, and it named the wrong
subject.** The three CKKS implementations are not in the same position, and this deployment runs the
one with nothing:

- **SEAL has no noise flooding at all.** Its `SECURITY.md` does not describe a parameter or a mode; it
  says the mitigation *"must be implemented by the application"*. So there is nothing to enable in
  node-seal, and the gap is upstream rather than a configuration this project declined to set.
- **HElib has it built into decryption**, with a settable accuracy target, per its `CKKS-security.md`.
  That document also gives the scaling law, which is the honest cost of the countermeasure: *"to
  withstand an attack that has access to the results from decryption of D ciphertexts, the bits
  parameter should be increased roughly by a factor of sqrt(D)"*.
- **OpenFHE has `NOISE_FLOODING_DECRYPT` plus `CKKS_M_FACTOR`**, and its security documentation states
  the tolerated query count: *"By default, OpenFHE chooses a configuration where a relatively large
  number of decryption queries of the same or related ciphertexts can be tolerated. The lower bound
  for the tolerated number of such decryption queries is N_d = 128 (but in practical computations, N_d
  is at least 10,000)."* It also scopes the attack the way this document should: it *"applies to the
  setting where decryption results need to be shared between multiple parties, e.g., in the threshold
  FHE setting"* - which is the server-side deployment, not the single-tab demo.

**Two things that stop noise flooding being read as a solved fix.** First, the countermeasure is itself
under attack: Guo, Nabokov, Suvanto and Johansson, *"Key Recovery Attacks on Approximate Homomorphic
Encryption with Non-Worst-Case Noise Flooding Countermeasures"*, USENIX Security 2024, targets exactly
the mitigation named above when the flooding is parameterised on average-case rather than worst-case
noise. Title, authors and venue are verified via the Lund University research portal; the **body could
not be loaded (HTTP 403), so no parameter, bound or number from it is quoted here** - it is recorded as
a named limit on the fix, not as a result. Second, the game is weaker than "malicious": Cheon, Choe,
Passelegue, Stehle and Suvanto (ePrint 2024/127, CCS 2024) give the adversary *"access to a decryption
oracle for ciphertexts for which it should know the underlying plaintexts. This includes ciphertexts
that it (honestly) encrypted and those obtained from the latter by evaluating circuits that it
chose."* **Honestly generated inputs suffice, and the attack is passive**, so "the parties here are
only semi-honest" is not a defence against it - which is worth stating because that sentence appears
elsewhere in this document as a real mitigation for other adversaries.

**The threshold setting OpenFHE points at now has a citation here.** This section had none, while the
quote above scopes the attack to exactly that case: Zhang, Zhou, Han, Liu and Huang, "Adaptively Secure
Threshold Decryption from LWE with Polynomial Modulus" (IACR ePrint 2026/1627, received 2026-08-06),
treats adaptive corruption and leakage from **decryption queries** using *"a refined polynomial noise
flooding technique based on a detailed min-entropy analysis"*. It matters here rather than in general
because `submodules/HEonGPU` ships N-out-of-N collective bootstrapping, so the threshold case is one
dependency away rather than hypothetical. Nothing here builds or calls it today: `HeongpuBackend` is a
build-gated reference adapter, and HEonGPU exposes no Python API.

**No standard requires testing for this.** The HomomorphicEncryption.org standard contains no
testing or conformance clause and states that CKKS "is not described in this standard"; ISO/IEC 28033 is not published, and the part that matters here is **Part 3** - mechanisms for arithmetic on approximate numbers, i.e. CKKS - which reached **FDIS stage 50.00 on 2026-06-11** (Part 1 general: DIS 40.60, 2025-12-11; Part 4: FDIS 50.00; Part 5: WD 20.60). Part 1 scopes correctness out. Naming "Parts 1-4 at DIS" was imprecise in both directions: Part 3 is further along than DIS and is the CKKS part, so this claim has a shelf life of months and carries a date for that reason. Checked 2026-08-10. So this section is a
statement of known limitations, not a conformance gap against any published requirement.

## Explicitly out of scope (PoC)
- Side-channel / fault attacks on the host.
- Formal verification of the protocols (referenced papers carry the proofs).
- Production key management / HSM integration (crypto-agility hooks are provided).

## Residual risk & honest limitations
- Quantum-channel results (QKD distance/QBER gains) and the QSMPC-FHE comparison row are
  **simulated**, not hardware-validated; photonic inference and qOLE are not implemented at all.
- The QHE homomorphic evaluation is exact for Clifford circuits with key tracking (the Python
  demo); non-Clifford (trainable) layers require magic-state/T-gate corrections in a real device
  (their cost is reported as a T-count, not executed fault-tolerantly here). The browser's
  comparison circuit applies its T gates to the padded state and never corrects or decrypts it, and
  the trained QCNN that predicts runs unmasked.
- PSI is a keyed hash over the quantum-safe channel: both banks HMAC their lists under one session
  key they both hold, so no third party sees the lists, but over a small ID space either bank can
  hash every candidate ID and recover the other's list (Pinkas, Schneider and Zohner, USENIX
  Security 2014, on hash-based PSI for low-entropy domains). It keeps neither list private from the
  other bank; a production PSI would use an OPRF-based protocol, and a malicious-secure OPRF PSI is
  future work.
- **Federated averaging is not a privacy guarantee -- and as shipped the secure aggregation hides
  nothing from the aggregator.** Pairwise-mask secure aggregation is intended to hide an *individual
  client's* update from the aggregator, and that is all it would claim. **Neither tier achieves even
  that.** In the browser (`web/src/lib/clientQFL.ts`) every pairwise mask is an HKDF output of the one
  PQC session key, which the aggregator also holds, so it can derive every mask. In Python
  (`qsh/ai/federated.py`) `SecureAggregator` is only ever constructed with no keys (`federated.py:219`,
  `tests/test_ai.py:26`), and the default pairwise keys are `sha256(f"pair-{i}-{j}")` -- derived from
  a public constant, so anyone can recompute every mask and unmask any individual update. The masks
  are real and cancel exactly; the secrecy is not. What would make the scheme hiding is a pairwise
  key agreement between each pair of clients, as Bonawitz et al. (CCS 2017) specify, with masks the
  aggregator cannot derive. It does not bound what the trained model itself leaks: gradient inversion against a shared
  model is an established attack. A formal claim would require a stated DP-SGD budget, which this
  path does not provide, so none is made.
- **Not everything the demo displays was computed.** **One** thing cannot be done in a browser at all
  and is labelled `SIMULATED` rather than quietly approximated: QKD has no browser API and cannot have
  one, so every QKD figure comes from an analytic channel model. *This bullet used to list three.* The
  other two were claims about specific libraries, and they no longer describe what ships: TFHE-rs' WASM
  build does not support FHE computations, but the demo does not use TFHE-rs - it runs **node-seal**
  (Microsoft SEAL compiled to WASM) for real CKKS (`web/src/lib/clientFHE.ts`); and JIFF is indeed
  unmaintained and the PSE MPC Framework sunset, but the demo runs **Polytune** authenticated garbling
  (`web/src/lib/clientMPC.ts`; it ran `mpc-framework` + `emp-wasm-engine` + `summon-ts` until Round 65).
  Both are real, measured, and
  described as such in `architecture.md` and `crypto-agility.md` - this bullet contradicted them.
  Caveat that *does* apply to the MPC path: both "parties" share one JS heap, so the party separation
  is notional (stated at `clientMPC.ts:13-15`), and it is a demonstration of the protocol, not two
  mutually distrusting hosts.
  **A SECOND CAVEAT, ADDED ROUND 55 AND RESOLVED IN ROUND 65, AND IT WAS A SUPPLY-CHAIN ONE.** The
  correction above replaced
  "JIFF is unmaintained and the PSE MPC Framework sunset" with the three packages that actually ship -
  and all three of those are now themselves archived upstream. Verified at the GitHub API on
  2026-08-22: `emp-wasm` (archived, last push **2025-06-13**), `mpc-framework` (archived,
  **2025-07-22**) and `summon` (archived, **2025-07-09**), each present under both `voltrevo/*` and the
  `privacy-ethereum/*` org it moved to. So the whole garbled-circuit tier - `emp-wasm-engine ^0.5.4`,
  `mpc-framework ^0.3.0`, `summon-ts ^0.6.2` - rested on code with no upstream maintainer and no
  security-patch path. That does not make the protocol wrong or the measurements false, and nothing
  here is being withdrawn; it means the tier carries a risk the CKKS tier does not, since `node-seal`
  tracks Microsoft SEAL. Note the shape: the previous bullet was corrected by naming the maintained
  replacements for two sunset dependencies, and the replacements have since been sunset in turn - a
  correction can go stale in the same way the claim it fixed did. The project's archived-upstream
  query covered submodules and not npm dependencies, which is why this went a year unnoticed. **Replacements were researched in Round 60, and nothing has been adopted.** The closest fit
  is `sine-fdn/polytune` with the Garble language (MIT, maintained, WRK17b multi-party authenticated
  garbling against malicious adversaries), at the cost of a Rust and wasm-pack build step and an alpha
  API; mpz would be a downgrade to semi-honest security. The register's Round 60 section carries the
  comparison and what remains unverified. **Round 63 prototyped it** (`prototypes/polytune-argmax/`, not
  part of the build): two Polytune parties complete the demo's argmax in one tab, correctly, about six
  times faster than emp-wasm at 2 classes, while sending about five times the bytes - the cost that
  would dominate on separate hosts, which was not timed. **Round 64 timed it**, between two tabs over
  netem-shaped real TCP: at 2 classes Polytune still finished first on all 18 links (1.09 to 3.33
  times faster), and at 10 classes emp-wasm did (1.47 and 2.75 times); the demo runs 2. **Round 65
  replaced the tier**: `clientMPC.ts` now runs Polytune, built by the web image's Rust stage from
  `wasm/qsh-mpc`, whose `cargo test` runs on every build, the public demo's included, and the three
  archived packages were removed. The residual risk moves with it: Polytune's latest release is a
  pre-release, so the crate is pinned to a commit, and a 10-class model under MPC would be slower than
  it was.
  **Round 68 reviewed the integration against that commit (47dc378).**
  - What held:
    - `mpc()`, Polytune's only public entry, always runs the preprocessing among the parties by OT
      extension. Its trusted-dealer mode is crate-private, so no dealer is involved.
    - Its randomness is `rand`'s thread-local CSPRNG, seeded from the OS, which on wasm32 means
      `crypto.getRandomValues`. Shared randomness comes from a commit-then-reveal coin toss. The fixed
      seeds in its source are in tests.
    - The statistical security parameter is 40.
  - What qualifies the claim: no security advisory is published for Polytune, but two issues are open.
    - #63: the OT extension uses KOS15's original consistency check, whose supporting lemma the
      SoftSpokenOT paper proved false. The attack it gives is impractical. Its practical attack needs
      a case the maintainers believe Polytune does not have. The corrected check exists in libOTe and
      CryProt, but not yet in Polytune.
    - #100: AES in counter mode as a PRG never repeats a block, which a birthday-bound test could
      notice. That is negligible at the output lengths here.
  - What it means here: neither issue changes the demo, whose two parties share one tab. Both qualify
    "malicious security" for a two-host deployment, as do the pre-release status and the absence of
    any audit in the repository.
  **Round 72 read further, at the pinned commit and in Polytune's history.**
  - No independent audit is published. Two pull requests (#34, merged 2025-01-07, and #36) apply
    feedback "from our security audit", but who audited, what was in scope and what was found are not
    disclosed. There is no SECURITY.md and no published advisory, and private vulnerability reporting
    is off.
  - #340 (2025-12-22) removed a fixed seed from the OT-extension sender's generator, a seed that a
    since-yanked pre-release had shipped. The pinned commit postdates it: its only fixed seeds are in
    test code.
  - #63, measured against the literature:
    - It is a gap in the proof at Polytune's parameter of 128, not a practical break. The one
      better-than-negligible attack (SoftSpokenOT, Appendix D) needs a field size divisible by 20.
    - On two hosts either party can play the attacking receiver, because each runs the extension in
      both directions. What is at stake is the other party's global key: its MAC key, and for the
      garbler its free-XOR offset.
  - A further question, ours and not yet answered:
    - The correlation check's challenge is drawn from the pairwise stream agreed by a coin toss at the
      start of preprocessing. That is before the receiver sends its matrix, so the receiver can
      compute the challenge in advance.
    - KOS15 samples the challenge after the matrix, and so did Polytune until commit 1f56564
      (2024-11-06), which moved it to save communication rounds.
    - None of the published analyses covers the earlier ordering. Its consequences are not assessed
      here.
    - A comment asking the maintainers was drafted for #63 on 2026-10-26.
  - So nothing here claims malicious security for two hosts at this commit. The demo, with both
    parties in one tab, is unaffected.
  - What was fixed: the integration, not the library.
    - A failure of either half of the hybrid is now reported as that half's failure, so a protocol
      abort no longer reads as "node-seal did not load".
    - The cost model's split is published only under `_representative` keys.
    - Each run checks the circuit's index against the same argmax in the clear (`mpc_matches_clear`).
    - The crate refuses more than 255 values before compiling.
    - `tests/test_the_polytune_integration_reports_what_happened.py` fails if the Polytune commit
      moves without this review being redone.
  Every result carries a
  `ProvenanceBadge` with one of exactly **four** states - `MEASURED (local WebGPU)`, `MEASURED (local
  WASM)`, `MEASURED (plain JS)`, `SIMULATED` - together with the runtime evidence for that claim (`adapter.info`,
  `crossOriginIsolated`, `ort.env.wasm.numThreads`, bytes transferred vs served from cache). A result
  labelled `MEASURED` that produced no compute or network activity is treated as a defect.
  (This said "five states" including `REMOTE` until Round 32. `REMOTE` was deleted in Round 24d when the
  hosted-GPU tier was declined, so the document described a badge state the code cannot emit - and the
  badge that *can* be emitted, `MEASURED (plain JS)`, was missing from the list.)
- **Published metrics are what this repository measured, not what the papers report.** Where a figure
  is carried from the literature instead of reproduced - the breast-cancer PRS AUROC being the clearest
  case, since 1000 Genomes has no case/control status to compute one from - it is labelled as external.
  Where a reproduction falls short of its published baseline, the gap and its cause are printed rather
  than closed by changing the metric.

- **No operator visibility into the public demo.** The client-only build serves no `/metrics` (404 by
  `Caddyfile.client-only`) and logs nothing server-side, so a demo that breaks in a visitor's browser is
  undetectable by the operator. Accepted for now; recorded here so it is a choice rather than an omission.
- **The demo's own attack surface belongs in this document.** Until Round 17 the demo stack ran an API
  container that left `/api/run/*` writable by anonymous visitors while the page displayed "No computation
  is sent to a server". The container was removed and `/api` now 404s, but that exposure was recorded only
  in `architecture.md`; a threat model should carry it.
- **No dependency advisory stays open (Rounds 59 to 65).** Round 59 found three production advisories
  and two development-tooling ones. The two react-router ones (an open redirect through a link target
  built from attacker input, and constructor injection during server-side hydration) were fixed by the
  upgrade to react-router 7.18.4 in Round 61; before that they were unreachable, because the app builds
  no link target from input and renders a declarative `<BrowserRouter>` with no hydration.
  `tests/test_navigation_targets_are_not_built_from_input.py` still fails the day a link target is
  built from a query parameter, because that would be an open redirect under any router version. The
  vite and esbuild advisories, dev-server issues unreachable from the public image, were fixed by vite
  6.4.5 in Round 62. The zod one, reached through the archived `mpc-framework`, left with that package
  in Round 65. `npm audit` now reports none, development tooling included; the register's Round 59 to
  65 sections carry the evidence.
  **Kept at zero since Round 67, rather than measured at zero by hand.** `scripts/audit_deps.sh` runs
  daily in `.github/workflows/audit.yml`, and whenever a manifest, a lockfile or the policy changes.
  It runs npm audit over `web/`, and cargo-deny over the Rust crates compiled into the MPC wasm, which
  had never been audited before. The cargo-deny policy (`wasm/qsh-mpc/deny.toml`) fails on every
  RustSec advisory and on any source other than crates.io and the pinned Polytune repository. It
  ignores one advisory: RUSTSEC-2025-0141, which says bincode 2.0.1 (pulled by Polytune) is
  unmaintained. That advisory reports no vulnerability, and there is no safe upgrade. A vulnerability
  in bincode would be a separate advisory, and it would fail the check.
- **There is no `REMOTE` provenance.** The scheme has exactly four states - `MEASURED (local WebGPU)`,
  `MEASURED (local WASM)`, `MEASURED (plain JS)` and `SIMULATED`. `remote` was **removed from the type**
  in Round 24d, not merely left unset: nothing in `web/src` calls the ZeroGPU Space, that Space returns
  HTTP 401, and a type member no code path can produce is the same defect as a metric with no reader.
  This bullet said "the five-state scheme below is four-state in practice" until Round 33, which
  described the state BEFORE the removal and contradicted the four-state count stated earlier in this
  same document. `measured-js` was added in Round 21: the bundled distilled model is a hand-written
  loop over plain JS numbers with no wasm module and no GPU, and it had been reporting
  `measured-wasm` - a claim wrong in the flattering direction.
- **The headline comparison row is simulated.** The QSMPC-FHE row is a state-vector simulation
  (8 qubits by default) of an untrained QOTP + Clifford+T circuit, and the simulated-vs-hardware caveat above
  covers it. The quantum-advantage score once derived from it is retired from display (the full
  deployment's `/api/benchmark` still returns it), and so is the 0-100 security score; see
  `../references/security-scoring-fact-check.md`.

## Genomics: this system protects the computation, and the computation is not where the leak is

Recorded 2026-08-10 after a literature check, because it is the sharpest gap in the project's threat
model and it is a gap of FRAMING rather than of implementation.

The genomics use case computes pooled allele counts under secure summation, so no party sees another
site's genotypes. That is a real input-confidentiality property and it is implemented. It is also
not the threat the genomics-privacy literature is concerned with.

**Pooled allele counts are the canonical genomic privacy failure.** Homer et al., PLoS Genetics
4(8):e1000167 (2008) - the paper that caused NIH to withdraw public aggregate GWAS data - states
that composite statistics across cohorts, such as allele frequency or genotype counts, do not mask
identity within genome-wide association studies. It detects an individual contributing under 0.1% of
a mixture from roughly 50,000 SNPs.

**Releasing frequencies is worse than releasing booleans.** A 2025 beacon-reconstruction result
(DOI 10.1093/bioinformatics/btaf273) reports genome reconstruction at **70% F1 using allele
frequencies against 53.4% using binary yes/no responses** - so publishing the counts this pipeline
computes raises reconstruction risk by about 17 points over publishing mere presence.

**A single released score can be enough, for a small score.** A Feb 2026 preprint (Nikitin and
Gursoy, DOI 10.64898/2026.02.16.706191) reconstructs genotypes from polygenic risk scores by solving
a subset-sum problem: of 4,723 PGS Catalog models, at least 447 would be vulnerable if patient-level
scores were shared, and a typical 27-locus PRS uniquely identifies 95% of individuals in a
450,000-participant cohort. The recovery covers scores of up to about 50 variants and the
vulnerability assessment about 80, so PGS000004's 313 variants sit outside the recovery regime,
though a high-precision individual score still carries uniqueness and linkage risk. Non-European
ancestries are at greater risk, because GWAS panels are less diverse.

### What this means here, stated precisely

Encryption of the computation is **orthogonal** to this threat. The secure summation is not
weakened by any of the above; it simply addresses a different adversary. The exposure is the
released aggregate, and the mitigation for that is **output perturbation** - differential privacy on
the published counts - which this project does **not** implement.

Nor does anyone else in deployed crypto-genomics: SF-GWAS (Nature Genetics, Feb 2025), PP-GWAS,
SECRET-GWAS, privateQTL and the CKKS polygenic-score system HEPRS (Cell Reports Methods, Jan 2026)
all decrypt and release plaintext statistics with no output noise. SF-GWAS argues explicitly that DP
is impractical at GWAS scale because releasing statistics for many variants would require
substantial noise; that view is now contested, since GOPHER (Nandi, Neel and Cho, arXiv:2512.04225)
releases complete genome-wide statistics at $`\epsilon = 3`$, reporting r = 0.85 to 0.86 in its own
evaluation. So this is a field-wide gap rather than an oversight peculiar to this project -
which is a reason to state it, not a reason to be quiet about it.

### The honest claim, and the one to avoid

**Defensible:** "no party learns another site's genotypes during the computation."

**Not defensible:** any implication that pooling under encryption makes the *released* counts safe to
publish. It does not, and the literature is unambiguous.

Closing this properly means adding calibrated noise to the released aggregate with a stated $`\epsilon`$,
and reporting the utility cost - a UK Biobank study (DOI 10.1093/jamia/ocaf090, Jun 2025) found odds
ratios destabilising below $`\epsilon \approx \ln 2`$, with some reversing sign. That is a design decision
with a real accuracy price, and it is recorded here as open rather than silently absent.
