# Threat model

## Adversaries
- **Harvest-Now-Decrypt-Later (HNDL)**: records ciphertext today to decrypt once a CRQC
  exists. Mitigation: PQC-first hybrid (ML-KEM+X25519) on every channel - confidential
  as long as *either* primitive holds; the X25519 half covers today, ML-KEM the quantum future.
- **Quantum adversary at Q-Day**: breaks RSA/ECC. Mitigation: ML-KEM/ML-DSA (FIPS 203/204);
  QKD adds information-theoretic key material as a *hybrid add-on*, never standalone - following BSI
  TR-02102-1 v2026-01, not NSA, which does not endorse QKD in any form for NSS. The usual
  citation for that is NSA CNSA 2.0, but no NSA page is retrievable (HTTP 403 to every automated
  fetch), so the quotable source is **BSI TR-02102-1 v2026-01**: QKD *"is currently not ready for
  use from a security point of view"*.
- **Honest-but-curious compute server** (delegated AI): would read inputs/model. Mitigation:
  QHE (QOTP perfect hiding) for the quantum path; CKKS for the classical path; secure
  aggregation hides per-client gradients in federated learning -- **but see the residual risk below:
  as shipped it is never keyed, so this mitigation is not currently in force.**
- **Malicious MPC participant**: injects bad inputs / aborts. Mitigation: identifiable-abort
  pattern (scaffold) and QOT test rounds; PSI reveals only the intersection.
- **QKD link attacker**: intercept-resend, PNS, Trojan-horse, detector-blinding, RNG
  manipulation. Mitigation: QBER monitoring + two-intensity decoy-state single-photon bounds; QRNG
  health + **SP 800-90B min-entropy** and section 4.4 continuous health checks guard the entropy source; the
  config-gated **leakage-resilient PQC+QKD combiner**. Note the property carefully: `src/qsh/pqc/transport.py`
  explicitly declines the "secure-if-either" framing this line used to use, because plain XOR already gives
  that and the cited paper never claims superiority over XOR. What it does claim is narrower -- if the
  combined output AND one input key are compromised, the remaining key material retains quantifiable
  min-entropy. Further, the `its_toeplitz` extractor's seed is `SHAKE-256` over the same two keys it
  extracts from (`transport.py:53`), and the Leftover Hash Lemma needs a seed independent of the source --
  so as implemented it is a **computationally-seeded** extractor, not an unconditionally
  information-theoretic one. The docstring says so; the label "ITS" does not; + QML-IDS (WS2) classifying 7 attack types against normal traffic (8 classes).
- **Network MITM / tampering**: Mitigation: AEAD (AES-256-GCM) record layer + ML-DSA
  message authentication over the hybrid session.

## Trust assumptions
- Endpoints (clients) are honest and hold their own secret keys.
- The QKD hardware is modelled by simulation; real deployments must address device
  imperfections (the WS2 IDS targets exactly these).
- The crypto/compute libraries are trusted as implemented (pin versions + review): server-side
  TenSEAL / OpenFHE / HEonGPU / liboqs / MPyC; client-side **node-seal** (CKKS) + **emp-wasm /
  mpc-framework** (garbled-circuit 2PC) for the in-browser demo; the **MCP Python SDK** (`MCPServer`, named `FastMCP` before v2.0.0) for
  the full-deployment MCP server.
- **MCP tool dispatch** (full deployment): the quantum-safe channel authenticates + seals the
  `tools/call` and its result (ML-DSA + AES-GCM), but the tool *logic* runs server-side and is trusted;
  the official filesystem MCP server is sandboxed to a use-case data dir. The public demo runs no
  server-side MCP (client mirror only), so it adds no server trust surface.

## Key management (what each key is, and what it is not)

Enumerated because "key management is out of scope" reads as *not yet built*, when in fact several
keys exist and one of them is publicly recomputable.

| Key | Lifetime | Where it lives | Caveat |
|---|---|---|---|
| Hybrid KEM (ML-KEM-768 + X25519) | Per session, ephemeral | Process / tab memory | Both halves genuinely computed on both sides. Never persisted, never escrowed. |
| QKD-derived key material | Per run, opt-in | Memory | Simulated channel - no QKD hardware is involved anywhere in this project. |
| ML-DSA signing keypair | Per run, discarded | Generated inside the signature stage | The demo now records the public half and the exact signed payload; the **backend** path still discards the public key and keeps only a 12-byte signature prefix, so a full-mode run's signature is not third-party verifiable. |
| Secure-aggregation pairwise masks | Per aggregation | Derived from `sha256(f"pair-{i}-{j}")` | **A public constant.** Anyone can recompute every mask. Wiring the PQC/QKD layer in is what would make this hiding; nothing does that yet. |
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
  security when the adversary sees DECRYPTION RESULTS - and their attack needs **one** decryption
  query, recovering the secret as `s = a^-1 (m'' - b)`. Demonstrated against HEAAN, SEAL, HElib and
  PALISADE, "within several seconds".
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

**How much of this reaches this project.** The public demo is the benign case and should be stated
as such rather than claimed as a defence: the CKKS keys are generated in the browser tab, used in
that tab, and the decrypted class is shown to the same person who holds the secret key. There is no
decryption oracle exposed to anyone else, so an IND-CPA-D adversary would have to be the key holder
attacking themselves.

**The server-side deployment is the case that would need work.** If a model owner ran CKKS on a
server and returned decrypted results to a client, that is precisely the CPAD setting, and the
countermeasure - noise flooding, with its documented precision cost - would be required before any
production claim. This project does not implement it, does not use DE-CKKS, and does not perform
CKKS-to-FHEW scheme switching (ePrint 2026/285 breaks PEGASUS-style switching in OpenFHE, so this is
worth stating explicitly rather than leaving to inference).

**No standard requires testing for this.** The HomomorphicEncryption.org standard contains no
testing or conformance clause and states that CKKS "is not described in this standard"; ISO/IEC 28033 is not published, and the part that matters here is **Part 3** - mechanisms for arithmetic on approximate numbers, i.e. CKKS - which reached **FDIS stage 50.00 on 2026-06-11** (Part 1 general: DIS 40.60, 2025-12-11; Part 4: FDIS 50.00; Part 5: WD 20.60). Part 1 scopes correctness out. Naming "Parts 1-4 at DIS" was imprecise in both directions: Part 3 is further along than DIS and is the CKKS part, so this claim has a shelf life of months and carries a date for that reason. Checked 2026-08-10. So this section is a
statement of known limitations, not a conformance gap against any published requirement.

## Explicitly out of scope (PoC)
- Side-channel / fault attacks on the host.
- Formal verification of the protocols (referenced papers carry the proofs).
- Production key management / HSM integration (crypto-agility hooks are provided).

## Residual risk & honest limitations
- Quantum-channel results (QKD distance/QBER gains, photonic inference, full qOLE) are
  **simulated**, not hardware-validated.
- The QHE homomorphic evaluation is exact for Clifford circuits with key tracking;
  non-Clifford (trainable) layers require magic-state/T-gate corrections in a real device
  (their cost is reported as a T-count, not executed fault-tolerantly here).
- PSI is semi-honest keyed-hash over the quantum-safe channel; a malicious-secure OPRF
  PSI is future work.
- **Federated averaging is not a privacy guarantee -- and as shipped it hides nothing.** The
  pairwise-mask secure aggregation in `qsh/ai/federated.py` is intended to hide an *individual client's*
  update from the aggregator, and that is all it would claim. **It does not currently achieve even that:**
  `SecureAggregator` is only ever constructed with no keys (`federated.py:175`, `tests/test_ai.py:26`), and
  the default pairwise keys are `sha256(f"pair-{i}-{j}")` -- derived from a public constant, so anyone can
  recompute every mask and unmask any individual update. The masks are real; the secrecy is not. Wiring the
  PQC/QKD layer into the aggregator is what would make the scheme hiding, and nothing does that yet. It does not bound what the trained model itself leaks: gradient inversion against a shared
  model is an established attack. A formal claim would require a stated DP-SGD budget, which this
  path does not provide, so none is made.
- **Not everything the demo displays was computed.** **One** thing cannot be done in a browser at all
  and is labelled `SIMULATED` rather than quietly approximated: QKD has no browser API and cannot have
  one, so every QKD figure comes from an analytic channel model. *This bullet used to list three.* The
  other two were claims about specific libraries, and they no longer describe what ships: TFHE-rs' WASM
  build does not support FHE computations, but the demo does not use TFHE-rs - it runs **node-seal**
  (Microsoft SEAL compiled to WASM) for real CKKS (`web/src/lib/clientFHE.ts`); and JIFF is indeed
  unmaintained and the PSE MPC Framework sunset, but the demo runs **`mpc-framework` + `emp-wasm-engine`
  + `summon-ts`** authenticated garbling (`web/src/lib/clientMPC.ts`). Both are real, measured, and
  described as such in `architecture.md` and `crypto-agility.md` - this bullet contradicted them.
  Caveat that *does* apply to the MPC path: both "parties" share one JS heap, so the party separation
  is notional (stated at `clientMPC.ts:10-12`), and it is a demonstration of the protocol, not two
  mutually distrusting hosts. Every result carries a
  `ProvenanceBadge` with one of exactly five states - `MEASURED (local WebGPU)`, `MEASURED (local
  WASM)`, `MEASURED (plain JS)`, `REMOTE`, `SIMULATED` - together with the runtime evidence for that claim (`adapter.info`,
  `crossOriginIsolated`, `ort.env.wasm.numThreads`, bytes transferred vs served from cache). A result
  labelled `MEASURED` that produced no compute or network activity is treated as a defect.
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
- **`REMOTE` is unreachable as deployed.** The five-state provenance scheme below is four-state in
  practice: no code path sets `remote`, and nothing in `web/src` calls the ZeroGPU Space. The four
  that do occur are `MEASURED (local WebGPU)`, `MEASURED (local WASM)`, `MEASURED (plain JS)` and
  `SIMULATED`. `measured-js` was added in Round 21: the bundled distilled model is a hand-written
  loop over plain JS numbers with no wasm module and no GPU, and it had been reporting
  `measured-wasm` - a claim wrong in the flattering direction.
- **The headline comparison row is simulated.** The QSMPC-FHE row that drives the quantum-advantage
  score is an 8-qubit state-vector simulation. The simulated-vs-hardware caveat above covers QKD,
  photonic inference and qOLE, but not this row - it should. (The 0-100 security score this bullet
  used to cite was retired in Round 22 and is no longer computed for display; see
  `../references/security-scoring-fact-check.md`.)

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

**A single released score can be enough.** A Feb 2026 preprint (DOI 10.64898/2026.02.16.706191)
reconstructs genotypes from polygenic risk scores by solving a subset-sum problem: of 4,723 PGS
Catalog models, at least 447 would be vulnerable if patient-level scores were shared, and a typical
27-locus PRS uniquely identifies 95% of individuals in a 450,000-participant cohort. Recovery then
re-identifies the individual *or a genetic relative* in a genealogy database at near 100% accuracy.
Non-European ancestries are at greater risk, because GWAS panels are less diverse.

### What this means here, stated precisely

Encryption of the computation is **orthogonal** to this threat. The secure summation is not
weakened by any of the above; it simply addresses a different adversary. The exposure is the
released aggregate, and the mitigation for that is **output perturbation** - differential privacy on
the published counts - which this project does **not** implement.

Nor does anyone else in deployed crypto-genomics: SF-GWAS (Nature Genetics, Feb 2025), PP-GWAS,
SECRET-GWAS, privateQTL and the CKKS polygenic-score system HEPRS (Cell Reports Methods, Jan 2026)
all decrypt and release plaintext statistics with no output noise. SF-GWAS argues explicitly that DP
is impractical at GWAS scale because releasing statistics for many variants would require
substantial noise. So this is a field-wide gap rather than an oversight peculiar to this project -
which is a reason to state it, not a reason to be quiet about it.

### The honest claim, and the one to avoid

**Defensible:** "no party learns another site's genotypes during the computation."

**Not defensible:** any implication that pooling under encryption makes the *released* counts safe to
publish. It does not, and the literature is unambiguous.

Closing this properly means adding calibrated noise to the released aggregate with a stated epsilon,
and reporting the utility cost - a UK Biobank study (DOI 10.1093/jamia/ocaf090, Jun 2025) found odds
ratios destabilising below epsilon about ln(2), with some reversing sign. That is a design decision
with a real accuracy price, and it is recorded here as open rather than silently absent.
