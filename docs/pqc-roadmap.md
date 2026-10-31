# PQC migration & threat roadmap (recommendations)

This project is designed to slot into the 2026-2030 post-quantum migration. The items
below are the recommended future tracks (PoC milestones M3/M4).

## Standards alignment (done / in progress)
- **NIST FIPS 203 (ML-KEM)** - implemented in the hybrid KEM. yes
- **NIST FIPS 204 (ML-DSA)** - implemented for message authentication. yes
- **NIST FIPS 205 (SLH-DSA)** - add as a hash-based signature option (not in this liboqs build). []
- **NSA CNSA 2.0** - national security systems (NSS) only: NSA *"does not mandate requirements on any
  other entity outside of NSS"* (CNSA 2.0 and Quantum Computing FAQ, Ver. 2.2, October 2026). CNSSP 15
  requires CNSA 2.0 algorithms in all new NSS acquisitions from **2027-01-01**; equipment that cannot
  support them is phased out by **2030-12-31**; CNSA 2.0 is mandatory for all use by **2031-12-31**;
  **2035** is the NSM-10 goal. A 2033 date appears only in the 2022 advisory's per-technology table,
  not in CNSSP 15 or the FAQ. The required parameter sets are ML-KEM-1024, ML-DSA-87, AES-256 and
  SHA-384/512; this repository's defaults (ML-KEM-768 + X25519, ML-DSA-65, HKDF-SHA256) are none of
  them, so nothing here is CNSA 2.0 by default (ML-KEM-1024 and ML-DSA-87 are selectable). NSA's QKD
  position is not "non-recommended standalone": the FAQ answers **"No"** to protecting an NSS with QKD,
  in any configuration. Verified 2026-10-23 against FAQ Ver. 2.2 and CNSSP 15; nsa.gov still returns
  HTTP 403 to scripts, so the FAQ was read from an archived copy of the PDF the NSA resource hub links.
  On QKD, **BSI TR-02102-1 v2026-01 section 2.1 "Use of Quantum-Safe Mechanisms"** (pp.28-29; the QKD
  paragraph is on p.29) says QKD *"is currently not ready for use from a security point of view... the
  BSI does not recommend QKD protocols at this time"*. What BSI does recommend is a hybrid of a
  quantum-safe and a classical key agreement (section 2.2), which is the ML-KEM + X25519 layer. Keeping
  QKD as an optional, simulated add-on to that hybrid, never standalone, is **this project's** choice;
  neither BSI nor NSA recommends it. yes

## Crypto-agility (done)
Algorithms resolve through `CryptoRegistry` at runtime - ML-KEM-768 <-> -1024, QKD <->
PQC-only, TenSEAL <-> OpenFHE. A future PQC algorithm is a registry entry + config change,
not a code rewrite. Add remote re-key + in-flight algorithm substitution next. []

## AI-QKD (WS1) + entropy: the optimizer is built and measured; the reference architecture's real-time loop is not
- **OptiQKD / AML-QKD** - an AI parameter optimizer (scikit-learn GBM/GP surrogate + gradient-free
  coordinate ascent) tunes the **two-intensity decoy-state** parameters `basis_prob` / `mu_signal` /
  **`mu_decoy`** / `pp_threshold` (Ma-Qi-Zhao-Lo 2005 single-photon Y1/e1 bounds - `mu_decoy` genuinely
  moves the SKR) to **raise the measured secret-key rate** vs the default baseline on the same
  simulated channel, and reports a **convergence metric** (evaluations-to-95 %). An **E91 / CHSH**
  entanglement protocol (`src/qsh/qkd/e91.py`) is selectable via `protocol="e91"`.
  (`src/qsh/qkd/optimize.py`; grounds the report's WS1 "OptiQKD" theme - OptiQKD arXiv:2603.04192v1 +
  AML-QKD arXiv:2603.04192v2, Mohamed et al. 2026, one paper in two versions: v1 reported +20-30 % and v2 revised that down to +14-25 % SKR via TCN+RL, in simulation. Benchmarking against v1 would be quoting a figure the authors superseded.)
  Opt-in per run; a simulation, not deployed QKD. yes
- **QRNG** - a local **state-vector quantum-measurement** entropy source with the **SP 800-90B section
  4.4 continuous health tests** (Repetition Count, cutoff $`C = 1 + \lceil -\log_2(\alpha)/H \rceil`$, and Adaptive
  Proportion over a 1024-sample window, both at $`\alpha = 2^{-20}`$, plus the section 4.3 start-up test over
  1024 samples) + a **SP 800-90B most-common-value min-entropy estimate**. Three SP 800-22 tests
  (monobit / runs / block-frequency) are also computed, but as an **output-stream sanity check only**:
  SP 800-90B section 4.1 states that SP 800-22-style tests *"will almost always fail, and thus are not
  useful for monitoring the noise source"*, so the pass/fail verdict is gated on the 90B tests. Round 17
  implemented the 90B tests; before that this line described an "SP 800-22-style health suite", which
  was exactly the conflation section 4.1 warns against. Post-processing = von Neumann /
  **Toeplitz (Leftover-Hash-Lemma-bounded output, epsilon-parameterised)** / **seedless two-source** extractor
  (`src/qsh/qrng/`); it **can** seed the BB84 basis/bit choices - **opt-in and off by default** (`qrng.enabled = False`), so the default run uses the original uniform-basis behaviour. A QRNG that targets
  full-entropy output would be an RBG3 construction under **NIST SP 800-90C** (final, 2025-09-25),
  which this simulated source does not claim to be. Optional real-hardware source = the **ANU QRNG**
  API (opt-in, online). Honestly labelled: sim, not HW. yes
- **Hybrid PQC+QKD key combiner** - `src/qsh/pqc/transport.py` offers a config-gated Toeplitz +
  Quantum-Leftover-Hash-Lemma combiner (`hybrid_combiner_mode="its_toeplitz"`) alongside the
  default dual-PRF HKDF - the session key stays secure if EITHER the PQC or the QKD input holds. It
  follows Giestinhas & Spiller, arXiv:2603.26907, 2026 (seedless QRNG bootstrapping + hybrid PQC-QKD
  key combination via strong extractors), but is seeded with SHAKE-256 of its own inputs, so the
  paper's leakage-resilience property, which needs an independent seed, does not hold: as
  implemented it is computational, not information-theoretic, and the demo's QKD input is a fixed
  constant. yes

## Reference implementations (2026, commercial-OK)
Grounded, actively-maintained OSS to deepen each workstream (all MIT / Apache-2.0 / BSD; adopt as the
scaffold hardens). QRNG + OptiQKD above are built from this set; the rest are adoptable next steps:

| WS | Reference implementations |
|----|---------------------------|
| WS1 AI-QKD | TNO-Quantum `communication.qkd_key_rate` (Apache-2.0), `sequence-toolbox/SeQUeNCe` (Argonne; GitHub reports the licence as NOASSERTION, so confirm before depending on it), `DLR-RM/stable-baselines3` (MIT) - **QKDNetSim removed from this table: it is GPL-2.0**, i.e. strong copyleft, and this table is specifically the commercial-OK set |
| WS2 QML-IDS | `rdisipio/qlstm` (PennyLane QLSTM), `ACE07-Sev/QSVM` (12-variant QSVM), `qiskit-community/qiskit-machine-learning` |
| WS4 QHExAI / QFL | `elucidator8918/QFL-MLNCP-NeurIPS` (QFed+FHE), `goldenskygiang/flower-fhe`, `ibarrond/Pyfhel`, OpenFHE, Flower |
| entropy | `dorahacksglobal/quantum-randomness-generator`, NIST SP 800-90B (health testing, required), NIST SP 800-22 (output sanity check only) |
| cross-cutting PQC/QKD | `open-quantum-safe/oqs-provider`, `rosenpass/rosenpass`, `arnika-project/arnika`, ETSI GS QKD 014 KME (`thomasarmel/qkd_kme_server`) |

## Recommended future workstreams
1. **NIST-aligned IDS against HNDL / Q-Day** - extend WS2 (QML-IDS) into a full network
   IDS plane: combine the QKD-attack classifier with classical NIDS (e.g. CICIDS-style
   flows), map detections to MITRE ATT&CK, and alert on harvesting patterns. Wire to
   SIEM/SOAR. (Report Phase-4 of WS2.)
2. **Hardware QKD field test** - replace the BB84 simulator with a real QKD testbed (or
   ETSI GS QKD 014 KME API) to validate the simulated QBER/SKR. (Milestone M4.)
3. **Malicious-secure MPC** - upgrade PSI to a post-quantum OPRF and add identifiable
   abort with ML-DSA-signed transcripts.
4. **Fault-tolerant QHE costing** - turn the measured T-counts into resource estimates
   under a chosen QEC code; track against Gidney's **<1 million noisy qubits** for RSA-2048 (arXiv:2505.15917) - note that is a *qubit* count; the "~1.1M T-gate" figure previously quoted here appears in none of the cited papers.
5. **GPU FHE at scale** - build the HEonGPU + PIGEON/hpmpc path (`scripts/build_gpu_mpc.sh`)
   for batched CNN/Transformer private inference; compare HEonGPU (GPU) vs OpenFHE (CPU) CKKS.
6. **Continuous PQC inventory** - scan dependencies/protocols for quantum-vulnerable
   crypto (a "crypto bill of materials") to drive migration priority.

## Timeline anchors

**Scope of this list, stated because it is a limit rather than a judgement.** Every instrument below is
a **US** one, apart from the BSI, Japanese and EU entries. The EU entry was read in its primary
document on 2026-08-15; `references/REFERENCES.md` records the two ways an earlier lead overstated it.

- **2026-12-31 / 2030-12-31 / 2035-12-31** - **EU: the NIS Cooperation Group PQC transition
  milestones.** *A Coordinated Implementation Roadmap for the Transition to Post-Quantum
  Cryptography*, Part 1, Version 1.1, EU PQC Workstream, dated 11.06.2025 - the first deliverable of
  the NIS CG work stream on PQC. Its section 4 timeline, in the document's own terms:
  - **by 31.12.2026** - First Steps implemented by all Member States; initial national PQC
    transition roadmaps established; transition planning and pilots **initiated** for high- AND
    medium-risk use cases
  - **by 31.12.2030** - Next Steps implemented; the transition for **high-risk** use cases
    **completed**; planning and pilots for medium-risk completed; quantum-safe upgrades enabled by
    default
  - **by 31.12.2035** - the transition for **medium-risk** completed, and for low-risk *"completed
    as much as feasible"*

  The executive summary's wording for 2035 is *"the transition should be completed for as many
  systems as practically feasible"* - weaker than a full-transition deadline, and worth stating
  because the lead that carried this for nine rounds recorded it as one. The roadmap is grounded in
  BSI's assessment of quantum-computer development and in *Securing Tomorrow, Today*, a joint
  statement by partners from 21 European states, and its steps are framed as supporting NIS2
  compliance rather than as a separate obligation.

  **What it means here:** nothing in this deliverable is a Member State system, so no milestone
  binds it. It is recorded because the demo is reachable from the EU and a roadmap with no EU
  milestone reads as parochial rather than scoped - which is what the Round 33 lead said, and was
  the one part of it that held.
- **2026-09-21 (passed)** - **FIPS 140-2 certificates moved to the CMVP Historical List.** NIST's own
  wording: *"FIPS 140-2 active modules can be used until this date for new systems. After this date,
  FIPS 140-2 validation certificates will be moved to the Historical List."* Federal agencies may now
  use those modules only in EXISTING systems; new systems need FIPS 140-3. Verified at csrc.nist.gov
  2026-08-06 and again 2026-10-23, after the date. It does not bind this
  repository - nothing here is FIPS-validated and nothing claims to be - but it binds anyone
  procuring a module to run this kind of workload, so it belongs in a migration roadmap.
- **2027-01-01 / 2030-12-31 / 2031-12-31** - CNSA 2.0 under CNSSP 15, national security systems only:
  new acquisitions, then phase-out of equipment that cannot support it, then all use. Not a target for
  this repository's defaults, which are not CNSA 2.0 parameter sets.
- **2029** - Cloudflare *targets* full PQ security for its product suite, *including PQ authentication*,
  and calls its intermediate milestones changeable (blog, 2026-04-07); Google sets 2029 as its own PQC
  **migration** timeline and prioritises PQC migration for authentication services (blog, 2026-03-25). Neither is a binding
  commitment, and neither post ranks itself against the other -> supply-chain expectations.
- **after 2030 / after 2035** - **NIST IR 8547**, *Transition to Post-Quantum Cryptography
  Standards* (final, October 2026, published 2026-10-27, doi:10.6028/NIST.IR.8547; Moody, Perlner,
  Regenscheid, Robinson and Cooper; it supersedes the 2024-11-12 initial public draft): every
  quantum-vulnerable public-key algorithm is deprecated after 2030 and disallowed under NIST
  standards and guidelines after 2035 - in Table 2, ECDSA and RSA signatures at 112 bits of security
  or more and EdDSA at 128 or more; in Table 4, key establishment by finite-field and elliptic-curve
  DH and MQV and by RSA, at 112 bits or more. Deprecated means usable at a risk the data owner must
  assess. The draft deprecated only 112-bit strength after 2030; the final also deprecates RSA-3072
  and P-256 after 2030. This replaces SP 800-57's projected disallowance of 112-bit public-key schemes
  on 2031-01-01, and it does not alter the earlier EO 14412 and OMB M-26-15 deadlines for federal
  high value assets and high impact systems (below). Plan field validation by 2027 Q1.
- **2030-12-31 / 2031-12-31** - **EO 14412**, *Securing the Nation Against Advanced Cryptographic Attacks*
  (signed 2026-06-22, 91 FR 38483): PQC key establishment on high-value assets and high impact systems by
  2030-12-31 and digital signatures by 2031-12-31 (section 4(b)). For contractors, section 6(c) directs
  the FAR Council to publish a *proposed* rule within 180 days (about 2026-12-19) amending the FAR to
  require covered contractors to comply by 2030-12-31 with NIST's FIPS, including all applicable FIPS
  incorporating PQC algorithms; none had appeared in the Federal Register as of 2026-10-30. **National Security Systems are excluded** and stay on CNSA 2.0. This is the binding US
  instrument for federal agencies; separately, EO 14306 and M-26-15 set TLS 1.3 by 2030-01-02.

  Note on the EO number, because secondary sources get it wrong: several write-ups call this **14409**.
  The Federal Register API returns **14412**, published 2026-06-25 at 91 FR 38483. The number above is
  correct.

  **There were TWO quantum executive orders published that day, and this roadmap cited one.** The other
  is *"Ushering in the Next Frontier of Quantum Innovation"*, Federal Register document **2026-12910**,
  **91 FR 38487**, published **2026-06-25** (EO 14412 is document 2026-12909 at 91 FR 38483). Recorded
  because a migration roadmap that cites one of two same-day quantum EOs looks like it missed one. Only
  its **title, document number, FR citation and date are verified** (Federal Register API, Round 32,
  re-stated here 2026-08-12); its text has not been read, so **it contributes no deadline to this list**
  and nothing below rests on it. EO 14412 remains the binding cryptographic instrument.
- **2026-10-22 / phased to 2035** - **OMB Memorandum M-26-15**, *Execution of the Migration to
  Post-Quantum Cryptography* (2026-06-24), which is how EO 14412 is actually executed and is newer than
  everything else in this list. Agency PQC Migration Plans were due to OMB and ONCD **within 120 days**
  of issuance, so about **2026-10-22** (now passed), and Action One is to mitigate as much quantum risk as feasible
  by **2030-12-31**. Five phases: **1** 2026-2027 strategy and discovery, **2** 2027-2028 pilots, **3**
  2028-2030 prioritised migration and key establishment, **4** 2031 signature migration, **5** 2035
  full migration. Not applicable to national security systems.

  Two consequences for this repository specifically:

  - Phases 3 and 4 each close with *"Ensure all systems are cryptographically agile."* That is a
    directive naming the exact property [`crypto-agility.md`](crypto-agility.md) is about, so that
    document is no longer only good practice.
  - The memorandum directs agencies to **align their plans with NIST IR 8547 "or successor
    document"**. IR 8547 stayed an initial public draft until NIST published the final on
    2026-10-27 (doi:10.6028/NIST.IR.8547), so the memorandum now points at a final document, and the
    2030 and 2035 dates above cite it.
- **2035** - Japan's Cabinet Secretariat interim report (Nov 2025) sets PQC migration at **2035**, with the
  detailed roadmap to be drafted during FY2026.

## Algorithm backups worth tracking

- **HQC** was selected in **March 2025** as the code-based KEM backup to ML-KEM, with standardisation
  expected 2026-2027. It matters to this project specifically because the whole architecture is a
  crypto-agility registry: a second KEM family resting on a different hardness assumption is exactly
  what that registry exists to make swappable. It is **not implemented**, and that is a deliberate gap
  rather than an oversight, because a backup KEM is only worth wiring once its parameter sets are
  final. Re-checked 2026-10-27: NIST has published no HQC draft standard (CSRC lists none), and liboqs
  **disables HQC by default since 0.13.0** after a "theoretical design flaw" advisory
  (GHSA-3rxw-4v8q-9gq5, CVSS 3.7), pending an updated specification. The venv's liboqs 0.14.0 lists
  HQC-128/192/256 and enables none of them. This line said "liboqs carries HQC" until Round 75, which
  is true of the source tree and not of any default build.

## Standards status re-checked 2026-10-27

Read at NIST's PQC news page and the pqc-forum on 2026-10-27. IR 8547's final appeared on
2026-10-27, after this table was read; its row was re-read at the CSRC page on 2026-10-30.

| Document | Status | Why it matters here |
|---|---|---|
| **NIST SP 800-227** - Recommendations for KEMs | **Final, 2025-09-18** | The combiner reference. This repository's HKDF combiner is a section 4.6.2 two-step KDF (equations 13-14) over the transcript section 4.6.3 binds; it is not that section's KeyCombineCCA_H. |
| **FIPS 206** (FN-DSA) | **No draft.** 2026-09-28: NIST's FIPS 206 team posted a new plan to the pqc-forum - a single fixed-point signing procedure, validatable by known-answer tests | Nothing to wire yet; the signature registry stays ML-DSA. |
| **HQC** (FIPS 207) | **No draft** as of 2026-10-27; liboqs disables it by default (see above) | The backup KEM stays unregistered. |
| **NIST SP 800-230** (initial public draft) - additional SLH-DSA parameter sets for limited signature use | Released 2026-04-13, comments closed 2026-06-12 | Not used here; tracked because SLH-DSA is the hash-based fallback. |
| **NIST SP 800-133 Rev. 3** (initial public draft) - key generation | Released 2026-04-17, comments closed 2026-06-16 | Governs how the demo's keys would have to be generated in an approved product. |
| **NIST IR 8610** - additional signatures, round 2 status | 2026-05-14, nine candidates advanced to round 3 | The on-ramp; no algorithm to adopt yet. |
| **NIST IR 8547** - transition to PQC | **Final, October 2026** (NIST news 2026-10-27, doi:10.6028/NIST.IR.8547; supersedes the 2024-11-12 draft) | Quantum-vulnerable public-key algorithms deprecated after 2030 and disallowed after 2035 (Tables 2 and 4). |

## Standards added since the last revision (verified 2026-08-03)

| Document | Status | Why it matters here |
|---|---|---|
| **NIST SP 800-90C** | **Final, 2025-09-25** | RBG constructions. A QRNG targeting full-entropy output is an RBG3 construction; the QRNG bullet above cites it beside SP 800-90B. |
| **NIST IR 8214C** - First Call for Multi-Party Threshold Schemes | **Final, Jan 2026** | Scope explicitly includes **FHE, MPC and ZKPs**. The most relevant NIST process to this project's WS3/WS4. |
| **NIST CSWP 39upd1** - Considerations for Achieving Crypto Agility | Final, 2026-06-29 | Supersedes the 2025-12 original; this is now *the* NIST crypto-agility reference, and `crypto-agility.md` cites it. |
| **ETSI GS QKD 020** - REST-based Interoperable KMS API | V1.1.1, 2026-06-29 | First cross-vendor KMS-to-KMS spec. `pqc/transport.py` targets GS QKD **014** only; 014 + 020 is the current pair. |
| **ETSI GS QKD 015 V2.1.1** - SDN control interface | Published 2022-04 | **Implemented in Round 18** (`qkd/sdn_controller.py`, mirrored in `web/src/lib/clientSdnController.ts`). Supplies the control plane the demo previously lacked, and its clause 1 scope statement is what settles that an MCP agent is an SAE rather than a controller. |
| **ETSI GS QKD 014 V1.1.1** - key delivery API | Published 2019-02, **still the only published edition** | **Implemented in Round 18** (`qkd/kme.py`). An ed2 stable draft (v1.3.1) existed as of 2025-06 but is not published, so "QKD 014 compliant" in this repository means the 2019 document. The mandated mutual TLS is **not** implemented - see `threat-model.md`. |
| **BSI TR-02102-1 v2026-01** | 2026-01-23 | The quotable basis for the hybrid PQC + classical key-agreement recommendation (section 2.2) and for BSI not recommending QKD (section 2.1). It does not recommend QKD as an add-on either: "QKD never standalone" is this project's rule, not BSI's. |
| **IACR ePrint 2026/935** - SoK: Private LLM Inference using Approximate HE | rev. 2026-06-03 | The citable ceiling: ~20 CKKS frameworks surveyed, largest demonstrated end-to-end model **8B parameters**, ~4 orders of magnitude slower than plaintext. Confirms nothing larger exists. |
| **IACR ePrint 2026/1231** - Hybrid AKE for QKD networks without signatures | 2026-06-10 | The closest published construction to `hybrid_combiner_mode="its_toeplitz"`: KEM-only hybrid AKE with a conditional information-theoretic guarantee when the QKD leg is uncompromised. This repository's combiner has no such guarantee: its seed is derived from its inputs, and its QKD leg is simulated. |
| **arXiv:2601.18511v2** - Scaling up FHE-based PPML | v2 2026-06-17 | Best published absolute: **Llama-3-8B under CKKS at 20 s prefill / 18 s per output token on 8x RTX PRO 6000**. Quote this rather than hand-waving FHE cost - and quote the PAPER, not the SoK that summarises it: `references/REFERENCES.md` now carries the primary entry with the abstract verbatim. Note the eight GPUs; this project has one and its FHE default is CPU. |

## EU AI Act Article 50

The analysis is [`docs/ai-act-art50-gap-analysis.md`](ai-act-art50-gap-analysis.md); this section is a
pointer and the dates.

- **Art. 50(1)**, telling a person they are interacting with an AI, applies from 2 August 2026, and the
  Agentic AI panel meets it with a disclosure shown unconditionally. Regulation (EU) 2026/1744 (the AI
  Act omnibus) gives it no transition.
- **Art. 50(2)**, machine-readable marking of generated content, is **not applicable**: the panel
  generates 96 tokens (Transformers.js) or 160 (WebLLM), under the 200-token "very short text"
  exception of the Code of Practice on Transparency of AI-generated Content (final 10 June 2026,
  assessed adequate on 8-9 July 2026; the Commission's Article 50 guidelines, C(2026) 5054, followed
  on 20 July 2026). The downloaded artifact's `provenance_note` block is voluntary provenance
  metadata, not an Article 50(2) marking layer, and it carries no compliance-status field.
- Systems placed on the market before 2 August 2026 have until 2 December 2026 for Art. 50(2) (new
  Art. 111(4)). Whether this demo was placed on the market before that date is not recorded in this
  repository; it does not change the conclusion above.
- Chapter VII (governance) has applied since 2 August 2025.
- Penalty exposure for Art. 50 non-compliance is up to EUR 15 million or 3 percent of worldwide annual
  turnover, whichever is higher.
- What would reopen Art. 50(2): the panel exceeding 200 tokens, generated text shipped inside a file
  container, or a usable C2PA text binding (Specification 2.4, 2026-04-01, defines text bindings; no
  released library implements them).

## Hybrid acceptance: the failure mode this design avoids, and where it would bite (added 2026-08-13)

**arXiv:2607.20800v3** (2026-07-23, revised 2026-08-12; PREPRINT - no journal_ref, no DOI),
*"Classical Acceptance Is Not Hybrid Authentication"*, tested **eight open-source path-validation
stacks over seven codebases, nine configurations over six certificate profiles**. On default paths,
**every stack that parsed a separable hybrid X.509 certificate accepted it** - and destroying the
post-quantum evidence while leaving the classical evidence valid **changed no verdict in any of the
27 cells**. Four of the stacks verify post-quantum signatures elsewhere on the same path, so
immature support does not explain it. Revoking a bound post-quantum certificate likewise changed no
verdict, because nothing consults it.

**This does not describe this project, and saying so precisely matters more than citing it.** The
finding is about *separable* hybrid credentials, where a verifier can reach a decision from the
classical half alone. This repository issues no X.509 at all: the hybrid is a **KEM combiner** -
ML-KEM-768/1024 and X25519 fed together through HKDF-SHA256 with the transcript bound in - c1 || c2 || ek1 || ek2, the ML-KEM ciphertext and the X25519 ephemeral public followed by both encapsulation keys, per NIST SP 800-227 section 4.6.3. The browser's call sites share one module, and
`npm run check:combiner` enumerates what it covers, so the count is derivable from a check rather
than restated in prose. `web/src/lib/clientMcp.ts` is deliberately NOT one of
them - it is a different composite scheme with its own separator, and the gate asserts they stay
apart
(`src/qsh/pqc/hybrid_kem.py`, mirrored in `web/src/lib/hybridCombiner.ts`, which `clientCompute.ts`
imports). Both secrets enter one hash, so there
is no path that reaches a session key from one of them; the code comment has said *"so neither half
can be cancelled out by an attacker who controls the other"* since it was written.

**Where it would bite if this project grew a transport.** `qkd/kme.py` records that a real ETSI
GS QKD 014 deployment authenticates SAEs by mutual TLS with the identity taken from the client
certificate, and that the in-process KME has no transport to attach that to. The moment that
transport exists, the certificate becomes the thing this paper is about, and `signatures.py`'s note
that ML-DSA is available *"for a hybrid signature if required by policy"* becomes a decision rather
than a capability. The paper's contribution is the shape of the answer: a **policy-parametric
contract** that carries the relying party's hybrid requirement into path validation and returns a
*labelled* acceptance, so "accepted" and "accepted on classical evidence alone" are distinguishable.

Two companion measurements from the same weeks, recorded as deployment context rather than as
requirements - both preprints: **arXiv:2608.02147** (2026-08-03) measures post-quantum TLS deployment
across UK internet sectors, and **arXiv:2607.29005** (2026-07-31) contrasts stated policy with
observed deployment. They bear on the timeline anchors above, not on this codebase.

## What a 2026 production deployment needs that this project does not implement

Verified 2026-08-09 against arXiv:2604.12985, "Quantum-safe IPsec in the banking industry"
(Vicente, Brito, Martin et al., 14 Apr 2026) - a real five-node deployment across a Madrid campus
plus private-cloud nodes in northern Spain and Mexico, mixing **DV-QKD and CV-QKD** from multiple
vendors.

The finding that matters here is the interface layer. That deployment had to integrate **three
mutually incompatible key-delivery interfaces: ETSI GS QKD 004, ETSI GS QKD 014, and Cisco SKIP**,
because a real multi-vendor network has all three. It concludes that quantum-safe communication in
financial networks is "not only technically feasible but also scalable, interoperable, and
resilient", and notes the standards gap it had to work around: PQC algorithms are not yet in
finalised IPsec standards.

**This project implements GS QKD 014 (`qkd/kme.py`) and GS QKD 015 (`qkd/sdn_controller.py`). It
does not implement GS QKD 004.** That is worth stating plainly because `sdn_controller.py:19` lists
"GS QKD 004, 014, 020" as the specs governing the quantum plane - which is correct as taxonomy and
which a reader may reasonably take as a claim of implementation.

The two are not interchangeable. **014 is a REST key-delivery API**: a stateless request for key
material by identifier. **004 is a stateful application interface**: an application opens a keyed
session and streams key material against it, with negotiated QoS. An application written against one
does not work against the other, which is exactly why the banking testbed needed both plus a
vendor-specific third.

Recorded as a gap rather than a task. Implementing 004 would be a real piece of work and this project
has no application that needs a key stream today - but "we implement the ETSI key-delivery interface"
is a claim that should say **which one**, and a buyer's technical due diligence will ask.

## The finance AML feature architecture, if that gap is ever closed

Round 24 measured why `finance_aml.gbm` sat at 0.0808 against a published 0.6323 - it has since
risen to 0.1519 by making the per-account aggregates causal - and ruled out the
threshold, unseen accounts, and the 4-hop reachability limit - see
`../references/finance-gap-investigation.md`. The residual is architectural: IBM's Graph Feature
Preprocessor differs from this pipeline in feature ARCHITECTURE, not in licence - the mechanism is
derived in `../references/finance-gap-investigation.md` and is not restated here. That difference is
implementable without `snapml` and is a bounded, well-specified workstream. A measured intermediate
step already exists: causal prior-at-time aggregation beat the shipped approach by 1.45x AUPRC at
matched feature count, with no leakage.
