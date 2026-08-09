# PQC migration & threat roadmap (recommendations)

This project is designed to slot into the 2026-2030 post-quantum migration. The items
below are the recommended future tracks (PoC milestones M3/M4).

## Standards alignment (done / in progress)
- **NIST FIPS 203 (ML-KEM)** - implemented in the hybrid KEM. yes
- **NIST FIPS 204 (ML-DSA)** - implemented for message authentication. yes
- **NIST FIPS 205 (SLH-DSA)** - add as a hash-based signature option (not in this liboqs build). []
- **NSA CNSA 2.0** - 2027 procurement mandate; QKD non-recommended standalone.
  `[UNVERIFIED - no primary NSA source is retrievable: nsa.gov and media.defense.gov return HTTP 403
  to every automated fetch. Secondary sources also report a 2033 milestone this section omits.]`
  For a *quotable* source on the QKD point, use **BSI TR-02102-1 v2026-01 section 2.1 "Use of Quantum-Safe Mechanisms"** (p.28): QKD
  *"is currently not ready for use from a security point of view... the BSI does not recommend QKD
  protocols at this time"*. The
  architecture keeps QKD strictly as a PQC-hybrid add-on, on BSI's authority. **NSA is not a source
  for this**: its CNSA 2.0 FAQ answers "No" to protecting an NSS with QKD and does not endorse a
  hybrid add-on either. yes

## Crypto-agility (done)
Algorithms resolve through `CryptoRegistry` at runtime - ML-KEM-768 <-> -1024, QKD <->
PQC-only, TenSEAL <-> OpenFHE. A future PQC algorithm is a registry entry + config change,
not a code rewrite. Add remote re-key + in-flight algorithm substitution next. []

## AI-QKD (WS1) + entropy (done)
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
  4.4 continuous health tests** (Repetition Count, cutoff `C = 1 + ceil(-log2(alpha)/H)`, and Adaptive
  Proportion over a 1024-sample window, both at alpha = 2^-20, plus the section 4.3 start-up test over
  1024 samples) + a **SP 800-90B most-common-value min-entropy estimate**. Three SP 800-22 tests
  (monobit / runs / block-frequency) are also computed, but as an **output-stream sanity check only**:
  SP 800-90B section 4.1 states that SP 800-22-style tests *"will almost always fail, and thus are not
  useful for monitoring the noise source"*, so the pass/fail verdict is gated on the 90B tests. Round 17
  implemented the 90B tests; before that this line described an "SP 800-22-style health suite", which
  was exactly the conflation section 4.1 warns against. Post-processing = von Neumann /
  **Toeplitz (Leftover-Hash-Lemma-bounded output, epsilon-parameterised)** / **seedless two-source** extractor
  (`src/qsh/qrng/`); it **can** seed the BB84 basis/bit choices - **opt-in and off by default** (`qrng.enabled = False`), so the default run uses the original uniform-basis behaviour. Optional real-hardware source = the **ANU QRNG**
  API (opt-in, online). Honestly labelled: sim, not HW. yes
- **Hybrid PQC+QKD key combiner** - `src/qsh/pqc/transport.py` offers a config-gated **leakage-resilient**
  Toeplitz + Quantum-Leftover-Hash-Lemma combiner** (`hybrid_combiner_mode="its_toeplitz"`) alongside the
  default dual-PRF HKDF - the session key stays secure if EITHER the PQC or the QKD input holds. (Grounds
  Giestinhas & Spiller, arXiv:2603.26907, 2026: seedless QRNG bootstrapping + hybrid PQC-QKD key combination
  via strong extractors.) yes

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
- **2026-09-21** - **FIPS 140-2 certificates move to the CMVP Historical List.** NIST's own wording:
  *"FIPS 140-2 active modules can be used until this date for new systems. After this date, FIPS
  140-2 validation certificates will be moved to the Historical List."* Federal agencies may then use
  those modules only in EXISTING systems, not for new procurement. Verified at csrc.nist.gov 2026-08-06.
  This is the nearest hard date in the whole table and it was missing from it. It does not bind this
  repository - nothing here is FIPS-validated and nothing claims to be - but it binds anyone
  procuring a module to run this kind of workload, so it belongs in a migration roadmap.
- **2027** CNSA 2.0 new-procurement mandate -> be compliant before bids.
- **2029** - Cloudflare commits its product suite to full PQ security *including PQ authentication*; Google sets 2029 as its PQC **migration** timeline (the weaker of the two commitments) -> supply-chain expectations.
- **2030** deprecation of **112-bit-strength** RSA/ECC and **2035** disallowance of all quantum-vulnerable public-key algorithms, per **NIST IR 8547 ipd** (Nov 2024 - *Initial Public Draft*, not final, not binding). RSA-3072 / P-256 and above carry no 2030 milestone; plan field validation by 2027 Q1.
- **2030-12-31 / 2031-12-31** - **EO 14412**, *Securing the Nation Against Advanced Cryptographic Attacks*
  (signed 2026-06-22, 91 FR 38483): PQC key establishment on high-value assets by 2030-12-31 and digital
  signatures by 2031-12-31, with a FAR rule putting contractors on the 2030 date. **National Security
  Systems are excluded** and stay on CNSA 2.0. This is the binding US instrument; IR 8547 is still a draft.
- **2035** - Japan's Cabinet Secretariat interim report (Nov 2025) sets PQC migration at **2035**, with the
  detailed roadmap to be drafted during FY2026.

## Algorithm backups worth tracking

- **HQC** was selected in **March 2025** as the code-based KEM backup to ML-KEM, with standardisation
  expected 2026-2027. It matters to this project specifically because the whole architecture is a
  crypto-agility registry: a second KEM family resting on a different hardness assumption is exactly
  what that registry exists to make swappable. It is **not implemented** - liboqs carries HQC, this
  repository's `CryptoRegistry` does not register it - and that is a deliberate gap rather than an
  oversight, because a backup KEM is only worth wiring once its parameter sets are final.

## Standards added since the last revision (verified 2026-08-03)

| Document | Status | Why it matters here |
|---|---|---|
| **NIST SP 800-90C** | **Final, 2025-09-25** | RBG constructions. A QRNG targeting full-entropy output is an RBG3 construction; cite alongside SP 800-90B in the QRNG section. |
| **NIST IR 8214C** - First Call for Multi-Party Threshold Schemes | **Final, Jan 2026** | Scope explicitly includes **FHE, MPC and ZKPs**. The most relevant NIST process to this project's WS3/WS4, and it was not cited at all. |
| **NIST CSWP 39upd1** - Considerations for Achieving Crypto Agility | Final, 2026-06-29 | Supersedes the 2025-12 original; this is now *the* NIST crypto-agility reference and should be cited in `crypto-agility.md`. |
| **ETSI GS QKD 020** - REST-based Interoperable KMS API | V1.1.1, 2026-06-29 | First cross-vendor KMS-to-KMS spec. `pqc/transport.py` targets GS QKD **014** only; 014 + 020 is the current pair. |
| **ETSI GS QKD 015 V2.1.1** - SDN control interface | Published 2022-04 | **Implemented in Round 18** (`qkd/sdn_controller.py`, mirrored in `web/src/lib/clientSdnController.ts`). Supplies the control plane the demo previously lacked, and its clause 1 scope statement is what settles that an MCP agent is an SAE rather than a controller. |
| **ETSI GS QKD 014 V1.1.1** - key delivery API | Published 2019-02, **still the only published edition** | **Implemented in Round 18** (`qkd/kme.py`). An ed2 stable draft (v1.3.1) existed as of 2025-06 but is not published, so "QKD 014 compliant" in this repository means the 2019 document. The mandated mutual TLS is **not** implemented - see `threat-model.md`. |
| **BSI TR-02102-1 v2026-01** | 2026-01-23 | The quotable basis for both "QKD never standalone" and the hybrid key-agreement recommendation - and it is fetchable, unlike NSA's pages. |
| **IACR ePrint 2026/935** - SoK: Private LLM Inference using Approximate HE | rev. 2026-06-03 | The citable ceiling: ~20 CKKS frameworks surveyed, largest demonstrated end-to-end model **8B parameters**, ~4 orders of magnitude slower than plaintext. Confirms nothing larger exists. |
| **IACR ePrint 2026/1231** - Hybrid AKE for QKD networks without signatures | 2026-06-10 | The closest published construction to `hybrid_combiner_mode="its_toeplitz"`: KEM-only hybrid AKE with a conditional information-theoretic guarantee when the QKD leg is uncompromised. |
| **arXiv:2601.18511v2** - Scaling up FHE-based PPML | v2 2026-06-17 | Best published absolute: **Llama-3-8B under CKKS at 20 s prefill / 18 s per output token on 8x RTX PRO 6000**. Quote this rather than hand-waving FHE cost. |

## EU AI Act Art. 50(2): partially met, and what the recognised route actually is

Verified 2026-08-09. **Deadline 2 December 2026**, which is the nearest unmet compliance date this
project has.

Art. 50(1) - telling the person they are interacting with an AI - has been implemented since
Round 21 and is applicable. **Art. 50(2) is a different obligation and is NOT implemented.** It
requires providers of AI systems generating synthetic audio, image, video or **text** to mark the
outputs in a **machine-readable** format, detectable as artificially generated. The Agentic AI panel
generates text with an in-browser LLM, so it is in scope.

Why there is still time, and exactly how much: the AI Omnibus grants generative systems already on
the market before 2 August 2026 until **2 December 2026** to meet the marking requirement. This demo
was live before that date, so the grace period applies - but it expires, and nothing in the codebase
implements marking today. A grep for `C2PA`, `watermark` or `machine-readable` over the web sources
returns nothing relevant.

What would satisfy it is a design question, not a code question yet: the plausible options are C2PA
Content Credentials on the generated text, or a documented in-band marker. Both need a decision
before implementation, and the decision should be recorded here the way the WebGL and ZeroGPU
declines were, rather than the item quietly rolling forward.

Penalty exposure for Art. 50 non-compliance is up to EUR 15 million or 3 percent of worldwide annual
turnover, whichever is higher - worth stating because it is the reason this is a dated item and not
a nice-to-have.

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
Preprocessor computes **per-transaction subgraph patterns over a graph updated as each transaction
arrives**, while this pipeline computes **per-account scalar aggregates**. That difference is
implementable without `snapml` and is a bounded, well-specified workstream. A measured intermediate
step already exists: causal prior-at-time aggregation beat the shipped approach by 1.45x AUPRC at
matched feature count, with no leakage.

### Round 24 update, researched 2026-08-09

**The recognised compliance route is now defined, and it is not "write a watermark".** The Commission
published its Article 50 guidelines on **20 July 2026**, alongside a **Code of Practice on marking and
labelling of AI-generated content** - final **10 June 2026**, assessed as adequate by the Commission
and the AI Board on **8-9 July 2026**. Adherence to that Code is the recognised way to demonstrate
compliance with Art. 50(2), (4) and (5). The requirement itself is that marks be "effective, reliable,
robust and interoperable" and machine-readable.

**What now ships**: the Agentic AI panel's downloadable artifact carries an `ai_generated` block -
`is_ai_generated`, the generating engine and model, the runtime, and an explicit scope and
compliance-status field. That is machine-readable and it travels with the artifact that leaves the
page.

**What it deliberately does not claim.** It marks the artifact, not the on-screen text. Token-level
schemes such as SynthID-Text mark the text itself by biasing the sampling loop, and the in-browser
Transformers.js path does not expose that loop, so implementing one here would mean forking the
inference runtime. And this project has **not** been assessed against the Code of Practice, so
`compliance_status` says `partial` rather than asserting compliance. Claiming more than was done is
the failure mode this repository has spent several rounds removing.

**C2PA is not the answer for text, at least not yet.** C2PA 2.2 (2025-05-01) is built around image,
video and audio assets; its text story is not established, and the searches that suggest otherwise
lead to vendor material rather than the specification.

Remaining decision, before **2 December 2026**: whether to seek assessment against the Code of
Practice, or to state publicly that this is a research demonstrator marking its artifacts on a
best-effort basis. Either is defensible; silence is not.
