# Use cases, workstreams and positioning

This is a **research / proof-of-concept**. Real implementations where tractable, honest
simulation/scaffolds where not - claims are cited to primary sources in
[`references/`](../references/REFERENCES.md) where one exists, and the corrections each fact-check made are
recorded in [`docs/deliverable-verification.md`](deliverable-verification.md) and
[`docs/verification-log.md`](verification-log.md).

> **What is new here, and what is not.** This project's verification rule is that a novelty claim
> must survive an adversarial search for prior art. Applied to its own headline claim the answer is
> asymmetric, so both halves are stated.
>
> **MPC in a browser is not new.** webSPDZ (IACR ePrint 2025/487) compiles MP-SPDZ to WebAssembly
> over WebRTC or WebSockets, and names JIFF and MPyC-Web as earlier browser MPC engines. The
> garbled-circuit half here re-implements a known idea.
>
> **Homomorphic evaluation in a browser is not new either.** node-seal, the SEAL build this demo
> runs, benchmarks homomorphic evaluation operations in Chrome, Firefox and Safari, and Attrapadung
> et al. (AsiaCCS 2018) shipped two-level homomorphic evaluation compiled to WebAssembly. Most
> deployed designs still encrypt client-side and evaluate server-side: IACR ePrint 2026/515 defines
> "client-side operations" as encryption and decryption while *"users offload large computations to
> servers"*, and Zama's TFHE-rs WASM API documents that it supports key generation, encryption and
> decryption but **not FHE computation**, citing WASM threading and the 2 GB memory ceiling. The
> narrower thing the search did not find is a peer-reviewed 2025-26 report of CKKS neural-network
> inference evaluated in a browser tab, and that is an absence of evidence rather than a proof.
> This demo evaluates a real CKKS dot product and a real CKKS convolution in the tab.
>
> **So the claim is the composition, not either half:** PQC key establishment to a simulated QKD link
> (it yields a key length, not key bytes) to secure multiparty computation to homomorphic evaluation to
> model inference, chained end to end in one browser tab with zero server compute. A search for an integrated browser system doing
> all of that returned nothing. Two caveats travel with it: the two MPC parties are notional - one
> tab, an in-memory transport, real protocol and bytes and wall-clock but no genuine party separation
> - and the quantum channel is simulated, because no browser API for QKD exists. And client-side is
> not automatically private: WWW 2026's WAMO paper argues browser deployment moves a model into a
> hostile environment, which is one reason the models here are public rather than treated as secrets.

> **Positioning - quantum-safe, not quantum-advantage.** The value here is **quantum-safe / PQC**, and
> what drives it is regulation (NIST FIPS 203/204/205; for national security systems only, CNSSP 15
> requires CNSA 2.0 algorithms in new acquisitions from 2027 and in all use by 2031; US EO 14412 moves
> federal high-value assets and high impact systems to PQC key establishment by 2030 and signatures by
> 2031; EU roadmap 2026/2030/2035; "harvest-now-decrypt-later"). A general **quantum *advantage*** for these workloads is **not** claimed
> as a market benefit - it is broadly unproven in 2026 (e.g. classical tensor-network simulation matching
> annealing results), and the numeric "quantum advantage" score the benchmark used to print is
> **retired from every page of the demo** (2026-10-23): its inputs measured different things on the two
> sides, and only the full deployment's `/api/benchmark` response still carries it. For the QCNN specifically the evidence is sharper than "unproven": Bermejo, Braccia,
> Rudolph, Holmes, Cincio and Cerezo show that randomly initialised QCNNs **can only act on
> low-bodyness measurements of their input and are efficiently classically simulable** with Pauli
> shadows, and argue that QCNN successes largely reflect benchmark simplicity
> (PRX Quantum 7, 020304, 2026; [arXiv:2408.12739](https://arxiv.org/abs/2408.12739)). The trained
> QCNNs here are simulable for a more direct reason: their blocks act on disjoint pairs of wires over
> a product angle encoding, so each is, up to its small final readout block, a tree of two-qubit
> blocks that a classical computer evaluates exactly, the regime of that paper's Section IV.B, "Classical datasets". Nothing on the QCNN path is encrypted, so
> it is not homomorphic evaluation either. On the same split, logistic regression beats every trained
> QCNN, and both figures are printed side by side on the page. Two circuit widths ship for telecom so
> the effect of the register size can be read rather than assumed: the eighth wire encodes no
> feature, and across five seeds seven wires beat eight on four (mean macro F1 0.5309 against 0.4846,
> paired p = 0.059), so the padding wire buys nothing measurable and may cost something
> ([`references/qcnn-width-ablation.md`](../references/qcnn-width-ablation.md)). Relatedly, Li, Wu, Li and Yung separate observable concentration from loss of
> parameter sensitivity and show that **avoiding concentration is necessary but not sufficient for
> trainability** ([arXiv:2603.18479](https://arxiv.org/abs/2603.18479), Mar 2026, rev. Jun 2026); the
> untrained circuit this project removed from its inference path failed exactly the concentration
> symptom, with |<Z0>| under 0.21 for every input tested.
>
> The production-relevant output is the **security posture table** - per-row facts, each
> traceable to FIPS 203/204 or to the run that produced it. The 0-100 composite that used to hold that
> role was **retired in Round 22**: 14 of its 17 weights carried no stated rationale and it scored a
> run with zero secure key bits 100/100 (see
> [`references/security-scoring-fact-check.md`](../references/security-scoring-fact-check.md)). The whole stack is deliberately a
> **hybrid** (PQC-classical + QKD/QHE quantum-simulated + FHE/MPC-classical + AI) - the pragmatic 2026
> pioneering-product shape.

---

## What it does

Four workstreams (WS), unified by one pipeline and a crypto-agility registry:

| WS | Theme | What's implemented |
|----|-------|--------------------|
| **WS1** | **AI-QKD** parameter optimization | BB84 **two-intensity decoy-state** simulation (Ma et al.'s own one-decoy bound, section 3.5 Eq. 41, which leaves the vacuum yield $`Y_0`$ to Eve), over **Rusca et al. 2018's stated detector set** (SNSPDs at efficiency 0.5, 1e-8 dark counts per gate, a 1% error floor, 1 GHz with a 100 ns dead time charged by their Eq. B3; a simulation set, not a measured link) (+ a selectable **E91/CHSH** protocol), QBER/SKR telemetry, eavesdropper detection (intercept-resend's 0.25 f disturbance composed independently with the channel error, a + b - 2ab, in both languages); a **published-reference decoy model** (`qkd/decoy_reference.py`) whose 13 tests assert Ma et al. PRA 72, 012326 (2005) values rather than this project's own - decoy reaches past 140 km where non-decoy dies near 40, and at the customary $`\mu = 0.1`$ a non-decoy link's tagged fraction exceeds 1 at **zero** distance, so it yields no key even back-to-back; a **QRNG** entropy source (state-vector quantum-measurement sim + SP 800-90B section 4.4 continuous health tests, an **SP 800-90B min-entropy estimate**, and a **Leftover-Hash-Lemma / seedless** Toeplitz extractor) that can seed the BB84/E91 basis choices in Python (opt-in, off by default; in the browser its bits feed only the health metrics); a config-gated PQC+QKD key combiner (HKDF, or a Toeplitz+QLHL extractor seeded with SHAKE-256 of its own inputs, so computational rather than information-theoretic); and **OptiQKD** - a measured AI optimizer (scikit-learn surrogate + gradient-free search) tuning decoy (`mu_signal` / `mu_decoy`) + basis params to raise the secret-key rate vs default (with a convergence metric); the reference framework reports +14-25 % in simulation |
| **WS2** | **QML-IDS** intrusion detection | 8 classes = **7 QKD attack types + normal traffic**; RandomForest baseline + PennyLane variational quantum classifier and a quantum-kernel SVC, both offline (the published `telecom_ids.qml` ONNX is the random forest, and the SVC's fidelity kernel equals the classical product kernel). Scored on macro F1, not attack-vs-normal F1 |
| **WS3** | **QSMPC x QKD** | Shamir + d-level QFT secure summation (faithful to the paper example), quantum oblivious transfer, and a keyed-hash PSI over the PQC channel (not private between the two banks over a small ID space; a production PSI would use an OPRF-based protocol) |
| **WS4** | **QHE x AI** | Quantum one-time pad (perfectly hiding as a scheme; the demo's pad is seeded); a trained variational QCNN that predicts in the clear (no T gate, no mask), beside an untrained QOTP + Clifford+T comparison circuit in the browser that counts T gates and predicts nothing (the Python module has no T gate); Rate-1 quantum **transport** (no `Eval`, so not QFHE); classical hybrid = CKKS **linear** layers (TenSEAL CPU / OpenFHE CPU, commercial-OK; HEonGPU is a build-gated GPU reference adapter that does not run here) + **MPyC** secret-sharing **non-linear** layers (ReLU/argmax) in the full deployment, a Polytune garbled-circuit 2PC between notional parties in the browser - incl. a **genuine FHE convolution** (im2col -> CKKS `dot_plain`), so the medical use case runs a real small CNN on **real MedMNIST images** with its convolution and dense layers under CKKS (the browser decrypts between layers and applies ReLU and pooling in plaintext) |

**Crypto-agility is the headline library strategy** - in the EU NIS Cooperation Group sense
([PQC Roadmap](https://digital-strategy.ec.europa.eu/en/library/coordinated-implementation-roadmap-transition-post-quantum-cryptography),
23 Jun 2025, section 3.2): *modular design that enables replacing the cryptographic components*,
explicitly **not** cipher-suite negotiation during protocol execution. Every algorithm
(KEM / QKD / FHE / **MPC** / signature) is resolved from a registry and selected by the
operator (config / WebUI) **before a run**, so a future PQC migration is a configuration
change, not a rewrite. There is **no in-band negotiation and no silent fallback**
(see [`docs/crypto-agility.md`](crypto-agility.md)).

Cross-cutting: **PQC-first hybrid** (ML-KEM-768 + X25519, ML-DSA-65 - the hybrid-first
approach the EU roadmap recommends), **crypto-agility**
(operator-selected, modular provider replacement), **federated learning with pairwise masking** (the
masks cancel exactly, but they hide nothing from a session-key holder in this demo) - including a **Quantum
Federated Learning** option (a PennyLane hybrid quantum-classical VQC over this repo's own FedAvg; in the
browser it runs on every finance and medical run rather than behind a toggle, over the page's own
bundled sample, as a logistic regression over a closed-form feature map with no circuit behind it), **explainable AI**
(exact linear-SHAP attributions, ML-DSA-signed for a tamper-evident audit record - aimed at EU AI Act
Art. 13, which binds high-risk systems only: from 2 Dec 2027 for Annex III and 2 Aug 2028 for Annex I
products after the AI Omnibus, and each use-case page states which, if either, would apply to it), an **E2E orchestration** engine with **Run / Pause / Resume / Step / Stop / Restart**
controls, live WebSocket event streaming, config-driven parameters editable in the WebUI (a per-page
**Parameters** panel - every QKD / QRNG / OptiQKD / IDS / data knob defaults from `config/defaults.py`
and is editable per run, with reset-to-defaults; no hardcoded values), and
high-DPI PNG / animated-GIF export of the run. An **Agentic AI, MCP** capability runs a small language
model **entirely in the browser** (Transformers.js on WASM by default and WebGPU on request, or WebLLM
on WebGPU; Apache-2.0) whose
request/response is sealed + signed over a **client-side quantum-safe MCP** channel - agentic AI with a
quantum-safe layer, zero server compute. It is **integrated into every use-case page** as a per-industry
panel (its own MCP tool - `score_transaction` / `query_cohort` / `pool_allele_counts` /
`inspect_qkd_telemetry` / `relay_mission_telemetry` - fed by that run's live metrics), so MCP is the
per-use-case **cross-cutting integration layer** (no standalone page - it lives inside each industry use
case). The **full
deployment** additionally runs a **real MCP server** (MCP Python SDK, `mcp>=1.2`: `MCPServer` on 2.x,
`FastMCP` on 1.x) exposing those 5 tools, bound to the
quantum-safe transport so a `tools/call` returns a **signed real result** (plus interop with the
official filesystem MCP server); the public demo keeps the zero-server client mirror.

The **public demo computes real encrypted inference in the browser** on the finance, medical and
genomics pages - node-seal **CKKS** (a genuine homomorphic linear layer) + a real **Polytune
garbled-circuit 2PC** for the non-linear argmax; telecom runs a CKKS check of its student and the
satellite page runs neither - so those FHE/MPC figures are **measured client-side, zero VPS compute**.
The model and the data are public, and the PSI and secure-aggregation steps keep nothing private
between the parties in one tab, so this demonstrates the computation rather than a privacy property.

### Industry use cases

| Use case | Pipeline | Industry |
|----------|----------|----------|
| `finance_aml` | PQC -> QKD -> **PSI** (cross-bank blacklists) -> FHE fraud inference | finance / AML |
| `medical_fl` | PQC -> QKD -> **federated training** + pairwise-masked aggregation -> private diagnosis | healthcare |
| `genomics` | PQC -> QKD -> **QSMS secure summation** (pooled allele counts) -> FHE | genomics |
| `telecom_ids` | PQC -> simulated QKD (optional eavesdropper) -> **QML-IDS** attack detection | telecom / defence |
| `satellite_comms` | PQC -> simulated **free-space QKD** (BB84 downlink) -> **QSDC** test with an inline eavesdrop check (it cannot close at the default link budget, and the run says why) | satellite |

**Maturity, stated in the same breath as the novelty.** CISA's Space Systems Critical Infrastructure
Working Group (*Space Systems Security and Resilience Landscape: Zero Trust in the Space
Environment*, April 2024) assessed both halves of that satellite row. Homomorphic encryption schemes
"have yet to be applied to space systems, and their technology readiness level remains low"; QKD
"has been tested in the space environment, but its technology readiness level remains low ...
quantum systems operate on specialized equipment that is costly". This is carried here rather than
somewhere quieter because it does two things at once: it independently corroborates the gap the
prior-art search found, and it sets the register - a research demonstrator, not something an operator
could fly. CISA attributes the QKD caution to NSA's published position; nsa.gov was not
machine-retrievable during verification, so cite CISA for the assessment and not NSA. Details and the
five motivations to avoid: [`references/satellite-motivation-fact-check.md`](../references/satellite-motivation-fact-check.md).

Each use case has its **own page** (`/usecase/:id`) - a **live architecture simulator** you Start /
Pause / Step / Stop / Restart. A **single Run** drives both the animated E2E protocol **and** the
benchmark: the benchmark is **assembled from that very run** (no separate "Run benchmark" button) -
the MPC-FHE row is built from the run's **measured FHE-linear + MPC-non-linear cost** (scaled to a
cost-model architecture on genomics, which has no encrypted student, and labelled so), shown next to
the QSMPC-FHE comparison, security posture, and (on all five use-case pages) a signed
SHAP explanation. The architecture canvas makes the **MPC-FHE secret-sharing over a LAN <-> WAN <-> LAN** link
explicit - linear layers cross as a **CKKS ciphertext**, non-linear layers as **secret-shares**. The
**E2E Orchestration** page draws the same canvas for any use case, and for its generic
demo between Party A and Party B, in an Architecture tab beside the stage-order graph. In
the **public demo** (`?demo=client`) every use case runs **entirely in the browser**, with **real,
measured FHE/MPC** (node-seal CKKS + a Polytune garbled-circuit 2PC) on the pages whose flow has a
private-inference stage, and an in-browser federated stage on every finance and medical run (the
"Quantum Federated Learning" label notwithstanding, nothing quantum is computed in it) - zero `/api`
compute; the **full local** deployment adds the server-side PennyLane QFL + the real MCP server. See [`docs/architecture.md`](architecture.md)
for the full diagram.
