# Annotated References

The `references/` folder holds the primary-source PDFs that ground this PoC. Each entry maps the
paper to the **workstream (WS)** it informs, the **module** that implements or uses it, and any
open-source code reused.

Every entry carries a `URL` and an `Added` line. `URL` is the canonical landing page, not a mirror.
`Added` is the date the PDF entered this repository, taken from `git log --diff-filter=A`, so it is
checkable rather than asserted; entries with no bundled PDF give the date the claim was last
verified against the source instead.

Markers:

- `[R]` the PDF is committed to this folder and you get it when you clone.
- `[R-local]` the PDF sits in this folder on the author's machine but is **gitignored**, so a clone
  does not contain it. Two are ~17 MB each and one is a copyrighted book. Fetch from the URL.
- `[link]` never bundled. Fetch from the URL.

An earlier revision marked the `[R-local]` entries `[R]`, which told a reader they had files that a
clone never contained.

Workstreams: **WS1** AI-QKD parameter optimisation and entropy, **WS2** QML-IDS, **WS3** QSMPC x QKD,
**WS4** QHE x AI. Cross-cutting: PQC-first hybrid, crypto-agility, XAI.

---

## WS1 - AI-QKD (real-time parameter optimization) + entropy (QRNG)

- **AML-QKD: Adaptive Machine Learning Framework for Real-time Parameter Tuning in QKD** - Mohamed,
  Kaldari and Al-Kuwari, Qatar Center for Quantum Computing (HBKU), **arXiv:2603.04192** [quant-ph]
  `[R-local]`
  URL: https://arxiv.org/abs/2603.04192
  Added: not committed (17 MB, `.gitignore:14`); claim re-verified 2026-08-04
  **v1** (4 Mar 2026) was titled *OptiQKD: A Machine Learning-Optimized Framework for Real-Time
  Parameter Tuning in Quantum Key Distribution*; **v2** (1 Jun 2026) retitled it AML-QKD.
  This is one paper in two versions, not two papers. An earlier revision of this file listed it
  twice and described v2 as "extending" v1; v2 *supersedes* v1 and revises the headline number
  **downward**: v1 claimed median SKR +20-30 percent, v2 claims +14-25 percent (QBER 3.0 to 1.5
  percent in both). The QRL/E91 result (+29.2 percent throughput) is labelled *exploratory* by the
  authors. Simulation only.
  Grounds `src/qsh/qkd/optimize.py`, a deterministic offline-safe sklearn-surrogate plus
  coordinate-ascent analogue over the two-intensity decoy-state BB84 model, with the `mu_decoy`
  single-photon bound (Ma, Qi, Zhao and Lo 2005) and a convergence metric, and the E91/CHSH model
  `src/qsh/qkd/e91.py`. Honest scope: a simulation, not deployed QKD, and gradient-free (no RL, no
  torch), so it is an analogue of the paper's theme rather than a reproduction of its method.
  Both the v1 and v2 PDFs are kept locally for provenance (`OptiQKD_*.pdf`, `AML-QKD_*.pdf`); cite v2.

- **Information-Theoretic Solutions for Seedless QRNG Bootstrapping and Hybrid PQC-QKD Key
  Combination** - Giestinhas and Spiller, University of York, arXiv:2603.26907v2 [quant-ph], 2026
  `[R-local]`
  URL: https://arxiv.org/abs/2603.26907
  Added: not committed (`.gitignore:15`); claim re-verified 2026-08-04
  Licence CC BY 4.0.
  Quantum Leftover Hash Lemma strong (Toeplitz, universal-2) extractors: (1) seedless QRNG
  bootstrapping from two independent entropy sources; (2) a QLHL-extractor **alternative** to XOR
  key-combining. It is **not** "secure-if-either" and not "better than XOR" - the paper never uses
  that phrase and explicitly says strong seeded extractors "present different characteristics",
  noting its own cost as "longer keys and a worse computational complexity". The property it does
  claim: if the combined output **and one input key** are compromised, the remaining key material
  retains quantifiable min-entropy. PQC keys are modelled as HILL (computational) min-entropy.
  Grounds the Round 16 QRNG and transport upgrade: `src/qsh/qrng/simulator.py` (LHL-bounded Toeplitz
  output plus seedless two-source extractor), `src/qsh/qrng/health.py` (SP 800-90B min-entropy
  estimate), and the config-gated ITS combiner in `src/qsh/pqc/transport.py` (`_mix_qkd_its`,
  `hybrid_combiner_mode="its_toeplitz"`).

---

## WS2 - QML-IDS (quantum machine learning for intrusion detection)

This section was promised by the preamble of every earlier revision of this file and never written,
while `src/qsh/qml_ids/` and the Telecom/Defense use case shipped. The gap mattered: WS2 is the one
workstream whose honest answer is negative, and the reasoning lived only in a training script's
docstring.

- **A Quantum Machine Learning Approach for Intrusion Detection Systems** - Al-Kuwari et al.,
  arXiv:2509.14282 [cs.CR], 16 Sep 2025 `[link]`
  URL: https://arxiv.org/abs/2509.14282
  Added: no PDF bundled; claim re-verified 2026-08-04
  Supplies the **class structure**: 8 classes = 7 attack types plus normal. That taxonomy is what
  `src/qsh/qml_ids/attacks.py` generates and what `scripts/train/telecom_ids.py` trains against.
  No number from this paper is reproduced or claimed here.
  It also **refutes a claim this project used to make**. An earlier revision of the training script
  and of this section asserted that every published QML-IDS number sits below a random forest on the
  same data. Table VIII of this paper reports a hybrid QLSTM at 94.7 percent accuracy and 94.7
  percent F1 against the authors' own random-forest baseline on their generated data. The
  defensible statement is narrower: published quantum results here are usually obtained on heavily
  subsampled and dimension-reduced inputs, which makes them hard to compare against a forest given
  the full feature set - not that quantum always loses.

- **Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study** - Engelen, Rimmer
  and Joosen, IEEE Security and Privacy Workshops (WTMC), 2021 `[link]`
  URL: https://doi.org/10.1109/SPW53761.2021.00009
  Added: no PDF bundled; claim re-verified 2026-08-04
  Why this project reports per-class F1 first and aggregate accuracy second. Engelen et al. showed
  aggregate random-forest accuracy on CIC-IDS2017 sits at 0.99 both before and after correcting the
  labels, while Bot F1 moves 0.60 to 0.99: the aggregate is blind to the fix. `telecom_ids.py`
  reports both for exactly this reason.

- **Withdrawn: a hybrid quantum-classical autoencoder IDS** - arXiv:2512.05069 `[link]`
  URL: https://arxiv.org/abs/2512.05069
  Added: no PDF bundled; withdrawal confirmed 2026-08-04
  Withdrawn **by its own authors** (v2, 2026-04-02) because the evaluation was "insufficient to
  fully support the paper's conclusions". Cited as evidence about the state of the field, not as a
  method. It is listed here so that a reader who finds v1 elsewhere learns it was retracted.

**What this repository measured**, and it is a negative result kept deliberately: on the synthetic
2400-sample, 8-class telemetry, RandomForest reaches macro F1 0.9866 while the
quantum-kernel SVC reaches 0.8316 and the variational QNN 0.4667. That comparison is not
like-for-like and the registry says so: both quantum models are **binary**, so on their own task they
score 0.9453 and 0.9333, and their macro F1 is depressed by the six classes they never
attempt. Even granting the binary figures, neither earns its qubits against a random forest that
takes a fraction of the 267 s the quantum pair needs. `quantum_beats_classical` is recorded as
`False` in `models/metrics.json`, which is the field this project would have to falsify to
claim otherwise.
The data is synthetic, and the reason was overstated in earlier revisions. The accurate version,
re-checked 2026-08-05:

- **UNSW-NB15** requires the authors' agreement for commercial use.
- **CIC-IDS2017 does not "state no licence at all".** The Canadian Institute for Cybersecurity's
  institute-level FAQ says *you may redistribute, republish, and mirror our datasets in any form*
  with citation, and places that under commercial use - while the CICIDS2017 dataset page itself
  frames the data as *publicly available for researchers* and names no licence. The two pages
  disagree. That is an unresolved ambiguity, not a permission, and it is the same class of question
  as the ImageNet one that Round 19 resolved by dropping the dependency rather than guessing.
- **The Engelen-corrected redistribution** is hosted separately and states nothing.
- **5G-NIDD was evaluated and rejected.** The adjudication lives in
  [`DATA_LICENSES.md`](../DATA_LICENSES.md) - it was written out in full in both files, and a licence
  finding with two homes is a licence finding that will be corrected in one of them.

Licence adjudications belong in [`DATA_LICENSES.md`](../DATA_LICENSES.md); this file cites the
papers. Where the two used to overlap, that file is the owner.

---

## WS3 - QSMPC x QKD (quantum-secure multi-party computation)

- **Quantum Secure Multiparty Computation based on Secure Summation and QKD** - Mandeep Kumar and
  Bhaskar Mondal (NIT Patna), Research Square, 3 Jul 2025 `[R]`
  URL: https://doi.org/10.21203/rs.3.rs-6706796/v1
  Added: 2026-06-27
  Licence **CC BY 4.0**, which requires attribution; an earlier revision of this entry named no
  authors.
  File: `Quantum_Secure_Multiparty_Computation_based_on_Secure_Summation_and_QKD.pdf`
  Core protocol for `src/qsh/qsmpc/secure_summation.py`: QKD-shared keys plus secure summation.

- **Efficient Simulation of Quantum Secure Multiparty Computation** - Sutradhar, arXiv:2501.10083,
  2025 `[R]`
  URL: https://arxiv.org/abs/2501.10083
  Added: 2026-06-27
  File: `Efficient_Simulation_of_Quantum_Secure_Multiparty_Computation.pdf`
  (t,n)-threshold QSMS = Shamir secret sharing plus QFT plus Pauli ops. Directly reimplemented in
  `secure_summation.py` and `shamir.py` (Qiskit-native, no hardware). Validated on the paper's
  worked example (t=3, n=7, d=11, X=2, Y=3, giving S=5).

- **Experimental secure multiparty computation from quantum oblivious transfer with bit commitment**
- Zhang et al., npj Quantum Information, 2026 `[R]`
  URL: https://www.nature.com/articles/s41534-026-01219-w (npj QI 12, 76, 2026)
  Preprint: https://arxiv.org/abs/2411.04558
  Added: 2026-06-27
  File: `Experimental_secure_multiparty_computation_from_quantum_oblivious_transfer_with_bit_commitment.pdf`
  Decoy-state BB84 plus SHA-256 commitment QOT, giving private set intersection on bank-blacklist
  data. Implemented in simulation in `src/qsh/qsmpc/qot.py` and the Finance AML/PSI use case.

- **Quantum-secure multiparty deep learning** - Sulimany, Vadlamani, Hamerly, Iyengar and
  Englund, Physical Review X 15, 041056, published 24 Dec 2025 `[R]`
  URL: https://doi.org/10.1103/k8wg-qmbh (APS moved to opaque DOIs in 2025, so the older
  `10.1103/PhysRevX.15.041056` pattern does not resolve)
  Preprint: https://arxiv.org/abs/2408.05629
  Added: 2026-06-27
  File: `Quantum-Secure_Multiparty_Deep_Learning.pdf`
  Photonic coherent linear-algebra inference; Holevo and Cramer-Rao leakage bounds (0.1 bit per
  weight, MNIST above 95 percent). The leakage-bound analysis is reimplemented in `src/qsh/qsmpc/`
  as security metrics.

- **Quantum-Safe Multi-Party Computation for Distributed AI Datasets** - Gopher Security blog, 2026
  `[R]`
  URL: https://gopher.security/blog
  Added: 2026-06-27
  File: `Quantum-Safe_Multi-Party_Computation_for_Distributed_AI_Datasets....pdf`
  A vendor blog post, not peer-reviewed, and weighted accordingly. qOLE and qMPSI plus PQC (ML-KEM)
  hybrid transport plus an MCP delegate-and-verify narrative. Informs `src/qsh/pqc/transport.py` and
  the end-to-end orchestration framing.

---

## WS4 - QHE x AI (quantum/classical homomorphic encryption + AI)

- **Reverse Delegated Training and Private Inference via Perfectly-Secure Quantum Homomorphic
  Encryption** - Ortega and Martin-Delgado, arXiv:2602.12712, 2026 `[R]`
  URL: https://arxiv.org/abs/2602.12712
  Added: 2026-06-27
  File: `Reverse_Delegated_Training_and_Private_Inference_via_Perfectly-Secure_Quantum_Homomorphic_Encryption.pdf`
  QOTP plus classical FHE hybrid, Clifford+T QCNN, Pauli-gate concealment for circuit privacy,
  MNIST 96.7 percent. **Relationship to this repository:** `src/qsh/qhe/qcnn.py` is an *independent*
  implementation of the paper's construction. It does not import, vendor or wrap the authors'
  reference code at `OrtegaSA/CQC-QHE-repo`; an earlier version of this line claimed code reuse that
  never existed, and the submodule was removed in Round 17. Cited as the source of the construction,
  which is the accurate relationship.

- **Rate-1 Quantum Fully Homomorphic Encryption** - Chardouvelis, Dottling and Malavolta, IACR
  ePrint 2020/1454 (TCC 2021) `[R]`
  URL: https://eprint.iacr.org/2020/1454
  Added: 2026-06-27
  File: `Rate-1_Quantum_Fully_Homomorphic_Encryption.pdf`
  Ciphertext-to-plaintext ratio near 1, malicious circuit privacy under LWE. Informs
  `src/qsh/qhe/rate1_qfhe.py` (QOTP plus rate-1 classical FHE keys) and the benchmark's
  communication-efficiency metric.

- **Leveraging ASIC AI Chips for Homomorphic Encryption (CROSS)** - Tong et al., IACR ePrint
  2026/160, HPCA 2026 `[R]`
  URL: https://eprint.iacr.org/2026/160
  Added: 2026-06-27
  File: `Leveraging_ASIC_AI_Chips_for_Homomorphic_Encryption.pdf`
  Maps FHE NTT and modular ops onto AI accelerators (BAT/MAT). Reference for the GPU benchmark's
  NTT and throughput discussion. Code: `EfficientHE/CROSS`, TPU-oriented, reference only.

- **EncFormer: Secure and Efficient Transformer Inference over Encrypted Data** - Zhu, Jin, Aung and
  Xiao (NUS and A*STAR), arXiv:2604.09975 [cs.CR], 11 Apr 2026 `[R]`
  URL: https://arxiv.org/abs/2604.09975
  Added: 2026-06-27
  File: `EncFormer_Secure_and_Efficient_Transformer_Inference_over_Encrypted_Data.pdf`
  **Manuscript submitted to IEEE TDSC: not accepted, not peer-reviewed.** Hybrid CKKS plus
  two-party computation, not FHE-only, and the text contains no TFHE, no quantization and no
  polynomial-softmax method; an earlier revision of this entry attributed all three to it and dated
  it 2024. Reported end-to-end BERT-base latency: 2.1 minutes LAN, 8.9 minutes WAN3; the authors put
  their secure pipeline at "roughly 3x10^4 slower" than the 4.5 ms plaintext baseline. Informs the
  Transformer workload in `src/qsh/benchmark/` as a cost reference, not as a performance target.

---

## QFL - quantum federated and delegated learning (medical and finance use cases)

- **Quantum Delegated and Federated Learning via Quantum Homomorphic Encryption** - Li and Deng,
  arXiv:2409.19359, 2024 `[R]`
  URL: https://arxiv.org/abs/2409.19359
  Added: 2026-06-27
  File: `Quantum_Delegated_and_Federated_Learning_via_Quantum_Homomorphic_Encryption.pdf`
  QOTP-encrypted delegated and federated learning; informs the `src/qsh/ai/federated.py` privacy
  layer.

- **Quantum Federated Learning: A Comprehensive Survey** - Nguyen, Uddin, Shaon, Rahman, Dobre and
  Niyato, arXiv:2508.15998, 21 Aug 2025 `[R]`
  URL: https://arxiv.org/abs/2508.15998
  Added: 2026-06-27
  **Preprint, under revision at IEEE COMST; not published.**
  File: `Quantum_Federated_Learning_A_Comprehensive_Survey.pdf`
  Taxonomy of fully, hybrid and hierarchical QFL, frameworks (PennyLane, Flower, NVFlare) and
  datasets. Design reference for the federated-learning architecture.

---

## QKD networking standards implemented in Round 18

Added in Round 18 and previously recorded only in source docstrings.

- **ETSI GS QKD 015 V2.1.1 (2022-04)** - SDN control interface for a software-defined QKD network
  `[link]`
  URL: https://www.etsi.org/deliver/etsi_gs/QKD/001_099/015/02.01.01_60/gs_QKD015v020101p.pdf
  YANG source: https://forge.etsi.org/rep/qkd/gs015-ctrl-int-sdn (revision 2022-01-30,
  BSD-3-Clause). The host serves an incomplete certificate chain, so some clients reject it; the
  spec PDF above carries the same module.
  Added: no PDF bundled; spec and YANG re-verified 2026-08-04
  Implemented by `src/qsh/qkd/sdn_controller.py` and mirrored in the browser by
  `web/src/lib/clientSdnController.ts`. Field names (`qkdn_id`, `qkdl_id`, `qkdl_type`,
  `virt_next_hop`, `phys_channel_att`, `skr`, `eskr`) are taken verbatim from the YANG; nothing is
  renamed to read more nicely. Clause 1 is what settles the agent question: "Application layer
  interfaces and quantum-channel interfaces are out of scope", so an MCP agent is a **SAE** (Secure
  Application Entity), an application-plane key consumer, never the controller.

- **ETSI GS QKD 014 V1.1.1 (2019-02)** - key delivery API `[link]`
  URL: https://www.etsi.org/deliver/etsi_gs/QKD/001_099/014/01.01.01_60/gs_QKD014v010101p.pdf
  Added: no PDF bundled; edition status re-verified 2026-08-04
  Implemented by `src/qsh/qkd/kme.py`. **V1.1.1 is still the only published edition**; an ed2 stable
  draft (v1.3.1) existed as of 2025-06 but is not published, so any "QKD 014 compliant" claim,
  including this one, targets the 2019 document. The spec mandates mutual TLS with the SAE identity
  taken from the client certificate, which an in-process model has no transport to attach to; that
  gap is stated in `KeyManagementEntity.authentication_note()` and in `docs/threat-model.md` rather
  than left for a reader to notice.

- **ETSI GS QKD 020** - interoperable KMS API `[link]`
  URL: https://forge.etsi.org/rep/qkd/gs020-interop-kms
  Also: https://www.etsi.org/committee/qkd - the committee landing page, which is what this entry
  used to point at. It is not the spec, and an audit of this file flagged it as one of two entries
  with no resolvable identifier. The forge repository carries the API definition itself.
  Added: publication date re-verified 2026-08-04; URL corrected 2026-08-07
  Published **2026-06-29**, not 2024 as is widely repeated, and it covers the KMS-to-KMS interface
  *within a trusted node* rather than general inter-domain transport. Not implemented here; recorded
  so the roadmap cites the right scope.

- **ITU-T Y.3804 / Y.3805** - QKDN control and management `[link]`
  URL: https://www.itu.int/rec/T-REC-Y.3804/en
  Added: re-verified 2026-08-04
  Cited in `docs/pqc-roadmap.md` as the ITU counterpart to the ETSI control plane. Not implemented.

---

## External standards and threat context (see `docs/pqc-roadmap.md`)

- NIST **FIPS 203/204/205** (ML-KEM, ML-DSA, SLH-DSA), 2024.
  URL: https://csrc.nist.gov/pubs/fips/203/final (204, 205 by the same pattern)
- NIST **SP 800-90B** - entropy sources, including the clause 4.4 continuous health tests
  (Repetition Count, Adaptive Proportion) that `src/qsh/qrng/health.py` implements.
  URL: https://csrc.nist.gov/pubs/sp/800/90/b/final
- NSA **CNSA 2.0** - 2027 procurement mandate. On QKD, NSA answers **"No"** to using it to protect a
  National Security System and says it "does not support the usage of QKD or QC". It does **not**
  recommend a PQC-hybrid QKD add-on either. The hybrid framing in this repository comes from **BSI
  TR-02102-1 v2026-01 section 2.2**, not from NSA.
  URL: https://www.nsa.gov/Cybersecurity/Post-Quantum-Cryptography-FAQs/
- **BSI TR-02102-1**, version 2026-01.
  URL: https://www.bsi.bund.de/EN/Themen/Unternehmen-und-Organisationen/Standards-und-Zertifizierung/Technische-Richtlinien/TR-nach-Thema-sortiert/tr02102/tr02102_node.html
- **HNDL** (Harvest Now, Decrypt Later) and Q-Day acceleration - Gidney, arXiv:2505.15917, 2025.
  URL: https://arxiv.org/abs/2505.15917
- **The Business of Quantum Technologies** - copyrighted book, `[R-local]`, deliberately not
  redistributed (`.gitignore:11`). Background for the private business notes only; no technical claim in this
  repository rests on it.

---

## OSS reused

Licences below are what each project publishes. **They are not uniformly commercial-OK** - `PaySim`
is GPL-3.0 and `MP-SPDZ` carries a CSIRO licence, so the previous heading ("all commercial-OK") was
wrong on its own table. Neither is a dependency: PaySim is cited only, and MP-SPDZ is a reference
baseline. Everything this project actually installs is MIT, Apache-2.0 or BSD.

`scripts/bootstrap_submodules.sh` fetches the subset that is still wired up; the submodules removed
in Round 17 are marked below.

| Code | Used by | Licence |
|---|---|---|
| `OrtegaSA/CQC-QHE-repo` | cited only - `qsh/qhe/qcnn.py` is an independent implementation of the paper, not a wrapper | Apache-2.0 |
| `open-quantum-safe/liboqs` and `liboqs-python` | `qsh/pqc/*` | MIT |
| `openfheorg/openfhe-development`, `microsoft/SEAL` | `qsh/fhe/*`, benchmark | BSD-2 / MIT |
| `tenseal` | `qsh/fhe/*`, benchmark | **Apache-2.0**, confirmed from the installed package metadata; an earlier revision of this table listed it under BSD-2/MIT |
| `Alisah-Ozcan/HEonGPU`, `chart21/hpmpc` (PIGEON), `kanav99/shark` | `qsh/benchmark/*`, our own glue | Apache-2.0 / MIT |
| `data61/MP-SPDZ` | benchmark CPU MPC baseline, reference only | CSIRO, **not a blanket commercial grant** |
| `lschoe/mpyc` | `qsh/mpc/secret_inference.py` - non-linear MPC (ReLU, argmax on shares). A normal PyPI dependency in `requirements-fhe.txt`; the submodule was dropped in Round 17 and nothing is vendored | MIT |
| `IBM/AMLSim`, `synthetichealth/synthea` | **cited only, not used.** Round 17 removed both submodules as unreferenced; `qsh/data/synthetic.py` mentions them in prose only and no code path calls them | Apache-2.0 |
| `EdgarLopezPhD/PaySim` | cited only, never cloned or vendored | **GPL-3.0** for the simulator source; the Kaggle output dataset is separately CC-BY-SA-4.0 |
| `openfhe` (OpenFHE 1.5.1, CKKS with bootstrapping) | `qsh/fhe/backends.py` (`OpenFHEBackend`) | BSD-2-Clause |
| **`s0l0ist/node-seal`** (SEAL compiled to WASM, CKKS) | `web/src/lib/clientFHE.ts` - real in-browser FHE | MIT |
| **`mpc-framework`, `emp-wasm-engine`, `summon-ts`** | `web/src/lib/clientMPC.ts` - real in-browser garbled-circuit 2PC | MIT |
| **`mcp`** (MCP Python SDK; the class is `MCPServer`, named `FastMCP` before v2.0.0) and the official `@modelcontextprotocol/server-filesystem` | `qsh/mcp/*` - real quantum-safe MCP server and interop | MIT |
| **`onnxruntime-web`** | `web/src/lib/onnxInference.ts` - browser inference. The `webgpu` and `wasm` builds are separate bundles and the demo picks one by probing for a real adapter | MIT |
| `PennyLane` | `qsh/ai/federated.py` - quantum federated learning (`kind="qfl"`), and `qsh/qml_ids/detector.py`. The FedAvg and secure-aggregation code is this repository's own; Flower is cited as prior art, not used as a dependency | Apache-2.0 |

## Baselines and corpora cited in code but previously absent here

Round 21 audit: these are quoted as authority in registry docstrings, model cards or the SPA's baked
config, and had no entry. A baseline a page prints beside a measured number is a citation, so it
belongs in this file rather than only in a comment.

- **Kotowski et al., "The OPS-SAT benchmark for detecting anomalies in satellite telemetry",
  *Scientific Data* 12 (2025).** DOI 10.1038/s41597-025-05035-3. Data: Zenodo
  10.5281/zenodo.12588359, CC BY 4.0. Source of the 30 published baselines `satellite_comms.opssat`
  is quoted against. Added: 2026-08-07.
- **Yang et al., "MedMNIST v2", *Scientific Data* 10:41 (2023).** DOI 10.1038/s41597-022-01721-8.
  Source of the BreastMNIST ResNet-18 AUC 0.901 / ResNet-50 AUC 0.857 baselines. Added: 2026-08-07.
- **Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models",
  NeurIPS 2023 Datasets and Benchmarks.** arXiv:2306.16424. AMLworld; source of the 0.6323 finance
  baseline and of the corpus's own synthetic-by-construction statement. Added: 2026-08-07.
- **Egressy et al., "Provably Powerful Graph Neural Networks for Directed Multigraphs" (Multi-PNA+EU),
  AAAI-24, vol. 38 no. 10, pp. 11838-11846.** DOI 10.1609/aaai.v38i10.29073. Source of the 0.6816
  finance ceiling. Added: 2026-08-07.
- **Liu et al., "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark"
  (TSB-AD), NeurIPS 2024 Datasets and Benchmarks.** arXiv:2412.20512. Source of the point-adjustment
  argument: a uniform random score ranks 26th of 32 detectors under point adjustment, which is why
  this project reports event-wise F-beta and never point-adjusted F1 as a headline. Added: 2026-08-07.
- **Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
  Thresholding", KDD 2018.** arXiv:1802.04431. Telemanom. Added: 2026-08-07.
- **Ma, Qi, Zhao & Lo, "Practical decoy state for quantum key distribution", *Phys. Rev. A* 72,
  012326 (2005).** DOI 10.1103/PhysRevA.72.012326. The single-photon bound OptiQKD tunes. Added:
  2026-08-07.
- **Lopez-Rojas, Elmir & Axelsson, "PaySim: A financial mobile money simulator for fraud detection",
  EMSS 2016.** The bundled finance demo sample; a SIMULATOR, not real transactions. Added: 2026-08-07.
- **Sharafaldin, Habibi Lashkari & Ghorbani, "Toward Generating a New Intrusion Detection Dataset and
  Intrusion Traffic Characterization", ICISSP 2018.** CIC-IDS2017. Added: 2026-08-07.
- **Wolberg, Street & Mangasarian, "Breast Cancer Wisconsin (Diagnostic)", UCI ML Repository (1995).**
  DOI 10.24432/C5DW2B. Relicensed by UCI to CC BY 4.0, so attribution is due. Added: 2026-08-07.
- **Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep
  Learning", *Cell* 172(5) (2018).** DOI 10.1016/j.cell.2018.02.010. The chest X-ray source behind
  PneumoniaMNIST. Added: 2026-08-07.
- **Mavaddat et al., "Polygenic Risk Scores for Prediction of Breast Cancer", *AJHG* 104(1) (2019).**
  DOI 10.1016/j.ajhg.2018.11.002. PGS Catalog PGS000001 / PPM017270; source of the C-index 0.628 the
  genomics card cites as external and explicitly does not measure. Added: 2026-08-07.
- **"SoK: Private LLM Inference", IACR ePrint 2026/935.** Quoted in `models/registry.py` and
  `models/distil.py` on why distillation is the practical route to FHE-feasible inference. Added:
  2026-08-07.
- **arXiv:2607.11969** - instability of overlap-based anomaly metrics. Quoted in the satellite model
  card and baked into `config.defaults.json`; the reason the event-wise F0.5 non-reproduction is
  reported as metric instability rather than as a training failure. Added: 2026-08-07.
- **EO 14412, "Securing the Nation Against Advanced Cryptographic Attacks"**, signed 2026-06-22,
  91 FR 38483. PQC key establishment on high-value assets by 2030-12-31, digital signatures by
  2031-12-31, FAR rule putting contractors on the 2030 date; National Security Systems excluded and
  remaining on CNSA 2.0. The binding US instrument, against NIST IR 8547 which is still a draft.
  Added: 2026-08-07.

## Security scoring: sources evaluated and DECLINED (Round 22)

Five candidate replacements for this project's security score were read in full and all five were
declined, with reasons. Recorded in `security-scoring-fact-check.md` so the evaluation is not
repeated: QERS (arXiv:2601.13399), CARS (IACR ePrint 2026/1467), the quantum-safe library assessment
(arXiv:2605.17061), and two vendor sources. The composite was replaced by per-dimension facts led by
the NIST security category (FIPS 203/204), not by another weighted score.

## Stress testing: what is prescribed and what was declined (Round 22)

`stress-testing-fact-check.md` records the full sweep of NIST (SP 800-227, FIPS 203, ACVP), ETSI
(GS QKD 010/014/015/016), ISO/IEC (23837-1/-2, 4922-1/-2, 28033) and the QKD attack literature. No
standard prescribes adversarial stress testing of an integrated PQC+QKD+FHE+MPC system; ISO/IEC
23837-2 endorses discretized parameter sweeping as the METHOD but explicitly supplies no thresholds.
Four axes are defensible with published reference values; detector blinding as a QBER anomaly, the
optical evaluation activities, and any aggregate stress score are declined with reasons.

## CKKS / FHE security under decryption queries (IND-CPA-D / CPAD)

Added Round 23, because the product returns decrypted inference results and the threat model was
silent on the notion that governs exactly that. Full discussion in `../docs/threat-model.md`.

- Li, Micciancio, "On the Security of Homomorphic Encryption on Approximate Numbers", EUROCRYPT
  2021, ePrint 2020/1533. Approximate FHE fails IND-CPA-D; the attack needs one decryption query.
- Li, Micciancio, Schultz, Sorrell, CRYPTO 2022, ePrint 2022/816. Noise flooding and its precision
  cost.
- Cheon, Choe, Passelegue, Stehle, Suvanto, ACM CCS 2024, ePrint 2024/127. Extends the attack class
  to exact FHE. (Distinct from ePrint 2024/116, Checri et al., CRYPTO 2024, filed three days apart
  by a different team at a different venue - easy to conflate, so both are listed.)
- Checri, Clet, Renard, Sirdey, "Impossibility of CPAD security for a class of FHE schemes",
  ePrint 2026/203. No HELLHO scheme - covering basic BFV, BGV, CKKS and TFHE - can be CPAD secure.
  Practical key recovery on DE-CKKS, supported in OpenFHE, in hours on a laptop.
- Santato, Zanotto, "How (not) to Switch FHE Schemes", ePrint 2026/285. Breaks PEGASUS-style
  CKKS-to-FHEW switching in OpenFHE. Not applicable here - this project does no scheme switching -
  and recorded so that stays a checked fact rather than an assumption.
- Ogilvie, ASIACRYPT 2025, ePrint 2025/1618. The constructive counterpoint: about two bits of extra
  precision loss can restore security against passive key recovery for standard parameters.
- Bossuat, Costache, Mouchet, Nurnberger, Troncoso-Pastoriza, ePrint 2024/853, IACR CiC 2025, DOI
  10.62056/a3n5tx4e-. NOTE the retitle at the 2025-08-11 revision: originally "Practical
  q-IND-CPA-D-Secure Approximate Homomorphic Encryption", now "Accurate and Composable Noise
  Estimates for CKKS with Application to Exact HE Computation". Citing the old title points at a
  paper that no longer exists under that name.

## QSDC security basis, and why its threshold equals BB84's (Round 24)

`qkd.pp_threshold = 0.11` looked for several rounds like the Shor-Preskill BB84 bound copied into a
protocol it does not govern. It is not. QSDC security is expressed as Wyner wiretap **secrecy
capacity** Cs = I(A:B) - I(A:E); over a binary symmetric channel under collective attack that is
Cs = [1 - h(e)] - h(e) = 1 - 2h(e), which reaches zero at **e = 0.110028**. Shor-Preskill's one-way
BB84 bound is the same expression, so the two thresholds coincide by identity rather than by
transcription.

The distinction still matters in this project, because the two are not interchangeable in effect: at
the repository's measured GLLP zero-key point of 0.098106 the decoy-state key rate is already zero
while QSDC retains Cs = +0.074. A satellite run that keeps sending under QSDC after BB84 has stopped
producing key is behaving correctly - QSDC carries the message on the channel itself and needs no
extractable key.

- Long and Liu, Phys. Rev. A 65, 032302 (2002) - the two-step QSDC protocol.
- Deng and Long, Phys. Rev. A 69, 052319 (2004) - DL04 one-time-pad QSDC.
- Wyner wiretap-based QSDC security analyses (Qi, Long et al.) - the secrecy-capacity formulation.
- Pan, Lin, Wu, Sun, Ruan, Yin, Long, "Experimental free-space quantum secure direct communication
  and its security analysis", arXiv:2005.05102 - free-space QSDC at 500 bps over 10 m with a mean
  QBER of 0.49% +/- 0.27%, the closest published analogue to this project's satellite use case.
