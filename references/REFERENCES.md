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
  MNIST 96.7 percent - those are the PAPER's properties, and two of them this repository does not
  reproduce. **Relationship to this repository:** `src/qsh/qhe/qcnn.py` is an *independent and
  PARTIAL* implementation of the paper's construction. It is not Clifford+T: the module builds
  `AngleEmbedding` plus CNOT/RY/RZ and contains no T gate, with `t_count()` returning a labelled
  Solovay-Kitaev estimate rather than a count. Pauli-gate concealment for CIRCUIT privacy is also not
  reproduced - `qotp.py` conceals a state, not a computation, because propagating the Pauli frame
  through the circuit breaks at T, which is the difficulty the paper exists to address. The browser
  circuit in `web/src/lib/quantumSim.ts` is the one that is genuinely Clifford+T, and it counts its
  T gates. It does not import, vendor or wrap the authors'
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

## QOTP-based quantum homomorphic encryption, and what WS4 can claim (Round 30)

Added 2026-08-11, verified at the arXiv abstract page.

- **Hernandez-Bueno, Lage, Higuero & Astorga, "Quantum Homomorphic Encryption: Towards Practical
  and Private Computation on Untrusted Quantum Hardware."** arXiv:2604.19256 [quant-ph],
  2026-04-21, 14 pp. A universal QHE framework derived from the Quantum One-Time Pad,
  information-theoretically secure, "exploiting the transformation properties of quantum gates
  under Pauli encryption" for non-interactive homomorphic evaluation of arbitrary Clifford+T
  circuits, with results from simulators and real quantum processors. **This is WS4, published
  four months before this round, and the repository did not cite it.** It is the correct
  positioning reference for `qotp_clifford_random_circuit_demo`, and it also makes the
  repository's honest disclosure - that the browser circuit is Clifford+T while the Python module
  contains no T gate - more informative rather than less. Added: 2026-08-11.
- **Joshi, Mishra & Karthikeyan, "Blind Transpiler: An open-source library for universally blind
  and homomorphic quantum computations."** arXiv:2607.17131, v1 2026-07-19, v2 2026-08-09.
  Transpiles Qiskit circuits to blind counterparts and demonstrates a blind variational quantum
  classifier on IRIS. The closest software prior art to this project's browser QCNN under QOTP.
  Added: 2026-08-11.
- **Liu, Gong, Tian, Zhang, Zheng & Xu, "Efficient Quantum Fully Homomorphic Encryption."**
  arXiv:2604.23490 [quant-ph], v1 2026-04-26, v2 2026-06-21, 32 pp. Reduces the quantum gadget
  from O(lambda^2) to O(lambda log^2 lambda) EPR pairs and moves every quantum resource to the
  server, so the client performs only classical LWE key generation and Pauli-key encryption under
  classical FHE. **Reframes rather than contradicts** this repository's "rate-1 transport, no
  Eval" caveat: as of April 2026 the direction of travel is a fully classical client, which is
  architecturally closer to the browser-tab story than to the caveat's pessimism. Added:
  2026-08-11.

**What this project can and cannot claim on this machinery, recorded here because the split
matters.** WS4(c), circuit privacy against a malicious server, is met for Clifford circuits: the
server holds the maximally-mixed state throughout and the client recovers the exact result by
tracking the Pauli frame. WS4(f), hiding the server's circuit from the client, is NOT met and is
in tension with the mechanism - the client must know the circuit to update its frame. The two are
now separate rows in `README.md` rather than one row that could only ever read PARTIAL.

## Prior art for the novelty claim, and the asymmetry it exposes (Round 30)

Added 2026-08-11. Checklist rule M4 requires a novelty claim to survive an adversarial prior-art
search. Run against this project's own headline claim, the search returned prior art for one half
and nothing for the other. Both are recorded, because reporting only the second would be the
omission this file exists to prevent.

- **Buchsteiner, Koch, Rotaru & Rechberger, "webSPDZ: Versatile MPC on the Web."** IACR ePrint
  2025/487, received 2025-03-14. Compiles MP-SPDZ's C++ backend to WebAssembly with Emscripten and
  moves party communication to WebRTC or WebSockets, supporting the 40-plus protocols MP-SPDZ
  implements. **Its abstract names two earlier browser MPC engines, JIFF and MPyC-Web.** So
  multiparty computation in a browser tab is established work and the garbled-circuit half of this
  demo is a re-implementation, not a first. `docs/threat-model.md` already mentioned JIFF; webSPDZ
  and MPyC-Web were absent. Added: 2026-08-11.
- **Aikata, Krieger & Sinha Roy, "Privacy at your Fingertips: Enabling Rapid Client-Side
  Operations in Fully Homomorphic Encryption."** IACR ePrint 2026/515, received 2026-03-13.
  Verified at the listing: "client-side operations" means encryption and decryption, and the
  abstract states that FHE "allows users to offload large computations to servers". It reduces
  client enc/decrypt cost by up to 97 percent - the client still does not evaluate. **This is the
  clearest evidence that the field's default division of labour is the opposite of this demo's**,
  which is why it is cited beside the novelty claim rather than in the FHE section. Added:
  2026-08-11.
- **Zama, TFHE-rs JS-on-WASM API documentation.** States that the WebAssembly client API supports
  key generation, encryption and decryption but **not FHE computation**, citing missing WASM
  threading and the 2 GB WASM memory ceiling. Documentation rather than a result, and the
  strongest single sentence supporting the in-tab evaluation claim: the leading FHE toolchain
  deliberately does not evaluate in the browser. Checked 2026-08-11.
- **Wang, Yu, Han, Gu, Xu & Huang, "WAMO: Toward Secure Browser Inference via Web Model
  Obfuscation in WebAssembly."** Proceedings of the ACM Web Conference 2026 (WWW '26), 2026-04-09,
  DOI 10.1145/3774904.3792297. **Carried as a partial contradiction, which is why it is worth
  having:** it argues that deploying a model in a browser exposes its code and parameters to an
  untrusted environment, against the intuition that client-side execution is automatically
  private. It does not threaten this project - the models here are public on Hugging Face, so
  extraction is not a harm - but it is the paper a reviewer will raise, and the threat model
  should name it. **Metadata verified via the OpenAlex record for that DOI; the full text was not
  retrieved (ACM DL refuses automated fetches), so nothing beyond title, authors, venue and date
  is asserted here.** Added: 2026-08-11.

**The negative results, stated as findings rather than left as absence.** No peer-reviewed
2025-2026 paper was located that performs CKKS homomorphic *evaluation* of a neural network inside
a browser tab with no server compute; every browser-FHE artefact found is encrypt-in-browser,
evaluate-on-server. No WebGPU-accelerated homomorphic-encryption or MPC paper was located for
2025-2026 either, which is consistent with the note in `web/src/lib/clientFHE.ts` that no
commercial-OK WebGPU FHE exists in 2026. Absence of evidence is recorded as absence of evidence:
the searches and their dates are here so the next round can re-run them rather than re-reason.

## Hybrid FHE/MPC private inference: the 2026 state of the art, and an attack on the nearest design

Added 2026-08-11. Each entry below was verified against the arXiv abstract page or the IACR listing
directly, not against another file in this repository. The architecture discussion lives in
[`docs/architecture.md`](../docs/architecture.md); this section is the provenance.

- **Biswas, Chartier, Dhasade, Jurien, Kerriou, Kermarec, Lemou, Tranie, de Vos & Vujasinovic,
  "Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption" (Safhire).**
  arXiv:2509.01253, submitted 1 Sep 2025. Linear layers encrypted server-side, non-linearities
  offloaded to the client **in plaintext**, which removes bootstrapping and allows exact activations;
  1.5x-10.5x lower latency than Orion. Model confidentiality is protected by **randomised shuffling**
  of the intermediates the client sees. This is the closest published architecture to this
  repository's, and the difference is precisely where the non-linear half runs. Added: 2026-08-11.
- **Kim & Lee, "Shuffling is Not Enough: Breaking Permutation-Based Model Confidentiality in Hybrid
  FHE Inference."** IACR eprint 2026/1527, ESORICS 2026 (received 2026-07-25). Recovers a
  permutation-invariant summary of a linear layer exactly and achieves perfect model
  distinguishability, needing `d+1` queries for a `d`-input layer - 5,712 for all linear layers of a
  ResNet-20. Breaks the shuffling-plus-noise defence that a plaintext-offload design depends on.
  **Carried as a trip-wire:** this repository sends the non-linear half as secret shares, so no
  plaintext intermediate exists to query, but the paper does not analyse the MPC-evaluated case - so
  the correct claim is that the attack's precondition is absent here, NOT that the MPC variant has
  been shown secure. If the non-linear path is ever moved to plaintext client offload, this applies
  directly. Added: 2026-08-11.
- **Xu, Lu, Yu, Yi, Lin, Wang & Li, "Breaking the Layer Barrier: Remodeling Private Transformer
  Inference with Hybrid CKKS and MPC" (BLB).** arXiv:2508.19525, Aug 2025. Splits at operator rather
  than layer granularity and fuses adjacent linear operators to minimise CKKS<->MPC conversions; 21x
  communication reduction vs BOLT (S&P'24) and 2x vs BumbleBee (NDSS'25), with 13x and 1.8x latency
  reductions under GPU acceleration. Already named in `README.md` and `docs/architecture.md` as the
  current end of the Gazelle -> Delphi -> BOLT -> Bumblebee line; recorded here with its numbers.
  Added: 2026-08-11.

**FHE maturity in 2026, for calibration rather than as a citation to lean on.** Surveys and vendor
retrospectives converge on roughly 100x-1000x overhead against plaintext, with what actually ships
being narrow: small-model inference - logistic regression, trees, small networks - completing in
seconds under CKKS. That is the honest frame for this project's T1 tier. The distilled students here
are small **because that is what runs under encryption today**, not because a larger model was
attempted and abandoned, and the measured teacher-student gaps are reported rather than hidden.

## Collaborative AML under FHE: the nearest published work to this repository's finance use case

Added 2026-08-11, verified at the arXiv abstract page.

- **Effendi & Chattopadhyay, "Privacy-Preserving Graph-Based Machine Learning with Fully
  Homomorphic Encryption for Collaborative Anti-Money Laundering."** arXiv:2411.02926, 5 Nov 2024
  (rev. 11 Nov 2024); SPACE 2024, DOI 10.1007/978-3-031-80408-3_6. TFHE via Zama Concrete ML, over
  two pipelines: a GNN with quantisation and pruning, and **graph-based XGBoost using a Graph
  Feature Preprocessor** - which is the same architecture as this repository's `finance_aml.gbm`
  teacher. Reports over 99 percent accuracy, F1, precision and recall on a **balanced** AML
  dataset in both plaintext and encrypted settings, and an 8 percent F1 improvement from graph
  features on the imbalanced version. Added: 2026-08-11.

**Read those two numbers together, because the pair is the lesson.** This repository's finance
teacher scores minority-class F1 **0.1519** at AMLworld's real prevalence of roughly 1 in 981. The
published work scores over 0.99 on a balanced set. Both are correct; they are not the same task,
and a reader comparing them without the prevalence would conclude this implementation is six times
worse than the state of the art rather than measured on harder rows. It is the same reason this
project refuses to quote aggregate accuracy for AML at all - at that prevalence a model predicting
"never launder" scores 99.9 percent accuracy - and it is why the finance page prints
minority-class F1 against an all-positive baseline instead.

**One practical note on reuse.** That paper's FHE layer is Zama Concrete ML, which this project
evaluated and declined. The licence objection stated more precisely, re-read at the primary source
2026-08-11: tfhe-rs is **BSD 3-Clause Clear**, whose distinguishing clause is "NO EXPRESS OR IMPLIED
LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE" - the copyright grant is permissive
and the **patent grant is expressly withheld**, which is a sharper and checkable objection than the
"require a paid licence" this file previously asserted about the vendor's terms. And separately,
Concrete ML's own release cadence had stalled at the time of the audit. So the nearest published
approach to this use case is built on a stack this repository cannot adopt commercially, which is
part of why the CKKS path here is TenSEAL and OpenFHE.

**Industry deployment, recorded as a vendor claim and nothing stronger.** Duality Technologies
publishes a case study in which two competing global banks jointly trained an AML model under its
platform without exchanging customer data, reporting a 34 percent improvement in detection. The
figure appears in the vendor's own marketing material and no independent write-up of the
engagement was locatable, so it is carried here as **`[UNVERIFIED - vendor claim]`** - useful as
evidence that cross-institution private AML training is being sold and bought in 2026, not as a
benchmark number. Checked 2026-08-11.

## Quantum machine learning: results that constrain what a QCNN can claim

Added 2026-08-11. Both verified at the arXiv abstract page.

- **Bermejo, Braccia, Rudolph, Holmes, Cincio & Cerezo, "Quantum Convolutional Neural Networks are
  Effectively Classically Simulable."** arXiv:2408.12739, 22 Aug 2024, revised 20 Feb 2026. Randomly
  initialised QCNNs can only operate on the information in **low-bodyness measurements** of their
  input states, and their behaviour is efficiently classically simulable by an algorithm equipped
  with Pauli shadows. The authors argue QCNN successes largely reflect benchmark simplicity, and that
  non-trivial datasets are a necessary ingredient for progress in QML. **Why it is cited here:** this
  repository's trained QCNN loses to logistic regression on the identical split (medical accuracy
  0.7326 vs 0.9535; finance minority-F1 0.7143 vs 0.8657). This paper is the reason that is the
  expected result rather than a failure of implementation, and it is quoted in both QCNN model cards.
  Added: 2026-08-11.
- **Li, Wu, Li & Yung, "Barren Plateaus Beyond Observable Concentration."** arXiv:2603.18479, 19 Mar
  2026, revised 19 Jun 2026. Unifies existing barren-plateau explanations under *observable
  concentration* and proves that avoiding it is **necessary but not sufficient** for trainability,
  identifying information loss outside the measurement light cone and correlated forward/backward
  blocks as further sources of gradient suppression. **Why it is cited here:** the untrained
  Clifford+T circuit this project removed from its inference path showed exactly the concentration
  symptom - |<Z0>| within +/-0.21 of zero for every input tested, with sign set by numerical noise -
  and this paper names the mechanism and warns that fixing concentration alone would not have made it
  trainable. Added: 2026-08-11.

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

Licences below are what each project publishes. **They are not uniformly commercial-OK** - `PaySim` is
GPL-3.0, so the previous heading ("all commercial-OK") was wrong on its own table. PaySim is cited
only, not a dependency. Everything this project actually installs is MIT, Apache-2.0 or BSD.

**MP-SPDZ correction, 2026-08-11.** This paragraph used to say MP-SPDZ "carries a CSIRO licence" and
listed it beside PaySim as a reason the set is not commercial-OK. Read at the primary source, its
`License.txt` says: *"CSIRO grants you a licence to the Software on the terms of the BSD 3-Clause
Licence."* The top-level grant is therefore plain BSD-3-Clause and belongs in the permissive set. The
real caveat is narrower and was attached to the wrong thing: **bundled third-party components** inside
the repository (SPDZ-2 / Bristol, SPDZ-BMR-ORAM / Bristol and Bar-Ilan, and others) carry their own
terms, which is what needs checking before depending on a specific protocol. Recorded because this
file's stated purpose is to stop exactly this kind of drift, and it had drifted in the direction of
being needlessly restrictive - which is still being wrong.

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

- **Ruszczak, Kotowski, Evans & Nalepa, "The OPS-SAT benchmark for detecting anomalies in
  satellite telemetry", *Scientific Data* 12, 710 (2025).** DOI 10.1038/s41597-025-05035-3. Data:
  Zenodo 10.5281/zenodo.12588358 (concept DOI), CC BY 4.0. Source of the 30 published baselines
  `satellite_comms.opssat` is quoted against. Added: 2026-08-07.
- **Kotowski, Haskamp, Andrzejewski, Ruszczak, Nalepa et al., "European Space Agency Benchmark for
  Anomaly Detection in Satellite Telemetry".** arXiv:2406.17826. Data: Zenodo
  10.5281/zenodo.12528696, CC BY 3.0 IGO. Source of the 0.968 event-wise F0.5 baseline
  `satellite_comms.telemanom` is quoted against - Supplementary Table 9 (anomalies alone), NOT
  Table 2 (all events), which is a different comparator and was once cited by mistake.
  Added: 2026-08-11.

> **These are two different papers by an overlapping group, and conflating them is an easy mistake to
> make - one that was made here.** OPS-SAT-AD is **Ruszczak** first (Scientific Data, Zenodo
> ...12588359, CC BY 4.0, the corpus `satellite_comms.opssat` trains on). ESA-ADB is **Kotowski**
> first (arXiv:2406.17826, Zenodo ...12528696, CC BY 3.0 IGO, the corpus `satellite_comms.telemanom`
> is scored against). Thirteen places in this repository credited OPS-SAT-AD to "Kotowski et al.",
> and the 2D overlay credited Ruszczak with the title of an unrelated algebraic-geometry paper. Both
> were fixed by checking Crossref and Zenodo, not by checking another file here. Before bulk-editing
> either name, confirm WHICH benchmark the line is about: the DOI and the Zenodo id are the reliable
> discriminators, and the author surname is not.
>
> A third variant to expect: the ESA-ADB **Zenodo record** lists **De Canio** first (contact person),
> then Kotowski and Haskamp as researchers, while the **paper** is Kotowski first. Citing the paper
> for the benchmark and the Zenodo DOI for the data - as the two entries above do - is correct and is
> not an inconsistency. Verified 2026-08-11 against Crossref, the arXiv listing and both Zenodo
> records. Note that the CC BY 3.0 IGO term appears on the Zenodo landing page but NOT in its REST
> metadata, which returns no licence at all - so an API-only check will wrongly read as unlicensed.

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
  DOI 10.1016/j.ajhg.2018.11.002. The paper behind **PGS000004 (PRS313)**, which the genomics use case
  has scored since Round 31. Its external baseline is now **AUROC 0.633 [0.624, 0.641]** from PGS Catalog
  **PPM001647** (Jia et al., JNCI Cancer Spectrum 2020, doi:10.1093/jncics/pkaa021, n=400,812 Europeans),
  cited and explicitly not measured here. The retired PGS000001 baseline was PPM017270's Harrell C-index
  0.628 - a different quantity on a different scale, which is one reason the switch was worth making.
  Added: 2026-08-07, revised 2026-08-11.
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

## QSDC security basis: secrecy capacity, not a fixed QBER (Round 24)

**This entry corrects an earlier version of itself.** That version said `qkd.pp_threshold = 0.11`
was correct for QSDC because the wiretap secrecy capacity `Cs = 1 - 2h(e)` has its root at 0.110028,
the same as Shor-Preskill, so the two coincided by identity. The identity is real but the conclusion
was wrong, because `Cs = 1 - 2h(e)` is the **loss-free** case.

The published criterion is `Cs > 0`, and `Cs` depends on channel loss:

    Cs = Q_Bob [ 1 - h(e) - g h(e_chk) ],    g = Q_Eve / Q_Bob = 10^(loss_dB / 10)

So the tolerable error rate falls fast as the link gets lossier: 0.110 at 0 dB, **0.048 at the 4.1 dB
Qi et al. measured on a real 1.5 km fibre link (g = 2.57)**, and ~0.0003 at the ~24.5 dB this
project's own free-space downlink reports. Applying 0.11 to a lossy channel is wrong in the
**permissive** direction - it delivers over a channel whose secrecy capacity is already zero. The
satellite path now computes `Cs` directly.

Three further facts that stop any single number being quoted safely:

- **The founding protocols specify no threshold at all.** Long and Liu (PRA 65, 032302, 2002) say
  "below certain threshod" (sic); Deng, Long and Liu (PRA 68, 042317, 2003) say "reasonably low";
  DL04 (Deng and Long, PRA 69, 052319, 2004) says "if the error rate is high ... the communication is
  halted". The numbers all come from later security analysis.
- **11% appears in the QSDC literature only where the protocol reduces to BB84.** Zhao et al.
  (Front. Phys. 19, 51201, 2024) give 0.11 for entanglement-distillation QSDC against 0.073 for their
  hybrid-entanglement scheme; Pan et al. (Sci. Adv. 11, eadt4627, 2025) state 11% and say plainly
  that their expression "is exactly the key rate of the BB84 QKD protocol". DL04 - the family every
  fibre experiment implements - is put at ~5.7% (Wu, Lin, Yin and Long, Quantum Eng. 1, e26, 2019).
- **There are two error rates and two aborts.** The forward-leg check error governs SECURITY; the
  return-leg QBER governs RELIABILITY and is bounded by the error-correcting capability, not by a
  security threshold. Collapsing them into one comparison is a modelling error in its own right.

**No standards body has specified QSDC**, and this was checked exhaustively rather than assumed.
All 14 published ETSI QKD deliverables were searched full-text (2.0 M characters): zero hits for
QSDC or "secure direct communication", against a control of 7 documents matching "BB84". ITU-T's
complete X.1700-X.1749 and Y.3800-Y.3835 sets are QKD or QRNG throughout, and the work programme
across every study period 2001-2028 and every status returns zero QSDC items against a control of
186 quantum-titled items. ISO/IEC JTC 1/SC 27's entire catalogue contains exactly two
quantum-titled standards, both parts of 23837, both QKD. IEEE has no QSDC standard - and no QKD one
either. China's national platform returns no data for the Chinese term, and the national quantum
terminology project scopes itself to QKD. QSDC is absent even from CEN/CENELEC's
pre-standardization roadmap, which mentions QKD 319 times and QSDC zero.

One terminology trap that produces false positives, and it is easy to fall into. In Chinese sources
**"quantum secure communication" is the QKD umbrella term, not QSDC** - they are different words and
different technologies. GB/T 42829-2023 and YD/T 4301-2023 are both QKD. A source claiming "China has
a quantum secure communication standard" is almost certainly pointing at one of those. Only the
distinct Chinese term for quantum secure DIRECT communication means QSDC, and no standards document
at any level contains it. Korea's TTA has the term in a terminology dictionary with no standard
number attached, which is the closest any body has come.

The correct phrasing is therefore **"no published standard; research-stage technology"** - not
"standardization in progress", because no work item exists at any body. Two citation traps worth
knowing: **"IEEE 1913-2019" is not a real published standard** (the PAR was withdrawn on 11 December
2024 and the scope was QKD anyway), and the 2024 IEEE COMST QSDC survey mentions standardization
exactly once, about QKD.

Primary sources: Qi, Sun, Lin, Yin, Long, Light Sci. Appl. 8, 22 (2019), doi:10.1038/s41377-019-0132-3
- the `Cs > 0` abort rule and the measured g = 2.57. Pan, Lin, Wu, Sun, Ruan, Yin, Long,
arXiv:2005.05102 - free-space QSDC, 500 bps over 10 m, QBER 0.49% +/- 0.27%, and the DBER/QBER
distinction. Long and Liu (2002), Deng, Long and Liu (2003), Deng and Long (2004) - the protocols.

## Round 31 additions, verified at primary source 2026-08-11

Every entry below was loaded at its arXiv abstract page, IACR eprint listing, Federal Register API
record or whitehouse.gov PDF before being written here. Nothing in this section was taken from a
secondary write-up or from another file in this repository. Where a claim could not be verified it is
marked, and where a check came back clean that is recorded too - a negative result that has been
established is worth keeping, because otherwise it gets re-litigated every round.

### PQC migration, now binding rather than advisory

- **OMB Memorandum M-26-15, "Execution of the Migration to Post-Quantum Cryptography."** Russell T.
  Vought, Director, Office of Management and Budget, 2026-06-24. Full 10-page PDF read from
  whitehouse.gov. The operational instrument behind EO 14412, and the most consequential item in this
  batch for [`docs/pqc-roadmap.md`](../docs/pqc-roadmap.md):
  - Action One: mitigate as much quantum risk as feasible **by 2030-12-31**.
  - Action Two: agency PQC Migration Plans to OMB and ONCD **within 120 days** (about 2026-10-22).
  - Five phases: **1** 2026-2027 strategy and discovery, **2** 2027-2028 pilots, **3** 2028-2030
    prioritised migration and key establishment, **4** 2031 signature migration, **5** 2035 full
    migration.
  - Phases 3 and 4 each end with the sentence *"Ensure all systems are cryptographically agile."* That
    is a mandated requirement naming the exact property [`docs/crypto-agility.md`](../docs/crypto-agility.md)
    claims, which changes that document's status from good practice to alignment with a directive.
  - **"Agencies must align their plans with NIST Internal Report (IR) 8547 ... or successor
    document."** This sharpens rather than contradicts the roadmap's line that IR 8547 "is still a
    draft, not binding": the draft is now normatively referenced by a binding memorandum. Both facts
    have to be stated together, because either alone misleads.
  - Does not apply to national security systems, consistent with what this repository already records
    about EO 14412.
- **Two citation checks recorded so they are not repeated.** Secondary sources (Wiz,
  gopher.security) label the executive order **14409**. The Federal Register API returns EO
  **14412**, "Securing the Nation Against Advanced Cryptographic Attacks", signed 2026-06-22,
  published 2026-06-25, **91 FR 38483**. This repository's number and FR citation are correct and the
  blogs are wrong. Separately, **NIST IR 8547 is confirmed still at Initial Public Draft** (published
  2024-11-12, comments closed 2025-01-10, no second draft and no final), so the roadmap's claim holds.

### QKD security proofs that supersede this repository's baseline

- **Sixto, Curras-Lorenzo, Pereira, Zapatero, Navarrete & Curty, "Finite-key security analysis of
  decoy-state QKD with source and detector imperfections."** arXiv:2606.29943, 2026-06-29. An
  analytical finite-key proof that carries state-preparation flaws, bit-and-basis side-channel leakage
  **and correlations**, setting-independent intensity fluctuations, and detection-efficiency mismatch
  simultaneously. **Directly relevant, and not comfortable reading:** `qkd/bb84.py::_decoy_secure_fraction`
  and `qkd/decoy_reference.py` both rest on Ma, Qi, Zhao & Lo (2005), and this paper's motivation is
  that existing proofs "rely on assumptions that are difficult to justify in realistic high-speed
  implementations, such as the independent and identically distributed nature of the emitted signals"
  - which is exactly what this repository assumes. The Ma et al. reproduction stays as the reference
  implementation because that is what the numbers on the page are computed against; this is the 2026
  replacement and the honest statement is that the demo has not adopted it.
- **Ivchenko, Khmelev, Kurochkin & Trushechkin, "Security of quantum key distribution with passive
  basis choice and detection-efficiency mismatch for a realistic satellite setup."** arXiv:2608.09793,
  2026-08-10. Passive-basis-choice BB84 with mismatch across four threshold detectors plus a
  decoy-state adaptation, parameterised on the **Micius satellite to Zvenigorod ground station** link -
  a published real-satellite parameterisation rather than a simulation. The string
  "detection-efficiency mismatch" appears nowhere else in this repository, and the satellite page's
  free-space link budget has no term for it.
- **Hossain & Hossain, "SATLOCK: Handover-Coupled Scheduling for Weather-Resilient Quantum Key
  Distribution over LEO Constellations."** arXiv:2607.17076, v1 2026-07-19, v2 2026-07-27. Composite
  channel model with atmospheric loss, pointing jitter, **Markov cloud cover, decoy-state estimation
  and finite-key correction**; an ILP upper bound; a decentralised DQN reaching 91.8% / 84.6% of ILP.
  This is the closest published prior art to `qkd/optimize.py` (the WS1 AI-QKD parameter optimizer)
  applied at constellation scale, and its finding that **finite-key effects eliminate short,
  low-elevation passes entirely** is a constraint the satellite path should state rather than a
  refinement.

### Secure inference and threshold FHE

- **Ma, Li & Schmid, "FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing."**
  arXiv:2606.09551, 2026-06-08. Compiles per-operator protocols into two batched FSS evaluations:
  1.24x-1.50x end-to-end speedup and 9-16% less online communication over the state of the art in
  FSS-based GPU secure inference. Targets exactly the layer `mpc/secret_inference.py` occupies, which
  is `mpyc` with pure-Python secret shares for ReLU and argmax. The honest read is that **the
  non-linear half is the active research frontier and this repository's implementation is the slow
  reference design**, not the state of the art - and that the leverage is in the protocol, not in
  changing the model to suit MPC.
- **Kim, Kim & Lee, "Revisiting Shamir Secret Sharing for Threshold Fully Homomorphic Encryption."**
  IACR eprint 2026/1559, received 2026-07-30. Reduces the ThFHE ciphertext modulus from `O(N)` to
  `O(t log(N/t^2))` bits, with every algorithm completing under 0.2 s at N=64 and scaling past N>=256.
  The only threshold-FHE anchor here was Chardouvelis, Doettling & Malavolta (2020/1454) in
  `qhe/rate1_qfhe.py`, which is the quantum-side citation; there was none for the classical
  multi-party side even though `submodules/HEonGPU` ships N-out-of-N collective bootstrapping. Bears
  on why [`docs/architecture.md`](../docs/architecture.md) splits linear-to-FHE and
  non-linear-to-MPC rather than using threshold decryption.

### The primary source for the FHE cost ceiling

- **Park, Park, Park, Ahn, Cheon, Hanrot, Kim, Park & Stehle, "Scaling up FHE-based
  Privacy-Preserving ML: Higher Throughput, Longer Inputs for LLama-3-8B."** arXiv:2601.18511, v1
  2026-01-26, v2 2026-06-17. Verbatim from the abstract: *"On 8 NVIDIA RTX PRO 6000 GPUs, 128
  encrypted tokens take 20s for summarization and 18s/token for generation (vastly outperforming the
  SOTA 295s on costlier H100 GPUs). For a heterogeneous 4096-token input (last 128 encrypted), it
  takes 64s for summarization and 22s/token for generation."*

  **Recorded here because it was cited without provenance.** `docs/pqc-roadmap.md` quoted the figure
  and `models/registry.py` attributed it to the secondary SoK (IACR 2026/935) rather than to this
  paper, while this file - the provenance file - did not list it at all. It is also the result that
  retired the registry's unscoped claim that homomorphic evaluation "will remain" cost-bounded: what
  survives is the scoped claim, that it is cost-bounded at a single device in a browser tab.

  Direction of travel, **metadata-verified only** (titles and dates from the arXiv API listing;
  abstracts not individually opened, so no number from these four should be quoted):
  arXiv:2604.04783 GPU acceleration of TFHE nonlinear layers for encrypted LLM inference (2026-04-06);
  arXiv:2602.11470 Cachemir, FHE generative inference with KV cache (2026-02-12); arXiv:2512.11269 a
  scalable multi-GPU framework for encrypted large-model inference (2025-12-12); arXiv:2602.22229
  FHECore, GPU microarchitecture for FHE.

### Negative results, re-verified 2026-08-11

Both support claims this repository already makes, and both are recorded so the search is not redone.

- **No in-browser FHE or MPC execution has been published since webSPDZ.** An IACR eprint full-text
  search for "WebAssembly" returns 8 results across all years, the newest crypto-relevant one still
  **2025/487 webSPDZ**; an arXiv sweep of WebAssembly, garbled-circuit and browser-plus-homomorphic
  queries, newest first, returns 40 papers back to 2026-02 with zero in-browser FHE or MPC results.
  The Round 30 finding therefore re-verifies, and the composition novelty claim rests on it.
- **Nothing in QHE or QOTP is newer than Round 30's citations.** An arXiv query on `cat:quant-ph AND
  abs:homomorphic`, newest first, puts arXiv:2607.17131 (Blind Transpiler, already cited) at the top
  of the relevant set; the only newer quant-ph hit is arXiv:2607.21449 (QuantumChain, blockchain-backed
  QFL, 2026-07-23), which is off-thesis.

### Verified but not yet cited from anywhere

Held here rather than dropped, because both bear on the WS4(c) / WS4(f) split and the client in this
project is a browser tab.

- **van Dam, Grimbergen & Wehner, "Verifiable blind quantum computing: Comparative analysis and design
  considerations for client architectures."** arXiv:2607.05650, 2026-07-06. Classifies BQC clients as
  emission-, measurement- and rotation-based and concludes that measurement-based remote state
  preparation and reflection-based teleportation are the default candidates.
- **Lee, Kim & Chung, "BPBO: Blindness-Preserving Brickwork Optimization by Certified Region
  Resynthesis."** arXiv:2606.29962, 2026-06-29. Grover-3 pattern reduced from 3x725 to 3x98 while
  preserving blindness.
