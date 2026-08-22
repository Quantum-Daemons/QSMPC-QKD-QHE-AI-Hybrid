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
  does not contain it. Fetch from the URL. Two are ~17 MB each and one is a copyrighted book;
  since Round 45 four more are excluded on LICENCE rather than size, adjudicated per file in
  [`DATA_LICENSES.md`](../DATA_LICENSES.md) section 8.
- `[link]` never bundled. Fetch from the URL.

An earlier revision marked the `[R-local]` entries `[R]`, which told a reader they had files that a
clone never contained. **It happened again in Round 45**, to four entries: that round excluded the
PDFs on licence, wrote the adjudication table, listed "leaving `REFERENCES.md` marking them `[R]`"
in its own plan as the defect it was most likely to ship, and shipped it. The marker is the claim
about what a clone contains, and it is checked by
`tests/test_bundled_reference_pdfs_carry_a_grant.py` from Round 46 onward.

Workstreams: **WS1** AI-QKD parameter optimisation and entropy, **WS2** QML-IDS, **WS3** QSMPC x QKD,
**WS4** QHE x AI. Cross-cutting: PQC-first hybrid, crypto-agility, XAI.

---

## WS1 - AI-QKD (real-time parameter optimization) + entropy (QRNG)

**A source-level fact-check current at 2026-08-22 lives in
[`2026-08-factcheck.md`](2026-08-factcheck.md).** It confirms the ETSI/NIST/Rusca citations in this
section at their cover pages and re-derives every internal figure exact, and it contradicts six
claims - the SP 800-90B health tests being run on conditioned rather than raw samples, the
`FiniteKeyParams` defaults still carrying the retired intensities, "two-intensity decoy-state bound"
naming a protocol the cited equations are not, two attributions in `decoy_reference.py`, permissive
terms in the E91 and QSDC models, and the ITU-T series now running to Y.3838 rather than Y.3835.
Each carries whether it was reproduced locally or is recorded as reported-and-unreproduced.

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

- **Quantum key distribution over a 2 km free-space channel with a high secure key rate** - Park,
  Kim, Im, Jo, Kim, Choi and Ihn, **arXiv:2607.24118v1** [quant-ph], submitted 2026-07-27 `[link]`
  URL: https://arxiv.org/abs/2607.24118
  Added: no PDF bundled; abstract loaded and claim verified 2026-08-15
  Licence **CC BY 4.0**, read from the abstract page's own licence line.
  **v1 only. It has an arXiv DataCite DOI (10.48550/arXiv.2607.24118) and NO journal DOI** - which
  is not the same thing as "no DOI", the phrasing a code comment used until Round 44. Every arXiv
  paper carries a DataCite DOI; what is absent here is peer review.
  Reports, verbatim from the abstract: a **100 MHz** repetition rate, **164.8 kbps** secure key rate
  and QBER of **approximately 3.3 %** over a 2 km outdoor free-space channel, with active beam-wander
  correction by fast-steering mirrors.
  **What it grounds, and only this:** the `source_rate_hz` / `free_space_source_rate_hz` defaults in
  `src/qsh/config/defaults.py`. Round 44 made the source repetition rate an explicit, operator-set
  assumption so ETSI GS QKD 015's `skr`/`eskr` could carry the bit/s the spec defines them as, and
  this paper's 100 MHz is the default that assumption starts from. It also anchors the 0.025
  misalignment floor in the free-space channel model.
  **Explicit negative, because a borrowed clock invites a borrowed result.** No other number from
  this paper is used. Its 164.8 kbps is quoted on the control-plane panel purely as the nearest
  measured figure this repository cites, labelled as belonging to a different channel - a 2 km
  horizontal path, not this model's 50 km of fibre or its ground-to-LEO slant range - and it is
  **neither reproduced nor claimed** anywhere in this project. A preprint is a weaker source than a
  refereed one and the demo's own bit/s is a simulation under an assumed clock; both facts travel
  with the number wherever it is shown.

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

**Workings for the 5G-NIDD metric profile live in
[`5g-nidd-metric-profile.md`](5g-nidd-metric-profile.md).** Round 49 published `telecom_ids.nidd` with
accuracy 0.7608 against literature figures of 0.90 to 0.98 and attributed the gap to row-identifier
leakage in the published work. Round 50 measured the cause instead and it is one column: `IdleTime`,
which `Combined.csv` does not carry and the release's `BTS1_BTS2_fields_preserved.csv` does. Adding it
gives 0.982897 with every identifier and host fingerprint dropped, so the published range is reachable
without leakage of any kind. The leakage-aware prior work Round 49 did not consult is cited in that
file. The identifier finding itself stands - three integer counters move the narrow configuration to
0.9997 - but it does not explain anyone else's numbers.

This section was promised by the preamble of every earlier revision of this file and never written,
while `src/qsh/qml_ids/` and the Telecom/Defense use case shipped. The gap mattered: WS2 is the one
workstream whose honest answer is negative, and the reasoning lived only in a training script's
docstring.

- **Resisting Quantum Key Distribution Attacks Using Quantum Machine Learning** - Al-Kuwari et al.,
  *IET Quantum Communication* 7(1), January 2026, DOI 10.1049/qtc2.70028
  (arXiv:2509.14282 [cs.CR], v1 16 Sep 2025, v2 19 Feb 2026) `[link]`
  TITLE CORRECTED Round 55. This entry carried "A Quantum Machine Learning Approach for Intrusion
  Detection Systems", which is not the title on v1, on v2, or in IET - a plausible-sounding
  reconstruction from the subject matter rather than a transcription. It is also no longer a preprint.
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

- **Network Attack Traffic Detection With Hybrid Quantum-Enhanced Convolution Neural Network** -
  Wang, Fok and Thing, arXiv:2504.20436 [cs.CR], 29 Apr 2025 `[link]`
  URL: https://arxiv.org/abs/2504.20436
  Added: no PDF bundled; recorded 2026-08-15 with the 5G-NIDD licence withdrawal below
  **The published comparison point on 5G-NIDD**, which is this paper's sole dataset. On the standard
  split it reports classical CNN 99.61 percent, QCNN-AnE 99.62 percent, QCNN-AmE 99.65 percent,
  Quan-ConvCNN 99.71 percent and QuanvolutionNN 98.67 percent accuracy. Two things have to be said
  before that ordering is quoted as a quantum win. The models are **hybrid quantum-classical CNNs,
  not pure Cong-style QCNNs**, so the name does not mean here what it means in the QCNN section
  below; and the authors' own claimed advantage is on **generalisation to unseen attacks**, not on
  the standard split, where a 0.10-point spread over a 99.61 percent classical baseline is not a
  result to lean on. Recorded because `DATA_LICENSES.md` section 5 requires every published model
  card to name the baseline it is measured against, and until this entry there was no such baseline
  on record for 5G-NIDD - which made the corpus look, from inside this repository, like a dataset
  with no external comparison point rather than one with a 2025 quantum paper already on it.

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
takes a small fraction of the run the quantum pair needs - `wall_clock_s` in
`models/metrics.json` covers the whole script and has been measured at 267.1 s and 341.5 s on the
same machine, so the absolute seconds are load-dependent and only the order of magnitude is a
claim. `quantum_beats_classical` is recorded as `False` in `models/metrics.json`, which is the
field this project would have to falsify to claim otherwise.

That last sentence was briefly false. Round 33 re-ran `telecom_ids.py` **without** `--quantum` to
add the CKKS student, and `record_metrics` replaces a record rather than merging it, so all eight
`quantum_*` fields - including the `quantum_beats_classical` this paragraph points at - were
deleted from `models/metrics.json` while three published claims went on citing them. Neither the
config-parity gate nor the new prose gate could see it: the prose fields were untouched and both
derived artifacts stayed consistent with each other. Restored in Round 34 by re-running with
`--quantum`; the values came back identical, which is the only reason the citations above were
recoverable rather than merely re-asserted.
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
- **5G-NIDD was evaluated and rejected, and that rejection is WITHDRAWN.** The dataset is
  **CC BY 4.0**, `access_type: Open`, at the authors' repository of record - Fairdata/Metax, DOI
  10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361. The rejection had been read off IEEE DataPort,
  which is a mirror, marks the dataset subscription-required and carries no licence string; absence
  of a grant at a mirror was recorded as absence of a grant. The adjudication still lives in
  [`DATA_LICENSES.md`](../DATA_LICENSES.md), where 5G-NIDD now sits in section 1 as usable with
  **attribution required** - CC BY 4.0 obliges anyone publishing the data, or a model trained on it,
  to credit the dataset authors.

  The previous revision of this bullet ended: *"a licence finding with two homes is a licence finding
  that will be corrected in one of them."* It was right, and it undercounted. The finding had
  **three** homes - this file, `DATA_LICENSES.md` and `src/qsh/models/registry.py` - the withdrawal
  was written into a **fourth**, `docs/deliverable-verification.md`, and for a full round that fourth
  file was the only place the truth was recorded while three others went on calling the dataset
  rejected. The sentence warning about duplication was itself one of the copies that went stale.
  Duplication is not fixed by noticing it in prose; it is fixed by there being one owner, which is
  the rule stated immediately below and which this entry existed in violation of.

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
- Zhang et al., npj Quantum Information, 2026 `[link]`
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
  `[link]`
  URL: https://gopher.security/blog
  Added: 2026-06-27
  File: `Quantum-Safe_Multi-Party_Computation_for_Distributed_AI_Datasets....pdf`
  A vendor blog post, not peer-reviewed, and weighted accordingly. qOLE and qMPSI plus PQC (ML-KEM)
  hybrid transport plus an MCP delegate-and-verify narrative. Informs `src/qsh/pqc/transport.py` and
  the end-to-end orchestration framing.

---

## WS4 - QHE x AI (quantum/classical homomorphic encryption + AI)

- **Reverse Delegated Training and Private Inference via Perfectly-Secure Quantum Homomorphic
  Encryption** - Ortega and Martin-Delgado, arXiv:2602.12712, 2026 `[link]`
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
  Xiao (NUS and A*STAR), arXiv:2604.09975 [cs.CR], 11 Apr 2026 `[link]`
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
  Effectively Classically Simulable."** **PRX Quantum 7, 020304 (2026), DOI 10.1103/8qt9-72ts**;
  arXiv:2408.12739, 22 Aug 2024, revised 20 Feb 2026. Upgraded from the preprint citation on
  2026-08-13: the journal reference and DOI were read from the arXiv API's `journal_ref` and
  `arxiv:doi` fields, so this is now a peer-reviewed citation rather than a preprint one. Randomly
  initialised QCNNs can only operate on the information in **low-bodyness measurements** of their
  input states, and their behaviour is efficiently classically simulable by an algorithm equipped
  with Pauli shadows. The authors argue QCNN successes largely reflect benchmark simplicity, and that
  non-trivial datasets are a necessary ingredient for progress in QML. **Why it is cited here:** this
  repository's trained QCNN loses to logistic regression on the identical split (medical accuracy
  **0.883721** vs **0.953488**; finance minority-class F1 **0.735294** vs **0.865672**). This paper is
  the reason that is the expected result rather than a failure of implementation, and it is quoted in
  both QCNN model cards. The 8-class telecom circuit added in Round 34 lands the same way - macro F1
  **0.510963** against a multinomial logistic regression at **0.976447** on the identical split, with
  all 8 classes predicted - which matters because it is a *different* readout (a 3-qubit basis
  marginal rather than a single $`\langle Z_0 \rangle`$ expectation) and a different task shape, and it did not change
  the direction of the result. **The figures in this bullet were 0.7326, 0.7143 and 0.8098 until
  Round 47, and then 0.860465, 0.84058 and 0.458324 for part of Round 47 itself**: measured first on
  a hardware-efficient ansatz (replaced in Round 46), then through a multiclass readout that binned
  the class index over a wire pooling had discarded (corrected in Round 47, which also made the
  fully-connected block per-survivor), and then on a circuit carrying two classes of analytically
  dead parameter (reordered later in the same round). All six artifacts were retrained twice; the
  current values are read from `models/*/weights.json`, which each card is now gated against. **The
  direction never moved - the classical baseline wins on all three use cases before and after every
  one of those changes - and only its size did**, which is the only thing this citation was ever
  quoted for.
  Added: 2026-08-11; telecom result and journal reference added 2026-08-13; metrics refreshed after
  each of the two Round 47 retrains.
- **Bowles, Ahmed & Schuld, "Better than classical? The subtle art of benchmarking quantum machine
  learning models."** arXiv:2403.07059, 11 Mar 2024, revised 14 Mar 2024. **Preprint - the arXiv API
  reports no `journal_ref` and no DOI, checked 2026-08-13**, so cite it as a preprint rather than a
  paper. 12 quantum models across 6 binary tasks and 160 datasets. Two findings: out-of-the-box
  classical models outperform the quantum classifiers overall, and **"removing entanglement from a
  quantum model often results in as good or better performance, suggesting that 'quantumness' may not
  be the crucial ingredient"**. **Why it is cited here:** the second finding is a claim about
  circuits, and this project has one, so it was CHECKED rather than repeated. Measured over five
  seeds per binary model with the CNOT ring skipped and nothing else changed - median minority-class
  F1 falls 0.714286 to 0.048544 on finance (entangled wins 5 of 5) and 0.870968 to 0.833333 on
  medical (4 of 5); telecom's 8-class macro F1 falls 0.809846 to 0.467059 on a single seed. **So the
  tendency they report does not reproduce on this circuit** - which is a statement about this
  circuit, not a counterexample to a survey of 160 datasets. Full workings, including the one-seed
  reading that pointed the opposite way before the sweep:
  [`qcnn-entanglement-ablation.md`](qcnn-entanglement-ablation.md). Note this cuts AGAINST the
  entry below it only in mechanism, not in verdict: entanglement mattering to a model that still
  loses to logistic regression is not evidence of advantage.
  **THOSE FIVE FIGURES BELONG TO A CIRCUIT THIS PROJECT NO LONGER SHIPS**, and they have not been
  re-measured. They were taken on the hardware-efficient ansatz - RY on every wire plus one CNOT ring,
  which `--no-entangle` removed completely - that Round 46 replaced with a real QCNN and Round 47
  corrected TWICE. On the current circuit the flag is also **not an exact ablation**: it gates the
  conv filter's CNOT and the survivor ring in the fully-connected block, but the pooling stage is a
  controlled rotation, which entangles, and it is not gated. So the sentence "the tendency does not
  reproduce on this circuit" now names a circuit that no longer exists, and re-running it is open
  work rather than a result carried forward. See the supersession banner on
  [`qcnn-entanglement-ablation.md`](qcnn-entanglement-ablation.md).
  Added: 2026-08-13; superseded-circuit note added in Round 47.
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
- **Liu, Qinghua and Paparrizos, John, "The Elephant in the Room: Towards A Reliable Time-Series
  Anomaly Detection Benchmark" (TSB-AD), NeurIPS 2024 Datasets and Benchmarks Track.** The
  proceedings entry IS the citation - **TSB-AD has no arXiv ID at all**, so there is no preprint
  identifier to give. Cited here as the benchmark that motivates reporting event-wise F-beta and never
  point-adjusted F1 as a headline (`src/qsh/models/metrics.py`, `event_wise_fbeta`). The specific
  point-adjustment number this entry used to quote is withdrawn; the replacement is the Kim et al.
  entry below. Added: 2026-08-07, **corrected 2026-08-12**.

> **This entry cited a different paper and quoted a number that is in neither.** Until 2026-08-12 it
> read "Liu et al. ... **arXiv:2412.20512**". That arXiv ID belongs to **Boniol, Liu, Huang, Palpanas
> and Paparrizos, "Dive into Time-Series Anomaly Detection: A Decade Review"** - a survey, not TSB-AD.
> The two papers share authors, which is how the conflation happened, and it is the same failure mode
> as the OPS-SAT / ESA-ADB mix-up above: an overlapping author set and a surname that is not a
> discriminator. Here there is no identifier to swap in, because TSB-AD never had one; an entry that
> carries an arXiv ID for TSB-AD is wrong by construction.
>
> **The "uniform random score ranks 26th of 32 detectors under point adjustment" claim is WITHDRAWN as
> unverified.** It could not be found in TSB-AD or anywhere else, and TSB-AD's own proceedings abstract
> does not fit its shape: that abstract evaluates **40 detection algorithms**, not 32. The claim was
> load-bearing - it was the whole stated reason for a metric choice - so it is replaced by a verbatim
> one rather than softened. Verified 2026-08-12.
>
> **The withdrawn number is quoted in seven places outside this file and they are NOT yet fixed.**
> Listed so the cleanup is a task rather than a discovery, because a correction that lands in the
> provenance file only is the weakest possible version of it: `README.md:823`;
> `report/technical-report.tex:201` and `:636`; `scripts/train/satellite_telemanom.py:10` and `:308`
> (which is what writes the string into `models/metrics.json:273`, whose `metric_note` therefore carries
> it, and into `web/src/lib/config.defaults.json:699` via the generated config); and
> `src/qsh/models/metrics.py:19`, which is the worst of them because it adds an inference the source
> never supported - "i.e. it beats six real detectors". Those files were owned by another worker while
> this correction was made, so they were deliberately left alone. **The replacement text for all seven
> is the Kim et al. entry below**, and the substitution is not merely a swap of citations: the Kim
> sentences are about a random and an untrained SCORE, so any rewrite that keeps the "beats six real
> detectors" gloss reintroduces the fabrication.
>
> **TSB-AD recommends VUS-PR and this repository does not use it.** The same abstract identifies
> **VUS-PR as the most reliable measure**. Since TSB-AD is cited here as the authority for rejecting
> point-adjusted F1, not adopting the measure it actually recommends needs a stated reason rather than
> silence: event-wise F-beta is what the satellite baselines this project is scored against are
> computed with (ESA-ADB Supplementary Table 9), so switching measures would break comparability with
> the published number on the page. That is a defensible choice made AGAINST the recommendation of the
> cited source, and a reviewer will ask - so it is recorded here as an open trade-off, not as agreement.

- **Kim, Choi, Choi, Lee and Yoon, "Towards a Rigorous Evaluation of Time-series Anomaly Detection",
  AAAI 2022.** arXiv:2109.05257 `[link]`
  URL: https://arxiv.org/abs/2109.05257
  **The load-bearing citation for rejecting point-adjusted F1**, in place of the withdrawn TSB-AD number
  above, and stronger: verbatim, *"even a random anomaly score can easily turn into a state-of-the-art
  TAD method"*, and separately, *"an untrained model obtains comparable detection performance to the
  existing methods even when PA is forbidden."* The first sentence is the point-adjustment failure
  itself; the second is the harder finding, because it holds with point adjustment switched **OFF** - so
  the missing control in this field is an untrained baseline, not a better measure.
  **This paper was already cited in code and never in this file.** `src/qsh/models/metrics.py:17`
  quotes it as "Kim et al. (AAAI-22)" for a third sentence, that point adjustment *"has a great
  possibility of overestimating the detection performance"* - a quote this round did not re-verify, so
  it is recorded as the code's, while the two sentences above were verified at the arXiv abstract page
  on 2026-08-12. What the repository does with it: `metrics.py` reports **event-wise F-beta with its
  event count as the headline** and computes `point_adjusted_f1` **only as a companion, to show the
  inflation** - on the satellite run, `point_adjusted_f1` 0.993691 against `event_wise_f0_5` 0.454545
  (`models/metrics.json`), which is the whole argument in two numbers from one run. That is
  the practice both papers argue for, and it does not change with this correction; only the citation
  supporting it does. Added: 2026-08-12.
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
- **Al Badawi, Alexandru, Polyakov and Vaikuntanathan, "SoK: Private LLM Inference using Approximate
  Homomorphic Encryption", IACR ePrint 2026/935**, revised 2026-06-03. Quoted in `models/distil.py` on
  why distillation is the practical route to FHE-feasible inference. Added: 2026-08-07, corrected
  2026-08-12. This entry also named `models/registry.py` until Round 33: Round 31 replaced that
  citation with the primary source (Park et al., arXiv:2601.18511v2), so the SoK has not appeared in
  the registry since, and two other entries in this file repeated the same stale attribution.
  **Extended 2026-08-12** with the half of it that bears on how this repository REPORTS rather than on
  what things cost (verified at the abstract, and extended here rather than filed twice): the SoK
  proposes a **"Private LLM Card System (PLCS)"** to standardise reporting because *"Frameworks differ
  in ciphertext packing layouts, model fidelity, software and hardware stacks, and reported metrics"*;
  it finds only **20% of surveyed implementations model-preserving**; and it puts the runtime gap at
  roughly **four orders of magnitude**. The 20% is the figure that matters most here, because the
  plaintext-versus-CKKS agreement this demo prints IS a model-fidelity claim, and this is the survey
  saying most published work does not make one.
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
  **The game conditions, verbatim, because they are weaker than readers assume** (verified at the
  abstract 2026-08-12): the adversary gets *"access to a decryption oracle for ciphertexts for which it
  should know the underlying plaintexts. This includes ciphertexts that it (honestly) encrypted and
  those obtained from the latter by evaluating circuits that it chose."* So **honestly generated inputs
  suffice and the attack is PASSIVE** - "the parties are only semi-honest" is not a defence against
  CPAD, which is the misreading this entry now exists to block.
- Checri, Clet, Renard, Sirdey, "Impossibility of CPAD security for a class of FHE schemes",
  ePrint 2026/203. No HELLHO scheme - covering basic BFV, BGV, CKKS and TFHE - can be CPAD secure.
  Practical key recovery on DE-CKKS, supported in OpenFHE, in hours on a laptop.
- Santato, Zanotto, "How (not) to Switch FHE Schemes", ePrint 2026/285. Breaks PEGASUS-style
  CKKS-to-FHEW switching in OpenFHE. Not applicable here - this project does no scheme switching -
  and recorded so that stays a checked fact rather than an assumption.
- Ogilvie, "IND-CPA-D and KR-D Security With Reduced Noise from the HintLWE Problem", ASIACRYPT 2025,
  ePrint 2025/1618. The constructive counterpoint, now with its title recorded and the claim verbatim
  (verified at the abstract 2026-08-12): *"an additional precision loss of as little as two bits is
  sufficient to restore a high level of security against passive key-recovery attacks for standard
  parameters."* Two bits is the number to quote; "about two bits of extra precision loss", which this
  entry said before, was a paraphrase of a verbatim sentence and there is no reason to paraphrase it.

**The implementations' own security documentation, added 2026-08-12.** This section cited six papers
and **not one of the three CKKS implementations**, while the demo ships one of them: `node-seal` IS
Microsoft SEAL compiled to WASM (`web/src/lib/clientFHE.ts`). Upstream's security statement for the
cipher on the page outranks any secondary summary of it. Full discussion in
[`../docs/threat-model.md`](../docs/threat-model.md); the load-bearing predicate is **crossing a trust
boundary**, not displaying a value.

- **Microsoft SEAL, `SECURITY.md`** `[link]`
  URL: https://github.com/microsoft/SEAL/blob/main/SECURITY.md
  Verified 2026-08-12. Verbatim: *"The CKKS scheme requires particular care because it is approximate:
  decryption returns the message plus a secret-key-dependent error term rather than the exact
  message"*; *"Releasing such decryptions to any party other than the secret key owner can therefore
  leak the secret key, even when only ordinary computation results are shared"*; *"Formally, the CKKS
  implementation in Microsoft SEAL does not provide IND-CPA-D security ...; it targets only IND-CPA
  security"*; *"Any mitigation, such as adding noise to decrypted values before they are shared (noise
  flooding), must be implemented by the application before a decrypted CKKS value crosses a trust
  boundary."* Bears on `web/src/lib/clientFHE.ts` and on the threat model's CPAD section. Note what the
  last sentence implies for this deployment: **SEAL provides no noise flooding**, so there is no
  parameter to set and the gap is upstream rather than declined.
- **IBM HElib, `CKKS-security.md`** `[link]`
  URL: https://github.com/homenc/HElib/blob/master/CKKS-security.md
  Verified 2026-08-12. The mechanism in one sentence: *"since the scheme itself adds some error, the
  attacker could learn something from the decryption result even if it knows what the decrypted result
  was supposed to be. Specifically, it learns the error, which may leak information about the secret
  key."* The query count Li and Micciancio is usually credited with is here rather than in that paper's
  reachable text: *"sometimes as few as a single decryption"*. And the cost of the fix: *"to withstand
  an attack that has access to the results from decryption of D ciphertexts, the bits parameter should
  be increased roughly by a factor of sqrt(D)"*. HElib implements noise flooding **inside decryption**
  with a settable accuracy target, which is the contrast that makes SEAL's position specific.
- **OpenFHE security documentation** `[link]`
  URL: https://openfhe.org/ - the security documentation on the project site; the deep-link path is
  deliberately not recorded, because only the text below was captured and a guessed path would be worse
  than a search for "decryption queries" from the landing page.
  Verified 2026-08-12. Verbatim: *"By default, OpenFHE chooses a configuration where a relatively large
  number of decryption queries of the same or related ciphertexts can be tolerated. The lower bound for
  the tolerated number of such decryption queries is N_d = 128 (but in practical computations, N_d is
  at least 10,000)."* It also scopes the threat: the attack *"applies to the setting where decryption
  results need to be shared between multiple parties, e.g., in the threshold FHE setting."* OpenFHE
  offers `NOISE_FLOODING_DECRYPT` and `CKKS_M_FACTOR`; it is the backend `qsh/fhe/backends.py` uses on
  the server side, so this is the one place in the project where the mitigation is configurable at all.
- **Guo, Nabokov, Suvanto, Johansson, "Key Recovery Attacks on Approximate Homomorphic Encryption with
  Non-Worst-Case Noise Flooding Countermeasures", USENIX Security 2024** `[link]`
  URL: https://portal.research.lu.se/ - search the title; the record path is not recorded here because
  the publication page, not a guessed permalink, is what was loaded.
  **Title, authors and venue VERIFIED** via the Lund University research portal, 2026-08-12. **Body NOT
  loadable (HTTP 403), so no parameter, bound or number from it is quoted anywhere in this
  repository.** Recorded because it attacks the very countermeasure the CPAD discussion presents as the
  fix: noise flooding parameterised on average-case rather than worst-case noise. A section that names
  noise flooding as the remedy and omits this reads as unaware of it.
- Bossuat, Costache, Mouchet, Nurnberger, Troncoso-Pastoriza, ePrint 2024/853, IACR CiC 2025, DOI
  10.62056/a3n5tx4e-. NOTE the retitle at the 2025-08-11 revision: originally "Practical
  q-IND-CPA-D-Secure Approximate Homomorphic Encryption", now "Accurate and Composable Noise
  Estimates for CKKS with Application to Exact HE Computation". Citing the old title points at a
  paper that no longer exists under that name.

## QSDC security basis: secrecy capacity, not a fixed QBER (Round 24)

**This entry corrects an earlier version of itself.** That version said `qkd.pp_threshold = 0.11`
was correct for QSDC because the wiretap secrecy capacity $`C_s = 1 - 2h(e)`$ has its root at 0.110028,
the same as Shor-Preskill, so the two coincided by identity. The identity is real but the conclusion
was wrong, because $`C_s = 1 - 2h(e)`$ is the **loss-free** case.

The published criterion is $`C_s > 0`$, and $`C_s`$ depends on channel loss:

$$
C_s \;=\; Q_{\text{Bob}}\bigl[\,1 - h(e) - g\,h(e_{\text{chk}})\bigr],
\qquad
g \;=\; \frac{Q_{\text{Eve}}}{Q_{\text{Bob}}} \;=\; 10^{\,\text{loss\_dB}/10}
$$

So the tolerable error rate falls fast as the link gets lossier: 0.110 at 0 dB, **0.048 at the 4.1 dB
Qi et al. measured on a real 1.5 km fibre link ($`g = 2.57`$)**, and ~0.0003 at the ~24.5 dB this
project's own free-space downlink reports. Applying 0.11 to a lossy channel is wrong in the
**permissive** direction - it delivers over a channel whose secrecy capacity is already zero. The
satellite path now computes $`C_s`$ directly.

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
All ETSI QKD deliverables then reachable were searched full-text (**"14 published" is one too many, corrected Round 55**: the portal shows **13 published**, plus GS QKD 003 V1.1.1 which is WITHDRAWN, and GS QKD 001 and 006 Stopped - 14 only if the withdrawn one is counted) (2.0 M characters): zero hits for
QSDC or "secure direct communication", against a control of 7 documents matching "BB84". ITU-T's
complete X.1700-X.1749 and Y.3800-**Y.3838** sets are QKD or QRNG throughout (this said Y.3835 until Round 51; Y.3836, Y.3837 and Y.3838 were approved 2026-08-13, Y.3836 still flagged
prepublished), and the work programme
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
- the $`C_s > 0`$ abort rule and the measured $`g = 2.57`$. Pan, Lin, Wu, Sun, Ruan, Yin, Long,
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
  IACR eprint 2026/1559, received 2026-07-30. Reduces the ThFHE ciphertext modulus from $`O(N)`$ to
  $`O(t \log(N/t^2))`$ bits, with every algorithm completing under 0.2 s at $`N = 64`$ and scaling past $`N \ge 256`$.
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

- **node-seal, `BENCHMARK.md`** (library benchmark page, last updated 2025-03-27).
  https://github.com/s0l0ist/node-seal/blob/main/BENCHMARK.md - fetched and read 2026-08-14.
  Microsoft SEAL v4.1.2 / node-seal v5.1.5 on an Apple M1 Pro; Chrome 134, Firefox 136, Safari 18.3.
  CKKS `EvaluateRotate` at `n=8192/log(q)=218` reads **6830 us in Chrome**; at `n=16384/log(q)=438`,
  **41200 us**. Ten iterations per benchmark, no error bars, timers quantised to 10 us (Chrome) and
  100 us (Firefox/Safari).

  **The only primary source found for browser-WASM CKKS per-primitive cost**, and the comparator this
  project's 3.10-4.39 ms per rotation is read against in
  [`encrypted-inference-performance.md`](encrypted-inference-performance.md). The `n=8192` row is
  comparable (same ring, similar modulus, both Chrome/WASM); the `n=16384` row is **not** - 438 bits
  against 200 here - and is recorded only so the parameter difference is visible.

- **Ali Sah Ozcan and Erkay Savas**, *GPU Implementations of Three Different Key-Switching Methods
  for Homomorphic Encryption Schemes*, IACR ePrint **2025/124** (November 2024).
  https://eprint.iacr.org/2025/124.pdf - PDF downloaded and text-extracted 2026-08-14.
  Table 5, CKKS relinearization: single-thread Microsoft SEAL on a Ryzen 9 7950X3D gives **1641 us**
  at ring 8192 and **14563 us** at 16384; HEonGPU on an RTX 4090 gives **41.8 us** and **84.1 us**.
  Measures relinearization, not rotation - both are key-switching, and node-seal's own table puts the
  two within 1% at both ring degrees, so it is a sound proxy but is cited as relinearization.

  **Recorded here because Round 38 introduced both sources and this file listed neither** - the same
  omission it records above for the Nature Communications figure, repeated by the round that read
  that entry.

  Four metadata-only "direction of travel" entries stood here - arXiv:2604.04783, 2602.11470,
  2512.11269, 2602.22229 - labelled "abstracts not individually opened, so no number from these four
  should be quoted". `docs/deliverable-verification.md` gave them a deadline: promoted by quoting a
  number with its section, or removed by Round 35. **Removed in Round 39, three rounds after the
  deadline the register set and never executed.** None was ever cited, so nothing depends on them;
  the arXiv IDs are preserved in this sentence for anyone who wants to re-open the search.

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

### Sweep of 2026-08-01 to 2026-08-14, run 2026-08-14

Thirty searches across seven areas; nothing found changes a claim this repository makes. The window is
thirteen days wide, so a null was the honest expectation - it is recorded with **what was searched**
rather than as a bare "nothing found", because an unrecorded null is indistinguishable from a search
nobody ran, and this project has twice re-derived a negative it had already established.

- **NIST SP 800-227: no change, checked at the primary source.** The CSRC document history reads
  exactly `01/07/25: SP 800-227 (Draft)` and `09/18/25: SP 800-227 (Final)` and nothing else - no
  errata, no update, no withdrawal, no follow-on. Section 4.6.3's IND-CCA-preserving combiner, which
  `src/qsh/pqc/hybrid_kem.py` implements, is unchanged.
- **FIPS 203 / 204 / 205: no change in window.** All three still show only `08/24/23 Draft` and
  `08/13/24 Final`. FIPS 204 has an errata spreadsheet dated **2026-07-31**, one day outside the
  window; it concerns the signature standard rather than the KEM. `csrc.nist.gov/news` carries two
  August 2026 items, neither PQC.
- **CKKS and homomorphic encryption: nothing that touches a claim here.** Nine in-window papers on the
  "homomorphic" query; the two that could plausibly bear on performance do not - one is a survey with
  no rotation timings and no WASM, the other is BGV matrix multiplication rather than CKKS rotations.
  The strong bootstrapping results that surface for "August 2026" (BootNet, "Bootstrapping is All You
  Need") are June 2026 and out of window. Note the asymmetry: **the 3.10-4.39 ms per rotation figure
  is a measurement of this stack, and no paper can invalidate a measurement** - only a comparative
  claim, and the one comparison made is already hedged as "the same order, this host faster".
- **QKD: one in-window record, not applicable.** arXiv:2608.04112 reports a secure-key-rate-per-pulse
  result in a four-dimensional fibre lab demonstration. This repository scopes hardware QKD out
  explicitly and validates its simulation against Ma et al. (2005). No new ETSI or ITU-T QKD standard
  in the window.
- **QCNN: null.** In-window hits are applications (fault classification, a CNN-QNN hybrid image
  classifier); no theory or benchmark result bearing on the QCNN claims made here.
- **MPC-FHE / QSMPC-FHE hybrids: null in window** for anything altering the model-strategy verdict.
- **ML-KEM and hybrid-combiner cryptanalysis: null**, with one item put on watch below.

**WATCH, not a change: ePrint 2026/1591, a claimed polynomial-time quantum algorithm for the Dihedral
Coset Problem.** https://eprint.iacr.org/2026/1591 - Daniel R. Simon (AWS Cryptography Group),
received 2026-08-03, three revisions to 2026-08-12, fetched and read as PDF 2026-08-14. Composed with
Regev's and Brakerski-Kirshanova-Stehle-Wen's reductions it would give polynomial-time quantum
algorithms for SVP and LWE, which is why it is here at all.

*Why it changes nothing today, from the document itself rather than from coverage of it:* the title
carries "[Preliminary Draft]"; four proofs are marked "(Sketch)"; the load-bearing final corollary
cites two references both listed as "Personal Communication", i.e. unpublished; the abstract carries
a self-correction noting that the first draft's Lemma 3 "incorrectly conflated pairwise independence
over the uniform vs. actual distribution". A full-text grep for `ml-kem|kyber|fips|dilithium|concrete
parameter|deployed` returns **zero hits** - the paper makes no claim against any deployed parameter
set. It is unrefereed and actively being revised.

*The honest nuance, which is the reason to record it rather than dismiss it:* **this repository's
ML-KEM + X25519 hybrid is not a hedge against this particular risk.** X25519 already falls to Shor,
so ML-KEM is the only quantum-resistant leg; if this result were confirmed AND shown to reach
ML-KEM's concrete parameters, the hybrid would not preserve quantum security. Neither condition holds
today. *Reopens if* the paper is refereed and published, or a version removes the sketch markers and
the personal-communication dependencies, or any analysis extends it to concrete ML-KEM parameters.

A related result surfaced while chasing this one and is recorded to stop it being re-found as news:
**ePrint 2026/155**, Wen and Zheng, Module-LWE and structured extrapolated dihedral cosets (CRYPTO
2026), proves a quantum polynomial-time *equivalence* in the setting ML-KEM uses. Original 2026-01-30,
last revised 2026-07-16 - **both before this window** - and it claims no attack and no weakening of
ML-KEM. Secondary coverage was pairing the two as though both were current.

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

## Round 32 additions, verified at primary source 2026-08-11

Same rule as the Round 31 section: every entry below was loaded at its arXiv abstract page, IACR eprint
listing, PubMed record or agency PDF before being written here. Nothing came from a secondary write-up.
Where a source could not be loaded it is marked and no number is taken from it.

### Homomorphic inference: the cost driver, and a design that contests this repository's split

- **Xie, Tan, Boddeti and Lu, "ATLAS: Automated Approximation of Transformers for Efficient Homomorphic
  Inference in One Hour."** arXiv:2607.23478, 2026-07-26 `[link]`
  URL: https://arxiv.org/abs/2607.23478
  Names **multiplicative depth as the dominant cost** of homomorphic transformer inference, with a
  per-layer configuration space reaching ~10^84 for a 12-layer BERT/ViT and 10^225 for 32-layer LLaMA3,
  and 35-50% of candidate configurations numerically invalid. This is a **primary source for the
  constraint `models/registry.py` asserts** in its T1 docstring, and Round 33 cited it there.
  This entry said the docstring "until now rested on a secondary systematisation (IACR 2026/935) plus
  Park et al.", which was wrong in both halves: Round 31 had already removed the 2026/935 citation
  from that file, and adding this reference in Round 32 did not put ATLAS into the docstring either -
  so a reference was recorded as the source for a code site it never touched. Contested by
  **arXiv:2608.00375** (Trimming) below, which carries no experiments.
- **Lin and Wu (NTU), "Fine-Grained and Runtime-Configurable Precision for Exact FHE Inference"
  (Tailor).** IACR ePrint 2026/1551, received 2026-07-29, approved 2026-08-03 `[link]`
  URL: https://eprint.iacr.org/2026/1551
  Per-layer precision becomes a runtime parameter under a single keygen; **ReLU and comparisons are
  evaluated exactly, not approximated**; TFHE and FINAL backends; MNIST at 2-bit output / 4-bit elsewhere
  reaches 96.44% in 222 s against uniform 4-bit's 96.17% in 236.1 s.
  **This is the most architecturally consequential item in the batch and it is carried as a named
  alternative, not a refinement.** This repository splits linear layers to FHE and non-linear layers to
  MPC *because* non-linearities are expensive under CKKS. Tailor evaluates them exactly under FHE with no
  MPC round at all. That does not refute the split - the demo's constraint is a browser tab, and Tailor's
  222 s is a server measurement on TFHE rather than CKKS - but "CKKS cannot do non-linearities cheaply"
  is no longer the whole story, and `docs/architecture.md` should name the alternative rather than imply
  none exists.
- **Gong, Tian, Zhang and Liu, "Verifiable and Collusion-Resistant Multi-Party Quantum Private Set
  Operations."** IACR ePrint 2026/1436, received 2026-07-14 `[link]`
  URL: https://eprint.iacr.org/2026/1436
  Combines **verifiable quantum FHE as the outsourced-evaluation layer with threshold FHE as the
  key-management mechanism**, instantiated on IBM Quantum Platform, extended to private set union, and
  resilient to collusion between the third party and participants. This is the QSMPC x QHE composition
  this repository is named after, in one protocol: `qhe/rate1_qfhe.py` had a quantum anchor
  (Chardouvelis-Doettling-Malavolta) and Round 31 added a classical ThFHE anchor (2026/1559) separately.
  Note Fengxia Liu is also first author of arXiv:2604.23490, already cited.

### QKD security proofs and hardware, receiver side

- **Navarrete, Pereira, Curras-Lorenzo, Mizutani, Curty and Tamaki, "Phase-error estimation for quantum
  key distribution with leaky receivers."** arXiv:2608.09674, 2026-08-10 `[link]`
  URL: https://arxiv.org/abs/2608.09674
  Finite-key phase-error framework carrying leakage about **both basis choice and measurement outcome**,
  requiring only a bound on side-channel state distinguishability, composable with existing
  hardware-imperfection proofs, motivated by detector backflashes and back-reflected Trojan-horse light.
  **The receiver-side counterpart to Round 31's transmitter-side arXiv:2606.29943**, published the same
  day as the already-cited 2608.09793 by an overlapping author set. Directly relevant to the
  `trojan_horse` and `wavelength_trojan` classes in `models/metrics.json`, which simulate receiver
  leakage and cite no security proof for it.
- **Konteli, Makris, Anastasiou, Tsimvrakidis and Kanellos, "Quantum-Level Crosstalk Characterization of
  a 16x16 MEMS Optical Switch for Dynamic Quantum Communications."** arXiv:2607.24299, 2026-07-27 `[link]`
  URL: https://arxiv.org/abs/2607.24299
  All-to-all quantum-level crosstalk measured with SNSPDs on a **commercial** 16x16 MEMS switch, with the
  theoretical impact expressed directly in **decoy-state BB84** terms. `qkd/sdn_controller.py` models
  ETSI GS QKD 015 `PHYS`/`VIRT` links and resource allocation with **no crosstalk term**; this is the
  measured physical-layer penalty a switched multi-user QKD network pays.

### Browser-side inference cost

- **Maczan, "Measuring and Reducing WebGPU Dispatch Overhead for LLM Inference."** arXiv:2608.08730,
  2026-08-09; accepted to the Fourth UK AI Conference 2026, to appear in PMLR `[link]`
  URL: https://arxiv.org/abs/2608.08730
  Sequential-dispatch measurement method; per-dispatch cost is independent of dtype; **at batch size 1
  the limit is dispatch overhead, not kernel quality**. A citable reason why in-tab inference is
  latency-bound in a way no smaller model or better graph fixes, for `web/src/lib/onnxInference.ts` and
  the T3 tier.

### Homomorphic encryption applied to polygenic risk scores - a gap, not an addition

`references/REFERENCES.md` had **no HE-genomics entry at all**, while the genomics use case is the one
whose model is FHE-native by construction. Both entries below also **falsify a claim this repository
makes**, in the direction of understating what is possible.

- **Knight, Li, Jensen, Yolou, Kockan and Gerstein, "Homomorphic encryption enables privacy preserving
  polygenic risk scores."** *Cell Reports Methods* 6(1):101271, published 2026-01-26 (epub 2026-01-08),
  DOI 10.1016/j.crmeth.2025.101271, PMID 41512862 `[link]`
  URL: https://doi.org/10.1016/j.crmeth.2025.101271
  Open-source HEPRS; three-party model (clinician / model developer / evaluator); tested on a
  schizophrenia model with **110,000 SNPs**; encrypted and plaintext PRSs "agree closely"; **"practical
  to use on a single CPU"**.
  **This is the citation that scopes the registry's T1 argument.** `genomics.prs` is 313 variants, and
  `registry.py` explains T1 smallness by homomorphic cost. For this use case that is the wrong reason:
  HEPRS runs roughly 350x more variants on one CPU, so the score is small because **PGS000004 has 313
  variants** - a property of Mavaddat et al. 2018, not of CKKS. The cost argument still holds for
  finance, medical and satellite, where the model is not a bare dot product.
- **Provatas, Galanopoulos and Georgakopoulos-Soares, "bioETH-PRS: Confidential Polygenic Risk Scoring
  without a Trusted Evaluator via Fully Homomorphic Encryption on a Programmable Blockchain."**
  arXiv:2605.21634 [q-bio.GN], 2026-05-20 `[link]`
  URL: https://arxiv.org/abs/2605.21634
  Integer-exact **TFHE** on an fhEVM, with a three-step fixed-point quantisation for representing signed
  GWAS weights as unsigned 64-bit integers at machine-epsilon reconstruction accuracy. A 2026 primary
  source for the signed-weight encoding problem `models/genomics_prs/weights.json` already solves, and an
  argument for why the evaluator role is worth separating.

### Medical: latent-space encoding, and a note this repository overstates

- **Kim, Jo, Ra and Park, "Privacy-Preserving Chest X-ray Classification in Latent Space with
  Homomorphically Encrypted Neural Inference."** arXiv:2506.15258, v1 2025-06-18, v2 2025-06-20 `[link]`
  URL: https://arxiv.org/abs/2506.15258
  VQGAN compresses images to latents at an optimum **downsampling factor of eight**; activations replaced
  with lower-degree polynomials; squeeze-and-excitation adapted to HE; two chest X-ray datasets.
  `models/metrics.json` calls `medical_fl.student`'s 3,226 parameters over a 14x14 input "a
  CKKS-feasibility budget". After this paper that is **a choice of input representation**, not a
  feasibility limit - and this repository is unusually well placed for the alternative, because the
  client is a browser tab already running `onnxruntime-web`, so an encoder could run locally in plaintext
  with no privacy loss at all. Recorded as the **sixth** falsifiable trigger in the additional-model
  decision (`docs/deliverable-verification.md`) rather than acted on. Corrected 2026-08-12: this entry
  and `models/metrics.json` both said the trigger was recorded while the table held five rows and none
  was this one, and the decision itself lived only in an untracked file, so the appeal could not be
  checked from the repository at all. The trigger is now written and the decision is tracked. Its
  condition is a measurement rather than a direction: the encoder's latent must fit 4096 CKKS slots at
  multiplicative depth 1 and run under `onnxruntime-web`. Note the paper's f=8 on a 28x28 input gives
  3.5x3.5, so the transfer is exactly what is unmeasured.

### Standards: nothing newer, and one uncited instrument

- **No PQC migration instrument newer than OMB M-26-15** (2026-06-24). The Federal Register API, term
  "post-quantum cryptography", publication date >= 2026-06-25, returns three documents; the only one
  after M-26-15 is **FCC 2026-15600**, 91 FR 48320, published 2026-07-31, whose full text mentions
  post-quantum cryptography **once**, as a question for comment: "should we account for that risk
  through post-quantum cryptography?" No date, no timeline - not a migration instrument.
- **NIST IR 8547 re-verified still at Initial Public Draft** (published 2024-11-12, comments closed
  2025-01-10, no second draft, no final). `docs/pqc-roadmap.md` holds. FIPS 206 IPD and an HQC draft:
  **NOT FOUND** - no June/July/August 2026 item for either on csrc.nist.gov/News.
- **A second quantum executive order was published the same day as EO 14412 and is cited nowhere here.**
  "Ushering in the Next Frontier of Quantum Innovation", Federal Register document 2026-12910,
  **91 FR 38487**, published 2026-06-25. EO 14412 is 2026-12909 at 91 FR 38483. Recorded because a
  roadmap that cites one of two same-day quantum EOs looks like it missed one.

### Negative results, re-verified 2026-08-11

- **No in-browser FHE or MPC has been published, third consecutive round.** An IACR full-text search for
  "WebAssembly browser" returns 3 results across all years, the newest crypto-relevant one still
  **2025/487 webSPDZ**. An arXiv sweep of (WebAssembly OR "in the browser" OR browser-based) against
  (homomorphic OR "secure multiparty" OR garbled OR "private inference"), newest first, returns nothing
  in-browser; a separate (WebGPU OR WebAssembly) sweep across cs.CR/cs.LG/cs.DC finds nothing evaluating
  ciphertext in a tab. The composition novelty claim rests on this and it continues to hold.
- **No encrypted-inference work specific to AML or to satellite telemetry.** Searches for (homomorphic OR
  secure inference OR privacy-preserving) against (money laundering OR fraud detection) surface only
  federated work in 2026; the same against (satellite OR telemetry OR spacecraft) surfaces no encrypted
  telemetry anomaly detection at all. So the encrypted tiers on those two pages are demonstrations rather
  than reproductions of anyone's deployment, which is what they should be described as.

## Round 33 additions, verified at primary source 2026-08-12

Same rule as Rounds 31 and 32: every entry was loaded at its arXiv abstract page, IACR eprint listing,
JMLR or PNAS record, Europe PMC record, vendor documentation page or repository file before being
written here. Where a source could not be loaded it is marked **METADATA-ONLY** or **NOT FOUND** and no
number is taken from it. Nothing in this section was promoted from metadata to verified.

Three items in this batch were **extended in place rather than filed again** - the CPAD section's Cheon
et al. (2024/127) and Ogilvie (2025/1618) entries, and the "SoK: Private LLM Inference" entry in the
baselines section - because a second entry for the same source is how this file drifts.

### Reporting: calibration, uncertainty and decision curves, which this file cited nowhere

A grep across the repository for `Vickers`, `decision curve`, `net benefit` and `calibrat` returned
**zero citations** while `src/qsh/models/metrics.py` implements net benefit, integrates it over a
threshold range, and selects an operating point by it. The method was named in a docstring and credited
to no publication.

- **Vickers and Elkin, "Decision curve analysis: a novel method for evaluating prediction models",
  *Medical Decision Making* 2006;26(6):565-574.** DOI 10.1177/0272989x06295361, PMID 17099194 `[link]`
  URL: https://doi.org/10.1177/0272989x06295361
  VERIFIED via Europe PMC, 2026-08-12. The source of the method `metrics.py::net_benefit` implements:
  its docstring says "Vickers-Elkin net benefit" and carried no citation, and this file had no entry.
  The implemented form `NB(p) = TP/N - (FP/N) * p/(1-p)` is **algebraically identical** to the published
  `sensitivity x prevalence - (1 - specificity) x (1 - prevalence) x w` with `w = p/(1-p)`, since
  `sensitivity x prevalence = TP/N` and `(1 - specificity) x (1 - prevalence) = FP/N` - a reader can
  check the identity without the PDF, which is why it is written out.
  **The load-bearing point is what the 2006 contribution actually was:** net benefit plotted **across**
  threshold probabilities. A single-threshold scalar is the thing decision curve analysis was invented
  to replace, so quoting one net-benefit number as a headline inverts the paper. This repository is on
  the right side of that by construction - `area_under_net_benefit` integrates over a stated range and
  `best_threshold_by_net_benefit` reports its margin over treat-all at the SAME threshold - and the
  citation makes that deliberate rather than accidental.
- **Vickers, van Calster and Steyerberg, "A simple, step-by-step guide to interpreting decision curve
  analysis", *Diagnostic and Prognostic Research* 3:18 (2019).** PMC6777022 `[link]`
  URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6777022/
  VERIFIED via Europe PMC, 2026-08-12. The practical companion: it gives the sensitivity/specificity
  form of the formula above and names **treat-all and treat-none as the standard reference lines** -
  which is exactly the pair `best_threshold_by_net_benefit` returns, so the reference lines on the page
  are the field's, not this project's invention.
- **Dimitriadis, Gneiting and Jordan, "Evaluating probabilistic classifiers: Reliability diagrams and
  score decompositions revisited", *PNAS* 118 (2021) e2016191118.** arXiv:2008.03033 `[link]`
  URL: https://arxiv.org/abs/2008.03033
  VERIFIED 2026-08-12. **CORP = Consistent, Optimally binned, Reproducible**, obtained by isotonic
  regression via the pool-adjacent-violators algorithm rather than by choosing a bin count. The
  citation to reach for **if** a reliability diagram is ever added; see the Riley entry below for why
  none is added now.
- **Roelofs, Cain, Shlens and Mozer, "Mitigating Bias in Calibration Error Estimation", AISTATS 2022.**
  arXiv:2012.08668 `[link]`
  URL: https://arxiv.org/abs/2012.08668
  VERIFIED 2026-08-12. Verbatim: *"binning-based estimators with bins of equal mass (number of
  instances) have lower bias than estimators with bins of equal width."* So an equal-width ECE, which
  is the default in most tutorials, is the biased choice - a one-line design constraint on any future
  calibration reporting here.
- **Riley et al., "Minimum sample size for external validation of a clinical prediction model with a
  binary outcome", *Statistics in Medicine* 2021.** DOI 10.1002/sim.9025 `[link]`
  URL: https://doi.org/10.1002/sim.9025
  **METADATA-ONLY.** The Wiley full text was not loaded; the thresholds below are corroborated across
  secondary sources only and are recorded at that strength deliberately: **at least 100 events and 100
  non-events** for a performance estimate, and **at least 200 events** for a flexible calibration curve.
  **This is the citation that rules OUT reliability diagrams on every page of this demo**, and the
  measured counts are attributable rather than asserted: medical **42** malignant cases in the scored
  split (`src/qsh/models/distil.py:194`), finance **164** positives in the browser's evaluation sample
  (`scripts/prep_model_inputs.py:125`), satellite **113** anomalous segments (`anomaly_rate_test`
  0.213611 x `n_test_segments` 529, `models/satellite_opssat/README.md`), and genomics **no outcome
  label at all** - 1000 Genomes carries no case/control status, which is why the PRS AUROC there is
  cited as external and never computed. So **no page clears the 200-event bar**, medical does not clear
  even the 100-event bar, and a smooth calibration curve drawn on any of them would be a picture of
  noise. The satellite figure above is the OPS-SAT segment model; the telemanom path is far worse
  (`n_events_scored` **3**, `models/metrics.json`), so the conclusion does not depend on which satellite
  model is meant. Recorded as the reason for an absence, which is the kind of decision this file exists to hold.
- **Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger, Cavallaro and Rieck, "Dos and Don'ts of
  Machine Learning in Computer Security", USENIX Security 2022.** arXiv:2010.09470; also
  *Communications of the ACM*, DOI 10.1145/3643456 `[link]`
  URL: https://arxiv.org/abs/2010.09470
  **Abstract VERIFIED** at arXiv 2026-08-12; the **individual pitfall texts are METADATA-ONLY**, so the
  two named below are carried as titles and prevalences rather than quoted. Pitfall **P7 Inappropriate
  Performance Measures** and **P8 Base Rate Fallacy**, present in **62%** and **46%** of their 30-paper
  survey. This is the canonical citation for why an aggregate metric hides per-class collapse - a claim
  the telecom page asserts with no source, while `metrics.py::per_class_f1` and the WS2 section of this
  file both rest on it. Engelen et al. (WTMC 2021, above) is the measured instance of the same failure;
  this is the systematic statement of it.

### Encrypted-versus-plaintext agreement: the two published reporting conventions

The demo prints a plaintext-versus-CKKS comparison and invented its own presentation of it. There are
two conventions in the literature and this repository should look like one of them.

- **Li, Wu and Miao, "Decision-Aware Quadratic ReLU Replacement for HE-Friendly Inference",
  arXiv:2605.22237, 2026-05-25** `[link]`
  URL: https://arxiv.org/abs/2605.22237
  VERIFIED at the arXiv HTML 2026-08-12. Its Table IX reports **"PH-mis"**, defined verbatim as *"the
  integer number of top-1 mismatches between plaintext polynomial evaluation and CKKS evaluation on the
  evaluated test set; it is zero for every selected row"*. So the published convention is an integer
  **COUNT**, expected **zero**, reported **beside** plaintext and ciphertext accuracy rather than folded
  into them. Bears on `web/src/lib/clientFHE.ts` (which measures `maxAbsError` against a plaintext
  reference) and on however the verification panel presents agreement.
- **Zama, Concrete-ML documentation** `[link]`
  URL: https://docs.zama.ai/concrete-ml
  VERIFIED 2026-08-12. Concrete-ML prints **"Similarity"**: the mean per-sample agreement between clear
  and FHE predictions, as a **percentage**. That is the de-facto convention from the vendor with the
  largest FHE-ML install base, and it is the complement of PH-mis rather than a rival to it - a count of
  disagreements and a percentage of agreements are the same measurement reported two ways, so quoting
  either is defensible and quoting neither is not. Also relevant to
  [`../docs/architecture.md`](../docs/architecture.md): Concrete-ML's **deployment model returns class
  scores and the CLIENT decrypts**, which is the mainstream construction for encrypted multiclass
  inference and is what this demo already does in a single tab.

### What the cost literature rules out, and what verifiability does not prove

- **Mazzone, Everts, Hahn and Peter, "Efficient Ranking, Order Statistics, and Sorting under CKKS",
  USENIX Security 2025.** arXiv:2412.15126 `[link]`
  URL: https://arxiv.org/abs/2412.15126
  VERIFIED at the abstract 2026-08-12. Argmin/argmax over a **128-element vector in 12.83 s**, with
  comparison depth *"of up to 2 (constant)"*. This is the measured cost that **rules out a homomorphic
  argmax in a browser tab** and therefore the published justification for the architecture's choice to
  send the non-linear half to MPC (`src/qsh/mpc/secret_inference.py`, `web/src/lib/clientMPC.ts`). The
  depth result also blocks the obvious rebuttal that the cost is a depth artefact: it is constant.
- **Peng et al., "A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning",
  arXiv:2502.18535**, revised 2026-03-29 `[link]`
  URL: https://arxiv.org/abs/2502.18535
  VERIFIED at the abstract 2026-08-12. Its taxonomy is **verifiable training / verifiable testing /
  verifiable inference**. Worth recording because it gives this project's dual-path comparison a name
  from the field: it occupies the **verifiable testing** slot, not verifiable inference.
- **Atapoor, Baghery, Pereira and Spiessens, "Verifiable FHE via Lattice-based SNARKs", IACR CiC 2024,
  ePrint 2024/032** `[link]`
  URL: https://eprint.iacr.org/2024/032
  VERIFIED at the abstract 2026-08-12. Verifies an approximate neural network of **three layers and more
  than 100 ciphertexts in "less than 1 second"**.
- **Tremblay Thibault and Walter (Zama), "Towards Verifiable FHE in Practice", ACM CCS 2025, ePrint
  2024/451** `[link]`
  URL: https://eprint.iacr.org/2024/451
  VERIFIED at the abstract 2026-08-12. TFHE **bootstrapping** proved with a prover under **20 min**, a
  proof of about **200 kB**, and a verifier under **10 ms**.

> **The load-bearing point about these last two is a NEGATIVE, and it constrains what this project may
> claim.** Verifiable FHE proves that **the server ran the circuit that was requested**. It does **not**
> prove that the result matches a plaintext reference. Those are different properties: the first is
> cryptographic integrity of the evaluation, the second is numerical fidelity of an approximate scheme.
> So this repository's plaintext-versus-CKKS comparison is **empirical fidelity, not cryptographic
> integrity**, and it has to say so - a reviewer who knows this literature will otherwise read a
> side-by-side agreement figure as an integrity claim, which nothing here establishes and which would
> need one of the two constructions above to establish. Recorded in
> [`../docs/architecture.md`](../docs/architecture.md) at the place the comparison is described.

### Conformal prediction on dependent data: the honest answer changed in 2024

Both entries matter because the pre-2024 answer - "split conformal is invalid on a time series" - is
what a reviewer would expect a demo to say about the satellite page's timeline data, and it is wrong.

- **Oliveira, Orenstein, Ramos and Romano, "Split Conformal Prediction and Non-Exchangeable Data",
  *JMLR* 25(225), 2024** `[link]`
  URL: https://jmlr.org/papers/v25/ - volume 25, paper 225; the per-paper permalink is not written out
  because it was not the URL loaded and a guessed one would point at someone else's paper.
  VERIFIED at the JMLR abstract 2026-08-12. Split conformal *"remains valid for many non-exchangeable
  processes by adding a small coverage penalty"*.
- **Barber and Pananjady, "Predictive inference for time series: why is split conformal effective
  despite temporal dependence?", arXiv:2510.02471**, revised 2026-01-10 `[link]`
  URL: https://arxiv.org/abs/2510.02471
  VERIFIED at the abstract 2026-08-12. Bounds the coverage loss by a **"switch coefficient"** and the
  bound is *"sharp over the class of stationary, beta-mixing processes"*.
  So the defensible statement about interval coverage on the satellite timeline is **"valid with a
  bounded penalty"**, not "invalid" and not "valid" unqualified. No conformal method is implemented here
  (a grep for `conformal` across the repository returns nothing), so this is recorded as the citation to
  use if intervals are ever added, and as the correction to the answer a reader is likely to assume.

### Genomics reporting: the reference implementation agrees, and one practice this repo does not follow

- **pgsc_calc (PGS Catalog Calculator) documentation** `[link]`
  URL: https://pgsc-calc.readthedocs.io/
  VERIFIED 2026-08-12. Verbatim: *"The distribution of PGS scores can differ between genetic ancestry
  groups, but this does not necessarily correspond to differences in risk (e.g., disease prevalence or
  mean biomarker values). These differences arise from changes in allele frequencies and linkage
  disequilibrium (LD) patterns."* It reports PGS as **density plots** and recommends interpreting
  results *"as distributions or deciles rather than single-point estimates"*. This is the **field
  reference implementation endorsing what the genomics page already says**, which is the cheapest kind
  of citation to add and the easiest to have missed.
- **Ding et al., "Polygenic scoring accuracy varies across the genetic ancestry continuum", *Nature*
  2023.** DOI 10.1038/s41586-023-06079-4 `[link]`
  URL: https://doi.org/10.1038/s41586-023-06079-4
  **METADATA-ONLY** - paywalled, abstract only, 2026-08-12. Accuracy *"decreases as a function of
  relatedness at the individual level when modelling genetic ancestry as a continuum"*. Recorded as the
  current best practice **against discrete ancestry bins**, because this repository uses discrete 1000
  Genomes super-populations. That is a defensible simplification for a demo - the bins are what the
  corpus ships - but it is the practice a genomics reviewer will raise, so it is written down as a known
  divergence rather than left to be discovered.

### New primary literature, 2026-08-01 to 2026-08-10, all verified at the abstract and all uncited

- **Malik, Anwar and Raza, "Beyond the Quantum Promise: A Security Analysis of Classical Control in
  Quantum Key Distribution."** arXiv:2608.07626, 2026-08-07 `[link]`
  URL: https://arxiv.org/abs/2608.07626
  **The highest-priority item in this batch.** The first hybrid **Tamarin** model of the ETSI and ITU-T
  QKD specifications, finding **three specification-level vulnerabilities** - subverted entanglement
  injection, basis-deferred measurement, message reflection - each arising from *"a classical
  control-plane omission in the procedure text"*, with verified countermeasures (**measurement
  commitment**, **identity-bound MACs**). `src/qsh/qkd/sdn_controller.py` models ETSI GS QKD 015
  `PHYS`/`VIRT` links and `qkd/kme.py` implements GS QKD 014, while
  [`../docs/threat-model.md`](../docs/threat-model.md) had **no classical-control-plane adversary at
  all**. Round 33 adds one there with a resolving condition, so the gap is falsifiable rather than
  merely noted. The finding to keep in view is that these are flaws in the SPECIFICATIONS this project
  implements faithfully.
- **Chiang, "Trimming: Decoupling Multiplicative Depth from Modulus Chains in RNS-CKKS via Rational
  Levels."** arXiv:2608.00375, 2026-08-01 `[link]`
  URL: https://arxiv.org/abs/2608.00375
  Referenced from `src/qsh/models/registry.py` in Round 33; this is the provenance entry.
  **CRITICAL CAVEAT: it has no experiments, and no number from it may be quoted.** Its abstract ends
  *"The proposed framework will be further validated through concrete implementation and experimental
  evaluation"*. It contests the depth-dominance premise of **ATLAS** (arXiv:2607.23478, above) as a
  **named direction only**, which is exactly how it should be cited: a stated alternative to a
  measured claim is not a counter-measurement.
- **Zhang, Zhou, Han, Liu and Huang, "Adaptively Secure Threshold Decryption from LWE with Polynomial
  Modulus."** IACR ePrint 2026/1627, received 2026-08-06 `[link]`
  URL: https://eprint.iacr.org/2026/1627
  Adaptive corruption and leakage from **decryption queries**, via *"a refined polynomial noise flooding
  technique based on a detailed min-entropy analysis"*. That is the same threat surface as the CPAD
  section above, which had **no threshold-setting citation** even though OpenFHE scopes the CPAD attack
  to precisely the threshold case and `submodules/HEonGPU` ships N-out-of-N collective bootstrapping.
- **Al-Mohammed and Al-Ali, "Machine Learning for Specialized QKD Aspects: A Survey of Adaptive
  Protocols, Free-Space Links, 6G Integration, and Steerability-Aware Security."** arXiv:2608.08280,
  2026-08-08 `[link]`
  URL: https://arxiv.org/abs/2608.08280
  Names *"the boundary between ML-based decision support and security certification"* as an open
  challenge. Independent support for a line this repository already carries at
  `references/model-choice-vs-2026-practice.md:200` - *"ML as an overlay for sub-threshold stealthy
  attacks, never as a replacement for decoy-state accounting"* - which until now rested on a single 2026
  study cited there and on no statement about the certification boundary itself. A survey is weak
  evidence for a mechanism and adequate evidence for a boundary, which is all it is used for here.
- **Chen et al., "DYNAFIX: Dynamic Fixed-Point Encoding for Arbitrary-Range MPC."** IACR ePrint
  2026/1612, received 2026-08-05 `[link]`
  URL: https://eprint.iacr.org/2026/1612
  Bears on `src/qsh/mpc/secret_inference.py` and on the fixed-point weight encoding in
  `models/genomics_prs/weights.json` - the same problem bioETH-PRS (arXiv:2605.21634, above) solves on
  the TFHE side, now on the MPC side.

### Negative results and closed searches, verified 2026-08-12

Recorded so no future round re-runs these. A measured horizon is more useful than "nothing found",
so each one names what was checked and where it stopped.

- **The window after 2026-08-11 is structurally EMPTY, and here are the horizons.** arXiv's newest
  cs.CR announcement batch is **2026-08-11**, with no 2026-08-12 batch; IACR ePrint's highest number is
  **2026/1631** (received 2026-08-07, approved 2026-08-10) and **2026/1632 returns HTTP 404**; the
  Federal Register API returns **total count 0** for "post-quantum cryptography" with `publication_date`
  greater than or equal to 2026-08-01; and NIST CSRC's **only** August 2026 item is **2026-08-06,
  NCCoE CSWP 36F on 5G NAS message security**, which is not PQC. So **no PQC instrument exists in
  August 2026 at all**, and the newest one remains OMB M-26-15 (2026-06-24). Any future round that wants
  to re-check should start from those four horizons rather than from a fresh sweep.

  **Round 43 correction, against this project's own interest: the CSRC count above is wrong by one.**
  The listing carries **two** August 2026 items, not one - 2026-08-06 CSWP 36F and **2026-08-05,
  *Transit Cybersecurity Framework Community Profile***. The second is a sector CSF profile and not a
  cryptographic instrument, so **the conclusion is unaffected**: there is still no PQC instrument in
  August 2026. The word "only" is what was wrong, and a stated horizon that undercounts is exactly
  what a later round trusts instead of re-running.

  **Method limit on that correction, stated because it bounds what the correction may be used for.**
  The re-check went through a summarising fetch rather than the CSRC publications API. That is enough
  to establish that a second August item exists - a correction that costs this project a claim, so the
  direction of error is safe - and **not** enough to assert an exhaustive August list. A round wanting
  an authoritative count should hit the API directly.

  **Horizons re-stamped 2026-08-15, at primary source.** The Federal Register API returns the literal
  `"count":0` for "post-quantum cryptography" published on or after 2026-08-01; arXiv cs.CR's newest
  submission is **2608.13465** (2026-08-13T16:46:56Z); IACR ePrint **2026/1644** is live and
  **2026/1645** returns 404. **2026-08-14 and 2026-08-15 are structurally empty** on both preprint
  servers, so a sweep of those two days is a sweep of an empty set and is recorded as one rather than
  as a search that found nothing.
- **No published in-browser FHE or MPC, and no prior art on VISUALISING encrypted inference - FIFTH
  consecutive round.** An IACR full-text search for "WebAssembly" returns **8 results across all
  years**, the newest crypto-relevant one still **2025/487 webSPDZ**. The composition novelty claim has
  now survived five independent attempts to refute it, which is worth more than any single sweep.
- **Bland and Altman (1986) is NOT FOUND and must not be cited.** The canonical method-comparison paper
  was not loaded: PubMed **2868172** returned a reCAPTCHA and Europe PMC returned navigation chrome
  only. It is the obvious citation for a plaintext-versus-ciphertext agreement plot, which is exactly
  why this negative is recorded - the temptation to cite it from memory is the hazard. Use the PH-mis
  count or Concrete-ML's Similarity convention above, both of which were actually loaded.
- **HEBench defines no error metric, tolerance or accuracy criterion for approximate schemes.** There is
  nothing in it to cite as a reporting standard, so this repository's existing note that HEBench covers
  only primitive operations is correct and stays. Checked 2026-08-12.
- **This folder cannot ground a UX or accessibility recommendation, and every round has been asked
  for one anyway. Measured 2026-08-15, said out loud here for the first time.** The standing
  per-round request asks for a UX review *grounded in* `references/`. A grep of all **16** markdown
  files for `usability`, `accessibility`, `WCAG`, `human-computer`, `human factors`, `affordance`,
  `user experience`, `HCI`, `cognitive load` and `eye-tracking` returns hits in exactly **one** file,
  and that file is THIS one - the sentence you are reading is the only match, because recording the
  absence created it. Excluding this file the count is **zero**, which is the substantive claim and
  is unchanged. Re-measured Round 49; the original "zero" was true when written and is now literally
  one by self-reference, which is worth stating rather than silently editing the number to something
  that reads as a real source having appeared. Widening to
  `interface design`, `visualis/visualiz`, `readability`, `legibility` and `colour-blind` returns
  **two**, and neither is a source: one is `he-scheme-selection.md` calling a twelve-coefficient
  model readable, and the other is the entry four bullets above recording that no prior art on
  visualising encrypted inference exists. The bundled PDFs are quantum and cryptography papers:
  **7 committed** to the repository, with 4 more present on a working checkout and deliberately not
  redistributed (`DATA_LICENSES.md` section 8 adjudicates each). This said "All **15**" until Round
  49, a count matching neither figure, and the one business-of-quantum title it alluded to is among
  the four that are NOT committed. **There is no HCI or human-factors source in this reference set at
  all.**

  So the UX rows do not rest on this folder, and any claim that they do would be false. What they
  rest on is stated instead: the **WCAG 2.2 success criteria themselves**, quoted by number in
  `docs/browser-verification-checklist.md`, plus that file's own measurement procedures - composite
  the alpha before reading a contrast ratio, read the `role` the element carries rather than the prop
  that was passed, enumerate the series from the source rather than from the last sweep's list. That
  is a defensible basis; it is simply a different one from a paper in this folder.

  **Not resolved by adding a citation to look resolved.** A single HCI paper bolted on here would let
  the round report "grounded in references/" while the rows continued to come from WCAG, which is the
  decoration-condition failure `QC16` describes. If an HCI source is ever added it has to be because
  a row actually derives from it. Recorded so that no future round answers this bullet by asserting
  a grounding it does not have - which is what nine rounds of silence amounted to.

### Leads recorded rather than cited

- **EU NIS Cooperation Group PQC roadmap milestones - DEADLINE EXECUTED Round 42, and the lead
  was wrong twice in the flattering direction.** The primary document is *A Coordinated Implementation Roadmap
  for the Transition to Post-Quantum Cryptography*, Part 1, Version 1.1, EU PQC Workstream,
  11.06.2025, the first deliverable of the NIS CG work stream on PQC. Fetched and read as PDF; its
  section 4 timeline is now quoted in [`../docs/pqc-roadmap.md`](../docs/pqc-roadmap.md).

  What this entry said, against what the document says:
  - *"full transition by 2035-12-31"* - **no.** 2035 is medium-risk completed and low-risk
    *"completed as much as feasible"*, and the executive summary reads *"completed for as many
    systems as practically feasible"*. The lead was stronger than its source.
  - *"`docs/pqc-roadmap.md` is **US-instrument-only**"* - **no.** That file already said *"Every
    instrument below is a US one, apart from the BSI and Japanese entries"* and recorded the EU set
    as *"Verified absent, not overlooked"*. The roadmap was precise; this lead dropped its qualifier.

  The dates themselves were right, which is how both overstatements survived being re-read.

  **Why it took nine rounds.** This entry ended *"Next round: find the Cooperation Group publication
  itself"* and was written in Round 33. `tests/test_register_deadlines.py` reads
  `docs/deliverable-verification.md` and matches `by **Round N**` or `BLOCKED UNTIL <date>`; this
  obligation was in a different file, in a third phrasing. Both halves of that gate's scope have been
  widened, and this sentence is the seed it was widened against.
- **91 FR 38487 was cited in this file and nowhere else; that is now closed.** The second same-day
  quantum executive order of 2026-06-25 ("Ushering in the Next Frontier of Quantum Innovation", FR
  document 2026-12910) was verified at the Federal Register API in Round 32 and recorded here, while
  `docs/pqc-roadmap.md` cited only EO 14412 / 91 FR 38483. Round 33 adds the cross-reference to that
  roadmap, stating only what was verified - title, document number, FR citation and date - because its
  text was never loaded and it therefore contributes no deadline.
