# Annotated References

The `references/` folder holds the primary-source PDFs that ground this PoC. Each entry maps the
paper to the **workstream (WS)** it informs, the **module** that implements/uses it, and any
**open-source code** reused. `[R]` = PDF present in this folder.

> Workstreams: **WS1** AI-QKD (parameter optimization) · **WS2** QML-IDS · **WS3** QSMPC×QKD ·
> **WS4** QHE×AI. Cross-cutting: PQC-first hybrid, crypto-agility, XAI.
>
> `[R]` = PDF bundled in this folder. `[arXiv]` = not bundled (large/third-party — fetch from the arXiv id).

---

## WS1 — AI-QKD (real-time parameter optimization) + entropy (QRNG)

- **AML-QKD: Adaptive Machine Learning Framework for Real-time Parameter Tuning in QKD** — Mohamed,
  Kaldari & Al-Kuwari, Qatar Center for Quantum Computing (HBKU), **arXiv:2603.04192** [quant-ph]
  `[arXiv]`. **v1** (4 Mar 2026) was titled *OptiQKD: A Machine Learning-Optimized Framework for
  Real-Time Parameter Tuning in Quantum Key Distribution*; **v2** (1 Jun 2026) retitled it AML-QKD.
  → **This is one paper in two versions, not two papers.** An earlier revision of this file listed it
  twice and described v2 as "extending" v1; v2 *supersedes* v1 and **revises the headline number
  downward**: v1 claimed median SKR **+20–30%**, v2 claims **+14–25%** (QBER 3.0%→1.5% in both). The
  QRL/E91 result (+29.2% throughput) is labelled *exploratory* by the authors. Simulation only.
  → Grounds `src/qsh/qkd/optimize.py` — a deterministic, offline-safe **sklearn-surrogate +
  coordinate-ascent** analogue over the two-intensity decoy-state BB84 model, with the `mu_decoy`
  single-photon bound (Ma-Qi-Zhao-Lo 2005) and a convergence metric — and the E91/CHSH model
  `src/qsh/qkd/e91.py`. Honest scope: a simulation (not deployed QKD), gradient-free (no RL/torch),
  so it is an analogue of the paper's theme rather than a reproduction of its method.
  → Note: `references/` holds **both** the v1 and v2 PDFs (`OptiQKD_*.pdf`, `AML-QKD_*.pdf`). They are
  the same arXiv entry at two revisions, kept for provenance; cite v2.

- **Information-Theoretic Solutions for Seedless QRNG Bootstrapping and Hybrid PQC-QKD Key Combination** — Giestinhas & Spiller, University of York, arXiv:2603.26907v2 [quant-ph], 2026 `[arXiv]`
  → **Quantum Leftover Hash Lemma** strong (Toeplitz / universal-2) extractors: (1) **seedless QRNG bootstrapping** from two independent entropy sources; (2) a QLHL-extractor **alternative** to XOR key-combining. It is **not** "secure-if-either" and not "better than XOR" — the paper never uses that phrase and explicitly says strong seeded extractors "present different characteristics", noting its own cost as "longer keys and a worse computational complexity". The property it does claim: if the combined output **and one input key** are compromised, the remaining key material retains quantifiable min-entropy. PQC keys are modelled as HILL (computational) min-entropy. Licence CC BY 4.0. Grounds the Round-16 QRNG/transport upgrade: `src/qsh/qrng/simulator.py` (LHL-bounded Toeplitz output + seedless two-source extractor), `src/qsh/qrng/health.py` (SP 800-90B min-entropy estimate), and the config-gated ITS combiner in `src/qsh/pqc/transport.py` (`_mix_qkd_its`, `hybrid_combiner_mode="its_toeplitz"`).

---

## WS3 — QSMPC × QKD (quantum-secure multi-party computation)

- **Quantum Secure Multiparty Computation based on Secure Summation and QKD** — Mandeep Kumar & Bhaskar
  Mondal (NIT Patna), Research Square, 3 Jul 2025, DOI `10.21203/rs.3.rs-6706796/v1`, **CC BY 4.0** `[R]`
  (CC BY requires attribution; an earlier revision of this entry named no authors)
  `Quantum_Secure_Multiparty_Computation_based_on_Secure_Summation_and_QKD.pdf`
  → core protocol for `src/qsh/qsmpc/secure_summation.py`. QKD-shared keys + secure summation.

- **Efficient Simulation of Quantum Secure Multiparty Computation** — Sutradhar, arXiv:2501.10083, 2025 `[R]`
  `Efficient_Simulation_of_Quantum_Secure_Multiparty_Computation.pdf`
  → **(t,n)-threshold QSMS** = Shamir secret sharing + QFT + Pauli ops. Directly reimplemented in
  `secure_summation.py` / `shamir.py` (Qiskit-native, no hardware). Validated on the paper's worked
  example (t=3, n=7, d=11, X=2, Y=3 → S=5).

- **Experimental SMPC from Quantum Oblivious Transfer with Bit Commitment** — Zhang et al., npj Quantum Inf., 2026 `[R]`
  `Experimental_secure_multiparty_computation_from_quantum_oblivious_transfer_with_bit_commitment.pdf`
  → decoy-state BB84 + SHA-256 commitment QOT → **private set intersection** on bank-blacklist data.
  Implemented (sim) in `src/qsh/qsmpc/qot.py` and the **Finance AML/PSI** use case.

- **Quantum-Secure Multiparty Deep Learning** — Sulimany et al., Phys. Rev. X 15, 041056, 2025 `[R]`
  `Quantum-Secure_Multiparty_Deep_Learning.pdf`
  → photonic coherent linear-algebra inference; Holevo/Cramér-Rao **leakage bounds** (0.1 bit/weight,
  MNIST >95%). The leakage-bound analysis is reimplemented in `src/qsh/qsmpc/` (security metrics).

- **Quantum-Safe Multi-Party Computation for Distributed AI Datasets** — Gopher Security blog, 2026 `[R]`
  `Quantum-Safe_Multi-Party_Computation_for_Distributed_AI_Datasets ... .pdf`
  → qOLE / qMPSI + **PQC (ML-KEM) hybrid transport** + MCP "delegate-and-verify" narrative. Informs
  `src/qsh/pqc/transport.py` and the E2E orchestration framing.

---

## WS4 — QHE × AI (quantum/classical homomorphic encryption + AI)

- **Reverse Delegated Training and Private Inference via Perfectly-Secure QHE** — Ortega & Martín-Delgado, arXiv:2602.12712, 2026 `[R]`
  `Reverse_Delegated_Training_and_Private_Inference_via_Perfectly-Secure_Quantum_Homomorphic_Encryption.pdf`
  → **QOTP + classical FHE** hybrid, Clifford+T QCNN, **Pauli-gate concealment** (circuit privacy),
  MNIST 96.7%. **Relationship to this repo:** `src/qsh/qhe/qcnn.py` is an *independent*
  implementation of the paper's construction. It does not import, vendor or wrap the authors'
  reference code at `OrtegaSA/CQC-QHE-repo` — an earlier version of this line claimed code reuse
  that never existed, and the submodule was removed in Round 17. Cited as the source of the
  construction, which is the accurate relationship.

- **Rate-1 Quantum Fully Homomorphic Encryption** — Chardouvelis, Döttling, Malavolta, IACR ePrint 2020/1454 (TCC 2021) `[R]`
  `Rate-1_Quantum_Fully_Homomorphic_Encryption.pdf`
  → ciphertext/plaintext ratio ≈ 1, malicious circuit privacy (LWE). Informs `src/qsh/qhe/rate1_qfhe.py`
  (QOTP + rate-1 classical FHE keys) and the benchmark's communication-efficiency metric.

- **Leveraging ASIC AI Chips for Homomorphic Encryption (CROSS)** — Tong et al., IACR ePrint 2026/160 / HPCA 2026 `[R]`
  `Leveraging_ASIC_AI_Chips_for_Homomorphic_Encryption.pdf`
  → maps FHE NTT/modular ops onto AI accelerators (BAT/MAT). Reference for the GPU benchmark's
  NTT/throughput discussion. Code: `EfficientHE/CROSS` (TPU-oriented; reference-only).

- **EncFormer: Secure & Efficient Transformer Inference over Encrypted Data** — Zhu, Jin, Aung & Xiao
  (NUS / A*STAR), **arXiv:2604.09975 [cs.CR], 11 Apr 2026** `[R]`
  `EncFormer_Secure_and_Efficient_Transformer_Inference_over_Encrypted_Data.pdf`
  → **Manuscript submitted to IEEE TDSC — not accepted, not peer-reviewed.** Hybrid **CKKS + two-party
  computation**, not FHE-only, and the text contains no TFHE, no quantization and no polynomial-softmax
  method — an earlier revision of this entry attributed all three to it, and dated it 2024. Reported
  end-to-end BERT-base latency: **2.1 min LAN, 8.9 min WAN3**; the authors put their secure pipeline at
  *"roughly 3x10^4 slower"* than the 4.5 ms plaintext baseline. Informs the **Transformer** workload in
  `src/qsh/benchmark/` as a cost reference, not as a performance target.

---

## QFL — quantum federated / delegated learning (medical + finance use cases)

- **Quantum Delegated and Federated Learning via Quantum Homomorphic Encryption** — Li & Deng, arXiv:2409.19359, 2024 `[R]`
  `Quantum_Delegated_and_Federated_Learning_via_Quantum_Homomorphic_Encryption.pdf`
  → QOTP-encrypted delegated/federated learning; informs `src/qsh/ai/federated.py` privacy layer.

- **Quantum Federated Learning: A Comprehensive Survey** — Nguyen, Uddin, Shaon, Rahman, Dobre & Niyato, arXiv:2508.15998, 21 Aug 2025 — **preprint, under revision at IEEE COMST; not published** `[R]`
  `Quantum_Federated_Learning_A_Comprehensive_Survey.pdf`
  → taxonomy (fully/hybrid/hierarchical QFL), frameworks (PennyLane, Flower, NVFlare…), datasets.
  Design reference for the federated-learning architecture.

---

## External standards & threat context (not in folder; see `docs/pqc-roadmap.md`)

- NIST **FIPS 203/204/205** (ML-KEM / ML-DSA / SLH-DSA), 2024.
- NSA **CNSA 2.0** — 2027 procurement mandate. On QKD, NSA answers **"No"** to using it to protect an NSS
  and *"does not support the usage of QKD or QC"* — it does **not** recommend a PQC-hybrid QKD add-on either.
  The hybrid framing in this repository comes from **BSI TR-02102-1 v2026-01 §2.2**.
- **HNDL** (Harvest-Now-Decrypt-Later) and **Q-Day** acceleration (Gidney arXiv:2505.15917, 2025).

---

## OSS reused (see `scripts/bootstrap_submodules.sh`, all commercial-OK)

| Code | Used by | License |
|---|---|---|
| `OrtegaSA/CQC-QHE-repo` | *cited only* — `qsh/qhe/qcnn.py` is an independent implementation of the paper, not a wrapper | Apache-2.0 |
| `open-quantum-safe/liboqs` (+ `liboqs-python`) | `qsh/pqc/*` | MIT |
| `openfheorg/openfhe-development`, `microsoft/SEAL` | `qsh/fhe/*`, benchmark | BSD-2 / MIT |
| `tenseal` | `qsh/fhe/*`, benchmark | **Apache-2.0** (confirmed from the installed package metadata; an earlier revision of this table listed it under BSD-2/MIT) |
| `Alisah-Ozcan/HEonGPU`, `chart21/hpmpc` (PIGEON), `kanav99/shark` | `qsh/benchmark/*` (our own glue) | Apache-2.0 / MIT |
| `data61/MP-SPDZ` | benchmark CPU MPC baseline | CSIRO |
| `lschoe/mpyc` (pip dependency, **not vendored** — `requirements-fhe.txt`; the submodule was dropped in Round 17) | `qsh/mpc/secret_inference.py` — non-linear MPC (ReLU/argmax on shares) | MIT |
| `IBM/AMLSim`, `synthetichealth/synthea` | **Cited only — not used.** Round 17 removed both submodules as unreferenced; `qsh/data/synthetic.py` mentions them in prose only, no code path calls them. | Apache-2.0 |
| `EdgarLopezPhD/PaySim` | *cited only - never cloned or vendored* | **GPL-3.0** (the simulator source; the Kaggle output dataset is separately CC-BY-SA-4.0) |
| `openfhe` (OpenFHE 1.5.1, CKKS + bootstrapping) | `qsh/fhe/backends.py` (`OpenFHEBackend`) | BSD-2-Clause (commercial-OK) |
| **`s0l0ist/node-seal`** (SEAL→WASM CKKS) | `web/src/lib/clientFHE.ts` — real in-browser FHE (demo) | MIT |
| **`mpc-framework` + `emp-wasm-engine` + `summon-ts`** | `web/src/lib/clientMPC.ts` — real in-browser garbled-circuit 2PC (demo) | MIT |
| **`mcp`** (MCP Python SDK / FastMCP) + official `@modelcontextprotocol/server-filesystem` | `qsh/mcp/*` — real quantum-safe MCP server + interop (full) | MIT |
| `PennyLane` | `qsh/ai/federated.py` — Quantum Federated Learning (`kind="qfl"`). The FedAvg and secure-aggregation code is this repository's own; Flower is cited as prior art below, not used as a dependency. | Apache-2.0 |
