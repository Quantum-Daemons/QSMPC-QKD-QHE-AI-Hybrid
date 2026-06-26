# Annotated References

The `references/` folder holds the primary-source PDFs that ground this PoC. Each entry maps the
paper to the **workstream (WS)** it informs, the **module** that implements/uses it, and any
**open-source code** reused. `[R]` = PDF present in this folder.

> Workstreams: **WS1** AI-QKD (parameter optimization) · **WS2** QML-IDS · **WS3** QSMPC×QKD ·
> **WS4** QHE×AI. Cross-cutting: PQC-first hybrid, crypto-agility, XAI.

---

## WS3 — QSMPC × QKD (quantum-secure multi-party computation)

- **Quantum Secure Multiparty Computation based on Secure Summation and QKD** — Research Square, 2025 `[R]`
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
  MNIST 96.7%. **Code reused:** `OrtegaSA/CQC-QHE-repo` → `submodules/CQC-QHE`, wrapped by
  `src/qsh/qhe/qcnn.py`.

- **Rate-1 Quantum Fully Homomorphic Encryption** — Chardouvelis, Döttling, Malavolta, IACR ePrint 2020/1454 (TCC 2021) `[R]`
  `Rate-1_Quantum_Fully_Homomorphic_Encryption.pdf`
  → ciphertext/plaintext ratio ≈ 1, malicious circuit privacy (LWE). Informs `src/qsh/qhe/rate1_qfhe.py`
  (QOTP + rate-1 classical FHE keys) and the benchmark's communication-efficiency metric.

- **Leveraging ASIC AI Chips for Homomorphic Encryption (CROSS)** — Tong et al., IACR ePrint 2026/160 / HPCA 2026 `[R]`
  `Leveraging_ASIC_AI_Chips_for_Homomorphic_Encryption.pdf`
  → maps FHE NTT/modular ops onto AI accelerators (BAT/MAT). Reference for the GPU benchmark's
  NTT/throughput discussion. Code: `EfficientHE/CROSS` (TPU-oriented; reference-only).

- **EncFormer: Secure & Efficient Transformer Inference over Encrypted Data** — 2024 `[R]`
  `EncFormer_Secure_and_Efficient_Transformer_Inference_over_Encrypted_Data.pdf`
  → encrypted Transformer (CKKS/TFHE + quantization, polynomial softmax). Informs the **Transformer**
  workload in `src/qsh/benchmark/`.

---

## QFL — quantum federated / delegated learning (medical + finance use cases)

- **Quantum Delegated and Federated Learning via Quantum Homomorphic Encryption** — Li & Deng, arXiv:2409.19359, 2024 `[R]`
  `Quantum_Delegated_and_Federated_Learning_via_Quantum_Homomorphic_Encryption.pdf`
  → QOTP-encrypted delegated/federated learning; informs `src/qsh/ai/federated.py` privacy layer.

- **Quantum Federated Learning: A Comprehensive Survey** — Nguyen et al., IEEE COMST, 2025 (arXiv:2508.15988) `[R]`
  `Quantum_Federated_Learning_A_Comprehensive_Survey.pdf`
  → taxonomy (fully/hybrid/hierarchical QFL), frameworks (PennyLane, Flower, NVFlare…), datasets.
  Design reference for the federated-learning architecture.

---

## External standards & threat context (not in folder; see `docs/pqc-roadmap.md`)

- NIST **FIPS 203/204/205** (ML-KEM / ML-DSA / SLH-DSA), 2024.
- NSA **CNSA 2.0** — 2027 procurement mandate; QKD non-recommended standalone → PQC-hybrid only.
- **HNDL** (Harvest-Now-Decrypt-Later) and **Q-Day** acceleration (Gidney arXiv:2505.15917, 2025).

---

## OSS reused (see `scripts/bootstrap_submodules.sh`, all commercial-OK)

| Code | Used by | License |
|---|---|---|
| `OrtegaSA/CQC-QHE-repo` | `qsh/qhe/qcnn.py` (WS4) | Apache-2.0 |
| `open-quantum-safe/liboqs` (+ `liboqs-python`) | `qsh/pqc/*` | MIT |
| `openfheorg/openfhe-development`, `microsoft/SEAL`, `tenseal` | `qsh/fhe/*`, benchmark | BSD-2 / MIT |
| `Alisah-Ozcan/HEonGPU`, `chart21/hpmpc` (PIGEON), `kanav99/shark` | `qsh/benchmark/*` (our own glue) | Apache-2.0 / MIT |
| `data61/MP-SPDZ` | benchmark CPU MPC baseline | CSIRO |
| `lschoe/mpyc` (vendored) | `qsh/mpc/secret_inference.py` — non-linear MPC (ReLU/argmax on shares) | MIT |
| `IBM/AMLSim`, PaySim, `synthetichealth/synthea` | `qsh/data/*` | Apache-2.0 |
| `openfhe` (OpenFHE 1.5.1, CKKS + bootstrapping) | `qsh/fhe/backends.py` (`OpenFHEBackend`) | BSD-2-Clause (commercial-OK) |
| **`s0l0ist/node-seal`** (SEAL→WASM CKKS) | `web/src/lib/clientFHE.ts` — real in-browser FHE (demo) | MIT |
| **`mpc-framework` + `emp-wasm-engine` + `summon-ts`** | `web/src/lib/clientMPC.ts` — real in-browser garbled-circuit 2PC (demo) | MIT |
| **`mcp`** (MCP Python SDK / FastMCP) + official `@modelcontextprotocol/server-filesystem` | `qsh/mcp/*` — real quantum-safe MCP server + interop (full) | MIT |
| `Flower` + `PennyLane` | `qsh/ai/federated.py` — Quantum Federated Learning (`kind="qfl"`) | Apache-2.0 |
