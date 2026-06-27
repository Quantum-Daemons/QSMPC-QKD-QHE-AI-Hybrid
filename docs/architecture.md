# Architecture

```
                         ┌──────────────────────────────────────────────┐
   Browser (SPA)         │            FastAPI backend (api/)             │
   React + Vite          │                                              │
   ├ @noble/post-quantum │  RunManager ── threads ── Pipeline           │
   ├ onnxruntime-web     │      │                       │               │
   ├ React Flow (DAG)    │   WebSocket  ◀── EventBus ◀──┤ stages        │
   └ recharts            │      │                       ▼               │
        ▲   │ REST/WS    │   /api/*                 ┌────────────┐       │
        │   ▼            │                          │  qsh core  │       │
   ┌────────────┐  HTTP  │                          └────────────┘       │
   │   Caddy    │◀───────┤  COOP/COEP, TLS, /api proxy, static SPA       │
   └────────────┘        └──────────────────────────────────────────────┘

   qsh core (src/qsh/) — the E2E security layer
   ┌───────────────────────────────────────────────────────────────────┐
   │ orchestration/  Pipeline · Stage · EventBus · CryptoRegistry        │
   │   pqc/      ML-KEM+X25519 hybrid KEM · ML-DSA sig · AEAD transport   │
   │   qkd/      BB84/decoy sim · channel/QBER · AI optimizer (WS1)       │
   │   qml_ids/  8 attack scenarios · RF + variational QNN (WS2)          │
   │   qsmpc/    Shamir+QFT secure sum · QOT · PSI (WS3)                  │
   │   qhe/      QOTP · Clifford+T QCNN · Rate-1 QFHE (WS4)               │
   │   fhe/      CKKS LINEAR layers: TenSEAL · OpenFHE (CPU) · HEonGPU GPU│
   │             + genuine FHE CONVOLUTION (conv.py: im2col → dot_plain)  │
   │   mpc/      NON-LINEAR (ReLU/argmax/maxpool): MPyC Shamir shares;     │
   │             layer specs (Conv/Dense/Pool/Flatten) + hybrid dispatch   │
   │   ai/       small CNN/Transformer · Flower-style FL + secure-agg     │
   │   benchmark/ hybrid MPC-FHE (linear→FHE+non-linear→MPC) vs QSMPC-FHE │
   │   usecases/ finance_aml · medical_fl · genomics · telecom_ids        │
   └───────────────────────────────────────────────────────────────────┘
```

## The pipeline contract
A `Pipeline` runs an ordered list of `Stage`s over a shared `Context`, emitting
structured `Event`s to an `EventBus`. The canonical chain:

```
PQCHandshake → QKDKeyAgreement → QSMPCCompute → PrivateInference → Result
```

Each stage resolves its algorithm from the `CryptoRegistry` (crypto-agility), so the
same chain runs PQC-only or PQC+QKD, TenSEAL or OpenFHE (linear-FHE), MPyC or MP-SPDZ
(non-linear-MPC), ML-KEM-768 or -1024 — chosen at runtime. Use cases swap in their own
data and (for medical) extra stages (`FederatedTrainingStage`, `QMLIDSStage`).

## Faithful hybrid private inference (linear → FHE, non-linear → MPC)
The 2026-standard private-inference split (Gazelle → Delphi → BOLT → Bumblebee → BLB; EncFormer)
is **linear layers under FHE** (matmul/conv as CKKS; GPU-accelerable via HEonGPU) and **non-linear
layers under MPC** (ReLU/argmax via secret-sharing secure comparison). `qsh.mpc.secret_inference`
implements it with TenSEAL/OpenFHE for the linear half and **MPyC** (Shamir shares, the CrypTen
`ReLU([x]) = [x]·[x>0]` pattern) for the non-linear half, converting CKKS↔shares at each boundary;
the benchmark reports the FHE-linear vs MPC-non-linear latency split + MPC communication. The linear
half includes a **genuine FHE convolution** (`fhe/conv.py`): im2col turns the encrypted image into
receptive-field patches and each output `(channel, position)` is one `dot_plain(patch_ct, kernel)` +
bias — the same CKKS primitive as a dense layer, no rotations. The engine dispatches on tagged layer
specs (`mpc/layers.py`: `ConvSpec`/`DenseSpec`/`PoolSpec`/`FlattenSpec`), so the **medical** use case
runs a real MedMNIST-scale CNN under FHE-linear conv + MPC `secure_relu`/`secure_maxpool` (the dense
MLP path is byte-identical via `normalize_layers`). FHE-feasible by design — MedMNIST-scale, not ResNet. The
**QSMPC-FHE** quantum analogue is Clifford gates (linear) vs T-gates (non-linear), both evaluated on
the encrypted state by QOTP/Rate-1 QFHE (ct/pt ≈ 1.3, no FHE↔MPC boundary). A **lightweight / medium
/ large** model-size selector scales the workload — light + medium run for real, large is projected
from measured per-op cost.

## SPA shape
Overview · Introduction & Value · Validations · E2E Orchestration (generic "run any pipeline") · Use Cases
index · and **5 per-use-case simulator pages** (`/usecase/:id`).

**Agentic AI · MCP** is a **cross-cutting per-use-case layer** (`web/src/components/AgenticAIPanel.tsx`) — a
reusable panel **embedded in every use-case page**, not a standalone page (the report frames MCP as the AI
integration layer under the quantum-safe channel, applied per use case; the standalone `/agentic-ai` page was
removed in Round 12 and now redirects to the Use Cases index). It runs a small language model **in the
browser** — Transformers.js (`@huggingface/transformers`, onnxruntime-web WebGPU/WASM) or WebLLM
(`@mlc-ai/web-llm`, WebGPU), both Apache-2.0 and **dynamically imported** (async chunks, never in the entry
bundle). The AI request/response is wrapped in a **client-side quantum-safe MCP channel**
(`web/src/lib/clientMcp.ts` — a zero-server mirror of `pqc/mcp_bridge.py` + `transport.py` via @noble
ML-KEM-768/ML-DSA-65 + WebCrypto HKDF/AES-256-GCM), with a **per-industry MCP tool** (`score_transaction` /
`query_cohort` / `pool_allele_counts` / `inspect_qkd_telemetry` / `relay_mission_telemetry`) and the live
run's metrics composed into the prompt. Everything fails soft (no WebGPU → Transformers.js WASM; the MCP
envelope always runs). Each use-case page
unifies a **live two-party (Alice↔Bob) sequence diagram** (driven by the running pipeline's
stages — Start/Pause/Step/Stop/Restart), that industry's benchmark, and a signed SHAP explanation
(finance/medical/genomics). **One Run drives both** (Round 7): the live pipeline runs, and the
`QuantumComparisonStage` assembles the benchmark **from that run** into `ctx.artifacts["benchmark_result"]`
(rides on `/api/runs/{id}`, no extra endpoint) — the MPC-FHE row is the run's real measured FHE-linear +
MPC-non-linear latency, next to the QSMPC-FHE comparison + advantage. There is **no separate "Run
benchmark" button**; `BenchmarkPanel` is a controlled results panel. The diagram makes the MPC-FHE
secret-sharing over **LAN↔WAN↔LAN** explicit (CKKS ciphertext for the linear half, secret-shares for
the non-linear half). The run engine is a shared hook (`web/src/lib/usePipelineRun.ts`) reused by the
Orchestration page and every use-case page; the demo mirrors it client-side via
`clientCompute.assembleBenchmark` (zero `/api` compute).

**Config single source of truth.** The benchmark cost model, per-use-case tier architectures, workload
multipliers, security/advantage scoring weights, demo data sizes and in-browser-AI knobs all live in
`src/qsh/config/defaults.py`. The backend reads the dataclasses directly; the SPA gets the same data from
`/api/config` (full mode) or the baked `web/src/lib/config.defaults.json` (demo, zero `/api`), generated
by `scripts/export_config.py`. So the demo's representative numbers can never drift from the backend
(`web/src/lib/config.ts` is the single frontend accessor; a test asserts the baked JSON == the config).

## Real client-side FHE/MPC, QFL, and a real MCP server (Round 11)
The public demo's privacy-preserving compute is **genuinely real and measured in the browser** (no
server, no cost-model estimate):
- **FHE** — `web/src/lib/clientFHE.ts` runs **node-seal** (Microsoft SEAL → WASM, MIT) **CKKS**: the
  feature vector is encrypted, the linear layer `W·x+b` is evaluated homomorphically (multiply-plain +
  rotate-and-add slot sum), and only the decrypted logits leave encryption. Reported FHE-linear ms,
  ciphertext bytes and the ≈1e-9 approximation error are real `performance.now()` measurements.
- **MPC** — `web/src/lib/clientMPC.ts` runs a real **garbled-circuit authenticated-garbling 2PC**
  (`mpc-framework` + `emp-wasm-engine` + `summon-ts`, all MIT): the logits are additively secret-shared,
  a summon-compiled argmax circuit privately reconstructs + compares them, and **both parties run loopback
  in one tab** (zero VPS) — the reported MPC ms + real garbled-circuit communication bytes are measured.
- **QFL** — Quantum Federated Learning is a per-use-case WebUI toggle. Full deployment:
  `ai.federated.federated_train(kind="qfl")` trains a hybrid **PennyLane VQC (`HybridQNN`) + classical**
  model with FedAvg under quantum-safe secure aggregation (CPU, since `default.qubit` simulates the VQC).
  Demo: `clientCompute.clientQFL` is an in-browser FedAvg of a logistic head over the `quantumSim` QCNN
  feature map, round accuracy genuinely computed. Both honest small-VQC scope (not fault-tolerant QFL).
- **Real MCP server** (`src/qsh/mcp/server.py`, full deployment only) — a **FastMCP** server (MCP Python
  SDK, MIT) exposing the 5 per-industry tools; `pqc/mcp_bridge.secure_agent_call` now **dispatches the
  real tool** and seals the **ML-DSA-signed result** back over the PQC(+QKD)+AES-GCM channel (disjoint
  nonce ranges per direction). `src/qsh/mcp/fs_interop.py` connects (stdio) to the **official
  `@modelcontextprotocol/server-filesystem`** and carries its real result under the same quantum-safe
  seal — real external-MCP interop. `/api/mcp/{tools,call,filesystem}`. The demo keeps the client-side
  `clientMcp.ts` mirror, so the VPS takes **zero per-user MCP load**.
- **Per-industry real data/models** — finance PaySim + XGBoost (distilled to a CKKS-linear surrogate),
  medical MedMNIST CNN, **genomics PGS-Catalog Polygenic Risk Score (linear → native CKKS dot)**, telecom
  QKD-attack data + Quantum-SVC, satellite ESA-ADB telemetry + a CKKS anomaly score. The FHE-*encrypted*
  inference is the FHE-feasible part; the full model is the real industry architecture (honest crypto scope).

**Backend layout (audited Round 11, kept lean).** Modules: `pqc/` (hybrid KEM, transport, signatures,
MCP bridge), `qkd/` (BB84, channel, QSDC), `qsmpc/` (QOT, Shamir, secure summation), `fhe/` (CKKS
inference, conv, backends), `mpc/` (secret inference, layers), `ai/` (models incl. `HybridQNN`, federated,
explainability), `qml_ids/`, `benchmark/`, `usecases/`, `orchestration/`, `mcp/` (new), `config/`
(single source). The demo going fully client-side did **not** orphan backend code: every `/api/*`
endpoint is still used by the full-mode SPA, the dense+conv hybrid dispatch is already unified
(`normalize_layers`), and config is centralized — so the reorg was an audit + the clean new `mcp/`
module rather than a churn of working, tested code (the full test suite stays green).

## Two compute topologies
- **Full** (GPU box): everything runs server-side — federated training (Flower + PennyLane QFL), GPU FHE
  (HEonGPU), GPU quantum sim (CUDA-Q / lightning.gpu), the real MCP server, Prometheus + Grafana.
- **Client-side build** (no GPU, no server-side ML): the static SPA runs **all** heavy compute in the
  browser (WASM/WebGPU PQC via @noble, **real CKKS FHE + garbled-circuit MPC**, ONNX inference, in-browser
  AI, JS BB84/PSI/SHAP/QFL, the client MCP mirror) — it makes **zero `/api` compute calls** (Validations'
  PQC/QKD/MCP are client-mirrored). It ships a **slim backend image** (`docker/Dockerfile.api.demo` —
  core+api only, no FHE/torch; ~2.9 GB smaller than the full image), optional and used only as a
  `?demo=full` fallback. Inference + an optional in-browser QFL toggle; the **full** topology additionally
  runs server-side federated learning + FHE inference + the MCP server. AI model weights stream from the
  HF/MLC CDN (browser-cached).

## Why CUDA-Q / cuQuantum (not qiskit-aer-gpu)
`qiskit-aer-gpu` cu11/cu12 wheels do not target Blackwell `sm_120`. On this hardware
the GPU quantum-simulation path is CUDA-Q + cuQuantum (PennyLane `lightning.gpu`,
SM 7.0+). Qiskit (CPU/Aer-CPU) is still used for circuit authoring and small sims.
