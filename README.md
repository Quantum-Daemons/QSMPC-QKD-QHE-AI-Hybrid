# QSMPC · QKD · QHE · AI — Quantum-Safe Hybrid Orchestration

> An end-to-end orchestration **security layer** that chains **QKD → QSMPC → QHE/FHE →
> privacy-preserving AI** on a **PQC-first hybrid** foundation, with industry use cases
> (finance / healthcare / genomics / telecom-defence / **satellite**), a **classical-vs-quantum
> benchmark** (latency, accuracy, numeric **security score**, **quantum advantage**, and an
> **Alice↔Bob** two-party E2E), and a researcher/business-facing **SPA** that runs **entirely
> client-side** in public-demo mode.

License: **Apache-2.0** · Python **3.12** · GPU: **NVIDIA RTX PRO 6000 Blackwell + CUDA 13**

This is a **research / proof-of-concept**. Real implementations where tractable, honest
simulation/scaffolds where not — every claim is tied to a primary source in
[`references/`](references/REFERENCES.md).

---

## What it does

Four workstreams (WS), unified by one pipeline and a crypto-agility registry:

| WS | Theme | What's implemented |
|----|-------|--------------------|
| **WS1** | **AI-QKD** parameter optimization | BB84/decoy-state simulation, QBER/SKR telemetry, eavesdropper detection, ML hook |
| **WS2** | **QML-IDS** intrusion detection | 8 QKD attack scenarios; RandomForest baseline + PennyLane variational quantum classifier |
| **WS3** | **QSMPC × QKD** | Shamir + d-level QFT secure summation (faithful to the paper example), quantum oblivious transfer, quantum-safe PSI |
| **WS4** | **QHE × AI** | Quantum one-time pad (perfect hiding), Clifford+T QCNN private inference, Rate-1 QFHE; classical hybrid = CKKS **linear** layers (TenSEAL CPU / OpenFHE CPU / HEonGPU GPU — commercial-OK) + **MPyC** secret-sharing **non-linear** layers (ReLU/argmax) — incl. a **genuine FHE convolution** (im2col → CKKS `dot_plain`) so the medical use case runs a real small CNN on **real MedMNIST images** under encryption |

**Crypto-agility is the headline library strategy**: every algorithm (KEM / QKD / FHE / **MPC** /
signature) is selected from a runtime registry and swappable from the WebUI — a future PQC
migration is a config change, not a rewrite (see [`docs/crypto-agility.md`](docs/crypto-agility.md)).

Cross-cutting: **PQC-first hybrid** (ML-KEM-768 + X25519, ML-DSA-65), **crypto-agility**
(swap algorithms at runtime), **secure-aggregation federated learning** — including a **Quantum
Federated Learning** option (a Flower + PennyLane hybrid quantum-classical VQC, mirrored in-browser as
a per-use-case toggle), **explainable AI**
(exact linear-SHAP attributions, ML-DSA-signed for a tamper-evident audit record — EU AI Act
Art. 13), an **E2E orchestration** engine with **Run / Pause / Resume / Step / Stop / Restart**
controls, live WebSocket event streaming, config-driven parameters editable in the WebUI, and
high-DPI PNG / animated-GIF export of the run. An **Agentic AI · MCP** capability runs a small language
model **entirely in the browser** (WebGPU via Transformers.js + WebLLM, Apache-2.0) whose
request/response is sealed + signed over a **client-side quantum-safe MCP** channel — agentic AI with a
quantum-safe layer, zero server compute. It is **integrated into every use-case page** as a per-industry
panel (its own MCP tool — `score_transaction` / `query_cohort` / `pool_allele_counts` /
`inspect_qkd_telemetry` / `relay_mission_telemetry` — fed by that run's live metrics), so MCP is the
per-use-case **cross-cutting integration layer** (no standalone page — it lives inside each industry use
case). The **full
deployment** additionally runs a **real FastMCP server** exposing those 5 tools, bound to the
quantum-safe transport so a `tools/call` returns a **signed real result** (plus interop with the
official filesystem MCP server); the public demo keeps the zero-server client mirror.

The **public demo computes real privacy-preserving inference in the browser** — node-seal **CKKS**
(a genuine homomorphic linear layer) + a real **emp-wasm garbled-circuit 2PC** for the non-linear
argmax — so its FHE/MPC figures are **genuinely measured client-side, zero VPS compute**.

### Industry use cases

| Use case | Pipeline | Industry |
|----------|----------|----------|
| `finance_aml` | PQC → QKD → **PSI** (cross-bank blacklists) → FHE fraud inference | finance / AML |
| `medical_fl` | PQC → QKD → **federated training** + secure aggregation → private diagnosis | healthcare |
| `genomics` | PQC → QKD → **QSMS secure summation** (pooled allele counts) → FHE | genomics |
| `telecom_ids` | PQC → QKD (w/ eavesdropper) → **QML-IDS** attack detection | telecom / defence |
| `satellite_comms` | PQC → **free-space QKD** (BB84 downlink) → **QSDC** direct telemetry (inline eavesdrop check) | satellite |

Each use case has its **own page** (`/usecase/:id`) — a **live two-party simulator** you Start /
Pause / Step / Stop / Restart. A **single Run** drives both the animated E2E protocol **and** the
benchmark: the benchmark is **assembled from that very run** (no separate "Run benchmark" button) —
the MPC-FHE row is the run's **real measured FHE-linear + MPC-non-linear latency**, shown next to the
QSMPC-FHE comparison, quantum advantage, security score, and (for finance/medical/genomics) a signed
SHAP explanation. The two-party diagram makes the **MPC-FHE secret-sharing over a LAN↔WAN↔LAN** link
explicit — linear layers cross as a **CKKS ciphertext**, non-linear layers as **secret-shares**. In
the **public demo** (`?demo=client`) every use case runs **entirely in the browser** with **real,
measured FHE/MPC** (node-seal CKKS + an emp-wasm garbled-circuit 2PC) and an optional in-browser
**Quantum Federated Learning** toggle — zero `/api` compute; the **full local** deployment adds the
server-side Flower + PennyLane QFL + the real MCP server. See [`docs/architecture.md`](docs/architecture.md)
for the full diagram.

---

## Quickstart

```bash
# 1. Python 3.12 venv + core (crypto + quantum CPU) stack
bash env/setup_venv.sh            # or: make setup
source .venv/bin/activate

# 2. Run the test suite (crypto correctness, FHE accuracy, E2E smoke)
make test                          # full suite — 116 passing, 1 skipped (use `-m "not slow"` to skip Node/GPU-gated)

# 3. Optional tiers
pip install -r requirements-ai.txt    # torch (CUDA 13 auto), Flower + PennyLane (QFL), XGBoost, MedMNIST, ONNX, MCP SDK
pip install -r requirements-fhe.txt   # TenSEAL + OpenFHE (commercial-OK CKKS)
pip install -r requirements-api.txt   # FastAPI backend

# 4. Run an E2E use case (writes results/ artifacts)
python -m qsh.orchestration.pipeline --usecase finance_aml --out results/

# 5. Classical-vs-quantum benchmark
bash scripts/run_benchmark.sh linear

# 6. Web: API + SPA (two terminals)
uvicorn api.main:app --port 8000           # backend
cd web && npm install && npm run dev        # SPA at http://localhost:5173
```

GPU quantum simulation (Blackwell): see **Dev environment** below — use CUDA-Q /
PennyLane `lightning.gpu`, **not** `qiskit-aer-gpu`.

---

## Repository layout

```
src/qsh/           the qsh package — orchestration, pqc, qkd, qml_ids, qsmpc, qhe,
                   fhe, mpc, ai, mcp (real FastMCP server), usecases, benchmark, data, monitoring
api/               FastAPI backend (REST + WebSocket + downloads + /metrics)
web/               React + Vite + TS SPA (Overview, Value, Validations, E2E Orchestration,
                   Use Cases index + 5 per-use-case simulator pages /usecase/:id)
docker/            Dockerfiles + compose (full + client-only builds) + Caddyfile
monitoring/        Prometheus config + Grafana notes
references/        primary-source PDFs + annotated REFERENCES.md
submodules/        upstream OSS (added by scripts/bootstrap_submodules.sh)
scripts/           bootstrap_submodules.sh · run_benchmark.sh · build_gpu_mpc.sh
docs/              architecture · crypto-agility · threat-model · pqc-roadmap
tests/             pytest suite
```

---

## Dev environment

Target box: **AlmaLinux 9.7, i5-13600K, 128 GB DDR5, NVIDIA RTX PRO 6000 Blackwell 96 GB,
CUDA 13.0**. Verified: Python 3.12.11, gcc 11.5, cmake 3.31, Node 22, Docker 29,
`torch 2.12+cu130` with `cuda.is_available() == True`.

### ⚠️ GPU quantum simulation on Blackwell
`qiskit-aer-gpu` (cu11/cu12 wheels) **does not support Blackwell `sm_120`**. Use:
- **CUDA-Q** (`cudaq`) and/or **PennyLane `lightning.gpu`** (wraps cuQuantum cuStateVec, SM 7.0+).
- **Qiskit / Qiskit-Aer (CPU)** for circuit authoring and small simulations (used here).
- See [`requirements-gpu.txt`](requirements-gpu.txt).

### FHE backends (commercial-OK only)
- **TenSEAL** (Apache-2.0, CPU CKKS via Microsoft SEAL) — the always-available default (`TenSEALBackend`).
- **OpenFHE** (BSD-2-Clause, v1.5.1 Apr-2026, CPU CKKS **with approximate bootstrapping**) — the live
  `OpenFHEBackend` via the `openfhe` wheel; it needs a recent libstdc++ (Ubuntu 24.04 / the Debian-trixie
  Docker image), and is import-guarded (falls back to TenSEAL) on older hosts.
- **HEonGPU** (Apache-2.0, GPU CKKS/BFV + collective bootstrapping) — the GPU path, exercised live by the
  benchmark glue (`gpu_mpc_adapter`); no Python binding, so it is the GPU benchmark backend, not a per-run selector option.

> **DESILO `desilofhe` was removed.** Its license is **non-commercial** — commercial use requires a separate
> paid license ([fhe.desilo.dev/latest/license](https://fhe.desilo.dev/latest/license/)). To keep the stack
> unambiguously commercial-OK, the FHE layer uses TenSEAL + OpenFHE + HEonGPU only.

### Non-linear MPC backend (the hybrid's other half)
- **MPyC** (MIT, pure-Python, v0.11 2026, vendored at `submodules/mpyc`) — Shamir secret-sharing with secure
  comparison drives the **non-linear** layers (ReLU/argmax) on shares, the canonical CrypTen `ReLU([x]) =
  [x]·[x>0]` pattern. Wired in `src/qsh/mpc/secret_inference.py`; selectable as the `mpc` crypto-agility family
  (`mpyc-shamir` default; MP-SPDZ / hpmpc-gpu as heavyweight references). The linear layers stay under FHE — see
  the hybrid benchmark below.

### Languages
Python-first. **C++** is used only via the optional GPU-MPC submodules (HEonGPU / hpmpc).
Go was evaluated and not needed; Julia is optional and deferred.

---

## OSS / submodules

Add upstream OSS (all commercial-OK, verified maintained ≤6 months as of June 2026):

```bash
bash scripts/bootstrap_submodules.sh         # light set (CQC-QHE, AMLSim, shark, mpyc)
QSH_SET=all bash scripts/bootstrap_submodules.sh   # + OpenFHE, SEAL, liboqs, MP-SPDZ,
                                                   #   HEonGPU, hpmpc, synthea
```

| Repo | Role | License |
|------|------|---------|
| open-quantum-safe/liboqs (+ `liboqs-python` pip) | PQC ML-KEM/ML-DSA | MIT |
| OrtegaSA/CQC-QHE-repo | QHE + Clifford+T QCNN (WS4) | Apache-2.0 |
| openfheorg/openfhe-development, microsoft/SEAL, `tenseal` (pip) | classical FHE (linear layers) | BSD-2 / MIT |
| **openfhe** (pip, CPU) | CKKS + approximate bootstrapping (commercial-OK) | BSD-2-Clause |
| **lschoe/mpyc** (vendored `submodules/mpyc`, pip) | **non-linear MPC** — Shamir secret-sharing ReLU/argmax (the hybrid's non-linear half) | **MIT** |
| Alisah-Ozcan/HEonGPU, chart21/hpmpc (PIGEON), kanav99/shark | GPU MPC-FHE benchmark (our own glue) | Apache-2.0 / MIT |
| data61/MP-SPDZ | heavyweight production MPC reference | CSIRO |
| IBM/AMLSim, synthetichealth/synthea | synthetic finance / EHR data | Apache-2.0 |

**Client-side / AI OSS (npm, all commercial-OK):** `node-seal` (SEAL→WASM CKKS, MIT) +
`mpc-framework`/`emp-wasm-engine`/`summon-ts` (garbled-circuit 2PC, MIT) for the demo's real in-browser
FHE/MPC; `@huggingface/transformers` + `@mlc-ai/web-llm` (Apache-2.0) for in-browser AI; `mcp` Python SDK
(MIT, FastMCP server) + the official `@modelcontextprotocol/server-filesystem` (MIT) for the real MCP
server (full deployment). Full table + dataset licenses in [`DATA_LICENSES.md`](DATA_LICENSES.md).

**Originality note:** the GPU MPC-FHE benchmark is our own integration over the *upstream*
projects (`src/qsh/benchmark/gpu_mpc_adapter.py` + `scripts/build_gpu_mpc.sh`). We do **not**
clone or vendor any mixed-authorship aggregate repo.

---

## Datasets

Free/public preferred, synthetic fallback (privacy-safe, dependency-free):
- **Finance/AML**: PaySim, Kaggle credit-card-fraud (CC0), IBM AMLSim, Elliptic — synthetic generator built in.
- **Medical**: MedMNIST v2 (CC-BY), Synthea (synthetic EHR), Wisconsin breast-cancer (bundled).
- **Genomics**: 1000 Genomes (CC-BY), OpenSNP (CC0) — synthetic genotypes by default (re-identification caveat).

Loaders + generators: `qsh.data.loaders` / `qsh.data.synthetic`.

---

## Benchmark — MPC-FHE vs QSMPC-FHE

On each use-case page the benchmark is **unified with the live simulator** (Round 7): a single **Run**
executes the E2E pipeline and the benchmark is **assembled from that run** — the MPC-FHE row carries the
run's real measured FHE-linear + MPC-non-linear latency, alongside the QSMPC-FHE comparison + quantum
advantage (the backend `QuantumComparisonStage` writes it into the ML-DSA-signed summary). The CLI below
runs the same comparison standalone.

`bash scripts/run_benchmark.sh linear` (or `python -m qsh.benchmark.runner --workload linear
--model-size medium --out results/`; artifacts → `results/benchmark_*.{csv,json}`, also viewable
via the benchmark panel on each use-case simulator page).

### Faithful hybrid: linear → FHE, non-linear → MPC

The classical path is a **genuine 2026-style MPC-FHE hybrid**, not CKKS-only. **Linear layers**
(matmul / conv) run **under CKKS/FHE** (TenSEAL or OpenFHE on CPU; HEonGPU on the Blackwell GPU),
while **non-linear layers** (ReLU, argmax) run **under secret-sharing MPC** — real Shamir
secure-comparison via **MPyC** (`src/qsh/mpc/secret_inference.py`), with a CKKS↔shares conversion at
each boundary. This mirrors the Gazelle → Delphi → BOLT → Bumblebee → **BLB** line and EncFormer, so
the benchmark reports the **FHE-linear vs MPC-non-linear latency split + MPC communication**, not a
single CKKS number. The **QSMPC-FHE** quantum analogue is **Clifford gates (linear, cheap) vs T-gates
(non-linear, expensive)** — but QOTP / Rate-1 QFHE evaluate *both* on the encrypted state (ct/pt ≈
1.3), so there is **no FHE↔MPC boundary**; the secure-comparison role is played by the QSMS/QOT
quantum-MPC path.

### Model-size tiers + per-use-case architectures

The model is **per use case**, not a single generic shape (`qsh.benchmark.runner.usecase_arch_layers` /
`usecase_arch_specs`, mirrored client-side): web-grounded 2026 parameter budgets — **finance** = tabular
fraud MLP, **medical** = a **genuine small CNN on real MedMNIST images**, **genomics** = SNP/variant risk
MLP. The **workload** selector changes the architecture *class* — `linear` (shallow) vs `cnn` vs
`transformer`.

**Medical + `cnn` is a real encrypted convolution, not a cost proxy.** When the medical use case loads
MedMNIST (`config['dataset']='medmnist'` / the CNN workload), the pipeline federated-trains a tiny CNN
(one conv block 1→4 ch 3×3, avg-pool, dense head at 14×14) and runs it under the hybrid as a **genuine FHE
convolution** — im2col → CKKS `dot_plain` per output channel (`src/qsh/fhe/conv.py`), with ReLU/max-pool
under MPC (`secure_relu` / `secure_maxpool`). The convolution arithmetic happens on the **encrypted image**;
the FHE-linear conv cost genuinely dominates (~4.5 s for one 14×14 image on CPU CKKS, decision faithful to
plaintext). This is FHE-feasible by design — a MedMNIST-scale CNN, **not** ResNet/ViT (FHE cannot run SOTA
CNNs/LLMs under encryption — a fundamental crypto-cost limit, sold honestly as a PoC).

For the other industries the encrypted hybrid preserves the plaintext decision regardless of weights, so
the metric is FHE/MPC **cost + fidelity**, not task SOTA; lightweight === the trained linear model, so SHAP
stays exact. The benchmark shows the architecture label + parameter count.

A **lightweight / medium / large** selector then scales it. **Lightweight + medium run end-to-end for real**
(CKKS-FHE linear + MPyC-MPC non-linear); **large is clearly labelled *projected*** — extrapolated from the
measured medium per-op cost (MAC / comparison counts) so the page stays responsive. Representative results on
the Blackwell box (TenSEAL CPU + MPyC, `linear` workload):

| Path | Backend | Size | Latency/inf | FHE-linear | MPC-non-linear | MPC comm | Acc | ct/pt | Security |
|------|---------|------|-------------|-----------|----------------|----------|-----|-------|----------|
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | lightweight | ~43 ms | ~39 ms | ~4 ms | ~0.5 KB | 1.0 | ~1393× | **68 / 100** |
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | medium | ~243 ms | ~215 ms | ~28 ms | ~4.5 KB | 1.0 | ~1393× | **68 / 100** |
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | large \* | ~1885 ms | ~1719 ms | ~165 ms | ~24.5 KB | 1.0 | — | **68 / 100** |
| MPC-FHE | HEonGPU CKKS (GPU sm_120) + MPyC | medium | batched | GPU | CPU | — | — | — | **68 / 100** |
| QSMPC-FHE | QOTP + QCNN (Clifford+T) | — | ~4 ms | Clifford | T-gates ×630 | n/a | 1.0 | **~1.28×** | **96 / 100** |

\* **projected** from measured medium per-op cost, not run end-to-end. GPU FHE wins on **batched**
workloads; for a tiny single-vector input the per-call kernel overhead can exceed CPU (raw encrypt
throughput was ~700 ops/s GPU vs ~34 CPU). The headline: the quantum **Rate-1** path is ~10³× more
communication-compact than classical CKKS, with information-theoretic (vs computational) security —
and the hybrid now shows *where* the classical cost goes (FHE dominates the linear layers; MPC adds
the non-linear secure comparisons + a few KB of communication).

**Three enriched dimensions** (Round 3):
- **Security score (0–100)** — `qsh/benchmark/security.py`: a transparent weighted sum over
  HNDL/PQC-hybrid + QKD, security model (information-theoretic ≫ computational), NIST KEM
  category, eavesdropper detection (QBER), and the ~0.1 bit/weight leakage bound (PRX
  *Quantum-Secure Multiparty Deep Learning*). Reflects the **crypto-agility** selection.
- **Quantum advantage (0–10)** — `qsh/benchmark/quantum_advantage.py`: weighted geometric mean of
  communication-compactness (≈10³×), latency, and the security jump → a tiered score
  (*strong* ≈ 7.5/10 on this box).
- **Per-use-case two-party simulator** — each industry has its own page (`/usecase/:id`) with a
  **live two-party (Alice↔Bob) sequence diagram** (`web/src/components/TwoPartyDiagram.tsx`) whose
  steps light up as that use case's pipeline runs — **Start / Pause / Resume / Step / Stop / Restart
  drive it**. Parties are use-case-specific (Bank A↔Bank B, Hospital↔Aggregator, Cohort↔Cohort,
  Carrier↔Eve, Ground-station↔LEO-satellite), grounded in `references/` (Zhang QOT npj 2026, Li&Deng
  QHE-FL, Sutradhar QSMS). Labels fill from live metrics (QBER / PSI / federated acc / QSDC secrecy);
  exports a high-DPI PNG + a multi-second GIF. A standalone per-party-timing micro-benchmark also
  exists at `qsh/benchmark/alice_bob_e2e.py` (`POST /api/alice-bob`).

The benchmark, security score, and quantum advantage are computed server-side **and** mirrored
client-side (`web/src/lib/clientCompute.ts`) so the public demo shows identical numbers with no backend.

---

## Deployment

- **Full deployment (federated learning + inference)** — Docker image build → server startup:
  ```bash
  git clone <repo> qsh && cd qsh
  docker compose -f docker/docker-compose.yml build              # build the images (full AI tier + OpenFHE + Node/MCP)
  docker images | grep -E 'docker-(api|web)'                     # the freshly-built images
  docker compose -f docker/docker-compose.yml up -d              # start: API (+AI tier) · SPA · Prometheus · Grafana
  docker compose -f docker/docker-compose.yml ps                 # all services healthy
  #  SPA  http://localhost:8080   ·   API  http://localhost:8000/api/health   ·   Grafana  http://localhost:3000
  # run a federated use case end-to-end (writes results/):
  docker compose -f docker/docker-compose.yml exec api \
    python -m qsh.orchestration.pipeline --usecase medical_fl --out results/
  ```
  The full `Dockerfile.api` (Debian-trixie) **bakes in the AI tier** (CPU torch + Flower + PennyLane + MedMNIST),
  the **live OpenFHE** CKKS backend, and **Node/npx** for the real MCP server, so a federated `medical_fl` run +
  the MCP server work out of the box. Fits a **6-core / 12 GB / 100 GB** box (CPU; GPU passthrough via host
  `nvidia-container-toolkit` + the `deploy.resources` GPU block is optional and faster). Ports are env-overridable
  (`WEB_PORT` / `API_PORT` / `GRAFANA_PORT`).
- **Client-side build (implementation capability)**: the SPA also builds **fully client-side** — every
  computation runs in the browser (real CKKS FHE + an emp-wasm garbled-circuit 2PC, ONNX inference, BB84 /
  PSI / SHAP / QFL), with **zero `/api` compute**. In this mode the system needs **neither the backend
  FHE/MPC compute path nor the frontend FHE/MPC backend-selectors** — the client owns both — so it requires
  no GPU and no server-side ML. The same pages and buttons work in both modes.
- **Monitoring**: see [`monitoring/README.md`](monitoring/README.md) (Prometheus + Grafana for the local
  full deployment; Zabbix dropped as redundant).

---

## WS3 + WS4 challenges addressed

The four workstreams target specific open problems in quantum-safe + private AI; here is how this
implementation maps to each, and where gaps remain.

**WS3 — QSMPC×QKD** (cross-org AI restricted by GDPR/HIPAA; classical SMPC is complex, HNDL-vulnerable, sometimes assumes a TTP; Agentic-AI/MCP lacks quantum-safe + private agent comms):

| Sub-problem | How QSH addresses it | Where |
|---|---|---|
| Raw data must not be shared | Federated learning + secure aggregation; PSI (blacklists never exchanged); FHE private inference | `ai/federated.py`, `qsmpc/qot.py`, `fhe/` |
| Classical SMPC protocol complexity | QSMS = Shamir + d-level QFT (no Beaver triples / garbled circuits) | `qsmpc/secure_summation.py` |
| HNDL-vulnerable HE / key exchange | PQC-first hybrid **ML-KEM + X25519** (+ optional QKD key material) | `pqc/hybrid_kem.py`, `pqc/transport.py` |
| TTP assumption | TTP-free: QSMS (entangled state, no dealer), keyed-hash PSI over the quantum-safe channel, pairwise-mask secure aggregation | `qsmpc/`, `ai/federated.py` |
| **Agentic-AI / MCP quantum-safe + private agent comms** | A **real FastMCP server** (5 per-industry tools, full deployment) bound to a quantum-safe channel: two agents establish a PQC(+QKD) session, the caller sends an **ML-DSA-signed, AES-GCM-sealed** `tools/call`, the callee **dispatches the real tool** and returns the **signed result** (authenticated, confidential, HNDL-resistant, no TTP); also interops with the official filesystem MCP server | `pqc/mcp_bridge.py`, `mcp/server.py`, `/api/mcp/*` |
| **Explainability / audit trail (XAI)** | Exact linear-SHAP feature attributions for each private-inference decision, **ML-DSA-signed** → tamper-evident, regulator-facing explanation (EU AI Act Art. 13, GDPR Art. 22, SR 11-7); surfaced on the finance/medical use-case cards | `ai/explainability.py`, `ExplainabilityStage` |

**WS4 — QHE×AI** (FHE 10³–10⁶× overhead; bootstrapping bottleneck; no quantum encoding; plus unsolved (a)–(f)):

| Sub-problem | How QSH addresses it | Where |
|---|---|---|
| (1)/(e) huge compute + ciphertext/plaintext overhead | Benchmark *measures* ~1393× CKKS vs **Rate-1 QFHE ≈ 1.27×** | `benchmark/`, `qhe/rate1_qfhe.py` |
| (2) bootstrapping bottleneck | **OpenFHE CKKS `EvalBootstrap`** (CPU) + **HEonGPU collective bootstrapping** (GPU) refresh ciphertext levels for deeper circuits — both commercial-OK | `fhe/backends.py` (`OpenFHEBackend(use_bootstrap=True)`) |
| (3)/(a) quantum encoding / practical QHE | QOTP + Clifford+T QCNN (CQC-QHE) + Rate-1 QFHE at small scale | `qhe/qotp.py`, `qhe/qcnn.py` |
| (b) QFL gradient leakage (arXiv:2508.15998) | Pairwise-mask secure aggregation + QOTP-delegated learning | `ai/federated.py` |
| (c)/(f) circuit privacy vs malicious server **and** hiding server circuit from a malicious client | **QOTP perfect hiding** (server sees the maximally-mixed state) + **Pauli-gate concealment** (wrong key ⇒ scrambled result) | `qhe/qotp.py` (`pauli_concealment_demo`) |
| (d) efficiency on AI/HPC infra | OpenFHE (CPU) + HEonGPU (GPU) + CROSS (reference), HEonGPU shown on the Blackwell GPU | `fhe/`, `benchmark/` |

**Residual gaps / recommendations**: malicious-secure MPC (PQ-OPRF PSI + identifiable abort);
hardware QKD field test (vs the BB84 simulator); larger fault-tolerant QHE; production bootstrapping
throughput. (A real quantum-safe MCP server + filesystem-MCP interop now ships in the full deployment.)
See [`docs/pqc-roadmap.md`](docs/pqc-roadmap.md).

---

## Concerns & recommendations

- **Simulation vs reality** — QKD distance/QBER gains, photonic inference, decoy-state, full
  qOLE, and the **free-space satellite QKD + QSDC** path are **simulated** (analytic channel +
  information-structure models), not hardware-validated. QHE evaluation is exact for Clifford
  circuits; non-Clifford layers are costed as T-gates, not run fault-tolerantly.
- **NSA stance** — QKD is used strictly as a PQC-hybrid add-on, never standalone (CNSA 2.0).
- **GPU/vendor & licensing** — Blackwell+CUDA13 narrows the quantum-GPU stack to CUDA-Q/cuQuantum.
  The FHE stack is deliberately **commercial-OK** (TenSEAL Apache-2.0, OpenFHE BSD-2, HEonGPU Apache-2.0);
  **DESILO was removed** because its license is non-commercial. Multiple swappable backends avoid lock-in.
- **Scope discipline** — QHE kept to 6–8 qubit QCNN for reproducibility. The demo's **QSMPC-FHE (quantum)
  comparison row is a genuine in-browser state-vector simulation** (8-qubit, QOTP + Clifford+T QCNN): its
  latency is **measured** and its T-gate count is **counted from the circuit** — a *simulation*, not quantum
  hardware (no QPU); under QOTP the ciphertext is the same #qubits as the plaintext (ct/pt = 1). The classical
  **MPC-FHE row is real measured crypto** (node-seal CKKS + emp-wasm 2PC) — so the demo shows measured, not
  estimated, numbers.
- **In-browser AI (demo)** — the per-use-case Agentic-AI panel runs **small** models client-side (Transformers.js
  on onnxruntime-web; WebLLM on WebGPU). The ONNX execution provider prefers **WebGPU → WebNN → WASM**,
  selected by a genuine `navigator.gpu.requestAdapter()` probe (so the reported backend is real, never a
  false "webgpu" that silently ran on WASM). Model weights stream from the HF/MLC **CDN** on first run
  (browser-cached after) — zero VPS bandwidth/compute. WebLLM needs WebGPU; without it the page falls
  back to Transformers.js (WASM) and the quantum-safe MCP envelope still runs. The in-browser models
  illustrate the agentic-AI + MCP flow; they are not the production inference path.
- **MCP** — the **full deployment ships a real MCP server**: a **FastMCP** server (MCP Python SDK, MIT)
  exposing the 5 per-industry tools, bound to the quantum-safe transport so a `tools/call` runs end-to-end
  under PQC+QKD+ML-DSA+AEAD and returns the **signed real result** (`src/qsh/mcp/server.py`,
  `pqc/mcp_bridge.secure_agent_call`, `/api/mcp/{tools,call}`); it also **interops with the official
  `@modelcontextprotocol/server-filesystem`** under the same quantum-safe seal (`mcp/fs_interop.py`,
  `/api/mcp/filesystem`). The public demo keeps the zero-server `clientMcp.ts` mirror (**zero per-user VPS
  load**). In the WebUI the in-browser agent is **integrated into each use-case page** (its own per-industry
  MCP tool, fed by the live run) — the standalone Agentic-AI page was **removed**, since the report frames MCP
  as a per-use-case cross-cutting layer, not a separate feature.
- **Demo vs full** — the public demo is **client-only** (zero `/api` compute) and ships a slim backend
  image (no FHE/torch). Its FHE/MPC is **genuinely real and measured in the browser**: **node-seal CKKS**
  (a real homomorphic linear layer, ~1e-9 error) + a real **emp-wasm garbled-circuit 2PC** for the
  non-linear argmax, both parties loopback in one tab (zero VPS). Federated learning is offered as an
  in-browser **Quantum Federated Learning** toggle (a real client-side FedAvg of a quantum-classical
  model); the full deployment runs the real **Flower + PennyLane** QFL + server-side FHE inference (incl.
  the genuine medical CNN under real encrypted convolution — in-browser, node-seal CKKS does the
  linear/argmax, so only the CNN's conv cost is approximated client-side).
- **Per-industry model realism (scorecard)** — the **crypto/protocol layer is real across all five
  industries** (PSI/QOT, QSMS secure-sum, BB84/QSDC, MPyC MPC, CKKS FHE, federated secure-aggregation),
  and **each vertical now runs a real industry-standard 2026 model** on commercial-OK data
  ([`DATA_LICENSES.md`](DATA_LICENSES.md)):
  - **finance** — a real **XGBoost** fraud model (the AML standard) distilled to a CKKS-linear surrogate
    for the encrypted inference (the tree ensemble runs plaintext; the surrogate runs under FHE);
  - **medical** — a genuine small **CNN on real MedMNIST images** under a real FHE convolution + MPC;
  - **genomics** — a real published **Polygenic Risk Score** (PGS Catalog PGS000001, 77-SNP breast-cancer)
    over **real 1000 Genomes phase-3 genotypes** (effect-allele dosages for those SNPs, from the open Ensembl/
    IGSR API — both the model *and* the genotypes are real), computed as an **exact CKKS dot-product** — the rare
    case that is *both* the clinical standard *and* FHE-native (matching the published HE-PRS work), per-SNP signed SHAP;
  - **telecom/defense** — a real classical RF + a **Quantum SVC** (quantum-kernel SVM, the 2026 QML-IDS
    standard, F1 ≥ 0.93) + a variational QNN, on literature-grounded QKD-attack telemetry;
  - **satellite** — a real linear **telemetry-anomaly classifier** run under CKKS on the encrypted
    downlinked telemetry (ESA-ADB-style; real ESA-ADB CC-BY-3.0-IGO substitutable).

  Honest crypto-scope: the **full-deployment model is the real industry architecture**; the
  **encrypted inference is the FHE-feasible part** (linear / PRS / small CNN / distilled surrogate),
  because FHE cannot run SOTA CNNs/LLMs/tree-ensembles under encryption (a fundamental crypto-cost
  limit). Sell it as a high-fidelity PoC, not a trained production system.
- **Public-demo data (real where commercial-OK, synthetic otherwise)** — the client-side demo loads **real
  commercial-OK datasets where shippable**, mirroring the full deployment: **medical** uses a real Wisconsin
  Diagnostic Breast-Cancer sample (UCI/sklearn, public domain) + a real logistic-regression model, and
  **genomics** uses the real PGS-Catalog PGS000001 PRS weights (77 SNPs, CC-BY 4.0) → a real in-browser
  **Polygenic Risk Score** with per-SNP signed SHAP — both tiny static JSON under `web/public/data/`, loaded by
  [`web/src/lib/realData.ts`](web/src/lib/realData.ts) and computed with the real in-browser node-seal CKKS +
  secret-sharing MPC. finance/telecom/satellite have no shippable commercial-OK real set, so they use
  representative synthetic features (logged as such in the run). The bundled sets are public / de-identified, so
  no real *personal* data leaves the browser — the privacy property is demonstrated on real, realistic inputs.
- **Config single-source** — the benchmark cost model, tier architectures, workload multipliers,
  security/advantage weights and demo data sizes live in one place (`src/qsh/config/defaults.py`),
  served at `/api/config` and baked into `web/src/lib/config.defaults.json` (regenerate via
  `python scripts/export_config.py`), so the demo's numbers cannot drift from the backend.
- **Future tracks** — see [`docs/pqc-roadmap.md`](docs/pqc-roadmap.md): a NIST-aligned IDS
  against HNDL/Q-Day, hardware QKD field test, malicious-secure MPC, and the GPU FHE-at-scale
  path (HEonGPU + PIGEON/hpmpc).

---

## References & license

- Primary sources: [`references/REFERENCES.md`](references/REFERENCES.md) (annotated, paper → code → workstream).
- Standards/threat context: [`docs/pqc-roadmap.md`](docs/pqc-roadmap.md).
- Code license: **Apache-2.0** ([`LICENSE`](LICENSE)). Upstream OSS retains its own licenses (table above).

> Not included in this public repository: private business notes, git-ignored by design.
