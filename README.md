# QSMPC, QKD, QHE, AI - Quantum-Safe Hybrid Orchestration

> An end-to-end orchestration **security layer** that chains **QKD -> QSMPC -> QHE/FHE ->
> privacy-preserving AI** on a **PQC-first hybrid** foundation, with industry use cases
> (finance / healthcare / genomics / telecom-defence / **satellite**), a **classical-vs-quantum
> benchmark** (latency, accuracy, a **security posture table**, **quantum advantage**, and an
> **Alice <-> Bob** two-party E2E), and a researcher/business-facing **SPA** that runs **entirely
> client-side** in public-demo mode.

License: **Apache-2.0**, Python **3.12**, GPU: **NVIDIA RTX PRO 6000 Blackwell + CUDA 13**

This is a **research / proof-of-concept**. Real implementations where tractable, honest
simulation/scaffolds where not - every claim is tied to a primary source in
[`references/`](references/REFERENCES.md).

> **What is new here, and what is not.** This project's verification rule is that a novelty claim
> must survive an adversarial search for prior art. Applied to its own headline claim the answer is
> asymmetric, so both halves are stated.
>
> **MPC in a browser is not new.** webSPDZ (IACR ePrint 2025/487) compiles MP-SPDZ to WebAssembly
> over WebRTC or WebSockets, and names JIFF and MPyC-Web as earlier browser MPC engines. The
> garbled-circuit half here re-implements a known idea.
>
> **FHE evaluation in a browser is where the search came back empty.** Every artefact found encrypts
> client-side and evaluates server-side: IACR ePrint 2026/515 defines "client-side operations" as
> encryption and decryption while *"users offload large computations to servers"*, and Zama's TFHE-rs
> WASM API documents that it supports key generation, encryption and decryption but **not FHE
> computation**, citing WASM threading and the 2 GB memory ceiling. This demo evaluates a real CKKS
> dot product and a real CKKS convolution in the tab.
>
> **So the claim is the composition, not either half:** PQC key establishment to QKD-derived key
> material to secure multiparty computation to homomorphic evaluation to model inference, chained end
> to end in one browser tab with zero server compute. A search for an integrated browser system doing
> all of that returned nothing. Two caveats travel with it: the two MPC parties are notional - one
> tab, an in-memory transport, real protocol and bytes and wall-clock but no genuine party separation
> - and the quantum channel is simulated, because no browser API for QKD exists. And client-side is
> not automatically private: WWW 2026's WAMO paper argues browser deployment moves a model into a
> hostile environment, which is one reason the models here are public rather than treated as secrets.

> **Positioning - quantum-safe, not quantum-advantage.** The value here is **quantum-safe / PQC**, a
> regulation-driven, fast-growing market (NIST FIPS 203/204/205; NSA CNSA 2.0 procurement mandate 2027;
> EU roadmap 2026/2030/2035; "harvest-now-decrypt-later"; Grand View Research sizes the PQC market at
> ~$1.6 B in 2025 reaching ~$20.5 B by 2033, a 37.8 % CAGR over 2026-2033, and the vendor spread on the
> base year is about 4x - see the market table on the Value page, which lists each house separately
> rather than blending them). A general **quantum *advantage*** for these workloads is **not** claimed
> as a market benefit - it is broadly unproven in 2026 (e.g. classical tensor-network simulation matching
> annealing results), so the numeric "quantum advantage" stays a **research / education** metric in the
> benchmark. For the QCNN specifically the evidence is sharper than "unproven": Bermejo, Braccia,
> Rudolph, Holmes, Cincio and Cerezo show that randomly initialised QCNNs **can only act on
> low-bodyness measurements of their input and are efficiently classically simulable** with Pauli
> shadows, and argue that QCNN successes largely reflect benchmark simplicity
> ([arXiv:2408.12739](https://arxiv.org/abs/2408.12739), rev. Feb 2026). That is the frame for this
> repository's own measured result - its trained QCNN loses to logistic regression on the same split,
> which is the expected outcome rather than a disappointing one, and both figures are printed side by
> side on the page. Two circuit widths ship for telecom so the effect of the register size can be
> read rather than assumed: the eighth wire encodes no feature and is still worth **0.079 macro F1**,
> while two of the eight classes score better without it
> ([`references/qcnn-width-ablation.md`](references/qcnn-width-ablation.md)). Relatedly, Li, Wu, Li and Yung separate observable concentration from loss of
> parameter sensitivity and show that **avoiding concentration is necessary but not sufficient for
> trainability** ([arXiv:2603.18479](https://arxiv.org/abs/2603.18479), Mar 2026, rev. Jun 2026); the
> untrained circuit this project removed from its inference path failed exactly the concentration
> symptom, with |<Z0>| under 0.21 for every input tested.
>
> The production-relevant output is the **security posture table** - per-row facts, each
> traceable to FIPS 203/204 or to the run that produced it. The 0-100 composite that used to hold that
> role was **retired in Round 22**: 14 of its 17 weights carried no stated rationale and it scored a
> run with zero secure key bits 100/100 (see
> [`references/security-scoring-fact-check.md`](references/security-scoring-fact-check.md)). The whole stack is deliberately a
> **hybrid** (PQC-classical + QKD/QHE quantum-simulated + FHE/MPC-classical + AI) - the pragmatic 2026
> pioneering-product shape.

---

## What it does

Four workstreams (WS), unified by one pipeline and a crypto-agility registry:

| WS | Theme | What's implemented |
|----|-------|--------------------|
| **WS1** | **AI-QKD** parameter optimization | BB84 **two-intensity decoy-state** simulation (Ma et al.'s Vacuum+Weak estimator with the vacuum yield modelled rather than measured - the paper's own two-intensity case, its section 3.5, bounds $`Y_0`$ instead) (+ a selectable **E91/CHSH** protocol), QBER/SKR telemetry, eavesdropper detection; a **published-reference decoy model** (`qkd/decoy_reference.py`) whose 13 tests assert Ma et al. PRA 72, 012326 (2005) values rather than this project's own - decoy reaches past 140 km where non-decoy dies near 40, and at the customary $`\mu = 0.1`$ a non-decoy link's tagged fraction exceeds 1 at **zero** distance, so it yields no key even back-to-back; a **QRNG** entropy source (state-vector quantum-measurement sim + SP 800-90B section 4.4 continuous health tests, an **SP 800-90B min-entropy estimate**, and a **Leftover-Hash-Lemma / seedless** Toeplitz extractor) that seeds the key material; a config-gated **leakage-resilient PQC+QKD key combiner**; and **OptiQKD** - a measured AI optimizer (scikit-learn surrogate + gradient-free search) tuning decoy (`mu_signal` / `mu_decoy`) + basis params to raise the secret-key rate vs default (with a convergence metric); the reference framework reports +14-25 % in simulation |
| **WS2** | **QML-IDS** intrusion detection | 8 classes = **7 QKD attack types + normal traffic**; RandomForest baseline + PennyLane variational quantum classifier. Scored on macro F1, not attack-vs-normal F1 |
| **WS3** | **QSMPC x QKD** | Shamir + d-level QFT secure summation (faithful to the paper example), quantum oblivious transfer, quantum-safe PSI |
| **WS4** | **QHE x AI** | Quantum one-time pad (perfect hiding), variational QCNN private inference (Clifford+T in the browser circuit; the Python module has no T gate), Rate-1 quantum **transport** (no `Eval`, so not QFHE); classical hybrid = CKKS **linear** layers (TenSEAL CPU / OpenFHE CPU / HEonGPU GPU - commercial-OK) + **MPyC** secret-sharing **non-linear** layers (ReLU/argmax) - incl. a **genuine FHE convolution** (im2col -> CKKS `dot_plain`) so the medical use case runs a real small CNN on **real MedMNIST images** under encryption |

**Crypto-agility is the headline library strategy** - in the EU NIS Cooperation Group sense
([PQC Roadmap](https://digital-strategy.ec.europa.eu/en/library/coordinated-implementation-roadmap-transition-post-quantum-cryptography),
23 Jun 2025, section 3.2): *modular design that enables replacing the cryptographic components*,
explicitly **not** cipher-suite negotiation during protocol execution. Every algorithm
(KEM / QKD / FHE / **MPC** / signature) is resolved from a registry and selected by the
operator (config / WebUI) **before a run**, so a future PQC migration is a configuration
change, not a rewrite. There is **no in-band negotiation and no silent fallback**
(see [`docs/crypto-agility.md`](docs/crypto-agility.md)).

Cross-cutting: **PQC-first hybrid** (ML-KEM-768 + X25519, ML-DSA-65 - the hybrid-first
approach the EU roadmap recommends), **crypto-agility**
(operator-selected, modular provider replacement), **secure-aggregation federated learning** - including a **Quantum
Federated Learning** option (a PennyLane hybrid quantum-classical VQC over this repo's own FedAvg; in the
browser it runs on every run rather than behind a toggle, over the page's own bundled sample), **explainable AI**
(exact linear-SHAP attributions, ML-DSA-signed for a tamper-evident audit record - aimed at EU AI Act
Art. 13, applicable 2 Dec 2027 for Annex III after the AI Omnibus deferral), an **E2E orchestration** engine with **Run / Pause / Resume / Step / Stop / Restart**
controls, live WebSocket event streaming, config-driven parameters editable in the WebUI (a per-page
**Parameters** panel - every QKD / QRNG / OptiQKD / IDS / data knob defaults from `config/defaults.py`
and is editable per run, with reset-to-defaults; no hardcoded values), and
high-DPI PNG / animated-GIF export of the run. An **Agentic AI, MCP** capability runs a small language
model **entirely in the browser** (WebGPU via Transformers.js + WebLLM, Apache-2.0) whose
request/response is sealed + signed over a **client-side quantum-safe MCP** channel - agentic AI with a
quantum-safe layer, zero server compute. It is **integrated into every use-case page** as a per-industry
panel (its own MCP tool - `score_transaction` / `query_cohort` / `pool_allele_counts` /
`inspect_qkd_telemetry` / `relay_mission_telemetry` - fed by that run's live metrics), so MCP is the
per-use-case **cross-cutting integration layer** (no standalone page - it lives inside each industry use
case). The **full
deployment** additionally runs a **real MCP server** (MCP Python SDK >=2.0 `MCPServer`; FastMCP is the
1.x fallback) exposing those 5 tools, bound to the
quantum-safe transport so a `tools/call` returns a **signed real result** (plus interop with the
official filesystem MCP server); the public demo keeps the zero-server client mirror.

The **public demo computes real privacy-preserving inference in the browser** - node-seal **CKKS**
(a genuine homomorphic linear layer) + a real **emp-wasm garbled-circuit 2PC** for the non-linear
argmax - so its FHE/MPC figures are **genuinely measured client-side, zero VPS compute**.

### Industry use cases

| Use case | Pipeline | Industry |
|----------|----------|----------|
| `finance_aml` | PQC -> QKD -> **PSI** (cross-bank blacklists) -> FHE fraud inference | finance / AML |
| `medical_fl` | PQC -> QKD -> **federated training** + secure aggregation -> private diagnosis | healthcare |
| `genomics` | PQC -> QKD -> **QSMS secure summation** (pooled allele counts) -> FHE | genomics |
| `telecom_ids` | PQC -> QKD (w/ eavesdropper) -> **QML-IDS** attack detection | telecom / defence |
| `satellite_comms` | PQC -> **free-space QKD** (BB84 downlink) -> **QSDC** direct telemetry (inline eavesdrop check) | satellite |

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
five motivations to avoid: [`references/satellite-motivation-fact-check.md`](references/satellite-motivation-fact-check.md).

Each use case has its **own page** (`/usecase/:id`) - a **live two-party simulator** you Start /
Pause / Step / Stop / Restart. A **single Run** drives both the animated E2E protocol **and** the
benchmark: the benchmark is **assembled from that very run** (no separate "Run benchmark" button) -
the MPC-FHE row is the run's **real measured FHE-linear + MPC-non-linear latency**, shown next to the
QSMPC-FHE comparison, quantum advantage, security posture, and (on all five use-case pages) a signed
SHAP explanation. The two-party diagram makes the **MPC-FHE secret-sharing over a LAN <-> WAN <-> LAN** link
explicit - linear layers cross as a **CKKS ciphertext**, non-linear layers as **secret-shares**. In
the **public demo** (`?demo=client`) every use case runs **entirely in the browser** with **real,
measured FHE/MPC** (node-seal CKKS + an emp-wasm garbled-circuit 2PC) and an optional in-browser
**Quantum Federated Learning** toggle - zero `/api` compute; the **full local** deployment adds the
server-side PennyLane QFL + the real MCP server. See [`docs/architecture.md`](docs/architecture.md)
for the full diagram.

---

## Quickstart

```bash
# 1. Python 3.12 venv + core (crypto + quantum CPU) stack
bash env/setup_venv.sh      # or: make setup
source .venv/bin/activate

# 2. Run the test suite (crypto correctness, FHE accuracy, E2E smoke)
make test             # pytest prints the count; the Node/OpenFHE-gated slow tests are deselected
# No count is quoted here. This line read "187 collected, 183 passed" while the suite had grown to
# 393 - and `Makefile` two files over already carries the same lesson from its own help text, which
# said 174 when the truth was 236. A count maintained by hand in a comment drifts; pytest reports it.
# `make test` already passes -m "not slow", which is what deselects the gated ones.
# `make test-all` runs every test, and on AlmaLinux 9.7 the OpenFHE bootstrap test SIGSEGVs inside
# EvalBootstrapSetup: fhe/backends.py guards against an ImportError, and a segfault is not one.

# 3. Optional tiers
pip install -r requirements-ai.txt  # torch (CUDA 13 auto), PennyLane (QFL), XGBoost, MedMNIST, ONNX, MCP SDK
pip install -r requirements-fhe.txt  # TenSEAL + OpenFHE (commercial-OK CKKS)
pip install -r requirements-api.txt  # FastAPI backend

# 4. Run an E2E use case (writes results/ artifacts)
python -m qsh.orchestration.pipeline --usecase finance_aml --out results/

# 5. Classical-vs-quantum benchmark
bash scripts/run_benchmark.sh linear

# 6. Web: API + SPA (two terminals)
uvicorn api.main:app --port 8000      # backend
cd web && npm install && npm run dev    # SPA at http://localhost:5173
```

GPU quantum simulation (Blackwell): see **Dev environment** below - use CUDA-Q /
PennyLane `lightning.gpu`, **not** `qiskit-aer-gpu`.

---

## Repository layout

```
src/qsh/      the qsh package - orchestration, pqc, qkd, qml_ids, qsmpc, qhe,
                   config, models, qrng, fhe, mpc, ai, mcp (real MCP server, SDK >=2.0), usecases, benchmark, data, monitoring
api/        FastAPI backend (REST + WebSocket + downloads + /metrics)
web/        React + Vite + TS SPA (Overview, Value, Validations, E2E Orchestration,
                   Use Cases index + 5 per-use-case simulator pages /usecase/:id)
docker/      Dockerfiles + compose (full + client-only builds) + Caddyfile
monitoring/    Prometheus config + Grafana notes
references/    primary-source PDFs + annotated REFERENCES.md
submodules/    upstream OSS (added by scripts/bootstrap_submodules.sh)
scripts/      bootstrap_submodules.sh . run_benchmark.sh . build_gpu_mpc.sh . run_gpu_mpc.sh
          export_config.py . prep_real_kaggle_data.py . publish_hf.py
web/scripts/  check_mlkem_modulus_vectors.mjs - `npm run check:pqc`, FIPS 203 section 7.2
          negative vectors + FIPS 204 round-trip against the browser PQC library
  train/      per-vertical training: finance_aml . medical_breastmnist . genomics_prs . satellite_telemanom
  data/      fetch_g1k_dosages.py (1000 Genomes provenance)
docs/       architecture . crypto-agility . threat-model . pqc-roadmap
            deliverable-verification . verification-log . deployment
tests/       pytest suite
```

---

## Dev environment

Target box: **AlmaLinux 9.7, i5-13600K, 128 GB DDR5, NVIDIA RTX PRO 6000 Blackwell 96 GB,
CUDA 13.0**. Verified: Python 3.12.11, gcc 11.5, cmake 3.31, Node 22, Docker 29,
`torch 2.13.0+cu130` with `cuda.is_available() == True`, nvcc V13.0.88, driver 580.105.08
(CUDA 13 wheels require 580.65.06 or newer on Linux). Re-measured 2026-08-14; the line read
`torch 2.12+cu130` until then. CUDA 13.0 became a **stable** PyTorch build target in 2.11, so this
box is no longer on a leading edge - `TR2`/`TR3` are what verify the capability, by launching a real
kernel and checking `get_arch_list()` contains the device's `sm_120`, rather than by trusting a
version string here.

### NOTE GPU quantum simulation on Blackwell
`qiskit-aer-gpu` (cu11/cu12 wheels) **does not support Blackwell `sm_120`**. Use:
- **CUDA-Q** (`cudaq`) and/or **PennyLane `lightning.gpu`** (wraps cuQuantum cuStateVec, SM 7.0+).
- **Qiskit / Qiskit-Aer (CPU)** for circuit authoring and small simulations (used here).
- See [`requirements-gpu.txt`](requirements-gpu.txt).

### FHE backends (commercial-OK only)
- **TenSEAL** (Apache-2.0, CPU CKKS via Microsoft SEAL) - the always-available default (`TenSEALBackend`).
- **OpenFHE** (BSD-2-Clause, v1.5.1 Apr-2026, CPU CKKS **with approximate bootstrapping**) - the live
  `OpenFHEBackend` via the `openfhe` wheel; it needs a recent libstdc++ (Ubuntu 24.04 / the Debian-trixie
  Docker image), and is import-guarded (falls back to TenSEAL) on older hosts.
- **HEonGPU** (Apache-2.0, GPU CKKS/BFV + collective bootstrapping) - the GPU path, exercised live by the
  benchmark glue (`gpu_mpc_adapter`); no Python binding, so it is the GPU benchmark backend, not a per-run selector option.

> **DESILO `desilofhe` was removed.** Its license is **non-commercial** - commercial use requires a separate
> paid license ([fhe.desilo.dev/latest/license](https://fhe.desilo.dev/latest/license/)). To keep the stack
> unambiguously commercial-OK, the FHE layer uses TenSEAL + OpenFHE + HEonGPU only.

### Non-linear MPC backend (the hybrid's other half)
- **MPyC** (MIT, pure-Python, v0.11 2026, a pip dependency - the submodule was dropped in Round 17) - Shamir secret-sharing with secure
  comparison drives the **non-linear** layers (ReLU/argmax) on shares, the canonical CrypTen `ReLU([x]) =
  [x], [x>0]` pattern. Wired in `src/qsh/mpc/secret_inference.py`; selectable as the `mpc` crypto-agility family
  (`mpyc-shamir` default; MP-SPDZ / hpmpc-gpu as heavyweight references). The linear layers stay under FHE - see
  the hybrid benchmark below.

### Languages
Python-first, plus **TypeScript** for the browser tier. **C++** is used only via the optional GPU-MPC
submodules (HEonGPU / hpmpc). Go was evaluated and not needed.

**Julia: DECLINED 2026-08-12 (Round 33), and it should have been declined twenty-nine plans ago.** It was
listed as "optional (defer)" in the first plan and this line said "optional and deferred" ever since. The
word appears in exactly one of thirty plan files and in no instruction after the first, and **no condition
was ever attached to it** - so it could not even fail the three-round rule in
[`docs/deliverable-verification.md`](docs/deliverable-verification.md), because an item with no condition
never becomes due. Two reasons, both checkable:

- **No in-browser path exists.** The browser tier is the deliverable, and Julia's WebAssembly target is
  still experimental in 2026: `ExportWebAssembly.jl` is marked work-in-progress, Julia's task runtime needs
  its stacks managed explicitly in the wasm heap rather than on the native WebAssembly stack, and only a
  few packages are built for the wasm platform through BinaryBuilder's experimental support. This is the
  same test that declined Rust for the client-side stack: not "is the language good" but "can the tab load
  it and evaluate".
- **There is no numerical gap for it to close.** The numerical work runs through numpy, torch and
  PennyLane, and the paths that dominate wall clock are already compiled native code - SEAL through
  Emscripten (`node-seal`), torch's kernels, `emp-wasm`. Julia's argument is speed against interpreted
  Python, and the hot loops here are not interpreted Python.

**Reopening condition (both halves required, so it can actually fire):** a numerical kernel appears here
that is neither a SEAL/`node-seal` call nor a torch/numpy call and is measured to be a run's bottleneck in
pure Python, **and** Julia's wasm target stops being experimental - a Julia release listing wasm32 as a
supported platform, or `ExportWebAssembly.jl` out of WIP.

---

## OSS / submodules

Only the two submodules that code actually builds against are vendored:

```bash
QSH_SET=all bash scripts/bootstrap_submodules.sh  # HEonGPU + hpmpc (the GPU MPC-FHE benchmark path)
# The bare command defaults to QSH_SET=min and both entries are tagged `all`, so it adds nothing.
```

Round 17 removed ten others (CQC-QHE, AMLSim, shark, mpyc, oqs-provider, openfhe, SEAL, liboqs,
MP-SPDZ, synthea). None was referenced by any import, script or Dockerfile; five were superseded by
the pip/npm package of the same project; and `.dockerignore` excludes `submodules/` wholesale, so
none of them ever reached an image. Cloning ~10 GB of C++ that nothing links against is a cost with
no matching capability. They remain **cited as upstream work** in
[`references/REFERENCES.md`](references/REFERENCES.md), which is the accurate relationship - using a
library through its published package does not require vendoring its source tree.

| Dependency | How it is used | How it is obtained | License |
|---|---|---|---|
| `liboqs-python` | PQC ML-KEM / ML-DSA / SLH-DSA | pip | MIT |
| `tenseal` | CKKS (default FHE backend) | pip | Apache-2.0 (over Microsoft SEAL, MIT) |
| `openfhe` | CKKS + approximate bootstrapping | pip; on RHEL 9 build with [`env/build_openfhe_el9.sh`](env/build_openfhe_el9.sh) | BSD-2-Clause |
| `mpyc` | **non-linear MPC** - Shamir ReLU/argmax, the hybrid's non-linear half | pip | MIT |
| `pennylane` | variational quantum layers (QFL, QML-IDS) | pip | Apache-2.0 |
| `qiskit` / `qiskit-aer` | circuit authoring + state-vector simulation | pip | Apache-2.0 |
| **Alisah-Ozcan/HEonGPU** | GPU CKKS for the MPC-FHE benchmark | **submodule** | Apache-2.0 |
| **chart21/hpmpc** (PIGEON) | GPU secret-sharing MPC for the same benchmark | **submodule** | MIT |

**Client-side / AI OSS (npm, all commercial-OK):** `@noble/post-quantum` **0.6.1** (MIT) for
in-browser ML-KEM and ML-DSA. The pin matters: **0.2.1 accepted all 2,595 malformed encapsulation
keys** in the C2SP/CCTV modulus set, because its polynomial coder took the compression path for
d=12 where FIPS 203 specifies plain ByteEncode12/ByteDecode12, making the section 7.2 modulus check
a no-op. 0.6.1 rejects all 2,595, and `npm run check:pqc` asserts it. Note the honest scope: noble
is self-audited and carries no side-channel protection, so this is conformance evidence, not
production assurance. Also `node-seal` 7.0.0, which bundles **Microsoft
SEAL 4.1.2**. Verified 2026-08-09: node-seal 7.0.0 (2025-11-08) is still the newest node-seal, so
this is not a stale pin here - but the gap to upstream SEAL is wider than it looks and is widening.
Microsoft SEAL has shipped **six releases since 4.1.2**, reaching **4.4.3 on 2026-08-04**. `CHANGES.md`
calls **4.4.0 a "Critical security update: all users should upgrade ... as soon as possible"**, which
hardened untrusted deserialization and fixed native memory-safety and concurrency defects across
ciphertext evaluation; 4.4.3 fixes null-output-pointer dereferences and C++ exceptions leaking across
the **C API** boundary (issues #753-#755). The C-API hedge applies to 4.4.3 - deserialization and
ciphertext evaluation are core C++, so 4.4.0's two most serious items **do** reach a WASM build. This
README said 4.4.0 was "described only as Major security update with no further detail" until Round 32;
that is the GitHub release body, not the changelog, and it under-stated a critical advisory. Full
analysis and what it means for this demo: [`docs/deliverable-verification.md`](docs/deliverable-verification.md)
and [`docs/threat-model.md`](docs/threat-model.md). **No CVE and no GHSA advisory
exists for SEAL**, so the version is recorded as a fact and no vulnerability is claimed
(SEAL -> WASM CKKS, MIT) +
`mpc-framework`/`emp-wasm-engine`/`summon-ts` (garbled-circuit 2PC, MIT) for the demo's real in-browser
FHE/MPC; `onnxruntime-web` (MIT) for the published industry models; `@huggingface/transformers` +
`@mlc-ai/web-llm` (Apache-2.0) for in-browser AI; `mcp` Python SDK (MIT) + the official
`@modelcontextprotocol/server-filesystem` (MIT) for the real MCP server (full deployment). Full table
+ dataset licenses in [`DATA_LICENSES.md`](DATA_LICENSES.md).

**Originality note:** the GPU MPC-FHE benchmark is our own integration over the *upstream*
projects (`src/qsh/benchmark/gpu_mpc_adapter.py` + `scripts/build_gpu_mpc.sh`). We do **not**
clone or vendor any mixed-authorship aggregate repo.

---

## Datasets

Bundled samples are real measurements where a licence permits it, synthetic where none does; the synthetic generators are an offline-CI fallback only
(logged as such, never a silent substitute):
- **Finance/AML** (demo sample): **PaySim simulator output** (Kaggle `ealaxi/paysim1`, CC-BY-SA-4.0) -> the 8 AML features. PaySim is a *simulator*: these are synthetic transactions from aggregated real mobile-money logs, not real transactions. The **training** corpus is IBM AMLworld HI-Small (CDLA-Sharing-1.0), 5,078,345 transactions at 1 laundering in 981. **AMLworld is synthetic too** - IBM generates it with a multi-agent virtual-world model and states "Everything is synthetic"; the NeurIPS 2023 D&B paper is titled *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. It is used because it is calibrated to real transaction statistics and its CDLA-Sharing terms permit publishing derived weights - **not** because it is real data. No commercial-OK corpus of real laundering transactions exists.
- **Medical**: MedMNIST v2 (CC-BY-4.0), Wisconsin breast-cancer (bundled demo sample; UCI, CC BY 4.0 - attribution required).
- **Genomics**: **real 1000 Genomes** phase-3 genotypes (Ensembl/IGSR - open *access*, but IGSR grants **no blanket licence**: "rights claimed on pieces of data vary") + **real PGS-Catalog** PGS000004 / PRS313 weights (**per-score licence**, EMBL-EBI terms by default). Neither is cleared for redistribution as commercial-OK; see [DATA_LICENSES.md](DATA_LICENSES.md).
- **Satellite**: **real NASA SMAP/MSL** spacecraft telemetry (Telemanom; Hundman et al. 2018; NASA JPL) - **no licence is granted by any provider** `[UNVERIFIED - reasoned inference]`. Telemanom's `LICENSE.txt` is a Caltech copyright assertion covering the *code*; the Kaggle mirror is labelled "Data files (c) Original Authors". 17 U.S.C. 105 removes US copyright only and is not a grant. Bundled demo sample only - the satellite **training** corpus is ESA-ADB (CC BY 3.0 IGO). See [DATA_LICENSES.md](DATA_LICENSES.md).
- **Telecom/defense**: literature-grounded synthetic QKD-attack telemetry (no commercial-OK real corpus of **QKD-layer** attacks exists - producing one needs a physical QKD testbed under attack).

Loaders + generators: `qsh.data.loaders` / `qsh.data.synthetic`.

---

## Benchmark - MPC-FHE vs QSMPC-FHE

On each use-case page the benchmark is **unified with the live simulator** (Round 7): a single **Run**
executes the E2E pipeline and the benchmark is **assembled from that run** - the MPC-FHE row carries the
run's real measured FHE-linear + MPC-non-linear latency, alongside the QSMPC-FHE comparison + quantum
advantage (the backend `QuantumComparisonStage` writes it into the ML-DSA-signed summary). The CLI below
runs the same comparison standalone.

`bash scripts/run_benchmark.sh linear` (or `python -m qsh.benchmark.runner --workload linear
--model-size medium --out results/`; artifacts -> `results/benchmark_*.{csv,json}`, also viewable
via the benchmark panel on each use-case simulator page).

### Faithful hybrid: linear -> FHE, non-linear -> MPC

The classical path is a **genuine 2026-style MPC-FHE hybrid**, not CKKS-only. **Linear layers**
(matmul / conv) run **under CKKS/FHE** (TenSEAL or OpenFHE on CPU; HEonGPU on the Blackwell GPU),
while **non-linear layers** (ReLU, argmax) run **under secret-sharing MPC** - real Shamir
secure-comparison via **MPyC** (`src/qsh/mpc/secret_inference.py`), with a CKKS <-> shares conversion at
each boundary. This mirrors the Gazelle -> Delphi -> BOLT -> Bumblebee -> **BLB** line and EncFormer, so
the benchmark reports the **FHE-linear vs MPC-non-linear latency split + MPC communication**, not a
single CKKS number. The **QSMPC-FHE** quantum analogue is **Clifford gates (linear, cheap) vs T-gates
(non-linear, expensive)** - but QOTP / Rate-1 QFHE evaluate *both* on the encrypted state (ct/pt approx.
1.3), so there is **no FHE <-> MPC boundary**; the secure-comparison role is played by the QSMS/QOT
quantum-MPC path.

### Model-size tiers + per-use-case architectures

The model is **per use case**, not a single generic shape (`qsh.benchmark.runner.usecase_arch_layers` /
`usecase_arch_specs`, mirrored client-side): web-grounded 2026 parameter budgets - **finance** = tabular
fraud MLP, **medical** = a **genuine small CNN on real MedMNIST images**, **genomics** = SNP/variant risk
MLP. There is **no workload selector** any more - see the removal note further down. It offered
`linear` against `cnn`, and once the CNN path began evaluating the published student's own weights
end to end under CKKS, `linear` was the only fabricated option left.

**Medical + `cnn` is a real encrypted convolution, not a cost proxy.** When the medical use case loads
MedMNIST (`config['dataset']='medmnist'` / the CNN workload), the pipeline federated-trains a tiny CNN
(one conv block 1 -> 4 ch 3x3, avg-pool, dense head at 14x14) and runs it under the hybrid as a **genuine FHE
convolution** - im2col -> CKKS `dot_plain` per output channel (`src/qsh/fhe/conv.py`), with ReLU/max-pool
under MPC (`secure_relu` / `secure_maxpool`). The convolution arithmetic happens on the **encrypted image**;
the FHE-linear conv cost genuinely dominates (~4.5 s for one 14x14 image on CPU CKKS, decision faithful to
plaintext). This is FHE-feasible by design - a MedMNIST-scale CNN, **not** ResNet/ViT (FHE cannot run SOTA
CNNs/LLMs under encryption - a fundamental crypto-cost limit, sold honestly as a PoC).

For the other industries the encrypted hybrid preserves the plaintext decision regardless of weights, so
the metric is FHE/MPC **cost + fidelity**, not task SOTA; lightweight === the trained linear model, so SHAP
stays exact. The benchmark shows the architecture label + parameter count.

There is **no lightweight / medium / large** selector any more either. It scaled invented
architectures - a "tabular fraud MLP" at 18/850/9,538 parameters against a real distilled student of
33 - so the sizes described models nothing had trained. What survives of the idea: **the measured
path runs end to end for real**
(CKKS-FHE linear + MPyC-MPC non-linear); **large is clearly labelled *projected*** - extrapolated from the
measured medium per-op cost (MAC / comparison counts) so the page stays responsive. Representative results on
the Blackwell box (TenSEAL CPU + MPyC, `linear` workload):

| Path | Backend | Size | Latency/inf | FHE-linear | MPC-non-linear | MPC comm | Acc | ct/pt | Security model |
|------|---------|------|-------------|-----------|----------------|----------|-----|-------|----------------|
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | lightweight | ~43 ms | ~39 ms | ~4 ms | ~0.5 KB | 1.0 | ~1483x | computational (RLWE) |
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | medium | ~243 ms | ~215 ms | ~28 ms | ~4.5 KB | 1.0 | ~1483x | computational (RLWE) |
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | large \* | ~1885 ms | ~1719 ms | ~165 ms | ~24.5 KB | 1.0 | - | computational (RLWE) |
| MPC-FHE | HEonGPU CKKS (GPU sm_120) + MPyC | medium | batched | GPU | CPU | - | - | - | computational (RLWE) |
| QSMPC-FHE | QOTP + QCNN (Clifford+T in the browser), no trained weights | - | ~4 ms | Clifford | T-gates x56 | n/a | n/a | **1.0 measured in QUBITS** (see note) | information-theoretic (QOTP) |

**Note on the `1.0` in the ct/pt column, because this README publishes two different numbers for
what looks like one quantity.** The row above says `1.0`; the WS4 table below says the measured
Rate-1 ratio is 1.2769 at 8 qubits, 1.0693 at 10 and 1.0173 at 12. Both are correct and they measure
different things. **`1.0` counts QUBITS**: under a QOTP the ciphertext is the same number of qubits as
the plaintext, which is the whole point of a rate-1 scheme and is exactly 1.0 by construction, key
excluded. **The 1.2769 family counts BYTES** of the serialised payload, which includes the roughly
1135-byte encapsulated key - fixed in size, so it dominates a small state and fades as the state
grows. Neither is the literature's 1.32x, which is a config constant feeding a static KPI and is not
a measurement of this code. See [`docs/architecture.md`](docs/architecture.md).

The **Security** column used to publish `68 / 100` and `96 / 100`. Those are the retired composite,
and that particular pair is the worst thing it produced: **16 of the 28 points separating them were
an artefact of which arguments two functions happened to supply** - one call passed no QBER while the
other passed a literal, and the same again for a leakage bound nothing measured. The column now
reports the security *model*, which is the part that was ever a property of the cryptography.

\* **projected** from measured medium per-op cost, not run end-to-end. GPU FHE wins on **batched**
workloads; for a tiny single-vector input the per-call kernel overhead can exceed CPU (raw encrypt
throughput was ~700 ops/s GPU vs ~34 CPU). The headline: the quantum **Rate-1** path is ~10^3x more
communication-compact than classical CKKS, with information-theoretic (vs computational) security -
and the hybrid now shows *where* the classical cost goes (FHE dominates the linear layers; MPC adds
the non-linear secure comparisons + a few KB of communication).

**Three enriched dimensions** (Round 3):
- **Security posture table** (replaced the 0-100 score in Round 22) - per-row facts, each either
  checkable against a published standard or produced by the run, and labelled which: NIST category
  and ciphertext/signature sizes from FIPS 203/204, QKD secure key bits, SKR, the QBER verdict taken
  from `secure_key_bits > 0` rather than a threshold, and the GLLP tagged fraction showing what the
  decoy states are buying. `qsh/benchmark/security.py` still computes the old composite for
  backwards compatibility but nothing renders it. Why it was retired:
  `references/security-scoring-fact-check.md`.
- **Quantum advantage (0-10)** - `qsh/benchmark/quantum_advantage.py`: weighted geometric mean of
  communication-compactness (approx. 10^3x), latency, and the security jump -> a tiered score
  (*dominant*). Recomputed 2026-08-09 from this table's own rows: **8.12 standalone**, uncapped. The
  finance E2E run reports **10/10**, and that is a CLAMP, not a maximum - the raw composite measured
  10.65, so the panel prints `raw 10.65 (clamped to 0-10)` beside the tier. A reader who sees 10 and
  concludes the run sits exactly at the ceiling is being misled by the clamp, which is why the raw
  value is printed. The earlier "9.2 on the finance E2E run" in this line was stale.
- **Per-use-case two-party simulator** - each industry has its own page (`/usecase/:id`) with a
  **live two-party (Alice <-> Bob) sequence diagram** (`web/src/components/TwoPartyDiagram.tsx`) whose
  steps light up as that use case's pipeline runs - **Start / Pause / Resume / Step / Stop / Restart
  drive it**. Parties are use-case-specific (Bank A <-> Bank B, Hospital <-> Aggregator, Cohort <-> Cohort,
  Carrier <-> Eve, Ground-station <-> LEO-satellite), grounded in `references/` (Zhang QOT npj 2026, Li&Deng
  QHE-FL, Sutradhar QSMS). Labels fill from live metrics (QBER / PSI / federated acc / QSDC secrecy);
  exports a high-DPI PNG + a multi-second GIF. A standalone per-party-timing micro-benchmark also
  exists at `qsh/benchmark/alice_bob_e2e.py` (`POST /api/alice-bob`).

The benchmark, security posture and quantum advantage are computed server-side **and** mirrored
client-side (`web/src/lib/clientCompute.ts`) so the public demo shows identical numbers with no backend.

---

## Deployment

Host-specific deployment - the port-collision workaround, the shared-proxy override and the
public-demo profile - is in [`docs/deployment.md`](docs/deployment.md). That file had **no inbound
link from any tracked file** until Round 43, and it is the only document carrying the workaround, so
a cloner hitting the collision had no way to find the answer that already existed.

A generated 14-section technical report is at
[`report/technical-report.pdf`](report/technical-report.pdf); build it with `make report`. Every
table in it is generated from the repository's own artefacts rather than transcribed.

- **Full deployment (federated learning + inference)** - Docker image build -> server startup:
  ```bash
  git clone <repo> qsh && cd qsh
  docker compose -f docker/docker-compose.yml build       # build the images (full AI tier + OpenFHE + Node/MCP)
  docker images | grep -E 'docker-(api|web)'           # the freshly-built images
  docker compose -f docker/docker-compose.yml up -d       # start: API (+AI tier) . SPA . Prometheus . Grafana
  docker compose -f docker/docker-compose.yml ps         # all services healthy
  # SPA http://localhost:8080  .  API http://localhost:8000/api/health  .  Grafana http://localhost:3000
  # run a federated use case end-to-end (writes results/):
  docker compose -f docker/docker-compose.yml exec api \
    python -m qsh.orchestration.pipeline --usecase medical_fl --out results/
  ```
  The full `Dockerfile.api` (Debian-trixie) **bakes in the AI tier** (CPU torch + PennyLane + MedMNIST),
  the **live OpenFHE** CKKS backend, and **Node/npx** for the real MCP server, so a federated `medical_fl` run +
  the MCP server work out of the box. Fits a **6-core / 12 GB / 100 GB** box (CPU; GPU passthrough via host
  `nvidia-container-toolkit` + the `deploy.resources` GPU block is optional and faster). Ports are env-overridable
  (`WEB_PORT` / `API_PORT` / `GRAFANA_PORT`).
- **Client-side build (implementation capability)**: the SPA also builds **fully client-side** - every
  computation runs in the browser (real CKKS FHE + an emp-wasm garbled-circuit 2PC, ONNX inference, BB84 /
  PSI / SHAP / QFL), with **zero `/api` compute**. In this mode the system needs **neither the backend
  FHE/MPC compute path nor the frontend FHE/MPC backend-selectors** - the client owns both - so it requires
  no GPU and no server-side ML. The same pages and buttons work in both modes.
- **Monitoring**: see [`monitoring/README.md`](monitoring/README.md) (Prometheus + Grafana for the local
  full deployment; Zabbix dropped as redundant).

---

## WS3 + WS4 challenges addressed

The four workstreams target specific open problems in quantum-safe + private AI; here is how this
implementation maps to each, and where gaps remain.

**WS3 - QSMPCxQKD** (cross-org AI restricted by GDPR/HIPAA; classical SMPC is complex, HNDL-vulnerable, sometimes assumes a TTP; Agentic-AI/MCP lacks quantum-safe + private agent comms):

| Sub-problem | How QSH addresses it | Where |
|---|---|---|
| Raw data must not be shared | Federated learning + secure aggregation; PSI (blacklists never exchanged); FHE private inference | `ai/federated.py`, `qsmpc/qot.py`, `fhe/` |
| Classical SMPC protocol complexity | QSMS = Shamir + d-level QFT (no Beaver triples / garbled circuits) | `qsmpc/secure_summation.py` |
| HNDL-vulnerable HE / key exchange | PQC-first hybrid **ML-KEM + X25519** (+ optional QKD key material) | `pqc/hybrid_kem.py`, `pqc/transport.py` |
| TTP assumption | TTP-free: QSMS (entangled state, no dealer), keyed-hash PSI over the quantum-safe channel, pairwise-mask secure aggregation | `qsmpc/`, `ai/federated.py` |
| **Agentic-AI / MCP quantum-safe + private agent comms** | A **real MCP server** (SDK >=2.0 `MCPServer`, formerly FastMCP) (5 per-industry tools, full deployment) bound to a quantum-safe channel: two agents establish a PQC(+QKD) session, the caller sends an **ML-DSA-signed, AES-GCM-sealed** `tools/call`, the callee **dispatches the real tool** and returns the **signed result** (authenticated, confidential, HNDL-resistant, no TTP); also interops with the official filesystem MCP server | `pqc/mcp_bridge.py`, `mcp/server.py`, `/api/mcp/*` |
| **Explainability / audit trail (XAI)** | Exact linear-SHAP feature attributions for each run's classification, **ML-DSA-signed** -> tamper-evident, regulator-facing explanation (EU AI Act Art. 13 - applicable 2 Dec 2027 for Annex III after the AI Omnibus deferral, not Aug 2026; GDPR Art. 22; SR 11-7); surfaced on all five use-case pages. On telecom and satellite, whose flows have no `private_inference` stage, the attribution is of the model that produced the reported class; on the other three it explains the bundled model beside the encrypted one, and the panel says which case applies | `ai/explainability.py`, `ExplainabilityStage` |

**WS4 - QHExAI** (FHE 10^3-10^6x overhead; bootstrapping bottleneck; no quantum encoding; plus unsolved (a)-(f)):

| Sub-problem | How QSH addresses it | Where |
|---|---|---|
| (1)/(e) huge compute + ciphertext/plaintext overhead | Benchmark *measures* ~1483x CKKS. For Rate-1 QFHE the ct/pt ratio is **measured and size-dependent** - 1.2769 at 8 qubits, 1.0693 at 10, 1.0173 at 12 - because the ~1135-byte encapsulated key is fixed while the state grows. The **1.32x** figure quoted elsewhere is the *literature* value in `config/defaults.py` feeding a static Overview KPI; it is **not** a measurement of this code, and an earlier revision of this row said the benchmark measured it. See `docs/architecture.md` | `benchmark/`, `qhe/rate1_qfhe.py` |
| (2) bootstrapping bottleneck | **OpenFHE CKKS `EvalBootstrap`** (CPU) + **HEonGPU collective bootstrapping** (GPU) refresh ciphertext levels for deeper circuits - both commercial-OK | `fhe/backends.py` (`OpenFHEBackend(use_bootstrap=True)`) |
| (3)/(a) quantum encoding / practical QHE | QOTP + a variational QCNN + Rate-1 QFHE at small scale. **Two different circuits, and only one is Clifford+T:** the BROWSER circuit (`web/src/lib/quantumSim.ts`) is built from H, CNOT and T and counts 56 real T gates, while the PYTHON module contains **no T gate at all** - it is `AngleEmbedding` plus CNOT/RY/RZ, and its `t_count()` is a labelled Solovay-Kitaev estimate of what a fault-tolerant deployment would cost, not a count of gates present. Earlier revisions called the Python module Clifford+T. | `qhe/qotp.py`, `qhe/qcnn.py`, `web/src/lib/quantumSim.ts` |
| (b) QFL gradient leakage (arXiv:2508.15998) | Pairwise-mask secure aggregation + QOTP-delegated learning | `ai/federated.py` |
| (c) circuit privacy vs a malicious server | **MET for Clifford circuits, and this row previously under-claimed it.** A random Clifford word over {H, S, CNOT} is evaluated on the QOTP-encrypted state; the client tracks the Pauli frame through the same word and recovers the plaintext result at fidelity 1.0, verified to depth 30 across seeds. The server sees the maximally-mixed state throughout, so both the input and the computation's operand are hidden from it. A working fixed-circuit version of this had been in the tree, tested, and cited in no document; the honest gap was that one circuit is not arbitrary circuits, which needed the S frame rule (`H` and `CNOT` alone reach only a subgroup) and a CNOT applier not hardcoded to wires 0 and 1. **Does NOT extend to T:** `T X T-dagger` is not Pauli, so the frame does not propagate, which is why the QCNN's non-linear layers are costed as magic-state T-gates instead. | `qhe/qcnn.py` (`qotp_clifford_random_circuit_demo`), `qhe/qotp.py` (`update_key_h/s/cnot`) |
| (f) hiding the server's circuit from a malicious client | **NOT met, and in tension with the mechanism rather than merely unimplemented.** The client can only decrypt by applying the circuit's action to its own Pauli frame, so under plain QOTP it must know the circuit - "the client updates its frame" and "the client does not learn the circuit" are close to mutually exclusive. Earning this needs a construction where that does not hold, which is what Ortega and Martin-Delgado's scheme addresses and this repository does not implement. Split from (c) so the one that is met is not held hostage to the one that is not. Cf. QOTPH (arXiv:2604.19256), which builds universal QHE from QOTP over Clifford+T with Pauli key updates and reports results on real quantum processors. | not implemented |
| (d) efficiency on AI/HPC infra | OpenFHE (CPU) + HEonGPU (GPU) + CROSS (reference), HEonGPU shown on the Blackwell GPU | `fhe/`, `benchmark/` |

**Residual gaps, each with the condition that would close it.** This was a four-item list with no
conditions and no dates until Round 33, which meant nothing on it could ever become due. Two of the four
turned out to be downstream of decisions already recorded in
[`docs/deliverable-verification.md`](docs/deliverable-verification.md), so they are one item each rather
than independent gaps:

| residual | condition that closes it |
|---|---|
| **PQ-OPRF PSI + identifiable abort** | Name the browser-loadable library that supplies each, at the point where it is claimed. The non-linear 2PC layer is **already malicious-secure** - `emp-wasm` is authenticated garbling (`emp-ag2pc` / `emp-agmpc`, Wang-Ranellucci-Katz CCS 2017) - so what is missing is these two primitives specifically: a post-quantum OPRF for the PSI, which is a semi-honest keyed hash today, and **identifiable** abort, which authenticated garbling does not provide. |
| **Hardware QKD field test** | A boundary of the deliverable, not a gap in it: no QKD device exists here and there is no browser API for one. If a characterised link ever exists, the first thing published is the **measured QBER against `qkd/decoy_reference.py`'s prediction for the same link budget** - and that is also the round in which the 2026 finite-key proofs (arXiv:2606.29943, arXiv:2608.09674) replace Ma et al. (2005), because it is the round their device inputs exist. |
| **Larger fault-tolerant QHE** | `t_count()` is a labelled Solovay-Kitaev **estimate** of a fault-tolerant deployment's cost. Replace it with an executed count when a backend exposing logical, error-corrected qubits is addressable from PennyLane. |
| **Production bootstrapping throughput** | Downstream of the OpenFHE `EvalBootstrapSetup` SIGSEGV already recorded as a permanent workaround on this host distribution. Publish the figure when `make test-all` completes without the segfault, or when the HEonGPU collective-bootstrap path is benchmarked end to end. |

(A real quantum-safe MCP server + filesystem-MCP interop now ships in the full deployment.)
See [`docs/pqc-roadmap.md`](docs/pqc-roadmap.md).

---

## Concerns & recommendations

- **The orchestration now runs the published models; before Round 21 it never had.** The pipeline
  truncated its features to 8 and called `inferLogits`, which builds a `[1,8]` tensor. No published
  model takes `[1,8]` - they take `[1,32]`, `[1,1,28,28]`, `[1,250,6]`, `[1,19]`, `[1,7]` - so the
  ONNX tier failed on every use case, every run silently dropped to the bundled 8-feature linear
  model, and the badge above it read MEASURED. The Models panel forty lines below advertised a
  ResNet-18 the page never ran. `runPublishedModel()` is now shared by the panel and the pipeline,
  and a **Published model** selector picks which one on the three use cases that publish several.
  Measured in-browser: `finance_aml.gbm` 1.4 ms `1x32`, `medical_fl.resnet18` 4.7 ms `1x1x28x28`,
  `telecom_ids.qml` 5.9 ms `1x7`, `satellite_comms.telemanom` 9.6 ms `1x250x6`.

  It is a **second** inference, not a replacement. The published models are trained on different
  corpora from the pipeline's own features - BreastMNIST images against Wisconsin tabular rows - so
  feeding one to the other would be the category error this project refuses elsewhere. Both run,
  both are measured, and each says which corpus it is on. A published model that exists and fails to
  run is reported as a failure with no substitute figure.

- **A retrain had silently unpublished two models.** `record_metrics` replaced the whole entry under
  a docstring that said "Merge", and `hf_revision` lives in that entry, so retraining finance and
  OPS-SAT to capture their `StandardScaler` dropped both pins. Nothing failed - the damage was three
  layers away in a browser, where `browserModels()` skips any spec without a revision, and the
  finance page had gone from one verifiable model to none while still printing that model's metrics.
  Republished; a retrain now demotes the sha to `hf_revision_invalidated_by_retrain` and the UI says
  "retrained since publication" rather than the identical-looking "never pinned".

- **A software WebGPU adapter is slower than no adapter.** Same model, same row, two browsers:
  SwiftShader gave `onnx-webgpu` at 12842 ms, no adapter gave `onnx-wasm` at 42 ms. `requestAdapter()`
  succeeds on a software rasteriser, so "is WebGPU available" answered yes and the code took the
  build that was 300x slower than the one it had just declined. Now declined by name.

- **Three of the four CKKS students can be verified in a browser.** They shipped no held-out sample,
  so selecting one reported DID NOT RUN. Samples ship model-ready with `preprocessing: []`, because
  the browser's `applyPreprocessing` is per-scalar and cannot apply a 32-length mean/scale vector or
  an area resize. Reproduced before publishing: medical accuracy 0.769231, OPS-SAT rare-class F1
  0.904348, finance rare-class F1 0.376130 on its shipped rows. `satellite_comms.student` still has
  none, and the blocker is its METRIC, not a missing scaler: it is scored on event-wise F0.5, which
  is defined over contiguous runs of a timeline and cannot be computed from independent rows.

- **The browser scores each student on the metric it was published with.** Three of the four record
  rare-class F1, and reporting accuracy for them would have confirmed a number nobody claimed - the
  finance student is about 99 percent accurate on a corpus that is 0.1 percent positive. Its F1 is
  also 6.1x higher on the shipped rows than on the full corpus, because the sample is stratified to
  contain positives; what survives the change of prevalence is the ratio to the all-positive
  baseline, 9.6x here against about 8.7x on the corpus, and the page says so beside the figure.

- **Two new trained models, and one of them is a negative result.** `medical_fl.resnet50` exists to
  test a published claim rather than to win one: MedMNIST v2 reports ResNet-50 at AUC 0.857 against
  ResNet-18's 0.901 on this 780-image corpus, so the deeper network is expected to lose. It does,
  and by more than the published gap - 0.7988 against our ResNet-18's 0.9147, both trained from
  random initialisation. A third consequence of the same over-parameterisation appears in the
  export: its int8 graph agrees with its own fp32 on
  0.9865 of the confident set against a 0.99 gate, so the int8 artefact was built, measured and
  **withheld** rather than shipped with the gate relaxed. It replaces a DenseNet-121 card that named
  a dataset with no loader and linked a Hugging Face repo that did not exist.
- **`satellite_comms.opssat` is the first corpus here whose DATA can be republished.** OPS-SAT-AD is
  CC BY 4.0 and 507 KB, so the whole published test partition ships in the model's own HF repo and
  the browser reproduces the exact figure the dataset's 30 published baselines are quoted against
  (AUC PR 0.9681 measured, against FCNN 0.979, XGBOD 0.975, RF+ICCS 0.963). It is a **different
  task** from the Telemanom forecaster on the same page - segment classification, not forecasting -
  and neither number is the other's baseline.
- **The encrypted path evaluates the published medical student itself.** All 3,226 parameters of it,
  fetched from its own pinned Hugging Face revision and run end to end under CKKS: convolution,
  average pool and both dense layers, with im2col patches packed into a single ciphertext. The run
  reports **152 counted rotations** - the quantity the 2026 literature identifies as the cost driver
  - rather than a latency multiplier standing in for it, and states whether the published weights
  drove the computation or only their shape did. If the fetch fails, no weights are invented.
  The **Workload selector is gone**: it offered Linear against CNN, and once CNN ran the real
  student, Linear was the only fabricated option left - an 8-feature dot product with a hand-written
  weight pattern matching no trained model. For finance, satellite and OPS-SAT the students genuinely
  ARE linear, which is why the selector only ever appeared on the medical page.
- **Every model in the Models panel carries a Run button, or a specific reason it has none.** Six can
  be verified in place; the rest are in three genuinely different states that are never collapsed
  into one message - never trained (the finance GNN, the telecom Transformer), deliberately
  unpublished on licence grounds (the PRS), or published without a held-out sample.
- **`finance_aml.gnn` is a decision, not a backlog item.** The AMLworld authors report that XGBoost
  with the Graph Feature Preprocessor outperforms their GNNs, so the model already shipped is the
  stronger one. More decisively, arXiv:2411.02926v2 ran this exact comparison under FHE and the
  **GNN pipeline never compiled** - it failed on ONNX operator support in Concrete ML - while the
  XGBoost pipeline completed at 88,000-165,000x overhead. Training it would mean building the weaker
  model to run a path the literature recorded as not working **in 2024**, and which DESIGN (arXiv:2507.05649) and TGHE (arXiv:2606.26664) have since done - so it is a dated result rather than a standing impossibility.

- **The model that runs under encryption is now published too.** Each use case has a teacher and a
  distilled student, and the student is what CKKS actually evaluates: 33 parameters for finance
  against a 141,154-parameter teacher, 3,226 against 11,171,266 for medical, 49 against 80,486 for satellite.
  Until Round 19 only the teachers were published, so the encrypted tier - the part the project is
  built to demonstrate - was the one part a visitor could not verify. All ten repos are public and
  pinned, and the evaluation sample for each lives in that model's own repo at the same revision as
  its weights.

  *Both corrected figures were Round-19-era and outlived what they described.* The finance teacher
  read **138,312** until Round 40 - the count from before the eleven per-account causal aggregates
  were added, against `models/metrics.json`'s `teacher_params: 141154` - and the sentence said
  **seven** repos through the Round 30 re-upload that took the registry to ten. The other three
  figures in the same sentence still match `metrics.json` exactly, which is why neither was noticed:
  a paragraph that is three-quarters right reads as right. `report/generated/macros.tex` derives the
  repo count (`\npublished`) from the registry, so this sentence is the second copy - and the second
  copy is the one that rots.

- **Retracted: the satellite forecaster is not worse than persistence.** Round 18 reported it as
  roughly 19-22x worse on next-step MAE. That was our measurement error: the probe resampled the
  telemetry at 60s while the published model is trained at 300s, and feeding a 300s model 60s data
  is the whole of the explanation. Re-measured at the trained cadence the model is about **2.9x
  better** than persistence - MAE 0.0314 against 0.0897 on the evaluation split, and 0.40x / 0.35x
  / 0.38x on train / test / whole series. The evaluation-sample generator hardcoded the same 60s;
  it now reads the cadence from the model record, because preprocessing has to come from the
  artefact rather than a constant in a second file.
- **CLOSED, and the explanation is the denominator.** This was carried as "the event-wise headline
  does not reproduce": retraining at the same configuration gave event-wise F0.5 of 0.4545 where the
  recorded figure was 0.7143. `references/satellite-event-count-investigation.md` closes it - the
  test split contains **3 scored events**, and the teacher flags 2 points out of 441,600, so the
  score is 1 hit against 2 flags. One extra false alarm walks the metric down a fixed ladder:
  0.714 / 0.455 / 0.333 / 0.263 / 0.217 / 0.185 for 0 to 5 false-alarm runs. It reproduces exactly as
  well as a 3-event denominator permits, which is not very well, but it is not mysterious.
  Point-adjusted F1 reproduces (0.9937 against 0.9945) for the same reason it is inflated: 441,600
  points rather than 3 events. Still read the event-wise number as one draw from a wide distribution -
  the general instability of overlap-based metrics is real (arXiv:2607.11969, 2026-07) - but the
  operative fact here is local and arithmetic.

- **The browser model tier now works, after a 401 and two wrong diagnoses.** Recorded in full
  because two of the three were mine, and because the shape of the mistake is instructive.

  Until Round 18 every published model returned **401**: `publish_hf.py` called
  `create_repo(private=False, exist_ok=True)`, which does not change the visibility of a repo that
  already exists, so `--public` uploaded files, reported success, and left them private.

  With that fixed the models downloaded and ONNX session creation hung. I blamed the graphs -
  `ai.onnx.ml` tree ensembles, then LSTMs, then `ConvInteger` quantisation - and built a per-artefact
  gate on that theory, revising it twice. Then I blamed the WebNN provider. Both were wrong, and both
  were written up as settled.

  The cause was a single import. `import * as ort from "onnxruntime-web"` resolves to the **JSEP**
  bundle - the WebGPU-enabled build - and on a browser with no WebGPU adapter its
  `InferenceSession.create` never returns, for any graph. The identical models create a session in
  **0.01-0.23 s** against the plain wasm build, which is what Node used and why Node looked healthy
  throughout. `loadOrt()` now imports `onnxruntime-web/webgpu` only when a real adapter exists and
  `onnxruntime-web/wasm` otherwise: the probe decides which **build** to load, not merely which
  provider to request.

  There was a grain of truth in the retracted diagnosis, found by re-testing with WebGPU actually
  enabled: the **webgpu build does not register the `ai.onnx.ml` operators**, so a
  `TreeEnsembleClassifier` fails there - cleanly, with `No Op registered` - while running fine on
  wasm. `ai.onnx.ml` really was unavailable, in one build, which happened to be the one being loaded
  unconditionally. A model in that domain now takes wasm even where an adapter exists.

  Verified on the live deploy, all three pages, in a Chrome launched with
  `--enable-unsafe-webgpu --enable-features=Vulkan,WebMachineLearningNeuralNetwork` (on Linux these
  cannot be set permanently through `chrome://flags`; the browser must be launched with them):

  | page | in-browser | expected | sample | provider |
  |---|---|---|---|---|
  | medical | **85.9%** | 85.9% | the entire official BreastMNIST test split, 156 rows | `onnx-webgpu` |
  | finance | **99.3%** | 99.3% | 8192 AMLworld rows at near-real 1-in-981 prevalence | `onnx-wasm` |
  | satellite | **MAE 0.0741** | 0.0741 | 128 ESA-ADB windows at the trained 300s cadence | `onnx-webgpu` |

  WebNN is probed by **building and executing a 2x2 matmul** - three cheaper checks all lied - and
  then deliberately left unused: with the flag the API works, but ORT's WebNN provider still cannot
  complete a session for the medical ResNet (162.7 s and a timeout, against 5.1 s on wasm).

  Two lessons, both ones this project already had: a capability check that tests for an API surface
  rather than a working capability is the same defect class as a label naming a protocol the code
  does not run - and when three different models fail three different ways, suspect the thing they
  share.

- **`EvalBootstrap` is described elsewhere in this README as the answer to the bootstrapping
  bottleneck, and on the documented box it is not runnable.** The OpenFHE bootstrap test **SIGSEGVs**
  inside `EvalBootstrapSetup` on AlmaLinux 9.7 - the import guard in `fhe/backends.py` catches an
  `ImportError`, and a segfault is not one. `make test` deselects it with `-m "not slow"`;
  `make test-all` still dies. Round 18 fixed the Makefile and this paragraph kept telling readers
  `make test` runs a bare `pytest` for three more rounds, which is the doc-versus-code divergence
  this list exists to catch - **and the sentence carried `(174 collected, 170 passed, 4 deselected)`
  until Round 41, which is the exact figure the install section 551 lines above names as the stale
  one.** Round 40 deleted one copy of that number and wrote a sentence explaining why, and left the
  other. Two copies, one deleted, is how this file has drifted before; no count is quoted here now.
- **The retired security score had two paths, and they disagreed for a long time.** The classical row
  was scored without QKD key material while the quantum row assumed 256 bits, and separately the
  classical row passed no QBER while the quantum row passed a literal `0.0128` and a literal leakage
  bound. **Sixteen of the 28 points in the headline 68-vs-96 gap were that asymmetry, not
  cryptography.** Round 22 closed both halves in both languages and then retired the composite
  entirely rather than keep rendering it. The lesson generalises past this metric: when two rows are
  compared, verify they were scored under ONE transport configuration before believing the gap.
- **ct/pt is not a constant.** The "~1483x" figure is `ciphertext_bytes / (n_in * 8)` and moves with the
  feature count - measured ~1392x at 30 features and ~5222x at 8. Read it as a property of a
  configuration, not of CKKS.
- **The encrypted medical CNN benchmark and the medical AUC use different datasets.** The AUC 0.9147 is
  BreastMNIST; the encrypted-convolution timing runs PneumoniaMNIST (`benchmark.medical_cnn.medmnist_name`).
  Both are MedMNIST v2, but they are not the same images.
- **`results/` retention.** It had none for three rounds and grew 953 -> 1133 -> 1421 files;
  `docker/docker-compose.yml` bind-mounts it, so a long-lived deployment grew without bound.
  `orchestration.keep_runs` (default 200, 0 disables) now prunes on persist, grouped by run stem
  so a run is kept or removed whole. Measured after the change: 600 files, 200 runs.
- **Recommendation - generate the benchmark table instead of maintaining it by hand.** Six of the ten
  numbers in the benchmark section had drifted from what the documented command produces. `tests/` already
  asserts demo samples against the registry; the same pattern over the model and benchmark tables would
  have caught the AUROC-vs-C-index, 2022-vs-2023 and "every published QML-IDS number" errors before a
  reader did.

- **Simulation vs reality** - QKD distance/QBER gains, photonic inference, decoy-state, full
  qOLE, and the **free-space satellite QKD + QSDC** path are **simulated** (analytic channel +
  information-structure models), not hardware-validated. QHE evaluation is exact for Clifford
  circuits; non-Clifford layers are costed as T-gates, not run fault-tolerantly.
- **NSA stance** - NSA does **not** endorse QKD in any form for national security systems. Its CNSA 2.0 /
  Quantum Computing FAQ answers *"Can I use a QKD system to protect my NSS from a quantum computer?"* with
  **"No"**, and says it *"does not support the usage of QKD or QC"* for NSS. Presenting QKD as a hybrid
  add-on is **this project's** design choice, taken from **BSI TR-02102-1 v2026-01 section 2.2**, not from NSA.
- **GPU/vendor & licensing** - Blackwell+CUDA13 narrows the quantum-GPU stack to CUDA-Q/cuQuantum.
  The FHE stack is deliberately **commercial-OK** (TenSEAL Apache-2.0, OpenFHE BSD-2, HEonGPU Apache-2.0);
  **DESILO was removed** because its license is non-commercial. Multiple swappable backends avoid lock-in.
- **Scope discipline** - QHE kept to 6-8 qubit QCNN for reproducibility. The demo's **QSMPC-FHE (quantum)
  comparison row is a genuine in-browser state-vector simulation** (8-qubit, QOTP + Clifford+T QCNN): its
  latency is **measured** and its T-gate count is **counted from the circuit** - a *simulation*, not quantum
  hardware (no QPU); under QOTP the ciphertext is the same #qubits as the plaintext (ct/pt = 1). The classical
  **MPC-FHE row is real measured crypto** (node-seal CKKS + emp-wasm 2PC) - so the demo shows measured, not
  estimated, numbers.
- **In-browser AI (demo)** - the per-use-case Agentic-AI panel runs **small** models client-side (Transformers.js
  on onnxruntime-web; WebLLM on WebGPU). The ONNX path uses **the WebGPU build only where a real `requestAdapter()` returns a HARDWARE
  adapter**, and the WASM build otherwise. A returned adapter is no longer sufficient: a software
  rasteriser (SwiftShader, lavapipe, llvmpipe, WARP) is declined by name, because it measured about
  300x slower than the wasm build it displaces - ResNet-50 at 12842 ms against 42 ms on the same
  model and row. Every shader dispatch is emulated on the CPU while wasm runs SIMD kernels natively,
  so declining it picks the faster of two CPU paths. The probe chooses the
  *build* (`onnxruntime-web/webgpu` vs `onnxruntime-web/wasm`), not merely the execution provider,
  because the JSEP bundle never completes session creation without an adapter - that, and not
  `ai.onnx.ml` or WebNN, was the cause of the Round 18 browser-tier hang. WebNN is probed by
  building and executing a 2x2 matmul, and then deliberately left unused: the API does work behind
  the flag, but ORT's WebNN provider cannot complete a session for the medical ResNet. A graph
  using `ai.onnx.ml` takes the WASM build whatever the adapter says, because ORT's CI builds the
  WebGPU wasm with `--disable_ml_ops`, so those kernels are compiled out rather than merely
  unpreferred. Because the choice comes from a real `requestAdapter()` call, the reported
  backend is never "webgpu" on a machine with no usable adapter. **It does not mean every node ran on
  the GPU**, and the sentence before this one is why: ORT assigns nodes to execution providers in
  preference order and falls back per node, so a graph can run partly on WebGPU and partly on the
  WASM floor while the badge shows a single state. Which nodes fell back is **not measured here** -
  closing that needs `startProfiling()`/`endProfiling()` or ORT's "kernel not found in registries"
  log, and the badge's two states cannot express a partial answer. Model weights stream from the HF/MLC **CDN** on first run
  (browser-cached after) - zero VPS bandwidth/compute. WebLLM needs WebGPU; without it the page falls
  back to Transformers.js (WASM) and the quantum-safe MCP envelope still runs. The in-browser models
  illustrate the agentic-AI + MCP flow; they are not the production inference path.
- **MCP** - the **full deployment ships a real MCP server**: an **`MCPServer`** server (MCP Python SDK >=2.0, MIT; the class was named `FastMCP` before v2.0.0 and `server.py` resolves either)
  exposing the 5 per-industry tools, bound to the quantum-safe transport so a `tools/call` runs end-to-end
  under PQC+QKD+ML-DSA+AEAD and returns the **signed real result** (`src/qsh/mcp/server.py`,
  `pqc/mcp_bridge.secure_agent_call`, `/api/mcp/{tools,call}`); it also **interops with the official
  `@modelcontextprotocol/server-filesystem`** under the same quantum-safe seal (`mcp/fs_interop.py`,
  `/api/mcp/filesystem`). The public demo keeps the zero-server `clientMcp.ts` mirror (**zero per-user VPS
  load**). In the WebUI the in-browser agent is **integrated into each use-case page** (its own per-industry
  MCP tool, fed by the live run) - the standalone Agentic-AI page was **removed**, since the report frames MCP
  as a per-use-case cross-cutting layer, not a separate feature.
- **Deploying behind an existing reverse proxy.** `docker/docker-compose.demo.yml` publishes host 80
  and 443, which is correct when the demo is the only service on the box. When something else
  already owns those ports - as on the public demo VPS, where a shared Caddy routes the hostname to
  this container by name - add the committed override:

  ```
  docker compose -p qsh-demo -f docker/docker-compose.demo.yml \
      -f docker/docker-compose.demo.behind-proxy.yml up -d --build
  ```

  It drops the host port publishing and joins the proxy's network (`PROXY_NETWORK`, default
  `pqcqkd_mgmt-net`), declared `external` so a missing network fails loudly instead of silently
  creating an empty one. Without it the bring-up loses the port race and leaves a container that
  `docker ps` reports as **Up** while it sits on **no network at all** and every request 502s. That
  happened on 2026-08-05 and took the public demo down for several minutes; the working topology
  existed only as a remembered command line, which is why it is a file now.

- **Demo vs full** - the public demo is **client-only** (zero `/api` compute) and ships **no backend
  image at all**: `docker/docker-compose.demo.yml` has no `api` service, and `/api/*` and `/metrics`
  return 404. (Round 17 removed the slim image this line used to credit - it omits FHE by design, so
  every server-routed run failed on `tenseal`, while the page still said no computation was sent.) Its FHE/MPC is **genuinely real and measured in the browser**: **node-seal CKKS**
  (a real homomorphic linear layer, ~1e-9 error) + a real **emp-wasm garbled-circuit 2PC** for the
  non-linear argmax, both parties loopback in one tab (zero VPS). Federated learning is offered as an
  in-browser **Quantum Federated Learning** toggle (a real client-side FedAvg of a quantum-classical
  model); the full deployment runs the **PennyLane** QFL over this repo's own FedAvg + secure aggregation, plus server-side FHE inference (incl.
  the genuine medical CNN under real encrypted convolution - in-browser, node-seal CKKS evaluates
  the published student END TO END: convolution, average pool and both dense layers, 152 counted
  rotations. Nothing is approximated. This line said "only the CNN's conv cost is approximated
  client-side" until Round 39, contradicting the passage above that already described the
  end-to-end path).
- **Per-industry models - three tiers, and the gap between them is the measurement** - the
  **crypto/protocol layer is real across all five industries** (PSI/QOT, QSMS secure-sum, BB84/QSDC,
  MPyC MPC, CKKS FHE, federated secure-aggregation). The AI side is organised in tiers because
  homomorphic evaluation is cost-bounded **at the operating point this demo targets** - a single
  device, in a browser tab, running CKKS through `node-seal` (SEAL compiled to WASM, single-threaded).
  Park et al. ([arXiv:2601.18511v2](https://arxiv.org/abs/2601.18511v2)) run Llama-3-8B end to end
  under CKKS, 128 encrypted tokens in 20 s - on **eight** NVIDIA RTX PRO 6000 GPUs, which is ten orders
  of magnitude away from a tab. So the constraint is real and it is about the *device*, not about the
  field: a demo cannot honestly claim both a state-of-the-art model *and* encrypted inference of that
  model **here**. This sentence read "and will stay that way ... (IACR ePrint 2026/935)" until Round 32,
  which was a prediction rather than a measurement and cited the systematisation instead of the paper:

  | tier | what it is | where it runs |
  |---|---|---|
  | **T1 crypto** | the FHE/MPC-feasible student | real CKKS + MPyC server-side; node-seal CKKS + emp-wasm 2PC in-browser |
  | **T2 industry** | the real literature model, trained here, published to the Hub | browser (**int8 ONNX where a quantised graph exists**; the two tree-ensemble models serve **fp32**, because their parameters live in node attributes and dynamic quantisation has nothing to quantise) and server (fp32 ONNX) |
  | **T3 large** | models too big to encrypt at all | WebGPU in-browser, or a hosted endpoint |

  The headline number is **`crypto_fidelity`**: how often the encrypted student reproduces the
  plaintext teacher's decision on held-out data, and what that costs in the industry's own metric.
  It is measured per run rather than asserted, and it is the direct answer to "what does the privacy
  actually cost". Everything below is what this repository measured, not what the papers report -
  published baselines are shown beside each figure, including where we fall short:

  | vertical | T2 model | data | measured | published baseline |
  |---|---|---|---|---|
  | **finance** | GFP + XGBoost (graph-feature preprocessing over the transaction multigraph) | IBM **AMLworld HI-Small**, 5,078,345 transactions, 1 laundering in 981 (CDLA-Sharing-1.0) | minority-class F1 **0.1519**, AUPRC **0.1233** vs a 0.0015 prevalence baseline | 0.632 (NeurIPS 2023 D&B) - see the gap note below |
  | **medical** | **ResNet-18** trained from random initialisation (no ImageNet weights, so no ImageNet terms attach) | MedMNIST v2 **BreastMNIST** (CC BY 4.0 **asserted by MedMNIST**; the upstream BUSI grant is **not verifiable** - see [DATA_LICENSES.md](DATA_LICENSES.md)) | AUC **0.9147** | 0.901 (Yang et al., Scientific Data 10:41, 2023) |
  | **genomics** | published **PRS** effect-weight vector - not a neural network | PGS Catalog **PGS000004 (PRS313)** over **2,504 real 1000 Genomes samples** | ancestry portability, below | AUROC 0.633 [0.624, 0.641] - *external citation, not measurable here: a reference panel carries no phenotype. Fitted WITH covariates a dot product does not have, and 8% of variants were missing in that cohort, so it is a citation and not a target* |
  | **satellite** | **Telemanom** 2xLSTM(80), seq 250, non-parametric dynamic threshold + pruning | **ESA-ADB Mission1** lightweight channels 41-46, multi-year real telemetry (CC-BY-3.0-IGO) | event-wise F0.5 **0.4545** over just **3 events** | 0.968 (ESA-ADB Supp. Table 9, anomalies only) |
  | **telecom** | RF baseline + quantum-kernel SVC + variational QNN | literature-grounded synthetic QKD-attack telemetry (7 attack types + normal) | macro F1 - **RF 0.987**, quantum-kernel SVC **0.832**, variational QNN **0.467** | no trustworthy single number exists - see below |

  Three of those deserve their honest caveat stated in the README rather than buried:

  - **finance falls well short of 0.632, and the reason recorded here for several rounds was wrong.**
    The note used to give two causes: the Proprietary `snapml` licence on IBM's Graph Feature
    Preprocessor, and 10-hop injected CYCLE patterns against 4-hop reachability here. The second does
    not survive arithmetic - the GFP paper (arXiv:2402.08593) decomposes its own contribution and caps
    multi-hop cycle and scatter-gather features at about **4% relative**, while fan-in/fan-out, which
    this project already reimplements, carries **more than 30%**. Four percent cannot explain a
    multiple-fold gap. The real cause is architectural rather than licence-related, and the mechanism
    is derived once in `references/finance-gap-investigation.md` rather than restated here. It is
    implementable without `snapml`; it is simply a larger piece of work. The
    licence exclusion is real but was never the mechanism. Investigating this also found and fixed a
    genuine defect - per-account aggregates spanned the whole training window, so a training row's
    feature included its own future while test rows got a stale table. Making them causal
    (prior-at-time) lifted minority-class F1 **0.0808 to 0.1519, +88%**, with leakage disproved by a
    truncation test rather than argued. Workings:
    [`references/finance-gap-investigation.md`](references/finance-gap-investigation.md).
  - **genomics cannot produce an AUROC here, so it does not print one.** 1000 Genomes is a reference
    panel with no case/control status; the published AUROC comes from an access-controlled phenotyped
    cohort (PPM001647, n=400,812). What *is* computable from real genotypes is the portability failure,
    and it is stark:
    **a single global 90th-percentile "high risk" threshold flags 16.2% of African-ancestry samples
    and 2.8% of European - a 5.8x difference driven entirely by ancestry, with no phenotype
    involved.** That is the clinically important finding, not a footnote. (The figures moved with the
    switch from PRS77 to PRS313 - 29.3%/1.0% and 29.5x were the 77-variant score's - and they are
    recomputed on every training run, so read `by_super_population.*.frac_above_global_cut` in
    `models/metrics.json` rather than this sentence.)
  - **satellite reports event-wise F-beta at beta=0.5, never point-adjusted F1.** On the same run the
    point-adjusted figure is **0.994** against an honest **0.4545** - under point adjustment "even a
    random anomaly score can easily turn into a state-of-the-art TAD method" (Kim et al., AAAI 2022,
    arXiv:2109.05257). This sentence cited TSB-AD for "a uniform random score ranks 26th of 32
    detectors" until Round 33, a figure that could not be verified at source. A windowed isolation
    forest runs as a trivial baseline every time (0.0002 here), because a deep model that cannot beat
    it has not earned its complexity. Only labels whose ESA-ADB Category is `Anomaly` count as
    positives; the 690 annotated rare *nominal* events do not.

  - **telecom reproduces the metric trap in-house.** The variational QNN scores **accuracy 0.875 but
    macro F1 0.467**. Stated precisely, because the earlier wording here said "on the same 8-class
    problem" and that was wrong: the QNN is a **binary** attack-vs-normal classifier - `qml_ids/detector.py`
    fits on `binary_labels(y)` and predicts `(preds > 0)` - so scoring its output against the 8-class
    labels guarantees a floor of zero on six of them by construction. The macro F1 is therefore not a
    like-for-like comparison, and the honest reading is narrower than the original claim: an aggregate
    can look respectable while the per-class picture is not, which is why per-class F1 is the headline
    everywhere here. The classical random forest, which **is** trained on all 8 classes, reaches macro
    F1 0.987; neither quantum model beats it.

  **Quantum-ML IDS is presented as an architecture showcase, not a performance claim.** On this
  project's own synthetic telemetry neither quantum model beats the classical random forest. Published
  results, however, are **mixed rather than uniformly below a random forest** - an earlier revision of
  this paragraph claimed they were, and the paper cited elsewhere in this README contradicts it:
  Al-Kuwari et al. (arXiv:2509.14282) Table VIII reports a hybrid QLSTM at 94.7% accuracy / 94.7% F1
  against their own random-forest baseline on the same generated data. What remains well-founded is
  caution about evaluation quality, not a universal ranking: arXiv:2512.05069 was withdrawn by its own
  authors for evaluation "insufficient to fully support the paper's conclusions".

- **There is no third (remote) tier, and that is a decision rather than an omission.**
  `spaces/qsh-inference/` was written to serve full-precision models from a Hugging Face ZeroGPU Space
  for anything too large to run in a tab or under encryption. It was **decided against in Round 24**
  and the record is in `docs/deliverable-verification.md`: the public demo is zero-server by design,
  so a remote provenance contradicts the architecture rather than extending it. The Space itself
  returns **HTTP 401** (re-checked 2026-08-12) and the org Space listing is empty. `ProvenanceBadge`
  therefore declares **four** states - `measured-webgpu`, `measured-wasm`, `measured-js`, `simulated` -
  and `remote` was **removed** from the type in Round 24d, because a type member no code path can
  produce is the same defect as a metric with no reader. **Reopening condition:** wire it only if the
  demo acquires a workload that genuinely cannot run client-side, and re-add the badge state in the
  same commit that first sets it, never before.
  For the record, the hosting facts that shaped the original plan: free accounts can create **Static
  Spaces only** (Gradio and Docker Spaces need a paid plan), with a standing exception of **two
  ZeroGPU Gradio Spaces** for accounts in good standing, which requires an account **older than 30
  days**.

- **Model artefacts live on the Hugging Face Hub, never in git** - `src/qsh/models/registry.py` is the
  single source of truth for repo ids, pinned revisions, artefact paths, dataset licences, published
  baselines and measured metrics. It flows through `frontend_config()` into the SPA, so no model id,
  URL or number is hard-coded in TypeScript. The browser fetches the ONNX graph from the CDN **at a
  pinned 40-character commit sha**, never `main`, so a visitor cannot silently get a different model
  from the one whose metrics the page is printing. Of the 15 registry specs, **11 declare a Hub repo
  and all 11 currently carry a pinned `hf_revision`**; the other 4 are not published at all.
  `onnxInference.ts` deliberately skips anything unpinned rather than fall back to `main`, so such a
  model is not fetched at all - the pinning guarantee holds, and the set it covers is worth stating
  as a number rather than implying. That number MOVES: a retrain demotes `hf_revision` to
  `hf_revision_invalidated_by_retrain` (`train.py`), because the recorded sha now points at weights
  the metrics no longer describe, and it is restored only by re-publishing. Round 49 hit exactly
  that - retraining finance for the calibration fix unpinned both finance models and emptied that
  use case's browser tier until the re-publish landed. Publish with `python scripts/publish_hf.py`; it defaults to
  **private** because the Hub's public-repo grant is "perpetual, irrevocable", and it refuses to
  publish publicly at all when the registry marks the training data non-publishable.

- **Public-demo data (real where the licence permits, synthetic otherwise)** - the client-side demo loads **real
  commercial-OK datasets where shippable**, mirroring the full deployment: **medical** uses a real Wisconsin
  Diagnostic Breast-Cancer sample (UCI dataset 17 via sklearn, **CC BY 4.0 - attribution required**) + a real logistic-regression model;
  **genomics** uses the real PGS-Catalog PGS000004 / PRS313 PRS weights (313 variants; **licensed per score** - PGS000004 carries the Catalog's default terms, *not* a blanket CC-BY grant) over real 1000 Genomes
  genotypes -> a real in-browser **Polygenic Risk Score** with per-SNP signed SHAP; **finance** uses a real
  **PaySim** transaction sample (CC-BY-SA-4.0) + the distilled CKKS-linear fraud model; and **satellite** scores
  **real NASA SMAP/MSL telemetry** (no licence stated by any provider - see DATA_LICENSES.md) with the real linear anomaly model - all tiny static JSON
  under `web/public/data/`, loaded by [`web/src/lib/realData.ts`](web/src/lib/realData.ts) and computed with the
  real in-browser node-seal CKKS + secret-sharing MPC. Only **telecom** has no commercial-OK real attack dataset
  - the narrower and correct statement, since **5G-NIDD is CC BY 4.0 and commercially usable**, and this repository trains on it. What does not exist is a commercial-OK corpus of QKD-LAYER attacks, which is a different problem needing a physical QKD testbed under attack - so the QKD IDS uses representative synthetic features (logged as such in the run).
  The bundled sets are public / de-identified, so no real *personal* data leaves the browser - the privacy property
  is demonstrated on real, realistic inputs.
- **Config single-source** - the benchmark cost model, tier architectures, workload multipliers,
  security/advantage weights and demo data sizes live in one place (`src/qsh/config/defaults.py`),
  served at `/api/config` and baked into `web/src/lib/config.defaults.json` (regenerate via
  `python scripts/export_config.py`), so the demo's numbers cannot drift from the backend.
- **Future tracks** - see [`docs/pqc-roadmap.md`](docs/pqc-roadmap.md): a NIST-aligned IDS
  against HNDL/Q-Day, hardware QKD field test, malicious-secure MPC, and the GPU FHE-at-scale
  path (HEonGPU + PIGEON/hpmpc).

---

## References & license

- Primary sources: [`references/REFERENCES.md`](references/REFERENCES.md) (annotated, paper -> code -> workstream).
- Standards/threat context: [`docs/pqc-roadmap.md`](docs/pqc-roadmap.md).
- Code license: **Apache-2.0** ([`LICENSE`](LICENSE)). Upstream OSS retains its own licenses (table above).

> Private business notes are kept outside this repository by design and are not published here.
