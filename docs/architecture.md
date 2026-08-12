# Architecture

```
                         +----------------------------------------------+
   Browser (SPA)         |            FastAPI backend (api/)             |
   React + Vite          |                                              |
   + @noble/post-quantum |  RunManager -- threads -- Pipeline           |
   + onnxruntime-web     |      |                       |               |
   + React Flow (DAG)    |   WebSocket  <-- EventBus <--+ stages        |
   + recharts            |      |                       v               |
        ^   | REST/WS    |   /api/*                 +------------+       |
        |   v            |                          |  qsh core  |       |
   +------------+  HTTP  |                          +------------+       |
   |   Caddy    |<-------+  COOP/COEP, TLS, /api proxy, static SPA       |
   +------------+        +----------------------------------------------+

   qsh core (src/qsh/) - the E2E security layer
   +-------------------------------------------------------------------+
   | orchestration/  Pipeline . Stage . EventBus . CryptoRegistry        |
   |   pqc/      ML-KEM+X25519 hybrid KEM . ML-DSA sig . AEAD transport   |
   |   qkd/      BB84/decoy sim . channel/QBER . AI optimizer (WS1)       |
   |   qml_ids/  7 attack types + normal . RF + variational QNN (WS2)     |
   |   qsmpc/    Shamir+QFT secure sum . QOT . PSI (WS3)                  |
   |   qhe/      QOTP+Clifford homomorphic . QCNN . Rate-1 transport (WS4)|
   |   fhe/      CKKS LINEAR layers: TenSEAL . OpenFHE (CPU) . HEonGPU GPU|
   |             + genuine FHE CONVOLUTION (conv.py: im2col > dot_plain)  |
   |   mpc/      NON-LINEAR (ReLU/argmax/maxpool): MPyC Shamir shares;     |
   |             layer specs (Conv/Dense/Pool/Flatten) + hybrid dispatch   |
   |   ai/       models . own FedAvg + pairwise-mask secure aggregation   |
   |   benchmark/ hybrid MPC-FHE (linear>FHE+non-linear>MPC) vs QSMPC-FHE |
   |   usecases/ finance_aml . medical_fl . genomics . telecom_ids .      |
   |             satellite_comms                                          |
   +-------------------------------------------------------------------+
```

**Rounds 15-16 additions (WS1 entropy + QKD protocols).** The `qkd/` module also provides a **QRNG entropy
backbone** (`qrng/`: state-vector `|+>` measurement, the **SP 800-90B section 4.4** continuous health
tests (Repetition Count + Adaptive Proportion) which gate the verdict, with SP 800-22 p-values reported
but NOT gating -- section 4.1 states those tests "are not useful for monitoring the noise source" -- + an **SP 800-90B**
most-common-value **min-entropy** estimate, and a **Leftover-Hash-Lemma-bounded Toeplitz / seedless two-source**
extractor - Giestinhas & Spiller, arXiv:2603.26907) that **can** seed the BB84/E91 basis+bit choices (**opt-in, off by default**: `qrng.enabled = False` in `src/qsh/config/defaults.py`, so BB84 entropy is byte-for-byte unchanged unless enabled), plus a selectable
**E91/CHSH** entanglement protocol (`qkd/e91.py`, `protocol="e91"`) alongside the two-intensity decoy-state BB84
(`bb84.py`, where `mu_decoy` genuinely moves the SKR - Ma-Qi-Zhao-Lo 2005; OptiQKD tunes it, arXiv:2603.04192).
The `pqc/` module adds a config-gated **leakage-resilient** PQC+QKD key combiner (`transport.py`,
`hybrid_combiner_mode = hkdf | its_toeplitz` - Toeplitz+QLHL; the session key stays secure if EITHER the PQC or
the QKD input retains entropy). The QRNG post-processing and the combiner mode are both runtime-selectable - see
`crypto-agility.md`.

**Round 18 addition (QKD networking control plane).** The demo talked about QKD networks and had no
control plane, while an Agentic-AI panel sat next to it implying the agent drove the quantum layer.
Two modules close that gap:

- `qkd/sdn_controller.py` implements **ETSI GS QKD 015 V2.1.1**, the SDN control interface. Field
  names (`qkdn_id`, `qkdl_id`, `qkdl_type`, `virt_next_hop`, `phys_channel_att`, `skr`, `eskr`) come
  verbatim from the published YANG module, so a reader can line the topology payload up against the
  standard without a translation table. The controller distinguishes a **PHYS** link (direct
  quantum channel) from a **VIRT** one (multi-hop through trusted relays) and makes every VIRT link
  state its trusted-node exposure - clause 5.4 relaying means an intermediate node holds the key in
  plaintext, which is the part of QKD networking a demo usually hides.
- `qkd/kme.py` implements the **ETSI GS QKD 014 V1.1.1** Key Management Entity: `status`,
  `get_key` and `get_key_with_ids`. The master SAE draws `(key_ID, key)` pairs and the slave later
  presents only the `key_ID`, so key material never crosses the application channel.

`web/src/lib/clientSdnController.ts` mirrors both for the browser, which is why the public demo can
draw three separate planes instead of implying two. **The MCP agent is a SAE** (Secure Application
Entity), an application-plane key consumer - GS QKD 015 clause 1 puts application-layer and
quantum-channel interfaces out of scope, so an agent that "controls QKD" would be a category error.
`tests/test_sdn_qkd.py` (13 tests) asserts the two properties that make this worth having rather
than decorative: a VIRT link must declare its trusted-node exposure, and a key ID must not be
redeemable by an SAE it was not issued to.

The quantum layer underneath remains **simulated** - there is no QKD hardware anywhere in this
project - and every payload says so in-band rather than leaving it to the reader.

## The pipeline contract
A `Pipeline` runs an ordered list of `Stage`s over a shared `Context`, emitting
structured `Event`s to an `EventBus`. The canonical chain:

```
PQCHandshake > QKDKeyAgreement > QSMPCCompute > PrivateInference > Result
```

Each stage resolves its algorithm from the `CryptoRegistry` (crypto-agility), so the
same chain runs PQC-only or PQC+QKD, TenSEAL or OpenFHE (linear-FHE), MPyC or MP-SPDZ
(non-linear-MPC), ML-KEM-768 or -1024 - **selected by the operator from configuration and
pinned into the run config before the run starts, never negotiated in-band between parties**
(see [`crypto-agility.md`](crypto-agility.md)). Use cases swap in their own
data and (for medical) extra stages (`FederatedTrainingStage`, `QMLIDSStage`).

## Faithful hybrid private inference (linear -> FHE, non-linear -> MPC)

### Where the non-linear half goes, and why that choice is a security decision

There are two live answers in the 2026 literature and this repository takes the more expensive one on
purpose.

**Offload the non-linearity to the client in plaintext.** Safhire (Biswas et al., *Practical and
Private Hybrid ML Inference with Fully Homomorphic Encryption*, arXiv:2509.01253, 1 Sep 2025) runs
linear layers encrypted on the server and sends intermediate values to the client, which evaluates
the activations in the clear. It is the closest published architecture to this one, and the payoff is
large: no bootstrapping, exact rather than polynomial-approximated activations, and 1.5x to 10.5x
lower latency than Orion. The cost is that the client now sees per-layer intermediates, so model
confidentiality has to be recovered by obfuscation - Safhire uses randomised shuffling of
those values.

**That obfuscation has since been broken.** Kim and Lee, *Shuffling is Not Enough: Breaking
Permutation-Based Model Confidentiality in Hybrid FHE Inference* (IACR eprint 2026/1527, ESORICS
2026), recover a permutation-invariant summary of a linear layer exactly, needing `d+1` queries for a
layer with `d` inputs - 5,712 queries for every linear layer of a ResNet-20 - and achieve perfect
model distinguishability. Shuffling plus noise, defended by an appeal to shuffle-model differential
privacy, does not hold.

**This repository sends the non-linear half as secret shares instead**, evaluated under MPyC secure
comparison, so the client never holds a plaintext intermediate to query. That is slower and it buys
something specific. **The attack paper does not analyse the MPC-evaluated case**, so the honest
statement is not "we are immune" but "the attack's precondition - a client that observes plaintext
intermediates - is absent here, and no one has published the MPC analogue." It is recorded as an open
question rather than a resolved one.

**This is a trip-wire, not just background.** If the non-linear path is ever changed to plaintext
client offload for the latency win, eprint 2026/1527 becomes directly applicable and the model
confidentiality claim must be withdrawn at the same time. Anyone making that change should read it
first.

**A third option exists as of 2026, and it is named here rather than left implied away.** The split
above rests on non-linearities being expensive under CKKS. Lin and Wu's **Tailor** (IACR ePrint
2026/1551, approved 2026-08-03) evaluates **ReLU and comparisons exactly under FHE** - not
polynomial-approximated - with per-layer precision as a runtime parameter under a single keygen,
reporting MNIST at 96.44% in 222 s. So there is a design that needs **no MPC round at all**.

That does not refute this repository's choice, and the reasons are specific rather than defensive.
Tailor's backends are TFHE and FINAL, not CKKS, so adopting it means changing scheme and not just
layer assignment; 222 s is a server measurement, against a demo whose budget is a page interaction in a
single-threaded WASM tab; and the exactness that makes it attractive is what makes it slow. But the
sentence "non-linearities go to MPC because CKKS cannot do them cheaply" is now only half the story,
and the half it omits is the one a reviewer will raise. Recorded as an alternative design with its
trade-off, not as a refinement of this one.

**The measured number that closes the argument for the specific non-linearity this demo needs.** The
final step of a classifier is an argmax, and there is now a published cost for doing that under CKKS:
Mazzone, Everts, Hahn and Peter, *Efficient Ranking, Order Statistics, and Sorting under CKKS* (USENIX
Security 2025, arXiv:2412.15126), evaluate argmin/argmax over a **128-element vector in 12.83 s**, with
comparison depth *"of up to 2 (constant)"*. That is a state-of-the-art result, not a naive baseline,
and the constant depth blocks the obvious rebuttal that the cost is a depth artefact to be optimised
away. Twelve seconds for one argmax **rules out a homomorphic argmax in a browser tab**, which is why
the logits go out as secret shares to a summon-compiled comparison circuit (`web/src/lib/clientMPC.ts`,
`src/qsh/mpc/secret_inference.py`) rather than staying under CKKS. The split is now a measured decision
rather than a stated preference.

### What the plaintext-versus-CKKS comparison proves, and what it does not

The demo measures its CKKS result against a plaintext reference computed in the same tab
(`web/src/lib/clientFHE.ts` returns `maxAbsError`, about 1e-9 on the linear path, and the convolution
path computes its own against a plaintext reference). That comparison is worth having and it is easy to
overstate, so the boundary is drawn here rather than left to a reader.

**It is empirical fidelity. It is not cryptographic integrity.** The verifiable-FHE literature proves a
different property: that **the server ran the circuit that was requested**. Atapoor, Baghery, Pereira
and Spiessens (*Verifiable FHE via Lattice-based SNARKs*, IACR CiC 2024, ePrint 2024/032) verify an
approximate neural network of three layers and more than 100 ciphertexts in *"less than 1 second"*, and
Tremblay Thibault and Walter (Zama, *Towards Verifiable FHE in Practice*, ACM CCS 2025, ePrint
2024/451) prove TFHE bootstrapping with a prover under 20 min, a proof of about 200 kB and a verifier
under 10 ms. Neither proves that the decrypted result matches a plaintext reference - that is not what
those proofs say - and conversely, an agreement figure proves nothing about whether the evaluator
followed the circuit. **This project implements neither construction**, so the honest reading of the
comparison is "the approximate scheme agreed with the exact one on this input", full stop. Anyone who
knows this literature will otherwise read a side-by-side agreement number as an integrity claim, and
nothing here establishes integrity.

In the taxonomy of Peng et al. (*A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning*,
arXiv:2502.18535) - verifiable training, verifiable testing, verifiable inference - this dual-path
comparison sits in **verifiable testing**. Naming the slot is useful precisely because it is the modest
one.

**How to report it, using someone else's convention rather than inventing one.** Two exist, and they
are the same measurement in two directions:

- **An integer count of disagreements, expected zero.** Li, Wu and Miao (arXiv:2605.22237) report
  **"PH-mis"** in Table IX, defined verbatim as *"the integer number of top-1 mismatches between
  plaintext polynomial evaluation and CKKS evaluation on the evaluated test set; it is zero for every
  selected row"* - reported **beside** plaintext and ciphertext accuracy, not folded into either.
- **A percentage of agreements.** Zama's Concrete-ML prints **"Similarity"**, the mean per-sample
  agreement between clear and FHE predictions. That is the convention from the vendor with the largest
  FHE-ML install base.

Either is defensible; neither is this project's invention; reporting an unlabelled error magnitude with
no agreement statistic is the option to avoid. Note also that Concrete-ML's deployment model **returns
class scores and has the CLIENT decrypt**, which is the mainstream construction for encrypted multiclass
inference and is what this demo already does - with the client and the key holder being the same tab.

### The split itself
The 2026-standard private-inference split (Gazelle -> Delphi -> BOLT -> Bumblebee -> BLB; EncFormer)
is **linear layers under FHE** (matmul/conv as CKKS; GPU-accelerable via HEonGPU) and **non-linear
layers under MPC** (ReLU/argmax via secret-sharing secure comparison). `qsh.mpc.secret_inference`
implements it with TenSEAL/OpenFHE for the linear half and **MPyC** (Shamir shares, the CrypTen
`ReLU([x]) = [x], [x>0]` pattern) for the non-linear half, converting CKKS <-> shares at each boundary;
the benchmark reports the FHE-linear vs MPC-non-linear latency split + MPC communication. The linear
half includes a **genuine FHE convolution** (`fhe/conv.py`): im2col turns the encrypted image into
receptive-field patches and each output `(channel, position)` is one `dot_plain(patch_ct, kernel)` +
bias - the same CKKS primitive as a dense layer, no rotations. The engine dispatches on tagged layer
specs (`mpc/layers.py`: `ConvSpec`/`DenseSpec`/`PoolSpec`/`FlattenSpec`), so the **medical** use case
runs a real MedMNIST-scale CNN under FHE-linear conv + MPC `secure_relu`/`secure_maxpool` (the dense
MLP path is byte-identical via `normalize_layers`). FHE-feasible by design - MedMNIST-scale, not ResNet. The
**QSMPC-FHE** quantum analogue is Clifford gates (linear) vs T-gates (non-linear), both evaluated on
the encrypted state by QOTP/Rate-1 QFHE (no FHE <-> MPC boundary). **On ct/pt:** the benchmark row the demo renders reports **1.0** - under QOTP the ciphertext is the same number of qubits as the plaintext (`clientCompute.ts`). The config's `cost_per_op.qsmpc_ct_pt_ratio = 1.32` is the *literature* figure for Rate-1 QFHE with the classical key included, and it feeds only the static blurb on the Overview page, never the benchmark. The **lightweight / medium / large** model-size selector was removed in Round 20 along with
`tier_architectures`: it scaled invented MLPs rather than the models that exist. The benchmark now
costs the student that actually runs.

## SPA shape
Overview, Introduction & Value, Validations, E2E Orchestration (generic "run any pipeline"), Use Cases
index, and **5 per-use-case simulator pages** (`/usecase/:id`).

**Agentic AI, MCP** is a **cross-cutting per-use-case layer** (`web/src/components/AgenticAIPanel.tsx`) - a
reusable panel **embedded in every use-case page**, not a standalone page (the report frames MCP as the AI
integration layer under the quantum-safe channel, applied per use case; the standalone `/agentic-ai` page was
removed in Round 12 and now redirects to the Use Cases index). It runs a small language model **in the
browser** - Transformers.js (`@huggingface/transformers`, onnxruntime-web WebGPU/WASM) or WebLLM
(`@mlc-ai/web-llm`, WebGPU), both Apache-2.0 and **dynamically imported** (async chunks, never in the entry
bundle). The AI request/response is wrapped in a **client-side quantum-safe MCP channel**
(`web/src/lib/clientMcp.ts` - a zero-server mirror of `pqc/mcp_bridge.py` + `transport.py` via @noble
ML-KEM-768/ML-DSA-65 + WebCrypto HKDF/AES-256-GCM), with a **per-industry MCP tool** (`score_transaction` /
`query_cohort` / `pool_allele_counts` / `inspect_qkd_telemetry` / `relay_mission_telemetry`) and the live
run's metrics composed into the prompt. Everything fails soft (no WebGPU -> Transformers.js WASM; the MCP
envelope always runs). Each use-case page
unifies a **live two-party (Alice <-> Bob) sequence diagram** (driven by the running pipeline's
stages - Start/Pause/Step/Stop/Restart), that industry's benchmark, and a signed SHAP explanation
(all five use-case pages). **One Run drives both** (Round 7): the live pipeline runs, and the
`QuantumComparisonStage` assembles the benchmark **from that run** into `ctx.artifacts["benchmark_result"]`
(rides on `/api/runs/{id}`, no extra endpoint) - the MPC-FHE row is the run's real measured FHE-linear +
MPC-non-linear latency, next to the QSMPC-FHE comparison + advantage. There is **no separate "Run
benchmark" button**; `BenchmarkPanel` is a controlled results panel. The diagram makes the MPC-FHE
secret-sharing over **LAN <-> WAN <-> LAN** explicit (CKKS ciphertext for the linear half, secret-shares for
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
- **FHE** - `web/src/lib/clientFHE.ts` runs **node-seal** (Microsoft SEAL -> WASM, MIT) **CKKS**: the
  feature vector is encrypted, the linear layer `W, x+b` is evaluated homomorphically (multiply-plain +
  rotate-and-add slot sum), and only the decrypted logits leave encryption. Reported FHE-linear ms,
  ciphertext bytes and the approx. 1e-9 approximation error are real `performance.now()` measurements.
- **MPC** - `web/src/lib/clientMPC.ts` runs a real **garbled-circuit authenticated-garbling 2PC**
  (`mpc-framework` + `emp-wasm-engine` + `summon-ts`, all MIT): the logits are additively secret-shared,
  a summon-compiled argmax circuit privately reconstructs + compares them, and **both parties run loopback
  in one tab** (zero VPS) - the reported MPC ms + real garbled-circuit communication bytes are measured.
- **QFL** - Quantum Federated Learning runs on every run; it is not a toggle. It performs a real
  client-side FedAvg over the page's own bundled sample, so it is part of what the demo does rather
  than something to opt into. Full deployment:
  `ai.federated.federated_train(kind="qfl")` trains a hybrid **PennyLane VQC (`HybridQNN`) + classical**
  model with FedAvg under quantum-safe secure aggregation (CPU, since `default.qubit` simulates the VQC).
  Demo: `clientCompute.clientQFL` is an in-browser FedAvg of a logistic head over the `quantumSim` QCNN
  feature map, round accuracy genuinely computed. Both honest small-VQC scope (not fault-tolerant QFL).
- **Real MCP server** (`src/qsh/mcp/server.py`, full deployment only) - an **`MCPServer`** server (MCP Python
  SDK, MIT) exposing the 5 per-industry tools; `pqc/mcp_bridge.secure_agent_call` now **dispatches the
  real tool** and seals the **ML-DSA-signed result** back over the PQC(+QKD)+AES-GCM channel (disjoint
  nonce ranges per direction). `src/qsh/mcp/fs_interop.py` connects (stdio) to the **official
  `@modelcontextprotocol/server-filesystem`** and carries its real result under the same quantum-safe
  seal - real external-MCP interop. `/api/mcp/{tools,call,filesystem}`. The demo keeps the client-side
  `clientMcp.ts` mirror, so the VPS takes **zero per-user MCP load**.
- **Per-industry real data/models** - finance PaySim + XGBoost (distilled to a CKKS-linear surrogate),
  medical MedMNIST CNN, **genomics PGS-Catalog Polygenic Risk Score (linear -> native CKKS dot)**, telecom
  QKD-attack data + Quantum-SVC, satellite ESA-ADB telemetry + a CKKS anomaly score. The FHE-*encrypted*
  inference is the FHE-feasible part; the full model is the real industry architecture (honest crypto scope).

**Backend layout (audited Round 11, kept lean).** Modules: `pqc/` (hybrid KEM, transport, signatures,
MCP bridge), `qkd/` (BB84, channel, QSDC), `qsmpc/` (QOT, Shamir, secure summation), `fhe/` (CKKS
inference, conv, backends), `mpc/` (secret inference, layers), `ai/` (models incl. `HybridQNN`, federated,
explainability), `qml_ids/`, `benchmark/`, `usecases/`, `orchestration/`, `mcp/` (new), `config/`
(single source). The demo going fully client-side did **not** orphan most backend code - with three
exceptions: `/api/mcp/tools`, `/api/mcp/call` and `/api/mcp/filesystem` have **no** reference anywhere in
`web/src`. The SPA's MCP path is `/api/demo/mcp` or the in-browser `clientMcp.ts` mirror, so those three
are reachable only by an external MCP client. Otherwise every `/api/*` endpoint is still used by the
full-mode SPA, the dense+conv hybrid dispatch is already unified
(`normalize_layers`), and config is centralized - so the reorg was an audit + the clean new `mcp/`
module rather than a churn of working, tested code (the full test suite stays green).

## Two compute topologies
- **Full** (GPU box): everything runs server-side - federated training (own FedAvg + PennyLane QFL), GPU FHE
  (HEonGPU), GPU quantum sim (CUDA-Q / lightning.gpu), the real MCP server, Prometheus + Grafana.
- **Client-side build** (no GPU, no server-side ML): the static SPA runs **all** heavy compute in the
  browser (WASM/WebGPU PQC via @noble, **real CKKS FHE + garbled-circuit MPC**, ONNX inference, in-browser
  AI, JS BB84/PSI/SHAP/QFL, the client MCP mirror) - it makes **zero `/api` compute calls** (Validations'
  PQC/QKD/MCP are client-mirrored). It ships **no backend at all**. A slim
  API image (core+api, no FHE/torch) used to run beside it as a `?demo=full` fallback. Two separate
  removals a round apart: **Round 17** stopped serving it after testing it on the deployed site, and
  **Round 18** deleted `docker/Dockerfile.api.demo` itself once nothing referenced it. The
  image excludes FHE by design, so any run it accepted failed at `No module named 'tenseal'` and
  ended in `State: error`, while the page still told the visitor no computation was sent to a
  server. It also left `/api/run/*` writable by anonymous visitors. The client-only image now
  carries `docker/env/Caddyfile.client-only`, which answers `/api` and `/metrics` with 404, and
  `VITE_CLIENT_ONLY=1` makes `?demo=` non-overridable - a client-only build is client-only.
  The **demo** topology therefore offers inference plus an optional in-browser QFL toggle; the **full**
  topology additionally runs server-side federated learning + FHE inference + the MCP server. AI model weights stream from the
  HF/MLC CDN (browser-cached).

## Why CUDA-Q / cuQuantum (not qiskit-aer-gpu)
`qiskit-aer-gpu` cu11/cu12 wheels do not target Blackwell `sm_120`. On this hardware
the GPU quantum-simulation path is CUDA-Q + cuQuantum (PennyLane `lightning.gpu`,
SM 7.0+). Qiskit (CPU/Aer-CPU) is still used for circuit authoring and small sims.
