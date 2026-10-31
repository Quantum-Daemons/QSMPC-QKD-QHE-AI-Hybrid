# Findings, concerns and recommendations

## WS3 + WS4 challenges addressed

The four workstreams target specific open problems in quantum-safe + private AI; here is how this
implementation maps to each, and where gaps remain.

**WS3 - QSMPCxQKD** (cross-org AI restricted by GDPR/HIPAA; classical SMPC is complex, HNDL-vulnerable, sometimes assumes a TTP; Agentic-AI/MCP lacks quantum-safe + private agent comms):

| Sub-problem | How QSH addresses it | Where |
|---|---|---|
| Raw data must not be shared | Federated learning: raw rows stay with each client, but the masked updates are not hidden from the aggregator (see (b) below). PSI: an HMAC of both banks' lists under one session key that both banks hold, so no third party sees the lists and the channel is PQC-protected, but over a small ID space either bank can hash every candidate ID and recover the other's list; a production PSI would use an OPRF-based protocol. FHE private inference | `ai/federated.py`, `qsmpc/qot.py`, `fhe/` |
| Classical SMPC protocol complexity | QSMS = Shamir + d-level QFT (no Beaver triples / garbled circuits) | `qsmpc/secure_summation.py` |
| HNDL-vulnerable HE / key exchange | PQC-first hybrid **ML-KEM + X25519** (+ simulated QKD key material in the full deployment only; the browser mixes none into any key) | `pqc/hybrid_kem.py`, `pqc/transport.py` |
| TTP assumption | No dealer in QSMS (entangled state) and no third party in the keyed-hash PSI, which runs between the two banks over the PQC channel. Pairwise-mask secure aggregation needs no dealer either, but its masks hide nothing from the aggregator here: in the browser every mask is an HKDF output of the one PQC session key, which the aggregator also holds, and in Python the default pair keys are a public constant | `qsmpc/`, `ai/federated.py` |
| **Agentic-AI / MCP quantum-safe + private agent comms** | A **real MCP server** (MCP Python SDK, `mcp>=1.2`; `MCPServer` on SDK 2.x, `FastMCP` on 1.x) (5 per-industry tools, full deployment) bound to a quantum-safe channel: two agents establish a PQC session, optionally mixed with a fixed 32-byte QKD-domain constant (no QKD key reaches this channel), the caller sends an **ML-DSA-signed, AES-GCM-sealed** `tools/call`, the callee **dispatches the real tool** and returns the **signed result** (authenticated, confidential, HNDL-resistant, no TTP); also interops with the official filesystem MCP server | `pqc/mcp_bridge.py`, `mcp/server.py`, `/api/mcp/*` |
| **Explainability / audit trail (XAI)** | Exact linear-SHAP feature attributions for each run's classification, **ML-DSA-signed** -> tamper-evident, regulator-facing explanation (EU AI Act Art. 13, for high-risk systems only - 2 Dec 2027 for Annex III and 2 Aug 2028 for Annex I after the AI Omnibus, not Aug 2026; medical and genomics would be Annex I products; finance AML is not high-risk for a bank but would be for law-enforcement profiling (Annex III 6(e)); the telecom IDS is not as a cybersecurity tool and is outside the Act if used exclusively for military, defence or national security purposes (Art. 2(3)); satellite would be only if its service is itself critical digital infrastructure and the detector a safety component, since space is a separate sector under the CER and NIS2 directives; GDPR Art. 22; SR 11-7); surfaced on all five use-case pages. On telecom and satellite, whose flows have no `private_inference` stage, the attribution is of the model that produced the reported class; on the other three it explains the bundled model beside the encrypted one, and the panel says which case applies | `ai/explainability.py`, `ExplainabilityStage` |

**WS4 - QHExAI** (FHE $`10^3`$ to $`10^6`$x overhead; bootstrapping bottleneck; no quantum encoding; plus unsolved (a)-(f)):

| Sub-problem | How QSH addresses it | Where |
|---|---|---|
| (1)/(e) huge compute + ciphertext/plaintext overhead | The benchmark measures the CKKS ciphertext/plaintext byte ratio for this configuration: about 1393x at 30 of 4,096 slots (the table in [`docs/models-and-benchmarks.md`](models-and-benchmarks.md)), about 12x at full packing, so it measures slot utilisation rather than CKKS itself. The client demo's 1483x (`cost_per_op.demo_ct_pt_ratio`) is a representative constant used when no ciphertext was measured, not a measurement. The Rate-1 QFHE module reports a byte ratio of a SIMULATED state vector (1.2769 at 8 qubits, 1.0693 at 10, 1.0173 at 12), which cannot be compared with it; in the paper's own units, qubits plus bits, its 8-qubit KEM-wrapped transport is about 1135x. The paper itself reports no concrete ratio. See `docs/architecture.md` | `benchmark/`, `qhe/rate1_qfhe.py` |
| (2) bootstrapping bottleneck | **PARTLY met, on the CPU only.** OpenFHE CKKS `EvalBootstrap` refreshes ciphertext levels: at ring dimension 65536 one bootstrap takes 7.7 to 9.1 s and returns a level-22 ciphertext at level 19, on the pinned OpenFHE 1.5 series (`docs/verification-log.md`). HEonGPU's collective bootstrapping is **not run here**: `HeongpuBackend` is a build-gated reference adapter that raises, because HEonGPU exposes no Python API, and no measured HEonGPU figure exists in this repository. Both libraries are commercial-OK | `fhe/backends.py` (`OpenFHEBackend(use_bootstrap=True)`, `HeongpuBackend`) |
| (3)/(a) quantum encoding / practical QHE | QOTP + a variational QCNN + a Rate-1 quantum transport (the module has no homomorphic `Eval`) at small scale. **Three different circuits, and only one is Clifford+T:** the BROWSER's QSMPC-FHE comparison circuit (`qotpQcnnInfer` in `web/src/lib/quantumSim.ts`) is untrained and built from H, CNOT and T on a padded state, counts 7n real T gates (56 at 8 qubits), and predicts nothing; the TRAINED QCNN that decides the class on the finance, medical and telecom pages when Inference is set to QCNN (`qcnnInferTrained`, RY, RZ, CNOT and CRY) has **no T gate and no mask**, and its blocks act on disjoint pairs of wires, so it is a tree of two-qubit blocks and classically simulable; and the PYTHON module (`qhe/qcnn.py`) contains **no T gate at all** - it is `AngleEmbedding` plus CNOT/RY/RZ, and its `t_count()` is a labelled Solovay-Kitaev estimate of what a fault-tolerant deployment would cost, not a count of gates present. | `qhe/qotp.py`, `qhe/qcnn.py`, `web/src/lib/quantumSim.ts` |
| (b) QFL gradient leakage (arXiv:2508.15998) | **NOT met.** Pairwise-mask secure aggregation runs and its masks cancel exactly, but they hide nothing from a holder of the session key: in the browser every mask is an HKDF output of the one PQC session key, which the aggregator also holds, and in Python the default pair keys are `sha256(f"pair-{i}-{j}")`, a public constant. Bonawitz et al. (CCS 2017) derive each pair's seed by a pairwise key agreement, and even a correct protocol does not bound what the aggregate leaks; no differential-privacy budget is applied. QOTP-delegated learning is not implemented | `ai/federated.py`, `web/src/lib/clientQFL.ts` |
| (c) circuit privacy vs a malicious server | **MET for Clifford circuits.** A random Clifford word over {H, S, CNOT} is evaluated on the QOTP-encrypted state; the client tracks the Pauli frame through the same word and recovers the plaintext result at fidelity 1.0, tested at depths 8, 12, 20 and 30, one seed each. The server sees the maximally-mixed state throughout, so both the input and the computation's operand are hidden from it. Arbitrary Clifford words need the S frame rule (`H` and `CNOT` alone reach only a subgroup) and a CNOT applier not hardcoded to wires 0 and 1, and both are implemented. **Does NOT extend to T:** `T X T-dagger` is not Pauli, so the frame does not propagate, which is why the QCNN's non-linear layers are costed as magic-state T-gates instead. | `qhe/qcnn.py` (`qotp_clifford_random_circuit_demo`), `qhe/qotp.py` (`update_key_h/s/cnot`) |
| (f) hiding the server's circuit from a malicious client | **NOT met, and in tension with the mechanism rather than merely unimplemented.** The client can only decrypt by applying the circuit's action to its own Pauli frame, so under plain QOTP it must know the circuit - "the client updates its frame" and "the client does not learn the circuit" are close to mutually exclusive. Earning this needs a construction where that does not hold, which is what Ortega and Martin-Delgado's scheme addresses and this repository does not implement. Split from (c) so the one that is met is not held hostage to the one that is not. Cf. QOTPH (arXiv:2604.19256), which builds universal QHE from QOTP over Clifford+T with Pauli key updates and reports results on real quantum processors. | not implemented |
| (d) efficiency on AI/HPC infra | **NOT met on GPU.** TenSEAL and OpenFHE run CKKS on the CPU, and that is what every measured figure here comes from. HEonGPU (GPU) is a build-gated reference adapter with no Python API and no measured run in this repository, and CROSS is not in this repository at all | `fhe/`, `benchmark/` |

**Residual gaps, each with the condition that would close it.** Two of the four are downstream of
decisions already recorded in [`docs/deliverable-verification.md`](deliverable-verification.md), so
they are one item each rather than independent gaps:

| residual | condition that closes it |
|---|---|
| **PQ-OPRF PSI + identifiable abort** | Name the browser-loadable library that supplies each, at the point where it is claimed. The non-linear 2PC layer **already runs a protocol designed for malicious security** - Polytune implements WRK17b multi-party authenticated garbling (Wang-Ranellucci-Katz, CCS 2017), secure against malicious adversaries controlling all but one party. Whether this pre-release implementation achieves that between hosts is open: the Round 68 and Round 72 reviews in `docs/threat-model.md` record a gap in the proof of its OT extension's check (upstream #63) and a check whose challenge is fixed earlier than the protocol specifies, so nothing here claims malicious security between hosts - so what is missing is these two primitives specifically: a post-quantum OPRF for the PSI, which today is a keyed hash under a key both banks hold, so over a small ID space it does not keep either list private from the other bank, and **identifiable** abort, which authenticated garbling does not provide. |
| **Hardware QKD field test** | A boundary of the deliverable, not a gap in it: no QKD device exists here and there is no browser API for one. If a characterised link ever exists, the first thing published is the **measured QBER against `qkd/decoy_reference.py`'s prediction for the same link budget** - and that is also the round in which the 2026 finite-key proofs (arXiv:2606.29943, arXiv:2608.09674) replace Ma et al. (2005), because it is the round their device inputs exist. |
| **Larger fault-tolerant QHE** | `t_count()` is a labelled Solovay-Kitaev **estimate** of a fault-tolerant deployment's cost. Replace it with an executed count when a backend exposing logical, error-corrected qubits is addressable from PennyLane. |
| **Production bootstrapping throughput** | OpenFHE's `EvalBootstrap` runs here (one bootstrap, 7.7 to 9.1 s at ring dimension 65536), but that is a single-ciphertext timing, not a throughput. Publish a throughput figure when a batched bootstrapping benchmark runs end to end, on the CPU path or on HEonGPU once it has a Python binding. |

(A real quantum-safe MCP server + filesystem-MCP interop now ships in the full deployment.)
See [`docs/pqc-roadmap.md`](pqc-roadmap.md).

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
  `telecom_ids.qml` 5.9 ms `1x7` (its published ONNX graph is a 200-tree random forest, not the
  quantum models its name suggests), `satellite_comms.telemanom` 9.6 ms `1x250x6`.

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
  **Round 60: a hardware adapter can be just as slow.** On an NVIDIA RTX PRO 6000 under Chrome 152 on
  Linux, one ResNet-18 `session.run` took 18216 ms on `onnx-webgpu`, and distilGPT-2 ran at 750 to
  1140 ms per token on WebGPU against about 9 ms on WASM. So the small models now run on WASM by
  default and on WebGPU only when the visitor selects it; other GPUs and browsers were not measured.

- **Three of the four CKKS students can be re-scored in a browser.** The Models panel runs each
  student's ONNX graph in plaintext on its held-out sample; this checks the published graph, not the
  encrypted evaluation. Samples ship model-ready with `preprocessing: []`, because
  the browser's `applyPreprocessing` is per-scalar and cannot apply a 32-length mean/scale vector or
  an area resize. Reproduced before publishing: medical accuracy 0.769231, OPS-SAT rare-class F1
  0.904348, finance rare-class F1 0.313187 on its shipped rows. `satellite_comms.student` has
  none, and the blocker is its METRIC, not a missing scaler: it is scored on event-wise F0.5, which
  is defined over contiguous runs of a timeline and cannot be computed from independent rows.

- **The browser scores each student on the metric it was published with.** Three of the four record
  rare-class F1, and reporting accuracy for them would have confirmed a number nobody claimed - the
  finance student is about 99 percent accurate on a test window that is 0.15 percent positive. Its
  F1 is 5.1x higher on the shipped rows (0.3132) than over the whole test window (0.0614), because
  the 8,192 rows are enriched to 2.0 percent laundering; not even the ratio to the all-positive
  baseline survives the change of prevalence (8.0x here against 20.2x on the whole window), so the
  sample checks that the browser reproduces the graph, not how good the student is, and the page
  says so beside the figure.

- **Two new trained models, and one of them is a negative result.** `medical_fl.resnet50` exists to
  test a published claim rather than to win one: MedMNIST v2 reports ResNet-50 at AUC 0.857 against
  ResNet-18's 0.901 on this 780-image corpus, so the deeper network is expected to lose. It does,
  and by more than the published gap - 0.7988 against our ResNet-18's 0.9147, both trained from
  random initialisation. A third consequence of the same over-parameterisation appears in the
  export: its int8 graph agrees with its own fp32 on
  0.9865 of the confident set against a 0.99 gate, so the int8 artefact was built, measured and
  **withheld** rather than shipped with the gate relaxed. It replaces a DenseNet-121 card that named
  a dataset with no loader and linked a Hugging Face repo that did not exist.
- **`satellite_comms.opssat` ships its whole published test partition.** OPS-SAT-AD is CC BY 4.0
  and its feature table (`dataset.csv`) is 507 KB, so the whole test partition ships in the model's
  own HF repo. The partition's labels are what the dataset's
  30 published baselines are quoted against (AUC PR 0.9681 measured offline by the training script,
  against FCNN 0.979, XGBOD 0.975, RF+ICCS 0.963); the browser re-scores the exported ONNX graph on
  the same 529 segments, which gives accuracy 0.9509 where the scikit-learn model records 0.9660,
  because 8 of the 529 predictions differ between the two. It is a **different
  task** from the Telemanom forecaster on the same page - segment classification, not forecasting -
  and neither number is the other's baseline.
- **The encrypted path evaluates the published medical student itself, one layer at a time.** All
  3,226 parameters of it, fetched from its own pinned Hugging Face revision. The convolution (im2col
  patches packed into a single ciphertext) and both dense layers run under CKKS; between them the
  client decrypts each layer's output and applies ReLU and average pooling in plaintext, and the
  CNN's argmax is taken in the clear (the garbled-circuit argmax on the medical page runs on the
  WDBC logistic regression, not on this CNN). The run
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
- **`finance_aml.gnn` is a decision, not a backlog item, and the reason is the encrypted path, not
  model strength.** On AMLworld HI-Small the published GNNs are the stronger models: the registry
  records GFP + XGBoost at minority-class F1 63.23 (Altman et al., NeurIPS 2023 D&B) against
  Multi-PNA+EU at 68.16 (AAAI 2024) and MEGA-PNA at 74.01 (arXiv:2412.00241v2, a preprint), and the
  XGBoost shipped here, without IBM's preprocessor, scores 15.19. The decision rests on the FHE
  path: arXiv:2411.02926v2 ran this comparison under FHE and the **GNN pipeline never compiled** -
  it failed on ONNX operator support in Concrete ML - while the XGBoost pipeline completed at an
  overhead of roughly 15,000x to 166,000x over its two datasets. That is a **2024** result, and
  DESIGN (arXiv:2507.05649) and TGHE (arXiv:2606.26664) have since run GNNs under FHE - so it is a
  dated result rather than a standing impossibility.

- **The models sized for encryption are published too.** Each use case has a teacher and a
  distilled student sized for CKKS: 33 parameters for finance against a 141,154-parameter teacher,
  3,226 against 11,171,266 for medical, 49 against 80,486 for satellite. In the browser, CKKS
  evaluates the medical student and the telecom student; the finance page's CKKS step runs a linear
  surrogate over the 8-feature PaySim demo sample rather than the 33-parameter AMLworld student, and
  the satellite page runs no CKKS step at all (the full deployment's Python pipeline scores
  satellite telemetry under CKKS). All eleven repos are public and
  pinned, and the evaluation sample for each lives in that model's own repo at the same revision as
  its weights - except the satellite student's, which publishes none by design, because its metric
  is defined over a timeline and cannot be re-scored from independent rows.

  `report/generated/macros.tex` derives the repo count (`\npublished`) from the registry, so this
  sentence is the second copy, and a test holds its count and its one exception to the registry.

- **Retracted: the satellite forecaster is not worse than persistence.** Round 18 reported it as
  roughly 19-22x worse on next-step MAE. That was our measurement error: the probe resampled the
  telemetry at 60s while the published model is trained at 300s, and feeding a 300s model 60s data
  is the whole of the explanation. Re-measured at the trained cadence the model was about **2.9x
  better** than persistence - MAE 0.0314 against 0.0897 on the evaluation split. The model retrained
  in Round 71, without the whole-series z-score described below, is about **3.1x better** over all
  441,600 test windows: MAE 0.0250 against 0.0764. The evaluation-sample generator hardcoded the same
  60s; it now reads the cadence from the model record, because preprocessing has to come from the
  artefact rather than a constant in a second file.
- **Retrained in Round 71: the published forecaster still carried a leak the trainer no longer
  had.** The trainer stopped fitting its z-score on the whole series in Round 34, test window
  included, and the model was never retrained, so the weights on the Hub and their `feature_spec.json`
  kept the whole-series constants. The retrained model flags 2 of 441,600 test points, as before,
  and neither falls inside one of the 3 scored events, so event-wise F0.5 is **0** (tp=0 fp=2 fn=3)
  and point-adjusted F1 falls from 0.9937 to 0 with it. That is the denominator problem below,
  measured once more, not a new property of the model.
- **CLOSED, and the explanation is the denominator.** The event-wise figure did not reproduce across
  identical-config retrains (0.7143, then 0.4545), and
  `references/satellite-event-count-investigation.md` explains why: the test split contains **3
  scored events**, so with one hit each extra false-alarm run walks the metric down a fixed ladder,
  0.714 / 0.455 / 0.333 / 0.263 / 0.217 / 0.185 for 0 to 5 false-alarm runs, and point-adjusted F1
  is inflated because it counts 441,600 points rather than 3 events. The current model flags 2 of
  441,600 test points and neither falls inside a scored event (tp=0 fp=2 fn=3), so both figures are
  0. Read the event-wise number as one draw from a wide distribution; the operative fact here is
  local and arithmetic.

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

  These are Round 18's figures. Round 71 rebuilt the finance and satellite samples from their
  held-out test windows - until then both had been drawn from the whole corpus, training rows
  included - and the run log prints the current figures with the split each sample comes from.

  WebNN is probed by **building and executing a 2x2 matmul** - three cheaper checks all lied - and
  then deliberately left unused: with the flag the API works, but ORT's WebNN provider still cannot
  complete a session for the medical ResNet (162.7 s and a timeout, against 5.1 s on wasm).

  Two lessons, both ones this project already had: a capability check that tests for an API surface
  rather than a working capability is the same defect class as a label naming a protocol the code
  does not run - and when three different models fail three different ways, suspect the thing they
  share.

- **`EvalBootstrap` runs on the documented box, and the test that proves it is not in
  `make test`.** The OpenFHE bootstrap test is marked slow, so `make test` deselects it with
  `-m "not slow"`; `scripts/ci_local.sh` runs it as its last step, and CI runs it on ubuntu-24.04 with
  `QSH_REQUIRE_OPENFHE_BOOTSTRAP=1`, so an unavailable OpenFHE fails rather than skips. A SIGSEGV
  inside `EvalBootstrapSetup` on AlmaLinux 9.7 came from the multiplicative depth on the
  bootstrapping path (4 against the 20 the setup needs), and a scaling technique `EvalBootstrap`
  rejects raised a separate error; both settings are fixed in `fhe/backends.py`, and three later runs
  on the same host exited 0 (both recorded in [`docs/verification-log.md`](verification-log.md)).
  The import guard there catches an `ImportError`, which a segfault is not, so the test runs the
  bootstrap in a subprocess and a native crash fails that test instead of killing the suite. No
  test count is quoted here: pytest reports
  it, and the quickstart notes in [`docs/development.md`](development.md) record why a hand-kept
  count drifts.
- **The retired security score had two paths, and they disagreed for a long time.** The classical row
  was scored without QKD key material while the quantum row assumed 256 bits, and separately the
  classical row passed no QBER while the quantum row passed a literal `0.0128` and a literal leakage
  bound. **Sixteen of the 28 points in the headline 68-vs-96 gap were that asymmetry, not
  cryptography.** Round 22 closed both halves in both languages and then retired the composite
  entirely rather than keep rendering it. The lesson generalises past this metric: when two rows are
  compared, verify they were scored under ONE transport configuration before believing the gap.
- **ct/pt is not a constant.** The measured ratio is `ciphertext_bytes / (n_in * 8)` and moves with the
  feature count - measured ~1393x at 30 features and ~5222x at 8. Read it as a property of a
  configuration, not of CKKS. The "~1483x" the client demo shows when it has no measured ciphertext
  is a representative constant (`cost_per_op.demo_ct_pt_ratio`), not a measurement.
- **The medical use-case pipeline and the medical AUC can use different datasets.** The AUC 0.9147 is
  BreastMNIST, and so is the benchmark's encrypted CNN (`benchmark.medical_cnn.medmnist_name` is
  `breastmnist`); the full deployment's medical use-case pipeline defaults to PneumoniaMNIST
  (`usecases/medical_fl`, unless the run config sets `medmnist_name`). Both are MedMNIST v2, but
  they are not the same images.
- **`results/` retention.** It had none for three rounds and grew 953 -> 1133 -> 1421 files;
  `docker/docker-compose.yml` bind-mounts it, so a long-lived deployment grew without bound.
  `orchestration.keep_runs` (default 200, 0 disables) now prunes on persist, grouped by run stem
  so a run is kept or removed whole. Measured after the change: 600 files, 200 runs.
- **Recommendation - generate the benchmark table instead of maintaining it by hand.** Six of the ten
  numbers in the benchmark section had drifted from what the documented command produces. `tests/` already
  asserts demo samples against the registry; the same pattern over the model and benchmark tables would
  have caught the AUROC-vs-C-index, 2022-vs-2023 and "every published QML-IDS number" errors before a
  reader did.

- **Simulation vs reality** - QKD distance/QBER gains, decoy-state, and the **free-space satellite
  QKD + QSDC** path are **simulated** (analytic channel + information-structure models), not
  hardware-validated; photonic inference and qOLE are not implemented at all. QHE evaluation is
  exact for Clifford circuits in the Python demo, with Pauli key updates; non-Clifford layers are
  costed as T-gates, not run fault-tolerantly, and the browser's comparison circuit applies its T
  gates to the padded state without correcting or decrypting it, so it measures cost and predicts
  nothing.
- **NSA and BSI stance** - NSA does **not** endorse QKD in any form for national security systems. Its
  CNSA 2.0 and Quantum Computing FAQ (Ver. 2.2, October 2026) answers *"Can I use a QKD system to protect
  my national security system from a quantum computer?"* with **"No"**, and its page on Quantum Key
  Distribution and Quantum Cryptography says NSA *"does not support the usage of QKD or QC to protect
  communications in National Security Systems"*. BSI does not recommend QKD either: TR-02102-1 v2026-01
  section 2.1 calls it *"currently not ready for use from a security point of view"*, and the hybrid it
  recommends (section 2.2) is a quantum-safe plus a classical key agreement - the ML-KEM + X25519 layer
  here. Offering QKD as an optional add-on to that hybrid is **this project's** design choice alone.
- **GPU/vendor & licensing** - Blackwell+CUDA13 narrows the quantum-GPU stack to CUDA-Q/cuQuantum.
  The FHE stack is deliberately **commercial-OK** (TenSEAL Apache-2.0, OpenFHE BSD-2, HEonGPU Apache-2.0);
  **DESILO was removed** because its license is non-commercial. Multiple swappable backends avoid lock-in.
- **Scope discipline** - the trained QCNNs ship at 7, 8 and 9 wires for reproducibility. The demo's
  **QSMPC-FHE (quantum) comparison row is a genuine in-browser state-vector simulation** of an untrained
  QOTP + Clifford+T circuit (8 qubits by default, not a trained QCNN): its latency is **measured** and its
  T-gate count is **counted from the circuit** - a *simulation*, not quantum hardware (no QPU). Its ct/pt
  is not rendered: a one-time pad maps n qubits to n qubits plus a 2n-bit classical key, so a qubit
  ratio cannot share a column with a CKKS byte ratio. The classical **MPC-FHE row is real measured
  crypto** (node-seal CKKS + Polytune 2PC) on the pages whose flow runs it.
- **In-browser AI (demo)** - the per-use-case Agentic-AI panel runs **small** models client-side (Transformers.js
  on onnxruntime-web; WebLLM on WebGPU). **Both the agent's Transformers.js models and the published ONNX
  models run on WASM by default, and on WebGPU only when the visitor selects it** (the Backend and Model
  backend controls), because on the machine measured the WebGPU path was the slow one by one to two
  orders of magnitude (see "A software WebGPU adapter is slower than no adapter" above). The model
  picker's sizes are computed from the bytes the config records for each backend's weights file, at a
  pinned commit: 236.7 MB of q8 weights for distilGPT-2 on WASM, 327.8 MB of fp32 on WebGPU. When
  WebGPU is selected, the ONNX path uses **the WebGPU build only where a real `requestAdapter()` returns a HARDWARE
  adapter**, and the WASM build otherwise. A returned adapter is not sufficient: a software
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
  (browser-cached after) - zero VPS bandwidth/compute. WebLLM needs WebGPU; without it WebLLM is
  disabled, Transformers.js runs on WASM as it does by default, and the quantum-safe MCP envelope still
  runs. The in-browser models
  illustrate the agentic-AI + MCP flow; they are not the production inference path.
- **MCP** - the **full deployment ships a real MCP server**: an **`MCPServer`** server (MCP Python SDK, MIT, `mcp>=1.2` in `requirements-ai.txt`; the class was named `FastMCP` before v2.0.0 and `server.py` resolves either)
  exposing the 5 per-industry tools, bound to the quantum-safe transport so a `tools/call` runs end-to-end
  under a PQC session (optionally mixed with a fixed 32-byte QKD-domain constant; the MCP path
  receives no QKD key) + ML-DSA + AEAD and returns the **signed real result** (`src/qsh/mcp/server.py`,
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
  return 404. Its FHE/MPC is **genuinely real and measured in the browser**: **node-seal CKKS**
  (a real homomorphic linear layer, ~1e-9 error) + a real **Polytune garbled-circuit 2PC** for the
  non-linear argmax, both parties loopback in one tab (zero VPS). In-browser federated learning runs
  on every run of the finance and medical pages, not behind a toggle: a real client-side FedAvg of
  a 4-weight logistic regression whose inputs are `qcnnExpectation`, a closed-form
  `tanh(mean(sin(x_i + 0.5 i)))`, so despite the "Quantum Federated Learning" label nothing quantum
  is computed in that stage; the full deployment runs the **PennyLane** QFL over this repo's own FedAvg + secure aggregation (whose default pair keys are a public constant), plus server-side FHE inference (incl.
  the medical CNN under real encrypted convolution). In the browser, node-seal CKKS evaluates the
  published medical student's convolution and both dense layers, 152 counted rotations, and the
  client decrypts between layers to apply ReLU and average pooling in plaintext.
- **Per-industry models - three tiers, and the gap between them is the measurement** - the
  **classical crypto/protocol layer is real wherever a use case's flow runs it** (PSI/QOT, QSMS
  secure-sum, MPyC MPC, CKKS FHE, federated secure-aggregation; the satellite page runs none of these
  in the browser, and the PSI and the secure aggregation keep nothing private between the parties,
  as above); BB84 and QSDC are simulated (analytic channel
  models), as "Simulation vs reality" under Concerns & recommendations says. The AI side is organised in tiers because
  homomorphic evaluation is cost-bounded **at the operating point this demo targets** - a single
  device, in a browser tab, running CKKS through `node-seal` (SEAL compiled to WASM, single-threaded).
  Park et al. ([arXiv:2601.18511v2](https://arxiv.org/abs/2601.18511v2)) run Llama-3-8B end to end
  under CKKS, 128 encrypted tokens in 20 s - on **eight** NVIDIA RTX PRO 6000 GPUs, which by the
  per-operation ratios this project cites (GPU against single-thread SEAL, browser against native
  SEAL) is roughly three to four orders of magnitude more compute than one tab, and the keys and
  ciphertexts would also have to fit the tab's WASM memory. So the constraint is real and it is
  about the *device*, not about the field: a demo cannot honestly claim both a state-of-the-art
  model *and* encrypted inference of that model **here**:

  | tier | what it is | where it runs |
  |---|---|---|
  | **T1 crypto** | the FHE/MPC-feasible student | real CKKS + MPyC server-side; node-seal CKKS + Polytune 2PC in-browser |
  | **T2 industry** | the real literature model, trained here, published to the Hub | browser (**int8 ONNX where a quantised graph exists**; the two tree-ensemble models serve **fp32**, because their parameters live in node attributes and dynamic quantisation has nothing to quantise) and server (fp32 ONNX) |
  | **T3 large** | models too big to encrypt at all | WebGPU in-browser, or a hosted endpoint |

  The headline numbers are **two agreements, one measured per run and one offline**. Per run, the
  browser reports `encrypted_agreement`: how often the CKKS evaluation reproduces the PLAINTEXT
  STUDENT's top-1 class over the rows it scored, which measures the encryption's noise (Concrete-ML
  calls the same quantity "Similarity"). Offline, from training, `models/metrics.json` records the
  student's `agreement` with its TEACHER on held-out data: finance 0.983, medical 0.827, OPS-SAT
  0.958, satellite 0.976, telecom 0.995, which measures what choosing an encryption-sized model
  costs, beside the industry's own metric. Everything below is what this repository measured, not
  what the papers report - published baselines are shown beside each figure, including where we
  fall short:

  | vertical | T2 model | data | measured | published baseline |
  |---|---|---|---|---|
  | **finance** | XGBoost over this project's own graph features (fan-in/fan-out and causal per-account aggregates over the transaction multigraph; IBM's Graph Feature Preprocessor is not used) | IBM **AMLworld HI-Small**, 5,078,345 transactions, 1 laundering in 981 (CDLA-Sharing-1.0) | minority-class F1 **0.1519**, AUPRC **0.1233** vs a 0.0015 prevalence baseline | 0.632 (NeurIPS 2023 D&B) - see the gap note below |
  | **medical** | **ResNet-18** trained from random initialisation (no ImageNet weights, so no ImageNet terms attach) | MedMNIST v2 **BreastMNIST** (CC BY 4.0 **asserted by MedMNIST**; the BUSI authors **publish no licence** - see [DATA_LICENSES.md](../DATA_LICENSES.md)) | AUC **0.9147** | 0.901 (Yang et al., Scientific Data 10:41, 2023) |
  | **genomics** | published **PRS** effect-weight vector - not a neural network | PGS Catalog **PGS000004 (PRS313)** over **2,504 real 1000 Genomes samples** | ancestry portability, below | AUROC 0.633 [0.624, 0.641] - *external citation, not measurable here: a reference panel carries no phenotype. Fitted WITH covariates a dot product does not have, and 8% of variants were missing in that cohort, so it is a citation and not a target* |
  | **satellite** | **Telemanom** 2xLSTM(80), seq 250, non-parametric dynamic threshold + pruning | **ESA-ADB Mission1** lightweight channels 41-46, multi-year real telemetry (CC-BY-3.0-IGO) | event-wise F0.5 **0** over just **3 events** (tp=0 fp=2 fn=3) | 0.968 (ESA-ADB Supp. Table 9, anomalies only) |
  | **telecom** | RF baseline + quantum-kernel SVC + variational QNN | literature-grounded synthetic QKD-attack telemetry (7 attack types + normal) | macro F1 - **RF 0.987**, quantum-kernel SVC **0.832**, variational QNN **0.467** | no trustworthy single number exists - see below |

  Three of those deserve their honest caveat stated here rather than buried:

  - **finance falls well short of 0.632, and the reason recorded here for several rounds was wrong.**
    The note used to give two causes: the Proprietary `snapml` licence on IBM's Graph Feature
    Preprocessor, and 10-hop injected CYCLE patterns against 4-hop reachability here. The second does
    not survive arithmetic - the GFP paper (arXiv:2402.08593) decomposes its own contribution, and in
    its Table 5 (absolute F1 points, XGBoost on HI-Small at batch size 128) multi-hop cycle and
    scatter-gather features add **1.7 points** (the paper says "up to 4%"), while fan-in/fan-out,
    which this project already reimplements, takes F1 from 19.75 to 56.88, **about 37 points**. A
    few points cannot explain a multiple-fold gap. The real cause is architectural rather than licence-related, and the mechanism
    is derived once in `references/finance-gap-investigation.md` rather than restated here. It is
    implementable without `snapml`; it is simply a larger piece of work. The
    licence exclusion is real but was never the mechanism. Investigating this also found and fixed a
    genuine defect - per-account aggregates spanned the whole training window, so a training row's
    feature included its own future while test rows got a stale table. Making them causal
    (prior-at-time) lifted minority-class F1 **0.0808 to 0.1519, +88%**, with leakage disproved by a
    truncation test rather than argued. Workings:
    [`references/finance-gap-investigation.md`](../references/finance-gap-investigation.md).
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
  - **satellite reports event-wise F-beta at $`\beta = 0.5`$, never point-adjusted F1.** On the run before
    the Round 71 retrain the point-adjusted figure was **0.994** against an honest **0.4545**, and on
    the retrain both are 0 because neither of its 2 flags falls inside a scored event - under point
    adjustment "even a
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
    F1 0.987; neither quantum model beats it. Two more facts travel with these figures: the published
    `telecom_ids.qml` ONNX graph, which the telecom page runs and verifies, is that random forest
    (the QSVC and the QNN run only offline, in the training script), and the QSVC's "fidelity
    quantum kernel" equals the classical product kernel $`\prod_i \cos^2((x_i - x'_i)/2)`$, so its
    0.832 is a classical kernel's score.

  **Quantum-ML IDS is presented as an architecture showcase, not a performance claim.** On this
  project's own synthetic telemetry neither quantum model beats the classical random forest. Published
  results, however, are **mixed rather than uniformly below a random forest** - an earlier revision of
  this paragraph claimed they were, and a paper this project cites in
  [`references/REFERENCES.md`](../references/REFERENCES.md) contradicts it:
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

- **Public-demo data (real or simulator output, and one sample with no licence)** - the client-side demo loads
  real or simulated datasets, mirroring the full deployment; every one but the satellite sample carries a licence that permits redistribution: **medical** uses a real Wisconsin
  Diagnostic Breast-Cancer sample (UCI dataset 17 via sklearn, **CC BY 4.0 - attribution required**) + a real logistic-regression model;
  **genomics** uses the real PGS-Catalog PGS000004 / PRS313 PRS weights (313 variants; **licensed per score** - PGS000004 carries the Catalog's default terms, *not* a blanket CC-BY grant) over real 1000 Genomes
  genotypes -> a plaintext in-browser **Polygenic Risk Score** over all 313 variants with per-SNP signed SHAP, while its CKKS step is a sign test on the 8 largest-weight variants (weights $`[-w/2, w/2]`$, bias 0); **finance** uses a
  **PaySim** simulator sample (synthetic transactions, CC-BY-SA-4.0) + a linear surrogate over its 8 features, which is what the finance page's CKKS step evaluates (the 33-parameter AMLworld student is re-scored only in plaintext ONNX); and **satellite** scores
  **real NASA SMAP/MSL telemetry** (no licence stated by any provider - see DATA_LICENSES.md) with a plaintext linear anomaly score - all tiny static JSON
  under `web/public/data/`, loaded by [`web/src/lib/realData.ts`](../web/src/lib/realData.ts). The finance, medical and genomics pages compute with the
  real in-browser node-seal CKKS + a Polytune garbled-circuit 2PC (notional parties); the satellite page runs neither. Only **telecom** has no commercial-OK real attack dataset
  - the narrower and correct statement, since **5G-NIDD is CC BY 4.0 and commercially usable**, and this repository trains on it. What does not exist is a commercial-OK corpus of QKD-LAYER attacks, which is a different problem needing a physical QKD testbed under attack - so the QKD IDS uses representative synthetic features (logged as such in the run).
  The bundled sets are public / de-identified, so no real *personal* data leaves the browser - the privacy property
  is demonstrated on real or realistic inputs.
- **Config single-source** - the benchmark cost model, tier architectures, workload multipliers,
  security/advantage weights and demo data sizes live in one place (`src/qsh/config/defaults.py`),
  served at `/api/config` and baked into `web/src/lib/config.defaults.json` (regenerate via
  `python scripts/export_config.py`), so the demo's numbers cannot drift from the backend.
- **Link templates** - each satellite downlink and inter-satellite terminal pair is one JSON file
  under `src/qsh/config/profiles/`, every value with the source's words or a stated assumption
  (the loader refuses a value with neither). A new file appears in the page's profile selector
  after `python scripts/export_config.py`, with no code change. Shipped: Micius to Xinglong (Liao
  2017) with the atmosphere at the midpoint or the upper end of the paper's 3-8 dB, without its
  vacuum state, and with an optional turbulence coupling at an assumed field stop; Micius to
  Zvenigorod (Khmelev et al. 2023); the Jinan-1 microsatellite (Li et al. 2025), with its one
  calibrated constant named in its file; Micius-type and CubeSat (Naughton et al. 2019)
  inter-satellite terminals. Published measurements the models are checked against live beside
  them, under `src/qsh/config/validation/`.
- **Future tracks** - see [`docs/pqc-roadmap.md`](pqc-roadmap.md): a NIST-aligned IDS
  against HNDL/Q-Day, hardware QKD field test, malicious-secure MPC, and the GPU FHE-at-scale
  path (HEonGPU + PIGEON/hpmpc).
