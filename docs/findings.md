# Findings, concerns and recommendations

## WS3 + WS4 challenges addressed

The four workstreams target specific open problems in quantum-safe + private AI; here is how this
implementation maps to each, and where gaps remain.

**WS3 - QSMPCxQKD** (cross-org AI restricted by GDPR/HIPAA; classical SMPC is complex, HNDL-vulnerable, sometimes assumes a TTP; Agentic-AI/MCP lacks quantum-safe + private agent comms):

| Sub-problem | How QSH addresses it | Where |
|---|---|---|
| Raw data must not be shared | Federated learning keeps raw rows with each client, the OPRF PSI shows Bank A only the shared accounts and the size of Bank B's list and Bank B only the size of Bank A's (semi-honest, not post-quantum), and FHE private inference evaluates the encrypted input; the masked updates are not hidden between the parties themselves ([what each step protects](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment)) | `ai/federated.py`, `qsmpc/psi.py`, `fhe/` |
| Classical SMPC protocol complexity | QSMS = Shamir + d-level QFT (no Beaver triples / garbled circuits) | `qsmpc/secure_summation.py` |
| HNDL-vulnerable HE / key exchange | PQC-first hybrid **ML-KEM + X25519** (+ simulated QKD key material in the full deployment only; the browser mixes none into any key) | `pqc/hybrid_kem.py`, `pqc/transport.py` |
| TTP assumption | No dealer in QSMS (entangled state), no third party in the OPRF PSI, which runs between the two banks over the PQC channel, and no dealer in pairwise-mask secure aggregation, whose mask seeds each pair of clients agrees by X25519 ([what each step protects](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment)) | `qsmpc/`, `ai/federated.py` |
| **Agentic-AI / MCP quantum-safe + private agent comms** | A **real MCP server** (MCP Python SDK, `mcp>=2.2.0`; the class is `MCPServer`, and `server.py` still resolves the 1.x `FastMCP`) (5 per-industry tools, full deployment) bound to a quantum-safe channel: two agents establish a PQC session, optionally mixed with a fixed 32-byte QKD-domain constant (no QKD key reaches this channel), the caller sends an **ML-DSA-signed, AES-GCM-sealed** `tools/call`, the callee **dispatches the real tool** and returns the **signed result** (authenticated, confidential, HNDL-resistant, no TTP); also interops with the official filesystem MCP server | `pqc/mcp_bridge.py`, `mcp/server.py`, `/api/mcp/*` |
| **Explainability / audit trail (XAI)** | Exact linear-SHAP feature attributions for each run's classification, **ML-DSA-signed** under a per-run key, so the record shows it is unchanged since signing and not who signed it: anyone can edit a copy and sign again ([limits](threat-model.md)); the attributions serve explanation duties such as (EU AI Act Art. 13, for high-risk systems only - 2 Dec 2027 for Annex III and 2 Aug 2028 for Annex I after the AI Omnibus, not Aug 2026; medical and genomics would be Annex I products; finance AML is not high-risk for a bank but would be for law-enforcement profiling (Annex III 6(e)); the telecom IDS is not as a cybersecurity tool and is outside the Act if used exclusively for military, defence or national security purposes (Art. 2(3)); satellite would be only if its service is itself critical digital infrastructure and the detector a safety component, since space is a separate sector under the CER and NIS2 directives; GDPR Art. 22; SR 11-7); surfaced on all five use-case pages. On telecom and satellite, whose flows have no `private_inference` stage, the attribution is of the model that produced the reported class; on the other three it explains the bundled model beside the encrypted one, and the panel says which case applies | `ai/explainability.py`, `ExplainabilityStage` |

**WS4 - QHExAI** (FHE $`10^3`$ to $`10^6`$x overhead; bootstrapping bottleneck; no quantum encoding; plus unsolved (a)-(f)):

| Sub-problem | How QSH addresses it | Where |
|---|---|---|
| (1)/(e) huge compute + ciphertext/plaintext overhead | The benchmark measures the CKKS ciphertext/plaintext byte ratio of this configuration, which tracks slot utilisation rather than CKKS itself, and renders no quantum ratio beside it, because none can share that column ([what ct/pt measures](models-and-benchmarks.md#what-ctpt-measures)) | `benchmark/`, `qhe/rate1_qfhe.py` |
| (2) bootstrapping bottleneck | **PARTLY met, on the CPU only.** OpenFHE CKKS `EvalBootstrap` refreshes ciphertext levels: at ring dimension 65536 one bootstrap takes 6.2 to 7.7 s and returns a level-22 ciphertext at level 19, on OpenFHE 1.6.1 (four runs on 2026-11-06; 7.7 to 9.1 s on the 1.5 series, `docs/verification-log.md`). HEonGPU's bootstrapping is **not run here**: since Round 86 `HeongpuBackend` runs the hybrid's linear half on the GPU, and its binding exposes the seven operations that needs and no bootstrap. Both libraries are commercial-OK | `fhe/backends.py` (`OpenFHEBackend(use_bootstrap=True)`, `HeongpuBackend`) |
| (3)/(a) quantum encoding / practical QHE | QOTP + a variational QCNN + a Rate-1 quantum transport (the module has no homomorphic `Eval`) at small scale. **Three different circuits, and only one is Clifford+T:** the BROWSER's QSMPC-FHE comparison circuit (`qotpQcnnInfer` in `web/src/lib/quantumSim.ts`) is untrained and built from H, CNOT and T on a padded state, counts 7n real T gates (56 at 8 qubits), and predicts nothing; the TRAINED QCNN that decides the class on the finance, medical and telecom pages when Inference is set to QCNN (`qcnnInferTrained`, RY, RZ, CNOT and CRY) has **no T gate and no mask**, and it is classically simulable at its 4 to 8 wires ([what the trained QCNNs are](use-cases.md#what-the-trained-qcnns-are)); and the PYTHON module (`qhe/qcnn.py`) contains **no T gate at all** - it is `AngleEmbedding` plus CNOT/RY/RZ, and its `t_count()` is a labelled Solovay-Kitaev estimate of what a fault-tolerant deployment would cost, not a count of gates present. | `qhe/qotp.py`, `qhe/qcnn.py`, `web/src/lib/quantumSim.ts` |
| (b) QFL gradient leakage (arXiv:2508.15998) | **NOT met.** Pairwise-mask secure aggregation runs with each pair's mask seed agreed by X25519, so an honest-but-curious aggregator cannot remove a mask, but there is no dropout recovery, no malicious security and no authentication of the public keys, and every client runs in one tab or one process ([what each step protects](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment)); even a complete protocol would not bound what the aggregate leaks, and no differential-privacy budget is applied. QOTP-delegated learning is not implemented | `ai/federated.py`, `web/src/lib/clientQFL.ts` |
| (c) input privacy under Clifford evaluation, and correct decryption | **Input privacy holds by the QOTP theorem, and correctness is tested; nothing checks a deviating server.** A random Clifford word over {H, S, CNOT} is evaluated on the QOTP-encrypted state; the client tracks the Pauli frame through the same word and recovers the plaintext result at fidelity 1.0, tested at depths 8, 12, 20 and 30, one seed each. The server sees the maximally-mixed state throughout, so both the input and the computation's operand are hidden from it. Arbitrary Clifford words need the S frame rule (`H` and `CNOT` alone reach only a subgroup) and a CNOT applier not hardcoded to wires 0 and 1, and both are implemented. **Does NOT extend to T:** `T X T-dagger` is not Pauli, so the frame does not propagate, which is why the QCNN's non-linear layers are costed as magic-state T-gates instead. | `qhe/qcnn.py` (`qotp_clifford_random_circuit_demo`), `qhe/qotp.py` (`update_key_h/s/cnot`) |
| (f) hiding the server's circuit from a malicious client | **NOT met, and in tension with the mechanism rather than merely unimplemented.** The client can only decrypt by applying the circuit's action to its own Pauli frame, so under plain QOTP it must know the circuit - "the client updates its frame" and "the client does not learn the circuit" are close to mutually exclusive. Earning this needs a construction where that does not hold, which is what Ortega and Martin-Delgado's scheme addresses and this repository does not implement. Split from (c) so the one that is met is not held hostage to the one that is not. Cf. QOTPH (arXiv:2604.19256), which builds universal QHE from QOTP over Clifford+T with Pauli key updates and reports results on real quantum processors. | not implemented |
| (d) efficiency on AI/HPC infra | **PARTLY met, on one GPU and for the linear half.** Since Round 86 HEonGPU runs the hybrid's linear half on the GPU through this project's binding, 19.5 times faster than TenSEAL on the CPU at both measured sizes, and the whole medium inference 5.7 times ([measured](models-and-benchmarks.md#ckks-on-each-backend-measured)). The MPC half is still MPyC on the CPU and is now three quarters of a GPU row's latency; nothing runs batched, bootstrapping runs only in OpenFHE on the CPU, the browser and the API image have no GPU path, and CROSS is not in this repository at all | `fhe/`, `benchmark/`, `native/heongpu_py/` |

**Residual gaps, each with the condition that would close it.** Two of the four are downstream of
decisions already recorded in [`docs/deliverable-verification.md`](deliverable-verification.md), so
they are one item each rather than independent gaps:

| residual | condition that closes it |
|---|---|
| **PQ-OPRF PSI + identifiable abort** | Name the browser-loadable library that supplies each, at the point where it is claimed. The non-linear 2PC layer **already runs a protocol designed for malicious security** - Polytune implements WRK17b multi-party authenticated garbling (Wang-Ranellucci-Katz, CCS 2017), secure against malicious adversaries controlling all but one party. Whether this pre-release implementation achieves that between hosts is open: the Round 68 and Round 72 reviews in `docs/threat-model.md` record a gap in the proof of its OT extension's check (upstream #63) and a check whose challenge is fixed earlier than the protocol specifies, so nothing here claims malicious security between hosts - so what is missing is these two primitives specifically: a post-quantum, malicious-secure OPRF for the PSI, which today is a semi-honest OPRF over ristretto255 ([what each step protects](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment)), and **identifiable** abort, which authenticated garbling does not provide. |
| **Hardware QKD field test** | A boundary of the deliverable, not a gap in it: no QKD device exists here and there is no browser API for one. If a characterised link ever exists, the first thing published is the **measured QBER against `qkd/decoy_reference.py`'s prediction for the same link budget** - and that is also the round in which the 2026 finite-key proofs (arXiv:2606.29943, arXiv:2608.09674) replace Ma et al. (2005), because it is the round their device inputs exist. |
| **Larger fault-tolerant QHE** | `t_count()` is a labelled Solovay-Kitaev **estimate** of a fault-tolerant deployment's cost. Replace it with an executed count when a backend exposing logical, error-corrected qubits is addressable from PennyLane. |
| **Production bootstrapping throughput** | OpenFHE's `EvalBootstrap` runs here (one bootstrap, 7.7 to 9.1 s at ring dimension 65536), but that is a single-ciphertext timing, not a throughput. Publish a throughput figure when a batched bootstrapping benchmark runs end to end, on the CPU path or on HEonGPU once it has a Python binding. |

(A real quantum-safe MCP server + filesystem-MCP interop now ships in the full deployment.)
See [`docs/pqc-roadmap.md`](pqc-roadmap.md).

## Per-industry models: three tiers

The **classical crypto/protocol layer is real wherever a use case's flow runs it** (PSI, QSMS
secure-sum, MPyC MPC, CKKS FHE, federated secure-aggregation), though the satellite page runs none of
these in the browser ([which pages encrypt](models-and-benchmarks.md#which-pages-encrypt)), the PSI
is semi-honest only, and the secure aggregation keeps nothing private between the parties, which all
run in one tab or one process
([what each step protects](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment)).
BB84 and QSDC are simulated (analytic channel models), as "Simulation vs reality" under
[Concerns and recommendations](#concerns--recommendations) says. The AI side is organised in tiers
because homomorphic evaluation is cost-bounded **at the operating point this demo targets** - a single
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
| **finance** | XGBoost over this project's own graph features (fan-in/fan-out and causal per-account aggregates over the transaction multigraph; IBM's Graph Feature Preprocessor is not used) | IBM **AMLworld HI-Small**, 5,078,345 transactions, 1 laundering in 981 (CDLA-Sharing-1.0) | minority-class F1 **0.1519**, AUPRC **0.1233** vs a 0.0015 prevalence baseline (the test window's positive rate; the corpus's is 1 in 981) | 0.632 (NeurIPS 2023 D&B) - see the gap note below |
| **medical** | **ResNet-18** trained from random initialisation (no ImageNet weights, so no ImageNet terms attach) | MedMNIST v2 **BreastMNIST** (CC BY 4.0 **asserted by MedMNIST**; the BUSI authors **publish no licence** - see [DATA_LICENSES.md](../DATA_LICENSES.md)) | AUC **0.9147** | 0.901 (Yang et al., Scientific Data 10:41, 2023) |
| **genomics** | published **PRS** effect-weight vector - not a neural network | PGS Catalog **PGS000072** (Graff et al. 2021, CC BY 4.0; 183 of its 187 variants scored) over **2,504 real 1000 Genomes samples** | ancestry portability, below | AUROC 0.637 - *external citation, not measurable here: a reference panel carries no phenotype. Fitted WITH covariates a dot product does not have, on 162 of the 187 variants and at 5 years, so it is a citation and not a target* |
| **satellite** | **Telemanom** 2xLSTM(80), seq 250, non-parametric dynamic threshold + pruning | **ESA-ADB Mission1** lightweight channels 41-46, multi-year real telemetry (CC-BY-3.0-IGO) | event-wise F0.5 **0** over just **3 events** (tp=0 fp=2 fn=3) | 0.968 (ESA-ADB Supp. Table 9, anomalies only) |
| **telecom** | RF baseline + quantum-kernel SVC + variational QNN | literature-grounded synthetic QKD-attack telemetry (7 attack types + normal) | macro F1 - **RF 0.987**, quantum-kernel SVC **0.832**, variational QNN **0.467** | no trustworthy single number exists - see below |

Four of those deserve their caveat stated here rather than buried.

### Finance: the gap to the published baseline is architectural

The XGBoost model scores minority-class F1 0.1519 against the published 0.632, and the cause is the
feature architecture, not the licence and not multi-hop reach. IBM's Graph Feature Preprocessor
computes per-transaction subgraph patterns over a dynamically updated graph, where this project
computes per-account scalar aggregates: a degree is one number per account, while the reference
feature is a pattern match in that transaction's own neighbourhood at the moment it arrives. The
mechanism is derived in
[`references/finance-gap-investigation.md`](../references/finance-gap-investigation.md), which also
rules out the decision threshold and accounts unseen in training. It is implementable without
`snapml`; it is simply a larger piece of work.

Two causes recorded here earlier do not hold. The Proprietary `snapml` licence on the preprocessor
is real, but it was never the mechanism. And 10-hop injected CYCLE patterns against 4-hop
reachability here does not survive arithmetic: the GFP paper (arXiv:2402.08593) decomposes its own
contribution, and in its Table 5 (absolute F1 points, XGBoost on HI-Small at batch size 128)
fan-in/fan-out, which this project already computes as per-account aggregates, takes F1 from 19.75
to 56.88, **about 37 points**, while multi-hop cycle and scatter-gather features add **1.7 points**
more, to 58.60 (the paper says "up to 4%"). A few points cannot explain a multiple-fold gap.

Investigating this also found and fixed a genuine defect: per-account aggregates spanned the whole
training window, so a training row's feature included its own future while test rows got a stale
table. A matched-harness experiment predicted that causal (prior-at-time) aggregation would beat the
shipped approach by 1.45x AUPRC at matched feature count; applied to the real trainer it lifted
minority-class F1 **0.0808 to 0.1519, +88%**. Leakage was disproved by a truncation test rather than
argued: every feature is bit-identical whether or not later rows exist.

### Genomics: no AUROC, and the portability gap instead

Genomics cannot produce an AUROC here, so it does not print one. 1000 Genomes is a reference
panel with no case/control status; the published AUROC comes from a phenotyped UK Biobank
cohort (PPM002038, n=216,273). What *is* computable from real genotypes is the portability failure,
and it is stark:
**a single global 90th-percentile "high risk" threshold flags 39.1% of East Asian-ancestry samples
against 4.0% of European and 1.7% to 2.9% of the other three super-populations - driven entirely
by ancestry, with no phenotype involved.** That is the clinically important finding, not a
footnote. The figures are recomputed on every training run, so read
`by_super_population.*.frac_above_global_cut` in `models/metrics.json` rather than this
sentence.

### Satellite: scored event-wise, never point-adjusted

Satellite reports event-wise F-beta at $`\beta = 0.5`$, never point-adjusted F1. On the run before
the Round 71 retrain the point-adjusted figure was **0.994** against an honest **0.4545**, and on
the retrain both are 0 because neither of its 2 flags falls inside a scored event - under point
adjustment "even a
random anomaly score can easily turn into a state-of-the-art TAD method" (Kim et al., AAAI 2022,
arXiv:2109.05257). This sentence cited TSB-AD for "a uniform random score ranks 26th of 32
detectors" until Round 33, a figure that could not be verified at source. A windowed isolation
forest runs as a trivial baseline every time (0.0002 here), because a deep model that cannot beat
it has not earned its complexity. Only labels whose ESA-ADB Category is `Anomaly` count as
positives; the 63 annotated rare *nominal* events on these six channels do not (78 in all of
Mission1; this said 690 until Round 86, which neither of ESA-ADB's label files supports).

### Telecom: the classical forest wins

**Telecom reproduces the metric trap in-house.** The variational QNN scores **accuracy 0.875 but
macro F1 0.467**, and both are the constant predictor's: the QNN is a **binary** attack-vs-normal
classifier (`qml_ids/detector.py` fits on `binary_labels(y)` and predicts `(preds > 0)`), it predicts
"attack" for every row, and on its split, where 7 of 8 test rows are attacks, that gives accuracy
$`7/8`$, attack F1 $`14/15`$ and macro F1 $`7/15`$, since the normal class scores zero. Its macro F1
averages two classes, not eight. The kernel SVC is binary too: its 0.832 is the mean of its attack F1
0.945 and its normal F1 0.718. Until Round 86 this paragraph, the registry and the training script set
both against the random forest's 8-class macro F1 0.987 on a different draw, and the earlier wording
explained the low score by a floor of zero on six classes. Neither was right.

The like-for-like comparison is the forest trained and scored on each quantum model's own split and
binary labels (`scripts/measure_telecom_quantum_like_for_like.py`, re-fitting the SVC to its recorded
figures first): **0.990** on the QNN's 240 test rows and **1.000** on the SVC's 120. Neither quantum
model beats it there either, and the registry records `quantum_beats_classical: false` on that basis.
An aggregate can look respectable while the per-class picture is not, which is why per-class F1 is the
headline everywhere here.
Two more facts travel with these figures: the published `telecom_ids.qml` ONNX graph, which the
telecom page runs and verifies, is that random forest, of 200 trees (the QSVC and the QNN run only
offline, in the training script), and the QSVC's "fidelity quantum kernel" equals the classical
product kernel $`\prod_i \cos^2((x_i - x'_i)/2)`$, so its 0.832 is a classical kernel's score.

**Quantum-ML IDS is presented as an architecture showcase, not a performance claim.** On this
project's own synthetic telemetry neither quantum model beats the classical random forest. Published
results, however, are **mixed rather than uniformly below a random forest** - an earlier revision of
this paragraph claimed they were, and a paper this project cites in
[`references/REFERENCES.md`](../references/REFERENCES.md) contradicts it:
Al-Kuwari et al. (arXiv:2509.14282) Table VIII reports a hybrid QLSTM at 94.7% accuracy / 94.7% F1
against their own random-forest baseline on the same generated data. What remains well-founded is
caution about evaluation quality, not a universal ranking: arXiv:2512.05069 was withdrawn by its own
authors for evaluation "insufficient to fully support the paper's conclusions".

---

## Concerns & recommendations

- **Planned, not started: CKKS-friendly students (the owner's decision, 2026-11-02).** The Round 86
  review found the encrypted tier evaluating students that are weak or fed out of domain, and the
  owner chose to plan a replacement as research rather than retrain now. The facts the plan starts from:
  - `finance_aml.student` (33 parameters) scores minority-class F1 0.0614 against its teacher's 0.1519
    over the same test window. A quadratic feature map costs no multiplicative depth, only slots, and
    it pays on OPS-SAT, whose 210-parameter student tracks its teacher (AUC-PR 0.968055 against
    0.968107); on finance it was measured and rejected, raising fidelity and lowering F1 (the
    trainer's `--student-degree` help records 0.0185 to 0.0115, on an earlier student). It also costs
    the browser: 560 features take a 1,024-slot block, 4 rows a ciphertext where 32 features fit 128.
  - `medical_fl.student` decrypts after every layer for its ReLU and pooling, and its encrypted input is
    a tabular row tiled to fill the image (review PI-10, now stated on the page). A student that runs
    whole under CKKS needs a polynomial activation, and more rescales than the 60+40+40+60 chain's two;
    a longer chain needs $`N = 16384`$, which Round 38 measured at about twice the cost of every FHE
    figure, so the plan must show what the depth buys before it pays for it.
  - The plan's rule: each candidate is measured against the shipped student on the same rows, with the
    decision fixed before the run, and is published only if it wins; a published one needs a
    republish, a new pin and a regenerated held-out sample. Until then the shipped students stay.

- **The orchestration now runs the published models; before Round 21 it never had.** The pipeline
  truncated its features to 8 and called `inferLogits`, which builds a `[1,8]` tensor. No published
  model takes `[1,8]` - they take `[1,32]`, `[1,1,28,28]`, `[1,250,6]`, `[1,19]`, `[1,7]` - so the
  ONNX tier failed on every use case, every run silently dropped to the bundled 8-feature linear
  model, and the badge above it read MEASURED. The Models panel forty lines below advertised a
  ResNet-18 the page never ran. `runPublishedModel()` is now shared by the panel and the pipeline,
  and a **Published model** selector picks which one on the three use cases that publish several.
  Measured in-browser: `finance_aml.gbm` 1.4 ms `1x32`, `medical_fl.resnet18` 4.7 ms `1x1x28x28`,
  `telecom_ids.qml` 5.9 ms `1x7` (its published ONNX graph is a random forest, not the quantum
  models its name suggests: [telecom](#telecom-the-classical-forest-wins)), `satellite_comms.telemanom`
  9.6 ms `1x250x6`.

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

- **A software WebGPU adapter is slower than no adapter.** Same model (ResNet-50), same row, two
  browsers: SwiftShader gave `onnx-webgpu` at 12842 ms, no adapter gave `onnx-wasm` at 42 ms.
  `requestAdapter()` succeeds on a software rasteriser, so "is WebGPU available" answered yes and the
  code took the build that was 300x slower than the one it had just declined: every shader dispatch
  is emulated on the CPU while wasm runs SIMD kernels natively. A software rasteriser (SwiftShader,
  lavapipe, llvmpipe, WARP) is now declined by name, which picks the faster of two CPU paths.
  **A hardware adapter can be just as slow.** On an NVIDIA RTX PRO 6000 under Chrome 152 on Linux,
  one ResNet-18 `session.run` took 18216 ms on `onnx-webgpu`, and distilGPT-2 ran at 750 to 1140 ms
  per token on WebGPU against about 9 ms on WASM. So both the agent's Transformers.js models and the
  published ONNX models run on WASM by default, and on WebGPU only when the visitor selects it; other
  GPUs and browsers were not measured.

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
  patches packed into a single ciphertext) and both dense layers (each layer's rows packed into one
  ciphertext) run under CKKS, each layer's bias added under encryption and every slot but its outputs
  masked and flooded; between layers the client decrypts the outputs and applies ReLU and average
  pooling in plaintext, and the CNN's argmax is taken in the clear (the garbled-circuit argmax on the
  medical page runs on the WDBC logistic regression, not on this CNN, over logits the client has
  already decrypted). The run reports **28 counted rotations** - the quantity the 2026 literature
  identifies as the cost driver - rather than a latency multiplier standing in for it, and says when
  the published weights could not be fetched, in which case the encrypted student does not run and
  no weights are invented.
  The **Workload selector is gone**: it offered Linear against CNN, and once CNN ran the real
  student, Linear was the only fabricated option left - an 8-feature dot product with a hand-written
  weight pattern matching no trained model. For finance, satellite and OPS-SAT the students genuinely
  ARE linear, which is why the selector only ever appeared on the medical page.
- **Every model in the Models panel carries a Run button, or a specific reason it has none.** Six can
  be verified in place; the rest are in three genuinely different states that are never collapsed
  into one message - never trained (the finance GNN, the telecom Transformer), served with the page
  rather than from the Hub (the PRS, whose weights are PGS000072's own, CC BY 4.0), or published
  without a held-out sample.
- **`finance_aml.gnn` is a decision, not a backlog item, and the reason is where the graph lives, not
  model strength.** On AMLworld HI-Small the published GNNs are the stronger models: the registry
  records GFP + XGBoost at minority-class F1 63.23 (Altman et al., NeurIPS 2023 D&B) against
  Multi-PNA+EU at 68.16 (AAAI 2024) and, in MEGA's published table (TMLR 08/2026, arXiv:2412.00241v3, Table 16), MEGA-PNA at 74.01 and Multi-FraudGT at 75.81, and the
  XGBoost shipped here, without IBM's preprocessor, scores 15.19. The decision rests on the operating
  point, as the registry states it: message passing needs each account's graph neighbourhood at
  scoring time, which this demo's browser tier does not hold, so every tier ships the model that
  scores one precomputed feature row per transaction. It is not an impossibility under FHE: Bilgi et
  al. (arXiv:2610.18054v2, preprint) run an edge-centric GNN under CKKS on the IBM AML benchmark at
  as little as 5.31 s per queried transaction with HEonGPU on one RTX 5090, the graph held in plaintext by
  the server, and DESIGN (arXiv:2507.05649) and TGHE (arXiv:2606.26664) run GNNs under FHE too. The
  2024 comparison of
  arXiv:2411.02926v2, where the GNN pipeline never compiled in Concrete ML while XGBoost ran at an
  overhead of roughly 15,000x to 166,000x, is dated and no longer the reason.

- **The models sized for encryption are published too.** Each use case has a teacher and a
  distilled student sized for CKKS: 33 parameters for finance against a 141,154-parameter teacher,
  3,226 against 11,171,266 for medical, 49 against 80,486 for satellite. Which of them CKKS
  evaluates in the browser, page by page, is under
  [which pages encrypt](models-and-benchmarks.md#which-pages-encrypt). All eleven repos are public and
  pinned, and the evaluation sample for each lives in that model's own repo at the same revision as
  its weights - except the satellite student's, which publishes none by design, because its metric
  is defined over a timeline and cannot be re-scored from independent rows.

  A test holds this sentence's count and its one exception to the registry.

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
  could not be set permanently through `chrome://flags` at Round 18; since then WebGPU is on by default for
  Intel Gen12+ from Chrome 144 and NVIDIA on Wayland from Chrome 147, per the
  [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) page of 2026-10-02, and other
  GPUs and WebNN still need flags):

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
- **The retired security score scored its two rows under different inputs**, so most of the gap it
  displayed was an artefact of the call sites
  ([why it was retired](../references/security-scoring-fact-check.md#why-the-existing-score-had-to-go)).
  The lesson generalises past this metric: when two rows are compared, verify they were scored under
  ONE transport configuration before believing the gap.
- **ct/pt is not a constant**: it moves with the feature count, so read it as a property of a
  configuration, not of CKKS ([what ct/pt measures](models-and-benchmarks.md#what-ctpt-measures)).
- **The medical use-case pipeline and the medical AUC use the same dataset since Round 86.** The AUC 0.9147 is
  BreastMNIST, and so are the benchmark's encrypted CNN and, since then, the full deployment's medical
  pipeline, which reads `benchmark.medical_cnn` and runs its CNN path from the page. Until then that
  pipeline defaulted to PneumoniaMNIST, a different MedMNIST v2 subset, and the page never reached it.
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
- **NSA and BSI stance** - neither recommends QKD: NSA answers "No" to QKD for a national security
  system in any configuration, BSI recommends a quantum-safe plus classical hybrid, and offering QKD as
  an optional add-on to that hybrid is **this project's** design choice alone
  ([their positions, in the roadmap](pqc-roadmap.md#standards-alignment-done--in-progress)).
- **GPU/vendor & licensing** - Blackwell+CUDA13 narrows the quantum-GPU stack to CUDA-Q/cuQuantum.
  The FHE stack is deliberately **commercial-OK** (TenSEAL Apache-2.0, OpenFHE BSD-2, HEonGPU Apache-2.0);
  **DESILO was removed** because its license is non-commercial. Multiple swappable backends avoid lock-in.
- **Scope discipline** - the trained QCNNs ship small, for reproducibility, at the widths
  [use-cases.md](use-cases.md#what-the-trained-qcnns-are) lists. The demo's
  **QSMPC-FHE (quantum) comparison row is a genuine in-browser state-vector simulation** of an untrained
  QOTP + Clifford+T circuit (8 qubits by default, not a trained QCNN): its latency is **measured** and its
  T-gate count is **counted from the circuit** - a *simulation*, not quantum hardware (no QPU). Its ct/pt
  is not rendered ([why](models-and-benchmarks.md#what-ctpt-measures)). The classical **MPC-FHE row is real measured
  crypto** (node-seal CKKS + Polytune 2PC) on the pages whose flow runs it.
- **In-browser AI (demo)** - the per-use-case Agentic-AI panel runs **small** models client-side (Transformers.js
  on onnxruntime-web; WebLLM on WebGPU). **Both the agent's Transformers.js models and the published ONNX
  models run on WASM by default, and on WebGPU only when the visitor selects it** (the Backend and Model
  backend controls), because on the machine measured the WebGPU path was the slow one ("A software
  WebGPU adapter is slower than no adapter", above). The model
  picker's sizes are computed from the bytes the config records for each backend's weights file, at a
  pinned commit: 236.7 MB of q8 weights for distilGPT-2 on WASM, 327.8 MB of fp32 on WebGPU. When
  WebGPU is selected, the ONNX path uses **the WebGPU build only where a real `requestAdapter()` returns a HARDWARE
  adapter**, and the WASM build otherwise: a returned adapter is not sufficient, and a software
  rasteriser is declined by name (above). The probe chooses the
  *build* (`onnxruntime-web/webgpu` vs `onnxruntime-web/wasm`), not merely the execution provider,
  because the JSEP bundle never completes session creation without an adapter - that, and not
  `ai.onnx.ml` or WebNN, was the cause of the Round 18 browser-tier hang. WebNN is probed and
  deliberately left unused (the browser model tier, above). A graph
  using `ai.onnx.ml` takes the WASM build whatever the adapter says, because ORT's CI builds the
  WebGPU wasm with `--disable_ml_ops`, so those kernels are compiled out rather than merely
  unpreferred. Because the choice comes from a real `requestAdapter()` call, the reported
  backend is never "webgpu" on a machine with no usable adapter. **It does not mean every node ran on
  the GPU**, and the sentence before this one is why: ORT assigns nodes to execution providers in
  preference order and falls back per node, so a graph can run partly on WebGPU and partly on the
  WASM floor while the badge shows a single state. Which nodes fell back is **not measured here** -
  closing that needs `startProfiling()`/`endProfiling()` or ORT's "kernel not found in registries"
  log, and the badge's two states cannot express a partial answer. Model weights come from Hugging Face at a pinned
  revision on first run and are browser-cached after, so they cost the demo host nothing. The engines' runtime files do
  come from the demo host: ONNX Runtime's wasm, and the WebLLM model libraries, built from the Apache-2.0 MLC-LLM source
  (`web/public/webllm/`, `scripts/build_webllm_libs.sh`). A run contacts the demo and Hugging Face, and no third
  host (checklist `CS12`). WebLLM needs WebGPU; without it WebLLM is
  disabled, Transformers.js runs on WASM as it does by default, and the quantum-safe MCP envelope still
  runs. The in-browser models
  illustrate the agentic-AI + MCP flow; they are not the production inference path.
- **MCP** - the **full deployment ships a real MCP server**: an **`MCPServer`** server (MCP Python SDK, MIT, `mcp>=2.2.0` in `requirements-ai.txt`; the class was named `FastMCP` before v2.0.0 and `server.py` resolves either)
  exposing the 5 per-industry tools, bound to the quantum-safe transport so a `tools/call` runs end-to-end
  under a PQC session (optionally mixed with a fixed 32-byte QKD-domain constant; the MCP path
  receives no QKD key) + ML-DSA + AEAD and returns the **signed real result** (`src/qsh/mcp/server.py`,
  `pqc/mcp_bridge.secure_agent_call`, `/api/mcp/{tools,call}`); it also **interops with the official
  `@modelcontextprotocol/server-filesystem`** under the same quantum-safe seal (`mcp/fs_interop.py`,
  `/api/mcp/filesystem`). The public demo keeps the zero-server `clientMcp.ts` mirror (**zero per-user VPS
  load**). In the WebUI the in-browser agent is **integrated into each use-case page** (its own per-industry
  MCP tool, fed by the live run) - the standalone Agentic-AI page was **removed**, since this project treats MCP
  as a per-use-case cross-cutting layer, not a separate feature.
- **Deploying behind an existing reverse proxy** needs the committed override and the project name
  `-p qsh-demo`, or the container comes up on no network and every request 502s while `docker ps`
  reports it **Up**: [the deployment runbook](deployment.md#demo-tier-behind-a-reverse-proxy-that-already-owns-80-and-443).

- **Demo vs full** - the public demo is **client-only** (zero `/api` compute) and ships **no backend
  image at all**: `docker/docker-compose.demo.yml` has no `api` service, and `/api/*` and `/metrics`
  return 404. Its FHE/MPC is **genuinely real and measured in the browser**: **node-seal CKKS**
  (a real homomorphic linear layer, ~1e-9 error) + a real **Polytune garbled-circuit 2PC** for the
  non-linear argmax, both parties loopback in one tab (zero VPS). In-browser federated learning runs
  on every run of the finance and medical pages, not behind a toggle: a real client-side FedAvg of
  a 4-weight logistic regression whose inputs are `qcnnExpectation`, a closed-form
  `tanh(mean(sin(x_i + 0.5 i)))`, so despite the "Quantum Federated Learning" label nothing quantum
  is computed in that stage; the full deployment runs the **PennyLane** QFL over this repo's own FedAvg + secure aggregation (each pair of clients agreeing its mask seed by X25519), plus server-side FHE inference (incl.
  the medical CNN under real encrypted convolution). In the browser, node-seal CKKS evaluates the
  published medical student's convolution and both dense layers, 28 counted rotations, each layer's
  bias added under encryption, and the client decrypts each layer's outputs between layers to apply
  ReLU and average pooling in plaintext.
- **There is no third (remote) tier, and that is a decision rather than an omission.**
  `spaces/qsh-inference/` was written to serve full-precision models from a Hugging Face ZeroGPU Space
  for anything too large to run in a tab or under encryption. It was **decided against in Round 24**
  and the record is in `docs/deliverable-verification.md`: the public demo is zero-server by design,
  so a remote provenance contradicts the architecture rather than extending it. The Space itself
  returns **HTTP 401** (re-checked 2026-08-12) and the org Space listing is empty, so `ProvenanceBadge`
  declares no remote state ([its four states](threat-model.md#what-a-results-provenance-badge-says)).
  **Reopening condition:** wire it only if the
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

- **Public-demo data (real or simulator output, and one payload without a licence grant)** - the client-side demo loads
  real or simulated datasets, mirroring the full deployment. Every one carries a licence that permits redistribution except the genomics payload's 1000 Genomes dosage slice, which has no licence instrument (IGSR's disclaimer; it ships under the cell budget of DATA_LICENSES.md section 4); the payload's PGS000072 weights are CC BY 4.0, with the credits that licence asks for. **medical** uses a real Wisconsin
  Diagnostic Breast-Cancer sample (UCI dataset 17 via sklearn, **CC BY 4.0 - attribution required**) + a real logistic-regression model;
  **genomics** uses the real PGS-Catalog PGS000072 weights (Graff et al., Nat Commun 12:970, 2021, **CC BY 4.0**; 183 of its 187 variants have a 1000 Genomes genotype and are scored) over real 1000 Genomes
  genotypes -> a **Polygenic Risk Score** over those 183 variants with per-SNP signed SHAP, and its CKKS step evaluates that full score: all 183 dosages in one ciphertext, the plaintext weights applied by one multiply-plain and a rotation sum, the decrypted score compared with the 90th percentile of the score over the 30 bundled individuals; **finance** uses a
  **PaySim** simulator sample (synthetic transactions, CC-BY-SA-4.0) + a linear surrogate over four of its columns (none built from PaySim's balance columns), which is what the finance page's one-row CKKS step evaluates (the 33-parameter AMLworld student is encrypted on 8,192 held-out test-window rows for the agreement figure since Round 84); and **satellite** scores
  **real OPS-SAT-AD telemetry segments** (CC BY 4.0, since Round 84; the NASA SMAP/MSL windows before them had no licence from any provider - see DATA_LICENSES.md) with a plaintext demo anomaly regression - all tiny static JSON
  under `web/public/data/`, loaded by [`web/src/lib/realData.ts`](../web/src/lib/realData.ts). The finance, medical and genomics pages compute with the
  real in-browser node-seal CKKS + a Polytune garbled-circuit 2PC (notional parties); the satellite page runs neither ([which pages encrypt](models-and-benchmarks.md#which-pages-encrypt)). Only **telecom** has no commercial-OK real attack dataset
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
  path (batching and bootstrapping on HEonGPU, and the non-linear half on PIGEON/hpmpc).
