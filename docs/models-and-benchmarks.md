# Models, datasets and the benchmark

## Datasets

Bundled samples are real measurements or simulator output. Every one the browser serves carries a
licence that permits redistribution except the genomics payload's 1000 Genomes dosage slice, which
has no licence instrument at all (IGSR's disclaimer; it ships under the cell budget of
[DATA_LICENSES.md](../DATA_LICENSES.md) section 4); its PGS000072 weights are CC BY 4.0 (Graff et al.,
Nat Commun 12:970, 2021). The synthetic
generators are an offline-CI fallback only (logged as such, never a silent substitute):
- **Finance/AML** (demo sample): **PaySim simulator output** (Kaggle `ealaxi/paysim1`, CC-BY-SA-4.0) -> eight AML-named columns, of which the demo's linear model reads four (amount, cross_border, night_tx, new_payee) and none built from PaySim's balance columns, which the dataset's description says must not be used for fraud detection. PaySim is a *simulator*: these are synthetic transactions from aggregated real mobile-money logs, not real transactions. The **training** corpus is IBM AMLworld HI-Small (CDLA-Sharing-1.0), 5,078,345 transactions at 1 laundering in 981. **AMLworld is synthetic too** - IBM generates it with a multi-agent virtual-world model and states "Everything is synthetic"; the NeurIPS 2023 D&B paper is titled *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. It is used because it is calibrated to real transaction statistics and its CDLA-Sharing terms permit publishing derived weights - **not** because it is real data. No commercial-OK corpus of real laundering transactions exists.
- **Medical**: MedMNIST v2 BreastMNIST for the trained models (CC BY 4.0 asserted by MedMNIST; the BUSI authors publish no licence - see [DATA_LICENSES.md](../DATA_LICENSES.md)), MedMNIST v2 PneumoniaMNIST as the full deployment's use-case default (CC BY 4.0), Wisconsin breast-cancer (bundled demo sample; UCI, CC BY 4.0 - attribution required). MedMNIST states that its data "is NOT intended for clinical use", and nothing here is either.
- **Genomics**: **real 1000 Genomes** phase-3 genotypes, read through the Ensembl REST API (open *access*, but IGSR attaches **no licence instrument**: "rights claimed on pieces of data vary"; cite The 1000 Genomes Project Consortium, Nature 526:68-74, 2015) + the **real PGS-Catalog** PGS000072 weights (Graff et al., Nat Commun 12:970, 2021, **CC BY 4.0**; 183 of the 187 variants have a 1000 Genomes genotype and are scored, the other 4 are left out). The weights are cleared for redistribution with credit; the genotypes are shipped as a slice under the cell budget, not under a licence. See [DATA_LICENSES.md](../DATA_LICENSES.md).
- **Satellite**: the browser demo scores **real OPS-SAT-AD** telemetry segments (ESA OPS-SAT; Ruszczak et al., Scientific Data 2025; CC BY 4.0) since Round 84, with a demo anomaly regression fitted on 370 test-partition segments and verdicts drawn from the other 159. The **real NASA SMAP/MSL** spacecraft telemetry it served before (Telemanom; Hundman et al. 2018; NASA JPL) has **no licence granted by any provider** (checked at each source on 2026-10-30), so it is neither served nor kept in the repository; the full deployment's CKKS telemetry stage scores one of the same OPS-SAT-AD segments per run, with the same demo regression as the browser ([which pages encrypt](#which-pages-encrypt)). Telemanom's `LICENSE.txt` is a Caltech copyright assertion covering the *code*; the Kaggle mirror is labelled "Data files (c) Original Authors". 17 U.S.C. 105 removes US copyright only, covers works of federal employees (JPL staff are Caltech employees), and is not a grant. Bundled demo sample only - the satellite **training** corpus is ESA-ADB (CC BY 3.0 IGO). See [DATA_LICENSES.md](../DATA_LICENSES.md).
- **Telecom/defense**: literature-grounded synthetic QKD-attack telemetry (no commercial-OK real corpus of **QKD-layer** attacks exists - producing one needs a physical QKD testbed under attack).

Loaders + generators: `qsh.data.loaders` / `qsh.data.synthetic`.

---

## Benchmark - MPC-FHE vs QSMPC-FHE

On each use-case page the benchmark is **unified with the live simulator**: a single **Run**
executes the E2E pipeline and the benchmark is **assembled from that run** - the MPC-FHE row carries the
run's measured FHE-linear + MPC-non-linear cost, alongside the QSMPC-FHE comparison row (the backend
`QuantumComparisonStage` writes the comparison into the ML-DSA-signed summary). On finance and medical
the latency is reported as measured; genomics has no encrypted student, and its row prices the
weighted sum that runs, at scale 1 and at every model size, and the row's label says so. The
retired quantum-advantage score is displayed nowhere and computed for no run: the browser and the
full deployment's `/api/benchmark` both return `advantage` as null. The CLI below runs the same
comparison standalone.

`bash scripts/run_benchmark.sh linear` (or `python -m qsh.benchmark.runner --workload linear
--model-size medium --out results/`; artifacts -> `results/benchmark_*.{csv,json}`, also viewable
via the benchmark panel on each use-case simulator page).

### Faithful hybrid: linear -> FHE, non-linear -> MPC

The classical path is an **MPC-FHE hybrid**, not CKKS-only. **Linear layers**
(matmul / conv) run **under CKKS/FHE** (TenSEAL or OpenFHE on the CPU, or HEonGPU on the GPU where
`scripts/build_heongpu.sh` has built it),
while **non-linear layers** (ReLU, argmax) run **under secret-sharing MPC** - Shamir secure
comparison via **MPyC** among three parties, each its own process over loopback TCP
(`src/qsh/mpc/parties.py`), with a **masked** CKKS <-> shares conversion at each boundary: the server
adds a mask to each linear output inside the ciphertext before the key holder decrypts it, and the
MPC opens re-masked results and, at the end, the class index to the key holder
(`src/qsh/mpc/secret_inference.py`); every CKKS slot the key holder can decode is masked, and what
that hides is in [the threat model's table](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment).
This mirrors the Gazelle ->
Delphi -> BOLT -> Bumblebee -> **BLB** line and EncFormer, so the benchmark reports the **FHE-linear
vs MPC-non-linear latency split + MPC communication**, not a single CKKS number. The **QSMPC-FHE**
row draws an analogy, not a fact about the gates, every one of which is a linear map: in QHE, Clifford gates are
cheap the way linear layers are, and T gates expensive the way non-linear ones are. The Python
QOTP demo evaluates Clifford gates on the encrypted state with Pauli key updates, the browser's
comparison circuit also applies T gates to the padded state but never corrects or decrypts it, so it
measures cost only, and the Rate-1 module is a transport with no `Eval`. There is **no FHE <-> MPC
boundary** on that side, and nothing there does a secure comparison: QSMS is a secure sum, and no flow runs the
QOT sketch.

### Per-use-case architectures

The benchmark's architecture is **per use case**, not a single generic shape
(`qsh.benchmark.runner.usecase_arch_layers` / `usecase_arch_specs`, mirrored client-side). Its
lightweight row is the trained linear model; its medium and large rows cost representative hidden-layer
shapes from `config.tier_architectures`, labelled "tabular fraud MLP" for finance, and **no use case
trains such an MLP**: finance's encrypted student is a 33-parameter logistic regression. Genomics
has no hidden layer at any size, because what runs is one weighted sum over its scored variants.
**Medical** is the
exception, a **genuine small CNN**: in the browser it reads a tabular (WDBC) row tiled to the image
size, run on the published student's own weights under CKKS, and real MedMNIST images only in the
full deployment. The pages have **no workload selector**;
the CLI's `--workload` still exists. The selector offered `linear` against `cnn`, and once the CNN
path began evaluating the published student's own weights under CKKS, `linear` was the only
fabricated option left on the medical page.

**Medical + `cnn` is a real encrypted convolution, not a cost proxy.** When the medical use case loads
MedMNIST (`config['dataset']='medmnist'` / the CNN workload), the pipeline federated-trains a tiny CNN
(one conv block 1 -> 4 ch 3x3, avg-pool, dense head at 14x14) and runs it under the hybrid as a **genuine FHE
convolution** - im2col -> CKKS `dot_plain` per output channel (`src/qsh/fhe/conv.py`), with ReLU
under MPC (`secure_relu`) and the average pool computed on each party's share locally (`secure_maxpool`
serves a max-pool layer). The convolution arithmetic happens on the **encrypted image**, with every
slot masked as above;
on the Blackwell box the FHE-linear conv cost dominates (~4.6 s for one 14x14 image on CPU CKKS, against
~2.0 s for the three MPC parties' 801 comparisons; decision faithful to plaintext). On a two-core machine
the three party processes share the cores and the MPC half can be the larger. This is FHE-feasible by
design for **this operating point** - a MedMNIST-scale CNN, **not** ResNet/ViT, because the demo targets
one browser tab running single-threaded WASM CKKS. It is not a limit of FHE itself: GPU systems have
run ResNet-20 and ResNet-50 and Llama-3-8B under CKKS ([`docs/findings.md`](findings.md)).

For the other industries the encrypted hybrid preserves the plaintext decision regardless of weights, so
the metric is FHE/MPC **cost + fidelity**, not task SOTA; lightweight === the trained linear model, so SHAP
stays exact. The benchmark shows the architecture label + parameter count.

The pages have **no lightweight / medium / large** selector either; the CLI's `--model-size` and the
config's `tier_architectures` still exist and produce the rows below. The selector scaled invented
architectures - a "tabular fraud MLP" at 18/850/9,538 parameters against a real distilled student of
33 - so the sizes described models nothing had trained. What survives of the idea: **the measured
path runs end to end for real**
(CKKS-FHE linear + MPyC-MPC non-linear); **large is clearly labelled *projected*** - extrapolated from the
measured medium per-op cost (MAC / comparison counts) so the page stays responsive. Results on the
development machine (`linear` workload, 6 samples), measured 2026-11-06 with the MPyC tier as three
parties and every slot masked, by `OMP_NUM_THREADS=6 python -m qsh.benchmark.runner --model-size <size>
--no-openfhe`. The rows are in `docs/measurements/hybrid-2026-11-06-*.json` and the machine is in
`fhe-backends-2026-11-06.json` beside them; the QSMPC-FHE row is the browser's, as its label says:

| Path | Backend | Size | Latency/inf | FHE-linear | MPC-non-linear | MPC comm | Acc | ct/pt | Security model |
|------|---------|------|-------------|-----------|----------------|----------|-----|-------|----------------|
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | lightweight | ~49 ms | ~44 ms | ~4 ms | ~15 KB | 1.0 | ~1393x | computational (RLWE) |
| MPC-FHE | HEonGPU CKKS (GPU) + MPyC (CPU) | lightweight | ~6.5 ms | ~2.3 ms | ~4 ms | ~15 KB | 1.0 | - | computational (RLWE) |
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | medium | ~297 ms | ~258 ms | ~39 ms | ~252 KB | 1.0 | ~1393x | computational (RLWE) |
| MPC-FHE | HEonGPU CKKS (GPU) + MPyC (CPU) | medium | ~52 ms | ~13 ms | ~39 ms | ~252 KB | 1.0 | - | computational (RLWE) |
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | large \* | ~2263 ms | ~2036 ms | ~226 ms | ~1438 KB | 1.0 | - | computational (RLWE) |
| QSMPC-FHE | QOTP + an untrained Clifford+T circuit (the browser's comparison circuit, not a trained QCNN) | - | ~3 ms | Clifford | T-gates x56 at 8 qubits | n/a | n/a | - (see note) | information-theoretic as a scheme (QOTP); not this run (seeded pad) |

**The GPU rows** run the same hybrid with its linear half in HEonGPU v2.1.1 on the GPU, through this
project's binding: 19.5 times faster than TenSEAL at both sizes (258 to 13.2 ms at medium), and the
whole inference 5.7 times at medium and 7.6 at lightweight. The MPC half is the same MPyC on the CPU,
so it is now three quarters of a GPU row's latency, 39 of 52 ms at medium. A GPU row's ct/pt is not
read: the binding returns no serialised ciphertext, and a size in GPU memory would match neither
kind of bytes [the ct/pt section](#what-ctpt-measures) counts. The operations themselves are timed
under [CKKS on each backend, measured](#ckks-on-each-backend-measured).

**What the MPC columns measure.** The non-linear layers run under MPyC as three
parties, each its own process, over loopback TCP: Shamir threshold t = 1, statistical security 40,
64-bit fixed point. "MPC comm" is the bytes the three parties sent, about 15,120 per secure comparison
and the same on every run. "MPC-non-linear" is the protocol's time on the parties' own clocks, about
2.3 ms per comparison here. Each MPC session also spends 0.16 to 0.20 s starting its processes, which
parties standing between requests would not, so that time is left out of the latency and reported as
`mpc_session_overhead_ms`. Loopback has no network delay, so a LAN or WAN adds a round trip per
communication round. Each row's `view_statistical_distance_max` bounds what the key holder's whole
view of one inference gives away, and `mask_statistical_distance_max` (9e-8 or less in the rows
recorded before Round 84) covers only the values that enter and leave the MPC through slot 0; what
the masking hides is in [the threat model's table](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment),
and what the bounds cover in [`architecture.md`](architecture.md#where-the-non-linear-half-goes-and-why-that-choice-is-a-security-decision).
Masking every slot adds about 3.2 ms per linear output to the 3.6 ms dot product, so an FHE-linear
latency recorded before Round 84 is lower than a run gives now.

The ct/pt column, and the dash in its QSMPC-FHE cell, are explained under
[What ct/pt measures](#what-ctpt-measures) below.

\* **projected** from measured medium per-op cost, not run end-to-end, and for the CPU row only. What
the hybrid shows is *where* the classical cost goes
(FHE dominates the linear layers; MPC adds the non-linear secure comparisons + a few KB of
communication).

**Three enriched dimensions**:
- **Security posture table** - per-row facts, each checkable
  against a published standard, measured by the run, or simulated by the run's analytic quantum-channel
  model, and labelled which: NIST category and ciphertext/signature sizes from FIPS 203/204, and, as
  simulated rows (no QKD device exists here), QKD secure key bits, SKR, the QBER verdict taken from
  `secure_key_bits > 0` rather than a threshold, and, on BB84 runs, the GLLP tagged fraction showing
  what the decoy states are buying (E91 uses none, so its table has no such row). It and the table's
  Security model column replaced the 0-100 composite, which `qsh/benchmark/security.py` still computes
  and the full deployment's `/api/benchmark` still returns in each row, but no page renders
  ([why it was retired](../references/security-scoring-fact-check.md#why-the-existing-score-had-to-go)).
- **Quantum advantage (0-10) - retired 2026-10-23, not rendered.** `qsh/benchmark/quantum_advantage.py`
  is kept for its own tests, as the 0-100 composite was, but no run computes it: the browser, the run
  artifact and the full deployment's `/api/benchmark` return `advantage` as null, and no panel, log
  line, diagram label or agent prompt shows it.
  Its three inputs did not measure the same thing on the two sides: a CKKS byte ratio divided by a qubit
  ratio that omits the classical key; a real CKKS and garbled-circuit inference latency divided by an
  8-qubit state-vector simulation of an untrained circuit (5794x, 8163x and 9610x across three runs of
  one page); and a 1.5x multiplier for a security label the run does not achieve. Gate:
  `tests/test_the_retired_comparisons_stay_retired.py`.
- **Per-use-case architecture simulator** - each industry has its own page (`/usecase/:id`) with a
  **live architecture canvas** (`web/src/components/UseCaseCanvas.tsx`, its stages in
  `web/src/lib/useCaseFlows.ts`): each party is a column of components and each stage a lane between
  them, whose components glow and whose lane flows, a token travelling in the data's direction, as
  that use case's pipeline runs - **Start / Pause / Resume / Step / Stop / Restart drive it**. It
  replaced a two-party sequence diagram in Round 80. Parties are use-case-specific and are the run's
  (Bank A <-> Bank B with four federated clients, four hospitals <-> the aggregator, four cohorts <->
  the analysis site, carrier <-> remote site with Eve on the link when the run sets her, ground station
  <- LEO satellite on the downlink), grounded in `references/` (Zhang QOT npj 2026, Li&Deng
  QHE-FL, Sutradhar QSMS). Labels fill from live metrics (QBER / PSI / federated acc / QSDC secrecy);
  exports a high-DPI PNG + a multi-second GIF. The legend lists only the lanes a canvas draws and the
  caption only what its run does (Round 81: telecom and satellite had listed CKKS, MPC and key lanes
  and an MPC-FHE inference they do not have). The E2E Orchestration page draws the same canvas, for
  the generic demo too, in an Architecture tab beside the stage-order graph. A standalone per-party-timing micro-benchmark also
  exists at `qsh/benchmark/alice_bob_e2e.py` (`POST /api/alice-bob`).

The benchmark and security posture are computed server-side **and** mirrored
client-side (`web/src/lib/clientCompute.ts`) so the public demo shows identical numbers with no backend.
In both, the QKD figures come from the simulated channel model.

### What ct/pt measures

**The CKKS ratio is a property of a configuration, not of CKKS.** The benchmark's ct/pt is the
ciphertext's bytes over the plaintext's, `ciphertext_bytes / (n_in * 8)`, so it moves with the
feature count and measures slot utilisation. **The two tiers also count different bytes for the same
ciphertext**, because they serialise it differently (Round 86, review PI-8; this paragraph quoted
both kinds in one sentence):

- the full deployment counts TenSEAL's `serialize()`, which compresses: about 334,000 bytes at
  $`N = 8192`$, so about 1393x at 30 features in 4,096 slots and about 5222x at 8;
- the browser counts node-seal's `saveToArray(ComprModeType.none)`, uncompressed: 393,329 bytes, so
  about 6146x at 8 features, 12,292x at the 4 of finance's PaySim row and 269x at genomics' 183.

At full packing the uncompressed ratio is 12x (Cheon, Kim, Kim and Song, ASIACRYPT 2017), and about
10x compressed. Measured on 2026-11-02 with TenSEAL 0.3.16 and node-seal 7.0.0. The 1483x the client
demo shows when it has measured no ciphertext is a representative constant
(`cost_per_op.demo_ct_pt_ratio`), not a measurement.

**The QSMPC-FHE cell is a dash, on either transport, because no quantum ratio can share that
column.** A ratio of `1.0`, the 1.2769 family the Rate-1 module reports, and a removed config
constant of 1.32 all fail to share it. `1.0` counts QUBITS only: a one-time pad maps $`n`$ qubits
to $`n`$ qubits, but it also needs a $`2n`$-bit classical key, which `1.0` leaves out (Ambainis,
Mosca, Tapp and de Wolf, FOCS 2000). The 1.2769 family (1.2769 at 8 qubits, 1.0693 at 10, 1.0173 at
12) divides by the BYTES of a classically simulated state vector of $`2^n`$ amplitudes, and a
quantum ciphertext has no size in bytes. In the units of the Rate-1 QFHE paper, qubits plus bits
(Chardouvelis, Doettling and Malavolta, TCC 2021, Def. 6.1), this module's 8-qubit KEM-wrapped
transport is about 1135x, tending to 3 as $`n`$ grows. And 1.32 is not in that paper, which reports
no concrete figure.

**So the quantum Rate-1 path is not about $`10^3`$ times more compact than classical CKKS, and its
security is not information-theoretic.** That ratio compared a simulated state vector's bytes with a
CKKS ciphertext at under 1% slot use; classical FHE reaches rate 1 as well (Gentry and Halevi, TCC
2019; Brakerski, Doettling, Garg and Malavolta, TCC 2019); and Rate-1 QFHE rests on quantum LWE, so
its security is computational.

### Where each published model came from

A published model's files are fixed by its Hugging Face revision (`hf_revision` in
`models/metrics.json`), and the browser fetches that revision and no other. The record of the code and
data that trained a model is weaker (Round 86, review PI-16):

- every recorded `git_sha` ends in `-dirty`: each model was trained from a working tree with uncommitted
  changes, so its commit alone does not reproduce it;
- no student records a `git_sha`, and only `finance_aml.gbm` records a digest of its training data
  (`data_digest`);
- the ids recorded before 2026-11-02 name commits of the history before it was rewritten, which is kept
  privately with a map from each old commit to its new one.

Nothing was retrained to close this. A retrain can record a clean commit and a data digest;
`qsh.models.train.git_sha()` marks a dirty tree and does not refuse one.

### What the pages show

Every page leads with its key points; method and history are folded, never a caveat that qualifies a
visible number.

- **Overview** - an architecture diagram of the stages as they run in the browser, each badged real,
  real protocol with notional parties, simulated, or simulated with a seeded pad. The simulated QKD link
  is drawn off the key path, because nothing it produces reaches the browser's session key.
- **Value** - the novelty sentence first, then a regulatory clock (a timeline by jurisdiction and a
  table) in which every date carries who it binds and how firm it is: requirement, proposed rule,
  draft, company target, deferred, or passed.
- **Validations** - the PQC primitives; the simulated BB84 channel with a sweep of QBER and secure bits
  against the eavesdropper fraction; the MCP agent call; and handshakes of the network layer's kinds,
  keyed with the page's own material and computed in the tab:
  - an ML-KEM-768 known-answer test against NIST ACVP cases (key generation, encapsulation, and
    decapsulation of valid and modified ciphertexts);
  - a WireGuard Noise_IKpsk2 handshake keyed by a preshared key from an in-tab ML-KEM-768 + X25519
    hybrid - the Noise core is checked against the vector cacophony publishes (a Haskell implementation's vector file, listed on the Noise wiki), and the WireGuard framing
    follows the whitepaper, for which no published vector exists;
  - an IKEv2 key schedule - the RFC 7296 derivation checked against NIST CAVP SP 800-135 vectors, the
    RFC 9370 ML-KEM-768 and RFC 8784 post-quantum preshared-key layers computed per the RFCs (they
    have no official vectors). ML-KEM in IKEv2 is still an Internet-Draft in the RFC Editor queue.

  Sources, commits and hashes: [`references/net-key-schedules.md`](../references/net-key-schedules.md).
  The tunnels themselves run in the companion network demo, pqc-qkd-HybridNetwork
  (WireGuard and strongSwan between containers; its QKD key servers are simulated too). Nothing here
  connects to it, and no figure from it is quoted.
- **Use-case pages** - model curves for the simulated channel, computed by the functions a run calls:
  BB84's rate with and without decoy states and E91's CHSH value against fibre length (telecom), and
  QSDC secrecy capacity against downlink range (satellite).
- **Satellite network** (the satellite card's Network tab) - what carries a key past the horizon a
  downlink cannot see over. A pass model, checked against Liao et al. 2017's pass of 19 December
  2016 counted the way the paper counts it (detections and signal-state sifted bits within 1
  percent); a trusted satellite relay as Micius flew it between Xinglong and Graz (Liao et al.
  2018), with the XOR on real random keys; and a single plane of satellites linked to their
  neighbours (Vergoossen et al. 2020), whose terminals load from templates. The same tab draws the
  model against all 23 Micius passes Liao et al. list: centred on them (median -0.03 dB), with the
  nights scattering 2.4 dB around it. Simulated: no satellite is involved. A pass's key is also
  shown as a composable finite key ([the QKD key blocks](#the-qkd-key-blocks)), and so is the relay's,
  each pass a block of its own, which is what its count of pass pairs for Liao et al.'s 100 kB relayed
  key uses; the constellation's keys are asymptotic bounds.
- **3D views** (three.js, loaded only where drawn) - the per-qubit reduced Bloch vectors through the
  QSMPC-FHE comparison circuit (the untrained QOTP + Clifford+T circuit, on one illustrative input, not
  the trained QCNN that decides a page's class), an illustration of the one-time pad on one qubit, and the satellite geometry the
  model itself uses: the orbit, the elevation a slant range implies, the arc above the minimum
  elevation, and a constellation's spacing, line-of-sight clearance and shadow.

#### Which pages encrypt

In the browser, the finance, medical and genomics pages run node-seal CKKS and a Polytune
garbled-circuit 2PC between notional parties in one tab; the telecom and satellite pages run a CKKS check of a
published student and no 2PC. The satellite page's anomaly verdict and SHAP explanation are a separate demo regression,
computed in plaintext on a bundled OPS-SAT-AD segment; its check scores `satellite_comms.opssat_student` on the benchmark's
529 test-partition segments, through the quadratic expansion and the scaler its ONNX graph applies. The full deployment scores one of those
segments, with the same demo regression, under CKKS (TenSEAL), and reports how far the decrypted
logits are from the plaintext ones. What CKKS evaluates in the browser: on the medical page the
Wisconsin logistic regression, whose logits the 2PC argmax then compares, and the published medical
student's convolution and both dense layers; the telecom student; the full 183-variant genomics
score; and, on finance, a linear surrogate over four columns of the PaySim demo sample, while the
33-parameter AMLworld student is encrypted on 8,192 held-out test-window rows for its agreement
figure. What each step hides, in either tier, is in
[the threat model's table](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment).

#### The QKD key blocks

Every BB84 key a run reports, on the fibre and from the satellite, is the composable finite key of a
block, computed by the function the run calls; E91 keeps its asymptotic count, labelled, and so does
the constellation on the Network tab. Every key rate on the site counts the
pulses sent; only signal pulses carry key. These are figures of the channel model, not of a link: no
satellite, ground station or quantum channel is involved.

- **Fibre block** - the use-case pages, the two-party demo and Validations page, and the Python API
  report the same key, because the model (`run_bb84` / `bb84()`) computes it. By default the block is
  a privacy-amplification block of 1e7 key-basis detections, the size Rusca et al.'s Table I uses and
  the class the Geneva systems run (Boaron et al. 2018; Grunenfelder et al. 2020); the pulses and
  time follow from the distance - 5.3 s at 50 km and 35 s at 100 km, carrying 5,214,327 and
  5,156,039 bits, about 5.2 Mbit either way. Seconds of link time and the run's own pulses are
  offered beside it.
- **Fibre intensities** - the fibre source's signal and decoy intensities are one of two pairs,
  chosen on the page and in a run's `qkd_intensities`. Both were found by the finite-key optimiser and
  they answer different questions on the shipped 50 km fibre: the default, signal 0.3416 and decoy
  0.1417, carries more key on the default block (5,214,327 bits against 4,977,636); signal 0.399 and
  decoy 0.098 needs the fewest pulses before any key exists (3,312,035 against 3,595,158, under the
  default statistics). The pair reaches the block, the bound, the distance chart and OptiQKD's
  baseline in both languages, which a test checks by running both stages. The satellite keeps its own
  profile's source.
- **Satellite block** - the satellite card's key is the composable finite key of a block of link
  time, not the asymptotic bound times a fixed pulse count. By default the block is one pass peaking
  at the page's range, the block Micius (Liao et al. 2017, 2018), Tiangong-2 and the
  Micius-Zvenigorod session (Khmelev et al. 2024) computed their keys on; pooled passes (Sidhu et al.
  2022) and a time held at one range are offered beside it. Held at 800 km, a key needs about 7 s of
  link time under the satellite's default statistics, and 33 s under Hoeffding's deviation alone.
- **The bounds** follow their sources: Lim et al. (Phys. Rev. A 89, 022307, 2014) for a signal, a
  weak decoy and the vacuum, as Micius sent, and Rusca et al. 2018 for one decoy, each as printed -
  $`21^2`$ where Lim et al.'s Eq. (5) and Rusca et al.'s (A21) print it, the vacuum bounded by the
  decoy's errors, which are tighter than the signal's, and $`	au_0`$ on the error term of (A16) only.
  The three transcription errors found while checking them against the papers are recorded with the
  Rusca et al. entry in [`references/REFERENCES.md`](../references/REFERENCES.md). The shipped fibre
  needs 5.35e6 pulses before any key under Hoeffding, and 3.60e6 under the default statistics.
- **The statistics** - on the fibre each count's expectation is bounded by the tighter of Yin et al.
  2020's Chernoff-type bound (Sci. Rep. 10, 14312, Eqs. (9)-(10)) and the published Hoeffding
  deviation on the basis total, at half the failure probability each: a union bound, so the security
  budget is unchanged. On the satellite the default is Chernoff alone, because there the halving costs
  more than the per-count choice gains (Liao's pass 226,072 bits against 226,056). Neither alone is
  tighter everywhere: Chernoff wins for a count under a quarter of its basis (the satellite's decoy
  and vacuum), Hoeffding for a larger one (the fibre's signal and decoy). Chernoff or Hoeffding alone
  can be chosen on the page.
- **Checked against Rusca et al.'s Table I**, with nothing fitted, under the Hoeffding statistics the
  table was computed with, and with the paper's global attenuation applied once, since it already
  includes the detector: the one-decoy 26 dB row to within 2 percent under the paper's dead-time
  convention, and the 26 and 46 dB rows within 5 and 7 percent under this project's, each below the
  table; the time to fill a block within 10 percent; the two-decoy rows within 20 percent and below
  them; and the ratio of the two protocols within 15 percent.
- **Checked against the flown keys**, which both papers key on both bases (Liao et al. distil theirs
  from all 1,671,072 matching-basis signal detections; Khmelev et al. apply their Eq. (7) to the data
  divided between X and Z). Keyed on the Z basis alone, as Lim et al.'s protocol does and every run
  does by default, Liao's pass carries 0.55 of its published 300,939 bits under Hoeffding's deviation
  (164,183 bits) and 0.75 (226,072 bits) under the satellite's default Chernoff statistics; keyed on
  both bases, with the failure probability split between them, it carries 1.08 (325,556 bits) and
  1.50 (450,923 bits). Khmelev et al.'s final key is 310,400 bits (629,000 before their
  detector-mismatch analysis, at an error-correction efficiency of 1.44 against 1.16 here): on one
  basis the model gives 0.56 of the 629,000 under Hoeffding and 0.69 under Chernoff, 1.40 of the final
  key, and on both bases 2.80 of it (868,462 bits). So the drop from the pass's asymptotic key
  (462,290 bits) to the one-basis finite key is mostly the basis convention, 95 percent of it under
  Chernoff and 54 percent under Hoeffding, not the statistics; keyed as the flights keyed theirs, the
  model exceeds both published keys. On Jinan-1's four published passes the model's one-basis
  whole-pass key is within a factor of 2.5 of the published key under Hoeffding, and 1.35 to 2.62
  times it under the default; Jinan-1 computes its keys on 500-kbit blocks inside a pass, which a
  whole-pass key is not held to.

## State-vector simulators, measured

Where a GPU simulator overtakes the CPU for the circuits the Python tiers run: four layers of $`R_Y`$ on every wire and a CNOT chain, with
$`\langle Z_0\rangle`$ read out, which forces the simulation to finish before the clock stops. Median of 7 runs after a first one, in
milliseconds, measured 2026-11-06 on this project's machine (RTX PRO 6000 Blackwell, driver 620.70.06, kernel 5.14.0-687.65.1,
Python 3.12.15) by `scripts/bench_quantum_devices.py`. The raw results and package versions are in `docs/measurements/`. A local measurement,
not something the demo does: the browser simulates its circuits in TypeScript.

| qubits | `default.qubit` | `lightning.qubit` | `lightning.gpu` | Qiskit Aer GPU | CUDA-Q `nvidia` (fp32) | CUDA-Q `nvidia-fp64` |
|---|---|---|---|---|---|---|
| 8 | 2.55 | 9.62 | 1.67 | 1.07 | 0.56 | 0.46 |
| 16 | 19.58 | 7.66 | 3.10 | 2.54 | 1.33 | 1.00 |
| 20 | 979.04 | 71.73 | 4.89 | 5.16 | 1.54 | 1.64 |
| 24 | - | 2,551.57 | 53.77 | 47.26 | 2.03 | 29.77 |
| 28 | - | - | 985.81 | 824.81 | 57.67 | 564.74 |
| 30 | - | - | 4,212.17 | 3,535.66 | 234.90 | 2,423.21 |

Every backend gives the same $`\langle Z_0\rangle`$ to $`5.5\times10^{-7}`$, the widest being single precision. What it shows:

- **At the sizes the shipped circuits use, 4 to 9 wires, the choice does not matter.** Every backend answers in a few milliseconds, so
  `quantum_sim.device` stays `default.qubit`, which runs everywhere and trains the hybrid network by backpropagation.
- **The GPU pays from about 18 qubits**, where `default.qubit` takes 77 ms against 3.6 ms on `lightning.gpu`, and at 20 qubits it is 200
  times faster. `qsh.quantum_device` lets a larger experiment use it without editing a module.
- **CUDA-Q in single precision is about 10 times faster than in double from 22 qubits up**, more than halving the memory traffic alone
  explains. The cause is not established here.
- **Qiskit Aer GPU 0.18.2 runs on this Blackwell GPU.** The earlier records said its wheels did not target `sm_120` and that it had no
  release since 2024; 0.18.2 was released on 2026-10-26, built on the CUDA 12 runtime, and ran these circuits.

CUDA-Q's `qpp-cpu` target is left out. On this hybrid CPU it took anywhere from 1 ms to 1 s for the same small register, depending on
OpenMP's thread count and wait policy and on other load, and nothing here uses it.


## CKKS on each backend, measured

The operations the hybrid's linear half is made of, on each backend, with the same inputs and a check that each answer matches
plaintext before its time is kept. Median of 20 runs after a first one, in milliseconds, measured 2026-11-06 on this project's
machine (RTX PRO 6000 Blackwell, driver 620.70.06, CUDA 13.5; Core i5-13600K) by
`OMP_NUM_THREADS=6 python scripts/bench_fhe_backends.py`, whose record is `docs/measurements/fhe-backends-2026-11-06.json`.
TenSEAL 0.3.16 and HEonGPU v2.1.1 run the configured parameters ($`N = 8192`$, moduli (60, 40, 40, 60), scale $`2^{40}`$,
128-bit security). OpenFHE 1.5.1 runs its own (depth 4, scale $`2^{50}`$, ring dimension 16384, 128 slots), so its column is
not the same computation. A GPU time includes a synchronisation before the clock stops: GPU work is asynchronous, and without
one a timer reads the launch.

| operation | TenSEAL (CPU) | OpenFHE (CPU, own parameters) | HEonGPU (GPU) | HEonGPU against TenSEAL |
|---|---|---|---|---|
| dot product, 33 values | 5.79 | 41.63 | 0.216 | 27x |
| dot product, 256 values | 7.45 | not run: 128 slots | 0.272 | 27x |
| encrypt 33 values | 2.53 | 5.30 | 0.105 | 24x |
| plaintext multiply | 0.84 | 1.56 | 0.040 | 21x |
| ciphertext add | 0.032 | 0.156 | 0.0073 | 4.4x |
| decrypt | 0.43 | 2.85 | 0.049 | 8.8x |
| mask every slot (a fresh encryption, added) | 2.13 | 6.37 | 0.115 | 19x |

- **The GPU is 19 to 27 times faster on the operations with arithmetic in them**, and 19.5 times on the hybrid's whole linear
  half (above). An addition is too small to gain much.
- **These became the GPU's figures after three fixes**, each found by measuring:
  - *The module carried no code for this GPU.* Built inside the binding's CMake project, HEonGPU took nvcc's default architecture,
    Turing's `sm_75`, and the driver recompiled its PTX for `sm_120` at load. `scripts/build_heongpu.sh` now builds for the GPU's
    own architecture and fails if the module carries no code for it. The times hardly moved, 0.2195 ms for the 33-value dot
    product either way, because at $`N = 8192`$ each operation is a few short kernels.
  - *The binding passed vectors as Python lists,* and building a list of 4,096 complex numbers took longer than the GPU took to
    mask them. With NumPy arrays, decrypt went from 0.169 to 0.049 ms, the mask from 0.383 to 0.115, and the hybrid's medium
    linear half from 24.9 to 13.2 ms.
  - *OpenFHE's column depends on OpenMP's thread count.* At OpenMP's default here, one thread per hardware thread (20 on this
    6 P-core, 8 E-core CPU), the 33-value dot product ranged from 81 to 577 ms as other processes came and went, because a
    parallel loop waits for its slowest thread. At 6 threads it held at 41 to 54 ms, and the table is at 6.
- **A GPU is not where this project's inference runs.** The demo's pages encrypt in the visitor's browser (node-seal), and the
  full deployment's image has no GPU code. This is the full deployment's hybrid on a machine that has one.
