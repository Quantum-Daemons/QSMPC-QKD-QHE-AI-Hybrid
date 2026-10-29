# Models, datasets and the benchmark

> Moved here from the README in Round 80, by the user's request to keep the README to a summary; the text is unchanged apart from its link paths, and the README links here.

## Datasets

Bundled samples are real measurements where a licence permits it, synthetic where none does; the synthetic generators are an offline-CI fallback only
(logged as such, never a silent substitute):
- **Finance/AML** (demo sample): **PaySim simulator output** (Kaggle `ealaxi/paysim1`, CC-BY-SA-4.0) -> the 8 AML features. PaySim is a *simulator*: these are synthetic transactions from aggregated real mobile-money logs, not real transactions. The **training** corpus is IBM AMLworld HI-Small (CDLA-Sharing-1.0), 5,078,345 transactions at 1 laundering in 981. **AMLworld is synthetic too** - IBM generates it with a multi-agent virtual-world model and states "Everything is synthetic"; the NeurIPS 2023 D&B paper is titled *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. It is used because it is calibrated to real transaction statistics and its CDLA-Sharing terms permit publishing derived weights - **not** because it is real data. No commercial-OK corpus of real laundering transactions exists.
- **Medical**: MedMNIST v2 (CC-BY-4.0), Wisconsin breast-cancer (bundled demo sample; UCI, CC BY 4.0 - attribution required).
- **Genomics**: **real 1000 Genomes** phase-3 genotypes (Ensembl/IGSR - open *access*, but IGSR grants **no blanket licence**: "rights claimed on pieces of data vary") + **real PGS-Catalog** PGS000004 / PRS313 weights (**per-score licence**, EMBL-EBI terms by default). Neither is cleared for redistribution as commercial-OK; see [DATA_LICENSES.md](../DATA_LICENSES.md).
- **Satellite**: **real NASA SMAP/MSL** spacecraft telemetry (Telemanom; Hundman et al. 2018; NASA JPL) - **no licence is granted by any provider** `[UNVERIFIED - reasoned inference]`. Telemanom's `LICENSE.txt` is a Caltech copyright assertion covering the *code*; the Kaggle mirror is labelled "Data files (c) Original Authors". 17 U.S.C. 105 removes US copyright only and is not a grant. Bundled demo sample only - the satellite **training** corpus is ESA-ADB (CC BY 3.0 IGO). See [DATA_LICENSES.md](../DATA_LICENSES.md).
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
while **non-linear layers** (ReLU, argmax) run **under secret-sharing MPC** - Shamir secure
comparison via **MPyC** among three parties, each its own process over loopback TCP
(`src/qsh/mpc/parties.py`), with a **masked** CKKS <-> shares conversion at each boundary: the server
masks each linear output inside the ciphertext before the key holder decrypts it, and the key holder
receives only re-masked results and, at the end, the class index (`src/qsh/mpc/secret_inference.py`;
the bound the masks give is reported per run, see `docs/architecture.md`). This mirrors the Gazelle ->
Delphi -> BOLT -> Bumblebee -> **BLB** line and EncFormer, so the benchmark reports the **FHE-linear
vs MPC-non-linear latency split + MPC communication**, not a single CKKS number. The **QSMPC-FHE**
quantum analogue is **Clifford gates (linear, cheap) vs T-gates (non-linear, expensive)** - but QOTP /
Rate-1 QFHE evaluate *both* on the encrypted state, so there is **no FHE <-> MPC boundary**; the
secure-comparison role is played by the QSMS/QOT quantum-MPC path. (This paragraph quoted a ct/pt of
about 1.3 for them, a figure the note under the benchmark table explains cannot share a column with
CKKS's byte ratio.)

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
on the Blackwell box the FHE-linear conv cost dominates (~4.6 s for one 14x14 image on CPU CKKS, against
~2.0 s for the three MPC parties' 801 comparisons; decision faithful to plaintext). On a two-core machine
the three party processes share the cores and the MPC half can be the larger. This is FHE-feasible by design - a MedMNIST-scale CNN, **not** ResNet/ViT (FHE cannot run SOTA
CNNs/LLMs under encryption - a fundamental crypto-cost limit, sold honestly as a PoC).

For the other industries the encrypted hybrid preserves the plaintext decision regardless of weights, so
the metric is FHE/MPC **cost + fidelity**, not task SOTA; lightweight === the trained linear model, so SHAP
stays exact. The benchmark shows the architecture label + parameter count.

There is **no lightweight / medium / large** selector any more either. It scaled invented
architectures - a "tabular fraud MLP" at 18/850/9,538 parameters against a real distilled student of
33 - so the sizes described models nothing had trained. What survives of the idea: **the measured
path runs end to end for real**
(CKKS-FHE linear + MPyC-MPC non-linear); **large is clearly labelled *projected*** - extrapolated from the
measured medium per-op cost (MAC / comparison counts) so the page stays responsive. Results on the
Blackwell box (TenSEAL CPU + MPyC, `linear` workload), measured 2026-10-27 with the MPyC tier as three
parties and masked conversions:

| Path | Backend | Size | Latency/inf | FHE-linear | MPC-non-linear | MPC comm | Acc | ct/pt | Security model |
|------|---------|------|-------------|-----------|----------------|----------|-----|-------|----------------|
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | lightweight | ~41 ms | ~38 ms | ~3 ms | ~15 KB | 1.0 | ~1393x | computational (RLWE) |
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | medium | ~267 ms | ~223 ms | ~43 ms | ~252 KB | 1.0 | ~1393x | computational (RLWE) |
| MPC-FHE | TenSEAL CKKS + MPyC (CPU) | large \* | ~2006 ms | ~1759 ms | ~247 ms | ~1438 KB | 1.0 | - | computational (RLWE) |
| MPC-FHE | HEonGPU CKKS (GPU sm_120) + MPyC | medium | batched | GPU | CPU | - | - | - | computational (RLWE) |
| QSMPC-FHE | QOTP + QCNN (Clifford+T in the browser), no trained weights | - | ~3 ms | Clifford | T-gates x56 | n/a | n/a | - (see note) | information-theoretic as a scheme (QOTP); not this run (seeded pad) |

**What the MPC columns measure, since Round 75.** The non-linear layers run under MPyC as three
parties, each its own process, over loopback TCP: Shamir threshold t = 1, statistical security 40,
64-bit fixed point. "MPC comm" is the bytes the three parties sent, about 15,120 per secure comparison
and the same on every run. "MPC-non-linear" is the protocol's time on the parties' own clocks, about
2.2 ms per comparison here. Each MPC session also spends about 0.25 s starting its processes, which
parties standing between requests would not, so that time is left out of the latency and reported as
`mpc_session_overhead_ms`. Loopback has no network delay, so a LAN or WAN adds a round trip per
communication round. The key holder decrypts only masked values and receives the class index; each
row's `mask_statistical_distance_max` bounds what its view of one inference gives away, 9e-8 or less
here (`docs/architecture.md` states what the bound does and does not cover).

Before Round 75 the tier ran as ONE in-process party, MPyC's defaults with its command line cleared:
m = 1 and t = 0, where a share is the value itself. Nothing crossed a wire, and the ~0.5, ~4.5 and
~24.5 KB this table printed were 256 B per comparison multiplied out. The measured figure is 59x that.

**Note on the dash in the QSMPC-FHE ct/pt cell.** Earlier revisions printed `1.0` here and 1.2769 in
the WS4 table below, and a config constant of 1.32 fed an Overview card. None of the three can share a
column with the CKKS byte ratio. `1.0` counts QUBITS only: a one-time pad maps $`n`$ qubits to $`n`$ qubits,
but it also needs a $`2n`$-bit classical key, which `1.0` leaves out (Ambainis, Mosca, Tapp and de Wolf,
FOCS 2000). The 1.2769 family (1.0693 at 10 qubits, 1.0173 at 12) divides by the BYTES of a classically
simulated state vector of $`2^n`$ amplitudes, and a quantum ciphertext has no size in bytes. In the units of
the Rate-1 QFHE paper, qubits plus bits (Chardouvelis, Doettling and Malavolta, TCC 2021, Def. 6.1), this
module's 8-qubit KEM-wrapped transport is about 1135x, tending to 3 as $`n`$ grows. And 1.32 is not in
that paper, which reports no concrete figure; the constant and the card are removed. See
[`docs/architecture.md`](architecture.md).

The **Security** column used to publish `68 / 100` and `96 / 100`. Those are the retired composite,
and that particular pair is the worst thing it produced: **16 of the 28 points separating them were
an artefact of which arguments two functions happened to supply** - one call passed no QBER while the
other passed a literal, and the same again for a leakage bound nothing measured. The column now
reports the security *model*, which is the part that was ever a property of the cryptography.

\* **projected** from measured medium per-op cost, not run end-to-end. GPU FHE wins on **batched**
workloads; for a tiny single-vector input the per-call kernel overhead can exceed CPU (raw encrypt
throughput was ~700 ops/s GPU vs ~34 CPU). What the hybrid shows is *where* the classical cost goes
(FHE dominates the linear layers; MPC adds the non-linear secure comparisons + a few KB of
communication). **Withdrawn 2026-10-23:** this paragraph used to call the quantum Rate-1 path about
$`10^3`$ times more compact than classical CKKS, with information-theoretic security. Neither half holds.
The ratio compared a simulated state vector's bytes with a CKKS ciphertext at under 1% slot use (about
12x at full packing; Cheon, Kim, Kim and Song, ASIACRYPT 2017); classical FHE reaches rate 1 as well
(Gentry and Halevi, TCC 2019; Brakerski, Doettling, Garg and Malavolta, TCC 2019); and Rate-1 QFHE
rests on quantum LWE, so its security is computational.

**Three enriched dimensions** (Round 3):
- **Security posture table** (replaced the 0-100 score in Round 22) - per-row facts, each checkable
  against a published standard, measured by the run, or simulated by the run's analytic quantum-channel
  model, and labelled which: NIST category and ciphertext/signature sizes from FIPS 203/204, and, as
  simulated rows (no QKD device exists here), QKD secure key bits, SKR, the QBER verdict taken from
  `secure_key_bits > 0` rather than a threshold, and, on BB84 runs, the GLLP tagged fraction showing
  what the decoy states are buying (E91 uses none, so its table has no such row). `qsh/benchmark/security.py` still computes the old composite for
  backwards compatibility but nothing renders it. Why it was retired:
  `references/security-scoring-fact-check.md`.
- **Quantum advantage (0-10) - retired 2026-10-23, not rendered.** `qsh/benchmark/quantum_advantage.py`
  is kept, as the 0-100 composite was, but no panel, log line, diagram label or agent prompt shows it.
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

### What the pages show (Round 58)

Every page leads with its key points; method and history are folded, never a caveat that qualifies a
visible number.

- **Overview** - an architecture diagram of the stages as they run in the browser, each badged real,
  real protocol with notional parties, simulated, or simulated with a seeded pad. The simulated QKD link
  is drawn off the key path, because nothing it produces reaches the browser's session key.
- **Value** - the novelty sentence first, then a regulatory clock (a timeline by jurisdiction and a
  table) in which every date carries who it binds and how firm it is: requirement, proposed rule,
  draft, company target, deferred, or passed.
- **Validations** - the PQC primitives; the simulated BB84 channel with a sweep of QBER and secure bits
  against the eavesdropper fraction; the MCP agent call; and the network layer's key schedules,
  computed in the tab:
  - an ML-KEM-768 known-answer test against NIST ACVP cases (key generation, encapsulation, and
    decapsulation of valid and modified ciphertexts);
  - a WireGuard Noise_IKpsk2 handshake keyed by a preshared key from an in-tab ML-KEM-768 + X25519
    hybrid - the Noise core is checked against the official cacophony vector, and the WireGuard framing
    follows the whitepaper, for which no published vector exists;
  - an IKEv2 key schedule - the RFC 7296 derivation checked against NIST CAVP SP 800-135 vectors, the
    RFC 9370 ML-KEM-768 and RFC 8784 post-quantum preshared-key layers computed per the RFCs (they
    have no official vectors). ML-KEM in IKEv2 is still an Internet-Draft in the RFC Editor queue.

  Sources, commits and hashes: [`references/net-key-schedules.md`](../references/net-key-schedules.md).
  The tunnels themselves run in the companion network demo, https://pqc-qkd-hybrid.daemons.jp/
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
  neighbours (Vergoossen et al. 2019), whose terminals load from templates. The same tab draws the
  model against all 23 Micius passes Liao et al. list: centred on them (median -0.03 dB), with the
  nights scattering 2.4 dB around it. Simulated: no satellite is involved. A pass's key is also
  shown as a composable finite key (below); the relay's and the constellation's keys are asymptotic
  bounds. Every key rate on the site counts the pulses sent; only signal pulses carry key.
- **Satellite key block** (Round 79) - the satellite card's key is the composable finite key of a
  block of link time, not the asymptotic bound times a fixed pulse count. By default the block is
  one pass peaking at the page's range, the block Micius (Liao et al. 2017, 2018), Tiangong-2 and
  the Micius-Zvenigorod session (Khmelev et al. 2023) computed their keys on; pooled passes (Sidhu
  et al. 2022) and a time held at one range are offered beside it. The bound follows the source:
  Lim et al. (Phys. Rev. A 89, 022307, 2014) for a signal, a weak decoy and the vacuum, as Micius
  sent, and Rusca et al. 2018 for one decoy. Checked, with nothing fitted: Rusca et al.'s Table I
  two-decoy rows within 20 percent and below them, the ratio of the two protocols within 15
  percent (Round 80: the factor of about 2 recorded before was a test channel applying the
  detector twice); Liao's pass at 0.55 of its published 300,939 bits and Khmelev's session at 0.56 of a
  Chernoff-bound key, the same analysis gap twice; Jinan-1's four published passes within a factor
  of 2.5. Writing it found two errors in the one-decoy bound, both on the safe side: a constant
  transcribed as 2 to the 21st where both papers print 21 squared, and the vacuum bounded by the
  signal's errors where the decoy's are tighter. The fibre threshold moved from 5.88e6 to 4.78e6
  pulses.
- **Fibre key block** (Round 80) - every fibre key is a composable finite key too: the use-case
  pages, the two-party demo and Validations page, and the Python API report the same one, because
  the model (`run_bb84` / `bb84()`) computes it. By default the block is a privacy-amplification
  block of 1e7 key-basis detections, the size Rusca et al.'s Table I uses and the class the Geneva
  systems run (Boaron et al. 2018; Grunenfelder et al. 2020); the pulses and time follow from the
  distance - 5.3 s at 50 km, 35 s at 100 km, about 5.1 Mbit either way. Seconds of link time and
  the run's own pulses are offered beside it. Checking that default found the reason this project
  had recorded its finite-key code as about 2x below Rusca et al.'s table since Round 47: its tests
  applied the detector on top of the paper's global attenuation, which already includes it. With
  the attenuation applied once the table's 26 dB row is reproduced to a few parts in a thousand
  under the paper's dead-time convention, and the 26 and 46 dB rows within 5 percent under this
  project's; the time to fill a block within 10 percent. E91 keeps its asymptotic count, labelled.
- **3D views** (three.js, loaded only where drawn) - the QCNN's per-qubit reduced Bloch vectors through
  the real circuit, an illustration of the one-time pad on one qubit, and the satellite geometry the
  model itself uses: the orbit, the elevation a slant range implies, the arc above the minimum
  elevation, and a constellation's spacing, line-of-sight clearance and shadow.
