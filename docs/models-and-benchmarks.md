# Models, datasets and the benchmark

## Datasets

Bundled samples are real measurements or simulator output. Every one the browser serves carries a
licence that permits redistribution except the genomics payload's 1000 Genomes dosage slice, which
has no licence instrument at all (IGSR's disclaimer; it ships under the cell budget of
[DATA_LICENSES.md](../DATA_LICENSES.md) section 4); its PGS000783 weights are CC BY 3.0 IGO (Kachuri et al.,
Nat Commun 11:6084, 2020). The synthetic
generators are an offline-CI fallback only (logged as such, never a silent substitute):
- **Finance/AML** (demo sample): **PaySim simulator output** (Kaggle `ealaxi/paysim1`, CC-BY-SA-4.0) -> eight AML-named columns, of which the full deployment's linear surrogate reads four (amount, cross_border, night_tx, new_payee) and none built from PaySim's balance columns, which the dataset's description says must not be used for fraud detection. PaySim is a *simulator*: these are synthetic transactions from aggregated real mobile-money logs, not real transactions. The **training** corpus of the AMLworld models is IBM AMLworld HI-Small (CDLA-Sharing-1.0), 5,078,345 transactions at 1 laundering in 981; the finance QCNN and the full deployment's four-column linear surrogate are fitted on the PaySim sample, and the page's federated step trains on its rows; since F7b the page's encrypted steps score AMLworld test-window rows (below). **AMLworld is synthetic too** - IBM generates it with a multi-agent virtual-world model and states "Everything is synthetic"; the NeurIPS 2023 D&B paper is titled *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. It is used because it is calibrated to real transaction statistics and its CDLA-Sharing terms permit publishing derived weights - **not** because it is real data. No real laundering-labelled corpus under a permissive licence was found (2026-11-22): Elliptic and Elliptic2 are CC BY-NC-ND 4.0, and the one real bank set found is GPL-3.0 with differentially private PCA features (arXiv:2507.08835); [`DATA_LICENSES.md`](../DATA_LICENSES.md#7-rejected-as-not-commercial-ok) lists what was checked.
- **Medical**: MedMNIST v2 BreastMNIST for the BreastMNIST models (CC BY 4.0 asserted by MedMNIST; the BUSI authors publish no licence - see [DATA_LICENSES.md](../DATA_LICENSES.md)), NIH ChestX-ray14 for the federated chest X-ray model (`medical_fl.cxr_fedavg`, under NIH's terms), UCI Heart Disease for the four-hospital federation and its differential-privacy arm, PhysioNet/CinC 2016 and CirCor for the heart-sound cards, BreastMNIST also for the full deployment's CNN path (`benchmark.medical_cnn`; PneumoniaMNIST until Round 86), Wisconsin breast-cancer (bundled demo sample; UCI, CC BY 4.0 - attribution required). MedMNIST states that its data "is NOT intended for clinical use", and nothing here is either.
- **Genomics**: **real 1000 Genomes** phase-3 genotypes, read through the Ensembl REST API (open *access*, but IGSR attaches **no licence instrument**: "rights claimed on pieces of data vary"; cite The 1000 Genomes Project Consortium, Nature 526:68-74, 2015) + the **real PGS-Catalog** PGS000783 weights (Kachuri et al., Nat Commun 11:6084, 2020, **CC BY 3.0 IGO**; since Round 86's batch 6, on the owner's choice of 2026-11-26, in place of PGS000072, whose 162 variants they re-weight by $`\beta / \mathrm{SE}`$; 158 of the 162 variants have a 1000 Genomes genotype and are scored, the other 4 are left out). The weights are cleared for redistribution with credit; the genotypes are shipped as a slice under the cell budget, not under a licence. See [DATA_LICENSES.md](../DATA_LICENSES.md).
- **Satellite**: the browser demo scores **real OPS-SAT-AD** telemetry segments (ESA OPS-SAT; Ruszczak et al., Scientific Data 2025; CC BY 4.0) since Round 84, with a demo anomaly regression fitted on 370 test-partition segments and verdicts drawn from the other 159. The **real NASA SMAP/MSL** spacecraft telemetry it served before (Telemanom; Hundman et al. 2018; NASA JPL) has **no licence granted by any provider** (checked at each source on 2026-10-30), so it is neither served nor kept in the repository; the full deployment's CKKS telemetry stage scores one of the same OPS-SAT-AD segments per run, with the same demo regression as the browser ([which pages encrypt](#which-pages-encrypt)). Telemanom's `LICENSE.txt` is a Caltech copyright assertion covering the *code*; the Kaggle mirror is labelled "Data files (c) Original Authors". 17 U.S.C. 105 removes US copyright only, covers works of federal employees (JPL staff are Caltech employees), and is not a grant. Bundled demo sample only - the satellite forecaster and its student are trained on ESA-ADB (CC BY 3.0 IGO), and the segment classifier and its student on OPS-SAT-AD. See [DATA_LICENSES.md](../DATA_LICENSES.md).
- **Telecom/defense**: synthetic QKD-attack telemetry for the QKD-layer detectors, and real 5G-NIDD traffic (CC BY 4.0) for `telecom_ids.nidd`. The class list comes from the literature; the signature values are this project's own illustrative design and are not fitted to any measurement (no public real capture of **QKD-layer** attacks was found: arXiv:2508.12749 reports anomalies recorded on an experimental receiver and states no data release).

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

### The finance models on 114 inputs, and the compact GNN

The four finance models trained on this project's features read the same 114 inputs (IBM's two GNNs, further down, read the transaction graph itself), computed in the open by `model_inputs` in `scripts/train/finance_aml.py` and by `scripts/train/finance_aml_fan.py`: the 32 per-account aggregates, causal so that each row sees only earlier transactions; 79 per-transaction window features, fan-in and fan-out, the spread of amounts and the time since each account's last edge in windows of one hour to three days, and scatter-gather counts; and 3 pair features, the earlier transactions between the same two accounts in each direction and whether this is their first contact (motivated by MEGA-GNN's count of illicit edges on single-edge pairs, 0.7292 against 0.0894 overall, reproduced here from the corpus). Like the Graph Feature Preprocessor paper's pipeline, a transaction's window and pair features see the transactions up to the end of its batch of 2,048, so this is micro-batch scoring and not a per-transaction real-time decision. That horizon is capped at the furthest any batch of the fit window reaches, 17 minutes: in the sparse last days of the corpus a batch takes days to fill, and more than a quarter of the test window's laundering rows would otherwise see over a day ahead. No label is read.

Measured on the whole held-out window (1,523,504 rows, 2,321 of them laundering, the last 30 percent in time order), with every threshold, scaling and seed chosen on the validation window:

| Model | Inputs | Minority-class F1 | AUPRC | Where it runs |
|---|---|---|---|---|
| CKKS linear student (`finance_aml.student`), 115 parameters | 114, distilled from the teacher | 0.3088 | 0.1840 | the browser, under CKKS |
| MLP (`finance_aml.mlp_large`), $`[114, 128, 64, 2]`$, 23,106 parameters, distilled | 114 | 0.6278 (seeds 0.594 to 0.637, median 0.624) | 0.6432 | the browser, CKKS and garbled circuits |
| XGBoost on the same inputs | 114 | 0.6076 | 0.6368 | comparator |
| Gradient-boosting teacher (`finance_aml.gbm`) | 114 | 0.6161 | 0.6428 | the browser (plaintext ONNX) and the full deployment |
| **Compact GNN (`finance_aml.gnn_compact`)**, 34,962 parameters, distilled | 114, and 16 neighbouring edges | **0.6585** (seeds 0.642 to 0.659, median 0.651) | **0.6687** | the browser, CKKS and garbled circuits |
| IBM's Multi-GIN+EU, official code, this split, five seeds | the raw edges; neighbours sampled from every edge of the test window | logged by IBM's loop $`0.6018 \pm 0.0260`$ | - | a server GPU |
| IBM's Multi-PNA+EU (`finance_aml.gnn`), official code, this split, five seeds, 64,374 parameters | the raw edges; neighbours sampled from every edge of the test window | logged by IBM's loop $`0.6487 \pm 0.0208`$; the exported seed 4, 0.6822 at a threshold tuned on validation | 0.6318 (five seeds $`0.6197 \pm 0.0304`$) | trained on a GPU; in the browser in plaintext on 16 held-out transfers (the IBM tab) |
| IBM's Multi-PNA+EU (`finance_aml.gnn_causal`), the same network under the causal protocol, five seeds of each time input | the raw edges with the time before the batch's end in place of the timestamp; neighbours from the transfer's batch of 2,048 or earlier ones, nothing past its horizon | logged by IBM's loop $`0.5628 \pm 0.0314`$; the exported seed 1, 0.6252 at a threshold tuned on validation | 0.5789 (five seeds $`0.5551 \pm 0.0416`$) | trained on a GPU; in the browser in plaintext on 16 held-out transfers (the IBM tab) |

**The compact GNN.** For a transaction from account u to account v, each endpoint carries 16 statistics and 4 neighbouring edges in each direction, the most recent ones up to the transaction's horizon. Each slot and each endpoint is embedded by a shared linear map without bias, so an empty slot stays exactly zero, and a ReLU; the slot embeddings are summed per direction, and a second layer per endpoint mixes the endpoint with its two sums before a head joins them with the MLP channel over the 114 inputs. The shape is fixed and public, so the evaluator learns nothing from it. Under encryption, sums over additive shares cost no AND gates, so the price is the 608 ReLUs and not the edges: the first layer runs under CKKS with one evaluation per output dimension serving every account at once (39 ciphertexts), and the ReLUs run in garbled circuits (52,655 AND gates), batched across a pool of Web Workers. A search of the literature to November 2026 found no private GNN in a browser and none under garbled circuits; the nearest, Bilgi et al. (arXiv:2610.18054) under CKKS with bootstrapping on a GPU and TGHE (arXiv:2606.26664) on capped ego trees, run on servers. That is a search result, not a claim of being first.

**Distilled from IBM's Multi-PNA+EU.** Both networks are trained towards $`(1-\alpha)\,y + \alpha\,p_{\mathrm{PNA}}`$ on each fit-window edge, $`p_{\mathrm{PNA}}`$ the probability IBM's official Multi-PNA+EU gives it on this project's windows (`scripts/train/ibm_multignn/`), with $`\alpha = 0.3`$ chosen on the validation window (0, 0.3, 0.5 and 0.7, three seeds each: mean validation AUPRC 0.5440, 0.5817, 0.5809, 0.5802 for the GNN). This is generalised distillation (Lopez-Paz, Bottou, Schoelkopf and Vapnik, ICLR 2016, arXiv:1511.03643): PNA's neighbourhoods on the fit window include transactions later than the one scored, which these models never see, and that reaches them only through the training targets, so nothing changes at inference and the test window plays no part. It lifted mean validation AUPRC from 0.4929 to 0.5821 for the MLP and the GNN's test F1 from 0.6283 to 0.6587, with the teacher as first dumped. That teacher's probabilities used the BatchNorm statistics IBM's loop saves, set by its test pass; since 2026-11-16 both networks are distilled toward seed 1's probabilities with the statistics recomputed on the fit window, at the same $`\alpha`$, and score 0.6585 (GNN) and 0.6278 (MLP); the MLP's trainer is deterministic, so its change is the teacher's alone (`docs/verification-log.md`). SALT-GNN's mean-and-spread aggregation was tried first and not adopted (`references/REFERENCES.md`).

**Against IBM's official models, run here on the same windows.** The official `IBM/Multi-GNN` code reproduces Multi-GIN+EU on the paper's own split at 0.6205 for seed 1 (the paper reports $`64.79 \pm 1.22`$ over five seeds), and on this project's 56/14/30 windows it logs $`0.6018 \pm 0.0260`$ over five seeds; Multi-PNA+EU, the paper's best ($`68.16 \pm 2.65`$), logs a test F1 of $`0.6487 \pm 0.0208`$ over five seeds at the epochs its loop keeps (seed 1 gave 0.6878 and 0.6616 in two runs of the same command, the spread GPU non-determinism gives). IBM's loop scores in training mode and saves BatchNorm statistics set by its test pass, so the exported model, seed 4 with its statistics recomputed on the fit window, is scored in eval mode at a threshold tuned on the validation window, as this project's models are: test F1 0.6822 and AUPRC 0.6318, and over five seeds $`0.6577 \pm 0.0226`$ and $`0.6197 \pm 0.0304`$ (`scripts/train/ibm_multignn/README.md`). IBM's neighbour sampling never returns 865 to 885 test edges per seed, and its evaluation drops them; one of them was laundering for seeds 2 and 5, none for seed 4. Their test graph holds every edge of the test window, so their neighbours include transactions later than the one scored; the compact GNN sees 16 edges and nothing past its batch, and its AUPRC, 0.6687, is above the exported PNA's, 0.6318, while its F1 is 0.02 below PNA's 0.6822 at a tuned threshold. The figures are reported side by side, not ranked. Under the causal protocol, which reads nothing past the horizon the page's own models give a transfer and in place of IBM's absolute timestamp reads each transfer's time before its batch's end (`lag`; the input set to zero, `drop`, was trained too, five seeds each, and does worse on validation), the same network logs $`0.5628 \pm 0.0314`$, and its exported seed 1 scores 0.6252 with an AUPRC of 0.5789, against the compact GNN's 0.6585 and 0.6687 on the same horizon; Multi-GIN+EU under it logs $`0.4095 \pm 0.0563`$ with `lag`, and less with `drop` (`scripts/train/ibm_multignn/README.md`).

**By the receiving account's in-degree** (the owner's choice of 2026-11-15; the finance page's IBM tab, computed offline). SALT-GNN (Losavio et al., arXiv:2607.10131) reports that graph models lose F1 where a transfer's receiver is dense, binning test transfers by the receiver's in-degree (1-4 to 100+) at a threshold of 0.5. `scripts/research/finance_degree_f1.py` bins this project's 1,523,504 test transfers the same way, from IBM's runs' saved scores, five seeds per protocol, counting the in-degree over every transfer of the file under IBM's protocol, as its test graph holds them, and over the transfers up to the end of the scored one's batch under the causal protocol; the record, `docs/measurements/finance-degree-f1-2026-11-20.json`, holds both windows for every model, both thresholds and a 95% bootstrap interval per bin. At 0.5, in each protocol's own window:

| model | overall | 5-9 | 50-99 (95% interval) |
|---|---|---|---|
| Multi-PNA+EU, IBM's protocol | $`0.638 \pm 0.025`$ | 0.473 | $`0.741 \pm 0.042`$ (0.694 to 0.780) |
| Multi-PNA+EU, causal | $`0.611 \pm 0.046`$ | 0.553 | $`0.642 \pm 0.097`$ (0.553 to 0.717) |
| Multi-GIN+EU, IBM's protocol | $`0.511 \pm 0.044`$ | 0.359 | $`0.714 \pm 0.049`$ (0.671 to 0.754) |
| Multi-GIN+EU, causal | $`0.440 \pm 0.064`$ | 0.375 | $`0.330 \pm 0.152`$ (0.261 to 0.391) |
| SALT-GNN's PNA, its Table 2 | $`0.629 \pm 0.032`$ | 0.516 | $`0.515 \pm 0.099`$ |
| SALT-GNN's GIN, its Table 2 | $`0.418 \pm 0.107`$ | 0.362 | $`0.509 \pm 0.106`$ |

On these windows the dense-receiver drop does not appear under IBM's protocol: both networks score higher in the 50-99 bin than overall, where the paper's PNA scores lower (its GIN scores higher); only GIN under the causal protocol falls there. The 5-9 bin is the lowest for three of the four runs here, as for most models in the paper; for GIN under the causal protocol the 50-99 bin is lower still. The 50-99 bin holds 167 of the 2,321 laundering transfers in IBM's window and 69 in the causal one, and the 100+ bin a single one, so the intervals are wide. The paper's split is temporal with no ratios stated and its degree window is unstated, so the rows are set side by side, not ranked; its code is not released, so no SALT model is reproduced. Each run left 859 to 895 test transfers unscored, counted as not flagged.

Against the published figures on this corpus: on HI-Small the Graph Feature Preprocessor paper (Blanusa et al., ICAIF 2024, Table 5) reports 19.75 percent for XGBoost on basic features, 56.88 to 59.71 with fan-in and fan-out, and 63.23 to 64.77 with its other features; Altman et al. (NeurIPS 2023) report 63.23 for GFP + XGBoost and 56.77 for PNA; Egressy et al. (AAAI 2024) 68.16 for Multi-PNA+EU; and Bilgi et al. 63.72 in plaintext for a GNN they also run under CKKS, at 5.31 s per transaction on a consumer GPU with bootstrapping at $`N = 65536`$. Their splits are 60/20/20 or 64/19/17 and their neighbourhoods include later transactions, so none of the rows above is a reproduction of those figures.

The pages show each network's logits and its decision, not a probability. Trained with class weights towards soft targets, the networks' raw probabilities are not risks: their Brier scores on test, 0.00177 for the MLP and 0.00181 for the GNN, are worse than giving every row the window's rate (0.00152). Each record carries a Platt map fitted on validation, as the teacher's and the student's do, and on the two networks it brings CORP's miscalibration term down 13- and 20-fold, to $`7.4 \times 10^{-5}`$ for the MLP and $`5.3 \times 10^{-5}`$ for the GNN. By CORP's discrimination term the GNN (0.000783) leads the teacher (0.000763) and the MLP (0.000741).

IBM's own Multi-PNA+EU (`finance_aml.gnn`) carried no such map until batch 4 (2026-11-25):
`docs/measurements/finance-gnn-calibration-2026-11-25.json` (`scripts/measure_finance_gnn_calibration.py`) reads the
probabilities its record was scored from, each seed's test AUPRC, rows and positives checked against the record, and
fits the same Platt map on validation. On the published seed, 4, its raw probabilities are overconfident in every bin
above 0.1 (in the 0.1 to 0.2 bin the mean prediction is 0.135 and the observed rate 0.024), with an expected calibration
error of $`2.5 \times 10^{-3}`$ over fifteen equal-mass bins; the map brings it to $`1.3 \times 10^{-4}`$ and CORP's
miscalibration term from $`2.2 \times 10^{-4}`$ to $`9.9 \times 10^{-6}`$, with the discrimination term unchanged, as a
monotone map must leave it. The other four seeds move the same way. The record holds each seed's ten-bin reliability
table with a Wilson interval per bin. On the owner's choice of 2026-11-25, the published seed's map is also in the model's
own record (`--metrics`), in the other finance records' form, so the finance page's model card and the Hub card show it.
The page still shows the GNN's logits and its decision, which compares the probability with a threshold tuned on
validation, and reads no score through the map. On test, 947,218 of the 1,522,629 scored rows lie below the map's logit
clip of $`10^{-12}`$ and tie there, which moves AUPRC from 0.63182937 to 0.63182826.

### The trained QCNN under CKKS

Finance, medical and telecom offer `QCNN (trained, under CKKS)` beside the plaintext `QCNN (trained, not encrypted)`. The readout is exactly $`\langle O\rangle = \sum_P c_P\,\phi_P(x)`$ over $`P \in \{I, X, Z\}^{\otimes n}`$, because the encoding is a product of real states ($`\phi_I = 1`$, $`\phi_X = \sin\theta`$, $`\phi_Z = \cos\theta`$ per wire, $`c_P = 2^{-n}\operatorname{Tr}(AP)`$ with $`A`$ the real part of $`U^\dagger O U`$ on the encoded basis states). So the client encrypts $`3^n`$ products and the server takes one inner product per readout: one multiplicative level at the frozen parameters, power-of-two rotation keys only, no relinearisation key.

Measured in Node on the shipped weights (`npm run check:qcnn-ckks`, the module the page runs): the form differs from the circuit by at most $`8 \times 10^{-15}`$; the decrypted readout by at most $`2 \times 10^{-9}`$; the class equals the plaintext circuit's on every row checked. Finance (4 qubits, 81 terms) takes about 0.1 s, medical (8 qubits, 6,561 terms in 2 ciphertexts) 0.15 s, and telecom (7 features, 2,187 terms, 8 readouts) 0.5 s, of which most is building the form and the keys. The accuracy is the plaintext QCNN's, since the class is the same.

**Compared with classical models, on the circuit's own split** (`classical_surrogate` and `classical_lr_*` in each artifact; the surrogate refits the same estimator on the same Pauli products):

| Page | Metric | Trained QCNN | LR on the same inputs | LR on the same $`3^n`$ products |
|---|---|---|---|---|
| Medical (8 wires) | accuracy | 0.9535 | 0.9535 | 0.9419 |
| Finance (4 wires) | minority-class F1 | 0.2286 | 0.4390 | 0.4889 |
| Telecom (8 wires) | macro F1 | 0.9444 | 0.9764 | 0.9723 |

**The satellite circuits** (Round 86, pre-registered in `references/satellite-qcnn-tn-preregistration.md`; `docs/measurements/satellite-qcnn-2026-11-24.json`) are scored on the benchmarks' own splits by average precision, five seeds each. OPS-SAT-AD, eight of nineteen features: QCNN 0.931 $`\pm`$ 0.003, logistic regression on the same inputs 0.881, the same estimator on the circuit's 6,561 products 0.943. ESA-ADB Mission1, 64-step windows of channels 41-46: 0.864 $`\pm`$ 0.006, 0.857 and 0.892. The surrogate leads on both, as on the other pages. Their tree network also runs under CKKS one row a slot (`clientQcnnTreeCkks.ts`; N = 16384, 360 bits): on all 115,064 ESA-ADB test windows the largest readout error was $`8.5 \times 10^{-8}`$ with no decision changed, at 0.28 ms and 3.8 kB a row on node, against 164 ms and 787 kB a row for the per-row form above (`docs/measurements/satellite-qcnn-tree-ckks-2026-11-24.json`).

The classical comparators are level or ahead on all three, which matches Bowles, Ahmed and Schuld (arXiv:2403.07059). Nothing here claims a quantum advantage. The circuit diagram on each of those pages (`Quantum circuits on this page`) is drawn from a recording of the circuit function, and the gates it draws are replayed against the simulator in the same check.

**The QCNN column is re-scored in the tab** (Round 86). Until then the pages printed it from the weights file and said so. Each page now ships its circuit's test rows, `public/data/<dir>/qcnn_heldout.json`: the last 15 percent of the trainer's seed-0 shuffle, in raw units, written by `scripts/data/build_qcnn_heldout.py` only after the trainer's own circuit reproduces every recorded figure from them. A run with a QCNN selected, and the `Re-score on the held-out rows` button in the circuit card, run the page's circuit over every row and compute the same metric: medical 0.9535 over 86 rows, finance 0.2286 over 600, telecom 0.9444 over 480 (and 0.9679 at 7 wires), each equal to the record, in about 10 to 170 ms. `npm run check:qcnn-rescore` asserts the equality for every served artifact and that one flipped label breaks it. The logistic-regression columns are still read from the weights files.

### Per-use-case architectures

The benchmark's architecture is **per use case**, not a single generic shape
(`qsh.benchmark.runner.usecase_arch_layers` / `usecase_arch_specs`, mirrored client-side). Its
lightweight row is the trained linear model; its medium and large rows cost the use case's trained model where one exists: **finance** the one published MLP, `finance_aml.mlp_large`, whichever size is asked for ([below](#the-finance-models-on-114-inputs-and-the-compact-gnn)); **medical** an MLP or a two-block CNN trained in the run on its own rows; and a use case with neither falls back to generic shapes with random weights, labelled as cost only. Genomics
has no hidden layer at any size, because what runs is one weighted sum over its scored variants.
**Medical** is the
exception, a **genuine small CNN**: in the browser it reads a held-out BreastMNIST test image at $`14 \times 14`$, its published
evaluation sample (a tabular row tiled to the image until Round 86), run on the published student's own weights under
CKKS, and MedMNIST images in the full deployment. The pages have **no workload selector**;
the CLI's `--workload` still exists. The selector offered `linear` against `cnn`, and once the CNN
path began evaluating the published student's own weights under CKKS, `linear` was the only
fabricated option left on the medical page.

**Medical + `cnn` is a real encrypted convolution, not a cost proxy.** When the medical use case loads
MedMNIST (`config['dataset']='medmnist'` / the CNN workload), the pipeline federated-trains a tiny CNN
(one conv block 1 -> 4 ch $`3 \times 3`$, avg-pool, dense head at $`14 \times 14`$) and runs it under the hybrid as a **genuine FHE
convolution** - im2col -> CKKS `dot_plain` per output channel (`src/qsh/fhe/conv.py`), with ReLU
under MPC (`secure_relu`) and the average pool computed on each party's share locally (`secure_maxpool`
serves a max-pool layer). The convolution arithmetic happens on the **encrypted image**, with every
slot masked as above;
on the Blackwell box the FHE-linear conv cost dominates (~4.6 s for one $`14 \times 14`$ image on CPU CKKS, against
~2.0 s for the three MPC parties' 801 comparisons; decision faithful to plaintext). On a two-core machine
the three party processes share the cores and the MPC half can be the larger. This is FHE-feasible by
design for **this operating point** - a MedMNIST-scale CNN, **not** ResNet/ViT, because the demo targets
one browser tab running single-threaded WASM CKKS. It is not a limit of FHE itself: GPU systems have
run ResNet-20 and ResNet-50 and Llama-3-8B under CKKS ([`docs/findings.md`](findings.md)).

For the other industries the encrypted hybrid preserves the plaintext decision regardless of weights, so
the metric is FHE/MPC **cost + fidelity**, not task SOTA; lightweight === the trained linear model, so SHAP
stays exact. The benchmark shows the architecture label + parameter count.

The pages have **no lightweight / medium / large** selector either; the CLI's `--model-size` and the
config's `generic_architecture` still exist and produce the rows below: with no use case, the trained linear model at lightweight and random weights on 16, and on 64 and 32, hidden units at medium and large (`tier_architectures` sizes the per-use-case pipeline). The selector scaled invented
architectures - a "tabular fraud MLP" at 18/850/9,538 parameters against a real distilled student of
33 at the time - so the sizes described models nothing had trained. Finance's page has one trained MLP and one switch for it. What survives of the idea: **the measured
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
parties, each its own process, over loopback TCP: Shamir threshold $`t = 1`$, statistical security parameter $`k = 40`$,
64-bit fixed point. "MPC comm" is the bytes the three parties sent, about 15,120 per secure comparison
and the same on every run. "MPC-non-linear" is the protocol's time on the parties' own clocks, about
2.3 ms per comparison here. Each MPC session also spends 0.16 to 0.20 s starting its processes, which
parties standing between requests would not, so that time is left out of the latency and reported as
`mpc_session_overhead_ms`. Loopback has no network delay, so a LAN or WAN adds a round trip per
communication round. Each row's `view_statistical_distance_max` bounds what the key holder's whole
view of one inference gives away, and `mask_statistical_distance_max` ($`9 \times 10^{-8}`$ or less in the rows
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
  Security model column replaced the 0-100 composite, which nothing computes since Round 86: no row,
  signed summary or download carries it; the runs recorded on 2026-11-06, before its removal, still hold its parts
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
client-side (`web/src/lib/clientCompute.ts`) so the public demo shows the same posture with no backend; the benchmark's figures are each tier's own measurement, and the next section says why their ct/pt differ.
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
  about 6146x at 8 features, 12,292x at 4 and 311x at genomics' 158 (269x at the 183 of PGS000072, the score until
  2026-11-26).

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
- of the students, only `satellite_comms.student` records a `git_sha`, since Round 86 gave it a trainer
  of its own, and only `finance_aml.gbm`, `finance_aml.mlp_large` and `finance_aml.gnn_compact` record a digest of
  their training data (`data_digest`);
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
  against the eavesdropper fraction; the MCP agent call; a secure argmax over a shaped network under two garbled-circuit engines, measured; the CBOM of the cryptography the project calls, read against NIST IR 8547 on a date the reader picks; and handshakes of the network layer's kinds,
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
  QSDC secrecy capacity against downlink range (satellite), and two-way QSDC against one-way quasi-QSDC per pulse
  against loss, with four published measurements.
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

#### The finance decision curve

The finance verification card draws net benefit across threshold probabilities for `finance_aml.gbm`, scored in the
tab on the 8,192 rows of its evaluation sample, against treat-all and treat-none (Vickers and Elkin 2006). Acting
when the calibrated risk is at least $`t`$ prices each false alarm at $`t/(1-t)`$ of a caught case:

$$NB(t) = \mu\,\mathrm{TPR}(t) - (1 - \mu)\,\mathrm{FPR}(t)\,\frac{t}{1 - t}$$

The sample draws each class of the test window on its own (164 laundering rows, 2.0%), so the rates are unbiased
and its prevalence is not; $`\mu`$ is the window's own, 2,321 of 1,523,504 (0.152%), which is the case-control
estimator of Pfeiffer and Gail (2020, eq. 11). The score passes through the record's Platt map first, and the
threshold range, 0.002 to 0.05, is the use case's decision context in the config, shared with the BFV scorecard's
fit. Measured on 2026-11-11 against the whole window with the published graph, and again on 2026-11-12 and on 2026-11-13 at the
pins its republishes moved (the record is the last, at `b74d16f`), with the same figures, per 10,000 transactions
([`measurements/finance-decision-curve-2026-11-11.json`](measurements/finance-decision-curve-2026-11-11.json)):

| $`t`$ | whole window | the page's sample, pointwise 95% | weighted by the sample's 2.0% | treat-all |
|---|---|---|---|---|
| 0.002 | 12.46 | 12.13 (11.25 to 13.01) | 165.52 | -4.77 |
| 0.005 | 11.59 | 11.47 (10.51 to 12.43) | 156.97 | -34.94 |
| 0.01 | 10.93 | 10.91 (9.88 to 11.93) | 148.51 | -85.62 |
| 0.02 | 10.11 | 10.38 (9.30 to 11.46) | 140.08 | -188.54 |
| 0.05 | 9.10 | 8.35 (7.17 to 9.53) | 112.11 | -510.28 |

The window lies inside the sample's interval at all 43 thresholds the page draws, at most 1.24 standard errors from
it. The third column is the error the estimator avoids. A perfect model would score 15.23, the window's rate per
10,000; the trainer's operating point, 0.7451 on the raw score, is a calibrated risk of 0.309, above the range. The
teacher on the 32 aggregates that this one replaced scored 5.44 at $`t = 0.01`$
([`measurements/finance-decision-curve-2026-11-06.json`](measurements/finance-decision-curve-2026-11-06.json)).
The map lowers this teacher's Brier score on the validation rows it was fitted on, from 0.00070321 to 0.00061622,
and raises it on the test window, from 0.00077072 to 0.00082539, where laundering is 0.152 percent of rows against
0.113 on validation, and on the test window it lowers the log-loss. Through the map the curve reaches a higher net
benefit than the raw score gives at 25 of the 43 thresholds, by at most 0.42 per 10,000, and a lower one at the other
18, by at most 0.27, so the choice between them does not change what the curve says. The record states the scores
beside the map.

The private-inference figure draws the same curve for the CKKS student, `finance_aml.student`, from the logits it
decrypts on the same 8,192 rows, over the same thresholds at the same rate. Its Platt map is fitted, as the teacher's
is, on the model's own probability, here $`q = \sigma(w^\top z + c)`$, over the validation window
(`student_calibration` in `scripts/train/finance_aml.py`, which a retrain runs; `scripts/finance_student_calibration.py`
applied it to the published weights without retraining). The student classifies at $`q = 0.5`$, where the trainer
folded its tuned threshold into the bias, a calibrated risk of 0.079. Measured the same way, per 10,000 transactions
([`measurements/finance-student-decision-curve-2026-11-11.json`](measurements/finance-student-decision-curve-2026-11-11.json)):

| $`t`$ | whole window | the page's sample, pointwise 95% | weighted by the sample's 2.0% | whole window, isotonic map | whole window, no map |
|---|---|---|---|---|---|
| 0.002 | 12.56 | 12.60 (12.11 to 13.09) | 189.70 | 12.52 | 11.93 |
| 0.005 | 10.83 | 10.86 (10.16 to 11.56) | 180.09 | 11.17 | 8.50 |
| 0.01 | 9.86 | 9.65 (8.72 to 10.57) | 166.47 | 10.05 | 3.84 |
| 0.02 | 8.11 | 7.96 (6.80 to 9.11) | 149.00 | 8.10 | -3.48 |
| 0.05 | 6.69 | 6.11 (4.66 to 7.56) | 120.08 | 6.01 | -13.14 |

The window lies inside the sample's interval at all 43 thresholds, at most 0.79 standard errors from it. Over the
whole window the student is level with the teacher at the bottom of the range, 12.56 against 12.46 at $`t = 0.002`$,
and below it from $`t \approx 0.0022`$, by 2.40 at $`t = 0.05`$; it stays above treat-none across the range, where
the 32-input student it replaced went below it from 0.033. On this student isotonic regression scores a lower Brier
score than Platt and a higher log-loss (0.00134853 against 0.00138016, and 0.00654373 against 0.00620012) and moves
the window curve by at most 0.68 per 10,000, at the top of the range; Platt is kept so that the two curves differ by
the model and not by the map, and the record says so. The calibrated student's Brier skill against the base rate is
$`1 - 0.00138016/0.00152114 = 0.0927`$, and the teacher's $`1 - 0.00082539/0.00152114 = 0.4574`$: as probability
forecasts both beat giving every row the window's rate, the student by much less. The last column, the student's own
probability read as a risk, negative from $`t \approx 0.015`$, is why the map is applied.

#### Which pages encrypt

In the browser, the finance, medical, genomics and DeFi lending pages run node-seal CKKS and Polytune
garbled circuits between notional parties in one tab, and on request the medical, genomics and lending pages run Lattigo's threshold CKKS or BGV, compiled to WebAssembly; the telecom page runs its
student under CKKS on one held-out row with its argmax in a garbled circuit, on the shares of the masked scores, and both the telecom and satellite pages run a CKKS check of each published student they carry, with no garbled circuit in that check. The satellite page's anomaly verdict and SHAP explanation are a separate demo regression,
computed in plaintext on a bundled OPS-SAT-AD segment; its check scores `satellite_comms.opssat_student` on the benchmark's
529 test-partition segments, through the quadratic expansion and the scaler its ONNX graph applies, and the forecaster's student, `satellite_comms.student`, is re-scored under CKKS on its whole published stretch, 8,640 forecasts. The full deployment scores one of those
segments, with the same demo regression, under CKKS (TenSEAL), and reports how far the decrypted
logits are from the plaintext ones. What CKKS evaluates in the browser: on the medical page the
Wisconsin logistic regression, whose logits the 2PC argmax then compares, and the published medical
student's convolution and both dense layers; the fully encrypted square student end to end, in its own
CKKS context (below); the chest X-ray model's own last layer over its 512 pooled features; the telecom student; the full 158-variant genomics
score; on finance, since F7b, the square network end to end on an AMLworld test-window transaction and on the 8,192
shipped test-window rows for its agreement figure (below), and the published MLP and the compact GNN on one held-out
transaction each, their ReLUs and argmax in garbled circuits; on the lending page, each check's margin; and, where a trained QCNN is chosen under CKKS, its input as a linear form. What each step hides, in either tier, is in
[the threat model's table](threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment).

#### The fully encrypted square student

`medical_fl.square_student` (Round 86, 2026-11-19, the owner's choice) runs end to end under CKKS in the tab, with no
layer decrypted in between: a convolution, a square, an average pool and a dense layer, multiplicative depth 3, so
`web/src/lib/clientFHESquare.ts` builds its own node-seal context rather than use the page's (60+40+40+60 bits, two
rescales), which the FHE.org poster describes. It was pre-registered (`references/medical-square-student-preregistration.md`
and its addendum for set A2) and measured by `scripts/train/medical_square_student.py`
(`docs/measurements/medical-square-student-2026-11-19.json`).

| | test AUC, five seeds | exported seed |
|---|---|---|
| square student (conv, BN folded, $`x^2`$, pool, dense 196 -> 2) | $`0.8435 \pm 0.0198`$ | 0.8198 |
| the published student's architecture, same recipe (control) | $`0.8182 \pm 0.0045`$ | - |

| parameter set | $`N`$ | modulus bits | scale | largest logit error, 156 images (TenSEAL) | in node-seal, 8 images | offered |
|---|---|---|---|---|---|---|
| A | 8,192 | 50+30+30+30+50 (190) | $`2^{30}`$ | $`8.2 \times 10^{-3}`$ | - | no: over the pre-registered $`10^{-3}`$ |
| A2 | 8,192 | 48+34+34+34+52 (202) | $`2^{34}`$ | $`3.1 \times 10^{-4}`$ | $`1.5 \times 10^{-5}`$ | yes |
| B | 16,384 | 60+40+40+40+60 (240) | $`2^{40}`$ | $`6.1 \times 10^{-5}`$ | $`6.7 \times 10^{-7}`$ | yes |

Both offered sets are within the HomomorphicEncryption.org standard's 128-bit post-quantum bound for a ternary secret
(202 bits at $`N = 8192`$, 411 at $`N = 16384`$); no decision changed under any set. Every vector is tiled with period
1,024, so the rotate-and-add leaves the logit in every slot and no partial sum (a weight times a pixel) is decrypted:
the depth-3 chain has no level left for the 0/1 mask `clientFHE.ts` uses. At its argmax, the pre-registered operating
point, it flags 66 of the 156 test images malignant (recall 0.738, specificity 0.693; accuracy 0.705, below the 0.731
of always answering benign). `tests/test_r86_square_student_ckks_in_node.py` runs the module under the real node-seal.

#### The 3D nodule student

The same design in three dimensions (Round 86, batch 4, on the owner's choices of 2026-11-24), on LIDC-IDRI chest CT
from TCIA (CC BY 3.0): one 3 x 3 x 3 convolution with four filters, batch norm folded, $`x^2`$, a 2 x 2 x 2 average
pool and a dense layer from $`4 \cdot 14^3`$ to two logits, on a 28 mm cube at 1 mm around a nodule. Pre-registered
before any image was fetched (`references/medical-nodule3d-square-student-preregistration.md`), with an addendum before
any model was trained (how the scans were fetched and verified, the training patients fetched a second time after a
naming bug, and the rule that picks the page's twelve test crops). Every file was checked against TCIA's MD5 and every
series against its image count (`docs/measurements/lidc-stream-2026-11-24.json`); the split is by patient
(1,178 labelled training nodules of 518 patients, 170 validation, 347 test). Record:
`docs/measurements/medical-nodule3d-square-student.json`.

| teacher candidate, out of fold over the training patients | AUC |
|---|---|
| 3D ResNet-18 from scratch (taught: the best of the three) | 0.874 |
| TotalSegmentator's nodule volume, a one-feature logistic regression | 0.786 |
| CT-FM's linear probe on its features averaged over a 64 mm cube | 0.641 |

| | test AUC, five seeds | exported seed |
|---|---|---|
| 3D square student | $`0.907 \pm 0.009`$ | 0.893 (95 % interval 0.848 to 0.933 over test patients) |
| the same network with ReLU (control) | $`0.882 \pm 0.027`$ | - |

| parameter set | $`N`$ | modulus bits | largest logit error, 347 test crops | decisions changed | median time a crop | bytes up, back |
|---|---|---|---|---|---|---|
| A2 | 8,192 | 48+34+34+34+52 (202) | $`3.2 \times 10^{-5}`$ | 0 | 1.7 s | 85.0 MB, 262 kB |
| B | 16,384 | 60+40+40+40+60 (240) | $`1.6 \times 10^{-5}`$ | 0 | 2.2 s | 84.9 MB, 525 kB |

All three pre-registered criteria are met. The exported seed is the best on validation, the rule fixed before any run,
and the lowest of the five on test. CT-FM's probe is reported as measured: its features are averaged over the whole
cube, of which the nodule is a small part, and the pre-registered probe was not tuned after the fact. The card on the
medical page (`Nodule3dCard.tsx`) draws one of twelve test crops as three planes with the exact contribution map and
runs it under CKKS in a Web Worker; in the preview a crop took 1.9 s under A2 and 2.4 s under B, keys included.
NoduleMNIST3D's official-split figures (ResNet-18 3D 0.863, ResNet-50 3D 0.875, auto-sklearn 0.914) are cited for
context and not compared: that split shares scans across its parts and labels differently. Membership inference
against these students is in `docs/measurements/mia-nodule3d-*.json`.

**At each step, as Lee et al. report theirs (Round 86, batch 8).** Pre-registered on the owner's choice of 2026-11-27
(`references/medical-nodule3d-reporting-preregistration.md` and its addendum), computed on the CPU over the same 347
test crops of 148 patients, each figure with a 95 percent interval over 2,000 resamples of the test patients
(`docs/measurements/medical-nodule3d-reporting-2026-11-27.json`):

| step | AUC | AUPRC |
|---|---|---|
| the teacher: the mean of the 3D ResNet-18's five fold networks | 0.941 (0.906 to 0.969) | 0.891 (0.819 to 0.946) |
| the ReLU control, five seeds | 0.882 (sd 0.027) | 0.767 (sd 0.037) |
| the square student, five seeds | 0.907 (sd 0.009) | 0.823 (sd 0.020) |
| the shipped seed (5), in the clear | 0.893 (0.848 to 0.933) | 0.809 (0.719 to 0.890) |
| the shipped seed under CKKS A2 or B, its decrypted logits | 0.893 (0.848 to 0.933) | 0.809 (0.719 to 0.890) |

A model that ranks at random has an AUPRC of 0.314, the test part's malignant share. The distillation costs about
0.03 of AUC against the teacher, whose five fold networks were each trained on three fifths of the training patients;
the square beats the ReLU control at either measure; and encryption costs nothing measurable: under both parameter sets
the decrypted logits give the clear ones' AUC and AUPRC to the sixth decimal, so no malignant-benign pair changed
order (largest logit error $`3.9 \times 10^{-5}`$ under A2, $`1.5 \times 10^{-5}`$ under B). The recomputed students' test AUCs match the record's within two of its
25,942 malignant-benign pairs, as the addendum allows: the record's came from GPU logits, these from the CPU.

#### A fully encrypted finance student, not adopted (F7)

The finance page runs two private paths: the linear student, which CKKS evaluates alone (test AUPRC 0.184, minority-class
F1 0.309), and the MLP, whose ReLUs run in garbled circuits (0.629 to 0.650 over its five seeds). Pre-registered before
any training (`references/finance-square-student-preregistration.md`; the owner's choices of 2026-11-25: one hidden
layer 128 wide, a ReLU control, every test row, the GPU), a network whose only non-linearity is a square, so that CKKS
evaluates it end to end, was trained by the MLP's recipe on its windows and 114 inputs. Record:
`docs/measurements/finance-square-student-2026-11-25.json`.

| | test AUPRC, five seeds | criterion |
|---|---|---|
| square: dense 114 -> 128, batch norm folded, $`x^2`$, dense 128 -> 2 | $`0.500 \pm 0.019`$ | 1, accuracy: at least the control's mean minus 0.02 - **not met** |
| control: the same network with a ReLU | $`0.549 \pm 0.017`$ | 2, no collapse: every square seed at least 0.15 and predicting both classes - met |

So the candidate is not adopted and the page does not change. Reported as pre-registered: the exported square seed
scores 0.521 (95 % interval 0.499 to 0.544 over test rows), and under CKKS, one row a slot, every one of the 1,523,504
test rows decrypts within $`3.1 \times 10^{-6}`$ of the plaintext logits under A2 and $`1.1 \times 10^{-7}`$ under B with
no decision changed (criterion 3 met), the largest plaintext logit 27.4 against A2's bound of $`2^{13}`$. A block of
4,096 rows took 0.6 s to encrypt and 9.9 s to evaluate under A2, with ten node processes running at once, since
node-seal's wasm is single-threaded. The square network is far ahead of the linear student it could replace in the
fully encrypted path (0.500 against 0.184) and behind the same network with a ReLU. On the owner's choice of 2026-11-25,
that comparison is pre-registered as F7b (`references/finance-square-student-page-preregistration.md`) before any of its
seeds was trained: its accuracy margins are stated as chosen after this result, and a new criterion asks for the page's
time and memory in a browser.

#### The square network on the finance page's fully encrypted path (F7b), adopted under A2

Pre-registered before any of its seeds was trained (`references/finance-square-student-page-preregistration.md`; record
`docs/measurements/finance-square-student-page-2026-11-25.json`): F7's network and recipe on new seeds 5 to 9, against
the published linear student on the same test window. F7's record had already given that comparison, so the accuracy
margins were chosen after F7 and say so; the new criterion was the page's own budget.

| criterion | result |
|---|---|
| 1, accuracy: mean test AUPRC at least the linear student's (0.184) plus 0.15, each seed plus 0.10 | $`0.500 \pm 0.018`$ (0.475 to 0.525): met |
| 2, no collapse | met |
| 3, fidelity under CKKS on every test row (1,523,504) | A2 within $`4.2 \times 10^{-6}`$, B within $`1.7 \times 10^{-7}`$, no decision changed: met |
| 4, one block of the shipped rows in a Web Worker in headless Chrome, at most 30 s and 1.5 GB, three runs | A2 (4,096 rows) 17.0 to 17.4 s, 0.087 GB: met; B (8,192 rows) 35.5 to 35.6 s: not met |

So the network is adopted under A2 and B is not offered. The exported seed, 9, scores 0.497 (95 % interval 0.476 to
0.519 over test rows). The browser harness (`web/scripts/measure_square_tabular_browser.mjs`) was smoke-tested once on
F7's exported seed after the pre-registration was committed and before F7b's seeds were trained; its A2 and B times
already sat on either side of the budget, which the record could not have changed, since time does not depend on the
seed.

On the owner's choices of 2026-11-25, after the result, the network replaces the linear student on the finance page,
and it ships with the site (`web/public/data/finance/square_student_weights.json`), not from the Hub. A Platt map was
fitted on the validation window after the run (`scripts/train/finance_square_student_page.py export`; Brier 0.00166 to
0.00109, CORP MCB $`6.5 \times 10^{-4}`$ to $`8.7 \times 10^{-5}`$, AUPRC unchanged, 5,862 test rows tied at the logit
clip) and is not part of the pre-registration. On the page:

- the private-inference figure scores the 8,192 shipped test-window rows through it, two A2 blocks in two Web Workers,
  about 17 s in this project's Chrome, and its decision curve reads the decrypted logits through that map;
- the pipeline's transaction is an AMLworld test-window row, not a PaySim row: the network evaluates it under CKKS in a
  Web Worker (a block of its own, about 17 s, since the work does not shrink with the rows), the garbled-circuit argmax
  runs over its decrypted logits, and the SHAP bars are its exact quadratic Shapley values
  (`web/src/lib/quadraticShap.ts`, held to the definition by brute force in `tests/test_explainability.py`). The finance
  QCNN reads the same transaction at its eight inputs since Round 86's batch 6 (a PaySim transaction of its own until
  then, in PaySim's four columns), and the run says so;
- the accuracy-against-cost plot prices the network's batch in place of the linear student's.

The linear student stays published on the Hub, and its record stays the comparison the BFV scorecard and
`scripts/measure_decision_curve.py` read.

#### The 5G-NIDD detector's CKKS student: telecom's encrypted path on real traffic

On the owner's choice of 2026-11-25, pre-registered before any student was trained, with two addenda before training
([`references/telecom-nidd-ckks-student-preregistration.md`](../references/telecom-nidd-ckks-student-preregistration.md);
`scripts/train/telecom_nidd_student.py`, `docs/measurements/telecom-nidd-student-2026-11-25.json`): a temperature-scaled
multinomial logistic regression on the 36 numeric 5G-NIDD features, standardised on the teacher's training split and
clipped to $`[-20, 20]`$. The distilled arm was dropped on the owner's choice of 2026-11-26, since it needed a download
the owner declined. On the 233,591 test flows: macro-F1 0.974 (criterion 1 asks 0.95; the gradient-boosted teacher
scores 0.997 on all 46 features, the repository's earlier linear probe 0.969), expected calibration error 0.0042 after a
temperature of 0.840. Under CKKS in the browser, on the 2,048 test flows the page ships (stratified, CC BY 4.0): 32
ciphertexts, about 7.4 s, the largest logit error $`2.3 \times 10^{-7}`$, no decision changed (criterion 2,
`tests/test_r86_nidd_student_under_ckks.py`). The telecom page draws it as a figure of its own beside the QKD student's,
which reads synthetic telemetry.

#### 5G-NIDD's two base stations, federated: the first station's own model is the best

On the owner's choice of 2026-11-26, pre-registered before any model was trained
([`references/telecom-nidd-federated-preregistration.md`](../references/telecom-nidd-federated-preregistration.md);
`scripts/train/telecom_nidd_federated.py`, `docs/measurements/telecom-nidd-federated-2026-11-26.json`). The corpus was
captured at two base stations, whose files the CSV concatenates; each station's fit, validation and test rows are the
student's that come from its file. Each station alone and both pooled run the student's own procedure; the federation
runs FedAvg for 40 rounds of 20 local L-BFGS-B iterations under the page's pairwise masks (`SecureAggregator`, the
browser's construction), each station weighted by its fit rows (0.478 and 0.522), the penalty and the temperature
chosen from masked sums of the stations' validation losses: 270 masked messages, every one checked to cancel exactly.
The stations exchanged only their row counts and each feature's mean and deviation in the clear, which reproduce the
student's standardisation: the centres within $`3.5 \times 10^{-8}`$ of a deviation, the deviations within
$`1.8 \times 10^{-7}`$ of themselves.

| Model | Station 1's test flows | Station 2's | Mean (FLamby's metric) |
|---|---|---|---|
| Station 1 alone | 0.986 | 0.983 | 0.985 |
| Station 2 alone | 0.935 | 0.983 | 0.959 |
| Federated | 0.959 | 0.982 | 0.971 |
| Pooled (the student) | 0.965 | 0.983 | 0.974 |

Macro-F1 over the nine classes. Criterion 1 fails: the federated model is not above each station alone, because station
1's own model is the best by the mean. It carries to station 2's flows as well as station 2's own model does, while
station 2's model does not carry back (SYNFlood 0.83, SYNScan 0.89, UDPScan 0.89 on station 1's flows); a single linear
model trained on both, federated or pooled, scores between the two on station 1's flows (SYNFlood 0.84 and 0.88). The
gradient-boosted teacher shows the same asymmetry across stations (0.995 one way, 0.902 the other). Criterion 2 holds:
the federation is 0.003 below pooling, and its pooled training objective 0.004 above the pooled fit's. Criterion 3
holds: under CKKS in the browser, on the student's 2,048 held-out flows (955 from station 1, 1,093 from station 2), 32
ciphertexts in about 7.4 s, the largest logit error $`2.1 \times 10^{-7}`$, no decision changed, and the accuracy by
station is the plaintext one (98.74 and 98.44 percent). The page's card states the comparison and its reason in the
record's figures, and after a run replays the last round's masked sum with fresh X25519 keys, 333 of 333 coordinates
equal to the shipped global model bit for bit. What it does not show: with two stations each can compute the other's
update from the global model and its own, so the masks hide the updates from the aggregator only; the mask seeds are
not post-quantum; no differential privacy is applied.

#### The finance QCNN on AMLworld, pre-registered, and on the page since Round 86's batch 6

On the owner's choice of 2026-11-25, pre-registered before any input was scored
([`references/finance-qcnn-amlworld-preregistration.md`](../references/finance-qcnn-amlworld-preregistration.md)):
eight of the 114 AMLworld inputs chosen on the fit window by $`|\mathrm{AUC} - 0.5|`$ (`format_risk` leads at 0.379),
the tree circuit's recipe on seeds 0 to 4, every laundering transaction of the fit window and four others each to fit
on, and the whole validation and test windows (`scripts/data/aml_qcnn_inputs.py`, the trainer's `finance_amlworld`
corpus; `docs/measurements/finance-qcnn-amlworld-2026-11-26.json`). Seed 0 at eight wires ran on the CPU; the other nine
ran on the GPU on the owner's choice of 2026-11-26.

| Wires | Test AUPRC by seed | Mean | Logistic regression, same inputs | Pauli surrogate |
|---|---|---|---|---|
| 8 | 0.267, 0.391, 0.262, 0.117, 0.029 | 0.213 | 0.370 | 0.111 |
| 4 | 0.064, 0.038, 0.037, 0.037, 0.073 | 0.050 | 0.031 | 0.042 |

Criterion 1 holds: every seed's test AUPRC is at least twice the test window's prevalence ($`0.0030`$; the lowest is
$`0.029`$). At eight wires the circuit trails logistic regression on the same inputs on average and varies widely by
seed; at four it is ahead of both comparators. The seed best on validation at eight wires is seed 3 (validation F1
0.179, test minority F1 0.193, the best of the five, test AUPRC 0.117). Criterion 2, the page's clear contraction
against the state vector, is checked on the seed the owner chooses to ship, and the PaySim QCNN stays on the page
until then. For scale, not like for like: F7b's square network on all 114 inputs scores 0.500.

That run selected each checkpoint and seed by a validation F1 at a logit of 0, a threshold the four-to-one fit sample
sets, so the comparison that followed re-ran the circuit selected by validation average precision, over twenty seeds,
beside the comparators the same inputs allow
([`references/finance-qcnn-amlworld-comparison-preregistration.md`](../references/finance-qcnn-amlworld-comparison-preregistration.md);
the finance page's comparison card states every arm with its interval). The seed best on validation, seed 6, scores test
average precision 0.442, at least twice the prevalence, and the page's tree contraction equals the state vector within
$`2.1 \times 10^{-6}`$ on all 1,523,504 test transactions with no decision changed, so criterion 1 holds, and on the
owner's choice of 2026-11-26 it is the finance page's QCNN (`scripts/export_finance_qcnn_amlworld.py`; the card is
[`models/qcnn_finance/README.md`](../models/qcnn_finance/README.md)). Its served file folds the fit window's
standardisation into its own moments, so the page feeds raw inputs, and the validation-best threshold into its bias:
minority-class F1 0.509 on the test window, against 0.497 for logistic regression on the same encoded inputs at its
own threshold.

The comparison on the test window (average precision with its 95 percent stratified bootstrap interval;
`docs/measurements/finance-qcnn-amlworld-comparison-2026-11-26.json`, written by
`scripts/train/finance_qcnn_comparison.py --record`):

| Model | Kind | Test average precision (95 % interval) |
|---|---|---|
| QCNN-1, the seed best on validation (seed 6) | simulated quantum circuit | 0.442 (0.421 to 0.463) |
| Logistic regression on the circuit's tanh-encoded inputs - not an arm | classical | 0.370 (0.347 to 0.392) |
| Quantum kernel SVM: the product encoding's fidelity | quantum kernel, computed classically | 0.330 (0.309 to 0.352) |
| QCNN-1, the 20 seeds' mean probability | simulated quantum circuit | 0.296 (0.276 to 0.315) |
| QCNN-2 (quantile encoding, encoded twice), the 20 seeds' mean | simulated quantum circuit | 0.275 (0.257 to 0.295) |
| Gradient-boosted trees | classical | 0.270 (0.250 to 0.292) |
| RBF support vector machine | classical | 0.262 (0.244 to 0.282) |
| Matrix-product-state classifier | classical tensor network | 0.256 (0.237 to 0.276) |
| Quantum kernel SVM: the ZZ feature map's fidelity | quantum kernel, computed classically | 0.195 (0.179 to 0.214) |
| QCNN-1 without its entangling gates, the 5 seeds' mean | simulated quantum circuit | 0.140 (0.130 to 0.154) |
| Gradient-boosted trees on the whole fit window | classical | 0.116 (0.105 to 0.129) |
| Quantum kernel SVM: the ZZ map's projected kernel | quantum kernel, computed classically | 0.109 (0.097 to 0.123) |
| Logistic regression on the standardised inputs, the whole fit window | classical | 0.036 (0.035 to 0.037) |
| Logistic regression on the standardised inputs | classical | 0.035 (0.034 to 0.036) |

Over its twenty seeds the circuit scores $`0.253 \pm 0.090`$, so the shipped seed is the
rule's pick, not the method's typical figure. The row that is not an arm, logistic regression on the circuit's own
tanh-encoded inputs, is the trainer's baseline as the served weights file records it, set beside the arms on the owner's
choice of 2026-11-26 after the record was first written (the arms' logistic regression reads the standardised inputs):
the shipped seed is ahead of it (paired difference 0.059 to 0.086), and it is ahead of the twenty seeds' mean
probability, whose interval lies below its own. Without its entangling gates the circuit scores 0.140 against
0.296; Bowles, Ahmed and Schuld's benchmark of quantum classifiers (arXiv:2403.07059, 2024) found
removing entanglement often as good or better on their small tasks, and classical models ahead overall; here removing
it lowers the figure, and one classical model, logistic regression on the encoded inputs, is ahead of the seed average,
while no classical arm scores above it. The circuit's largest Schmidt rank, every seed, cut and layer, traced in double
precision on 1,000 test rows, is 8 of the 16 an eight-wire state can reach.

#### The DeFi QCNN on Aave v3 liquidations, pre-registered (Round 86, batch 8)

The DeFi page's private check computes Aave V3's own rule, a health factor below 1, so it has no learned classifier for
a QCNN to reproduce; on the owner's choices of 2026-11-26 and 2026-11-27 this asks a question beside it instead,
pre-registered before the archive was downloaded
([`references/defi-qcnn-aave-v3-preregistration.md`](../references/defi-qcnn-aave-v3-preregistration.md), with an
addendum after the download and before any label, and one moving its second arm to the GPU). A borrow on Aave v3
(Polygon) is positive when its account is liquidated within 30 days, from FinSurvival's Borrow-to-Liquidated table in
Spadea and Seneviratne's archive (CC BY 4.0), with windows by time (70, 15 and 15 percent of the borrows) and eight of
the 80 numeric inputs chosen on the fit window alone by $`|\mathrm{AUC} - 0.5|`$. The arms follow the finance
comparison's recipe: the tree circuit, three layers, 120 epochs and the checkpoint by validation average precision, 20
seeds at eight wires and 20 at four, and 5 without the entangling gates, all on the GPU; beside them logistic regression,
gradient-boosted trees, logistic regression on the circuit's own Pauli products, and, not like for like, the booster on
all 80 inputs and the whole fit window (`scripts/data/defi_qcnn_inputs.py`, `scripts/train/defi_qcnn_comparison.py`;
the record `docs/measurements/defi-qcnn-aave-v3-2026-11-27.json`).

The eight inputs are counts and sums of the account's deposits and liquidations, so a model on them means something only
if they describe the account before the borrow. FinSurvival's appendix B.4 describes them as the account's transactions
in the past; Spadea and Seneviratne's paper does not define them. Checked on the table after the pre-registration, and
not one of its criteria (`scripts/data/check_defi_inputs_are_prior_history.py`;
`docs/measurements/defi-inputs-prior-history-2026-11-27.json`): along each account's borrows in time, no count of
liquidations or deposits ever decreases; after each of the 27,307 liquidated borrows that the same account follows with
another borrow, that later borrow counts at least one more liquidation, so no borrow's count already included its own
outcome; and at an account's first borrow, a liquidation is already counted for 0.19 percent of the accounts liquidated
later and 0.03 percent of the others, where a count over the whole history would be positive for every account
liquidated later.

The comparison, the validation window's average precision beside the test window's with its 95 percent stratified
bootstrap interval; 1.08 percent of the validation window's 182,350 borrows are liquidated within 30 days and 1.24
percent of the test window's 182,351:

| Model | Kind | Validation | Test (95 % interval) |
|---|---|---|---|
| Gradient-boosted trees on all 80 inputs, the whole fit window (reference) | classical | 0.0885 | 0.1714 (0.1555 to 0.1879) |
| QCNN, four wires: the 20 seeds' ensemble | quantum circuit, simulated classically | 0.1559 | 0.0812 (0.0749 to 0.0881) |
| QCNN, eight wires: the seed best on validation (seed 19) | quantum circuit, simulated classically | 0.2723 | 0.0642 (0.0592 to 0.0699) |
| QCNN without entanglement: the 5 seeds' ensemble | product circuit, simulated classically | 0.2013 | 0.0575 (0.0526 to 0.0630) |
| QCNN, eight wires: the 20 seeds' ensemble | quantum circuit, simulated classically | 0.2048 | 0.0411 (0.0391 to 0.0433) |
| Logistic regression on the same inputs | classical | 0.1451 | 0.0322 (0.0304 to 0.0341) |
| Gradient-boosted trees on the same inputs | classical | 0.1949 | 0.0235 (0.0214 to 0.0249) |
| Logistic regression on the circuit's Pauli products | classical surrogate | 0.0372 | 0.0144 (0.0134 to 0.0155) |

Both criteria hold. The seed best on validation at eight wires, seed 19, scores test average precision 0.0642 (0.0592 to
0.0699), above the $`0.0247`$ that twice the prevalence asks, and the page's tree contraction of it equals the state
vector within $`2.5 \times 10^{-6}`$ on all 182,351 test borrows with no decision changed
(`scripts/check_defi_qcnn_export.py`). What the figures do not carry is a ranking. Every model on these eight inputs
scores lower on the later test window than on validation, the shipped seed from 0.272 to 0.064, logistic regression from
0.145 to 0.032 and the booster from 0.195 to 0.023; only the booster on all 80 inputs scores higher, from 0.089 to 0.171.
The test column ranks these models under a shift between the windows, not the methods, and no figure here says a quantum
circuit is better at this task. Over its 20 seeds the eight-wire circuit scores $`0.045 \pm 0.024`$ and their mean
probability 0.041, so the shipped seed is the rule's pick, not the method's typical figure. At four wires the 20 seeds'
mean probability scores 0.081 and, without its entangling gates, the circuit's five seeds' scores 0.058, both above the
eight-wire ensemble's: Bowles, Ahmed and Schuld's benchmark of quantum classifiers (arXiv:2403.07059, 2024) found
removing entanglement often as good or better on their small tasks. Every circuit here acts on at most eight wires, so
its state is at most 256 amplitudes and is simulated exactly on a classical computer; no quantum advantage is claimed.

On the page, `DefiQcnnCard.tsx` below the lending panels reads `web/public/data/defi/qcnn_comparison.json` and
`web/public/data/defi/defi_qcnn.json` (`scripts/export_defi_qcnn.py`: the seed's weights and tree network, and 4,096 of
the test window's borrows drawn uniformly at seed 0, each its eight standardised inputs and its label, no account or other
identifier), and re-scores that sample in the tab with the page's own contraction: average precision 0.0803 over its 51
liquidated borrows, which the tab reproduces within $`10^{-9}`$ (`tests/test_r86_defi_qcnn.py`). The pre-registration
left three choices to the owner after these figures; on 2026-11-27: seed 19 ships, the card is shown open, and nothing
goes to the Hugging Face Hub, since a Hub card would first need the inputs' EDA and calibration and the test figure is
weak and tied to the shift between the windows.

#### Calibrated thresholds for the satellite forecaster, not adopted

On the owner's choice of 2026-11-25, pre-registered before any threshold was fitted
([`references/satellite-threshold-calibration-preregistration.md`](../references/satellite-threshold-calibration-preregistration.md)):
the shipped CKKS student's forecasts on ESA-ADB's Mission1 lightweight channels, its errors smoothed as Telemanom
smooths them, and three rules that turn them into flags, each fitted on the validation months alone and scored once on
the test period's 29 anomalies (`scripts/measure_satellite_thresholds.py`,
`docs/measurements/satellite-thresholds-2026-11-25.json`).

| Rule | Corrected event-wise $`F_{0.5}`$, anomalies | tp / fp / fn | Share of test steps flagged |
|---|---|---|---|
| Telemanom's NDT with ESA-ADB's pruning (the page's) | 0.940 | 22 / 0 / 7 | 0.09 % |
| POT, $`\alpha = 10^{-4}`$ (primary) | 0.260 | 4 / 8 / 25 | 0.65 % |
| Weighted conformal, $`\alpha = 10^{-4}`$ (primary) | 0.044 | 7 / 182 / 22 | 1.22 % |

At $`\alpha = 10^{-3}`$ and $`10^{-5}`$ the extreme-value rule scores 0.228 and 0.270 and the conformal one 0.001 and
0.140. The NDT arm reproduces the student's record (0.93977). A threshold fixed on three months does not carry across
seven years of telemetry whose nominal level drifts, which is what Telemanom's windowed, pruned threshold is built for;
the extreme-value pipeline this compared against recomputes its threshold monthly (arXiv:2609.10017), and this did not.
None of the calibrated rules approaches NDT on the primary metric, so the page's rule is unchanged; whether the page
shows this comparison is the owner's choice, as the pre-registration says. VUS-PR of the same score, which no threshold changes, is 0.656 (VUS-ROC 0.980, window 97).

#### The PSI among banks under MPC

The finance page runs a private set intersection among three to five banks as one garbled circuit under Polytune, in a Web Worker (`wasm/qsh-mpc`, `web/src/lib/clientMultiPsi.ts`). Each bank brings eight generated accounts, hashed to 32 bits, and three are shared by construction. The circuit costs $`k((n-1)k \cdot 31 + (n-1)(k-1) + (n-2) + 32)`$ AND gates for $`n`$ banks of $`k`$ accounts. Measured in a browser tab on 2026-11-08, on a build of this tree, while other work ran on the machine:

| banks | AND gates | traffic | messages | wall-clock |
|---|---|---|---|---|
| 3 | 4,344 | 30.5 MB | 1,262 | 2.6 s |
| 4 | 6,392 | 88.2 MB | 3,411 | 7.4 s |
| 5 | 8,440 | 191.6 MB | 6,884 | 15.4 s |

Traffic grows much faster than the circuit because WRK17b authenticates each AND gate between every pair of parties. Every result is compared with the intersection in the clear, which is possible only because all the banks are in one tab.

#### The CNN student under MPC

The published medical student, on one held-out BreastMNIST image per run, with no activation in either party's hands: CKKS for the convolution and both dense layers, a mask over every slot before the client decrypts, Polytune garbled circuits for the 800 ReLUs and the argmax, and average pooling on each
party's own share ([what it hides and what it does not](threat-model.md#the-medical-cnn-under-mpc)). It runs in a Web Worker. Measured on 2026-11-08 under Node on this project's machine, over all 156 images of the official test split:

| | |
|---|---|
| class from the 2PC argmax against the clear network | the same on 156 of 156 |
| accuracy on the 156 images | 0.7692, the recorded figure |
| malignant images detected | 24 of 42, as the page's clear evaluation |
| largest logit error, shares against the clear network | $`4.9 \times 10^{-4}`$; the smallest logit gap is 0.018 |
| statistical distance of everything masked, per run | $`2^{-22.5}`$ to $`2^{-24.2}`$ |
| AND gates | 69,167: 800 ReLUs at 86 each, and the 2-class argmax's 367 |
| garbled-circuit traffic | 174.4 MB in 15 batches and the argmax |
| CKKS | 28 rotations and 6 conjugations, 3.5 MB of ciphertexts, 14.7 MB of keys |
| wall-clock, median | 17.1 s: garbled circuits 16.9 s, CKKS 0.18 s |

A ReLU costs 86 AND gates at a 20-bit ring and a 48-bit output mask because its adders are written bit by bit, one AND per bit, where Garble's own `+` costs three. Compiling all 800 at once took 75 s and 4 GB natively, so they run in batches of 56, compiled once.

The weights did not change. Retraining to a polynomial activation, as Flash does (arXiv:2401.16732), was rejected for this path: Flash's protocol squares a masked integer in BFV, and with CKKS reals and a $`2^{30}`$ mask the square is about $`2^{60}`$, whose decode error is about 700.

#### The QKD key blocks

Every BB84 key a run reports, on the fibre and from the satellite, is the composable finite key of a
block, computed by the function the run calls; E91 keeps its asymptotic count, labelled, and so does
the constellation on the Network tab. Every key rate on the site counts the
pulses sent; only signal pulses carry key. These are figures of the channel model, not of a link: no
satellite, ground station or quantum channel is involved.

- **Fibre block** - the use-case pages, the two-party demo and Validations page, and the Python API
  report the same key, because the model (`run_bb84` / `bb84()`) computes it. By default the block is
  a privacy-amplification block of $`10^{7}`$ key-basis detections, the size Rusca et al.'s Table I uses and
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
  decoy's errors, which are tighter than the signal's, and $`\tau_0`$ on the error term of (A16) only.
  The three transcription errors found while checking them against the papers are recorded with the
  Rusca et al. entry in [`references/REFERENCES.md`](../references/REFERENCES.md). The shipped fibre
  needs $`5.35 \times 10^{6}`$ pulses before any key under Hoeffding, and $`3.60 \times 10^{6}`$ under the default statistics.
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
  basis the model gives 0.54 of the 629,000 under Hoeffding and 0.66 under Chernoff, 1.34 of the final
  key, and on both bases 2.67 of it (829,658 bits). Their session is counted from the 28 deg where their
  stable tracking started to 1,100 km on the way down (Round 86; it was 1,100 km on both sides, which
  gave 0.56, 0.69, 1.40 and 2.80). So the drop from the pass's asymptotic key
  (462,290 bits) to the one-basis finite key is mostly the basis convention, 95 percent of it under
  Chernoff and 54 percent under Hoeffding, not the statistics; keyed as the flights keyed theirs, the
  model exceeds both published keys. On Jinan-1's four published passes the model's one-basis
  whole-pass key is within a factor of 2.5 of the published key under Hoeffding, and 1.35 to 2.62
  times it under the default; Jinan-1 computes its keys on 500-kbit blocks inside a pass, which a
  whole-pass key is not held to.
- **Jinan-1's pass of 25 September 2022, second by second (Round 86, batch 6, post hoc).** From the paper's source
  data (Zenodo 14732295, CC BY 4.0: Fig. 4b's counts and slant range, Fig. 4c's QBER and key), 353 link seconds,
  background off, against the `jinan1_2022_jinan` profile at each second's range. The comparison was pre-registered
  ([`references/satellite-jinan1-pass-preregistration.md`](../references/satellite-jinan1-pass-preregistration.md)),
  but its windows failed, because Fig. 4c's QBER is one value per key block; the windows the satellite card uses were
  chosen after those figures were seen (its addendum 2), and the card says so. The profile's one constant, the
  station efficiency 0.1683, against the data's fit of 0.1667: $`-0.04`$ dB, so it stays. After that fit the
  per-second residuals deviate by 0.29 dB with a trend of $`+0.07`$ dB per air mass, and the fall with elevation is
  within 0.15 and 0.26 dB of the model's at the link's two ends; over 18 choices of the background window, the link
  threshold and the model's own background term the mean residual stays within 0.07 dB. QBER over the key blocks:
  1.04 percent measured, 0.90 percent for the model's signal state. The model's finite key over the simulated pass
  is 1.38 times the pass's 406,784 bits, before the 800-ps gate (about 80 percent of the signal kept) and the 10
  percent of detections sampled for the QBER, which the second analysis adds
  ([`references/satellite-jinan1-pass-v2-preregistration.md`](../references/satellite-jinan1-pass-v2-preregistration.md),
  committed before any of its figures).

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
- **CUDA-Q in single precision is about 10 times faster than in double from 24 qubits up** (3.1 times at 22, 14.6 at 24, 9.8 to 10.3 from 26 to 30), more than halving the memory traffic alone
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
128-bit security). OpenFHE 1.6.1 runs its own (depth 4, scale $`2^{50}`$, ring dimension 16384, 128 slots), so its column is
not the same computation. A GPU time includes a synchronisation before the clock stops: GPU work is asynchronous, and without
one a timer reads the launch.

| operation | TenSEAL (CPU) | OpenFHE (CPU, own parameters) | HEonGPU (GPU) | HEonGPU against TenSEAL |
|---|---|---|---|---|
| dot product, 33 values | 5.91 | 15.31 | 0.215 | 27x |
| dot product, 256 values | 7.62 | not run: 128 slots | 0.274 | 28x |
| encrypt 33 values | 2.59 | 4.04 | 0.104 | 25x |
| plaintext multiply | 0.84 | 1.06 | 0.040 | 21x |
| ciphertext add | 0.032 | 0.085 | 0.0075 | 4.3x |
| decrypt | 0.44 | 2.99 | 0.052 | 8.5x |
| mask every slot (a fresh encryption, added) | 2.14 | 5.05 | 0.114 | 19x |

- **The GPU is 19 to 28 times faster on the operations with arithmetic in them**, and 19.5 times on the hybrid's whole linear
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
    parallel loop waits for its slowest thread. At 6 threads it held at 41 to 54 ms on OpenFHE 1.5.1, and the table is at 6.
  - *OpenFHE moved to 1.6.1 (Round 86), and its column was measured again:* the 33-value dot product reads 15.31 ms
    against 41.63 on 1.5.1, the same command on a quiet machine, twice, within 1 percent. The other backends' columns
    came out within 6 percent of their earlier figures.
- **A GPU is not where this project's inference runs.** The demo's pages encrypt in the visitor's browser (node-seal), and the
  full deployment's image has no GPU code. This is the full deployment's hybrid on a machine that has one.

## The published graphs on WebGPU beside WASM, measured

Every page with a published ONNX graph lets a visitor pick WebGPU for it (the Model backend select), and until Round 86's
batch 8 only the medical page had that measured. Each use-case page now carries a card, "WebGPU beside WASM, on this
page's fp32 graphs" (`web/src/components/BackendCompareCard.tsx`), that runs a model's fp32 graph in the tab on WASM and
then on WebGPU over every row of its published evaluation sample, compares every output value and each row's decision by
the sample's own rule, and times both (`compareBackends` in `web/src/lib/onnxInference.ts`, the rule and the tolerance in
`web/src/lib/backendCompare.ts`). Within tolerance means every value within $`10^{-3}`$ of the WASM run's and no decision
changed beyond a near tie; the tolerance was stated before any figure was measured.

- **fp32 only, by the owner's choice.** The page loads a model's int8 graph where one ships, and ONNX Runtime Web's WebGPU
  provider registers no kernel for an int8 graph's quantised nodes, so those nodes would run on the CPU and the time would
  describe neither backend. An int8 graph is compared as its fp32 twin, pinned at the same revision and checked against its
  SHA-256 as every graph is; the int8 graph stays on WASM.
- **An `ai.onnx.ml` graph is not compared** (the tree ensembles and the ML-domain linear students): the WebGPU build
  registers none of its operators, so it runs on WASM only, and the card says so on its row.
- **The WebGPU session is opened with WebGPU alone,** so a backend that cannot start fails the comparison instead of opening
  a WASM session that would be timed as WebGPU. Nodes the WebGPU provider has no kernel for still run on the CPU, one by
  one, as ONNX Runtime places them; which nodes those are is not visible to the page.

On a local client-only build of this branch in the owner's Chrome (NVIDIA RTX PRO 6000 Blackwell, as the browser names
the adapter; cross-origin isolated), 2026-11-27, one comparison per model; times are medians of three runs over every row
and of five on one row, after a first run, in milliseconds:

| page | model (graph) | rows | largest difference (largest value) | decisions changed | all rows, WASM / WebGPU | one row, WASM / WebGPU |
|---|---|---|---|---|---|---|
| finance | MLP on per-transaction features | 8,192 | $`3.81 \times 10^{-6}`$ (13.7) | 0 | 4.1 / 6.6 | 0.025 / 2.9 |
| finance | IBM Multi-PNA+EU, IBM's protocol (a graph sample) | 16 seeds | $`1.34 \times 10^{-5}`$ (25.0) | 0 | 50.2 / 30.8 | none: one graph |
| finance | IBM Multi-PNA+EU, causal protocol (a graph sample) | 16 seeds | $`7.63 \times 10^{-6}`$ (17.4) | 0 | 47.0 / 28.5 | none: one graph |
| medical | ResNet-18 on BreastMNIST (the fp32 twin) | 156 | $`1.72 \times 10^{-5}`$ (14.1) | 0 | 344 / 5.6 | 2.4 / 7.0 |
| medical | ResNet-50 on BreastMNIST | 156 | $`4.67 \times 10^{-5}`$ (114) | 0 | 653 / 5.9 | 4.8 / 4.1 |
| medical | CKKS-evaluable CNN student | 156 | $`1.67 \times 10^{-6}`$ (4.51) | 0 | 0.59 / 2.8 | 0.040 / 0.69 |
| medical | fully encrypted CNN student (square) | 156 | $`1.67 \times 10^{-6}`$ (5.54) | 0 | 0.56 / 1.1 | 0.12 / 1.7 |
| medical | ResNet-18 on ChestX-ray14, FedAvg (the fp32 twin) | 16 | $`3.10 \times 10^{-6}`$ (5.01) | 0 | 336 / 6.2 | 21.7 / 4.4 |
| satellite | Telemanom LSTM forecaster (the fp32 twin) | 128 windows | $`2.74 \times 10^{-6}`$ (2.59) | none: a forecast | 39.1 / 16.2 | 1.2 / 12.5 |
| telecom | CKKS-linear IDS student | 600 | $`3.81 \times 10^{-6}`$ (24.0) | 0 | 0.13 / 0.57 | 0.030 / 0.53 |
| telecom | federated 5G-NIDD detector | 2,048 | $`3.81 \times 10^{-5}`$ (259) | 0 | 1.9 / 0.71 | 0.10 / 0.55 |

- **Every graph agrees within the tolerance, and no decision changed.** The largest difference is
  $`4.67 \times 10^{-5}`$, on ResNet-50, whose largest value is 114.
- **Over a whole sample, WebGPU is the faster backend for the three ResNets,** by 54 to 111 times here (ResNet-50: 653
  against 5.9 ms), for the graph networks and the LSTM by 1.6 to 2.4 times, and for the federated detector's 2,048 rows by
  2.7 times. WASM is faster for the finance MLP's 8,192 rows (4.1 against 6.6 ms), the two medical CNN students and the
  telecom student, and on one row for every model but ResNet-50 and the ChestX-ray14 ResNet: a WebGPU run pays for moving
  the input to the GPU and the result back, which a small graph or a single row does not repay.
- **One comparison per model, so the times are indicative.** The telecom student's WebGPU median over every row read 161 ms
  on a first comparison and 0.57 ms on a second, minutes apart; the differences and the decisions were the same both times.
- **On the deploy** the card runs the same way; its figures are in checklist CS23.

## Secret-shared matrix multiplication on the GPU, measured

hpmpc (MIT), the MPC framework under PIGEON, multiplies two secret-shared $`N \times N`$ matrices twenty times under its
Trio protocol, three parties on loopback sharing this machine's GPU, either with its tiled CPU loop or with a CUTLASS
GEMM on the GPU (`scripts/bench_hpmpc_gpu.sh`,
[`measurements/hpmpc-gpu-2026-11-06.json`](measurements/hpmpc-gpu-2026-11-06.json)). No stage or page runs it. Wall
time of party 0, the median of three runs (one for the CPU at 4096), each started with the 1-minute load under 4:

| $`N`$ | CPU | GPU | CPU / GPU | sent by party 0 |
|---|---|---|---|---|
| 128 | 0.023 s | 0.319 s | 0.07 | 1.3 MB |
| 256 | 0.084 s | 0.318 s | 0.26 | 5.2 MB |
| 512 | 0.616 s | 0.776 s | 0.79 | 21 MB |
| 1024 | 4.75 s | 0.685 s | 6.9 | 84 MB |
| 2048 | 38.6 s | 1.98 s | 19.5 | 336 MB |
| 4096 | 316 s | 5.75 s | 54.9 | 1,342 MB |

The GPU pays from $`N = 1024`$; below it a fixed cost of about 0.3 s, there at every size of the GPU's runs, decides.
The parties send the same volume either way, because the GPU replaces only the local products. The GPU path is checked
before it is timed: hpmpc's matrix-multiplication test passes with the GPU and fails with it hidden.

Built at hpmpc `155c935` with one line changed in the script's clone: the benchmark's GPU branch calls `T::GEMM` where
no `T` is declared, at that commit and at upstream's head, so it does not compile; the clone calls the share type's
`S::GEMM`. CUTLASS v4.8.1, headers only, for `sm_120`. The matrix multiplication's GPU path links the GEMM object
alone.

### The convolution layer on the GPU, checked and measured

The same protocol runs one flexNN convolution layer on three paths: the CPU; `im2col` on the CPU with the GEMM above on
the GPU; and the whole convolution on the GPU, CUTLASS's implicit-GEMM convolution (`PARTS=conv
scripts/bench_hpmpc_gpu.sh`,
[`measurements/hpmpc-gpu-conv-2026-11-06.json`](measurements/hpmpc-gpu-conv-2026-11-06.json)). hpmpc's source labels its
three layers CryptGPU's figure 1, a to c: function 33 is an $`n \times n \times 3`$ input under an $`11 \times 11`$
kernel to 64 channels at stride 4, function 34 a batch of $`k`$ such $`32 \times 32 \times 3`$ inputs, and function 35 an
$`n \times n \times 512`$ input under a $`3 \times 3`$ kernel to 512 channels. Wall time of party 0, median of three
runs, each started with the 1-minute load under 4:

| layer | CPU | GPU GEMM | GPU convolution | CPU / GPU GEMM | CPU / GPU convolution | sent by party 0 |
|---|---|---|---|---|---|---|
| 33, $`n = 64`$ | 0.042 s | 0.241 s | 0.265 s | 0.18 | 0.16 | 1.2 MB |
| 33, $`n = 128`$ | 0.153 s | 0.532 s | 0.276 s | 0.29 | 0.56 | 4.9 MB |
| 33, $`n = 256`$ | 0.572 s | 0.720 s | 0.306 s | 0.79 | 1.87 | 20 MB |
| 33, $`n = 512`$ | 2.24 s | 1.33 s | 0.873 s | 1.69 | 2.57 | 83 MB |
| 33, $`n = 1024`$ | 9.11 s | 3.33 s | 1.42 s | 2.73 | 6.42 | 333 MB |
| 34, $`k = 8`$ | 0.049 s | 0.296 s | 0.485 s | 0.17 | 0.10 | 2.0 MB |
| 34, $`k = 32`$ | 0.202 s | 0.519 s | 1.31 s | 0.39 | 0.15 | 8.0 MB |
| 34, $`k = 128`$ | 0.726 s | 1.48 s | 4.65 s | 0.49 | 0.16 | 32 MB |
| 34, $`k = 512`$ | 2.88 s | 5.29 s | 18.0 s | 0.54 | 0.16 | 129 MB |
| 35, $`n = 8`$ | 0.701 s | 0.440 s | 0.404 s | 1.59 | 1.74 | 2.6 MB |
| 35, $`n = 16`$ | 2.70 s | 0.639 s | 0.495 s | 4.22 | 5.45 | 10 MB |
| 35, $`n = 32`$ | 11.7 s | 1.21 s | 0.827 s | 9.6 | 14.1 | 42 MB |
| 35, $`n = 64`$ | 47.2 s | 2.39 s | 0.882 s | 19.7 | 53.5 | 168 MB |

The deep layer is where the GPU pays: function 35 is ahead from $`n = 8`$ and 53.5 times faster at $`n = 64`$ with the
whole convolution there. The wide shallow one pays from $`n = 256`$. The batch never does, and the GPU convolution's time
grows by about 35 ms an image, because the layer calls the GPU once per image (`local_batch = 1` in flexNN's
`convolutional_layer.h`). Every path sends the same volume at the same size.

**Checked before it is timed.** hpmpc has no test of its convolution, so the script inserts one into function 33 for
its own builds: small fixed integers through the secret-shared layer, the input in fixed point and the weights as
integers so the truncated output is the plain convolution, revealed and compared with a plain convolution in the
program and, as a sum over the 576 outputs, with an independent NumPy one (1,036,170). Every path reproduces it exactly
on all three parties; with the GPU hidden, neither GPU path does. The check is taken out of the clone before any run is
timed, and a timed run that prints it is refused.

The GPU convolution needs two more changes in the clone, printed by the script: CUTLASS 4.8.1's convolution header calls a
function from problem-size headers it does not include, so the object is compiled with `-include` of both; and hpmpc's
16-bit convolution fails a CUTLASS static assertion, so that instantiation is commented out, as upstream already does for
the 8-bit one. And one build flag: functions 33 and 34 do not compile with hpmpc's default `PRINT_TIMINGS 1`, because
`print.hpp` declares the layer-statistics functions only for function 35 and above while flexNN calls one, so every
convolution build sets `PRINT_TIMINGS=0`. It governs per-layer statistics, which one layer never prints; the time comes
from the protocol executor either way.
