# Annotated References

The `references/` folder holds the primary-source PDFs that ground this PoC. Each entry maps the
paper to the **workstream (WS)** it informs, the **module** that implements or uses it, and any
open-source code reused.

Entries for a paper, standard or library carry a `URL`, and an `Added` line where the date is known; older entries and notes
may lack either. `URL` is the canonical landing page, not a mirror.
`Added` is the date the PDF entered this repository, taken from `git log --diff-filter=A`, so it is
checkable rather than asserted; entries with no bundled PDF give the date the claim was last
verified against the source instead.

Markers:

- `[R]` the PDF is committed to this folder and you get it when you clone.
- `[R-local]` the PDF sits in this folder on the author's machine but is **gitignored**, so a clone
  does not contain it. Fetch from the URL. Two are ~17 MB each and one is a copyrighted book;
  since Round 45 four more are excluded on LICENCE rather than size, adjudicated per file in
  [`DATA_LICENSES.md`](../DATA_LICENSES.md) section 8.
- `[link]` never bundled. Fetch from the URL.

An earlier revision marked the `[R-local]` entries `[R]`, which told a reader they had files that a
clone never contained. **It happened again in Round 45**, to four entries: that round excluded the
PDFs on licence, wrote the adjudication table, listed "leaving `REFERENCES.md` marking them `[R]`"
in its own plan as the defect it was most likely to ship, and shipped it. The marker is the claim
about what a clone contains, and it is checked by
`tests/test_bundled_reference_pdfs_carry_a_grant.py` from Round 46 onward.

Workstreams: **WS1** AI-QKD parameter optimisation and entropy, **WS2** QML-IDS, **WS3** QSMPC x QKD,
**WS4** QHE x AI. Cross-cutting: PQC-first hybrid, crypto-agility, XAI.

---

## WS1 - AI-QKD (real-time parameter optimization) + entropy (QRNG)

**A source-level fact-check current at 2026-08-22 lives in
[`2026-08-factcheck.md`](2026-08-factcheck.md).** It confirms the ETSI/NIST/Rusca citations in this
section at their cover pages and re-derives every internal figure exact, and it contradicts six
claims - the SP 800-90B health tests being run on conditioned rather than raw samples, the
`FiniteKeyParams` defaults still carrying the retired intensities, "two-intensity decoy-state bound"
naming a protocol the cited equations are not, two attributions in `decoy_reference.py`, permissive
terms in the E91 and QSDC models, and the ITU-T series running past Y.3835 (to Y.3839, prepublished, as of 2026-11-03).
Each carries whether it was reproduced locally or is recorded as reported-and-unreproduced.

- **AML-QKD: Adaptive Machine Learning Framework for Real-time Parameter Tuning in QKD** - Mohamed,
  Kaldari and Al-Kuwari, Qatar Center for Quantum Computing (HBKU), **arXiv:2603.04192** [quant-ph]
  `[R-local]`
  URL: https://arxiv.org/abs/2603.04192
  Added: not committed (17 MB, `.gitignore` rules `references/OptiQKD_*.pdf` and
  `references/AML-QKD_*.pdf`); claim re-verified 2026-08-04
  **v1** (4 Mar 2026) was titled *OptiQKD: A Machine Learning-Optimized Framework for Real-Time
  Parameter Tuning in Quantum Key Distribution*; **v2** (1 Jun 2026) retitled it AML-QKD.
  This is one paper in two versions, not two papers. An earlier revision of this file listed it
  twice and described v2 as "extending" v1; v2 *supersedes* v1 and revises the headline number
  **downward**: v1 claimed median SKR +20-30 percent, v2 claims +14-25 percent (QBER 3.0 to 1.5
  percent in both). The QRL/E91 result (+29.2 percent throughput) is labelled *exploratory* by the
  authors. Simulation only.
  Grounds `src/qsh/qkd/optimize.py`, a deterministic offline-safe sklearn-surrogate plus
  coordinate-ascent analogue over the two-intensity decoy-state BB84 model, with the `mu_decoy`
  single-photon bound (Ma, Qi, Zhao and Lo 2005) and a convergence metric, and the E91/CHSH model
  `src/qsh/qkd/e91.py`. Honest scope: a simulation, not deployed QKD, and gradient-free (no RL, no
  torch), so it is an analogue of the paper's theme rather than a reproduction of its method.
  Both the v1 and v2 PDFs are kept locally for provenance (`OptiQKD_*.pdf`, `AML-QKD_*.pdf`); cite v2.

- **Quantum key distribution over a 2 km free-space channel with a high secure key rate** - Park,
  Kim, Im, Jo, Kim, Choi and Ihn, **arXiv:2607.24118v1** [quant-ph], submitted 2026-07-27 `[link]`
  URL: https://arxiv.org/abs/2607.24118
  Added: no PDF bundled; abstract loaded and claim verified 2026-08-15
  Licence **CC BY 4.0**, read from the abstract page's own licence line.
  **v1 only. It has an arXiv DataCite DOI (10.48550/arXiv.2607.24118) and NO journal DOI** - which
  is not the same thing as "no DOI", the phrasing a code comment used until Round 44. Every arXiv
  paper carries a DataCite DOI; what is absent here is peer review.
  Reports, verbatim from the v1 abstract: a **100 MHz** repetition rate, **164.8 kbps** secure key
  rate and QBER of **approximately 3.3 %** over a 2 km outdoor free-space channel, with active
  beam-wander correction by fast-steering mirrors.
  **v2 (2026-10-16), read 2026-10-27:** an **average finite-key** secure key rate of **110 kbps** at an
  average QBER of **3.28 %**, same link and clock. The code, the config and the control-plane panel
  quote v2 since Round 75.
  **What it grounds, and only this:** the `source_rate_hz` / `free_space_source_rate_hz` defaults in
  `src/qsh/config/defaults.py`. Round 44 made the source repetition rate an explicit, operator-set
  assumption so ETSI GS QKD 015's `skr`/`eskr` could carry the bit/s the spec defines them as, and
  this paper's 100 MHz is the default that assumption starts from. It anchored the free-space
  misalignment floor (0.025) until Round 75; a 2 km horizontal ground link is not a satellite
  downlink, and the floor now comes from Liao et al. 2017's measured QBER (the entry below).
  **Explicit negative, because a borrowed clock invites a borrowed result.** No other number from
  this paper is used. Its rate (v2: 110 kbps) is quoted on the control-plane panel purely as the nearest
  measured figure this repository cites, labelled as belonging to a different channel - a 2 km
  horizontal path, not this model's 50 km of fibre or its ground-to-LEO slant range - and it is
  **neither reproduced nor claimed** anywhere in this project. A preprint is a weaker source than a
  refereed one and the demo's own bit/s is a simulation under an assumed clock; both facts travel
  with the number wherever it is shown.

- **Satellite-to-ground quantum key distribution** - Liao et al., *Nature* **549**, 43-47 (2017).
  arXiv:1707.00542 `[link]`
  URL: https://arxiv.org/abs/1707.00542
  VERIFIED 2026-10-27 in the arXiv text. The source of the satellite page's channel since Round 75,
  value by value: the 300 mm satellite telescope's "near-diffraction-limited far-field divergence of
  ~10 urad"; Xinglong's receiving telescope "with an aperture of 1 m"; "diffraction loss ... 22 dB at
  1200 km"; atmosphere "3 dB to 8 dB" and pointing "less than 3 dB" at 1200 km; detectors
  "efficiency 50%, dark counts <25 Hz" with a "2-ns time window" (so 5e-8 per detector per pulse); an
  "overall optical efficiency including the receiving telescope and the fiber coupling ... ~16%"; a
  100 MHz source; QBER "with an average of 1.1%"; sifted rates "from ~12 kbit/s at 645 km to ~1
  kbit/s at 1200 km"; a final key of about 1.1 kbit/s with finite-key analysis. Micius ran THREE
  intensities, "mu_s=0.8, mu_1=0.1, mu_2=0", "sent with probabilities of 50%, 25%, and 25%", and since
  Round 77 so does the satellite model, with the vacuum yield measured (Ma et al. 2005, Eqs. 34 and
  37). The atmosphere now scales with air mass from an anchor at 1200 km (the midpoint, 5.5 dB, by the
  user's choice), and no link is operated below "10 deg elevation angle", where "A single-orbit
  experiment ends".
  ROUND 77 CORRECTED THE CROSS-CHECK. Round 75 compared the "~12 kbit/s at 645 km" with half of every
  detection at $`\mu = 0.8`$, as if every pulse were a signal, and found 12.2. Liao counted signal states
  only: 1,671,072 sifted of 3,551,136 detections is 0.47057, against 0.47059 for signal-only
  sifting at 50 / 25 / 25 and 0.5 for all states. Counted that way, the model of Round 75 was 2.9 dB
  low at 645 km and 2.5 dB high at 1200 km. The model of Round 77 integrates the whole pass of 19
  December 2016 (a 500 km circular orbit, closest approach 645 km, the data window under 1200 km)
  to 276.5 s against 273 s, 3,519,419 detections (-0.9%) and 1,655,574 sifted bits (-0.9%); the
  end rates read -0.4 dB at 645 km and +2.0 dB at 1200 km, so the model falls 8.4 dB over the pass
  where Liao's rate fell 10.8. Asserted in `tests/test_the_satellite_channel_is_sourced.py`. The 1.2 m
  aperture the repository used before is Yin et al., Science 356, 1140 (2017)'s stations
  (METADATA-ONLY here), not Xinglong.

- **Finite-key analysis for the 1-decoy state QKD protocol** - Rusca, Boaron, Grunenfelder, Martin
  and Zbinden, *Appl. Phys. Lett.* **112**, 171104 (2018). arXiv:1801.03443 `[link]`
  URL: https://arxiv.org/abs/1801.03443
  VERIFIED 2026-10-27 in the arXiv text. Two roles here. Its finite-key bound is what
  `src/qsh/qkd/finite_key.py` implements (Round 47), checked against its Table I. And since Round 76
  its stated simulation parameters ARE the fibre channel, value by value: "a repetition rate of
  1 GHz"; SNSPDs "which have a dead-time $`\tau_{DT}`$ = 100 ns, dark-count rate (DCR) of 10 Hz
  which correspond to $`p_{DC} = 10^{-8}`$ and an efficiency ($`\eta_{det}`$ around 50%)"; "a
  typical value $`p_{Err}`$ of 1%"; and Appendix B's dead-time correction, Eq. (B3),
  $`c_{dt} = 1 / (1 + R P t_{DT})`$, modelled in
  `qkd/channel.py::dead_time_factor` counting every click where (B3) sums the Z-basis ones. A
  simulation set rather than a measurement; the experiment that realised the protocol is the
  Boaron entry below. Table I's distances follow 0.2 dB per km.

- **Efficient decoy-state quantum key distribution with quantified security** - Lucamarini, Patel,
  Dynes, Frohlich, Sharpe, Dixon, Yuan, Penty and Shields, *Opt. Express* **21**, 24550 (2013).
  arXiv:1310.0240 `[link]`
  URL: https://arxiv.org/abs/1310.0240
  VERIFIED 2026-10-27 in the arXiv text. A MEASURED 50 km fibre link at 1 GHz with gated
  self-differencing InGaAs detectors: "detection efficiency of 20.5%, dark count probability per gate
  of $`2.1 \times 10^{-5}`$ and after pulse probability 5.25%", intensities 0.425 / 0.044 / 0.001 (two decoys),
  QBER 4.26% in Z, and secure rates of 1.09 Mbps (efficient BB84) and 0.63 Mbps (standard) at 50 km.
  Cited in `QKDConfig` for its smallest positive-key sample (1.4e5 counts). In Round 76 it was the
  yardstick that showed the old fibre set was about 3x optimistic, and it was not adopted: it ran
  three intensities, and under this project's 1-decoy finite-key analysis its noise needs 7e8 to 2e9
  pulses for a key.

- **Secure quantum key distribution over 421 km of optical fiber** - Boaron et al., *Phys. Rev. Lett.*
  **121**, 190502 (2018). arXiv:1807.03222 `[link]`
  URL: https://arxiv.org/abs/1807.03222
  VERIFIED 2026-10-27 in the arXiv text. The experimental realisation of the 1-decoy protocol modelled
  here, at 2.5 GHz: SNSPDs whose "maximum efficiencies ... are between 40 to 60%" at "a dark count rate
  (DCR) of 0.1 Hz", ultra-low-loss fibre "of about 0.16 dB/km (0.17 dB/km including the connections
  loss)", QBER in Z 0.4-0.7% from 252 to 355 km, and 6.5 bps over 405 km. Not adopted for the channel:
  its distances and fibre are far from the 50 km standard-fibre default.

- **Fast single-photon detectors and real-time key distillation enable high secret-key-rate quantum
  key distribution systems** - Grunenfelder et al., *Nat. Photon.* **17**, 422 (2023).
  arXiv:2210.16126 `[link]`
  URL: https://arxiv.org/abs/2210.16126
  VERIFIED 2026-10-27 in the arXiv text. 64 Mbps over 10.0 km and 3.0 Mbps over 102.4 km at 2.5 GHz,
  simplified BB84 with one decoy, multi-pixel SNSPDs (64% efficiency at high count rates, 82% system
  detection efficiency), QBER in Z 0.3-0.4%. The "2.5 GHz in the 64 Mbps fibre demonstration" an old
  comment in `clientSdnController.ts` quoted without a source. Not adopted: ultra-low-loss fibre and
  detectors well beyond the simulation set.

- **10-Mb/s quantum key distribution** - Yuan et al., *J. Lightwave Technol.* **36**, 3427 (2018).
  arXiv:1807.04484 `[link]`
  URL: https://arxiv.org/abs/1807.04484
  VERIFIED 2026-10-27 in the arXiv text. 13.72 Mb/s over 2 dB (10 km) at 1 GHz, InGaAs detectors with
  "a detection efficiency of 31%, a combined dark count rate of 450 kHz and an afterpulsing probability
  of 4.4%", QBER 3.07% of which afterpulsing contributes 2.2%, and a measured error-correction
  efficiency of 1.32-1.36 at 3% QBER. Read as the InGaAs comparison in Round 76; not adopted.

- **Information-Theoretic Solutions for Seedless QRNG Bootstrapping and Hybrid PQC-QKD Key
  Combination** - Giestinhas and Spiller, University of York, arXiv:2603.26907v2 [quant-ph], 2026
  `[R-local]`
  URL: https://arxiv.org/abs/2603.26907
  Added: not committed (`.gitignore` rule
  `references/Information-Theoretic_Solutions_for_Seedless_QRNG_*.pdf`); claim re-verified 2026-08-04
  Licence CC BY 4.0.
  Quantum Leftover Hash Lemma strong (Toeplitz, universal-2) extractors: (1) seedless QRNG
  bootstrapping from two independent entropy sources; (2) a QLHL-extractor **alternative** to XOR
  key-combining. It is **not** "secure-if-either" and not "better than XOR" - the paper never uses
  that phrase and explicitly says strong seeded extractors "present different characteristics",
  noting its own cost as "longer keys and a worse computational complexity". The property it does
  claim: if the combined output **and one input key** are compromised, the remaining key material
  retains quantifiable min-entropy. PQC keys are modelled as HILL (computational) min-entropy.
  Grounds the Round 16 QRNG and transport upgrade: `src/qsh/qrng/simulator.py` (LHL-bounded Toeplitz
  output plus seedless two-source extractor), `src/qsh/qrng/health.py` (SP 800-90B min-entropy
  estimate), and the config-gated Toeplitz+QLHL combiner in `src/qsh/pqc/transport.py` (`_mix_qkd_its`,
  `hybrid_combiner_mode="its_toeplitz"`). **As implemented that combiner is computational, not
  information-theoretic.** Its Toeplitz seed is SHAKE-256 of a label and of the same two keys it
  extracts from (`transport.py`, `tseed`), while the paper names the absence of correlation between
  input and seed as a necessary condition for its leftover hash lemma, so the seed and the input here
  are fully correlated. And the QKD-domain input it is given in the browser and by the full
  deployment's MCP endpoints is a fixed 32-byte constant, which carries no entropy at all; only the
  full deployment's own QKD path folds 256 or more QKD bits into its key, by HKDF-SHA256. Corrected
  2026-10-30: this line called it "the config-gated ITS combiner".

---

## WS2 - QML-IDS (quantum machine learning for intrusion detection)

**Workings for the 5G-NIDD metric profile live in
[`5g-nidd-metric-profile.md`](5g-nidd-metric-profile.md).** Its finding is that the published file holds one attacker's UDP flood twice under contradictory labels, which sets a ceiling of 0.7685 on the rows as published; the model is fitted with the Benign copy removed and scores 0.9979. Round 49 published `telecom_ids.nidd` with
accuracy 0.7608 against literature figures of 0.90 to 0.98 and attributed the gap to row-identifier
leakage in the published work. Round 50 measured the cause as one column, `IdleTime`, which `Combined.csv` does not carry, and refitted on it to reach 0.982897. Round 86 read that column's values: one per attack session and base station, so it names the file a flow came from, and the 0.98 measured how well a model recognises the file. The model is refitted without it, at accuracy 0.7616 and macro F1 0.9275, against a ceiling of 0.7685 for these columns and 0.42 and 0.61 across base stations, and that file carries the correction. The leakage-aware prior work Round 49 did not consult is cited there. The identifier finding itself stands - three integer counters move the narrow configuration to
0.9997 - but it does not explain anyone else's numbers.

This section was promised by the preamble of every earlier revision of this file and never written,
while `src/qsh/qml_ids/` and the Telecom/Defense use case shipped. The gap mattered: WS2 is the one
workstream whose honest answer is negative, and the reasoning lived only in a training script's
docstring.

- **Resisting Quantum Key Distribution Attacks Using Quantum Machine Learning** - Al-Kuwari et al.,
  *IET Quantum Communication* 7(1), January 2026, DOI 10.1049/qtc2.70028
  (arXiv:2509.14282 [cs.CR], v1 16 Sep 2025, v2 19 Feb 2026) `[link]`
  TITLE CORRECTED Round 55. This entry carried "A Quantum Machine Learning Approach for Intrusion
  Detection Systems", which is not the title on v1, on v2, or in IET - a plausible-sounding
  reconstruction from the subject matter rather than a transcription. It is also no longer a preprint.
  URL: https://arxiv.org/abs/2509.14282
  Added: no PDF bundled; claim re-verified 2026-08-04
  Supplies the **class structure**: 8 classes = 7 attack types plus normal. That taxonomy is what
  `src/qsh/qml_ids/attacks.py` generates and what `scripts/train/telecom_ids.py` trains against.
  No number from this paper is reproduced or claimed here.
  It also **refutes a claim this project used to make**. An earlier revision of the training script
  and of this section asserted that every published QML-IDS number sits below a random forest on the
  same data. Table VIII of this paper reports a hybrid QLSTM at 94.7 percent accuracy and 94.7
  percent F1 against the authors' own random-forest baseline on their generated data. The
  defensible statement is narrower: published quantum results here are usually obtained on heavily
  subsampled and dimension-reduced inputs, which makes them hard to compare against a forest given
  the full feature set - not that quantum always loses.

- **Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study** - Engelen, Rimmer
  and Joosen, IEEE Security and Privacy Workshops (WTMC), 2021 `[link]`
  URL: https://doi.org/10.1109/SPW53761.2021.00009
  Added: no PDF bundled; claim re-verified 2026-08-04
  Why this project reports per-class F1 first and aggregate accuracy second. Engelen et al. showed
  aggregate random-forest accuracy on CIC-IDS2017 sits at 0.99 both before and after correcting the
  labels, while Bot F1 moves 0.60 to 0.99: the aggregate is blind to the fix. `telecom_ids.py`
  reports both for exactly this reason.

- **Network Attack Traffic Detection With Hybrid Quantum-Enhanced Convolution Neural Network** -
  Wang, Fok and Thing, arXiv:2504.20436 [cs.CR], 29 Apr 2025 `[link]`
  URL: https://arxiv.org/abs/2504.20436
  Added: no PDF bundled; recorded 2026-08-15 with the 5G-NIDD licence withdrawal below
  **The published comparison point on 5G-NIDD**, which is this paper's sole dataset. On the standard
  split it reports classical CNN 99.61 percent, QCNN-AnE 99.62 percent, QCNN-AmE 99.65 percent,
  Quan-ConvCNN 99.71 percent and QuanvolutionNN 98.67 percent accuracy. **These are accuracies at
  telling malicious from benign traffic, not at telling the attack types apart**: the paper's Table II
  labels each traffic type malicious or benign, and its unseen-attack experiment scores a test set
  that is all malicious (read in the arXiv text 2026-10-30). So the 9-class accuracy of
  `telecom_ids.nidd` is not comparable with them. Two things have to be said
  before that ordering is quoted as a quantum win. The models are **hybrid quantum-classical CNNs,
  not pure Cong-style QCNNs**, so the name does not mean here what it means in the QCNN section
  below; and the authors' own claimed advantage is on **generalisation to unseen attacks**, not on
  the standard split, where a 0.10-point spread over a 99.61 percent classical baseline is not a
  result to lean on. Recorded because `DATA_LICENSES.md` section 5 requires every published model
  card to name the baseline it is measured against, and until this entry there was no such baseline
  on record for 5G-NIDD - which made the corpus look, from inside this repository, like a dataset
  with no external comparison point rather than one with a 2025 quantum paper already on it.
  Of its 28 inputs (Table III) exactly one, `Offset`, differs between the two copies of the flood in
  the published file; the same counts in its Table II show that it used those rows. See
  [`5g-nidd-metric-profile.md`](5g-nidd-metric-profile.md).

- **Withdrawn: a hybrid quantum-classical autoencoder IDS** - arXiv:2512.05069 `[link]`
  URL: https://arxiv.org/abs/2512.05069
  Added: no PDF bundled; withdrawal confirmed 2026-08-04
  Withdrawn **by its own authors** (v2, 2026-04-02) because the evaluation was "insufficient to
  fully support the paper's conclusions". Cited as evidence about the state of the field, not as a
  method. It is listed here so that a reader who finds v1 elsewhere learns it was retracted.

**What this repository measured**, and it is a negative result kept deliberately: on the synthetic
8-class telemetry, RandomForest reaches macro F1 0.9866 on its 600 test rows, while the
quantum-kernel SVC reaches 0.8316 and the variational QNN 0.4667. Those three are not one quantity:
both quantum models are **binary**, each trained on its own smaller draw (60 and 120 rows per class),
so their macro F1 averages attack and normal, and the QNN predicts "attack" for every row, which is
what 0.4667 is. Until Round 86 this paragraph put the gap down to "the six classes they never
attempt"; that was wrong. Like for like, a forest trained and scored on each quantum model's own split
and labels reaches 0.9903 and 1.0 (`docs/measurements/telecom-quantum-like-for-like-2026-11-07.json`). Even granting the binary figures, neither earns its qubits against a random forest that
takes a small fraction of the run the quantum pair needs - `wall_clock_s` in
`models/metrics.json` covers the whole script and has been measured at 267.1 s and 341.5 s on the
same machine, so the absolute seconds are load-dependent and only the order of magnitude is a
claim. `quantum_beats_classical` is recorded as `False` in `models/metrics.json`, which is the
field this project would have to falsify to claim otherwise.

Two facts about what runs, checked 2026-10-30. The published `telecom_ids.qml` artefact, the model
the telecom page runs and verifies, is the RandomForest: a 200-tree ONNX tree ensemble with a 7-wide
input. The quantum-kernel SVC and the QNN are trained offline and are not run by the page. And the
SVC's fidelity kernel embeds one input and un-embeds the other with no entangling gate, so it equals
the classical product kernel $`\prod_i \cos^2((x_i - x'_i)/2)`$ (to 2.2e-16 on 200 random pairs,
PennyLane 0.45.1): its 0.8316 is the score of a kernel a laptop evaluates in closed form.

That last sentence was briefly false. Round 33 re-ran `telecom_ids.py` **without** `--quantum` to
add the CKKS student, and `record_metrics` replaces a record rather than merging it, so all eight
`quantum_*` fields - including the `quantum_beats_classical` this paragraph points at - were
deleted from `models/metrics.json` while three published claims went on citing them. Neither the
config-parity gate nor the new prose gate could see it: the prose fields were untouched and both
derived artifacts stayed consistent with each other. Restored in Round 34 by re-running with
`--quantum`; the values came back identical, which is the only reason the citations above were
recoverable rather than merely re-asserted.
The data is synthetic, and the reason was overstated in earlier revisions. The accurate version,
re-checked 2026-08-05:

- **UNSW-NB15** requires the authors' agreement for commercial use.
- **CIC-IDS2017 does not "state no licence at all".** The Canadian Institute for Cybersecurity's
  institute-level FAQ says *you may redistribute, republish, and mirror our datasets in any form*
  with citation, and places that under commercial use - while the CICIDS2017 dataset page itself
  frames the data as *publicly available for researchers* and names no licence. The two pages
  disagree. That is an unresolved ambiguity, not a permission, and it is the same class of question
  as the ImageNet one that Round 19 resolved by dropping the dependency rather than guessing.
- **The Engelen-corrected redistribution** is hosted separately and states nothing.
- **5G-NIDD was evaluated and rejected, and that rejection is WITHDRAWN.** The dataset is
  **CC BY 4.0**, `access_type: Open`, at the authors' repository of record - Fairdata/Metax, DOI
  10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361. The rejection had been read off IEEE DataPort,
  which is a mirror, marks the dataset subscription-required and carries no licence string; absence
  of a grant at a mirror was recorded as absence of a grant. The adjudication still lives in
  [`DATA_LICENSES.md`](../DATA_LICENSES.md), where 5G-NIDD now sits in section 1 as usable with
  **attribution required** - CC BY 4.0 obliges anyone publishing the data, or a model trained on it,
  to credit the dataset authors.

  The previous revision of this bullet ended: *"a licence finding with two homes is a licence finding
  that will be corrected in one of them."* It was right, and it undercounted. The finding had
  **three** homes - this file, `DATA_LICENSES.md` and `src/qsh/models/registry.py` - the withdrawal
  was written into a **fourth**, `docs/deliverable-verification.md`, and for a full round that fourth
  file was the only place the truth was recorded while three others went on calling the dataset
  rejected. The sentence warning about duplication was itself one of the copies that went stale.
  Duplication is not fixed by noticing it in prose; it is fixed by there being one owner, which is
  the rule stated immediately below and which this entry existed in violation of.

Licence adjudications belong in [`DATA_LICENSES.md`](../DATA_LICENSES.md); this file cites the
papers. Where the two used to overlap, that file is the owner.

---
- **Gillani, Baig, Shah, Ali and Siddiqui, "How Quantum Is the Advantage? A Fair, Calibration- and
  Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion
  Detection"**, arXiv:2608.18155 v1, 2026-08-13, preprint `[link]`
  URL: https://arxiv.org/abs/2608.18155
  Added: 2026-11-08
  Read in the arXiv PDF. Hybrid VQCs and quantum-kernel SVMs against five tuned classical models on four
  NIDS datasets, 8 PCA components on 8 qubits, simulation only, three seeds for quantum models. Best
  classical against leading 8-qubit quantum F1: 0.782/0.730, 0.891/0.836, 1.000/0.968, 0.991/0.949.
  Parameter-matched classical controls remove the aggregate gain. Two NSL-KDD results survive
  false-discovery correction (an IQP kernel over its random-feature surrogate, and a 4-qubit hybrid at 1%
  FPR); only the first survives Holm. F1 near 0.98 under within-train cross-validation falls to about
  0.78 on the official NSL-KDD test split. It tests neither this project's circuits nor its corpora.

## WS3 - QSMPC x QKD (quantum-secure multi-party computation)

- **Quantum Secure Multiparty Computation based on Secure Summation and QKD** - Mandeep Kumar and
  Bhaskar Mondal (NIT Patna), Research Square, 3 Jul 2025 `[R]`
  URL: https://doi.org/10.21203/rs.3.rs-6706796/v1
  Added: 2026-06-27
  Licence **CC BY 4.0**, which requires attribution; an earlier revision of this entry named no
  authors.
  File: `Quantum_Secure_Multiparty_Computation_based_on_Secure_Summation_and_QKD.pdf`
  Core protocol for `src/qsh/qsmpc/secure_summation.py`: QKD-shared keys plus secure summation.

- **An efficient simulation for quantum secure multiparty computation** - Sutradhar and Om, Scientific Reports **11**,
  2206 (2021), DOI `10.1038/s41598-021-81799-z`, CC BY 4.0 `[R]`
  URL: https://www.nature.com/articles/s41598-021-81799-z
  Added: 2026-06-27 (the bundled posting); the article cited here from 2026-11-08
  File: `Efficient_Simulation_of_Quantum_Secure_Multiparty_Computation.pdf`, which is the later single-author posting of the same protocol:
  K. Sutradhar, arXiv:2501.10083 (2025). Its abstract paraphrases this article's sentence by sentence, with the same worked example and simulation, and its reference list
  cites the article. This entry cited only the posting until 2026-11-08 (`DATA_LICENSES.md` section 8 has the licence of each).
  (t,n)-threshold QSMS = Shamir secret sharing plus QFT plus Pauli ops. Directly reimplemented in
  `secure_summation.py` and `shamir.py` (Qiskit-native, no hardware). Validated on the paper's
  worked example (t=3, n=7, d=11, X=2, Y=3, giving S=5).

- **Quantum convolutional neural networks** - Cong, Choi and Lukin, Nature Physics **15**, 1273 (2019), DOI `10.1038/s41567-019-0648-8`,
  arXiv:1810.03787 `[link]`
  URL: https://arxiv.org/abs/1810.03787
  Added: 2026-11-08
  The source of the name QCNN and of the convolution-and-pooling structure that `tests/test_the_qcnn_has_the_structure_the_name_claims.py` holds the trained circuits to.

- **Quantum convolutional neural network for classical data classification** - Hur, Kim and Park, Quantum Machine Intelligence **4**, 3
  (2022), DOI `10.1007/s42484-021-00061-x`, arXiv:2108.00661 `[link]`
  URL: https://arxiv.org/abs/2108.00661
  Added: 2026-11-08
  The SU(4) two-qubit block telecom's circuit uses on every neighbouring pair (`scripts/train/qcnn_variational.py`).

- **Experimental secure multiparty computation from quantum oblivious transfer with bit commitment**
- Zhang et al., npj Quantum Information, 2026 `[link]`
  URL: https://www.nature.com/articles/s41534-026-01219-w (npj QI 12, 76, 2026)
  Preprint: https://arxiv.org/abs/2411.04558
  Added: 2026-06-27
  File: none in the repository since Round 45 (CC BY-NC-ND 4.0, excluded; `DATA_LICENSES.md`
  section 8)
  Decoy-state BB84 plus SHA-256 commitment QOT, giving private set intersection on bank-blacklist
  data. `src/qsh/qsmpc/qot.py` sketches the 1-of-2 QOT primitive. **The finance use case does not
  use it.** Its set intersection is a classical, semi-honest DH-OPRF PSI on RFC 9497's OPRF over
  ristretto255 (`src/qsh/qsmpc/psi.py`, `web/src/lib/oprfPsi.ts`; the RFC 9497 entry below), which is
  not post-quantum. Corrected 2026-10-30: this line said the use case implemented the paper's
  construction in simulation.

- **RFC 9497, "Oblivious Pseudorandom Functions (OPRFs) Using Prime-Order Groups"** - Davidson,
  Faz-Hernandez, Sullivan and Wood, IRTF (CFRG), December 2023 `[link]`
  URL: https://www.rfc-editor.org/rfc/rfc9497.txt (read 2026-11-01; SHA-256 of the text
  `95160517617374655fe23af2c6cec311eae326deb7689239f97641fd5a560432`)
  The finance PSI's OPRF, mode 0x00 with the suite ristretto255-SHA512: the context string (section
  3.1), GenerateKeyPair and DeriveKeyPair (3.2, 3.2.1), Blind, BlindEvaluate, Finalize and Evaluate
  (3.3.1), and the suite's HashToGroup, HashToScalar and serialization (4.1), implemented in
  `src/qsh/qsmpc/oprf.py` and run in the browser with `@noble/curves`' `ristretto255_oprf`. Section
  7.1 gives the server *"Unconditional input secrecy"* (it *"does not learn anything about the client
  input x, even with unbounded computation"*); section 7.2.1 rests pseudorandomness on the One-More
  Gap Computational Diffie-Hellman assumption, which is why the PSI is not post-quantum; section
  7.2.3 puts the cost of Q BlindEvaluate queries to one key at about $`\log_2(Q)/2`$ bits and names
  rotating keys as a mitigation (the PSI's key is fresh per run); section 7.4 requires constant-time
  operations on secret data, which the Python implementation does not provide. Appendix A.1.1's two
  OPRF-mode vectors are checked in both tiers by `tests/test_r84_oprf_psi.py`.
- **RFC 9496, "The ristretto255 and decaf448 Groups"** - de Valence, Grigg, Hamburg, Lovecruft,
  Tankersley and Valsorda, IRTF (CFRG), December 2023 `[link]`
  URL: https://www.rfc-editor.org/rfc/rfc9496.txt (read 2026-11-01; SHA-256 of the text
  `d875c6e703827da487eae8f0656fd25e4a278f1dcb0f06041669adad27c03ff3`)
  The prime-order group under the OPRF. `src/qsh/qsmpc/oprf.py` implements section 4: the constants
  (4.1), SQRT_RATIO_M1 (4.2), Decode, Encode and Equals (4.3.1 to 4.3.3) and the element derivation
  from 64 bytes (4.3.4). Appendix A's generator multiples, invalid encodings, element-derivation
  outputs and square-root ratios are checked by `tests/test_r84_oprf_psi.py`.
- **RFC 9380, "Hashing to Elliptic Curves"** - Faz-Hernandez, Scott, Sullivan, Wahby and Wood, IRTF
  (CFRG), August 2023 `[link]`
  URL: https://www.rfc-editor.org/rfc/rfc9380.txt (read 2026-11-01; SHA-256 of the text
  `a2e6ab02b2117d6f52ac293020857752972b2e23ae005c33f612e40436ff4b6d`)
  Section 5.3.1, expand_message_xmd, and Appendix B, hash_to_ristretto255 (the element derivation of
  64 expanded bytes), which RFC 9497's ristretto255-SHA512 suite uses for HashToGroup. Appendix K.3's
  SHA-512 vectors check `expand_message_xmd` in `tests/test_r84_oprf_psi.py`.

- **Quantum-secure multiparty deep learning** - Sulimany, Vadlamani, Hamerly, Iyengar and
  Englund, Physical Review X 15, 041056, published 24 Dec 2025 `[R]`
  URL: https://doi.org/10.1103/k8wg-qmbh (APS moved to opaque DOIs in 2025, so the older
  `10.1103/PhysRevX.15.041056` pattern does not resolve)
  Preprint: https://arxiv.org/abs/2408.05629
  Added: 2026-06-27
  File: `Quantum-Secure_Multiparty_Deep_Learning.pdf`
  Photonic coherent linear-algebra inference; Holevo and Cramer-Rao leakage bounds (0.1 bit per
  weight, MNIST above 95 percent). Nothing in this repository implements the leakage analysis: its
  0.1 bit per weight was quoted as a literature constant in the retired security score
  (`src/qsh/benchmark/security.py`, removed in Round 86). Corrected 2026-10-30: this line said the analysis was
  reimplemented in `src/qsh/qsmpc/` as security metrics.

- **Quantum-Safe Multi-Party Computation for Distributed AI Datasets** - Gopher Security blog, 2026
  `[link]`
  URL: https://gopher.security/blog
  Added: 2026-06-27
  File: none in the repository since Round 45 (no grant found; `DATA_LICENSES.md` section 8)
  A vendor blog post, not peer-reviewed, and weighted accordingly. qOLE and qMPSI plus PQC (ML-KEM)
  hybrid transport plus an MCP delegate-and-verify narrative. Informs `src/qsh/pqc/transport.py` and
  the end-to-end orchestration framing.

---

- **Kolesnikov, Matania, Pinkas, Rosulek and Trieu, "Practical Multi-party Private Set Intersection from Symmetric-Key
  Techniques"**, CCS 2017, IACR ePrint 2017/799 `[link]`
  URL: https://eprint.iacr.org/2017/799
  Added: 2026-11-08
  Read at the abstract. Any number of semi-honest participants, from an oblivious programmable PRF; five parties with
  $`2^{20}`$ items each in 72 s, or about 22 s in a weaker model. Cited in `docs/threat-model.md` for what a dedicated protocol
  reaches: this project's PSI among banks is one generic circuit and stays at eight accounts each.
- **Srikumar, Hill and Hollenberg, "A kernel-based quantum random forest for improved classification"**, Quantum Machine
  Intelligence, 2024; arXiv:2210.02355 `[link]`
  URL: https://arxiv.org/abs/2210.02355
  Added: 2026-11-08
  Read at the abstract. Decision trees of QSVM nodes, with a low-rank Nystrom approximation of the kernel; the abstract
  compares the forest with QSVMs. Cited in the register's decision not to build a quantum random forest here.
- **Rasyidi, Alhussein, Muhaidat and Damiani, "Hybrid Quantum-Classical Autoencoders for Unsupervised Network Intrusion
  Detection"**, arXiv:2512.05069 `[link]`
  URL: https://arxiv.org/abs/2512.05069
  Added: 2026-11-08
  Read at the abstract. The arXiv page marks v2 (2026-04-02) as withdrawn pending additional experiments and analysis, so no
  figure from it may be used. Cited in the register's decision not to build a VAE filter here.

## WS4 - QHE x AI (quantum/classical homomorphic encryption + AI)

- **Reverse Delegated Training and Private Inference via Perfectly-Secure Quantum Homomorphic
  Encryption** - Ortega and Martin-Delgado, arXiv:2602.12712, 2026 `[link]`
  URL: https://arxiv.org/abs/2602.12712
  Added: 2026-06-27
  File: none in the repository since Round 45 (arXiv's default licence grants third parties nothing;
  `DATA_LICENSES.md` section 8)
  Read in the arXiv text (v1, 13 Feb 2026) on 2026-10-30: QOTP encryption with a
  teleportation-based, non-interactive evaluation of T gates, and no classical FHE, on QCNNs
  decomposed into Clifford+T. Two scenarios: reverse delegated training on MNIST digits 0 and 1
  reduced to two principal components (test accuracy 0.967 after decryption, 0.496 before), and
  private inference with an eight-qubit network trained in plaintext with PennyLane on 16x16 MNIST
  images of a binary task (accuracy 0.967), whose encrypted inference is shown on five test samples.
  Pauli-gate concealment gives "probabilistic model protection" of the server's circuit. Those are
  the PAPER's properties, and two of them (Clifford+T in Python, and circuit concealment) this
  repository does not reproduce. Corrected 2026-10-30: this line read "QOTP plus classical FHE
  hybrid ... MNIST 96.7 percent", which named a classical FHE the paper does not use and read as
  encrypted inference on full MNIST. **Relationship to this repository:** `src/qsh/qhe/qcnn.py` is an
  *independent and PARTIAL* implementation of the paper's construction. It is not Clifford+T: the
  module builds `AngleEmbedding` plus CNOT/RY/RZ and contains no T gate, with `t_count()` returning
  a labelled Solovay-Kitaev estimate rather than a count. Pauli-gate concealment for CIRCUIT privacy
  is also not reproduced - `qotp.py` conceals a state, not a computation, because propagating the
  Pauli frame through the circuit breaks at T, which is the difficulty the paper exists to address.
  The browser circuit in `web/src/lib/quantumSim.ts` is the one that is genuinely Clifford+T, and it
  counts its T gates. It does not import, vendor or wrap the authors' reference code at
  `OrtegaSA/CQC-QHE-repo`; an earlier version of this line claimed code reuse that never existed,
  and the submodule was removed in Round 17. Cited as the source of the construction, which is the
  accurate relationship.

- **Rate-1 Quantum Fully Homomorphic Encryption** - Chardouvelis, Dottling and Malavolta, IACR
  ePrint 2020/1454 (TCC 2021) `[R]`
  URL: https://eprint.iacr.org/2020/1454
  Added: 2026-06-27
  File: `Rate-1_Quantum_Fully_Homomorphic_Encryption.pdf`
  Rate approaching 1 ASYMPTOTICALLY: rate is plaintext over evaluated-ciphertext size, qubits plus
  bits (Def. 6.1), and it tends to 1 only as the security parameter and the message grow (Sec. 6.2),
  reached by moving the Pauli key into the rate-1 classical FHE of Brakerski, Doettling, Garg and
  Malavolta (TCC 2019). Malicious circuit privacy, under quantum LWE - computational, not
  information-theoretic. **The paper reports no concrete ratio**: the 1.32 this repository once called
  its "literature value" is not in it. That constant and the Overview card it fed were removed on
  2026-10-23, and the quantum-advantage score built on the same comparison was retired from display
  then. The module is kept (`src/qsh/benchmark/quantum_advantage.py`), but neither tier computes it
  for a run and no page shows it. Informs
  `src/qsh/qhe/rate1_qfhe.py` (QOTP plus a KEM-masked key; no Eval).
- **Classical rate-1 FHE, the counterpart the comparison above omitted** `[R]`
  Gentry and Halevi, "Compressible FHE with Applications to PIR", TCC 2019, LNCS 11892, pp. 438-464,
  doi:10.1007/978-3-030-36033-7_17 (ePrint 2019/733): rate $`1-\epsilon`$ under (ring) LWE, with a
  concrete rate-4/9 PIR. Brakerski, Doettling, Garg and Malavolta, "Leveraging Linear Decryption:
  Rate-1 Fully-Homomorphic Encryption and Time-Lock Puzzles", TCC 2019, LNCS 11892, pp. 407-437,
  doi:10.1007/978-3-030-36033-7_16 (ePrint 2019/720): rate $`1-1/\lambda`$ from LWE. Because classical
  FHE also reaches rate 1 - and rate-1 QFHE is built on the second - a gap between a quantum and a
  classical ciphertext ratio is not a quantum advantage. Added: 2026-10-23.

- **Leveraging ASIC AI Chips for Homomorphic Encryption (CROSS)** - Tong et al., IACR ePrint
  2026/160, HPCA 2026 `[R]`
  URL: https://eprint.iacr.org/2026/160
  Added: 2026-06-27
  File: `Leveraging_ASIC_AI_Chips_for_Homomorphic_Encryption.pdf`
  Maps FHE NTT and modular ops onto AI accelerators (BAT/MAT). Reference for the GPU benchmark's
  NTT and throughput discussion. Code: `EfficientHE/CROSS`, TPU-oriented, reference only.

- **EncFormer: Secure and Efficient Transformer Inference over Encrypted Data** - Zhu, Jin, Aung and
  Xiao (NUS and A*STAR), arXiv:2604.09975 [cs.CR], 11 Apr 2026 `[link]`
  URL: https://arxiv.org/abs/2604.09975
  Added: 2026-06-27
  File: none in the repository since Round 45 (arXiv's default licence grants third parties nothing;
  `DATA_LICENSES.md` section 8)
  **Manuscript submitted to IEEE TDSC: not accepted, not peer-reviewed.** Hybrid CKKS plus
  two-party computation, not FHE-only, and the text contains no TFHE, no quantization and no
  polynomial-softmax method; an earlier revision of this entry attributed all three to it and dated
  it 2024. Reported end-to-end BERT-base latency: 2.1 minutes LAN, 8.9 minutes WAN3; the authors put
  their secure pipeline at "roughly $`3 \times 10^4`$ slower" than the 4.5 ms plaintext baseline. Informs the
  Transformer workload in `src/qsh/benchmark/` as a cost reference, not as a performance target.

---

## QOTP-based quantum homomorphic encryption, and what WS4 can claim (Round 30)

Added 2026-08-11, verified at the arXiv abstract page.

- **Hernandez-Bueno, Lage, Higuero & Astorga, "Quantum Homomorphic Encryption: Towards Practical
  QOTP-Encrypted Computation on Gate-Based Hardware."** arXiv:2604.19256 [quant-ph], v1 2026-04-21
  (14 pp, titled "...Towards Practical and Private Computation on Untrusted Quantum Hardware"),
  **v2 2026-09-15, 28 pp, retitled** - checked at arXiv on 2026-10-23. A universal QHE framework derived from the Quantum One-Time Pad,
  information-theoretically secure, "exploiting the transformation properties of quantum gates
  under Pauli encryption" for non-interactive homomorphic evaluation of arbitrary Clifford+T
  circuits, with results from simulators and real quantum processors. **This is WS4, published
  four months before this round, and the repository did not cite it.** It is the correct
  positioning reference for `qotp_clifford_random_circuit_demo`, and it also makes the
  repository's honest disclosure - that the browser circuit is Clifford+T while the Python module
  contains no T gate - more informative rather than less. Added: 2026-08-11.
- **Joshi, Mishra & Karthikeyan, "Blind Transpiler: An open-source library for universally blind
  and homomorphic quantum computations."** arXiv:2607.17131, v1 2026-07-19, v2 2026-08-09.
  Transpiles Qiskit circuits to blind counterparts and demonstrates a blind variational quantum
  classifier on IRIS. The closest software prior art to this project's browser QCNN under QOTP.
  Added: 2026-08-11.
- **Liu, Gong, Tian, Zhang, Zheng & Xu, "Efficient Quantum Fully Homomorphic Encryption."**
  arXiv:2604.23490 [quant-ph], v1 2026-04-26, v2 2026-06-21, 32 pp. Reduces the quantum gadget
  from $`O(\lambda^2)`$ to $`O(\lambda \log^2 \lambda)`$ EPR pairs and moves every quantum resource to the
  server, so the client performs only classical LWE key generation and Pauli-key encryption under
  classical FHE. **Reframes rather than contradicts** this repository's "rate-1 transport, no
  Eval" caveat: as of April 2026 the direction of travel is a fully classical client, which is
  architecturally closer to the browser-tab story than to the caveat's pessimism. Added:
  2026-08-11. **Caution, 2026-10-23:** v3 (2026-10-13) carries the authors' comment that the
  conclusion contains mistakes that need to be corrected. It is NOT withdrawn - the abstract stands -
  but treat its conclusions as provisional until a corrected version appears, and do not build an
  argument on them.

**What this project can and cannot claim on this machinery, recorded here because the split
matters.** WS4(c), circuit privacy against a malicious server, is met for Clifford circuits: the
server holds the maximally-mixed state throughout and the client recovers the exact result by
tracking the Pauli frame. WS4(f), hiding the server's circuit from the client, is NOT met and is
in tension with the mechanism - the client must know the circuit to update its frame. The two are
now separate rows in `docs/findings.md` (the README's until Round 80) rather than one row that
could only ever read PARTIAL.

## Prior art for the novelty claim, and the asymmetry it exposes (Round 30)

Added 2026-08-11. This project's verification rule requires a novelty claim to survive an
adversarial prior-art search. Run against this project's own headline claim, the search returned
prior art for one half and, read correctly, for the other too (the negative results below were
narrowed on 2026-10-30); it found nothing for the whole chain. All of it is recorded, because
reporting only the absence would be the omission this file exists to prevent.

- **Buchsteiner, Koch, Rotaru & Rechberger, "webSPDZ: Versatile MPC on the Web."** IACR ePrint
  2025/487, received 2025-03-14. Compiles MP-SPDZ's C++ backend to WebAssembly with Emscripten and
  moves party communication to WebRTC or WebSockets, supporting the 40-plus protocols MP-SPDZ
  implements. **Its abstract names two earlier browser MPC engines, JIFF and MPyC-Web.** So
  multiparty computation in a browser tab is established work and the garbled-circuit half of this
  demo is a re-implementation, not a first. `docs/threat-model.md` already mentioned JIFF; webSPDZ
  and MPyC-Web were absent. Added: 2026-08-11.
- **Aikata, Krieger & Sinha Roy, "Privacy at your Fingertips: Enabling Rapid Client-Side
  Operations in Fully Homomorphic Encryption."** IACR ePrint 2026/515, received 2026-03-13.
  Verified at the listing: "client-side operations" means encryption and decryption, and the
  abstract states that FHE "allows users to offload large computations to servers". It reduces
  client enc/decrypt cost by up to 97 percent - the client still does not evaluate. **This is the
  clearest evidence that the field's default division of labour is the opposite of this demo's**,
  which is why it is cited beside the novelty claim rather than in the FHE section. Added:
  2026-08-11.
- **Zama, TFHE-rs JS-on-WASM API documentation.** States that the WebAssembly client API supports
  key generation, encryption and decryption but **not FHE computation**, and gives no reason. It
  notes separately that without WASM threading key generation can be too slow, and that some
  parameter sets' keys exceed what it calls WASM's 2 GB memory limit (the JS API limit for a 32-bit
  memory is 4 GiB). Documentation rather than a result, and the strongest single sentence supporting
  the in-tab evaluation claim: the leading FHE toolchain does not evaluate in the browser. Checked
  2026-08-11; re-read 2026-11-08.
- **Wang, Yu, Han, Gu, Xu & Huang, "WAMO: Toward Secure Browser Inference via Web Model
  Obfuscation in WebAssembly."** Proceedings of the ACM Web Conference 2026 (WWW '26), 2026-04-09,
  DOI 10.1145/3774904.3792297. **Carried as a partial contradiction, which is why it is worth
  having:** it argues that deploying a model in a browser exposes its code and parameters to an
  untrusted environment, against the intuition that client-side execution is automatically
  private. It does not threaten this project - the models here are public on Hugging Face, so
  extraction is not a harm - but it is the paper a reviewer will raise, and the threat model
  should name it. **Metadata verified via the OpenAlex record for that DOI; the full text was not
  retrieved (ACM DL refuses automated fetches), so nothing beyond title, authors, venue and date
  is asserted here.** Added: 2026-08-11. [Re-checked 2026-11-15: Crossref gives pp. 2812-2823 and
  publication on 2026-04-12, and the abstract, read through Semantic Scholar's API, says browser deployment
  "exposes model code and parameters to untrusted environments", the point the value page makes; the page now
  links the DOI.]

**The negative results, stated as findings rather than left as absence.** No peer-reviewed 2025-2026
paper was located that performs CKKS homomorphic *evaluation* of a neural network inside a browser
tab with no server compute. That is the whole negative, and it is narrower than this paragraph used
to state it ("every browser-FHE artefact found is encrypt-in-browser, evaluate-on-server", corrected
2026-10-30): homomorphic evaluation in a browser exists. node-seal, the WebAssembly build of SEAL
this demo uses, evaluates in the browser, and its own `BENCHMARK.md` (Round 31 additions, below)
times evaluation operations such as `EvaluateRotate` in Chrome, Firefox and Safari; and Attrapadung
et al. (AsiaCCS 2018; Round 83 additions, below) shipped two-level homomorphic encryption compiled
to WebAssembly for browsers. No WebGPU-accelerated homomorphic-encryption or MPC paper was located
for 2025-2026 either, which is consistent with the note in `web/src/lib/clientFHE.ts` that no
commercial-OK WebGPU FHE exists in 2026. Absence of evidence is recorded as absence of evidence: the
searches and their dates are here so the next round can re-run them rather than re-reason.

## Hybrid FHE/MPC private inference: the 2026 state of the art, and an attack on the nearest design

Added 2026-08-11. Each entry below was verified against the arXiv abstract page or the IACR listing
directly, not against another file in this repository. The architecture discussion lives in
[`docs/architecture.md`](../docs/architecture.md); this section is the provenance.

- **Biswas, Chartier, Dhasade, Jurien, Kerriou, Kermarec, Lemou, Tranie, de Vos & Vujasinovic,
  "Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption" (Safhire).**
  TMLR; arXiv:2509.01253v2, revised 2026-10-16. Linear layers encrypted server-side, non-linearities
  offloaded to the client **in plaintext**, which removes bootstrapping and allows exact activations;
  up to 7.6x lower latency than Orion and 7.4x lower than Concrete-ML (v1 said 1.5x-10.5x). Model confidentiality is protected by **randomised shuffling**
  of the intermediates the client sees. This is the closest published architecture to this
  repository's, and the difference is precisely where the non-linear half runs. Added: 2026-08-11.
- **Kim & Lee, "Shuffling is Not Enough: Breaking Permutation-Based Model Confidentiality in Hybrid
  FHE Inference."** IACR eprint 2026/1527, ESORICS 2026 (received 2026-07-25). Recovers a
  permutation-invariant summary of a linear layer exactly and achieves perfect model
  distinguishability, needing `d+1` queries for a `d`-input layer - 5,712 for all linear layers of a
  ResNet-20. Breaks the shuffling-plus-noise defence that a plaintext-offload design depends on.
  **Carried as a trip-wire; in the full deployment's server tier it was crossed until Round 84.** This
  repository sends the non-linear half as secret shares, through masked conversions since Round 75,
  meant to let the key holder decrypt only masked intermediates. With the TenSEAL backend the mask
  covered slot 0 only: `src/qsh/mpc/secret_inference.py` added the bias plus the mask as a scalar to
  `ct.dot(w)`, TenSEAL broadcasts that scalar to every CKKS slot, and the dot product's rotation sum
  leaves partial sums of the products in the other slots. Decoding the whole plaintext with the
  client's own secret key, slot differences cancelled the mask and exposed each product, so the
  client recovered the unmasked output and the whole weight row in ONE query (confirmed locally
  2026-10-30; `tests/test_mpc.py` read slot 0 only, which is why it passed). Since Round 84 every
  slot the client can decode is masked independently, real and imaginary parts, by a vector encrypted
  under its key (`src/qsh/fhe/masking.py`), and `tests/test_r84_c1_slot_masking.py` decodes every
  slot and finds no product; what the client decodes is bounded per run by
  `view_statistical_distance`, statistically rather than perfectly, and semi-honest with an honest
  majority. On OpenFHE the imaginary parts are not flooded, and such runs say so. The public demo
  has no server and its models are public, so it was never affected. Corrected 2026-10-30: this
  said the precondition was absent, up to the masks' stated statistical bound. **Until Round 75 this entry said "no
  plaintext intermediate exists to query" while the conversions decrypted every linear output in the
  clear and opened every ReLU result to all parties**, so the precondition was present then too. If
  the non-linear path is ever moved to plaintext client offload, this applies directly. Added:
  2026-08-11.
- **Xu, Lu, Yu, Yi, Lin, Wang & Li, "Breaking the Layer Barrier: Remodeling Private Transformer
  Inference with Hybrid CKKS and MPC" (BLB).** arXiv:2508.19525, Aug 2025. Splits at operator rather
  than layer granularity and fuses adjacent linear operators to minimise CKKS<->MPC conversions; 21x
  communication reduction vs BOLT (S&P'24) and 2x vs BumbleBee (NDSS'25), with 13x and 1.8x latency
  reductions under GPU acceleration. Already named in `docs/models-and-benchmarks.md` (the README's text until Round 80) and
  `docs/architecture.md` as the
  current end of the Gazelle -> Delphi -> BOLT -> Bumblebee line; recorded here with its numbers.
  Added: 2026-08-11.
- **Lehmkuhl, Mishra, Srinivasan and Popa, "Muse: Secure Inference Resilient to Malicious Clients."** USENIX Security 2021 `[link]`
  URL: https://www.usenix.org/system/files/sec21-lehmkuhl.pdf
  A malicious client extracts model weights from semi-honest additive-share protocols by malleating its share, at 22 to 312 times fewer queries than the best black-box extraction, and the attack scales with the parameters, not the depth. Why `docs/threat-model.md` does not claim malicious security for the medical CNN under MPC. Added: 2026-11-08 (abstract, sections 1-2).
- **Rathee et al., "CrypTFlow2: Practical 2-Party Secure Inference."** ACM CCS 2020, arXiv:2010.06457 `[link]`
  URL: https://arxiv.org/abs/2010.06457
  A GC ReLU at 32 bits costs 4l-2 = 126 AND gates (Table 2): the literature check on `web/src/lib/clientHybridCnn.ts`'s 86. Added: 2026-11-08.
- **Huang, Lu, Hong and Ding, "Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference."** USENIX Security 2022 `[link]`
  URL: https://www.usenix.org/system/files/sec22-huang-zhicong.pdf
  RLWE with coefficient encoding, shares over a ring; 12 fractional bits, the figure this project's fixed point follows. Added: 2026-11-08.
- **Roh et al., "Flash: A Hybrid Private Inference Protocol for Deep CNNs with High Accuracy and Low Latency on CPU."** arXiv:2401.16732v2 `[link]`
  URL: https://arxiv.org/abs/2401.16732
  ReLU replaced by $`x^2 + x`$ with a retrained, batch-normalised network, the activation protocol squaring a masked integer in BFV. Considered and not adopted for the medical CNN (`docs/models-and-benchmarks.md`). Added: 2026-11-08.
- **Zhu, Jin, Aung and Xiao, "FESC: Remodeling Long-Context Private Inference with Encrypted State-Space Models."** arXiv:2608.17442v2 (2026-08-19), preprint `[link]`
  URL: https://arxiv.org/abs/2608.17442
  The same CKKS-to-MPC boundary as EncFormer and BLB, a mask uniform over the plaintext ring. Added: 2026-11-08.
- **Wang, Ranellucci and Katz, "Global-Scale Secure Multiparty Computation."** ACM CCS 2017 `[link]`
  URL: https://acmccs.github.io/papers/p39-wangA.pdf
  The authenticated garbling Polytune implements. Added: 2026-11-08.
- **Juvekar, Vaikuntanathan & Chandrakasan, "GAZELLE: A Low Latency Framework for Secure Neural
  Network Inference."** arXiv:1801.05507; USENIX Security 2018. Its "encryption switching protocols"
  convert between homomorphic and garbled-circuit encodings at each layer boundary by masking: the
  server adds a random mask to the encrypted linear output before the client decrypts it, exactly,
  over the scheme's plaintext ring. The shape `src/qsh/mpc/secret_inference.py` follows since Round
  75, with real-valued masks because CKKS is approximate. Until Round 84 the mask was one scalar per
  output and protected slot 0 only (the Kim and Lee entry above); since then it is independent in
  every slot, which is closer to GAZELLE's random element of the plaintext ring, though statistical
  rather than exact. Added: 2026-10-27; the slot-0 limit added 2026-10-30, and its fix in Round 84.
- **Mishra, Lehmkuhl, Srinivasan, Zheng & Popa, "Delphi: A Cryptographic Inference Service for Neural
  Networks."** IACR ePrint 2020/050; USENIX Security 2020. Inference between two parties "without
  revealing either party's data", with linear layers on additive secret shares prepared under
  homomorphic encryption and ReLU in garbled circuits; neither party holds an intermediate in the
  clear. Cited for the same conversions. Added: 2026-10-27.

**FHE maturity in 2026, for calibration rather than as a citation to lean on.** Surveys and vendor
retrospectives converge on roughly 100x-1000x overhead against plaintext, with what actually ships
being narrow: small-model inference - logistic regression, trees, small networks - completing in
seconds under CKKS. That is the honest frame for this project's T1 tier. The distilled students here
are small **because that is what runs under encryption in one browser tab, on single-threaded
WebAssembly**, not because a larger model was attempted and abandoned, and the measured
teacher-student gaps are reported rather than hidden. On servers much larger models run under
encryption: Park et al. (Round 31 additions, below) summarise 128 encrypted tokens of Llama-3-8B in
20 s on eight GPUs. Corrected 2026-10-30: this said "because that is what runs under encryption
today", which those server results contradict.

## Collaborative AML under FHE: the nearest published work to this repository's finance use case

Added 2026-08-11, verified at the arXiv abstract page.

- **Effendi & Chattopadhyay, "Privacy-Preserving Graph-Based Machine Learning with Fully
  Homomorphic Encryption for Collaborative Anti-Money Laundering."** arXiv:2411.02926, 5 Nov 2024
  (rev. 11 Nov 2024); SPACE 2024, DOI 10.1007/978-3-031-80408-3_6. TFHE via Zama Concrete ML, over
  two pipelines: a GNN with quantisation and pruning, and **graph-based XGBoost using a Graph
  Feature Preprocessor** - which is the same architecture as this repository's `finance_aml.gbm`
  teacher. Reports over 99 percent accuracy, F1, precision and recall on a **balanced** AML
  dataset in both plaintext and encrypted settings, and an 8 percent F1 improvement from graph
  features on the imbalanced version. Added: 2026-08-11.

**Read those two numbers together, because the pair is the lesson.** This repository's finance
teacher scores minority-class F1 **0.1519** at AMLworld's real prevalence of roughly 1 in 981 (0.6161 since
Round 86, on 114 inputs; the lesson below is unchanged). The
published work scores over 0.99 on a balanced set. Both are correct; they are not the same task,
and a reader comparing them without the prevalence would conclude this implementation is six times
worse than the state of the art rather than measured on harder rows. It is the same reason this
project refuses to quote aggregate accuracy for AML at all - at that prevalence a model predicting
"never launder" scores 99.9 percent accuracy - and it is why the finance page prints
minority-class F1 against an all-positive baseline instead.

**One practical note on reuse.** That paper's FHE layer is Zama Concrete ML, which this project
evaluated and declined. The licence objection stated more precisely, re-read at the primary source
2026-08-11: tfhe-rs is **BSD 3-Clause Clear**, whose distinguishing clause is "NO EXPRESS OR IMPLIED
LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE" - the copyright grant is permissive
and the **patent grant is expressly withheld**, which is a sharper and checkable objection than the
"require a paid licence" this file previously asserted about the vendor's terms. And separately,
Concrete ML's own release cadence had stalled at the time of the audit. So the nearest published
approach to this use case is built on a stack this repository cannot adopt commercially, which is
part of why the CKKS path here is TenSEAL and OpenFHE.

**Industry deployment, recorded as a vendor claim and nothing stronger.** Duality Technologies
publishes a case study in which two competing global banks jointly trained an AML model under its
platform without exchanging customer data, reporting a 34 percent improvement in detection. The
figure appears in the vendor's own marketing material and no independent write-up of the
engagement was locatable, so it is carried here as **`[UNVERIFIED - vendor claim]`** - useful as
evidence that cross-institution private AML training is being sold and bought in 2026, not as a
benchmark number. Checked 2026-08-11.

## Quantum machine learning: results that constrain what a QCNN can claim

Added 2026-08-11. Both verified at the arXiv abstract page.

- **Bermejo, Braccia, Rudolph, Holmes, Cincio & Cerezo, "Quantum Convolutional Neural Networks are
  Effectively Classically Simulable."** **PRX Quantum 7, 020304 (2026), DOI 10.1103/8qt9-72ts**;
  arXiv:2408.12739, 22 Aug 2024, revised 20 Feb 2026. Upgraded from the preprint citation on
  2026-08-13: the journal reference and DOI were read from the arXiv API's `journal_ref` and
  `arxiv:doi` fields, so this is now a peer-reviewed citation rather than a preprint one. Randomly
  initialised QCNNs can only operate on the information in **low-bodyness measurements** of their
  input states, and their behaviour is efficiently classically simulable by an algorithm equipped
  with Pauli shadows. The authors argue QCNN successes largely reflect benchmark simplicity, and that
  non-trivial datasets are a necessary ingredient for progress in QML. **Why it is cited here, and
  the section that applies.** This repository's QCNNs take classical data in a product angle
  embedding, one RY per feature, which is the case of the paper's Sec. IV.B, "Classical datasets"
  (pp. 020304-10 and -11, Table I; Sec. IV.2 in the arXiv HTML): with a tensor-product embedding the
  whole model is classically simulable with no quantum device, shown at n = 8 qubits on pairs of
  MNIST, Fashion-MNIST, EuroSAT and GTSRB classes, and the authors conclude that QCNN-based QML for
  classical data "appears to be an ill-motivated task" (read in the published PDF 2026-10-30). The
  finance and medical circuits here are moreover trees of two-qubit blocks on disjoint pairs, which a
  chain of small density-matrix maps reproduces exactly at any width (all six artefacts shipped
  before Round 84 to 1.6e-6, checked 2026-10-30); telecom's circuit since Round 84 is Hur, Kim and
  Park's SU(4) block on every neighbouring pair, which is not a tree, and at 7 or 8 qubits the state
  vector is a few hundred amplitudes anyway.
  **The paper does not predict that logistic regression beats the circuit; that is measured here.**
  The trained QCNN loses to or ties logistic regression on the identical split (finance
  minority-class F1 **0.228571** vs **0.439024** at the shipped seed since its Round 84 retrain on
  the four columns the linear model reads, and over five seeds the circuit leads on 2 and trails on
  3, a mean difference of +0.0426 that five seeds do not separate; medical accuracy **0.883721** vs **0.953488**
  until Round 84, and since its retrain at c = 0.5 a tie, **0.953488** each at the shipped seed, with
  logistic regression ahead by 0.0186 over five seeds), and Bowles
  et al. (next entry) report the same direction across 160 datasets. Corrected 2026-10-30: this said
  the paper "is the reason that is the expected result", resting on its random-initialisation result
  (Sec. III.A), which assumes independent random gates that the trained circuits do not have. The
  QCNN model cards quote it. The 8-class telecom circuit added in Round 34 lands the same way - macro F1
  **0.510963** against a multinomial logistic regression at **0.976447** on the identical split, with
  all 8 classes predicted (the Round 84 SU(4) circuit narrows that to **0.944439** at 8 wires and
  **0.967932** at 7, still behind) - which matters because it is a *different* readout (a 3-qubit basis
  marginal rather than a single $`\langle Z_0 \rangle`$ expectation) and a different task shape, and it did not change
  the direction of the result. **The figures in this bullet were 0.7326, 0.7143 and 0.8098 until
  Round 47, and then 0.860465, 0.84058 and 0.458324 for part of Round 47 itself**: measured first on
  a hardware-efficient ansatz (replaced in Round 46), then through a multiclass readout that binned
  the class index over a wire pooling had discarded (corrected in Round 47, which also made the
  fully-connected block per-survivor), and then on a circuit carrying two classes of analytically
  dead parameter (reordered later in the same round). All six artifacts were retrained twice; the
  current values are read from `models/*/weights.json`, which each card is now gated against. **The
  direction never moved - the classical baseline wins on all three use cases before and after every
  one of those changes - and only its size did**, which is the only thing this citation was ever
  quoted for.
  Added: 2026-08-11; telecom result and journal reference added 2026-08-13; metrics refreshed after
  each of the two Round 47 retrains.
- **Bowles, Ahmed & Schuld, "Better than classical? The subtle art of benchmarking quantum machine
  learning models."** arXiv:2403.07059, 11 Mar 2024, revised 14 Mar 2024. **Preprint - the arXiv API
  reports no `journal_ref` and no DOI, checked 2026-08-13**, so cite it as a preprint rather than a
  paper. 12 quantum models across 6 binary tasks and 160 datasets. Two findings: out-of-the-box
  classical models outperform the quantum classifiers overall, and **"removing entanglement from a
  quantum model often results in as good or better performance, suggesting that 'quantumness' may not
  be the crucial ingredient"**. **Why it is cited here:** the second finding is a claim about
  circuits, and this project has one, so it was CHECKED rather than repeated. Measured over five
  seeds per binary model with the CNOT ring skipped and nothing else changed - median minority-class
  F1 falls 0.714286 to 0.048544 on finance (entangled wins 5 of 5) and 0.870968 to 0.833333 on
  medical (4 of 5); telecom's 8-class macro F1 falls 0.809846 to 0.467059 on a single seed. **So the
  tendency they report does not reproduce on this circuit** - which is a statement about this
  circuit, not a counterexample to a survey of 160 datasets. Full workings, including the one-seed
  reading that pointed the opposite way before the sweep:
  [`qcnn-entanglement-ablation.md`](qcnn-entanglement-ablation.md). Note this cuts AGAINST the
  entry below it only in mechanism, not in verdict: entanglement mattering to a model that still
  loses to logistic regression is not evidence of advantage.
  **THOSE FIVE FIGURES BELONG TO A CIRCUIT THIS PROJECT NO LONGER SHIPS**, and they have not been
  re-measured. They were taken on the hardware-efficient ansatz - RY on every wire plus one CNOT ring,
  which `--no-entangle` removed completely - that Round 46 replaced with a real QCNN and Round 47
  corrected TWICE. On the current circuit the flag is also **not an exact ablation**: it gates the
  conv filter's CNOT and the survivor ring in the fully-connected block, but the pooling stage is a
  controlled rotation, which entangles, and it is not gated. So the sentence "the tendency does not
  reproduce on this circuit" now names a circuit that no longer exists, and re-running it is open
  work rather than a result carried forward. See the supersession banner on
  [`qcnn-entanglement-ablation.md`](qcnn-entanglement-ablation.md).
  Added: 2026-08-13; superseded-circuit note added in Round 47.
- **Li, Wu, Li & Yung, "Barren Plateaus Beyond Observable Concentration."** arXiv:2603.18479, 19 Mar
  2026, revised 19 Jun 2026. Unifies existing barren-plateau explanations under *observable
  concentration* and proves that avoiding it is **necessary but not sufficient** for trainability,
  identifying information loss outside the measurement light cone and correlated forward/backward
  blocks as further sources of gradient suppression. **Why it is cited here:** the untrained
  Clifford+T circuit this project removed from its inference path showed exactly the concentration
  symptom - $`\lvert\langle Z_0\rangle\rvert`$ within $`\pm 0.21`$ of zero for every input tested, with sign set by numerical noise -
  and this paper names the mechanism and warns that fixing concentration alone would not have made it
  trainable. Added: 2026-08-11.

## QFL - quantum federated and delegated learning (medical and finance use cases)

- **Quantum Delegated and Federated Learning via Quantum Homomorphic Encryption** - Li and Deng,
  *Research Directions: Quantum Technologies* **3**, e3, DOI `10.1017/qut.2025.2` (the bundled PDF, CC BY 4.0, is this version of
  record); preprint arXiv:2409.19359, 2024 `[R]`
  URL: https://doi.org/10.1017/qut.2025.2
  Added: 2026-06-27
  File: `Quantum_Delegated_and_Federated_Learning_via_Quantum_Homomorphic_Encryption.pdf`
  QOTP-encrypted delegated and federated learning; informs the `src/qsh/ai/federated.py` privacy
  layer. **What that layer's pairwise masks do, stated 2026-11-01.** In both tiers each client draws
  an X25519 key pair per run and each pair of clients derives its mask seed from their shared secret
  by HKDF-SHA256, the pairwise key agreement of Bonawitz et al. (Round 83 additions, below), so the
  aggregator, which receives only the public keys and the masked updates, cannot remove a mask on its
  own. Their double masking and secret-shared keys for dropouts, their malicious-security protocol and
  any authentication of the public keys are not implemented, and every client runs in one tab or one
  process; since Round 85 both tiers mask modulo $`2^{32}`$, so the masks cancel exactly
  ([what each step protects](../docs/threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment)).

- **Quantum Federated Learning: A Comprehensive Survey** - Nguyen, Uddin, Shaon, Rahman, Dobre and
  Niyato, arXiv:2508.15998, 21 Aug 2025 `[R]`
  URL: https://arxiv.org/abs/2508.15998
  Added: 2026-06-27
  **Preprint, under revision at IEEE COMST; not published.**
  File: `Quantum_Federated_Learning_A_Comprehensive_Survey.pdf`
  Taxonomy of fully, hybrid and hierarchical QFL, frameworks (PennyLane, Flower, NVFlare) and
  datasets. Design reference for the federated-learning architecture.

---
- **Rajabi and Toorani, "Secure Aggregation for Privacy-Preserving Federated Learning on Clinical EEG
  Data"**, arXiv:2607.28191 v2, 2026-09-07; a version is to appear at HotDiSec 2026 (ESORICS workshop)
  `[link]`
  URL: https://arxiv.org/abs/2607.28191
  Added: 2026-11-08
  Read in the arXiv v2 PDF. Four variants of Bell et al.'s SecAgg+ on Flower (semi-honest, malicious,
  each with an optional notary): pairwise masks over a sparse graph, a self-mask, Shamir-shared
  recovery and L2 clipping, on TUH Abnormal EEG with up to 70 simulated clients. No dropouts or active
  server behaviour were injected, so recovery and the malicious checks are built but not measured.
  Cited at `secureAggregate` in `web/src/lib/clientQFL.ts` as what a complete version adds; this
  project's all-pairs masks with no self-mask are not that protocol.
- **Gholizade, Ruffini, Ducange and Marcelloni, "Federated Explainable Artificial Intelligence: Roles,
  Architectures, Evaluation, and Open Challenges"**, Expert Systems with Applications 331, 133183
  (2026), DOI 10.1016/j.eswa.2026.133183; arXiv:2607.13045 v1, 2026-06-15 `[link]`
  URL: https://doi.org/10.1016/j.eswa.2026.133183
  Added: 2026-11-08
  Read in part in the arXiv PDF. A survey and taxonomy of federated explainability. It lists
  explanation-centric security threats among its open challenges, meaning leakage through explanations
  and explanation poisoning by adversarial clients, and it covers secure aggregation, SMPC and
  homomorphic encryption applied to SHAP values. Cited for that gap only. This project's per-run ML-DSA
  signature on an explanation detects accidental change, and addresses neither of those threats.

## QKD networking standards implemented in Round 18

Added in Round 18 and previously recorded only in source docstrings.

- **ETSI GS QKD 015 V2.1.1 (2022-04)** - SDN control interface for a software-defined QKD network
  `[link]`
  URL: https://www.etsi.org/deliver/etsi_gs/QKD/001_099/015/02.01.01_60/gs_QKD015v020101p.pdf
  YANG source: https://forge.etsi.org/rep/qkd/gs015-ctrl-int-sdn (revision 2022-01-30,
  BSD-3-Clause), the `v2.1.1` tag, which the spec's Annex A names. The spec PDF gives the URLs and not the
  module text. Forge served an incomplete certificate chain until at least August 2026; on 2026-11-03 it
  verified.
  Added: no PDF bundled; spec and YANG re-verified 2026-08-04
  Implemented by `src/qsh/qkd/sdn_controller.py` and mirrored in the browser by
  `web/src/lib/clientSdnController.ts`. Field names (`qkdn_id`, `qkdl_id`, `qkdl_type`,
  `virt_next_hop`, `phys_channel_att`, `skr`, `eskr`) are taken verbatim from the YANG; nothing is
  renamed to read more nicely. Clause 1 is what settles the agent question: "Application layer
  interfaces and quantum-channel interfaces are out of scope", so an MCP agent is a **SAE** (Secure
  Application Entity), an application-plane key consumer, never the controller.

- **ETSI GS QKD 014 V1.1.1 (2019-02)** - key delivery API `[link]`
  URL: https://www.etsi.org/deliver/etsi_gs/QKD/001_099/014/01.01.01_60/gs_QKD014v010101p.pdf
  Added: no PDF bundled; edition status re-verified 2026-08-04
  Implemented by `src/qsh/qkd/kme.py`. **V1.1.1 is still the only published edition**; an ed2 stable
  draft (v1.3.1) existed as of 2025-06 but is not published, so any "QKD 014 compliant" claim,
  including this one, targets the 2019 document. The spec mandates mutual TLS with the SAE identity
  taken from the client certificate, which an in-process model has no transport to attach to; that
  gap is stated in `KeyManagementEntity.authentication_note()` and in `docs/threat-model.md` rather
  than left for a reader to notice.

- **ETSI GS QKD 020** - interoperable KMS API `[link]`
  URL: https://forge.etsi.org/rep/qkd/gs020-interop-kms
  Also: https://www.etsi.org/committee/qkd - the committee landing page, which is what this entry
  used to point at. It is not the spec, and an audit of this file flagged it as one of two entries
  with no resolvable identifier. The forge repository carries the API definition itself.
  Added: publication date re-verified 2026-08-04; URL corrected 2026-08-07
  Published **2026-06-29**, not 2024 as is widely repeated, and it covers the KMS-to-KMS interface
  *within a trusted node* rather than general inter-domain transport. Not implemented here; recorded
  so the roadmap cites the right scope.

- **ITU-T Y.3804 / Y.3805** - QKDN control and management `[link]`
  URL: https://www.itu.int/rec/T-REC-Y.3804/en
  Added: re-verified 2026-08-04
  Cited in `docs/pqc-roadmap.md` as the ITU counterpart to the ETSI control plane. Not implemented.

---
- **Barral et al., "Interconnecting Regional QKD Networks: Hybrid Key Delivery Across Quantum
  Domains"**, IEEE Transactions on Quantum Engineering 7, 4102117 (2026), DOI
  10.1109/TQE.2026.3728800; arXiv:2604.20376 v2 `[link]`
  URL: https://doi.org/10.1109/TQE.2026.3728800
  Added: 2026-11-08
  Read in the arXiv v2 PDF. A key-management overlay on ETSI GS QKD 014 and 020 joining QKD islands
  over classical links, each hop keyed by the XOR of a Kyber secret and, where a link exists, a QKD
  key. Eight nodes on four QKD links at four sites in two Spanish regions, about 500 km apart (the
  abstract says three regional subnetworks; the body says two regions and four sites). About 2-3 kb/s
  pairwise, about 0.5 kb/s over the 120 km link. Routes are computed once at start-up and the
  controller functions sit inside the key-management nodes, so this is not a GS QKD 015
  implementation. The authors disclaim information-theoretic security on any path with a PQC-only hop.

## External standards and threat context (see `docs/pqc-roadmap.md`)

- **CycloneDX 1.7** (Ecma-424, second edition), Bill of Materials specification, release 1.7.2, 2026-09-17, `cryptographic-asset` components and the algorithm-family registry `[link]`
  URL: https://github.com/CycloneDX/specification
  Added: 2026-11-08
  The schema is vendored in `schemas/cyclonedx/` at tag 1.7.2 and read there. Ecma's page lists ECMA-424, second edition, December 2025, as defining CycloneDX v1.7. `web/public/cbom/qsh.cdx.json` follows it.
- NIST **FIPS 203/204/205** (ML-KEM, ML-DSA, SLH-DSA), 2024.
  URL: https://csrc.nist.gov/pubs/fips/203/final (204, 205 by the same pattern)
- NIST **SP 800-90B** - entropy sources, including the clause 4.4 continuous health tests
  (Repetition Count, Adaptive Proportion) that `src/qsh/qrng/health.py` implements.
  URL: https://csrc.nist.gov/pubs/sp/800/90/b/final
- NIST **IR 8547**, *Transition to Post-Quantum Cryptography Standards*, **final, October 2026**
  (announced 2026-10-27), Moody, Perlner, Regenscheid, Robinson and Cooper, doi:10.6028/NIST.IR.8547.
  It follows the initial public draft of 2024-11-12.
  URL: https://csrc.nist.gov/pubs/ir/8547/final (PDF:
  https://nvlpubs.nist.gov/nistpubs/ir/2026/NIST.IR.8547.pdf)
  VERIFIED in the final PDF 2026-10-30. Table 2 (quantum-vulnerable signatures: ECDSA and RSA at 112
  bits of classical security or more, EdDSA at 128 or more) and Table 4 (key establishment:
  finite-field and elliptic-curve DH and MQV, and RSA, at 112 bits or more) read *"Deprecated after
  2030"* and *"Disallowed after 2035"*, so RSA-3072 and P-256 are deprecated after 2030 as well, which
  the draft did not do. Section 4: *"NIST will deprecate quantum-vulnerable public-key cryptography
  after 2030 and disallow its use under NIST standards and guidelines after 2035"*, and the window
  *"does not alter the earlier deadlines"* that EO 14412 and OMB M-26-15 set for high value assets and
  high impact systems (PQC key establishment by 2030-12-31, signatures by 2031-12-31). It replaces SP
  800-57 Part 1's projected disallowance of 112-bit public-key schemes on 2031-01-01. Deprecated means
  the algorithm may still be used at a risk the data owner must examine.
- NSA **CNSA 2.0** - national security systems only (*"NSA does not mandate requirements on any
  other entity outside of NSS"*). Under CNSSP 15, CNSA 2.0 algorithms are required in all new NSS
  acquisitions from 2027-01-01, equipment that cannot support them is phased out by 2030-12-31, and
  CNSA 2.0 is mandatory for all use by 2031-12-31. On QKD, the CNSA 2.0 and Quantum Computing FAQ
  (Ver. 2.2, October 2026) answers **"No"** to using it to protect a National Security System, and
  on hybrids it says NSA *"will not require NSS developers to use hybrid certified products"*. The
  sentence that NSA *"does not support the usage of QKD or QC to protect communications in National
  Security Systems"* is not in the FAQ (it was attributed to the FAQ until 2026-10-24). It is on
  NSA's "Quantum Key Distribution (QKD) and Quantum Cryptography (QC)" page, and the archived copy
  of the Post-Quantum Cybersecurity Resources address carries the same text (Wayback copies of
  2026-10-16 and 2026-10-20, read 2026-10-30, because nsa.gov returns HTTP 403 to scripts). NSA does
  **not** recommend a PQC-hybrid QKD add-on, and neither does BSI: TR-02102-1 v2026-01 recommends a
  quantum-safe plus classical hybrid (section 2.2) and does not recommend QKD (section 2.1). The
  optional QKD add-on in this repository is the project's own choice.
  URL (FAQ Ver. 2.2): https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0__FAQ_v2.2.PDF
  (linked from https://www.nsa.gov/Cybersecurity/Post-Quantum-Cryptography-Resource-Hub/; nsa.gov
  returns HTTP 403 to scripts, so the 2026-10-23 check read an archived copy). The same PDF, byte for
  byte (U/OO/6072712-26, October 2026), is also served at
  https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF, the address search
  indexes list (Wayback copies of 2026-10-20 and 2026-10-23, compared 2026-10-30).
  URL (resources page): https://www.nsa.gov/Cybersecurity/Post-Quantum-Cybersecurity-Resources/
  URL (QKD and QC page): https://www.nsa.gov/Cybersecurity/Quantum-Key-Distribution-QKD-and-Quantum-Cryptography-QC/
  The URL this entry used to give, `nsa.gov/Cybersecurity/Post-Quantum-Cryptography-FAQs/`, could not be
  confirmed (UNVERIFIED - needs confirmation) and is no longer cited.
- **BSI TR-02102-1**, version 2026-01.
  URL: https://www.bsi.bund.de/EN/Themen/Unternehmen-und-Organisationen/Standards-und-Zertifizierung/Technische-Richtlinien/TR-nach-Thema-sortiert/tr02102/tr02102_node.html
- **HNDL** (Harvest Now, Decrypt Later) and Q-Day acceleration - Gidney, arXiv:2505.15917, 2025.
  URL: https://arxiv.org/abs/2505.15917
- **The Business of Quantum Technologies** - copyrighted book, `[R-local]`, deliberately not
  redistributed (`.gitignore` rule `references/The Business of Quantum Technologies.pdf`). Background
  reading only; no technical claim in this repository rests on it.

---
- ANSSI, BSI, NLNCSA and the Swedish National Communications Security Authority (Swedish Armed Forces),
  **Position Paper on Quantum Key Distribution**, January 2024 (BSI dates it 2024-01-26) `[link]`
  URL: https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/Crypto/Quantum_Positionspapier.html
  Added: 2026-11-08
  Read in full in the BSI copy. QKD is usable only in niche cases for now and lacks protocol standards,
  comprehensive security proofs and evaluation methods; PQC and symmetric keying take priority. The two
  fragments quoted in `src/qsh/qkd/sdn_controller.py` were compared byte for byte with the executive
  summary and match. Section 5 states the priority as PQC in hybrid solutions.

## OSS reused

Licences below are what each project publishes. **They are not uniformly commercial-OK** - `PaySim` is
GPL-3.0, so the previous heading ("all commercial-OK") was wrong on its own table. PaySim is cited
only, not a dependency. Everything this project actually installs is MIT, Apache-2.0 or BSD.

**MP-SPDZ correction, 2026-08-11.** This paragraph used to say MP-SPDZ "carries a CSIRO licence" and
listed it beside PaySim as a reason the set is not commercial-OK. Read at the primary source, its
`License.txt` says: *"CSIRO grants you a licence to the Software on the terms of the BSD 3-Clause
Licence."* The top-level grant is therefore plain BSD-3-Clause and belongs in the permissive set. The
real caveat is narrower and was attached to the wrong thing: **bundled third-party components** inside
the repository (SPDZ-2 / Bristol, SPDZ-BMR-ORAM / Bristol and Bar-Ilan, and others) carry their own
terms, which is what needs checking before depending on a specific protocol. Recorded because this
file's stated purpose is to stop exactly this kind of drift, and it had drifted in the direction of
being needlessly restrictive - which is still being wrong.

`scripts/bootstrap_submodules.sh` fetches the subset that is still wired up; the submodules removed
in Round 17 are marked below.

| Code | Used by | Licence |
|---|---|---|
| `OrtegaSA/CQC-QHE-repo` | cited only - `qsh/qhe/qcnn.py` is an independent implementation of the paper, not a wrapper | Apache-2.0 |
| `open-quantum-safe/liboqs` and `liboqs-python` | `qsh/pqc/*` | MIT |
| `openfheorg/openfhe-development`, `microsoft/SEAL` | `qsh/fhe/*`, benchmark | BSD-2 / MIT |
| `tenseal` | `qsh/fhe/*`, benchmark | **Apache-2.0**, confirmed from the installed package metadata; an earlier revision of this table listed it under BSD-2/MIT |
| `enCRYPTON-labs/HEonGPU` (moved from `Alisah-Ozcan/HEonGPU`), `chart21/hpmpc` (PIGEON) | git submodules. HEonGPU v2.1.1 is the GPU CKKS backend since Round 86, through this project's binding (`native/heongpu_py`, `qsh/fhe/backends.py`), measured in `docs/models-and-benchmarks.md`; hpmpc is not run yet. `kanav99/shark`, listed here until 2026-10-30, was removed in Round 17 and nothing uses it | Apache-2.0 / MIT |
| `data61/MP-SPDZ` | not in the repository (its submodule was removed in Round 17, unreferenced); cited as a CPU MPC reference only | BSD-3-Clause, per CSIRO's grant (the correction above); its bundled third-party components carry their own terms |
| `lschoe/mpyc` | `qsh/mpc/secret_inference.py` - non-linear MPC (ReLU, argmax, max-pool on shares), run as three party processes by `qsh/mpc/parties.py` since Round 75. A normal PyPI dependency in `requirements-fhe.txt`; the submodule was dropped in Round 17 and nothing is vendored | MIT |
| `IBM/AMLSim`, `synthetichealth/synthea` | **cited only, not used.** Round 17 removed both submodules as unreferenced; `qsh/data/synthetic.py` mentions them in prose only and no code path calls them | Apache-2.0 |
| `EdgarLopezPhD/PaySim` | cited only, never cloned or vendored | **GPL-3.0** for the simulator source; the Kaggle output dataset is separately CC-BY-SA-4.0 |
| `openfhe` (OpenFHE 1.6.1, CKKS with bootstrapping) | `qsh/fhe/backends.py` (`OpenFHEBackend`) | BSD-2-Clause |
| **`s0l0ist/node-seal`** (SEAL compiled to WASM, CKKS) | `web/src/lib/clientFHE.ts` - real in-browser FHE | MIT |
| **`polytune`, `garble_lang`** (sine-fdn) | `web/src/lib/clientMPC.ts` via `wasm/qsh-mpc` - real in-browser garbled-circuit 2PC (replaced `mpc-framework` + `emp-wasm-engine` + `summon-ts` in Round 65); dependency notices in `/licenses/qsh-mpc.txt` | MIT |
| **`mcp`** (MCP Python SDK; the class is `MCPServer`, named `FastMCP` before v2.0.0) and the official `@modelcontextprotocol/server-filesystem` | `qsh/mcp/*` - real quantum-safe MCP server and interop | MIT |
| **`onnxruntime-web`** | `web/src/lib/onnxInference.ts` - browser inference. The `webgpu` and `wasm` builds are separate bundles; the demo uses WASM unless the visitor selects WebGPU and a real adapter exists | MIT |
| **`@huggingface/transformers`, `@mlc-ai/web-llm`** | `web/src/lib/clientAI.ts` - the Agentic AI panel's engines, dynamically imported. Transformers.js's ONNX Runtime wasm comes from this site. WebLLM's model libraries are built here from MLC-LLM and TVM at the commits upstream's own libraries name (`scripts/build_webllm_libs.sh`; `web/public/webllm/*/MANIFEST.json`), and the models' licences are in `DATA_LICENSES.md` section 5b | Apache-2.0 |
| `PennyLane` | `qsh/ai/federated.py` - quantum federated learning (`kind="qfl"`), and `qsh/qml_ids/detector.py`. The FedAvg and secure-aggregation code is this repository's own; Flower is cited as prior art, not used as a dependency | Apache-2.0 |

## Baselines and corpora cited in code but previously absent here

Round 21 audit: these are quoted as authority in registry docstrings, model cards or the SPA's baked
config, and had no entry. A baseline a page prints beside a measured number is a citation, so it
belongs in this file rather than only in a comment.

- **Ruszczak, Kotowski, Evans & Nalepa, "The OPS-SAT benchmark for detecting anomalies in
  satellite telemetry", *Scientific Data* 12, 710 (2025).** DOI 10.1038/s41597-025-05035-3. Data:
  Zenodo 10.5281/zenodo.12588359, version 1, CC BY 4.0: the record the dataset's README cites and
  the file the loader fetches. The concept DOI 10.5281/zenodo.12588358 resolves to version 2 (record
  15108715), whose files differ and which adds an MIT licence file from KP Labs (`DATA_LICENSES.md`,
  read 2026-10-31). Source of the 30 published baselines `satellite_comms.opssat` is quoted against.
  Added: 2026-08-07.
- **Kotowski, Haskamp, Andrzejewski, Ruszczak, Nalepa et al., "European Space Agency Benchmark for
  Anomaly Detection in Satellite Telemetry".** arXiv:2406.17826. Data: Zenodo
  10.5281/zenodo.12528696, CC BY 3.0 IGO. Source of the 0.968 event-wise F0.5 baseline
  `satellite_comms.telemanom` is quoted against - Supplementary Table 9 (anomalies alone), NOT
  Table 2 (all events), which is a different comparator and was once cited by mistake.
  Added: 2026-08-11.

> **These are two different papers by an overlapping group, and conflating them is an easy mistake to
> make - one that was made here.** OPS-SAT-AD is **Ruszczak** first (Scientific Data, Zenodo
> ...12588359, CC BY 4.0, the corpus `satellite_comms.opssat` trains on). ESA-ADB is **Kotowski**
> first (arXiv:2406.17826, Zenodo ...12528696, CC BY 3.0 IGO, the corpus `satellite_comms.telemanom`
> is scored against). Thirteen places in this repository credited OPS-SAT-AD to "Kotowski et al.",
> and the 2D overlay credited Ruszczak with the title of an unrelated algebraic-geometry paper. Both
> were fixed by checking Crossref and Zenodo, not by checking another file here. Before bulk-editing
> either name, confirm WHICH benchmark the line is about: the DOI and the Zenodo id are the reliable
> discriminators, and the author surname is not.
>
> A third variant to expect: the ESA-ADB **Zenodo record** lists **De Canio** first (contact person),
> then Kotowski and Haskamp as researchers, while the **paper** is Kotowski first. Citing the paper
> for the benchmark and the Zenodo DOI for the data - as the two entries above do - is correct and is
> not an inconsistency. Verified 2026-08-11 against Crossref, the arXiv listing and both Zenodo
> records. Note that Zenodo's legacy REST JSON (`/api/records/12528696`) returns no licence for
> ESA-ADB, while its InvenioRDM JSON (the record's JSON export) carries `CC BY 3.0 IGO` with the
> legal-code link (both read 2026-10-31), so a check of the legacy API alone reads the corpus as
> unlicensed.

- **Yang et al., "MedMNIST v2", *Scientific Data* 10:41 (2023).** DOI 10.1038/s41597-022-01721-8.
  Source of the BreastMNIST ResNet-18 AUC 0.901 / ResNet-50 AUC 0.857 baselines. Added: 2026-08-07.
- **Doerrich, Di Salvo, Brockmann and Ledig, "Rethinking model prototyping through the MedMNIST+ dataset
  collection"**, *Scientific Reports* 15, 7669 (2025), DOI 10.1038/s41598-025-92156-9; arXiv:2404.15786 `[link]`
  URL: https://doi.org/10.1038/s41598-025-92156-9
  Added: 2026-11-08
  Read in the arXiv v3 PDF: Table S2 and the training pipeline; the journal's supplement was not opened. BreastMNIST,
  end-to-end training, ImageNet-pretrained ResNet-18, images zero-padded from 28 by 28 to 224 by 224: AUC
  $`87.61 \pm 1.16`$ and accuracy $`83.97 \pm 1.81`$ (percent), over three seeds. Cited in `docs/findings.md` for the
  seed-to-seed spread a BreastMNIST AUC carries; this project's pipeline differs, so the numbers are not like for like.
- **Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models",
  NeurIPS 2023 Datasets and Benchmarks.** arXiv:2306.16424. AMLworld; source of the 0.6323 finance
  baseline and of the corpus's own synthetic-by-construction statement. Added: 2026-08-07.
- **Egressy et al., "Provably Powerful Graph Neural Networks for Directed Multigraphs" (Multi-PNA+EU),
  AAAI-24, vol. 38 no. 10, pp. 11838-11846.** DOI 10.1609/aaai.v38i10.29073. Source of the 0.6816
  finance ceiling. Added: 2026-08-07.
- **Liu, Qinghua and Paparrizos, John, "The Elephant in the Room: Towards A Reliable Time-Series
  Anomaly Detection Benchmark" (TSB-AD), NeurIPS 2024 Datasets and Benchmarks Track.** The
  proceedings entry IS the citation - **TSB-AD has no arXiv ID at all**, so there is no preprint
  identifier to give. Cited here as the benchmark that motivates reporting event-wise F-beta and never
  point-adjusted F1 as a headline (`src/qsh/models/metrics.py`, `event_wise_fbeta`). The specific
  point-adjustment number this entry used to quote is withdrawn; the replacement is the Kim et al.
  entry below. Added: 2026-08-07, **corrected 2026-08-12**.

> **This entry cited a different paper and quoted a number that is in neither.** Until 2026-08-12 it
> read "Liu et al. ... **arXiv:2412.20512**". That arXiv ID belongs to **Boniol, Liu, Huang, Palpanas
> and Paparrizos, "Dive into Time-Series Anomaly Detection: A Decade Review"** - a survey, not TSB-AD.
> The two papers share authors, which is how the conflation happened, and it is the same failure mode
> as the OPS-SAT / ESA-ADB mix-up above: an overlapping author set and a surname that is not a
> discriminator. Here there is no identifier to swap in, because TSB-AD never had one; an entry that
> carries an arXiv ID for TSB-AD is wrong by construction.
>
> **The "uniform random score ranks 26th of 32 detectors under point adjustment" claim is WITHDRAWN as
> unverified.** It could not be found in TSB-AD or anywhere else, and TSB-AD's own proceedings abstract
> does not fit its shape: that abstract evaluates **40 detection algorithms**, not 32. The claim was
> load-bearing - it was the whole stated reason for a metric choice - so it is replaced by a verbatim
> one rather than softened. Verified 2026-08-12.
>
> **Resolved, checked 2026-10-30: none of the places listed below quotes the number as a finding any
> more.** Some no longer carry it, and the rest record its withdrawal. The paragraph is kept as it
> was written.
>
> **The withdrawn number was quoted in seven places outside this file. All are corrected since** (2026-11-08: only correction notes remain, at `src/qsh/models/metrics.py` and `scripts/train/satellite_telemanom.py`).
> Listed so the cleanup was a task rather than a discovery, because a correction that lands in the
> provenance file only is the weakest possible version of it: `README.md:823`;
> `report/technical-report.tex:201` and `:636`; `scripts/train/satellite_telemanom.py:10` and `:308`
> (which is what writes the string into `models/metrics.json:273`, whose `metric_note` therefore carries
> it, and into `web/src/lib/config.defaults.json:699` via the generated config); and
> `src/qsh/models/metrics.py:19`, which is the worst of them because it adds an inference the source
> never supported - "i.e. it beats six real detectors". Those files were owned by another worker while
> this correction was made, so they were deliberately left alone. **The replacement text for all seven
> is the Kim et al. entry below**, and the substitution is not merely a swap of citations: the Kim
> sentences are about a random and an untrained SCORE, so any rewrite that keeps the "beats six real
> detectors" gloss reintroduces the fabrication.
>
> **TSB-AD recommends VUS-PR, and since Round 86 the satellite page reports it beside the event-wise
> F0.5.** The same abstract identifies **VUS-PR as the most reliable measure**. The headline stays the
> event-wise F-beta, because that is what the satellite baselines this project is scored against are
> computed with (ESA-ADB Supplementary Table 9), and switching measures would break comparability with
> the published number on the page. VUS-PR is computed beside it, offline, with TSB-AD 1.5's own code
> (`scripts/measure_satellite_vus_pr.py`, `docs/measurements/satellite-vus-pr-2026-11-14.json`) on the
> published forecaster's continuous scores: the threshold-free question the F0.5 does not ask. No
> published VUS-PR exists for ESA-ADB Mission1's lightweight channels (search of 2026-11-14), so the
> figure stands against shuffled and random-walk scores only, never against another detector.

- **Kim, Choi, Choi, Lee and Yoon, "Towards a Rigorous Evaluation of Time-series Anomaly Detection",
  AAAI 2022.** arXiv:2109.05257 `[link]`
  URL: https://arxiv.org/abs/2109.05257
  **The load-bearing citation for rejecting point-adjusted F1**, in place of the withdrawn TSB-AD number
  above, and stronger: verbatim, *"even a random anomaly score can easily turn into a state-of-the-art
  TAD method"*, and separately, *"an untrained model obtains comparable detection performance to the
  existing methods even when PA is forbidden."* The first sentence is the point-adjustment failure
  itself; the second is the harder finding, because it holds with point adjustment switched **OFF** - so
  the missing control in this field is an untrained baseline, not a better measure.
  **This paper was already cited in code and never in this file.** `src/qsh/models/metrics.py:17`
  quotes it as "Kim et al. (AAAI-22)" for a third sentence, that point adjustment *"has a great
  possibility of overestimating the detection performance"* - a quote this round did not re-verify, so
  it is recorded as the code's, while the two sentences above were verified at the arXiv abstract page
  on 2026-08-12. What the repository does with it: `metrics.py` reports **event-wise F-beta with its
  event count as the headline** and computes `point_adjusted_f1` **only as a companion, to show the
  inflation** - on the satellite run, `point_adjusted_f1` 0.993691 against `event_wise_f0_5` 0.454545
  (`models/metrics.json`), which is the whole argument in two numbers from one run. (That run's; the
  Round 71 retrain, without the whole-series z-score, scores 0 on both, because neither of its 2
  flagged points falls inside one of the 3 scored events.) That is
  the practice both papers argue for, and it does not change with this correction; only the citation
  supporting it does. Added: 2026-08-12.
- **Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
  Thresholding", KDD 2018.** arXiv:1802.04431. Telemanom. Added: 2026-08-07.
- **Ma, Qi, Zhao & Lo, "Practical decoy state for quantum key distribution", *Phys. Rev. A* 72,
  012326 (2005).** DOI 10.1103/PhysRevA.72.012326. The single-photon bound OptiQKD tunes. Added:
  2026-08-07.
- **Lopez-Rojas, Elmir & Axelsson, "PaySim: A financial mobile money simulator for fraud detection",
  EMSS 2016.** The bundled finance demo sample; a SIMULATOR, not real transactions. Added: 2026-08-07.
- **Sharafaldin, Habibi Lashkari & Ghorbani, "Toward Generating a New Intrusion Detection Dataset and
  Intrusion Traffic Characterization", ICISSP 2018.** CIC-IDS2017. Added: 2026-08-07.
- **Wolberg, Mangasarian, Street & Street, "Breast Cancer Wisconsin (Diagnostic)", UCI Machine
  Learning Repository (1993).** DOI 10.24432/C5DW2B. Relicensed by UCI to CC BY 4.0, so attribution
  is due. Added: 2026-08-07. Corrected 2026-10-30 to the creators, order and year the UCI page asks
  to be cited with; this entry read "Wolberg, Street & Mangasarian ... (1995)".
- **Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep
  Learning", *Cell* 172(5) (2018).** DOI 10.1016/j.cell.2018.02.010. The chest X-ray source behind
  PneumoniaMNIST. Added: 2026-08-07.
- **Mavaddat et al., "Polygenic Risk Scores for Prediction of Breast Cancer and Breast Cancer Subtypes",
  *AJHG* 104(1) (2019).**
  DOI 10.1016/j.ajhg.2018.11.002. The paper behind **PGS000004 (PRS313)**, which the genomics use case does not
  score: the article is CC BY-NC-ND 4.0 and the weights appear only in its supplementary data.
- **Graff et al., "Cross-cancer evaluation of polygenic risk scores for 16 cancer types in two large
  cohorts", *Nat Commun* 12:970 (2021).** DOI 10.1038/s41467-021-21288-z, CC BY 4.0. The paper behind
  **PGS000072**, the breast-cancer score the genomics use case scores (187 variants; the score sums
  risk-allele counts times the log odds ratio from each source GWAS). Its external baseline is **AUROC
  0.637** from PGS Catalog **PPM002038** (Kachuri et al., *Nat Commun* 11:6084, 2020,
  doi:10.1038/s41467-020-19600-4; n=216,273 UK Biobank women, with covariates, 162 of the 187 SNPs, AUC
  at 5 years), cited and explicitly not measured here. Added: 2026-10-31.
- **Lambert, Wingfield et al., "Enhancing the Polygenic Score Catalog with tools for score calculation
  and ancestry normalization", *Nat Genet* 56:1989-1994 (2024).** DOI 10.1038/s41588-024-01937-x. The
  citation the PGS Catalog's REST `/rest/info` asks for. Added: 2026-10-31.
- **Al Badawi, Alexandru, Polyakov and Vaikuntanathan, "SoK: Private LLM Inference using Approximate
  Homomorphic Encryption", IACR ePrint 2026/935**, revised 2026-10-23 (its third revision). Quoted in `models/distil.py` on
  why distillation is the practical route to FHE-feasible inference. Added: 2026-08-07, corrected
  2026-08-12. This entry also named `models/registry.py` until Round 33: Round 31 replaced that
  citation with the primary source (Park et al., arXiv:2601.18511v2), so the SoK has not appeared in
  the registry since, and two other entries in this file repeated the same stale attribution.
  **Extended 2026-08-12** with the half of it that bears on how this repository REPORTS rather than on
  what things cost (verified at the abstract, and extended here rather than filed twice): the SoK
  proposes a **"Private LLM Card System (PLCS)"** to standardise reporting because *"Frameworks differ
  in ciphertext packing layouts, model fidelity, software and hardware stacks, and reported metrics"*;
  it finds only **5 of the 21 surveyed frameworks model-preserving** (the revision of 2026-10-23); and
  it puts the runtime gap at roughly **four orders of magnitude**. That count is the figure that
  matters most here, because the
  plaintext-versus-CKKS agreement this demo prints IS a model-fidelity claim, and this is the survey
  saying most published work does not make one.
- **Lyu, "Did We Actually Fix It? An Independent Adversarial Stress-Test of Post-Point-Adjustment
  Evaluation Metrics for Time-Series Anomaly Detection", arXiv:2607.11969** (v1 2026-07-12, v2
  2026-07-19). Quoted in the satellite model card and baked into `config.defaults.json` as the
  reason the event-wise F0.5 non-reproduction is reported as metric instability rather than as a
  training failure. Added: 2026-08-07. **Read at the abstract 2026-10-30, it does not support that
  use.** Its subject is no-skill detectors gaming the metrics that replaced point adjustment (PA%K,
  range-based and affiliation precision and recall, VUS): under one honest run none is gameable on
  its six benchmarks, and under best-of-N reporting affiliation-F1 and every ROC-based metric inflate
  steeply while the PR-based metrics and PA%K stay near-flat. The event-wise F-score this project
  reports is not among the metrics it names. The local reason is enough on its own: three scored
  events cannot rank detectors (`satellite-event-count-investigation.md`).
- **EO 14412, "Securing the Nation Against Advanced Cryptographic Attacks"**, signed 2026-06-22,
  91 FR 38483. PQC key establishment on high-value assets and high impact systems by 2030-12-31,
  digital signatures by 2031-12-31 (section 4(b)); section 6(c) directs the FAR Council to publish a
  *proposed* rule within 180 days (about 2026-12-19) that would put covered contractors on FIPS,
  including the PQC FIPS, by 2030-12-31 - none in the Federal Register as of 2026-10-30 (this entry
  used to describe that rule as if it existed); National Security Systems excluded and remaining on
  CNSA 2.0. The binding US instrument for those systems. NIST IR 8547, final since October 2026
  (doi:10.6028/NIST.IR.8547, External standards above), sets NIST's own schedule - quantum-vulnerable
  public-key algorithms deprecated after 2030 and disallowed after 2035 - and leaves these earlier
  dates in place. Added: 2026-08-07; the IR 8547 sentence corrected 2026-10-30 (it called IR 8547
  "still a draft").

## Security scoring: sources evaluated and DECLINED (Round 22)

Five candidate replacements for this project's security score were read in full and all five were
declined, with reasons. Recorded in `security-scoring-fact-check.md` so the evaluation is not
repeated: QERS (arXiv:2601.13399), CARS (IACR ePrint 2026/1467), the quantum-safe library assessment
(arXiv:2605.17061), and two vendor sources. The composite was replaced by per-dimension facts led by
the NIST security category (FIPS 203/204), not by another weighted score.

## Stress testing: what is prescribed and what was declined (Round 22)

`stress-testing-fact-check.md` records the full sweep of NIST (SP 800-227, FIPS 203, ACVP), ETSI
(GS QKD 010/014/015/016), ISO/IEC (23837-1/-2, 4922-1/-2, 28033) and the QKD attack literature. No
standard prescribes adversarial stress testing of an integrated PQC+QKD+FHE+MPC system; ISO/IEC
23837-2 endorses discretized parameter sweeping as the METHOD but explicitly supplies no thresholds.
Four axes are defensible with published reference values; detector blinding as a QBER anomaly, the
optical evaluation activities, and any aggregate stress score are declined with reasons.

## CKKS / FHE security under decryption queries (IND-CPA-D / CPAD)

Added Round 23, because the product returns decrypted inference results and the threat model was
silent on the notion that governs exactly that. Full discussion in `../docs/threat-model.md`.

- Li, Micciancio, "On the Security of Homomorphic Encryption on Approximate Numbers", EUROCRYPT
  2021, ePrint 2020/1533. Approximate FHE fails IND-CPA-D; the attack needs one decryption query.
- Li, Micciancio, Schultz, Sorrell, CRYPTO 2022, ePrint 2022/816. Noise flooding and its precision
  cost.
- Cheon, Choe, Passelegue, Stehle, Suvanto, ACM CCS 2024, ePrint 2024/127. Extends the attack class
  to exact FHE. (Distinct from ePrint 2024/116, Checri et al., CRYPTO 2024, filed three days apart
  by a different team at a different venue - easy to conflate, so both are listed.)
  **The game conditions, verbatim, because they are weaker than readers assume** (verified at the
  abstract 2026-08-12): the adversary gets *"access to a decryption oracle for ciphertexts for which it
  should know the underlying plaintexts. This includes ciphertexts that it (honestly) encrypted and
  those obtained from the latter by evaluating circuits that it chose."* So **honestly generated inputs
  suffice and the attack is PASSIVE** - "the parties are only semi-honest" is not a defence against
  CPAD, which is the misreading this entry now exists to block.
- Checri, Clet, Renard, Sirdey, "Impossibility of CPAD security for a class of FHE schemes",
  ePrint 2026/203. No HELLHO scheme - covering basic BFV, BGV, CKKS and TFHE - can be CPAD secure.
  Practical key recovery on DE-CKKS, supported in OpenFHE, in hours on a laptop.
- Santato, Zanotto, "How (not) to Switch FHE Schemes", ePrint 2026/285. Breaks PEGASUS-style
  CKKS-to-FHEW switching in OpenFHE. Not applicable here - this project does no scheme switching -
  and recorded so that stays a checked fact rather than an assumption.
- Ogilvie, "IND-CPA-D and KR-D Security With Reduced Noise from the HintLWE Problem", ASIACRYPT 2025,
  ePrint 2025/1618. The constructive counterpoint, now with its title recorded and the claim verbatim
  (verified at the abstract 2026-08-12): *"an additional precision loss of as little as two bits is
  sufficient to restore a high level of security against passive key-recovery attacks for standard
  parameters."* Two bits is the number to quote; "about two bits of extra precision loss", which this
  entry said before, was a paraphrase of a verbatim sentence and there is no reason to paraphrase it.

**The implementations' own security documentation, added 2026-08-12.** This section cited six papers
and **not one of the three CKKS implementations**, while the demo ships one of them: `node-seal` IS
Microsoft SEAL compiled to WASM (`web/src/lib/clientFHE.ts`). Upstream's security statement for the
cipher on the page outranks any secondary summary of it. Full discussion in
[`../docs/threat-model.md`](../docs/threat-model.md); the load-bearing predicate is **crossing a trust
boundary**, not displaying a value.

- **Microsoft SEAL, `SECURITY.md`** `[link]`
  URL: https://github.com/microsoft/SEAL/blob/main/SECURITY.md
  Verified 2026-08-12. The upstream security statement for the cipher on the page. Four of its
  sentences are quoted verbatim in the threat model's CPAD section
  ([`../docs/threat-model.md`](../docs/threat-model.md#what-ckks-does-and-does-not-promise-here)): that CKKS decryption returns the message plus a
  secret-key-dependent error, that releasing such decryptions can leak the key, that SEAL's CKKS
  targets IND-CPA and not IND-CPA-D, and that noise flooding is the application's job before a value
  crosses a trust boundary, so SEAL itself provides none. Bears on `web/src/lib/clientFHE.ts`.
- **IBM HElib, `CKKS-security.md`** `[link]`
  URL: https://github.com/homenc/HElib/blob/master/CKKS-security.md
  Verified 2026-08-12. Three passages are quoted verbatim in the same section: the mechanism (a
  decryption result leaks the error, and with it the key), the query count Li and Micciancio is
  usually credited with, which is here rather than in that paper's reachable text, and the scaling law
  for HElib's own noise flooding, which it implements inside decryption with a settable accuracy
  target - the contrast that makes SEAL's position specific.
- **OpenFHE security documentation** `[link]`
  URL: https://openfhe.org/ - the security documentation on the project site; the deep-link path is
  deliberately not recorded, because only the text quoted in the threat model was captured and a
  guessed path would be worse than a search for "decryption queries" from the landing page.
  Verified 2026-08-12. The tolerated number of decryption queries and the scope it gives the attack
  (decryption results shared between parties, as in threshold FHE) are quoted verbatim in the same
  section, with the two options OpenFHE offers, `NOISE_FLOODING_DECRYPT` and `CKKS_M_FACTOR`.
- **Guo, Nabokov, Suvanto, Johansson, "Key Recovery Attacks on Approximate Homomorphic Encryption with
  Non-Worst-Case Noise Flooding Countermeasures", USENIX Security 2024** `[link]`
  URL: https://portal.research.lu.se/ - search the title; the record path is not recorded here because
  the publication page, not a guessed permalink, is what was loaded.
  **Title, authors and venue VERIFIED** via the Lund University research portal, 2026-08-12. **Body NOT
  loadable (HTTP 403), so no parameter, bound or number from it is quoted anywhere in this
  repository.** Recorded because it attacks the very countermeasure the CPAD discussion presents as the
  fix: noise flooding parameterised on average-case rather than worst-case noise. A section that names
  noise flooding as the remedy and omits this reads as unaware of it.
- Bossuat, Costache, Mouchet, Nurnberger, Troncoso-Pastoriza, ePrint 2024/853, IACR CiC 2025, DOI
  10.62056/a3n5tx4e-. NOTE the retitle at the 2025-08-11 revision: originally "Practical
  q-IND-CPA-D-Secure Approximate Homomorphic Encryption", now "Accurate and Composable Noise
  Estimates for CKKS with Application to Exact HE Computation". Citing the old title points at a
  paper that no longer exists under that name.
- **Chan, Mazzanti, Swaminathan, Vega, Mocskos and Venkatagiri, "When and Where Faults Matter: A Study
  of Transient Errors in CKKS Multiplication"**, arXiv:2608.11147 v1, 2026, preprint `[link]`
  URL: https://arxiv.org/abs/2608.11147
  Added: 2026-11-08
  Read in the arXiv PDF. Single-bit transient faults injected into ciphertext polynomials during
  ciphertext multiplication, on the authors' own unoptimised CKKS (no NTT, no RNS, $`N = 64`$ in the
  figure). A flip in both uses of a component leaves a valid ciphertext with a small extra error; a flip
  in one use breaks the cancellation at decryption. Neither SEAL nor node-seal was tested.
- **Mazzanti, Chan, Swaminathan, Vega, Mocskos and Venkatagiri, "On the Sensitivity to Errors in
  Homomorphic Computing: Single Transient Bit-flip Client-side Error Characterization"**,
  arXiv:2608.11155 v1, 2026, preprint `[link]`
  URL: https://arxiv.org/abs/2608.11155
  Added: 2026-11-08
  Read in the arXiv PDF. Each bit of each plaintext and ciphertext coefficient flipped in turn, on the
  authors' own CKKS. With NTT or RNS+NTT arithmetic any single flip ended in catastrophic decryption or
  decoding failure. Whether those failures are flagged is not stated. Neither SEAL nor node-seal was
  tested.

## QSDC security basis: secrecy capacity, not a fixed QBER (Round 24)

**This entry corrects an earlier version of itself.** That version said `qkd.pp_threshold = 0.11`
was correct for QSDC because the wiretap secrecy capacity $`C_s = 1 - 2h(e)`$ has its root at 0.110028,
the same as Shor-Preskill, so the two coincided by identity. The identity is real but the conclusion
was wrong, because $`C_s = 1 - 2h(e)`$ is the **loss-free** case.

The published criterion is $`C_s > 0`$, and $`C_s`$ depends on channel loss:

$$
C_s \;=\; Q_{\text{Bob}}\bigl[\,1 - h(e) - g\,h(e_x + e_z)\bigr],
\qquad
g \;=\; \frac{Q_{\text{Eve}}}{Q_{\text{Bob}}} \;=\; 10^{\,\text{loss\_dB}/10}
$$

That is Qi et al.'s Eq. 8, with $`g`$ counting "the efficiency of the detector and channel loss" of the
back channel. The code computes it with $`e_x = e_z = e`$. So the tolerable error rate falls fast as
the link gets lossier: **0.0757 at 0 dB, 0.0286 at Qi et al.'s own $`g = 2.57`$** (their back
channel's 4.1 dB, in a 1.5 km fibre system whose total loss was 25.1 dB), and about $`6.5 \times
10^{-5}`$ on this project's 800 km downlink, where $`g \approx 533`$: Eve's gap counts 27.3 dB, the
channel's 32.2 dB (18.1 dB of diffraction, 3.2 dB of atmosphere, 3.0 dB of pointing and 8.0 dB of
station optics) less the station's optics, plus the detector's 3.0 dB (recomputed 2026-10-30 in
`channel.py`). Those are the channel's figures since Round 77, which scaled the atmosphere by air
mass. Until 2026-10-30 this entry carried Round 75's: 37.0 dB with the detector, $`g \approx 5{,}000`$
and about $`5.5 \times 10^{-6}`$. Round 75's had in turn replaced figures computed with a pooled
$`h(e_{\text{chk}})`$ and no detector term (0.110 / 0.048 / ~0.0003), and a downlink of 24.5 dB
that had no receiver-optics loss. Applying 0.11 to a lossy channel is wrong in the
**permissive** direction - it delivers over a channel whose secrecy capacity is already zero. The
satellite path now computes $`C_s`$ directly.

Three further facts that stop any single number being quoted safely:

- **The founding protocols specify no threshold at all.** Long and Liu (PRA 65, 032302, 2002) say
  "below certain threshod" (sic); Deng, Long and Liu (PRA 68, 042317, 2003) say "reasonably low";
  DL04 (Deng and Long, PRA 69, 052319, 2004) says "if the error rate is high ... the communication is
  halted". The numbers all come from later security analysis.
- **11% appears in the QSDC literature only where the protocol reduces to BB84.** Zhao et al.
  (Front. Phys. 19, 51201, 2024) give 0.11 for entanglement-distillation QSDC against 0.073 for their
  hybrid-entanglement scheme; Pan et al. (Sci. Adv. 11, eadt4627, 2025) state 11% and say plainly
  that their expression "is exactly the key rate of the BB84 QKD protocol". DL04 - the family every
  fibre experiment implements - has a ~5.7% figure attributed to Wu, Lin, Yin and Long (Quantum Eng. 1,
  e26, 2019), not confirmed at source: the full text is behind a subscription and the abstract gives no number.
- **There are two error rates and two aborts.** The forward-leg check error governs SECURITY; the
  return-leg QBER governs RELIABILITY and is bounded by the error-correcting capability, not by a
  security threshold. Collapsing them into one comparison is a modelling error in its own right.

**No standards body has specified QSDC**, and this was checked exhaustively rather than assumed.
All ETSI QKD deliverables then reachable were searched full-text (**"14 published" is one too many, corrected Round 55**: the portal shows **13 published**, plus GS QKD 003 V1.1.1 which is WITHDRAWN, and GS QKD 001 and 006 Stopped - 14 only if the withdrawn one is counted) (2.0 M characters): zero hits for
QSDC or "secure direct communication", against a control of 7 documents matching "BB84". ITU-T's
complete X.1700-X.1749 and Y.3800-**Y.3839** sets are QKD or QRNG throughout (Y.3839 (10/26), *QKD networks - Framework of cryptographic application registration*, approved 2026-10-19 and prepublished, read 2026-11-03; this said Y.3835 until Round 51; Y.3836, Y.3837 and Y.3838 were approved 2026-08-13 and **all three are flagged prepublished**, not just Y.3836 - corrected Round 56, re-fetched per recommendation: Y.3835 reads "In force" alone at 9,796 B while Y.3836/3837/3838 read "In force" AND "prepublished" at 9,647/9,590/9,515 B. The ceiling then held: Y.3839 and Y.3840 returned **200 with a zero-length body**, as Y.3840 and Y.3841 still do on 2026-11-05, which is this project's established discriminator for a recommendation that does not exist - and note that a browser User-Agent collapses ALL of these to a 245 B stub, so the UA that defeats ETSI's bot block destroys the signal here), and the work programme
across every study period 2001-2028 and every status returns zero QSDC items against a control of
186 quantum-titled items. ISO/IEC JTC 1/SC 27's entire catalogue contains exactly two
quantum-titled standards, both parts of 23837, both QKD. IEEE has no QSDC standard - and no QKD one
either. China's national platform returns no data for the Chinese term, and the national quantum
terminology project scopes itself to QKD. QSDC is absent even from CEN/CENELEC's
pre-standardization roadmap, which mentions QKD 319 times and QSDC zero.

One terminology trap that produces false positives, and it is easy to fall into. In Chinese sources
**"quantum secure communication" is the QKD umbrella term, not QSDC** - they are different words and
different technologies. GB/T 42829-2023 and YD/T 4301-2023 are both QKD. A source claiming "China has
a quantum secure communication standard" is almost certainly pointing at one of those. Only the
distinct Chinese term for quantum secure DIRECT communication means QSDC, and no standards document
at any level contains it. Korea's TTA has the term in a terminology dictionary with no standard
number attached, which is the closest any body has come.

The correct phrasing is therefore **"no published standard; research-stage technology"** - not
"standardization in progress", because no work item exists at any body. Two citation traps worth
knowing: **"IEEE 1913-2019" is not a real published standard** (the PAR was withdrawn on 11 December
2024 and the scope was QKD anyway), and the 2024 IEEE COMST QSDC survey mentions standardization
exactly once, about QKD.

Primary sources: Qi, Sun, Lin, Yin, Long, Light Sci. Appl. 8, 22 (2019), doi:10.1038/s41377-019-0132-3
- the $`C_s > 0`$ abort rule and the measured $`g = 2.57`$. Pan, Lin, Wu, Sun, Ruan, Yin, Long,
arXiv:2005.05102 - free-space QSDC, 500 bps over 10 m, QBER $`0.49\% \pm 0.27\%`$, and the DBER/QBER
distinction. Long and Liu (2002), Deng, Long and Liu (2003), Deng and Long (2004) - the protocols.

## One-way quasi-QSDC: what the satellite page sends when two-way QSDC cannot close (Round 86, read at source 2026-11-09)

**Two-way QSDC stops early.** Qi et al.'s bound, per received carrier scaled by the gain, is

$$C_s = Q^{\mathrm{Bob}}\,\bigl[\,1 - h(e) - g\,h(e_x + e_z)\,\bigr],$$

their Eq. 8, with $`g`$ the gap between Eve's channel and Bob's, counting the channel and the detector and not the station's optics. At this page's error floor $`e = 0.0111`$ it is positive only for $`g \le (1 - h(e)) / h(2e)`$, or 7.7 dB, against 27.26 dB on the default downlink. A floor of 0.1% would lift that to 16.8 dB. Zhang et al. (Light Sci. Appl. 11, 83, 2022, doi:10.1038/s41377-022-00769-w), with the same structure in their Eq. 10, measured $`e_x = 0.001`$ and $`e_z = 0.0003`$ at 6 dB, carried 22.4 kbps there and 0.54 bps at 15.8 dB of ultra-low-loss fibre, and give 18.4 dB as the most their system could tolerate.

**STIKE removes the penalty for loss by adding a key.** Pan et al., "Simultaneous transmission of information and key exchange using the same photonic quantum states", Sci. Adv. 11, eadt4627 (2025), doi:10.1126/sciadv.adt4627. Alice one-time pads the message from a pool both ends hold, masks it (INCUM), and sends it one way. Bob's untriggered pulses are treated as lost, the key for them is returned, and the rest distils new key. Their Eq. 2 is the BB84 key rate, $`C_s = 1 - H_2(\delta_p) - H_2(\delta_b)`$, which they say outright. With decoy states, per pulse,

$$C_s = q\,Q_\mu\bigl[-f H_2(E_\mu) + \Delta_1\bigl(1 - H_2(e_1)\bigr)\bigr],\qquad C_r = q\,Q_\mu\bigl[1 - f H_2(E_\mu)\bigr],$$

their Eqs. 3 and 4, with the recovered share $`P_{\mathrm{rec}} = 1 - q\,Q_\mu`$. Measured, at a 1.25 GHz clock over fibre of 0.2 dB/km with 6.5 dB of optics at Bob and 20% detectors: 34.08 kbps at 50.3 km (QBER 2.94%), 11.36 kbps at 81.1 km (3.20%) and 2.38 kbps at 104.8 km (3.60%), the last over a week, at spreading ratios of 1:192, 1:768 and 1:3840 with an LDPC code. Their secrecy capacities at 81.1 and 104.8 km exceed the upper limit of two-way QSDC. They derive the capacity in the asymptotic regime and leave the finite-length analysis, and the strategy for keeping the pool filled, open.

**The nearest thing to a space link.** Sun et al., "High-speed phase-encoded quantum secure direct communication over 11.4 km heterogeneous free-space and fiber links", arXiv:2602.06624 (2026), preprint: STIKE over 1.4 km of urban air and 10 km of fibre, the same clock, detector efficiency and receiver optics, 4.22 kbps with a mean QBER of 2.38%, and a simulated reach past 30 km of free space. Their Table S1 is what `tests/test_r86_stike.py` reproduces: 17.83 dB of channel, $`Q_\mu = 5.31 \times 10^{-4}`$ and $`Q_\nu = 2.09 \times 10^{-4}`$, with key generated at 13.51 and 30.42 kbps on two days.

**What was not found, and nearby work that is different.** No source read reports QSDC or quasi-QSDC over a satellite link. A 2023 review (Advanced Devices & Instrumentation, doi:10.34133/adi.0004) lists the feasibility of satellite-to-ground QSDC among the open problems. Vazquez-Castro, Winter and Zbinden, arXiv:2409.05694, analyse a keyless scheme for optical inter-satellite links in the asymptotic regime, resting on a limit to the share of Bob's detected photon energy that Eve can collect; only its abstract was read, and it is not modelled here.

**What the page does with this.** It keeps Eq. 8's verdict, sends the telemetry by Eqs. 3 and 4's model where a pool is configured, and plots both against loss. The published points on the plot are each paper's rate over its clock at its own channel, optics and detector, and $`\text{loss} = 0.2L + 6.5 + 10\log_{10}(1/0.2)`$ dB for the fibre runs. Zhang et al.'s 18.4 dB counts the channel alone, a different definition, so it is here and not on the axis. See `docs/threat-model.md`.

## Round 31 additions, verified at primary source 2026-08-11

Every entry below was loaded at its arXiv abstract page, IACR eprint listing, Federal Register API
record or whitehouse.gov PDF before being written here. Nothing in this section was taken from a
secondary write-up or from another file in this repository. Where a claim could not be verified it is
marked, and where a check came back clean that is recorded too - a negative result that has been
established is worth keeping, because otherwise it gets re-litigated every round.

### PQC migration, now binding rather than advisory

- **OMB Memorandum M-26-15, "Execution of the Migration to Post-Quantum Cryptography."** Russell T.
  Vought, Director, Office of Management and Budget, 2026-06-24. Full 11-page PDF read from
  whitehouse.gov. The operational instrument behind EO 14412, and the most consequential item in this
  batch for [`docs/pqc-roadmap.md`](../docs/pqc-roadmap.md):
  - Action One: mitigate as much quantum risk as feasible **by 2030-12-31**.
  - Action Two: agency PQC Migration Plans to OMB and ONCD **within 120 days** (due about 2026-10-22, now passed).
  - Five phases: **1** 2026-2027 strategy and discovery, **2** 2027-2028 pilots, **3** 2028-2030
    prioritised migration and key establishment, **4** 2031 signature migration, **5** 2035 full
    migration.
  - Phases 3 and 4 each end with the sentence *"Ensure all systems are cryptographically agile."* That
    is a mandated requirement naming the exact property [`docs/crypto-agility.md`](../docs/crypto-agility.md)
    claims, which changes that document's status from good practice to alignment with a directive.
  - **"Agencies must align their plans with NIST Internal Report (IR) 8547 ... or successor
    document."** This sharpens rather than contradicts the roadmap's line that IR 8547 "is still a
    draft, not binding": the draft is now normatively referenced by a binding memorandum. Both facts
    have to be stated together, because either alone misleads. [Superseded 2026-10-27: NIST
    published IR 8547 final, doi:10.6028/NIST.IR.8547, which deprecates all quantum-vulnerable
    public-key algorithms after 2030 and disallows them after 2035; the memorandum now points at a
    final document.]
  - Does not apply to national security systems, consistent with what this repository already records
    about EO 14412.
- **Two citation checks recorded so they are not repeated.** Secondary sources (Wiz,
  gopher.security) label the executive order **14409**. The Federal Register API returns EO
  **14412**, "Securing the Nation Against Advanced Cryptographic Attacks", signed 2026-06-22,
  published 2026-06-25, **91 FR 38483**. This repository's number and FR citation are correct and the
  blogs are wrong. Separately, **NIST IR 8547 is confirmed still at Initial Public Draft** (published
  2024-11-12, comments closed 2025-01-10, no second draft and no final), so the roadmap's claim holds.
  [Superseded 2026-10-27: NIST published IR 8547 final, doi:10.6028/NIST.IR.8547, which deprecates
  all quantum-vulnerable public-key algorithms after 2030 and disallows them after 2035.]

### QKD security proofs that supersede this repository's baseline

- **Sixto, Curras-Lorenzo, Pereira, Zapatero, Navarrete & Curty, "Finite-key security analysis of
  decoy-state QKD with source and detector imperfections."** arXiv:2606.29943, 2026-06-29. An
  analytical finite-key proof that carries state-preparation flaws, bit-and-basis side-channel leakage
  **and correlations**, setting-independent intensity fluctuations, and detection-efficiency mismatch
  simultaneously. **Directly relevant, and not comfortable reading:** `qkd/bb84.py::_decoy_secure_fraction`
  and `qkd/decoy_reference.py` both rest on Ma, Qi, Zhao & Lo (2005), and this paper's motivation is
  that existing proofs "rely on assumptions that are difficult to justify in realistic high-speed
  implementations, such as the independent and identically distributed nature of the emitted signals"
  - which is exactly what this repository assumes. The Ma et al. reproduction stays as the reference
  implementation because that is what the numbers on the page are computed against; this is the 2026
  replacement and the honest statement is that the demo has not adopted it.
- **Ivchenko, Khmelev, Kurochkin & Trushechkin, "Security of quantum key distribution with passive
  basis choice and detection-efficiency mismatch for a realistic satellite setup."** arXiv:2608.09793,
  2026-08-10. Passive-basis-choice BB84 with mismatch across four threshold detectors plus a
  decoy-state adaptation, parameterised on the **Micius satellite to Zvenigorod ground station** link -
  a published real-satellite parameterisation rather than a simulation. The string
  "detection-efficiency mismatch" appears nowhere else in this repository, and the satellite page's
  free-space link budget has no term for it.
- **Hossain & Hossain, "SATLOCK: Handover-Coupled Scheduling for Weather-Resilient Quantum Key
  Distribution over LEO Constellations."** arXiv:2607.17076, v1 2026-07-19, v2 2026-07-27. Composite
  channel model with atmospheric loss, pointing jitter, **Markov cloud cover, decoy-state estimation
  and finite-key correction**; an ILP upper bound; a decentralised DQN reaching 91.8% / 84.6% of ILP.
  This is the closest published prior art to `qkd/optimize.py` (the WS1 AI-QKD parameter optimizer)
  applied at constellation scale, and its finding that **finite-key effects eliminate short,
  low-elevation passes entirely** is a constraint the satellite path should state rather than a
  refinement.
- **Ferreira, "AKE Protocol Combining PQC and QKD"**, IACR Communications in Cryptology 2(4), 2026,
  DOI 10.62056/aesgsgbmo; IACR ePrint 2026/036 is a later major revision (revised 2026-06-18) `[link]`
  URL: https://cic.iacr.org/p/2/4/4
  Added: 2026-11-08
  Read in the journal's PDF: the main body and the proof of the QKD-only case. The ePrint body refused
  automated access, so what the major revision changes is not known here. A hybrid AKE of the author's
  own design (ephemeral KEM, signatures, a carried shared state, one BB84 run) whose Extract/Expand
  chain takes the QKD key only in its last Extract. Where the QKD key is the only unrevealed secret,
  the key-indistinguishability bound carries the QKD secrecy term times $`n_T (n_P n_S)^2`$; with
  vendors' claimed bounds of $`10^{-9}`$ to $`10^{-10}`$ and at least $`2^{35}`$ runs the author finds
  it not meaningful, and calls the bounds non-tight. Direct quantum channel, no trusted nodes, no side
  channels, BB84 without decoy states. The proof relies on HMAC being a dual PRF, the assumption
  `_mix_qkd` in `src/qsh/pqc/transport.py` states, but it does not analyse that function, which takes
  the QKD key as salt where the paper's chain takes it as input. The QKD term is the same for any
  combiner, so the paper does not compare HKDF with Toeplitz. This project has no real QKD leg to price.

### Secure inference and threshold FHE

- **Ma, Li & Schmid, "FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing."**
  arXiv:2606.09551, 2026-06-08. Compiles per-operator protocols into two batched FSS evaluations:
  1.24x-1.50x end-to-end speedup and 9-16% less online communication over the state of the art in
  FSS-based GPU secure inference. Targets exactly the layer `mpc/secret_inference.py` occupies, which
  is `mpyc` with pure-Python secret shares for ReLU and argmax. The honest read is that **the
  non-linear half is the active research frontier and this repository's implementation is the slow
  reference design**, not the state of the art - and that the leverage is in the protocol, not in
  changing the model to suit MPC.
- **Kim, Kim & Lee, "Revisiting Shamir Secret Sharing for Threshold Fully Homomorphic Encryption."**
  IACR eprint 2026/1559, received 2026-07-30. Reduces the ThFHE ciphertext modulus from $`O(N)`$ to
  $`O(t \log(N/t^2))`$ bits, with every algorithm completing under 0.2 s at $`N = 64`$ and scaling past $`N \ge 256`$.
  The only threshold-FHE anchor here was Chardouvelis, Doettling & Malavolta (2020/1454) in
  `qhe/rate1_qfhe.py`, which is the quantum-side citation; there was none for the classical
  multi-party side even though `submodules/HEonGPU` ships N-out-of-N collective bootstrapping. Bears
  on why [`docs/architecture.md`](../docs/architecture.md) splits linear-to-FHE and
  non-linear-to-MPC rather than using threshold decryption.

### The primary source for the FHE cost ceiling

- **Park, Park, Park, Ahn, Cheon, Hanrot, Kim, Park & Stehle, "Scaling up FHE-based
  Privacy-Preserving ML: Higher Throughput, Longer Inputs for LLama-3-8B."** arXiv:2601.18511, v1
  2026-01-26, v2 2026-06-17. Verbatim from the abstract: *"On 8 NVIDIA RTX PRO 6000 GPUs, 128
  encrypted tokens take 20s for summarization and 18s/token for generation (vastly outperforming the
  SOTA 295s on costlier H100 GPUs). For a heterogeneous 4096-token input (last 128 encrypted), it
  takes 64s for summarization and 22s/token for generation."*

  **Recorded here because it was cited without provenance.** `docs/pqc-roadmap.md` quoted the figure
  and `models/registry.py` attributed it to the secondary SoK (IACR 2026/935) rather than to this
  paper, while this file - the provenance file - did not list it at all. It is also the result that
  retired the registry's unscoped claim that homomorphic evaluation "will remain" cost-bounded: what
  survives is the scoped claim, that it is cost-bounded at a single device in a browser tab.

- **node-seal, `BENCHMARK.md`** (library benchmark page, last updated 2025-03-27).
  https://github.com/s0l0ist/node-seal/blob/7.0.0/BENCHMARK.md (deleted from `main` on 2026-10-31) - fetched and read 2026-08-14.
  Microsoft SEAL v4.1.2 / node-seal v5.1.5 on an Apple M1 Pro; Chrome 134, Firefox 136, Safari 18.3.
  CKKS `EvaluateRotate` at `n=8192/log(q)=218` reads **6830 us in Chrome**; at `n=16384/log(q)=438`,
  **41200 us**. Ten iterations per benchmark, no error bars, timers quantised to 10 us (Chrome) and
  100 us (Firefox/Safari).

  **The only primary source found for browser-WASM CKKS per-primitive cost**, and the comparator this
  project's 3.10-4.39 ms per rotation is read against in
  [`encrypted-inference-performance.md`](encrypted-inference-performance.md). The `n=8192` row is
  comparable (same ring, similar modulus, both Chrome/WASM); the `n=16384` row is **not** - 438 bits
  against 200 here - and is recorded only so the parameter difference is visible.

- **Ali Sah Ozcan and Erkay Savas**, *GPU Implementations of Three Different Key-Switching Methods
  for Homomorphic Encryption Schemes*, IACR ePrint **2025/124** (November 2024).
  https://eprint.iacr.org/2025/124.pdf - PDF downloaded and text-extracted 2026-08-14.
  Table 5, CKKS relinearization: single-thread Microsoft SEAL on a Ryzen 9 7950X3D gives **1641 us**
  at ring 8192 and **14563 us** at 16384; HEonGPU on an RTX 4090 gives **41.8 us** and **84.1 us**.
  Measures relinearization, not rotation - both are key-switching, and node-seal's own table puts the
  two within 1% at both ring degrees, so it is a sound proxy but is cited as relinearization.

  **Recorded here because Round 38 introduced both sources and this file listed neither** - the same
  omission it records above for the Nature Communications figure, repeated by the round that read
  that entry.

  Four metadata-only "direction of travel" entries stood here - arXiv:2604.04783, 2602.11470,
  2512.11269, 2602.22229 - labelled "abstracts not individually opened, so no number from these four
  should be quoted". `docs/deliverable-verification.md` gave them a deadline: promoted by quoting a
  number with its section, or removed by Round 35. **Removed in Round 39, three rounds after the
  deadline the register set and never executed.** None was ever cited, so nothing depends on them;
  the arXiv IDs are preserved in this sentence for anyone who wants to re-open the search.

### Negative results, re-verified 2026-08-11

Both support claims this repository already makes, and both are recorded so the search is not redone.

- **No in-browser FHE or MPC execution has been published since webSPDZ.** An IACR eprint full-text
  search for "WebAssembly" returns 8 results across all years, the newest crypto-relevant one still
  **2025/487 webSPDZ**; an arXiv sweep of WebAssembly, garbled-circuit and browser-plus-homomorphic
  queries, newest first, returns 40 papers back to 2026-02 with zero in-browser FHE or MPC results.
  The Round 30 finding therefore re-verifies, and the composition novelty claim rests on it.
- **Nothing in QHE or QOTP is newer than Round 30's citations.** An arXiv query on `cat:quant-ph AND
  abs:homomorphic`, newest first, puts arXiv:2607.17131 (Blind Transpiler, already cited) at the top
  of the relevant set; the only newer quant-ph hit is arXiv:2607.21449 (QuantumChain, blockchain-backed
  QFL, 2026-07-23), which is off-thesis.

### Sweep of 2026-08-01 to 2026-08-14, run 2026-08-14

Thirty searches across seven areas; nothing found changes a claim this repository makes. The window is
thirteen days wide, so a null was the honest expectation - it is recorded with **what was searched**
rather than as a bare "nothing found", because an unrecorded null is indistinguishable from a search
nobody ran, and this project has twice re-derived a negative it had already established.

- **NIST SP 800-227: no change, checked at the primary source.** The CSRC document history reads
  exactly `01/07/25: SP 800-227 (Draft)` and `09/18/25: SP 800-227 (Final)` and nothing else - no
  errata, no update, no withdrawal, no follow-on. Section 4.6.3's IND-CCA-preserving combiner, which
  `src/qsh/pqc/hybrid_kem.py` implements, is unchanged.
- **FIPS 203 / 204 / 205: no change in window.** All three still show only `08/24/23 Draft` and
  `08/13/24 Final`. FIPS 204 has an errata spreadsheet dated **2026-07-31**, one day outside the
  window; it concerns the signature standard rather than the KEM. `csrc.nist.gov/news` carries two
  August 2026 items, neither PQC.
- **CKKS and homomorphic encryption: nothing that touches a claim here.** Nine in-window papers on the
  "homomorphic" query; the two that could plausibly bear on performance do not - one is a survey with
  no rotation timings and no WASM, the other is BGV matrix multiplication rather than CKKS rotations.
  The strong bootstrapping results that surface for "August 2026" (BootNet, "Bootstrapping is All You
  Need") are June 2026 and out of window. Note the asymmetry: **the 3.10-4.39 ms per rotation figure
  is a measurement of this stack, and no paper can invalidate a measurement** - only a comparative
  claim, and the one comparison made is already hedged as "the same order, this host faster".
- **QKD: one in-window record, not applicable.** arXiv:2608.04112 reports a secure-key-rate-per-pulse
  result in a four-dimensional fibre lab demonstration. This repository scopes hardware QKD out
  explicitly and validates its simulation against Ma et al. (2005). No new ETSI or ITU-T QKD standard
  in the window.
- **QCNN: null.** In-window hits are applications (fault classification, a CNN-QNN hybrid image
  classifier); no theory or benchmark result bearing on the QCNN claims made here.
- **MPC-FHE / QSMPC-FHE hybrids: null in window** for anything altering the model-strategy verdict.
- **ML-KEM and hybrid-combiner cryptanalysis: null**, with one item put on watch below.

**WATCH, not a change: ePrint 2026/1591, a claimed polynomial-time quantum algorithm for the Dihedral
Coset Problem.** https://eprint.iacr.org/2026/1591 - Daniel R. Simon (AWS Cryptography Group),
received 2026-08-03, three revisions to 2026-08-12, fetched and read as PDF 2026-08-14. Composed with
Regev's and Brakerski-Kirshanova-Stehle-Wen's reductions it would give polynomial-time quantum
algorithms for SVP and LWE, which is why it is here at all.

*Why it changes nothing today, from the document itself rather than from coverage of it:* the title
carries "[Preliminary Draft]"; four proofs are marked "(Sketch)"; the load-bearing final corollary
cites two references both listed as "Personal Communication", i.e. unpublished; the abstract carries
a self-correction noting that the first draft's Lemma 3 "incorrectly conflated pairwise independence
over the uniform vs. actual distribution". A full-text grep for `ml-kem|kyber|fips|dilithium|concrete
parameter|deployed` returns **zero hits** - the paper makes no claim against any deployed parameter
set. It is unrefereed and actively being revised.

*The honest nuance, which is the reason to record it rather than dismiss it:* **this repository's
ML-KEM + X25519 hybrid is not a hedge against this particular risk.** X25519 already falls to Shor,
so ML-KEM is the only quantum-resistant leg; if this result were confirmed AND shown to reach
ML-KEM's concrete parameters, the hybrid would not preserve quantum security. Neither condition holds
today. *Reopens if* the paper is refereed and published, or a version removes the sketch markers and
the personal-communication dependencies, or any analysis extends it to concrete ML-KEM parameters.

A related result surfaced while chasing this one and is recorded to stop it being re-found as news:
**ePrint 2026/155**, Wen and Zheng, Module-LWE and structured extrapolated dihedral cosets (CRYPTO
2026), proves a quantum polynomial-time *equivalence* in the setting ML-KEM uses. Original 2026-01-30,
last revised 2026-07-16 - **both before this window** - and it claims no attack and no weakening of
ML-KEM. Secondary coverage was pairing the two as though both were current.

### Verified but not yet cited from anywhere

Held here rather than dropped, because both bear on the WS4(c) / WS4(f) split and the client in this
project is a browser tab.

- **van Dam, Grimbergen & Wehner, "Verifiable blind quantum computing: Comparative analysis and design
  considerations for client architectures."** arXiv:2607.05650, 2026-07-06. Classifies BQC clients as
  emission-, measurement- and rotation-based and concludes that measurement-based remote state
  preparation and reflection-based teleportation are the default candidates.
- **Lee, Kim & Chung, "BPBO: Blindness-Preserving Brickwork Optimization by Certified Region
  Resynthesis."** arXiv:2606.29962, 2026-06-29. Grover-3 pattern reduced from 3x725 to 3x98 while
  preserving blindness.

### Read for the lending page (Round 86), each on 2026-11-09

- **Aave, `LiquidationLogic.sol`**, `aave/aave-v3-core`, GitHub `[link]`
  URL: https://github.com/aave/aave-v3-core/blob/master/contracts/protocol/libraries/logic/LiquidationLogic.sol
  Added: 2026-11-09
  Read at source. `DEFAULT_LIQUIDATION_CLOSE_FACTOR = 0.5e4`, `MAX_LIQUIDATION_CLOSE_FACTOR = 1e4`, `CLOSE_FACTOR_HF_THRESHOLD = 0.95e18`, and the close factor is the default above the threshold and the maximum at or below it. The page re-derives this and copies no code. That repository's licence (Business Source License 1.1) names a change date of 27 January 2023 at the latest; the current `aave-dao/aave-v3-origin` is under the same licence with a change date of 2027-03-06 at the latest.
- **Aave, "Aave V4's New Liquidation Engine"**, blog, 11 December 2025 `[link]`
  URL: https://aave.com/blog/aave-v4-liquidations
  Added: 2026-11-09
  A Target Health Factor set by governance for each Spoke replaces the fixed percentage of the debt, and the bonus grows as the health factor falls. The lending page models V3's rule and says so.
- **Martinico, Calabrese, Rubin and Ciampi, "Crypto-Asset Collateralised Loans in Open Finance via Robust Fully Homomorphic Encryption"**, ACNS 2026, LNCS 16572, published 24 July 2026, DOI 10.1007/978-3-032-32575-4_18 `[link]`
  URL: https://doi.org/10.1007/978-3-032-32575-4_18
  Added: 2026-11-09
  ABSTRACT ONLY, read in a browser. A lender scores a borrower by a privacy-preserving function of net worth, with blockchain assets as collateral, on a new notion, Robust FHE, secure against a corrupted decryptor, proved in the UC model. It does not describe a liquidation check. The closest published design found; the lending page is semi-honest and gives no such guarantee.
- **Sachan, Fickett, Buchinger and Miller, "Post-Quantum Secure Federated DeFi for Inclusive Banking"**, arXiv:2606.10658, preprint, June 2026 `[link]`
  URL: https://arxiv.org/abs/2606.10658
  Added: 2026-11-09
  ABSTRACT ONLY. Banks pool encrypted data for lending decisions under lattice-based FHE, tested on agricultural lending. Neither liquidation nor a health factor appears in the abstract.
- **Sadeghi and Feinstein, "Liquidation Dynamics in DeFi and the Role of Transaction Fees"**, arXiv:2602.12104, preprint, February 2026 `[link]`
  URL: https://arxiv.org/abs/2602.12104
  Added: 2026-11-09
  ABSTRACT ONLY. A liquidator's dynamic program with a constant-product market maker as the oracle, oracle manipulation by sandwiching, and fees as a defence. The page's market has no oracle to manipulate, which is one of the things it does not show.
- **Spadea and Seneviratne, "From Risk to Rescue: An Agentic Survival Analysis Framework for Liquidation Prevention"**, arXiv:2604.14583, preprint, April 2026 `[link]`
  URL: https://arxiv.org/abs/2604.14583
  Added: 2026-11-09
  ABSTRACT ONLY. An agent on Aave V3 that estimates liquidation risk by survival analysis, validated on a simulator rather than on live data. No cryptography. Context for what a simulated market can and cannot show.

- **Who holds the key in a private liquidation check**: the 2026 sources on custody and liquidation, read at origin, are in [`defi-lending-key-custody.md`](defi-lending-key-custody.md).
- **Flood, Engelen, Aspinall and Desmet, "Bad Design Smells in Benchmark NIDS Datasets"**, IEEE EuroS&P 2024, pp. 658-675 `[link]`
  URL: https://detlearsom.github.io/pubs/BadSmells-EuroSP24-finaldraft.pdf
  Added: 2026-11-09
  Read in the authors' final draft. Audits seven benchmark NIDS datasets and reports labels leaked via features and millions of near-duplicate flows, and warns that standard randomised splits leak where similar attacks cross categories. The source for treating identical flows across classes as a measurement.
- **Arp et al., "Dos and Don'ts of Machine Learning in Computer Security"**, USENIX Security 2022, pp. 3971-3988 `[link]`
  URL: https://www.usenix.org/conference/usenixsecurity22/presentation/arp
  Added: 2026-11-09
  Read in the authors' PDF. Names data snooping (P3) and spurious correlations (P4) among ten pitfalls, the second being artifacts unrelated to the security problem that create shortcuts between classes.
- **Siriwardhana et al., "Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)"**, IEEE Data Descriptions 2, 358-369, 2025, doi:10.1109/IEEEDATA.2025.3592888 `[link]`
  URL: https://ieeexplore.ieee.org/document/11098458
  Added: 2026-11-09
  Read in full, open access under CC BY. The testbed, the session schedule with the 5-minute overlap of the two attackers, GTP removal with TraceWrangler and flows from Argus. It says the data were labelled and each attack given its own label and gives no method. The form of citation the dataset's record asks for.
- **Siriwardhana et al., scripts for 5G-NIDD** `[link]`
  URL: https://github.com/yushan1986/5g-nidd
  Added: 2026-11-09
  Read the README, the attack commands, the loading script and `IP_Change.ipynb`. There is no labelling step. The repository carries no licence, so nothing from it is reused here.
- **Window statistics from the 5G-NIDD captures**: the study and what it found, in [`5g-nidd-window-features.md`](5g-nidd-window-features.md).
- **Python walkthroughs on the demo**: marimo, JupyterLite and pyvista-wasm assessed against the demo's constraints, in [`notebook-walkthroughs.md`](notebook-walkthroughs.md).
- **Mirsky, Doitshman, Elovici and Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection"**, NDSS 2018, arXiv:1802.09089 `[link]`
  URL: https://arxiv.org/abs/1802.09089
  Added: 2026-11-09
  Read for its feature extractor, AfterImage: statistics per host, channel and socket over damped windows with decay $`d_\lambda(t) = 2^{-\lambda t}`$, 115 features. The source of the idea of several time windows per channel in the window study; that study uses plain sliding windows and a supervised booster, and not Kitsune's autoencoders.
- **Samarakoon, Siriwardhana, Porambage, Liyanage, Chang, Kim, Kim and Ylianttila, "5G-NIDD"**, arXiv:2212.01298, 2022 `[link]`
  URL: https://arxiv.org/abs/2212.01298
  Added: 2026-11-09
  Read in its HTML version, section V. Captures on 26 and 29 August 2022, each attack in its own session and each session at two base stations; 112 Argus features; a multiclass ANOVA ranking whose top ten include `Seq` and `Offset`, and accuracies around 0.99 on those ten.

### Read in Round 86, cited by nothing yet

- **Harth-Kitzerow, Schiller, Schwanke, Prantl and Carle, "Benchmarking Secure Multiparty Computation
  Frameworks for Real-World Workloads in Diverse Network Settings"**, IACR ePrint 2026/183, preprint,
  2026 `[link]`
  URL: https://eprint.iacr.org/2026/183
  Added: 2026-11-08
  ABSTRACT ONLY: the PDF refused automated access, so no figure from it may be used here. Six
  reference use cases derived from more than 50 MPC applications, implemented on HPMPC, MPyC, MP-SPDZ
  and MOTION and benchmarked under varied bandwidth, latency, packet loss and input size.
- **Mazzone, Micali and Pronesti, "Secure Change-Point Detection for Time Series under Homomorphic
  Encryption"**, Proceedings on Privacy Enhancing Technologies 2026(2), 127-142, DOI
  10.56553/popets-2026-0040; arXiv:2601.05865 `[link]`
  URL: https://doi.org/10.56553/popets-2026-0040
  Added: 2026-11-08
  Read in the arXiv PDF, with the main figures matched against the published PDF. CUSUM over encrypted
  block summaries under CKKS (OpenFHE, ring degree $`2^{17}`$, no bootstrapping), run by an
  honest-but-curious server for a single client. One million points in 101.52 s (mean) to 179.66 s
  (frequency) on a 64-core server using up to 48.9 GB, with plaintext-level accuracy on synthetic and
  three real series. It locates one change point assumed to exist; multiple change points and the
  no-change case are left open. Not comparable with an in-tab student evaluated on satellite telemetry.

## Round 32 additions, verified at primary source 2026-08-11

Same rule as the Round 31 section: every entry below was loaded at its arXiv abstract page, IACR eprint
listing, PubMed record or agency PDF before being written here. Nothing came from a secondary write-up.
Where a source could not be loaded it is marked and no number is taken from it.

### Homomorphic inference: the cost driver, and a design that contests this repository's split

- **Xie, Tan, Boddeti and Lu, "ATLAS: Automated Approximation of Transformers for Efficient Homomorphic
  Inference in One Hour."** arXiv:2607.23478, 2026-07-26 `[link]`
  URL: https://arxiv.org/abs/2607.23478
  Names **multiplicative depth as the dominant cost** of homomorphic transformer inference, with a
  per-layer configuration space reaching about $`10^{84}`$ for a 12-layer BERT/ViT and $`10^{225}`$ for 32-layer LLaMA3,
  and 35-50% of candidate configurations numerically invalid. These figures are v1's (2026-07-26);
  v2 (2026-08-27) is a revised text, so quote them with their version. This is a **primary source for the
  constraint `models/registry.py` asserts** in its T1 docstring, and Round 33 cited it there.
  This entry said the docstring "until now rested on a secondary systematisation (IACR 2026/935) plus
  Park et al.", which was wrong in both halves: Round 31 had already removed the 2026/935 citation
  from that file, and adding this reference in Round 32 did not put ATLAS into the docstring either -
  so a reference was recorded as the source for a code site it never touched. Contested by
  **arXiv:2608.00375** (Trimming) below, which carries no experiments.
- **Lin and Wu (NTU), "Fine-Grained and Runtime-Configurable Precision for Exact FHE Inference"
  (Tailor).** IACR ePrint 2026/1551, received 2026-07-29, approved 2026-08-03 `[link]`
  URL: https://eprint.iacr.org/2026/1551
  Per-layer precision becomes a runtime parameter under a single keygen; **ReLU and comparisons are
  evaluated exactly, not approximated**; TFHE and FINAL backends; MNIST at 2-bit output / 4-bit elsewhere
  reaches 96.44% in 222 s against uniform 4-bit's 96.17% in 236.1 s.
  **This is the most architecturally consequential item in the batch and it is carried as a named
  alternative, not a refinement.** This repository splits linear layers to FHE and non-linear layers to
  MPC *because* non-linearities are expensive under CKKS. Tailor evaluates them exactly under FHE with no
  MPC round at all. That does not refute the split - the demo's constraint is a browser tab, and Tailor's
  222 s is a server measurement on TFHE rather than CKKS - but "CKKS cannot do non-linearities cheaply"
  is no longer the whole story, and `docs/architecture.md` should name the alternative rather than imply
  none exists.
- **Gong, Tian, Zhang and Liu, "Verifiable and Collusion-Resistant Multi-Party Quantum Private Set
  Operations."** IACR ePrint 2026/1436, received 2026-07-14 `[link]`
  URL: https://eprint.iacr.org/2026/1436
  Combines **verifiable quantum FHE as the outsourced-evaluation layer with threshold FHE as the
  key-management mechanism**, instantiated on IBM Quantum Platform, extended to private set union, and
  resilient to collusion between the third party and participants. This is the QSMPC x QHE composition
  this repository is named after, in one protocol: `qhe/rate1_qfhe.py` had a quantum anchor
  (Chardouvelis-Doettling-Malavolta) and Round 31 added a classical ThFHE anchor (2026/1559) separately.
  Note Fengxia Liu is also first author of arXiv:2604.23490, already cited.

### QKD security proofs and hardware, receiver side

- **Navarrete, Pereira, Curras-Lorenzo, Mizutani, Curty and Tamaki, "Phase-error estimation for quantum
  key distribution with leaky receivers."** arXiv:2608.09674, 2026-08-10 `[link]`
  URL: https://arxiv.org/abs/2608.09674
  Finite-key phase-error framework carrying leakage about **both basis choice and measurement outcome**,
  requiring only a bound on side-channel state distinguishability, composable with existing
  hardware-imperfection proofs, motivated by detector backflashes and back-reflected Trojan-horse light.
  **The receiver-side counterpart to Round 31's transmitter-side arXiv:2606.29943**, published the same
  day as the already-cited 2608.09793 by an overlapping author set. Directly relevant to the
  `trojan_horse` and `wavelength_trojan` classes in `models/metrics.json`, which simulate receiver
  leakage and cite no security proof for it.
- **Konteli, Makris, Anastasiou, Tsimvrakidis and Kanellos, "Quantum-Level Crosstalk Characterization of
  a 16x16 MEMS Optical Switch for Dynamic Quantum Communications."** arXiv:2607.24299, 2026-07-27 `[link]`
  URL: https://arxiv.org/abs/2607.24299
  All-to-all quantum-level crosstalk measured with SNSPDs on a **commercial** 16x16 MEMS switch, with the
  theoretical impact expressed directly in **decoy-state BB84** terms. `qkd/sdn_controller.py` models
  ETSI GS QKD 015 `PHYS`/`VIRT` links and resource allocation with **no crosstalk term**; this is the
  measured physical-layer penalty a switched multi-user QKD network pays.

### Browser-side inference cost

- **Maczan, "Measuring and Reducing WebGPU Dispatch Overhead for LLM Inference."** arXiv:2608.08730,
  2026-08-09; accepted to the Fourth UK AI Conference 2026, to appear in PMLR `[link]`
  URL: https://arxiv.org/abs/2608.08730
  Sequential-dispatch measurement method; per-dispatch cost is independent of dtype; **at batch size 1
  the limit is dispatch overhead, not kernel quality**. A citable reason why in-tab inference is
  latency-bound in a way no smaller model or better graph fixes, for `web/src/lib/onnxInference.ts` and
  the T3 tier.

### Homomorphic encryption applied to polygenic risk scores - a gap, not an addition

`references/REFERENCES.md` had **no HE-genomics entry at all**, while the genomics use case is the one
whose model is FHE-native by construction. Both entries below also **falsify a claim this repository
makes**, in the direction of understating what is possible.

- **Knight, Li, Jensen, Yolou, Kockan and Gerstein, "Homomorphic encryption enables privacy preserving
  polygenic risk scores."** *Cell Reports Methods* 6(1):101271, published 2026-01-26 (epub 2026-01-08),
  DOI 10.1016/j.crmeth.2025.101271, PMID 41512862 `[link]`
  URL: https://doi.org/10.1016/j.crmeth.2025.101271
  Open-source HEPRS; three-party model (clinician / model developer / evaluator); tested on a
  schizophrenia model with **110,000 SNPs**; encrypted and plaintext PRSs "agree closely"; **"practical
  to use on a single CPU"**.
  **This is the citation that scopes the registry's T1 argument.** `genomics.prs` scores 183 variants, and
  `registry.py` explains T1 smallness by homomorphic cost. For this use case that is the wrong reason:
  HEPRS runs roughly 600x more variants on one CPU, so the score is small because **PGS000072 has 187
  variants** - a property of Graff et al. 2021, not of CKKS. The cost argument still holds for
  finance, medical and satellite, where the model is not a bare dot product.
- **Provatas, Galanopoulos and Georgakopoulos-Soares, "bioETH-PRS: Confidential Polygenic Risk Scoring
  without a Trusted Evaluator via Fully Homomorphic Encryption on a Programmable Blockchain."**
  arXiv:2605.21634 [q-bio.GN], 2026-05-20 `[link]`
  URL: https://arxiv.org/abs/2605.21634
  Integer-exact **TFHE** on an fhEVM, with a three-step fixed-point quantisation for representing signed
  GWAS weights as unsigned 64-bit integers at machine-epsilon reconstruction accuracy. A 2026 primary
  source for the signed-weight encoding problem `models/genomics_prs/weights.json` already solves, and an
  argument for why the evaluator role is worth separating.

### Medical: latent-space encoding, and a note this repository overstates

- **Kim, Jo, Ra and Park, "Privacy-Preserving Chest X-ray Classification in Latent Space with
  Homomorphically Encrypted Neural Inference."** arXiv:2506.15258, v1 2025-06-18, v2 2025-06-20 `[link]`
  URL: https://arxiv.org/abs/2506.15258
  VQGAN compresses images to latents at an optimum **downsampling factor of eight**; activations replaced
  with lower-degree polynomials; squeeze-and-excitation adapted to HE; two chest X-ray datasets.
  `models/metrics.json` calls `medical_fl.student`'s 3,226 parameters over a 14x14 input "a
  CKKS-feasibility budget". After this paper that is **a choice of input representation**, not a
  feasibility limit - and this repository is unusually well placed for the alternative, because the
  client is a browser tab already running `onnxruntime-web`, so an encoder could run locally in plaintext
  with no privacy loss at all. Recorded as the **sixth** falsifiable trigger in the additional-model
  decision (`docs/deliverable-verification.md`) rather than acted on. Corrected 2026-08-12: this entry
  and `models/metrics.json` both said the trigger was recorded while the table held five rows and none
  was this one, and the decision itself lived only in an untracked file, so the appeal could not be
  checked from the repository at all. The trigger is now written and the decision is tracked. Its
  condition is a measurement rather than a direction: the encoder's latent must fit 4096 CKKS slots at
  multiplicative depth 1 and run under `onnxruntime-web`. Note the paper's $`f = 8`$ on a 28x28 input gives
  3.5x3.5, so the transfer is exactly what is unmeasured.

### Standards: nothing newer, and one uncited instrument

- **No PQC migration instrument newer than OMB M-26-15** (2026-06-24). The Federal Register API, term
  "post-quantum cryptography", publication date on or after 2026-06-25, returns three documents; the only one
  after M-26-15 is **FCC 2026-15600**, 91 FR 48320, published 2026-07-31, whose full text mentions
  post-quantum cryptography **once**, as a question for comment: "should we account for that risk
  through post-quantum cryptography?" No date, no timeline - not a migration instrument.
- **NIST IR 8547 re-verified still at Initial Public Draft** (published 2024-11-12, comments closed
  2025-01-10, no second draft, no final). `docs/pqc-roadmap.md` holds. [Superseded 2026-10-27: NIST
  published IR 8547 final, doi:10.6028/NIST.IR.8547, which deprecates all quantum-vulnerable
  public-key algorithms after 2030 and disallows them after 2035.] FIPS 206 IPD and an HQC draft:
  **NOT FOUND** - no June/July/August 2026 item for either on csrc.nist.gov/News.
- **A second quantum executive order was published the same day as EO 14412 and is cited nowhere here.**
  "Ushering in the Next Frontier of Quantum Innovation", Federal Register document 2026-12910,
  **91 FR 38487**, published 2026-06-25. EO 14412 is 2026-12909 at 91 FR 38483. Recorded because a
  roadmap that cites one of two same-day quantum EOs looks like it missed one.

### Negative results, re-verified 2026-08-11

- **No in-browser FHE or MPC has been published, third consecutive round.** An IACR full-text search for
  "WebAssembly browser" returns 3 results across all years, the newest crypto-relevant one still
  **2025/487 webSPDZ**. An arXiv sweep of (WebAssembly OR "in the browser" OR browser-based) against
  (homomorphic OR "secure multiparty" OR garbled OR "private inference"), newest first, returns nothing
  in-browser; a separate (WebGPU OR WebAssembly) sweep across cs.CR/cs.LG/cs.DC finds nothing evaluating
  ciphertext in a tab. The composition novelty claim rests on this and it continues to hold.
  [Corrected 2026-10-30: the heading was too broad when written. webSPDZ, which this bullet names, is
  MPC in a browser; node-seal, which this demo uses, evaluates homomorphically in the browser and
  benchmarks it there; and Attrapadung et al. (AsiaCCS 2018; Round 83 additions, below) shipped
  two-level homomorphic encryption in WebAssembly. What these searches support is narrower: no
  peer-reviewed 2025-26 report of CKKS neural-network inference evaluated in a browser tab.]
- **No encrypted-inference work specific to AML or to satellite telemetry.** Searches for (homomorphic OR
  secure inference OR privacy-preserving) against (money laundering OR fraud detection) surface only
  federated work in 2026; the same against (satellite OR telemetry OR spacecraft) surfaces no encrypted
  telemetry anomaly detection at all. So the encrypted tiers on those two pages are demonstrations rather
  than reproductions of anyone's deployment, which is what they should be described as.
  [Corrected 2026-10-30: wrong for AML when written. This file's own entry for Effendi and
  Chattopadhyay (SPACE 2024), added the same day, is encrypted collaborative-AML inference under TFHE.
  The satellite half still holds as far as the searches reached, and on the satellite page nothing
  runs under encryption at all: its anomaly score is a plaintext computation in the tab.]

## Round 33 additions, verified at primary source 2026-08-12

Same rule as Rounds 31 and 32: every entry was loaded at its arXiv abstract page, IACR eprint listing,
JMLR or PNAS record, Europe PMC record, vendor documentation page or repository file before being
written here. Where a source could not be loaded it is marked **METADATA-ONLY** or **NOT FOUND** and no
number is taken from it. Nothing in this section was promoted from metadata to verified.

Three items in this batch were **extended in place rather than filed again** - the CPAD section's Cheon
et al. (2024/127) and Ogilvie (2025/1618) entries, and the "SoK: Private LLM Inference" entry in the
baselines section - because a second entry for the same source is how this file drifts.

### Reporting: calibration, uncertainty and decision curves, which this file cited nowhere

A grep across the repository for `Vickers`, `decision curve`, `net benefit` and `calibrat` returned
**zero citations** while `src/qsh/models/metrics.py` implements net benefit, integrates it over a
threshold range, and selects an operating point by it. The method was named in a docstring and credited
to no publication.

- **Vickers and Elkin, "Decision curve analysis: a novel method for evaluating prediction models",
  *Medical Decision Making* 2006;26(6):565-574.** DOI 10.1177/0272989x06295361, PMID 17099194 `[link]`
  URL: https://doi.org/10.1177/0272989x06295361
  VERIFIED via Europe PMC, 2026-08-12. The source of the method `metrics.py::net_benefit` implements:
  its docstring says "Vickers-Elkin net benefit" and carried no citation, and this file had no entry.
  The implemented form `NB(p) = TP/N - (FP/N) * p/(1-p)` is **algebraically identical** to the published
  `sensitivity x prevalence - (1 - specificity) x (1 - prevalence) x w` with `w = p/(1-p)`, since
  `sensitivity x prevalence = TP/N` and `(1 - specificity) x (1 - prevalence) = FP/N` - a reader can
  check the identity without the PDF, which is why it is written out.
  **The load-bearing point is what the 2006 contribution actually was:** net benefit plotted **across**
  threshold probabilities. A single-threshold scalar is the thing decision curve analysis was invented
  to replace, so quoting one net-benefit number as a headline inverts the paper. This repository is on
  the right side of that by construction - `area_under_net_benefit` integrates over a stated range and
  `best_threshold_by_net_benefit` reports its margin over treat-all at the SAME threshold - and the
  citation makes that deliberate rather than accidental.
- **Vickers, van Calster and Steyerberg, "A simple, step-by-step guide to interpreting decision curve
  analysis", *Diagnostic and Prognostic Research* 3:18 (2019).** PMC6777022 `[link]`
  URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6777022/
  VERIFIED via Europe PMC, 2026-08-12. The practical companion: it gives the sensitivity/specificity
  form of the formula above and names **treat-all and treat-none as the standard reference lines** -
  which is exactly the pair `best_threshold_by_net_benefit` returns, so the reference lines on the page
  are the field's, not this project's invention.
- **Pfeiffer and Gail, "Estimating the decision curve and its precision from three study designs",
  *Biometrical Journal* 62(3):764-776 (2020; online 2019-08-08).** DOI 10.1002/bimj.201800240, PMID 31394013,
  PMC8620346 `[link]`
  URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8620346/
  VERIFIED at the PMC full text 2026-11-06. Section 3.2.2, eq. 11, estimates net benefit from case-control data
  when the event rate $`\mu`$ is known:

  $$\widehat{NB}(t) = \mu\{1 - G_m(t)\} - (1 - \mu)\{1 - K_n(t)\}\,\frac{t}{1 - t}$$

  where $`G_m`$ and $`K_n`$ are the empirical distributions of the risk among the $`m`$ cases and $`n`$ controls, so
  $`1 - G_m(t)`$ and $`1 - K_n(t)`$ are the true- and false-positive rates; section 3.3 gives pointwise intervals as
  $`\widehat{NB}(t) \pm 1.96\,\widehat{\mathrm{var}}^{1/2}`$. **This is the estimator the finance page's decision curve
  uses** (`web/src/lib/decisionCurve.ts`, `src/qsh/models/metrics.py::net_benefit_at_prevalence`): the page's
  evaluation sample draws each class of the test window on its own and is enriched to 164 events in 8,192 rows, and
  the window's own 2,321 of 1,523,504 is the known rate. The variance used there is the one of two independent
  binomial proportions, written out in `metrics.py`, not the paper's influence-function form, which its online
  appendix holds and which was not read.
- **Dimitriadis, Gneiting and Jordan, "Evaluating probabilistic classifiers: Reliability diagrams and
  score decompositions revisited", *PNAS* 118 (2021) e2016191118.** arXiv:2008.03033 `[link]`
  URL: https://arxiv.org/abs/2008.03033
  VERIFIED 2026-08-12. **CORP = Consistent, Optimally binned, Reproducible**, obtained by isotonic
  regression via the pool-adjacent-violators algorithm rather than by choosing a bin count. The
  citation to reach for **if** a reliability diagram is ever added; see the Riley entry below for why
  none is added now.
- **Roelofs, Cain, Shlens and Mozer, "Mitigating Bias in Calibration Error Estimation", AISTATS 2022.**
  arXiv:2012.08668 `[link]`
  URL: https://arxiv.org/abs/2012.08668
  VERIFIED 2026-08-12. Verbatim: *"binning-based estimators with bins of equal mass (number of
  instances) have lower bias than estimators with bins of equal width."* So an equal-width ECE, which
  is the default in most tutorials, is the biased choice - a one-line design constraint on any future
  calibration reporting here.
- **Van Calster, Nieboer, Vergouwe, De Cock, Pencina and Steyerberg, "A calibration hierarchy for risk
  models was defined: from utopia to empirical data", *Journal of Clinical Epidemiology* 74 (2016)
  167-176.** DOI 10.1016/j.jclinepi.2015.12.005 `[link]`
  URL: https://doi.org/10.1016/j.jclinepi.2015.12.005
  VERIFIED 2026-10-27 in the authors' accepted manuscript (KU Leuven Lirias), bibliographic data at
  Crossref. It recommends at least 100 events and 100 non-events for the calibration intercept and
  slope, and *"at least 200 events and 200 non-events to derive flexible calibration curves"*; with
  less data such curves may be unstable and can be omitted. **This is the source of the floor that
  `web/src/lib/calibration.ts` and a row of the private verification checklist apply.** Until Round
  74 the project attributed
  the floor to Riley et al., the entry below, and a Round 51 chart drew curves below it regardless.
- **Riley et al., "Minimum sample size for external validation of a clinical prediction model with a
  binary outcome", *Statistics in Medicine* 2021.** DOI 10.1002/sim.9025 `[link]`
  URL: https://doi.org/10.1002/sim.9025
  **METADATA-ONLY.** The Wiley full text was not loaded; the thresholds below are corroborated across
  secondary sources only and are recorded at that strength deliberately: **at least 100 events and 100
  non-events** for a performance estimate, and **at least 200 events** for a flexible calibration curve.
  **Corrected Round 74:** those two rules of thumb are Collins et al. (2016) and Van Calster et al.
  (2016, the entry above). This paper argues that sample size should be calculated for the setting and
  that the rules can be too few, so it supports the floor as a minimum and is not its source.
  The paragraph below said this was "the citation that rules OUT reliability diagrams"; the ruling is
  Van Calster's, and the measured counts are attributable rather than asserted: medical **42** malignant cases in the scored
  split (`src/qsh/models/distil.py:194`), finance **164** positives in the browser's evaluation sample
  (`scripts/prep_model_inputs.py:125`), satellite **113** anomalous segments (`anomaly_rate_test`
  0.213611 x `n_test_segments` 529, `models/satellite_opssat/README.md`), and genomics **no outcome
  label at all** - 1000 Genomes carries no case/control status, which is why the PRS AUROC there is
  cited as external and never computed. So **no binary page clears the 200-event bar**, medical does
  not clear even the 100-event bar, and a smooth calibration curve drawn on any of them would be a
  picture of noise. (Telecom: its top-label question - was the winning class right -
  had 250 wrong predictions on the NIDD sample in Round 74, which cleared it; on the sample of the model
  refitted in Round 86 it has 34 wrong, which does not, nor do the 8 wrong on the QML sample.) The satellite figure above is the OPS-SAT segment model; the telemanom path is far worse
  (`n_events_scored` **3**, `models/metrics.json`), so the conclusion does not depend on which satellite
  model is meant. Recorded as the reason for an absence, which is the kind of decision this file exists to hold.
- **Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger, Cavallaro and Rieck, "Dos and Don'ts of
  Machine Learning in Computer Security", USENIX Security 2022.** arXiv:2010.09470; also
  *Communications of the ACM*, DOI 10.1145/3643456 `[link]`
  URL: https://arxiv.org/abs/2010.09470
  **Abstract VERIFIED** at arXiv 2026-08-12; the **individual pitfall texts are METADATA-ONLY**, so the
  two named below are carried as titles and prevalences rather than quoted. Pitfall **P7 Inappropriate
  Performance Measures** and **P8 Base Rate Fallacy**, present in **62%** and **46%** of their 30-paper
  survey. This is the canonical citation for why an aggregate metric hides per-class collapse - a claim
  the telecom page asserts with no source, while `metrics.py::per_class_f1` and the WS2 section of this
  file both rest on it. Engelen et al. (WTMC 2021, above) is the measured instance of the same failure;
  this is the systematic statement of it.
- **ISO/IEC 5259 series, "Artificial intelligence - Data quality for analytics and machine learning
  (ML)"**: Part 1 (overview, terminology, examples), Part 2 (data quality measures), Part 3 (management
  requirements and guidelines), Part 4 (process framework), all 2024; Part 5 (governance framework),
  2025; ISO/IEC TR 5259-6:2026 (visualization framework, a Technical Report) `[link]`
  URL: https://www.evs.ee/en/evs-en-iso-iec-5259-2-2025
  Added: 2026-11-08
  METADATA ONLY. Editions and scopes are from the Estonian standards body's catalogue and, for Part 6,
  the publisher's front-matter preview; iso.org refused automated access and no clause has been read.
  Part 2's scope is a data quality model, measures, and guidance on reporting them.
- **Collins, Moons, Dhiman, Riley, Beam, Van Calster et al., "TRIPOD+AI statement: updated guidance for
  reporting clinical prediction models that use regression or machine learning methods"**, BMJ 385,
  e078378, 2024, DOI 10.1136/bmj-2023-078378, PMC11019967 `[link]`
  URL: https://doi.org/10.1136/bmj-2023-078378
  Added: 2026-11-08
  The checklist read in full through NCBI's XML; supplementary tables not read. Minimum reporting recommendations
  for clinical prediction model studies: 27 main items, 52 subitems, replacing TRIPOD 2015. Used here for
  item 7 (pre-processing), item 11 (missing data) and item 12e (performance measures and their rationale).
  Outside clinical prediction, as with this project's telemetry, it applies only by analogy.

### Encrypted-versus-plaintext agreement: the two published reporting conventions

The demo prints a plaintext-versus-CKKS comparison and invented its own presentation of it. There are
two conventions in the literature and this repository should look like one of them.

- **Li, Wu and Miao, "Decision-Aware Quadratic ReLU Replacement for HE-Friendly Inference",
  arXiv:2605.22237, 2026-05-25** `[link]`
  URL: https://arxiv.org/abs/2605.22237
  VERIFIED at the arXiv HTML 2026-08-12. Its Table IX reports **"PH-mis"**, defined verbatim as *"the
  integer number of top-1 mismatches between plaintext polynomial evaluation and CKKS evaluation on the
  evaluated test set; it is zero for every selected row"*. So the published convention is an integer
  **COUNT**, expected **zero**, reported **beside** plaintext and ciphertext accuracy rather than folded
  into them. Bears on `web/src/lib/clientFHE.ts` (which measures `maxAbsError` against a plaintext
  reference) and on however the verification panel presents agreement.
- **Zama, Concrete-ML documentation** `[link]`
  URL: https://docs.zama.ai/concrete-ml
  VERIFIED 2026-08-12. Concrete-ML prints **"Similarity"**: the mean per-sample agreement between clear
  and FHE predictions, as a **percentage**. That is the de-facto convention from the vendor with the
  largest FHE-ML install base, and it is the complement of PH-mis rather than a rival to it - a count of
  disagreements and a percentage of agreements are the same measurement reported two ways, so quoting
  either is defensible and quoting neither is not. Also relevant to
  [`../docs/architecture.md`](../docs/architecture.md): Concrete-ML's **deployment model returns class
  scores and the CLIENT decrypts**, which is the mainstream construction for encrypted multiclass
  inference and is what this demo already does in a single tab.
- **Lee, Kang, Lee, Choi, Eom, Deryabin, Lee, Lee, Yoo, Kim and No, "Privacy-Preserving Machine Learning with
  Fully Homomorphic Encryption for Deep Neural Network", arXiv:2106.07229 (v1, 2021-06-14, the only version).**
  `[link]`
  URL: https://arxiv.org/abs/2106.07229
  **Published as** IEEE Access 10, 30039-30054 (2022), DOI 10.1109/ACCESS.2022.3159694, CC BY 4.0 (Crossref record
  read 2026-11-06). The arXiv record carries no journal reference, which is why the entry below said none existed;
  the journal version is the one to cite, and its abstract reports **98.43%** identical results and an accuracy of
  **92.43% $`\pm`$ 2.65%** against the plaintext ResNet-20's 91.89%. The pages cite it since Round 86. The preprint's
  figures, for the record - verified at the arXiv abstract 2026-11-06, verbatim: *"We numerically verify that the proposed model with the CIFAR-10 dataset shows 98.67%
  identical results to the original ResNet-20 model with non-encrypted data. The classification accuracy of the
  proposed model is 90.67%"*. The model is the standard ResNet-20 under RNS-CKKS **with bootstrapping**, its ReLU
  evaluated by a polynomial approximation. **It enters as a relation, not a number to beat** (the owner's choice,
  Round 36): every agreement figure on this project's pages encrypts one linear layer at one multiplicative level,
  with no bootstrapping and no approximated activation, so it does far less homomorphic arithmetic and should agree
  more often. `EncryptedAgreementChart` states that under its figure.

### What the cost literature rules out, and what verifiability does not prove

- **Mazzone, Everts, Hahn and Peter, "Efficient Ranking, Order Statistics, and Sorting under CKKS",
  USENIX Security 2025.** arXiv:2412.15126 `[link]`
  URL: https://arxiv.org/abs/2412.15126
  VERIFIED at the abstract 2026-08-12. Argmin/argmax over a **128-element vector in 12.83 s**, with
  comparison depth *"of up to 2 (constant)"*. This is the measured cost that **rules out a homomorphic
  argmax in a browser tab** and therefore the published justification for the architecture's choice to
  send the non-linear half to MPC (`src/qsh/mpc/secret_inference.py`, `web/src/lib/clientMPC.ts`). The
  depth result also blocks the obvious rebuttal that the cost is a depth artefact: it is constant.
- **Peng et al., "A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning",
  arXiv:2502.18535**, revised 2026-03-29 `[link]`
  URL: https://arxiv.org/abs/2502.18535
  VERIFIED at the abstract 2026-08-12. Its taxonomy is **verifiable training / verifiable testing /
  verifiable inference**. Worth recording because it gives this project's dual-path comparison a name
  from the field: it occupies the **verifiable testing** slot, not verifiable inference.
- **Atapoor, Baghery, Pereira and Spiessens, "Verifiable FHE via Lattice-based SNARKs", IACR CiC 2024,
  ePrint 2024/032** `[link]`
  URL: https://eprint.iacr.org/2024/032
  VERIFIED at the abstract 2026-08-12. Verifies an approximate neural network of **three layers and more
  than 100 ciphertexts in "less than 1 second"**.
- **Tremblay Thibault and Walter (Zama), "Towards Verifiable FHE in Practice", ACM CCS 2025, ePrint
  2024/451** `[link]`
  URL: https://eprint.iacr.org/2024/451
  VERIFIED at the abstract 2026-08-12. TFHE **bootstrapping** proved with a prover under **20 min**, a
  proof of about **200 kB**, and a verifier under **10 ms**.

> **The load-bearing point about these last two is a NEGATIVE, and it constrains what this project may
> claim.** Verifiable FHE proves that **the server ran the circuit that was requested**. It does **not**
> prove that the result matches a plaintext reference. Those are different properties: the first is
> cryptographic integrity of the evaluation, the second is numerical fidelity of an approximate scheme.
> So this repository's plaintext-versus-CKKS comparison is **empirical fidelity, not cryptographic
> integrity**, and it has to say so - a reviewer who knows this literature will otherwise read a
> side-by-side agreement figure as an integrity claim, which nothing here establishes and which would
> need one of the two constructions above to establish. Recorded in
> [`../docs/architecture.md`](../docs/architecture.md) at the place the comparison is described.

### Conformal prediction on dependent data: the honest answer changed in 2024

Both entries matter because the pre-2024 answer - "split conformal is invalid on a time series" - is
what a reviewer would expect a demo to say about the satellite page's timeline data, and it is wrong.

- **Oliveira, Orenstein, Ramos and Romano, "Split Conformal Prediction and Non-Exchangeable Data",
  *JMLR* 25(225), 2024** `[link]`
  URL: https://jmlr.org/papers/v25/ - volume 25, paper 225; the per-paper permalink is not written out
  because it was not the URL loaded and a guessed one would point at someone else's paper.
  VERIFIED at the JMLR abstract 2026-08-12. Split conformal *"remains valid for many non-exchangeable
  processes by adding a small coverage penalty"*.
- **Barber and Pananjady, "Predictive inference for time series: why is split conformal effective
  despite temporal dependence?", arXiv:2510.02471**, revised 2026-01-10 `[link]`
  URL: https://arxiv.org/abs/2510.02471
  VERIFIED at the abstract 2026-08-12. Bounds the coverage loss by a **"switch coefficient"** and the
  bound is *"sharp over the class of stationary, beta-mixing processes"*.
  So the defensible statement about interval coverage on the satellite timeline is **"valid with a
  bounded penalty"**, not "invalid" and not "valid" unqualified. No conformal method is implemented here
  (a grep for `conformal` across the repository returns nothing), so this is recorded as the citation to
  use if intervals are ever added, and as the correction to the answer a reader is likely to assume.

### Genomics reporting: the reference implementation agrees, and one practice this repo does not follow

- **pgsc_calc (PGS Catalog Calculator) documentation** `[link]`
  URL: https://pgsc-calc.readthedocs.io/
  VERIFIED 2026-08-12. Verbatim: *"The distribution of PGS scores can differ between genetic ancestry
  groups, but this does not necessarily correspond to differences in risk (e.g., disease prevalence or
  mean biomarker values). These differences arise from changes in allele frequencies and linkage
  disequilibrium (LD) patterns."* It reports PGS as **density plots** and recommends interpreting
  results *"as distributions or deciles rather than single-point estimates"*. This is the **field
  reference implementation endorsing what the genomics page already says**, which is the cheapest kind
  of citation to add and the easiest to have missed.
- **Ding et al., "Polygenic scoring accuracy varies across the genetic ancestry continuum", *Nature*
  2023.** DOI 10.1038/s41586-023-06079-4 `[link]`
  URL: https://doi.org/10.1038/s41586-023-06079-4
  **METADATA-ONLY** - paywalled, abstract only, 2026-08-12. Accuracy *"decreases as a function of
  relatedness at the individual level when modelling genetic ancestry as a continuum"*. Recorded as the
  current best practice **against discrete ancestry bins**, because this repository uses discrete 1000
  Genomes super-populations. That is a defensible simplification for a demo - the bins are what the
  corpus ships - but it is the practice a genomics reviewer will raise, so it is written down as a known
  divergence rather than left to be discovered.

### New primary literature, 2026-08-01 to 2026-08-10, all verified at the abstract and all uncited

- **Malik, Anwar and Raza, "Beyond the Quantum Promise: A Security Analysis of Classical Control in
  Quantum Key Distribution."** arXiv:2608.07626, 2026-08-07 `[link]`
  URL: https://arxiv.org/abs/2608.07626
  **The highest-priority item in this batch.** The first hybrid **Tamarin** model of the ETSI and ITU-T
  QKD specifications, finding **three specification-level vulnerabilities** - subverted entanglement
  injection, basis-deferred measurement, message reflection - each arising from *"a classical
  control-plane omission in the procedure text"*, with verified countermeasures (**measurement
  commitment**, **identity-bound MACs**). `src/qsh/qkd/sdn_controller.py` models ETSI GS QKD 015
  `PHYS`/`VIRT` links and `qkd/kme.py` implements GS QKD 014, while
  [`../docs/threat-model.md`](../docs/threat-model.md) had **no classical-control-plane adversary at
  all**. Round 33 adds one there with a resolving condition, so the gap is falsifiable rather than
  merely noted. The finding to keep in view is that these are flaws in the SPECIFICATIONS this project
  implements faithfully.
- **Chiang, "Trimming: Decoupling Multiplicative Depth from Modulus Chains in RNS-CKKS via Rational
  Levels."** arXiv:2608.00375, 2026-08-01 `[link]`
  URL: https://arxiv.org/abs/2608.00375
  Referenced from `src/qsh/models/registry.py` in Round 33; this is the provenance entry.
  **CRITICAL CAVEAT: it has no experiments, and no number from it may be quoted.** Its abstract ends
  *"The proposed framework will be further validated through concrete implementation and experimental
  evaluation"*. It contests the depth-dominance premise of **ATLAS** (arXiv:2607.23478, above) as a
  **named direction only**, which is exactly how it should be cited: a stated alternative to a
  measured claim is not a counter-measurement.
- **Bian, Choe, Kim and Lee**, IACR ePrint 2026/2490 (2026-10), to appear at IEEE S&P 2027 `[link]`
  URL: https://eprint.iacr.org/2026/2490
  Theory: a malicious evaluator holding the public coefficient-to-slot keys reacts to a client's 1-bit accept or reject.
  Verification: implemented end to end on Lattigo v6.2.0, the current release, with each query re-randomised by a fresh
  encryption of zero, so simple detection does not see it. Result: recovery of a ternary secret key. Conclusion for this
  project: answer the same either way rather than rely on detection or rate limits (`docs/threat-model.md`).
- **IACR ePrint 2026/2575** `[link]`
  URL: https://eprint.iacr.org/2026/2575
  Cited in `docs/threat-model.md` beside 2026/2490 for the same section. Read 2026-11-03.
- **IACR ePrint 2026/2546** (CCS 2026), **2026/1671**, **2026/2524** (GCKKS), **2026/2675** `[link]`
  URLs: https://eprint.iacr.org/2026/2546 , /2026/1671 , /2026/2524 , /2026/2675
  In order: IND-CPA-D attacks on BFV and BGV with OpenFHE BFV key recovery in minutes to hours; a machine-checked flooding
  bound with a square-root-of-queries loss; two ways of keeping precision after flooding, the second for threshold CKKS.
  Abstracts read 2026-11-03; no number from them is used here.
- **Zhang, Zhou, Han, Liu and Huang, "Adaptively CCA-Secure Threshold Decryption from Lattices with
  Polynomial Modulus."** IACR ePrint 2026/1627, received 2026-08-06, fourth revision 2026-09-19 (it was first
  titled "Adaptively Secure Threshold Decryption from LWE with Polynomial Modulus") `[link]`
  URL: https://eprint.iacr.org/2026/1627
  Adaptive corruption and leakage from **decryption queries**, in the random-oracle model with bounded
  queries, via polynomial noise flooding and a coupled refined leftover hash lemma. That is the same threat surface as the CPAD
  section above, which had **no threshold-setting citation** even though OpenFHE scopes the CPAD attack
  to precisely the threshold case and `submodules/HEonGPU` ships N-out-of-N collective bootstrapping.
- **Al-Mohammed and Al-Ali, "Machine Learning for Specialized QKD Aspects: A Survey of Adaptive
  Protocols, Free-Space Links, 6G Integration, and Steerability-Aware Security."** arXiv:2608.08280,
  v1 2026-08-08, v2 2026-08-18 `[link]`
  URL: https://arxiv.org/abs/2608.08280
  Names *"the boundary between ML-based decision support and security certification"* as an open
  challenge. Independent support for a line this repository already carries at
  `references/model-choice-vs-2026-practice.md:200` - *"ML as an overlay for sub-threshold stealthy
  attacks, never as a replacement for decoy-state accounting"* - which until now rested on a single 2026
  study cited there and on no statement about the certification boundary itself. A survey is weak
  evidence for a mechanism and adequate evidence for a boundary, which is all it is used for here.
  **Read in the body in Round 70 (2026-10-25).** Its section V-B reports deep-RL adaptive optimisation
  at SKR +15-20 percent and QBER -30-40 percent under noisy conditions, citing Shingne et al. (DRL with
  a VAE); the abstract gives no figures. This is consistent with AML-QKD v2's +14-25 percent. Neither
  figure is comparable with the browser's OptiQKD gain, which is measured over this demo's default
  parameters (symmetric bases), and the page and the run log say so.
- **Chen et al., "DYNAFIX: Dynamic Fixed-Point Encoding for Arbitrary-Range MPC."** IACR ePrint
  2026/1612, received 2026-08-05 `[link]`
  URL: https://eprint.iacr.org/2026/1612
  Bears on `src/qsh/mpc/secret_inference.py` and on the fixed-point weight encoding in
  `models/genomics_prs/weights.json` - the same problem bioETH-PRS (arXiv:2605.21634, above) solves on
  the TFHE side, now on the MPC side.

### Negative results and closed searches, verified 2026-08-12

Recorded so no future round re-runs these. A measured horizon is more useful than "nothing found",
so each one names what was checked and where it stopped.

- **The window after 2026-08-11 is structurally EMPTY, and here are the horizons.** arXiv's newest
  cs.CR announcement batch is **2026-08-11**, with no 2026-08-12 batch; IACR ePrint's highest number is
  **2026/1631** (received 2026-08-07, approved 2026-08-10) and **2026/1632 returns HTTP 404**; the
  Federal Register API returns **total count 0** for "post-quantum cryptography" with `publication_date`
  greater than or equal to 2026-08-01; and NIST CSRC's **only** August 2026 item is **2026-08-06,
  NCCoE CSWP 36F on 5G NAS message security**, which is not PQC. So **no PQC instrument exists in
  August 2026 at all**, and the newest one remains OMB M-26-15 (2026-06-24). Any future round that wants
  to re-check should start from those four horizons rather than from a fresh sweep.

  **Round 43 correction, against this project's own interest: the CSRC count above is wrong by one.**
  The listing carries **two** August 2026 items, not one - 2026-08-06 CSWP 36F and **2026-08-05,
  *Transit Cybersecurity Framework Community Profile***. The second is a sector CSF profile and not a
  cryptographic instrument, so **the conclusion is unaffected**: there is still no PQC instrument in
  August 2026. The word "only" is what was wrong, and a stated horizon that undercounts is exactly
  what a later round trusts instead of re-running.

  **Method limit on that correction, stated because it bounds what the correction may be used for.**
  The re-check went through a summarising fetch rather than the CSRC publications API. That is enough
  to establish that a second August item exists - a correction that costs this project a claim, so the
  direction of error is safe - and **not** enough to assert an exhaustive August list. A round wanting
  an authoritative count should hit the API directly.

  **Horizons re-stamped 2026-08-15, at primary source.** The Federal Register API returns the literal
  `"count":0` for "post-quantum cryptography" published on or after 2026-08-01; arXiv cs.CR's newest
  submission is **2608.13465** (2026-08-13T16:46:56Z); IACR ePrint **2026/1644** is live and
  **2026/1645** returns 404. **2026-08-14 and 2026-08-15 are structurally empty** on both preprint
  servers, so a sweep of those two days is a sweep of an empty set and is recorded as one rather than
  as a search that found nothing.
- **No published in-browser FHE or MPC, and no prior art on VISUALISING encrypted inference - FIFTH
  consecutive round.** An IACR full-text search for "WebAssembly" returns **8 results across all
  years**, the newest crypto-relevant one still **2025/487 webSPDZ**. The composition novelty claim has
  now survived five independent attempts to refute it, which is worth more than any single sweep.
- **Bland and Altman (1986) is NOT FOUND and must not be cited.** The canonical method-comparison paper
  was not loaded: PubMed **2868172** returned a reCAPTCHA and Europe PMC returned navigation chrome
  only. It is the obvious citation for a plaintext-versus-ciphertext agreement plot, which is exactly
  why this negative is recorded - the temptation to cite it from memory is the hazard. Use the PH-mis
  count or Concrete-ML's Similarity convention above, both of which were actually loaded.
- **HEBench defines no error metric, tolerance or accuracy criterion for approximate schemes.** There is
  nothing in it to cite as a reporting standard, so this repository's existing note that HEBench covers
  only primitive operations is correct and stays. Checked 2026-08-12.
- **This folder cannot ground a UX or accessibility recommendation, and every round has been asked
  for one anyway. Measured 2026-08-15, said out loud here for the first time.** The standing
  per-round request asks for a UX review *grounded in* `references/`. A grep of all **16** markdown
  files for `usability`, `accessibility`, `WCAG`, `human-computer`, `human factors`, `affordance`,
  `user experience`, `HCI`, `cognitive load` and `eye-tracking` returns hits in exactly **one** file,
  and that file is THIS one - the sentence you are reading is the only match, because recording the
  absence created it. Excluding this file the count is **zero**, which is the substantive claim and
  is unchanged. Re-measured Round 49; the original "zero" was true when written and is now literally
  one by self-reference, which is worth stating rather than silently editing the number to something
  that reads as a real source having appeared. Widening to
  `interface design`, `visualis/visualiz`, `readability`, `legibility` and `colour-blind` returns
  **two**, and neither is a source: one is `he-scheme-selection.md` calling a twelve-coefficient
  model readable, and the other is the entry four bullets above recording that no prior art on
  visualising encrypted inference exists. The bundled PDFs are quantum and cryptography papers:
  **7 committed** to the repository, with 4 more present on a working checkout and deliberately not
  redistributed (`DATA_LICENSES.md` section 8 adjudicates each). This said "All **15**" until Round
  49, a count matching neither figure, and the one business-of-quantum title it alluded to is among
  the four that are NOT committed. **There is no HCI or human-factors source in this reference set at
  all.**

  So the UX rows do not rest on this folder, and any claim that they do would be false. What they
  rest on is stated instead: the **WCAG 2.2 success criteria themselves**, quoted by number in
  `docs/browser-verification-checklist.md`, plus that file's own measurement procedures - composite
  the alpha before reading a contrast ratio, read the `role` the element carries rather than the prop
  that was passed, enumerate the series from the source rather than from the last sweep's list. That
  is a defensible basis; it is simply a different one from a paper in this folder.

  **Not resolved by adding a citation to look resolved.** A single HCI paper bolted on here would let
  the round report "grounded in references/" while the rows continued to come from WCAG, which is the
  decoration-condition failure, a deferral whose stated condition is not the one that binds. If an
  HCI source is ever added it has to be because
  a row actually derives from it. Recorded so that no future round answers this bullet by asserting
  a grounding it does not have - which is what nine rounds of silence amounted to.

### Leads recorded rather than cited

- **EU NIS Cooperation Group PQC roadmap milestones - DEADLINE EXECUTED Round 42, and the lead
  was wrong twice in the flattering direction.** The primary document is *A Coordinated Implementation Roadmap
  for the Transition to Post-Quantum Cryptography*, Part 1, Version 1.1, EU PQC Workstream,
  11.06.2025, the first deliverable of the NIS CG work stream on PQC. Fetched and read as PDF; its
  section 4 timeline is now quoted in [`../docs/pqc-roadmap.md`](../docs/pqc-roadmap.md).

  What this entry said, against what the document says:
  - *"full transition by 2035-12-31"* - **no.** 2035 is medium-risk completed and low-risk
    *"completed as much as feasible"*, and the executive summary reads *"completed for as many
    systems as practically feasible"*. The lead was stronger than its source.
  - *"`docs/pqc-roadmap.md` is **US-instrument-only**"* - **no.** That file already said *"Every
    instrument below is a US one, apart from the BSI and Japanese entries"* and recorded the EU set
    as *"Verified absent, not overlooked"*. The roadmap was precise; this lead dropped its qualifier.

  The dates themselves were right, which is how both overstatements survived being re-read.

  **Why it took nine rounds.** This entry ended *"Next round: find the Cooperation Group publication
  itself"* and was written in Round 33. `tests/test_register_deadlines.py` reads
  `docs/deliverable-verification.md` and matches `by **Round N**` or `BLOCKED UNTIL <date>`; this
  obligation was in a different file, in a third phrasing. Both halves of that gate's scope have been
  widened, and this sentence is the seed it was widened against.
- **91 FR 38487 was cited in this file and nowhere else; that is now closed.** The second same-day
  quantum executive order of 2026-06-25 ("Ushering in the Next Frontier of Quantum Innovation", FR
  document 2026-12910) was verified at the Federal Register API in Round 32 and recorded here, while
  `docs/pqc-roadmap.md` cited only EO 14412 / 91 FR 38483. Round 33 adds the cross-reference to that
  roadmap, stating only what was verified - title, document number, FR citation and date - because its
  text was never loaded and it therefore contributes no deadline.

## Round 70 re-check, oldest-verified areas first, read at primary source 2026-10-25

### New primary literature since August 2026, verified at the abstract or the body

- **arXiv:2608.08280** (Al-Mohammed and Al-Ali, the 2026-08 survey of ML for QKD) was already entered
  above, first cited on 2026-08-12 for the certification boundary. Round 70 read its body, and its
  section V-B figures are recorded in that entry rather than in a second one (merged in Round 73).
- **Laschet, Lendvay, Lorunser, James, Torresetti and Colombo, "Software Defined Networks Key Relay
  for Large-Scale Quantum Key Distribution Networks."** arXiv:2608.17539, 2026-08-18 `[link]`
  URL: https://arxiv.org/abs/2608.17539
  The paper describes an SDN controller that aggregates link KPIs and selects relay paths by Dijkstra
  or by max-min capacity with load balancing. For multiple domains it adds an oblivious multi-party
  protocol, so providers need not disclose their networks. The demo's SDN panel builds one
  trusted-relay chain when a span is exceeded and selects among no paths. The panel's "path
  computation" is the control plane's role as the standards define it, not a claim of path
  optimisation.

### Standards and specifications, re-checked

- **FIPS 206 (FN-DSA) and an HQC draft: still none.** The NIST PQC project page (last modified
  2026-08-05) lists Falcon as "selected for ongoing standardization; that process is underway" and
  HQC as selected, with no initial public draft of either. `docs/pqc-roadmap.md` holds.
- **ETSI QKD.** GS QKD 014 V1.1.1 is still the only edition, GS QKD 015 is V2.1.1 and GS QKD 020 is
  V1.1.1 (2026-06), as the August fact-check recorded. GR QKD 007 V1.2.1 (2026-01, vocabulary) is a
  group report and changes no interface this project models.
- **Model Context Protocol.** The current specification revision is **2026-07-28**
  (`schema/2026-07-28/schema.ts`). It describes stateless, self-contained requests with per-request
  capability negotiation, and opt-in extensions (Tasks, Skills over MCP, MCP Apps). The channel's
  messages follow it since Round 86: requests "**MUST** include a string or integer ID", required `_meta`
  fields, `resultType` on every result, and a custom transport "**MUST** preserve the JSON-RPC message
  format, the message patterns, and the per-request metadata model". Before that this entry said the
  channel depended on none of it, which was true only because its messages were not MCP. The Python
  SDK's latest release is 2.3.0 (2026-10-02). `qsh/mcp/server.py` accepts both the 2.x `MCPServer` and
  the 1.x `FastMCP` spelling; the floor is `mcp>=2.2.0` (`requirements-ai.txt`), since `mcp>=1.2` admits
  releases with high-severity advisories.
- **AML-QKD.** arXiv:2603.04192 is still at v2 (2026-06-01). The entry above holds.

## Round 77 additions: the satellite pass, the network beyond the horizon, and the link templates

Read at primary source on 2026-10-28 (arXiv full texts, plus the published text of Naughton et al.
on UNSWorks). Each value a template carries is quoted in its JSON file under
`src/qsh/config/profiles/`, and `tests/test_link_profiles_are_sourced.py` refuses a value with
neither a quote nor a stated assumption.

- **Kasten and Young, "Revised optical air mass tables and approximation formula"**, Appl. Opt. 28,
  4735 (1989). METADATA-ONLY `[doi]` The formula `1 / (sin(g) + 0.50572 (g + 6.07995)^-1.6364)` the
  downlink's atmosphere scales by (`channel.py::air_mass`, `quantumSim.ts::airMass`). Bourgoin et al.
  (below) show the same scaling reproduces a MODTRAN elevation curve.
- **Liao et al., "Satellite-Relayed Intercontinental Quantum Network"**, Phys. Rev. Lett. 120,
  030501 (2018). arXiv:1801.04418 `[link]` URL: https://arxiv.org/abs/1801.04418
  VERIFIED 2026-10-28 in the arXiv text. The trusted relay the Network tab models: Micius "performs
  bitwise exclusive OR operations between the two keys and relays the result to one of the ground
  stations", "a secret key is created between China and Europe at locations separated by 7600 km";
  the stations' coordinates and altitudes; "circles Earth every 94 minutes"; "passes over the three
  ground stations allowing for a downlink for a duration of ~300 s". Per-pass final keys of 61 to 398
  kb, each station's listed total 400 to 833 kb (the text's "final key length ranging from 400 kb to
  833 kb" is those totals, not a per-pass figure); "we establish a 100 kB secure key between Xinglong
  and Graz", more than Xinglong's listed 400 kb, so it drew on passes the table does not list. The
  model reproduces the distances from the coordinates within 1 percent (7,621 and 2,524 km against
  7,600 and 2,500).
- **Vergoossen, Loarte, Bedington, Kuiper and Ling, "Satellite constellations for trusted node QKD
  networks"**, arXiv:1903.07845 (2019); published as "Modelling of satellite constellations for
  trusted node QKD networks", Acta Astronautica 173, 164-171 (2020), doi:10.1016/j.actaastro.2020.02.010.
  `[link]` URL: https://arxiv.org/abs/1903.07845
  VERIFIED 2026-10-28 in the arXiv text. The constellation the Network tab models: "a
  string-of-pearls configuration of satellites in the same plane (intra-planar ISL)", "Satellites
  have one Micius-type Earth-pointing terminal, and where specified another terminal available for
  inter-satellite links", QKD "only during passes where both the satellite and the ground station are
  in eclipse", and inter-satellite links giving "a 20-100% increase in message size". Its Table 3
  models Liao's 19 December pass (3,926,729 detections, 1,963,364 sifted, 521,513 secret bits); its
  sifted count is exactly half its detections, so it counts every state where Liao counted signals.
  It states no ISL distance, loss, aperture or divergence; the Micius-type terminal values are Liao
  2017's.
- **Khmelev et al., "Eurasian-Scale Experimental Satellite-based Quantum Key Distribution with
  Detector Efficiency Mismatch Analysis"** (the Micius pass over Zvenigorod), arXiv:2310.17476
  (2023); published in Opt. Express 32(7), 11964 (2024), doi:10.1364/OE.511772 (abstract: "a total
  final key length of 310 kbits"). `[link]` URL: https://arxiv.org/abs/2310.17476
  VERIFIED 2026-10-28 in the arXiv text. The template `micius_2022_zvenigorod`: a 600 mm telescope
  with obstruction 0.73 at 198 m, an atmospheric extinction coefficient "kappa = 0.22 +/- 0.04, which
  corresponds to clear weather conditions" (0.88 dB at the zenith, magnitudes per air mass), channel
  efficiencies of 21, 35, 37 and 19 percent, detectors of about 60 percent, $`290 \pm 60`$ background
  clicks per second, operating elevations "from 20 deg to 90 deg". Its measured sifted rate, on
  signal states, "changes from around 20.7 kbit/s at 600 km to 2.9 kbit/s at 1,100 km": this model
  gives -0.3 dB at 600 km and +1.75 dB at 1,100 km. The paper attributes its own model's misfit to
  "additional losses induced by pointing errors"; on both Micius passes checked here the measured
  rate falls about 2 dB further than diffraction and air mass predict.
- **Naughton et al., "Design considerations for an optical link supporting intersatellite quantum key
  distribution"**, Opt. Eng. 58(1), 016106 (2019), doi:10.1117/1.OE.58.1.016106. VERIFIED 2026-10-28
  in the published text (UNSWorks, CC BY-NC-ND). The template `cubesat_naughton_2019`: "80-mm diameter
  optical terminals", "a clear aperture of 80 mm and a 13% obscuration", "an assumed average optical
  transmission factor of 0.7", and a divergence of "6.1 urad at the 1/e2 intensity values ... half of
  the full beam angular dimension" for a link of "satellite separations up to around 400 km". Its own
  link is entanglement-based; the template runs Micius's decoy source, and says so.
- **Sidhu, McCarthy, Paterson and Oi**, arXiv:2602.11833v1 (2026). `[link]`
  URL: https://arxiv.org/abs/2602.11833v1
  v2 (2026-10-29) is retitled *Operational performance of finite-key entanglement-based satellite quantum
  key distribution*, adds Walk and Eisert as authors and rewords the sentence below; the 10 degrees and the
  0.73 dB zenith atmosphere are unchanged. VERIFIED 2026-10-28 in the v1 text: a "minimum elevation of theta_min = 10 deg for both links to
  account for local restrictions", the same cut as Liao 2017's and the default here. Its MODTRAN
  atmosphere at 785 nm is 0.73 dB at the zenith.
- **Bourgoin et al., "A comprehensive design and performance analysis of LEO satellite quantum
  communication"**, New J. Phys. 15, 023006 (2013). arXiv:1211.2733 `[link]`
  URL: https://arxiv.org/abs/1211.2733
  VERIFIED 2026-10-28 from its figure: a MODTRAN zenith loss of about 3.2 dB at 830 nm (rural
  aerosols, 5 km visibility) whose elevation curve equals the zenith transmittance raised to the
  Kasten-Young air mass. Not a template: a hazy-sky atmosphere, about three times the clear-sky
  values above.
- **Li et al., "Microsatellite-based real-time quantum key distribution"** (Jinan-1), Nature 640, 47-54 (2025),
  arXiv:2408.10994; the Table 1 used here is arXiv v1's (the version of record reports up to 1.07
  million bits in one pass and moves the per-pass results to Extended Data, not read). `[link]` URL: https://arxiv.org/abs/2408.10994
  READ 2026-10-28. Not a template yet: it states no dark counts, atmosphere, pointing loss or
  satellite optics efficiency, and fitting them to its own pass would make that pass a circular
  check. Its divergence is a "full angle at 1/e2 of maximum" (9 to 10 urad), a Gaussian convention 3
  dB away from this model's top-hat law.
- **Gandara Gandara et al., "Atmospheric channel model for QKD over satellite megaconstellations"**,
  arXiv:2610.27856 (2026-10-20). READ 2026-10-28. Not adopted: its elevation axis behaves as a zenith
  angle, its validation compares against a review's summary numbers, and it drops Liao's "at 1200 km"
  qualifier on the 3-8 dB.

## Round 78 additions: the low-elevation question, verified at primary source

Read 2026-10-28. What each grounds is stated; where a source was read and not adopted, so is the
reason.

- **Liao et al. 2017, Extended Data Table 2** (Nature 549, 43). VERIFIED in the published table
  image. Every Micius pass over Xinglong from September 2016 to May 2017 - 23 rows of highest altitude
  angle, shortest distance, peak and average sifted rate and QBER - now a validation set
  (`src/qsh/config/validation/liao_2017_xinglong_passes.json`). The 19 December 2016 pass is listed at
  47.79 deg and 645.08 km, which implies a 491.5 km orbit that night against the paper's "~500 km".
- **Zhang et al., "Astronomical Observing Conditions at Xinglong Observatory from 2007 to 2014"**,
  arXiv:1601.08114 (2016). `[link]` URL: https://arxiv.org/abs/1601.08114
  VERIFIED in the arXiv text: the DIMM's 2014 "Mean and median seeing are around 1.9'' and 1.7''",
  December's median 2.05''. Grounds `free_space_seeing_arcsec`.
- **Liu, Giordano, Yao, Vernin, Chadid, Wang, Yin and Wang, "Optical Turbulence Characterization at
  LAMOST Site: Observations and Models"**, arXiv:1505.05436 (MNRAS). `[link]`
  URL: https://arxiv.org/abs/1505.05436
  VERIFIED in the arXiv text: Eq. (1) $`r_0 = 0.185\,\lambda^{6/5} \left(\int C_n^2\,dh\right)^{-3/5}`$
  and Eq. (2) $`\epsilon_0 = 0.98\,\lambda / r_0`$, at LAMOST, which is at Xinglong. Grounds the
  optional turbulence coupling (`channel.py::turbulence_coupling_db`); the slant path's
  $`\sec\zeta`$ is this project's step.
- **Lu et al., "Micius quantum experiments in space"**, Rev. Mod. Phys. 94, 035001 (2022),
  arXiv:2208.10236, Eq. (9). VERIFIED in the arXiv text: the expected mispointing loss
  $`w^2 / (w^2 + 4\sigma_p^2)`$ for Gaussian jitter (after Toyoshima and Araki 1998). Grounds the Jinan-1
  template's pointing loss.
- **Li et al., "Microsatellite-based real-time quantum key distribution"** (Jinan-1), Nature 2025,
  arXiv:2408.10994. Now a TEMPLATE (`jinan1_2022_jinan`), with its stated values quoted, its unstated
  ones marked as assumptions from the sources above, and one constant - the station efficiency -
  calibrated to its 25 September 2022 pass. Its Table 1 lists 21 passes (its title says 20).
- **Vasylyev, Vogel and Moll, "Satellite-mediated quantum atmospheric links"**, Phys. Rev. A 99,
  053830 (2019), arXiv:1901.07452; **Liorni, Kampermann and Bruss, "Satellite-based links for quantum
  key distribution: beam effects and weather dependence"**, New J. Phys. 21, 093055 (2019),
  arXiv:1906.08115. READ 2026-10-28. Not adopted: both put downlink beam wander and broadening well
  under the residual in question.
- **Khmelev et al. 2023**, Fig. 3a, re-read: the model the paper fits matches the rising half of its
  pass and the data fall below it only on the falling half, which an elevation-only term cannot
  produce.


## Round 79 additions: the satellite key block and its finite-key bound, verified at primary source

Read 2026-10-28. The user chose a finite key for the downlink and asked the published practice to
decide its block. What each source grounds is stated; where one was read and not adopted, so is the
reason.

- **Lim, Curty, Walenta, Xu and Zbinden, "Concise security bounds for practical decoy-state quantum
  key distribution"**, Phys. Rev. A 89, 022307 (2014). arXiv:1311.7129 `[link]`
  URL: https://arxiv.org/abs/1311.7129
  VERIFIED in the arXiv text and, for the constant in Eq. (5), in the rendered page: Eqs. (1) to (5)
  with $`\epsilon_{sec} = 21\epsilon`$, the conditions $`\mu_1 > \mu_2 + \mu_3`$ and
  $`\mu_2 > \mu_3 \ge 0`$, and $`\gamma`$'s factor $`21^2 / a^2`$. Implemented as `finite_key.vacuum_decoy_key_length` and `finiteKey.ts`, for a source
  that sends a signal, a weak decoy and the vacuum.
- **Rusca, Boaron, Grunenfelder, Martin and Zbinden, "Finite-key analysis for the 1-decoy state
  QKD protocol"**, Appl. Phys. Lett. 112, 171104 (2018). arXiv:1801.03443 `[link]`
  URL: https://arxiv.org/abs/1801.03443
  RE-READ for this round: Table I's 2-decoy rows (236 kHz, 2,503, 197 and 14.1 Hz at
  $`n_Z = 10^7`$; 355 kHz, 3,881, 333 and 30.7 at $`10^9`$), the vacuum held at
  $`\mu_3 = 10^{-6}`$ (Fig. 4), and the vacuum share
  "always greater than 11%" at the optimum (Appendix C). Two findings about this repository's own
  1-decoy implementation: (A21) prints $`21^2`$, which the code had as $`2^{21}`$; and (A16)
  bounds the vacuum by "the errors relative to one intensity" without naming it, where the code had
  the signal's and the decoy's is tighter at every row. RE-READ in Round 82, on the rendered page at
  300 dpi: (A16) puts $`\tau_0`$ on the error term only,
  $`s^u_{Z,0} = 2\left(\tau_0 \frac{e^{k}}{p_k}\left(m_{Z,k} + \delta(m_Z, \epsilon_1)\right) + \delta(n_Z, \epsilon_1)\right)`$,
  as (A13)-(A15) derive it; the code had $`\tau_0`$ on $`\delta(n_Z, \epsilon_1)`$ too, which
  understated the vacuum and overstated the key. With the printed form the 26 dB row reads
  1.5 percent below the table under the paper's dead-time convention, where the old form matched it
  to a few parts in a thousand: the table may have been computed the old way.
- **Liao et al. 2017** (Nature 549, 43), arXiv:1707.00542, re-read for its key: "mu_s=0.8, mu_1=0.1,
  mu_2=0", "a hamming algorithm is used for error correction", "the intensity fluctuation for the
  signal state and decoy state (<5%)", and a "secure final key of 300,939 bits when the statistical
  failure probability is set to be 10-9". It does not name its finite-key analysis.
- **Khmelev et al. 2023** (arXiv:2310.17476), re-read for its key: "utilize Chernoff bounds with a
  failure probability of 10-9", "the secure final key is 629,000 bits", error correction at "an
  inefficiency fec = 1.44", and 310,400 bits with detector-efficiency mismatch, the key "finally
  shared". Their Eq. (7) is applied "to the divided data among two bases X and Z": "both bases are
  used for key generation". How the model's key on the same session compares, on one basis and on
  both: [`../docs/models-and-benchmarks.md`](../docs/models-and-benchmarks.md#the-qkd-key-blocks). Its
  Chernoff bounds cite Zhang et al. 2017 (its ref. [46]), below.
- **Yin, Zhou, Gu, Xie, Lu and Chen, "Tight security bounds for decoy-state quantum key
  distribution"**, Sci. Rep. 10, 14312 (2020). arXiv:2002.06530 `[link]`
  URL: https://arxiv.org/abs/2002.06530
  VERIFIED on the rendered pages (Round 82): for a sum of independent Bernoulli variables with
  probabilities not necessarily equal, the expectation $`x^*`$ from the observed value $`x`$, with
  failure probability $`\epsilon`$ and $`\beta = \ln(1/\epsilon)`$, is at most
  $`x + \beta + \sqrt{2\beta x + \beta^2}`$, Eq. (9), and at least
  $`x - \beta/2 - \sqrt{2\beta x + \beta^2/4}`$, Eq. (10), floored at zero; closed forms slightly looser
  than the roots of their Eqs. (7)-(8). Their Appendix B calls the inverse-solution Chernoff method of
  Zhang et al. 2017 not rigorous, since it puts the observed value where the bound needs the
  expectation. Grounds `finite_key.py::_chernoff_upper` / `_chernoff_lower`, checked in
  `tests/test_the_finite_key_statistics_are_yin_2020s.py` by drawing sums and counting misses.
- **Zhang, Zhao, Razavi and Ma, "Improved key-rate bounds for practical decoy-state quantum-key-
  distribution systems"**, Phys. Rev. A 95, 012333 (2017). arXiv:1611.02524 `[link]`
  URL: https://arxiv.org/abs/1611.02524
  The Chernoff bounds Khmelev et al. 2023 used. Not implemented: on Liao's pass and Khmelev's session
  its analytic form, as Yin et al.'s Appendix B gives it, differs from Yin et al.'s by under 0.3
  percent of the key, and Yin et al. question its rigour.
- **Liao et al., "Space-to-ground quantum key distribution using a small-sized payload on Tiangong-2
  space lab"**, Chin. Phys. Lett. 34, 090302 (2017). VERIFIED in the published text: one pass, "a
  mean sifted key rate of 783 bps in 143 s", "112031 bits sifted key", "a 13,119 bit secure key".
  Grounds that a flown downlink's finite key is computed over one pass.
- **Sidhu, Brougham, McArthur, Pousa and Oi, "Finite key effects in satellite quantum key
  distribution"**, npj Quantum Inf. 8, 18 (2022). arXiv:2012.07829 `[link]`
  URL: https://arxiv.org/abs/2012.07829
  VERIFIED in the arXiv text: "Systems with zero single-overpass SKL can generate positive key by
  accumulating signals from several overpasses (Fig. 6)", with the M-pass key at least M times one
  pass's. Grounds the pooled-passes block and its test.
- **Bourgoin et al., "Experimental quantum key distribution with simulated ground-to-satellite
  photon losses and processing limitations"**, Phys. Rev. A 92, 052339 (2015). arXiv:1512.05789
  `[link]` URL: https://arxiv.org/abs/1512.05789
  READ: per-pass finite keys, and three passes pooled where one made none. Cited as practice, not
  checked against: its passes are a ground emulation.
- **Li et al. 2025** (Jinan-1, arXiv:2408.10994), re-read for its blocks: error correction on
  100-kbit packets and privacy amplification on 500-kbit blocks inside one orbit, the block keys
  summed per orbit. Its Table 1's four passes used in Round 78 are now checked as finite keys.
- **Orsucci et al. (DLR), arXiv:2404.05668 (2024)**; **Sidhu et al., Commun. Phys. (2023),
  arXiv:2301.13209**; **Mannalath et al., arXiv:2605.29706 and arXiv:2607.17690 (2026)**. READ by the
  round's research: single pass, pooled passes and sub-pass segments as the options a design weighs,
  and pulse-count thresholds for a positive key. Not adopted as checks: they are models, not passes.

## Round 80 additions: the fibre key's block, verified at primary source

Read 2026-10-28. The user chose a finite key at every fibre site and asked the published practice to
decide its block. What each source grounds is stated.

- **Rusca et al. 2018** (arXiv:1801.03443), re-read for its channel and its blocks: $`n_Z`$, the Z-basis
  detections, "defines the privacy amplification (PA) block size"; Table I's 1e7 and 1e9 rows with the
  time each block takes (14 s at 26 dB, 20 min at 46 dB, for 1e7); and "the detector and the internal
  losses of Bob's apparatus are included in the global attenuation". The last is the finding of the
  round: this repository's two Table I tests had applied a 50 percent detector on top of it.
- **Boaron et al., "Secure quantum key distribution over 421 km of optical fiber"**, Phys. Rev. Lett.
  121, 190502 (2018). arXiv:1807.03222 `[link]` URL: https://arxiv.org/abs/1807.03222
  VERIFIED in the arXiv text: privacy-amplification blocks of 8.2e6 bits to 302 km, and at 405 and
  421 km reduced "by more than a factor of ten" to keep the acquisition "shorter than one day".
- **Grunenfelder et al., "Performance and security of 5 GHz repetition rate polarization-based quantum
  key distribution"**, Appl. Phys. Lett. 117, 144003 (2020). arXiv:2007.15447 `[link]`
  URL: https://arxiv.org/abs/2007.15447
  VERIFIED in the arXiv text: "one privacy amplification block consists of 4192 error correction
  blocks" of 1944 bits, 8,149,248 bits, at $`\epsilon_{sec} = 10^{-9}`$ and $`\epsilon_{cor} = 10^{-15}`$.
- **Li et al., "High-rate quantum key distribution exceeding 110 Mb/s"**, Nat. Photon. 17, 416 (2023).
  arXiv:2307.02364 `[link]` URL: https://arxiv.org/abs/2307.02364
  VERIFIED in the arXiv text and supplement: the 1-decoy protocol "within a realistic finite-key size
  ($`n_Z \le 10^8`$)", Rusca et al.'s bound, and a block time of 1.58 s at 50 km.
- **Lucamarini et al.**, Opt. Express 21, 24550 (2013), arXiv:1310.0240, and IEEE JSTQE (2015),
  arXiv:1503.07335. READ by the round's research: a fixed 20 minute acquisition, the minority choice,
  and the smallest positive samples (1.4e5 counts; 1.6e4 sifted bits).
- **Yuan et al. (Toshiba)**, J. Lightwave Technol. 36, 3427 (2018), arXiv:1807.04484; **the Cambridge
  network** (npj Quantum Inf. 2019); **the Tokyo QKD Network** (arXiv:1103.3566); **SECOQC** (New J.
  Phys. 2009); **SwissQuantum** (arXiv:1203.4940). READ by the round's research: deployed systems fix
  the block as bits (1e6 to 1e8) and let the time follow; none fixes it as a pulse count.
- **Lim et al. 2014** re-read: the "post-processing block size" is $`n_X`$, a count of key-basis
  detections, evaluated at 1e4 to 1e9.
- **ETSI GS QKD 005 (2010) and GS QKD 016 V2.1.1 (2024)**, read by the round's research: the block
  size must be specified; neither gives a value. A search-engine summary claiming ETSI recommends 1e8
  or more is not supported by either document.

## Round 83 additions: sources cited without an entry, and the sources this round's corrections rest on

Read 2026-10-30 at the primary source each entry names. The first two were cited elsewhere in the
repository with no entry here; the other three ground corrections made on that date in this file and
in the technical report.

- **Huang, Maeng and Suh, "Beyond Latency: A System-Level Characterization of MPC and FHE for
  PPML"**, arXiv:2604.00169 [cs.CR], v1 2026-03-31, v2 2026-04-21, whose comment says it carries minor
  corrections to the ISPASS 2026 accepted paper `[link]`
  URL: https://arxiv.org/abs/2604.00169
  Read at the abstract. One characterisation of two MPC variants (arithmetic/binary sharing
  conversion, and function secret sharing) and FHE for private inference on CNN and Transformer
  models, across LAN and WAN settings, model sizes, batch sizes and input lengths, measuring energy
  and monetary cost as well as performance, with offline and online computation and communication
  counted. Cited for system-level cost by the use-case pages' footnote
  (`web/src/pages/UseCasePage.tsx`, under a shortened title), with no entry here until now. No number
  on the page or here is taken from it.
- **Yugay, Cui, Pei and Zaytsev, "Strong Linear Baselines Strike Back: Closed-Form Linear Models as
  Gaussian Process Conditional Density Estimators for TSAD"**, arXiv:2602.00672 [cs.LG], 2026-01-31
  `[link]`
  URL: https://arxiv.org/abs/2602.00672
  Read at the abstract: an autoregressive anomaly score with the closed-form OLS solution
  *"consistently matches or outperforms state-of-the-art deep detectors"*, read as a finite-history
  Gaussian process conditional density, and the authors ask future work to include strong linear
  baselines and to build benchmarks that show where deep models help. Discussed, and an earlier
  misreading of it corrected, in [`model-choice-vs-2026-practice.md`](model-choice-vs-2026-practice.md),
  with no entry here until now. It concerns an unsupervised score fitted on the series itself, not a
  student distilled from a teacher, so it does not by itself support the satellite's linear tier.
- **Pinkas, Schneider and Zohner, "Faster Private Set Intersection Based on OT Extension"**, 23rd
  USENIX Security Symposium, San Diego, 2014 `[link]`
  URL: https://www.usenix.org/system/files/conference/usenixsecurity14/sec14-paper-pinkas.pdf
  Read in the proceedings PDF. Section 1.1: the naive solution, in which both parties hash their
  inputs and compare the hashes, *"is insecure if the input domain is not large or does not have high
  entropy"*, because one party can hash every likely item and compare. A keyed hash whose key both
  parties hold is the same construction between those two parties. It is why the finance use case's
  intersection is an OPRF whose key only one bank holds (the RFC 9497 entry in WS3); nothing here
  implements this paper's protocol.
- **Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal and Seth, "Practical Secure
  Aggregation for Federated Learning on User-Held Data"**, arXiv:1611.04482 (2016-11-14, NIPS 2016
  workshop on Private Multi-Party Machine Learning);
  full version "Practical Secure Aggregation for Privacy Preserving Machine Learning", IACR ePrint
  2017/281, published as "Practical Secure Aggregation for Privacy-Preserving Machine Learning",
  ACM CCS 2017, pp. 1175-1191, DOI 10.1145/3133956.3133982 `[link]`
  URL: https://arxiv.org/abs/1611.04482
  Read in the arXiv text. Each pair of users agrees a secret by Diffie-Hellman key agreement and
  expands it with a PRG into a mask, uniform over $`[0, R)^k`$ with the sum taken mod $`R`$, so the
  masked inputs *"appear uniformly random to the server"* up to their sum; the lower-ordered user of
  a pair adds the expansion and the higher subtracts it (Protocol 3). Protocol 1 adds the secret
  sharing that lets the server finish a round without users who dropped, Protocol 2 a second,
  per-user mask so that this recovery cannot expose a user who did not drop, and Protocol 4 notes
  that a server relaying unauthenticated keys can mount a man-in-the-middle attack. This
  project's pairwise masks follow Protocol 3's key agreement (the QFL section), with X25519 and
  HKDF-SHA256 as its key agreement and SHAKE-256 as its PRG, and implement neither the dropout
  recovery nor key authentication. The ePrint listing was read for the full version's title, authors and its
  abstract, which states security proofs in the honest-but-curious and malicious settings; its PDF
  refused automated access. The CCS venue, pages and DOI are from the DOI's Crossref record
  (2026-11-01).
- **Attrapadung, Hanaoka, Mitsunari, Sakai, Shimizu and Teruya, "Efficient Two-level Homomorphic
  Encryption in Prime-order Bilinear Groups and A Fast Implementation in WebAssembly"**, ACM AsiaCCS
  2018, DOI 10.1145/3196494.3196552 `[link]`
  URL: https://github.com/herumi/she-wasm (the authors' library, which gives this citation)
  Read at the library's README: lifted-ElGamal encryption that *"can add two ciphertexts many times,
  and can multiply two ciphertexts once"*, compiled to WebAssembly for Node.js and browsers, under a
  BSD-3-Clause licence. The ACM page was not opened. Cited as prior art for homomorphic evaluation in
  a browser, which this project's novelty statement had said its search did not find. Two-level HE is
  not fully homomorphic, and neither is this demo's levelled CKKS without bootstrapping, so the
  comparison is like for like.

## Private inference of a trained QCNN (Round 86), read at primary source 2026-11-09

The page's trained QCNN ran on plaintext features and said so. These are the sources the private form rests on, and the search that found nothing nearer.

- **Bermejo, Braccia, Rudolph, Holmes, Cincio and Cerezo, "Quantum Convolutional Neural Networks are Effectively Classically Simulable", PRX Quantum 7, 020304 (2026), doi:10.1103/8qt9-72ts, arXiv:2408.12739v2.** Sec. IV.B and Table I: classical data in a product (tensor-product) angle encoding needs no quantum resources; the paper calls QCNNs on classical data ill-motivated. This is why the circuit here is simulated classically and described as such.
- **Schuld, "Supervised quantum machine learning models are kernel methods", arXiv:2101.11020.** A quantum model is linear in the encoded density matrix. For the product RY encoding used here that gives the form over the $`3^n`$ Pauli products: $`\langle O\rangle = \sum_P c_P\,\phi_P(x)`$, with $`c_P = 2^{-n}\operatorname{Tr}(A P)`$ and $`\phi_I = 1,\ \phi_X = \sin\theta,\ \phi_Z = \cos\theta`$ on each wire. `web/src/lib/qcnnPauli.ts` and `src/qsh/qhe/qcnn_pauli.py` compute it, and `npm run check:qcnn-ckks` holds the identity to rounding (at most 8e-15 on the shipped weights).
- **Schreiber, Eisert and Meyer, "Classical surrogates for quantum learning models", Phys. Rev. Lett. 131, 100803 (2023), arXiv:2206.11740,** and **Bowles, Ahmed and Schuld, "Better than classical? The subtle art of benchmarking quantum machine learning models", arXiv:2403.07059.** Why the QCNN's figures are read against a classical model on the same split: the raw-input logistic regression the trainer records, and logistic regression on the same Pauli products, which `scripts/train/qcnn_classical_surrogate.py` adds. The QCNN is trained on labels and has no teacher to compare with.
- **Quantum routes, and why none runs in a browser.** Broadbent and Jeffery (CRYPTO 2015, arXiv:1412.8766) need EPR registers or auxiliary states per T gate; Dulek, Schaffner and Speelman (CRYPTO 2016, arXiv:1603.09717) put one EPR gadget per T gate in a quantum evaluation key. **Ortega and Martin-Delgado, arXiv:2602.12712 (v1, 13 Feb 2026, no journal reference)** is the only server-held-QCNN private inference found: simulation only, 8 qubits on MNIST at 16x16 with amplitude encoding, plaintext accuracy 0.967, the homomorphic evaluation run on five samples (Fig. 10), two ancilla qubits returned to the client per T gate, and no runtime, communication or concrete T-count reported; its Sec. VI.1 says algorithmic guarantees for the server's model cannot be given for all weights.
- **Search negative, 2026-11-09.** Homomorphic encryption of the input to a simulated QNN or VQC, multi-party computation for QNN inference, and production use of private QNN inference: arXiv API on titles and abstracts, 17 web searches, GitHub. Nothing found. Semantic Scholar and DBLP rate-limited the search and ePrint PDFs were blocked, so absence here is not proof, and no claim of priority is made.
- **What the page says about flooding.** Li and Micciancio (EUROCRYPT 2021) show CKKS decryptions can leak the key when results return to an adversary; the client here holds the key and the result stays with it. `clientQcnnCkks.ts` still floods the slots it does not read, as `clientFHE.ts` does, so the ciphertext the client decrypts carries fresh encryption randomness.

## The finance MLP and what it is compared against (Round 86), read at primary source 2026-11-09

- **Blanusa, Cravero Baraja, Anghel, von Niederhausern, Altman, Pozidis and Atasu, "Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection", ACM ICAIF 2024 (extended version arXiv:2402.08593).** Sec. 2.2: graph-pattern features are computed after inserting a batch of edges into an in-memory graph over a time window; batches of 128 and 2,048 on the AML datasets. Table 5, HI-Small, minority-class F1 (percent) at batch 128 and 2048: XGBoost basic features 19.75, fan-in and fan-out 56.88 and 59.71, multi-hop 58.60 and 61.01, vertex statistics 63.23 and 64.77; LightGBM basic 21.30, fan-in and fan-out 50.85 and 49.73. The library is proprietary, which is why `scripts/train/finance_aml_fan.py` computes the same kind of feature in the open; 60/20/20 temporal split.
- **Altman, Blanusa, von Niederhausern, Egressy, Anghel and Atasu, NeurIPS 2023 Datasets and Benchmarks.** Table 2 on HI-Small: GIN 28.70, GIN+EU 47.73, PNA 56.77, GFP+LightGBM 62.86, GFP+XGBoost 63.23. Figure 1 shows transactions in tabular and graph form and Figure 2 the eight laundering patterns modelled.
- **Egressy, von Niederhausern, Blanusa, Altman, Wattenhofer and Atasu, AAAI 2024, arXiv:2306.11586.** Table 2 on AML Small HI: Multi-PNA+EU $`68.16 \pm 2.65`$. This is the architecture of the registry's never-trained `finance_aml.gnn` card, whose official code is `IBM/Multi-GNN` (Apache-2.0).
- **Bilgi, Mutluergil, Yildirim, Atasu and Savas, arXiv:2610.18054v2 (14 Oct 2026).** A GNN on IBM AML under CKKS, on HEonGPU with $`N = 65536`$ and bootstrapping: Table 2 plaintext minority-class F1 on Small-HI $`63.72 \pm 1.86`$, hidden dimension 16, 64 and 128 at 57.20, 63.72 and 65.09 (Table 4); 5.31 s per transaction on an RTX 5090, and the FHE and plaintext decisions agree on 99.25 percent of 2,000 seed edges. The nearest published work to private AML inference; it needs the graph and a GPU, and this page's model needs neither.
- **Literature on the MLP's shape:** QUOTIENT (Agrawal et al., CCS 2019) 2 x 124 on tabular data; Wodzinski et al., arXiv:2611.02856, a pyramid $`[h, \lfloor h/2 \rfloor]`$ on clinical tabular data. **Search negative, 2026-11-09:** no paper found runs a trained MLP on finance tabular data under a CKKS and MPC hybrid.

## Model-explanation figures (Round 86), read at primary source 2026-11-09

- **Transaction graphs.** Altman et al., NeurIPS 2023 Datasets and Benchmarks: Figure 1 shows transactions in tabular and graph form and Figure 2 the eight laundering patterns modelled (fan-out, fan-in, gather-scatter, scatter-gather, simple cycle, random, bipartite, stack). This is the published precedent for the finance page's figure. The READMEs of `IBM/AMLSim` and `IBM/AML-Data` contain no graph figure, and the SAML-D paper (Oztas et al., ICEBE 2023) is behind a bot check on the Bournemouth server and was not read, so no claim is made about it. HI-Small's pattern file is not in this project's local corpus, so the shapes are derived from the laundering-labelled edges of the test window.
- **Occlusion sensitivity.** Zeiler and Fergus, "Visualizing and Understanding Convolutional Networks", ECCV 2014, arXiv:1311.2901, Sec. 4.2: occlude portions of the input image with a grey square and monitor the classifier's output. It needs no gradients and no extra model, which suits a 3,226-parameter network in a browser tab. It is a coarse explanation of the model, not of the run under MPC, and the page says so.

## The compact transaction GNN and how to improve it (Round 86), read at primary source 2026-11-10

What the search found and what it did not. No private GNN in a browser, none under garbled circuits, and no HE and garbled-circuit hybrid for a GNN were found in three searches to November 2026; that is a search result, not a claim of being first. The nearest work is server-side: Bilgi et al. and TGHE below.

- **Bilgi, Mutluergil, Yildirim, Atasu and Savas, arXiv:2610.18054v2.** The server holds the global incidence matrices and the weights in plaintext and the banks' edge features are encrypted (Sec. 1, 4). CKKS with $`N = 65536`$ and bootstrapping on HEonGPU; 5.31 s per transaction on an RTX 5090 at batch 1 (Table 7), not counting key generation, encryption, decryption or the network (Sec. 6.2). No F1 is computed under FHE; the FHE and plaintext decisions agree on 99.25 percent of 2,000 seed edges. Table 5, the polynomial activation fine-tuned on Small-HI over 3 seeds: cross-entropy alone $`50.93 \pm 2.24`$, with distillation $`60.90 \pm 3.22`$. Table 6, one run per depth: 37.71, 53.73 and 64.70 at $`d = 1, 3, 5`$, against 63.72 for ReLU.
- **TGHE, arXiv:2606.26664 (ICWS 2026).** An edge client holds the CKKS key and the features, the cloud the weights; ego trees capped at 10 hop-1 and 5 hop-2 neighbours, quadratic activations, SEAL $`N = 16384`$, and the server learns template signatures, degree counts and plaintext edge metadata (Sec. V-D). AUC 0.7959 against 0.8460 for the plaintext GEARSage backbone on DGraphFin (Table II); 115.9 s for 10,000 queries on 16 vCPUs (Table III).
- **MEGA-GNN, arXiv:2412.00241v3 (31 Aug 2026).** Aggregating each account's parallel edges per counterparty before message passing, on AML Small HI: GIN $`46.50 \pm 4.11`$ to $`69.98 \pm 2.02`$ and PNA $`62.96 \pm 1.43`$ to $`73.65 \pm 0.36`$ (Table 7); grouping at random instead gives 61.21 and 66.89 (Table 6). Table 14: 72.92 percent of illicit edges sit on account pairs with exactly one edge, and 91.06 percent of all edges on pairs with two or more; this project reproduces 0.7292 and 0.0894 from the corpus. The split is 64/19/17 and the test graph holds every transaction (Appendix C.4).
- **SALT-GNN, arXiv:2607.10131.** On HI-Small (Table 2, 5 seeds), overall F1: PNA $`0.629 \pm 0.032`$, FraudGT $`0.679 \pm 0.018`$, SALT-GAT $`0.694 \pm 0.007`$, SALT-Trans $`0.716 \pm 0.018`$. Table 6: aggregating by the mean only 0.690, by mean and standard deviation 0.712, the full model 0.716; without degree scalers the overall score is 0.713 and the 50-99 in-degree bin falls from 0.741 to 0.616. Strictly temporal splits.
- **Egressy et al., AAAI 2024, and the official code `IBM/Multi-GNN`.** A 60-20-20 split by time (App. E.1), and the test graph is every edge of the corpus (`data_loading.py` assigns the whole edge list to the test data), so the 100 sampled neighbours per hop of a test edge can come from later in time.

**What the published figures are not comparable with.** The figures above sample neighbours from a graph that includes transactions later than the one scored, on splits of 60/20/20 or 64/19/17. This project's models see nothing past the end of the transaction's batch of 2,048 rows and at most 17 minutes past it, on a 56/14/30 split. They are reported side by side, not ranked against each other.

**What was tried here, chosen on the validation window only (three seeds each).** The pair counts that MEGA's statistic motivates, computed up to each transaction's horizon (`finance_aml_fan.pair_features`), lifted mean validation AUPRC from 0.4775 to 0.5233 for the MLP and from 0.5283 to 0.5452 for the GNN, and were adopted. MEGA's counterparty grouping of the slots, 4 distinct counterparties per direction from the 32 most recent edges with their count and total, did not change it (0.5449 against 0.5452) and was not. The neighbours' own windowed statistics lowered it (0.5072 to 0.5211 against 0.5262, at the same learning rate and weight decay) and were left out. Of the cheap next steps, SALT-GNN's aggregation was then tried on the shipped model's code path, five seeds each, endpoint scaling as shipped (2026-11-10): the sum as shipped, rerun, reached mean validation AUPRC $`0.5470 \pm 0.0057`$; adding the sum of the squared slot embeddings, which under encryption costs one CKKS square and no garbled ReLU, lowered it to $`0.5345 \pm 0.0056`$ and was not adopted; SALT-GNN's own mean and standard deviation over the filled slots, which would need a division and a square root in garbled circuits, reached $`0.5536 \pm 0.0111`$, a difference within one standard deviation of the seeds and not reflected on test (mean test F1 0.6125 against 0.6186), so the shipped sum stands. Distillation from IBM's Multi-PNA+EU, run on this project's windows (`scripts/train/ibm_multignn/`), was then tried, with $`\alpha`$ the teacher's weight in the target: mean validation AUPRC 0.5440, 0.5817, 0.5809 and 0.5802 at 0, 0.3, 0.5 and 0.7 (three seeds each), so 0.3 was adopted for the GNN and, by the same validation comparison (0.4929 to 0.5821), for the MLP. Generalised distillation in the sense of Lopez-Paz, Bottou, Schoelkopf and Vapnik, ICLR 2016, arXiv:1511.03643: the teacher's neighbourhoods include later transactions of the fit window, which reach the students only through their training targets. Three further steps were then measured on the validation window and none was adopted (2026-11-11): more neighbouring edges for the distilled GNN, 4, 6 or 8 per direction (608, 736 and 864 ReLUs), gave mean validation AUPRC 0.5810, 0.5815 and 0.5803 over three seeds, within noise, so the cheapest stays; distilling PNA into the XGBoost teacher, with the class weight given per row by the hard label because `scale_pos_weight` multiplies only rows labelled exactly 1, lowered its validation AUPRC from 0.5730 at $`\alpha = 0`$ to 0.5302 at 0.3 and 0.5417 at 0.5; and training the linear CKKS student towards PNA's blend instead of the teacher's probabilities lowered it from 0.1192 to 0.0732, and towards the mean of the two to 0.0957.

## A run's timeline and its per-party view (Round 86), read at primary source 2026-11-14

The owner's choice of that day: explain a run rather than add cryptography. The note is
`references/run-timeline-and-party-views.md`.

- **Garbled Circuit Visualizer** (stringhandler, https://stringhandler.github.io/garbled-circuit-viz, no date; read: html).
  Reset, Back, Forward, Run All and Next with a step counter and a phase bar, and a panel per party: Alice holds both
  labels of each wire, Bob the one he receives. The pattern of the replay controls and "Who holds what".
- **Mitchell and Pearce, *How Federated Learning Protects Privacy*, PAIR Explorables, November 2022**
  (https://pair.withgoogle.com/explorables/federated-learning/; read: html). Each device's update masked with shared
  random numbers and only the sum seen, which stays the same while each device's message changes.
- **Hohman, Conlen, Heer and Chau, *Communicating with Interactive Articles*, Distill, 2020**
  (https://distill.pub/2020/communicating-with-interactive-articles/; read: html). Details on demand, in "Reducing
  Cognitive Load": inspect a part while the overview stays in view.

