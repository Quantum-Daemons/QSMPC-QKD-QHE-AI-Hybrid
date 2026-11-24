# Evaluating the privacy technologies here: a three-layer plan, checked against the repository (2026-11-22)

On the owner's instruction of 2026-11-22, a design memo of 2026-11-18 for a "PETs evaluation tool" was read as data and
each of its proposals checked against this repository and at its primary source. The memo proposes three layers:
ingest the SBOM and CBOM, audit the cryptographic parameters and component versions statically, and audit what the
outputs leak (membership inference, model inversion, the differential-privacy budget). Every tool and paper named
below was read at its repository, package index or publisher on 2026-11-22. The memo's own cautions hold here already:
public or synthetic data only (`DATA_LICENSES.md`), and an audit runs against this project's own models, never a
third party's system.

## Layer 1: SBOM and CBOM ingestion

- **Here already.** CycloneDX 1.7 SBOMs for the Python, web, MPC-wasm and threshold-wasm builds (`cyclonedx-bom`
  8.0.1, `cargo-cyclonedx` 0.5.10) and a CBOM (`scripts/generate_cbom.py`), validated in CI against the vendored 1.7.2
  schema (`scripts/validate_cyclonedx.py`).
- **Not adopted.** cdxgen's crypto mode covers Java keystores and certificates and JavaScript/TypeScript source, not
  this repository's Python or Rust, and from v13 it is the `@cdxgen/cdxgen` package. IBM's sonar-cryptography bundles
  analysers under the Sonar Source-Available License v1.0, which is not OSI-approved, and needs a SonarQube server
  (CBOMkit generates with it). OWASP Dependency-Track cannot ingest cryptographic assets yet (its issues #3145 and
  #6191 are open) and is a server, outside the client-side rule.

## Layer 2: parameters and versions

- **Here already.** `tests/test_r86_cbom_he_parameters.py` holds every HE parameter set the CBOM names to the 2018 HE
  Standard and to the guidelines of Bossuat et al., *Security Guidelines for Implementing Homomorphic Encryption*,
  IACR Communications in Cryptology 1(4), 2025 (doi:10.62056/anxra69p1). npm audit, pip-audit and cargo-deny run
  daily (`.github/workflows/audit.yml`).
- **Candidate: OSV-Scanner over the SBOMs, offline.** (Adopted on 2026-11-23 in #90, over every tracked lockfile
  rather than the SBOMs: the `osv` job and `scripts/scan_lockfiles.sh`.) Apache-2.0, v2.7.1 (2026-11-09). It finds CycloneDX SBOMs by
  the `*.cdx.json` names these already have, and checks them against the OSV database; online it sends package names,
  versions and hashes to OSV.dev, so only `--offline` fits. A cross-check of the three audits, not a replacement.
- **Candidate: lattice-estimator for parameter sets outside the guidelines' tables.** Its README gives LGPL-3.0-or-later
  and the repository has no LICENSE file; it runs in SageMath, so in CI only, never distributed. The guidelines'
  tables cover ternary and Gaussian secrets; anything else needs the estimator at a pinned commit.

## Layer 3: what the outputs leak

- **Membership inference: nothing here yet** when this was written; run on 2026-11-22 to 23 (#88): LiRA and RMIA over
  64 shadow models on OPS-SAT and the telecom forest, the BreastMNIST students and the heart federation. The current attacks are LiRA (Carlini et al., *Membership Inference
  Attacks From First Principles*, IEEE S&P 2022, arXiv:2112.03570; reference code in `tensorflow/privacy`,
  `research/mi_lira_2021`) and RMIA (Zarifzadeh, Liu and Shokri, *Low-Cost High-Power Membership Inference Attacks*,
  ICML 2024, arXiv:2312.03262; implemented in ML Privacy Meter, MIT, `modules/mia/attacks/rmia.py`, last pushed
  2025-04-26, its 2023 PyPI release older than the code, so a pinned commit). An audit reports the true-positive rate
  at a false-positive rate of $`10^{-3}`$ or below, per record and over several seeds, because an average hides the
  records most exposed (Aerni, Zhang and Tramer, *Evaluations of Machine Learning Privacy Defenses are Misleading*,
  CCS 2024, arXiv:2404.17399). IBM's Adversarial Robustness Toolbox (MIT) has only the older shadow-model and label-only attacks.
- **Differential privacy: accounting here, no empirical audit** when this was written; since #88 the one-run audit
  after Steinke, Nasr and Jagielski (`docs/measurements/dp-one-run-audit-2026-11-23.json`). The murmur study accounts DP-SGD by Renyi DP and
  `qsh.ai.dp` composes zCDP releases. The empirical check is a canary audit: Steinke, Nasr and Jagielski, *Privacy
  Auditing with One (1) Training Run*, NeurIPS 2023 (arXiv:2305.08846), which gives a lower bound on the epsilon a run
  actually spends, to set beside the claimed one. The current accountants are PRV and PLD (Gopi, Lee and Wutschitz,
  NeurIPS 2021, arXiv:2106.02848, the Opacus default); `dp_accounting` 0.7.0 (Apache-2.0) offers PLD for a
  cross-check of this repository's own accountant.
- **Repeated queries.** Each DP release on a page was accounted per run, and reruns in one tab were not composed. From
  2026-11-23 the tab keeps a ledger of its releases (`web/src/lib/dpLedger.ts`) and each DP card shows the tab's total
  beside the run's own, rho added across releases (Bun and Steinke, TCC 2016); `tests/test_r86_dp_ledger.py` holds it
  to `qsh.ai.dp` and checks that every release the page makes is recorded.
- **Inversion and the decryption surface: covered.** The pages return decrypted scores to the party that owns the
  input, and the weights are published, so extraction gains nothing. The surface that matters is a decryption, or the
  reaction to one, reaching another party: [`docs/threat-model.md`](../docs/threat-model.md) covers Li and Micciancio
  (EUROCRYPT 2021) and Bian, Choe, Kim and Lee (ePrint 2026/2490, to appear at IEEE S&P 2027, demonstrated on Lattigo
  v6.2.0, the version the threshold module uses), with the rule that the client never tells an evaluator whether a
  decryption decoded. The threshold module imports no bootstrapping package, so it holds no coefficient-to-slot keys.

## Reconstruction and inversion, read 2026-11-22

The memo also proposes model-inversion and reconstruction tests. What applies here depends on what an adversary holds:
the weights are published, and the federations aggregate updates.

- **Logistic models and small MLPs: an informed adversary reconstructs a record in closed form.** Balle, Cherubin and
  Hayes, *Reconstructing Training Data with Informed Adversaries*, IEEE S&P 2022 (doi:10.1109/SP46214.2022.9833677),
  give the attack for an adversary who knows every training record but one. Run on planted canaries, it measures what
  the published students and the federated heart model reveal. Its authors' companion code
  (`facebookresearch/bounding_data_reconstruction`) is CC BY-NC 4.0, so it is reimplemented here, not copied.
- **Gradients in a federation.** Geiping et al., *Inverting Gradients* (NeurIPS 2020, arXiv:2003.14053), after Zhu, Liu
  and Han, *Deep Leakage from Gradients* (NeurIPS 2019, arXiv:1906.08935): one example's update to a layer with a bias
  gives its input exactly. Secure aggregation leaves an honest server only the sum; a server that tampers with the
  model can still isolate records, which is outside what the pages simulate. For the chest X-ray federation, a run with
  `breaching` (MIT) or LeakPro (Apache-2.0) at a positive control and at the real local configuration would measure it,
  on the GPU.
- **DP-trained models.** Hayes, Mahloujifar and Balle, *Bounding Training Data Reconstruction in DP-SGD* (NeurIPS 2023,
  arXiv:2302.07225), and a k-candidate canary game run in one training run (Mahloujifar, Melis and Chaudhuri, *Auditing
  f-Differential Privacy in One Run*, ICML 2025, arXiv:2410.22235), reported beside the 1/k baseline and the bound that
  riskcal (MIT) computes after Kulynych et al. (NeurIPS 2025, arXiv:2507.06969).
- **Attribute inference on the tabular heart data**, reported as success over an imputation baseline, because earlier
  attacks did no better than imputing (Jayaraman and Evans, *Are Attribute Inference Attacks Just Imputation?*, CCS
  2022, doi:10.1145/3548606.3560663).
- **Not run, because the result would mislead.** Inverting a linear student returns its published weight direction,
  and inverting the binary BreastMNIST classifier returns a class prototype; the telemetry and synthetic-data models
  hold no person's record; and reconstructing from the ResNets is exploratory work whose failure would prove nothing.

Reporting follows from this: the threat model and the unit of privacy (record, patient or hospital), where the noise is
added, canary success with its confidence interval beside the baseline and any DP bound, and that an attack gives a
lower bound only. Metrics are published, never a reconstructed record or image.

## Results of the CPU phase (2026-11-23)

On the owner's choices of 2026-11-22 (the CPU phase first; GPU jobs asked one by one), each record is in
`docs/measurements/` with its script.

- **The DP-SGD accountant** (`dp-accountant-crosscheck-2026-11-23.json`): at the murmur study's settings the
  repository's Renyi-DP accountant is within 1 percent of dp_accounting's RDP and never below its PLD: noise multiplier
  1.89138 gives dp_accounting RDP 5.6114 and PLD 5.1434;
  3.45294 gives 2.6082 and 2.3904. The tighter PLD accountant would
  spend about 9 percent less epsilon for the same noise.
- **Membership inference, OPS-SAT** (`mia-opssat-2026-11-23.json`, 64 shadow models): the gradient-boosted teacher,
  LiRA AUC 0.583 and a true-positive rate of 2.55 percent at a false-positive rate of 0.1
  percent, 44 of 1698 records exposed in most targets; the CKKS student, LiRA AUC
  0.532 and 0.30 percent. RMIA agrees (AUC 0.600 and 0.525); its rate at 0.1
  percent is not resolvable, because its top score ties across more non-members than that.
- **Membership inference, the telecom forest** (`mia-telecom-qml-2026-11-23.json`): RMIA AUC 0.867, 22.64 percent
  at 1 percent; LiRA AUC 0.705. A forest memorises its training points; these are generated rows.
- **Membership inference, the BreastMNIST students** (`mia-breastmnist-students-2026-11-23.json`, 64 shadow models,
  on the CPU): the CNN student, LiRA AUC 0.611 and 0.27 percent at a false-positive rate of 0.1 percent, none of 562
  records exposed in most targets; the square-activation student, LiRA AUC 0.617 and 0.43 percent, 2 records exposed.
  RMIA agrees (AUC 0.612 and 0.631) and matches ML Privacy Meter's exactly; its rate at 0.1 percent is not resolvable,
  as for OPS-SAT. The teacher is held fixed: the local graph, its node metadata set aside, has the published pin's
  sha256, and every shadow student is distilled from its labels by the student's own recipe, on 281 records, about half
  the 546 the published students saw. So this is what a student's own training adds. Membership in the teacher's
  training set needs the teacher retrained for every shadow, a GPU job.
- **Membership inference, the heart federation** (`mia-fedheart-2026-11-23.json`, 64 shadow federations through the
  page's Python twin; the script's FedAvg, pooled and DP arms give the twin's weights exactly): FedAvg, LiRA AUC 0.504
  and 0.21 percent at a false-positive rate of 0.1 percent; the pooled model, LiRA AUC 0.541 and 0.36 percent; the
  record-level DP federation, LiRA AUC 0.548 with clipping alone, 0.544 at epsilon 8 and 0.507 at epsilon 1. RMIA
  matches ML Privacy Meter's exactly for every arm. At epsilon 1 every measured rate is under the guarantee's bound,
  exp(epsilon) times the false-positive rate plus delta: 0.0977 against 0.2718 at a false-positive rate of 10 percent.
  At epsilon 8 that bound is above 1 at these rates, so no attack could test it there.
- **Attribute inference, the heart models** (`attribute-inference-fedheart-2026-11-23.json`): asked a patient's sex
  from everything else known about them, no attack beats imputation with the same knowledge and no model (Jayaraman
  and Evans, CCS 2022). On the training rows, model inversion (Fredrikson et al.) gets 75.72 percent from the FedAvg
  model and 75.51 from the pooled one, where answering "male" every time gets 75.72; the loss attack (Yeom et al.)
  gets 35.39 and 64.61; imputation from all the features gets 76.75. For a logistic model the loss attack's guess
  depends on the diagnosis alone (the record checks this), so its lead of 2.80 points on the training rows over the
  test rows for the pooled model comes from the two splits' own mix of sex and diagnosis, not from memorisation.
- **A one-run DP audit** (`dp-one-run-audit-2026-11-23.json`): Steinke, Nasr and Jagielski's bound (NeurIPS 2023,
  arXiv:2305.08846, Corollary 5.4 and their Appendix D's search), implemented here and checked against the four values
  that appendix prints. The canaries are gradients, one per coordinate, seen by an adversary who observes every release
  and knows every other contribution, the strongest setting the paper audits. The heart federation's DP rows, 14,000
  canaries over 1,000 runs, give lower bounds of 2.99 at epsilon 8, 1.76 at epsilon 4, 0.74 at epsilon 2, 0.37 at
  epsilon 1 (each truncated, never rounded up). The murmur study's DP-SGD, 5,160 canaries, gives 1.97 against its accountant's 5.66 at noise multiplier
  1.89138, and 1.11 against 2.61 at 3.45294. Every arm is under its accountant's epsilon. Both seeded bugs are caught:
  the federation with a sixteenth of the variance gives 2.20 against the 1 its accountant states, and DP-SGD with
  every person in every step gives 6.17 against 5.66. A bound under the accountant's epsilon is consistent with the
  accounting and proves nothing more; the gap at the sampled DP-SGD is the paper's own finding for one-run audits.
- **Reconstruction by an informed adversary, the OPS-SAT student** (`reconstruction-opssat-student-2026-11-23.json`):
  from the published preprocessing mean, every one of the 1,594 training records within 0.1 percent (median
  relative error 7.1e-05); from the weights solved to the optimum, 98.93 percent within 10 percent and
  89.40 percent within 1; from the weights as the trainer's solver left them, 2.26 percent within 10. The
  preprocessing a CKKS client needs is a release of training statistics, and a model on private data would need them
  and its weights computed under differential privacy; this telemetry is public, so it shows the mechanism.

- **Reconstruction by an informed adversary, the QKD-attack student and the heart federation** (2026-11-24, the
  owner's choice that day; `reconstruction-telecom-student-2026-11-24.json`, `reconstruction-fedheart-2026-11-24.json`).
  The QKD-attack student, a multinomial logistic model fitted on the forest's soft labels (1,800 generated rows): from
  the published preprocessing mean, all 1,800 within 0.1 percent; from the weights solved to the optimum,
  87.39 percent within 0.1 and 97.94 percent within 1; from the weights as the trainer's solver left them (34
  iterations), 8.39 percent within 10. The retrained student is the published one to within 0.0003 in every weight.
  The heart federation (486 training rows over four hospitals): from each hospital's published mean, every row within
  0.1 percent; from the pooled model solved to the optimum, every row within 0.1 percent; from the pooled model as the
  page trains it (Adam, 50 epochs), 13.37 percent within 10; from the FedAvg model as the page trains it, none within
  10. The closed form needs the exact optimum, which an early-stopped or federated model does not reach; that is not a
  defence, because a release of training statistics already gives every record back, and the DP arms are audited by the
  one-run audit instead. The finance student was not run: its AMLworld features are not on this machine since the data
  drive failed on 2026-11-24, and its route through the published mean is the same as the OPS-SAT student's.

The CPU phase of 2026-11-23 left two parts of the plan above undone: the informed reconstruction had run on the
OPS-SAT student only (`docs/measurements/reconstruction-opssat-student-2026-11-23.json`), and the k-candidate canary
game with riskcal's bound was not built. On the owner's choice of 2026-11-24 the reconstruction now covers the
QKD-attack student and the heart federation (above; the finance student is not run), and the canary game is next. The
GPU jobs, on the owner's approval of 2026-11-23:

- **Membership inference, BreastMNIST's whole pipeline** (`mia-breastmnist-pipeline-2026-11-23.json`, 64 shadows): each
  shadow retrains the ResNet-18 teacher by its published recipe on its members, then distils both students from it,
  and all three are scored on the true label. The teacher leaks: LiRA AUC 0.770, 8.72 percent of members found at a
  false-positive rate of 0.1 percent, 18.10 at 1 and 40.94 at 10, and 44 of 562 records exposed in most targets.
  Through the students the pipeline leaks far less: the CNN student, LiRA AUC 0.594 and 0.47 percent at 0.1; the
  square student, LiRA AUC 0.619 and 0.36 percent. RMIA matches ML Privacy Meter's exactly for all three. A rerun of four
  shadows gave the same figures bit for bit (cuDNN deterministic). The teacher's figure is the one that matters for a
  deployment: it is the model trained on the patients' images, and a pipeline that publishes it publishes this.

- **Gradient inversion, the chest X-ray federation** (`gradient-inversion-cxr-2026-11-23.json`): Geiping et al.'s
  attack (NeurIPS 2020; cosine gradient matching, a total-variation prior, signed Adam), reimplemented, on one gradient
  step at the published weights, which first reproduce all 16 sample rows' classes of the published graph (margins
  within 0.009 of its int8 build). This is the defender's worst case, a single step on few films with the labels known,
  which a real FedAvg update after local epochs over thousands of augmented films is not. At 12,000 iterations, half
  the paper's budget for ImageNet, no film came back recognisable: median SSIM 0.215 with one film per gradient and
  0.170 with four, below a flat grey image (0.391) and below the most similar other film (0.444), and none of the 4
  films was reconstructed closer to itself than its nearest other film. That bounds this attack at this budget; it is
  not evidence that a hospital's gradient is safe to show the aggregator, which longer runs, restarts or a learned image
  prior could exploit further. In a deployment the aggregator would see only the hospitals' sum (secure aggregation),
  or a DP update; this offline study sends the updates in the clear.

The BitcoinTemporalGraph check, asked and approved, ran on 2026-11-23 and shipped in #90: entity types predicted from
a labelled node's own features and, by a GNN, from its one-hop neighbourhood too (`docs/measurements/btg-check-2026-11-23.json`).

## What each use case could add (status on 2026-11-24)

| Use case | Membership-inference audit | DP audit | Other | Note |
|---|---|---|---|---|
| Finance | not run: the GBM, MLP, compact GNN and the two Multi-PNA+EU models, on AMLworld | not run: the counting release (`dp_count`) | informed reconstruction of the student: not run, its features are not on this machine since 2026-11-24 | the training data is synthetic, so a figure measures memorisation, not exposure of people |
| Medical, 2D | done: BreastMNIST's students (teacher fixed) and the whole pipeline (teacher retrained, GPU); the heart federation's five arms | done: the one-run audit of the heart federation's noise and of the murmur DP-SGD | done: attribute inference (heart), gradient inversion (chest X-ray); informed reconstruction of the heart models (2026-11-24); the k-candidate canary game: on the owner's choice of 2026-11-24 | records of real patients, all public |
| Medical, 3D (LIDC-IDRI) | to be chosen after one teacher fit is timed (the owner's choice of 2026-11-24): 64 retrained shadows as for BreastMNIST, or fewer | none: not trained under DP | none | CT scans of real patients, public (TCIA, CC BY 3.0) |
| Genomics | none: the score's weights are published, not trained here | the discrete-Gaussian counts: not audited | | |
| Telecom | done: the QKD-attack forest (generated rows); not run: the 5G-NIDD classifier (real flows, CC BY 4.0) | none | done: informed reconstruction of the QKD-attack student (2026-11-24) | |
| Satellite | done: OPS-SAT's teacher and student; not run: Telemanom | none | done: informed reconstruction of the OPS-SAT student | telemetry, not personal data |
| DeFi | none | none | | the liquidation check is arithmetic on balances and the hour's public prices, not a trained model |

Every recorded result is shown on the Validations page from 2026-11-24 (the owner's choice that day): the card
`web/src/components/PrivacyAudits.tsx` draws `web/src/lib/privacyAudits.json`, which `scripts/export_privacy_audits.py`
copies from the records with each record's SHA-256, and `tests/test_r86_privacy_audits_card.py` holds the card to the
file and the file to the records. Metrics only, as above.

Not adopted: CrypTen, archived since 2024; MP-SPDZ, TenSEAL, OpenFHE's Python binding, Opacus and the
estimator on the public page, all server-side; and any attack aimed at a system that is not this project's.

## Re-checked on 2026-11-24

On the owner's instruction of 2026-11-24 the memo's four sections were read again against the repository and at source.

- **Still as written above.** OWASP Dependency-Track's issues #3145 and #6191 are open; cdxgen's newest release is
  v13.8.0 (2026-11-14); OSV-Scanner's is still v2.7.1; ML Privacy Meter's newest commit is still 2025-04-26 (each read
  through the GitHub API).
- **The HE parameter check, and an ISO standard on its way.** The ISO/IEC 28033 series, *Information security - Fully
  homomorphic encryption* (ISO/IEC JTC 1/SC 27), read on iso.org in a browser on 2026-11-24: Part 1, *General*, FDIS,
  stage 50.00; Part 2, BGV and BFV "and parameter selection for various security levels", DIS, stage 40.60 (close of
  voting); Part 3, CKKS with the same, stage 60.00 (under publication); Part 4, arithmetic by look-up-table evaluation,
  stage 60.00; Part 5, scheme switching, a working draft. None is published, so
  `tests/test_r86_cbom_he_parameters.py` still rests on the HE Standard (2018) and on Bossuat et al. (2025). When
  Part 3 is published, holding the CKKS sets to its tables needs the standard bought (ISO sells it; it is not
  redistributed here), which is the owner's decision. NIST's threshold call (NIST IR 8214C) lists FHE as its category
  S5, with preview talks in 2026; no submission deadline for full packages was found.
- **A cheaper membership attack, for models too costly to retrain 64 times.** Bertran et al., *Scalable Membership
  Inference Attacks via Quantile Regression* (NeurIPS 2023, arXiv:2307.03694; code `amazon-science/quantile-mia`,
  Apache-2.0, last pushed 2024-04-04): one quantile regressor fitted to the scores of non-members, no shadow models;
  its authors report it competitive with shadow-model attacks. A candidate for the 3D pipeline if retraining its teacher
  64 times is too long.
- **The repository's own settings.** On the owner's choice of 2026-11-24, secret scanning, push protection and
  Dependabot security updates were turned on (the GitHub API confirms each); Dependabot alerts and private
  vulnerability reporting were on already.
