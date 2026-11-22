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
- **Candidate: OSV-Scanner over the SBOMs, offline.** Apache-2.0, v2.7.1 (2026-11-09). It finds CycloneDX SBOMs by
  the `*.cdx.json` names these already have, and checks them against the OSV database; online it sends package names,
  versions and hashes to OSV.dev, so only `--offline` fits. A cross-check of the three audits, not a replacement.
- **Candidate: lattice-estimator for parameter sets outside the guidelines' tables.** Its README gives LGPL-3.0-or-later
  and the repository has no LICENSE file; it runs in SageMath, so in CI only, never distributed. The guidelines'
  tables cover ternary and Gaussian secrets; anything else needs the estimator at a pinned commit.

## Layer 3: what the outputs leak

- **Membership inference: nothing here yet.** The current attacks are LiRA (Carlini et al., *Membership Inference
  Attacks From First Principles*, IEEE S&P 2022, arXiv:2112.03570; reference code in `tensorflow/privacy`,
  `research/mi_lira_2021`) and RMIA (Zarifzadeh, Liu and Shokri, *Low-Cost High-Power Membership Inference Attacks*,
  ICML 2024, arXiv:2312.03262; implemented in ML Privacy Meter, MIT, `modules/mia/attacks/rmia.py`, last pushed
  2025-04-26, its 2023 PyPI release older than the code, so a pinned commit). An audit reports the true-positive rate
  at a false-positive rate of $`10^{-3}`$ or below, per record and over several seeds, because an average hides the
  records most exposed (Aerni, Zhang and Tramer, *Evaluations of Machine Learning Privacy Defenses are Misleading*,
  CCS 2024, arXiv:2404.17399). IBM's Adversarial Robustness Toolbox (MIT) has only the older shadow-model and label-only attacks.
- **Differential privacy: accounting here, no empirical audit.** The murmur study accounts DP-SGD by Renyi DP and
  `qsh.ai.dp` composes zCDP releases. The empirical check is a canary audit: Steinke, Nasr and Jagielski, *Privacy
  Auditing with One (1) Training Run*, NeurIPS 2023 (arXiv:2305.08846), which gives a lower bound on the epsilon a run
  actually spends, to set beside the claimed one. The current accountants are PRV and PLD (Gopi, Lee and Wutschitz,
  NeurIPS 2021, arXiv:2106.02848, the Opacus default); `dp_accounting` 0.7.0 (Apache-2.0) offers PLD for a
  cross-check of this repository's own accountant.
- **Repeated queries.** Each DP release on a page is accounted per run; reruns in one tab are not composed.
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
- **Reconstruction by an informed adversary, the OPS-SAT student** (`reconstruction-opssat-student-2026-11-23.json`):
  from the published preprocessing mean, every one of the 1,594 training records within 0.1 percent (median
  relative error 7.1e-05); from the weights solved to the optimum, 98.93 percent within 10 percent and
  89.40 percent within 1; from the weights as the trainer's solver left them, 2.26 percent within 10. The
  preprocessing a CKKS client needs is a release of training statistics, and a model on private data would need them
  and its weights computed under differential privacy; this telemetry is public, so it shows the mechanism.

Still to run in the CPU phase: the BreastMNIST students (the local teacher checkpoint is to be tied to the published one
first), the federated heart model (its training is TypeScript, run through the page's own module), the canary game with
the one-run DP audit, and attribute inference against imputation.

## What each use case could add

| Use case | Membership-inference audit | DP audit | Note |
|---|---|---|---|
| Finance | the GBM, MLP, compact GNN and the two Multi-PNA+EU models, on AMLworld | the counting release (`dp_count`) | the training data is synthetic, so the figure measures memorisation, not exposure of people |
| Medical | the BreastMNIST ResNets and students; the chest X-ray FedAvg model, where a server sees every client's update | the murmur DP-SGD model; the federated heart model's noise | the one use case on records of real patients, all public |
| Genomics | none: the score's weights are published, not trained here | the discrete-Gaussian counts | |
| Telecom | the 5G-NIDD classifier (real flows, CC BY 4.0) | none | the QKD-attack model trains on generated rows |
| Satellite | the OPS-SAT and Telemanom models | none | telemetry, not personal data |
| DeFi | none | none | the liquidation check is arithmetic on balances and the hour's public prices, not a trained model |

Not adopted: CrypTen, archived since 2024; MP-SPDZ, TenSEAL, OpenFHE's Python binding, Opacus and the
estimator on the public page, all server-side; and any attack aimed at a system that is not this project's.
