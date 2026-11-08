# Datasets and model sources - licences, attribution, and what may be published

Four different things carry four different licences, and conflating them is how a project ends up
redistributing something it may not:

1. **Training corpora** - large, fetched on demand, never committed and never redistributed whole;
   small slices of some ship as evaluation samples and overlays (sections 2b and 4).
2. **Bundled demo samples** - small, committed under `data/` and `web/public/data/`, so tests and the
   public demo run offline.
3. **Trained model weights** - derived artefacts, published to the Hugging Face Hub *only* when the
   training corpus's licence permits it.
4. **Bundled reference PDFs** - third-party papers committed under `references/` and redistributed
   verbatim to anyone who clones. Section 8.

Bucket 4 was added in Round 45, and its absence was not a missing row. It was a **structural blind
spot**: `references/` held 11 tracked PDFs totalling 14.5 MB - the largest single redistribution
category in the repository, about 45% of everything tracked - and none of the three buckets above
covered "a third-party document we hand out unchanged". The file whose stated job is *what may be
published* had never had them in scope, so nothing was wrong in it; the question had simply never
been asked here.

The project's own code is Apache-2.0. Everything below keeps its own licence.

**Re-verified 2026-10-25 (Round 70), each at its source of record, and none has changed:**
- WDBC is CC BY 4.0 (UCI, DOI 10.24432/C5DW2B).
- OPS-SAT-AD is CC BY 4.0 (Zenodo). Read again on 2026-10-31, its two versions are different files.
  This project uses **version 1**, record 12588359 (doi:10.5281/zenodo.12588359, published
  2024-06-29), the record the dataset's own README cites: the loader fetches its `dataset.csv`
  (`qsh.models.datasets.OPSSAT_FETCH`), and the local copy matches that file's MD5,
  `c12fb83863afa4463dec07303a20bb14` (507,568 B). The concept DOI 10.5281/zenodo.12588358 now
  resolves to **version 2**, record 15108715 (created 2025-03-30), whose `dataset.csv` (507,550 B)
  and `segments.csv` (18,987,091 B, against version 1's 17,970,946 B) are different files, and which
  adds a `LICENSE` file, *"MIT License for OPS-SAT Dataset"*, "Copyright (c) 2024 KP Labs", granting
  rights to *"this software, data and associated documentation files"*. Both records' metadata state
  CC BY 4.0. Nothing here ships version 2's files, so its MIT notice does not bind them; a citation of
  the concept DOI points at version 2 and is an open change (section 2c).
- The ESA Anomaly Dataset is CC BY 3.0 IGO (Zenodo).
- AMLworld is CDLA-Sharing-1.0 (Kaggle, version 8, updated 2025-07-08).
- PaySim is CC BY-SA 4.0 (Kaggle).
- 5G-NIDD is CC BY 4.0, with open access (Fairdata Metax, modified 2025-08-14).
- gnomAD is CC0, and its attribution is requested, not required; its terms also bind users not to
  attempt to re-identify participants.
- MedMNIST BreastMNIST stays Unresolved, for the upstream reason given below.

Checked again at each source on 2026-10-30: the same licences hold. That check also found that
CSE-CIC-IDS2018's own page grants redistribution (section 3) and that the NASA SMAP/MSL sample has
no grant at any source (section 2, which records its removal). Section 2c lists what each licence
asks of a shipped file and where the file carries it.

`src/qsh/models/registry.py` is the machine-readable version of this file: each model spec carries
its dataset, licence, source URL, the date the licence text was last read, and a `publishable` flag
that `scripts/publish_hf.py` enforces. The script **refuses** to create a public repo when that flag
is false, because the Hub's public grant is *"perpetual, irrevocable, worldwide, royalty-free,
non-exclusive"* - a licensing mistake there cannot be undone.

---

## 1. Training corpora (fetched on demand; the full corpora are never redistributed)

| Vertical | Corpus | Licence | Weights publishable? | Notes |
|---|---|---|---|---|
| **Finance** | [IBM AMLworld HI-Small](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) - 5,078,345 transactions, 5,177 laundering (1 in 981); **synthetic** | **CDLA-Sharing-1.0** | **Yes** | **AMLworld is synthetic, like PaySim.** IBM's own README: *"The data is generated using a multi-agent virtual world model... Everything is synthetic"*, and the NeurIPS 2023 D&B paper is *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. It is the training corpus because it is calibrated to real transaction statistics **and** its licence permits publishing derived weights - not because it is real. There is no commercial-OK corpus of real laundering transactions. CDLA-Sharing section 1.11 defines *Results* as the outputs of Computational Use, and section 3.5 states *"This Agreement imposes no obligations or restrictions on Your Use or Publication of Results."* Trained weights are Results; the ShareAlike reciprocity attaches to redistributing the **Data**, and this repository does redistribute Data: the 8,192-row evaluation samples in the two finance Hub repos and in git (section 4), and `web/public/data/finance/real_2d.json` (section 2b). Those computed-feature rows are Enhanced Data, so CDLA-Sharing-1.0 binds them: the same unmodified agreement (sections 3.1(a) and 3.3), prominent notices on the changed files (3.1(b)), preserved credit (3.1(c)), and the agreement's text, name or a link (3.3). Section 2c says what the files still lack. Test-fixture feature rows (`fixtures.npy`) are git-ignored and excluded from every upload. |
| **Medical** | [MedMNIST v2 BreastMNIST](https://medmnist.com/) - 780 breast-ultrasound images | **CC BY 4.0 asserted by MedMNIST; the BUSI authors publish no licence** | **Yes, by decision (2026-10-26)** | Attribution: MedMNIST v2, Yang et al., *Scientific Data* 10:41 (2023), and MedMNIST v1, Yang, Shi and Ni, IEEE ISBI 2021 - MedMNIST asks for both - plus the BUSI source paper below. MedMNIST also states that its data is not for clinical use. **Because the test split ships in three public Hub repos and in git, no file in this repository may say BreastMNIST cannot be redistributed**. **Checked again 2026-10-26, at every source the BUSI authors publish.** The host page asks only for a citation in every capture from 2021 to 2026; the article's supplements hold the Baheya Hospital ethics committee's approval to publish the dataset publicly and the waiver of consent, and no licence; the authors made no repository deposit; and every Kaggle mirror carries a licence its uploader chose, none of them an author. So the data was released publicly with no terms, and the only licence anyone states is MedMNIST's own CC BY 4.0. The three Hugging Face repos stay public on that basis, by decision. Unlike CIC-IDS2017 (declined) and ImageNet (dropped), whose providers state conflicting or restrictive terms, BUSI's state none. **Checked 2026-08-10 and downgraded.** MedMNIST licenses itself CC BY 4.0, but BreastMNIST is a resize of BUSI (Al-Dhabyani et al., *Data in Brief* 28:104863, 2020), where the CC BY attaches to the ARTICLE - PMC states no separate data licence, and the dataset host asks only "If you use this dataset, please cite". A citation request is not a grant. Same ambiguity class as CIC-IDS2017 (declined) and ImageNet (dropped); unresolved rather than disproven, and recorded because three HF repos ship on it. Separately, BUSI is contaminated - see `registry.py` `DS_BREASTMNIST` for the measured train/test duplicate with contradictory labels. |
| **Medical (second model)** | [MedMNIST v2 BreastMNIST](https://medmnist.com/) | **CC BY 4.0 asserted by MedMNIST; the BUSI authors publish no licence** | **Yes, by decision (2026-10-26)** | **Corrected Round 55.** This cell read a flat "CC BY 4.0 / Yes" while saying in its own next clause that it is "the same corpus as the ResNet-18 row above" - and that row was downgraded on 2026-08-10 with the reasoning spelled out. The downgrade reached the registry and one of the four places that state this licence; the row that names itself a duplicate of the downgraded one was not among them. See the row above for the full argument: the CC BY on BUSI attaches to the ARTICLE, and a citation request is not a grant. The same corpus as the ResNet-18 row above, deliberately: `medical_fl.resnet50` exists to reproduce a published comparison, and a comparison on different data is not one. Both are scored on the identical 156-image official test split, which ships whole in each model's own HF repo. **Camelyon17-WILDS was removed in Round 20**, along with the `medical_fl.densenet` spec that named it: there was no loader, no `wilds` dependency, no training script and no artefact directory, and the card advertised a Hugging Face repo that did not exist. A planned dataset with no code path is a claim, not a plan. |
| **Genomics** | [PGS Catalog](https://www.pgscatalog.org/downloads/) scoring files [PGS000072](https://www.pgscatalog.org/score/PGS000072/) (`CC_Breast`, 187 variants, Graff et al. 2021 - **the active score**) and [PGS000001](https://www.pgscatalog.org/score/PGS000001/) (alias `PRS77_BC`, 77 SNPs, Mavaddat et al. 2015 - retired) + [1000 Genomes phase 3](https://www.internationalgenome.org/) via the Ensembl REST API | **PGS000072: CC BY 4.0** (its publication); 1000 Genomes: no licence instrument | **Weights: yes, with the credit** (no Hub repo; they ship with the page). Genotypes: no | See section 4. The score's publication is Graff RE et al., *Cross-cancer evaluation of polygenic risk scores for 16 cancer types in two large cohorts*, Nat Commun 12:970 (2021), doi:10.1038/s41467-021-21288-z, CC BY 4.0. The PGS Catalog asks to be cited as Lambert, Wingfield et al., *Enhancing the Polygenic Score Catalog with tools for score calculation and ancestry normalization*, Nat Genet 56:1989-1994 (2024), doi:10.1038/s41588-024-01937-x (the citation its REST `/rest/info` names, read 2026-10-31). |
| **Medical (full deployment's CNN path)** | [MedMNIST v2 BreastMNIST](https://medmnist.com/) | **CC BY 4.0 asserted by MedMNIST; the BUSI authors publish no licence** | **Yes, by decision (2026-10-26)** | The full deployment's medical CNN path (`src/qsh/usecases/medical_fl/`, `src/qsh/data/loaders.py`), read from `benchmark.medical_cnn` and downloaded at image build in `docker/Dockerfile.api`: the same corpus and terms as the two rows above. Until Round 86 this path defaulted to PneumoniaMNIST (CC BY 4.0; Kermany et al., *Cell* 172(5):1122-1131, 2018), a different subset from every model's, and the image baked that; nothing uses PneumoniaMNIST now. |
| **Medical (pretraining)** | **none** - the published weights are trained from random initialisation | n/a | **Yes** | `scripts/train/medical_breastmnist.py`, whose default is random initialisation at both the command line and `build_resnet()` (the function defaulted to `IMAGENET1K_V1` until Round 86, and this cell named a `--from-scratch` flag that does not exist). `--pretrained` is for local experiments: its weights carry the unresolved ImageNet redistribution question, and `publish_hf.py` refuses any model whose manifest records a pretrained source. `models/metrics.json` records `transfer_learning: none`, so no ImageNet term attaches to the published artefact. |
| **Satellite** | [ESA Anomaly Dataset](https://zenodo.org/records/12528696), Mission1 lightweight channels 41-46 | **CC BY 3.0 IGO** | **Yes** | Missions are anonymised `Mission1/2/3` - they are **not** named Sentinel anything. Multi-year telemetry, ~15.4M samples per channel; Mission1 spans 14 years by the benchmark paper's Table 1 (Mission2, 3.5). Cite the benchmark as its README now asks, Kotowski et al., *European Space Agency Dataset and Benchmark for Real-World Anomaly Detection in Spacecraft Time Series*, Journal of Data-centric Machine Learning Research (DMLR), 2026 (earlier version arXiv:2406.17826), and the dataset record (creators De Canio, Kotowski and Haskamp); Zenodo record 15237121 (v2, 2025-04-17) supersedes the v1 record 12528696 linked here, under the same CC BY 3.0 IGO. Only labels whose `anomaly_types.csv` Category is `Anomaly` count as positives; the annotated **rare *nominal* events** are not faults and scoring them as such would inflate the anomaly rate. |
| **Satellite (second model)** | [OPS-SAT-AD](https://doi.org/10.5281/zenodo.12588359), version 1 | **CC BY 4.0** | **Yes - including the data itself** | Article: Ruszczak, Kotowski, Evans and Nalepa, *The OPS-SAT benchmark for detecting anomalies in satellite telemetry*, Scientific Data 12, 710 (2025), doi:10.1038/s41597-025-05035-3. Dataset, which CC BY 4.0 asks to be credited too: Ruszczak, Kotowski, Nalepa and Evans, *OPSSAT-AD - anomaly detection dataset for satellite telemetry*, Zenodo, doi:10.5281/zenodo.12588359 (version 1, 2024), the version this project uses; the concept DOI 10.5281/zenodo.12588358 resolves to version 2, whose files differ (the list above). Code MIT at `kplabs-pl/OPS-SAT-AD`. 2,123 annotated segments across 9 channels from ESA's OPS-SAT CubeSat, 20.4% anomalous, with 19 precomputed per-segment features and the benchmark's own train/test split. Its feature table (`dataset.csv`) is 507 KB; the raw `segments.csv` is about 18 MB. It is not the only corpus whose data ships here - slices of WDBC, PaySim, AMLworld, BreastMNIST, ESA-ADB and gnomAD ship too (sections 2, 2b and 4). The published test partition ships entire inside the model's own HF repo as `eval_sample.json`, and the browser re-scores the exported graph on it. A **different task** from ESA-ADB above: segment classification, not forecasting, so the two scores are not comparable and neither is the other's baseline. |
| **Telecom (declared, NOT implemented)** | [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), [Engelen-corrected](https://intrusion-detection.distrinet-research.be/WTMC2021/) | **the provider's two pages disagree** | **No** | Re-checked 2026-08-05, and the earlier wording here was too strong. The CIC's institute-level FAQ states *"You may redistribute, republish, and mirror our datasets in any form; however, any use or redistribution of the data must include a citation"*, and places it under commercial use. The CICIDS2017 dataset page itself says only *"publicly available for researchers"* and names no licence. So terms ARE stated, they just conflict - which is an unresolved ambiguity, not a permission, and the same class of question Round 19 settled by removing the ImageNet dependency rather than guessing. The Engelen redistribution is separately hosted and states nothing. **Absent a consistent affirmative grant, neither is bundled or redistributed.** **No code path loads it:** `cicids2017_fixed_dir()` is never called; the shipped telecom model trains on synthetic telemetry. Cite Engelen, Rimmer & Joosen (WTMC 2021) *and* Sharafaldin et al. (2018). |
| **Telecom (network-layer; adopted and IMPLEMENTED)** | [5G-NIDD](https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361) - labelled attack traffic captured on a live 5G test network. **113 columns** in `BTS1_BTS2_fields_preserved.csv`, of which 66 are dropped as leaks, identifiers, constants, duplicates or all-null, leaving **47 features**. Corrected Round 55: this cell said "NOT yet implemented", "no code path loads it yet" and "52 columns", and all three were false - `datasets.py` fetches it, `scripts/train/telecom_nidd.py` trains on all 1,215,890 flows, and `telecom_ids.nidd` ships with a full metrics record. The attribution obligation itself WAS discharged and renders in `ModelPanel`; only the prose went stale | **CC BY 4.0** | **Yes - attribution required** | **This row is a withdrawn rejection, and the withdrawal is the point.** Until Round 47 this dataset sat in section 7 as NOT commercial-OK, on the reasoning that IEEE DataPort marks it subscription-required and that neither that record nor the authors' NetsLab page carries a licence string. Both of those observations were true and the conclusion drawn from them was still wrong: IEEE DataPort is a **mirror**, and the authors' repository of record is Fairdata/Metax, DOI [10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361](https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361), which returns **CC BY 4.0** with `access_type: Open` and `restriction_grounds: []`. The error was possible because the adjudication stopped at the first host that came up and read silence there as silence everywhere - and this file's own *"SILENCE IS NOT PERMISSION"* rule (section 8) says an absent grant is not a permission; it does not say an absent grant **at a mirror** is evidence about the source. Checking a licence means checking the repository of record, which for a Finnish research dataset is Fairdata, not a US mirror that fronts a paywall. **CC BY 4.0 REQUIRES ATTRIBUTION**: publishing the data, or any model trained on it, must credit the dataset authors as named in the Metax record - the same obligation the WDBC and OPS-SAT-AD rows carry, and a permissive licence does not remove it. **A code path loads it: `datasets.py` -> `scripts/train/telecom_nidd.py`**, and adopting it is not a drop-in: the shipped telecom model classifies QKD-layer attacks from QKD physical observables, 5G-NIDD classifies network-layer attacks from flow statistics, and the measured overlap is zero in both the label space and the feature space (`docs/deliverable-verification.md`). It is therefore a **separate** network-layer model rather than a replacement for the synthetic telemetry in section 3. The published comparison point on this corpus is Wang, Fok & Thing, arXiv:2504.20436 - recorded in `references/REFERENCES.md` under WS2, because section 5 requires each published card to name the baseline it is measured against. |

**ESA-ADB channels are distributed as Python pickles.** Unpickling arbitrary downloaded bytes
executes whatever the stream asks for, so `qsh.models.datasets` loads them through a **restricted
unpickler** with an explicit class allowlist, derived by parsing the opcodes with `pickletools`
(which reads without executing). That allowlist is enforced at `find_class`, so the load is safe by
construction rather than because someone inspected one file once.

## 2. Bundled demo samples (committed, offline)

| File | Licence | Source |
|---|---|---|
| `data/finance/paysim_sample.tsv`, `web/public/data/finance/paysim_sample.json` | **CC BY-SA 4.0 (ShareAlike)** | Kaggle [ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1), Lopez-Rojas et al. 2016. **PaySim is a simulator: these are synthetic transactions** generated from aggregated real mobile-money logs - the Kaggle title is *"Synthetic Financial Datasets For Fraud Detection"*. Earlier revisions of this file called them "real PaySim transactions", which was wrong. Fraud is **oversampled** for demo balance; the simulator's base rate is 0.13%. **ShareAlike applies to these bundled files**: they are Adapted Material and must be redistributed under CC BY-SA 4.0, not under the project's Apache-2.0. Since Round 84 `paysim_sample.json` carries `licence`, `licence_url` (the CC BY-SA 4.0 deed), `attribution` (Lopez-Rojas, Elmir and Axelsson, EMSS 2016) and `modifications`; its rows are unchanged. The demo's linear model reads four of the eight columns (amount, cross_border, night_tx, new_payee); device_risk and geo_mismatch are computed from PaySim's balance columns, which the dataset's description says must not be used for fraud detection because fraudulent transactions are cancelled; since Round 84 no shipped model reads them, and they stay in the TSV as its generator wrote them. Its 400 rows are the fitting rows of the demo's logistic surrogate, and the private-inference figure no longer encrypts them. |
| `data/genomics/pgs000072_breast_cancer.tsv`, `g1k_dosages_pgs000072_slice.tsv`, `pgs000001_breast_cancer.tsv`, `g1k_dosages.tsv`, `g1k_panel.tsv` | see section 4 | PGS Catalog PGS000072 (active; CC BY 4.0) and PGS000001 (retired) + 1000 Genomes phase 3 (read through the Ensembl REST API; IGSR). The scoring file is the Catalog's, unchanged, under one first line this project added that records its source, checksum, licence and citations. Cell counts are tabulated in section 4. The 2,504-individual matrices (`g1k_dosages_full.tsv` for the retired score, `g1k_dosages_pgs000072.tsv` for the active one) are **not committed**: they are gitignored, and `scripts/data/fetch_g1k_dosages.py` regenerates the active one. |
| `data/satellite/smap_msl_telemetry.tsv` and `web/public/data/satellite/telemetry.json` (NASA SMAP/MSL) | **no licence granted by any provider** (checked again at source on 2026-10-30) | **Removed from the tree in Round 84, and nothing reads either.** OPS-SAT-AD (CC BY 4.0, next row) replaced them: the browser's satellite anomaly verdict and SHAP score its segments in plaintext, and the full deployment's CKKS telemetry stage scores the same segments with the same demo regression, read by `qsh.data.loaders.load_opssat_segments` from a byte-identical copy under `data/satellite/`. The registry keeps the record as `DS_SMAP_MSL`, "no licence granted for the data", attached to no model, loaded by nothing and not publishable. Why there is no licence, which is why neither file ships (this file's own rule, section 8: *"SILENCE IS NOT PERMISSION"*): it is not a US Government work, since 17 U.S.C. 101 defines one as prepared by a federal officer or employee, JPL is a Caltech-managed FFRDC whose staff are Caltech employees, and the upstream licence file claims Caltech copyright. NASA SMAP/MSL via [Telemanom](https://github.com/khundman/telemanom), Hundman et al. 2018 (KDD), NASA JPL. The Telemanom repository is **self-contradictory about its own code**: the README shows an Apache-2.0 badge while the only licence file, `LICENSE.txt`, is a Caltech/JPL BSD-3-style copyright assertion (*"Copyright (c) 2018, California Institute of Technology... All rights reserved"*). Either way it covers the **code**; **the telemetry carries no licence statement anywhere**, and the Kaggle mirror it links to is labelled *"Data files (c) Original Authors"*. The public-domain reading rests on NASA SPD-41a (*"should be released with a Creative Commons Zero license"* - a *should*, not a grant) plus 17 U.S.C. section 105, which removes **US** copyright only and does not clear the work outside the United States. Treated as an inference, not a licence. **ESA-ADB (CC BY 3.0 IGO) is the satellite training corpus.** |
| `web/public/data/medical/breast_cancer_sample.json` | **CC BY 4.0** | Wisconsin Diagnostic Breast Cancer, [UCI ML dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), redistributed in scikit-learn. Credit as UCI asks: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. doi:10.24432/C5DW2B. Since Round 84 the file holds all 569 rows in eight of the 30 columns, z-scored, split into the 398 the bundled regression and its scaler were fitted on (`samples`) and the 171 held out from both (`heldout`, with mean radius and mean texture in source units for the overlay plane), with the class recoded to 1 = malignant. It carries `licence_url` and a `modifications` note stating these changes (section 2c). Earlier revisions of this file said "public domain"; the UCI dataset page states *"This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license"*, so **attribution is required** - being bundled inside scikit-learn does not remove it. Commercial use is permitted. |
| `web/public/data/satellite/opssat_segments.json` and its byte-identical copy `data/satellite/opssat_segments.json` (Round 84) | **CC BY 4.0** | OPS-SAT-AD, the satellite anomaly-demo sample in place of SMAP/MSL, scored in plaintext by the browser and under CKKS by the full deployment, whose API image ships the copy under `data/`: the 19 published per-segment features of 159 segments of the benchmark's test partition, z-scored, with their labels, and a demo logistic regression fitted on 370 other segments of that partition (stratified, seed 0). Built by `scripts/data/build_heldout_payloads.py` from `models/satellite_opssat/eval_sample.json`, which the repository already tracks; nothing was downloaded. Credit: Ruszczak, Kotowski, Evans and Nalepa, Scientific Data 12, 710 (2025), and the dataset record, Ruszczak, Kotowski, Nalepa and Evans, Zenodo, doi:10.5281/zenodo.12588359 (version 1, whose rows these are). The file carries the licence link and a change note; its credit names the concept DOI, 10.5281/zenodo.12588358, which resolves to version 2 (section 2c). The registry's `BUNDLED_DEMO_SAMPLES["satellite_comms"]` and `DEMO_SAMPLE_FILES` name the browser's copy. |

> **One authority, checked by a test.** Every string in this section is generated from
> `BUNDLED_DEMO_SAMPLES` in [`src/qsh/models/registry.py`](src/qsh/models/registry.py) and reaches the
> browser through `frontend_config()` -> `web/src/lib/config.defaults.json`.
> [`tests/test_demo_provenance.py`](tests/test_demo_provenance.py) asserts the shipped JSON, the baked
> config and the registry all carry the same text, and fails on any file that restates one of the
> corrected claims. This exists because the demo went on printing *"real PaySim mobile-money
> transactions"* through four separate rounds of correction: the string lived in six places and fixing
> four of them looked like fixing it.

> **Why PaySim is a demo sample and not the finance training corpus.** Whether CC BY-SA's ShareAlike
> propagates from a training set to model weights is **legally unsettled**. AMLworld's CDLA-Sharing
> answers that question explicitly in its own text (section 3.5), so it is the corpus the published finance
> model is trained on. PaySim stays as a bundled demo sample, which is plain redistribution under
> CC BY-SA with attribution - a case the licence does cover.

## 2b. Derived artefacts: 2D overlay slices and trained weights the demo serves

These are committed and served by the demo, so they are redistribution and belong in this register.
They are a different category from a demo sample: an overlay slice is a projection of a corpus, and
QCNN or student weights are a model fitted on one.

| File | Licence | Source and reasoning |
|---|---|---|
| `web/public/data/satellite/real_2d.json` | **CC BY 4.0** | OPS-SAT-AD test partition (529 segments), two of its published per-segment features. Ruszczak, Kotowski, Evans & Nalepa, *The OPS-SAT benchmark for detecting anomalies in satellite telemetry*, Scientific Data 12, 710 (2025), DOI 10.1038/s41597-025-05035-3; data Zenodo 10.5281/zenodo.12588359 (version 1, whose rows these are), credited in the file too, where the credit names the concept DOI, 10.5281/zenodo.12588358, which resolves to version 2 (section 2c). Redistributable, and it is the corpus the published OPS-SAT models train on (their train partition; these segments are held out from them). Since Round 84 it is generated from the tracked `models/satellite_opssat/eval_sample.json` (the same rows in the same order) and carries `licence_url` and `modifications`. |
| `web/public/data/medical/real_2d.json` | **CC BY 4.0** | Wisconsin Diagnostic Breast Cancer (UCI dataset 17), attribution required - same corpus and terms as `breast_cancer_sample.json` in section 2. Flagged `is_training_corpus: false` in the payload because the published medical models train on BreastMNIST: this is a **different cohort**, and the page says so. Since Round 84 the provenance no longer says BreastMNIST "cannot be redistributed" (it now says the 156-image test split is published on MedMNIST's CC BY 4.0, consistent with section 1), it states that these rows are the bundled regression's corpus with its 398 / 171 split, and it carries `licence_url` and `modifications` (two of 30 columns, rounded, class recoded). |
| `web/public/data/finance/real_2d.json` | **CDLA-Sharing-1.0 (SHARE-ALIKE)** | IBM AMLworld HI-Small held-out rows. Unlike trained weights, this slice **is** Data under CDLA-Sharing, so the reciprocity in section 3 of that licence does attach. The payload names the licence, credits Altman et al. and, since Round 84, carries the agreement's link (`licence_url`, cdla.dev/sharing-1-0) and a change notice (`modifications`), both rendered under the overlay. |
| `web/public/data/finance/student_heldout.json` (Round 84) | **CDLA-Sharing-1.0 (SHARE-ALIKE)** | The same 8,192 test-window rows as `real_2d.json`, all 32 computed graph features, rounded to 7 significant figures: what the browser encrypts `finance_aml.student` on. Copied from the tracked `models/finance_aml_gbm/eval_sample.json` by `scripts/data/build_heldout_payloads.py`; Enhanced Data under the same agreement, with its link, the credit and a change notice in the file. |
| `web/public/data/finance/student_weights.json` (Round 84) | as `models/finance_aml_student/weights.json` (section 5) | A byte-identical copy of the published student's weights, bundled so the encrypted path runs that model without a Hub fetch. Trained weights are Results under CDLA-Sharing-1.0 (sections 1.11 and 3.5), so no share-alike attaches. |
| `web/public/data/genomics/real_2d.json`, `data/genomics/gnomad_af_pgs000072.json`, `data/genomics/gnomad_af.json` | **CC0 1.0** | gnomAD genome allele frequencies, read through Ensembl's REST variation endpoint (release 116, populations `gnomADg:*`), not from any gnomAD service; Ensembl's response does not name the gnomAD release, so none is claimed here. `gnomad_af_pgs000072.json` is the **active** cache, written by `scripts/fetch_gnomad_af.py`: for each of the 183 PGS000072 variants the page scores, keyed by rsID, the frequencies of the allele the page's dosages count (resolved on the same Ensembl record), with that allele's frequency over 1000 Genomes' 503 Europeans beside them. `real_2d.json` plots 177 of the 183, gnomAD's East Asian against its non-Finnish European group: Ensembl lists no frequency of the effect allele in one of the two groups for the other 6, and every kept point's non-Finnish European frequency is within 0.1 of its 1000 Genomes European one. `gnomad_af.json` is the retired rsID-keyed cache for PGS000001's 77 variants, which nothing reads. CC0 waives attribution; the overlay credits gnomAD with the link back its terms ask for, states their one condition (that users not attempt to re-identify participants), and carries the CC0 link and a change note. It is a different thing from the page's own genotypes, 30 1000 Genomes individuals in the browser payload (section 4). |
| `models/qcnn_medical/weights.json`, `web/public/data/medical/qcnn_weights.json` | **CC BY 4.0** | Variational QCNN angles (8 wires; the 9-wire artifacts were deleted in Round 84) fitted on 8 of the 30 WDBC features. Attribution-only, so publishing weights raises no reciprocity question; the files' `provenance` names the corpus, its licence with the link, and UCI's credit. |
| `models/qcnn_finance/weights.json`, `web/public/data/finance/qcnn_weights.json` | **CC BY-SA 4.0 (ShareAlike), and this is a deliberate exception** | Fitted on the PaySim demo sample (since its Round 84 retrain, the four columns the finance linear model reads, at 4 wires; the 9-wire artifacts were deleted in Round 84), which is CC BY-SA. See the note below. The files' `provenance` names PaySim, its licence with the link, and the PaySim credit. |
| `models/qcnn_telecom/weights.json`, `web/public/data/telecom/qcnn_weights.json`, `models/qcnn_telecom_q7/weights.json`, `web/public/data/telecom/qcnn_weights_q7.json` | **Apache-2.0** (generated here; see the note below) | Fitted on synthetic QKD-attack telemetry produced by this repository's own generator (`qsh.qml_ids.attacks`; class structure per Al-Kuwari et al., arXiv:2509.14282), at 8 and 7 wires. No third-party terms attach; the files' `provenance` states the licence. |
| `web/public/data/telecom/student_weights.json` (identical to `models/telecom_ids_student/weights.json`) | **Apache-2.0** (generated here; see the note below) | The 64-parameter telecom CKKS student (8 class scores over 7 features), distilled from `telecom_ids.qml` on the same synthetic telemetry. Its `licence` field states it. |
| `web/public/data/telecom/real_2d.json` | **Apache-2.0** (generated here; see the note below) | The 600 rows held out from `generate(n_per_class=300, seed=42)` by the telecom trainer's stratified split - the rows `telecom_ids.qml` and `telecom_ids.student` were scored on and never fitted on - not the 3,200-row seed-0 corpus a QCNN trains on. Its `licence` and `licence_url` fields state the licence. |
| `web/public/data/telecom/student_heldout.json` (Round 84) | **Apache-2.0** (generated here; see the note below) | The same 600 held-out rows with all 7 features, rounded to 7 significant figures: what the browser encrypts `telecom_ids.student` on. Regenerated from the repository's own generator and checked against `models/telecom_qml_ids/eval_sample.json` by `scripts/data/build_heldout_payloads.py`. |

> **Generated data and the weights fitted on it are Apache-2.0, the repository's licence.** No
> third-party terms attach to the rows `qsh.qml_ids.attacks` writes or to the weights fitted on them,
> and the repository grants them under its own licence: the registry records the synthetic telemetry
> as `Apache-2.0`, each file above states it (a `licence` field with its link, or the weights'
> `provenance`), and the `qsh-telecom-qml-ids` Hub card names it as `license: apache-2.0`.

> **The finance QCNN is trained on PaySim, which section 2 says the published models are not - and
> that is a real tension, recorded rather than smoothed over.** The rule above exists because
> ShareAlike propagation from a training set to model weights is legally unsettled, so AMLworld,
> whose licence answers the question in its own text, is the corpus every **published** finance model
> uses. The QCNN cannot follow that rule without breaking a different one. Its four inputs are the
> demo pipeline's model columns - `amount`, `cross_border`, `night_tx`, `new_payee` - which is what the browser actually feeds it. Two of the sample's other columns, device_risk and geo_mismatch, are computed from PaySim's balance columns, which the dataset's description says must not be used for fraud detection; since its Round 84 retrain this artifact reads the four columns the linear model reads, and neither of those two.
> AMLworld reaches the browser as 32 z-scored graph features, a different space entirely. Training on
> AMLworld and serving PaySim-shaped inputs would reintroduce precisely the feature-space mismatch
> that made the first trained QCNN meaningless, where seven of eight features pinned to -1. Between
> shipping a model that cannot work and accepting a ShareAlike obligation, the second is the smaller
> problem.
>
> So the obligation is accepted rather than argued away. These weights are offered under **CC BY-SA
> 4.0**, not the project's Apache-2.0, exactly as the bundled PaySim rows in section 2 are. They are
> **deliberately absent from the Hugging Face publication path** - `models/metrics.json` has no
> `qcnn` entry and no publish script references one - so the boundary being held is that PaySim
> derivatives stay inside this repository and its demo, under the source licence, and are never
> published as standalone model artefacts.

## 2c. What the licences ask of shipped files, and where each file carries it

Read at the licence texts on 2026-10-30. Every licence here except CC0 asks for more than a name when
its data is shared: a link to the licence (or its text), the credit the creator asks for, and a note
of the changes. A shipped file travels alone, to the Hub and out of it, so each carries them itself:

| Licence | Asks for, when its data is shared | Shipped files it covers | Where the file carries it |
|---|---|---|---|
| CC BY 4.0 (section 3(a)) | credit as the creator asks, the licence's text or a link to it, and a note of any changes | WDBC sample and overlay; BreastMNIST (as MedMNIST asserts it), OPS-SAT-AD and 5G-NIDD evaluation samples; OPS-SAT-AD overlay and anomaly-demo segments; medical QCNN weights | the samples' and overlays' `licence_url`, `attribution` and `modifications` fields (UCI's order and year for WDBC, MedMNIST v2 and v1 and the BUSI article for BreastMNIST, the OPS-SAT-AD dataset record); the QCNN weights' `provenance` (link and UCI's credit) |
| CC BY-SA 4.0 (sections 3(a) and 3(b)) | as CC BY, and adaptations under CC BY-SA 4.0 or a compatible licence | PaySim sample (`paysim_sample.tsv` and `.json`), finance QCNN weights | the JSON's fields; the TSV's header lines (licence and link, credit, changes); the weights' `provenance` |
| CC BY 3.0 IGO (section 4) | the licence or its URI with every copy, the credit, and changes identified | ESA-ADB evaluation sample (128 windows) | the sample's `licence_url`, `attribution` (the DMLR 2026 paper and the Zenodo record) and `modifications` |
| CDLA-Sharing-1.0 (sections 3.1 and 3.3) | the same agreement, prominent notices on changed files, preserved credit, and the agreement's text, name or link | AMLworld evaluation samples, the finance overlay and `finance/student_heldout.json` | each file's `licence`, `licence_url`, `attribution` and `modifications` |
| CC0 1.0 | nothing; gnomAD's terms add one condition, no attempt to re-identify participants | gnomAD frequencies and the genomics overlay | the overlay, and the `_source` of both caches, the active `gnomad_af_pgs000072.json` and the retired `gnomad_af.json` |
| CC BY 4.0 (PGS000072's publication), with the PGS Catalog's EMBL-EBI Terms of Use; IGSR's disclaimer (1000 Genomes, no licence) | credit, the licence link and a note of changes; the PGS Catalog's citation; the 1000 Genomes consortium paper | PGS000072 weights (scoring file, browser payload, `weights.json`); the 1000 Genomes dosage slices | the browser payload's `attribution`, `genotypes_attribution`, `licence_url` and `modifications`; the scoring file's first line and the dosage files' headers |
| Apache-2.0 (section 4) | a copy of the licence, and the notices kept | the telecom data and weights generated here (section 2b) and the telecom QML evaluation sample | each file's `licence` and `licence_url`, or the weights' `provenance` |

The change notes record a few kinds of change: rows drawn from a held-out split or rebalanced by
class, a subset of columns, standardised, resized or rounded values, computed features, and a recoded
binary label. Each Hub card names its licence by the Hub's identifier where one exists
(`cc-by-4.0`, `cdla-sharing-1.0`, `apache-2.0`), and otherwise as `license: other` with a name and a
`license_link` (CC BY 3.0 IGO; BreastMNIST's CC BY 4.0, as MedMNIST asserts it).

OPS-SAT-AD is cited as version 1 (Zenodo record 12588359), the version whose rows these files ship;
the concept DOI 10.5281/zenodo.12588358 now resolves to version 2 (record 15108715), whose files
differ and whose MIT notice would have to travel with them if they were ever used.

## 3. Telecom has no real dataset in the demo, and says so

**The reason stated here was wrong, and it is corrected rather than quietly swapped.** Every earlier
revision opened *"No commercial-OK real network-attack corpus exists"*, resting on UNSW-NB15
requiring the authors' agreement for commercial use and the CIC-IDS family being research-use. The
first holds; the second holds only for CIC-IDS2017, whose pages disagree (section 1), because
CSE-CIC-IDS2018's own page grants redistribution, republication and mirroring in any form with a
citation and a link to its AWS registry page. And the sentence generalised them into a claim about
the whole field, which 5G-NIDD falsifies: CC BY 4.0, Open, section 1. Nothing ships from
CSE-CIC-IDS2018; it is a second commercially usable real network-attack corpus beside 5G-NIDD. The
narrower statement that survives is the real reason and always was - **no commercial-OK corpus of real QKD-layer attacks exists**, because producing one
needs a physical QKD testbed under attack. 5G-NIDD is network-layer traffic and a different problem,
not a substitute, which is why adopting it changes what this project may publish without changing
what the telecom demo runs on. The demo therefore uses **synthetic**
telemetry whose class structure follows Al-Kuwari et al., arXiv:2509.14282, **16 Sep 2025** (later in *IET Quantum Communication* 7(1), e70028) - **8 classes = 7 attack types + normal traffic**, not "eight attack scenarios". The run logs
this as synthetic; it is never presented as measured on real traffic.

## 4. Genomics - the strictest case

- **Published evaluation samples (`models/*/eval_sample.json`), committed and redistributed.** Each
  model's held-out sample ships inside that model's own Hugging Face repo at the same pinned
  revision as the weights, so a sample can never be paired with a different revision of the model it
  scores; ten of the eleven repos ship one (all but the satellite student's). They carry their source
  corpus's licence: AMLworld under CDLA-Sharing-1.0, BreastMNIST
  under CC BY 4.0 **as asserted by MedMNIST, the BUSI authors publishing no licence of their own** (see the
  medical rows in section 1 - this is the surface where that ambiguity actually binds, because the
  sample is REDISTRIBUTED), ESA-ADB under CC BY 3.0 IGO, OPS-SAT-AD and 5G-NIDD under CC BY 4.0, and
  the telecom QML sample, generated here, under Apache-2.0 (section 2b). Each file states its
  `provenance`, its `licence` and the licence's link (`licence_url`), the credit its licence asks for
  (`attribution`) and how its rows differ from the source (`modifications`); each Hub card names its
  licence by the Hub's identifier where one exists, and by name and link otherwise (section 2c). The
  full corpora are **not** redistributed and are not in this repository.

- **The encrypted-tier students are published too.** `finance_aml.student` (33 params),
  `medical_fl.student` (3226), `satellite_comms.student` (49) and `satellite_comms.opssat_student`
  (210) are the models sized for CKKS. They are distilled from the teachers on the same corpora and
  inherit the same licences, and their small `weights.json` files are also tracked in git. Which of
  them CKKS evaluates in the browser is in
  [`docs/models-and-benchmarks.md`](docs/models-and-benchmarks.md#which-pages-encrypt).

- **ImageNet-derived weights: the question was removed, not answered.** An earlier revision of this
  file analysed whether ImageNet-pretrained weights could be redistributed and concluded it could not
  be settled: ImageNet's terms bind "the Database" to non-commercial research and say nothing about a
  model trained on it, while torchvision is BSD-3 for the software and silent on the weights. Rather
  than publish on an open question under an irrevocable grant, the dependency was dropped -
  `medical_fl.resnet18` is now trained from random initialisation.

  The trade was expected to cost accuracy and did not: AUC went from 0.8876 to **0.9147**, above the
  MedMNIST v2 published ResNet-18 baseline of 0.901, while accuracy and macro F1 each fell about a
  point. The earlier text predicted "the metric will be lower than 0.878 and that is the honest
  trade"; that prediction was wrong and is recorded here rather than quietly replaced. All eleven Hub
  repos, seven teachers and four encrypted students, are public.

- **PGS Catalog is not blanket CC BY 4.0: a score's terms are its publication's.** The scoring-file
  format defines an optional `#license` line, and PGS000072's file has none. Its REST record carries
  the Catalog's pass-through to the EMBL-EBI Terms of Use and asks users to observe *"any licensing
  restrictions set by the authors"*. PGS000072's publication, Graff et al., Nat Commun 12:970
  (2021), is **CC BY 4.0** (Crossref `vor` and `tdm`, Europe PMC "cc by", read 2026-10-31); the
  article lists the score's 187 variants in its Supplementary Data 2 and weights each by the log odds
  ratio its source GWAS reported, and the Catalog's weights are those log odds ratios. The scoring
  file's `#citation` line names the
  article's bioRxiv preprint (doi:10.1101/2020.01.18.911578), for which Crossref and Europe PMC record
  no licence; Europe PMC links it as the preprint of the Nat Commun article, which is the Catalog's
  publication record (PGP000050). The bundled file's first line, written by this project, records
  this, and `scripts/train/genomics_prs.py` re-reads it at run time (`audit_licence()`) and
  **records** it. Hub publication is gated separately, by `dataset.publishable` in
  `src/qsh/models/registry.py`, which `scripts/publish_hf.py` enforces: for `genomics.prs` it is
  `True` (CC BY 4.0, with the credit), and the model has no Hub repo, so the weights ship with the
  page only.
- **1000 Genomes / IGSR is not public domain.** Verbatim from the
  [IGSR disclaimer](https://www.internationalgenome.org/IGSR_disclaimer): *"restrictions on different
  pieces of data within IGSR and rights claimed on pieces of data vary"* and *"It remains the
  responsibility of users to ensure that their exploitation of the data does not infringe any of the
  rights of third parties."* IGSR asks users to cite the consortium's paper, 1000 Genomes Project
  Consortium, *A global reference for human genetic variation*, Nature 526:68-74 (2015),
  doi:10.1038/nature15393; the genomics payload, the registry's attribution and the dosage files'
  headers carry it. The genotypes are read through the Ensembl REST API's genotype endpoint
  (`scripts/data/fetch_g1k_dosages.py`), which fills in no genotype: 183 of PGS000072's 187 variants
  have a 1000 Genomes phase 3 genotype for every one of the 2,504 individuals, and the other 4 (one
  that Ensembl does not map to GRCh38, two with no 1000 Genomes genotypes there and one with them for
  1,582 individuals) are declared `#unscored` in the dosage files and the payload and left out of
  the score.
- **What per-sample genotype data this repository does redistribute, stated plainly.** Measured
  2026-10-31, complete:

  | tracked file | individuals | variants | genotype cells |
  |---|---|---|---|
  | `data/genomics/g1k_dosages.tsv` (PRS77, retired score) | 400 | 77 | 30,800 |
  | `data/genomics/g1k_dosages_pgs000072_slice.tsv` (PGS000072, **active**) | 97 | 183 | 17,751 |
  | `data/genomics/pgs000001_prs_retired.json` (retired, **no longer served**) | 30 | 77 | 2,310 |
  | `web/public/data/genomics/pgs000072_breast_cancer.json` (**active** browser payload) | 30 | 183 | 5,490 |
  | **total tracked** | | | **56,351** |
  | of which **served to browsers** | | | **5,490** |

  All four are effect-allele dosages at published PRS loci for 1000 Genomes samples. Note the second row
  of totals: the retired browser payload is not under `web/public/`, so what the public demo actually
  serves is 5,490 cells - one payload, 30 individuals, whose 1000 Genomes sample IDs it lists.
  Tracked-in-git and served-to-the-public are different exposures.

  **The budget this section is cited for, now that it exists here.** Three source files -
  `src/qsh/data/loaders.py`, `tests/test_prs_real_genotypes.py` and
  `scripts/data/build_prs_browser_payload.py` - cite "DATA_LICENSES section 4" for a figure of about
  **30,800 genotype cells**, and until now this section stated a shape (400 x 77) and made no
  cell-budget argument at all. The authority they appealed to had not been written. It is written now,
  and the reasoning is the one those files assumed: **the unit that matters for redistribution is
  genotype cells, not rows.** 400 x 77 is the slice this section originally justified, so 30,800 cells
  is the per-file ceiling the code holds itself to. The PGS000072 CI slice is 97 individuals (the
  first 48 AFR and the first 49 EUR of the panel in sample-ID order) x 183 variants, 17,751 cells.

  This is not a privacy escalation: IGSR describes the samples as "openly consented", its consent items
  include public release of "extensive individual data from the study of the samples (but no individual
  identifiers or medical information)", and the full genotypes are downloadable from IGSR/ENA without
  access control, so a few-hundred-variant subset exposes nothing IGSR does not already publish. The
  sample IDs the payload lists (`HG00103`) are the panel's own public codes, which IGSR publishes beside
  the genotypes, not the individual identifiers that consent item rules out; they make each row
  reproducible from the source (owner's decision 2026-11-05, after reading those pages, and keeping them
  supersedes an early plan's "never publish per-sample genotypes" for this corpus). IGSR names no licence
  instrument: it says the data is "made available according to the Fort Lauderdale Agreement" and is
  "available without embargo", and that users must not infringe third-party rights. It *is*
  redistribution, so the IGSR disclaimer and attribution apply, and it is recorded here rather than
  glossed. Read 2026-11-05: <https://www.internationalgenome.org/sample_collection_principles>,
  <https://www.internationalgenome.org/IGSR_disclaimer/>,
  <https://www.internationalgenome.org/faq/do-i-need-permission-to-use-igsr-data-in-my-own-scientific-research/>.
  The **2,504-individual** matrices used for the ancestry-stratification analysis are deliberately
  **not** committed (git-ignored). `scripts/data/fetch_g1k_dosages.py` regenerates the active score's
  (and its CI slice). The published output is the aggregate per-super-population statistics, and the
  full PGS000072 matrix, 2,504 x 183 = 458,232 cells, would be about fifteen times the per-file budget
  for no demo benefit.
- **Model weights carry no genotypes.** The PRS effect weights and the aggregate metrics are tracked
  in git (section 5); nothing genomic is pushed to the Hub: `genomics.prs` has no Hub repo, and the
  genotypes' record (`DS_1000G`) sets `publishable=False`.
- **Nothing openSNP-derived is published.** The archive was CC0 and existing copies remain legally
  usable, but the maintainers deleted it in April 2025 specifically over DNA-misuse concerns. That is
  a request worth honouring regardless of what the licence permits.

## 5. Published model weights

The published models' graphs live on the Hugging Face Hub (repo ids in `src/qsh/models/registry.py`),
pinned by 40-character commit sha, and never in git (`.gitignore` excludes `*.onnx`, `*.safetensors`,
`*.pt`, `*.npy`; CI fails the build if one is staged). Small JSON weight files are a different matter and are tracked in
git: the QCNN angles and the telecom student (section 2b), the four Hub students' `weights.json`, the
finance scorecard, and the PGS000072 effect weights the page scores (183 of them in
`models/genomics_prs/weights.json` and in the browser payload
`web/public/data/genomics/pgs000072_breast_cancer.json`; all 187 in the bundled scoring file). Those
ship under their publication's **CC BY 4.0** (Graff et al., Nat Commun 12:970, 2021), with the
credit, the licence link and the change note the payload carries, and with the PGS Catalog's own
terms - the EMBL-EBI Terms of Use: cite the score's publication and use it *"in accordance with any
licensing restrictions set by the authors"* (the PGS000072 record, read 2026-10-31). The score has no
Hub repo. PGS000004 (PRS313) is not shipped: its source paper, Mavaddat et al., *Am. J. Hum. Genet.*
104(1):21-34 (2019), doi:10.1016/j.ajhg.2018.11.002, is published under
**CC BY-NC-ND 4.0** (its version of record, per Crossref, read 2026-10-31), and its weights appear
only in that article's supplementary data. Each repo's model
card carries the dataset, its licence, the measured metrics and the published baseline they are
measured against - including where the model falls short - and names its licence by the Hub's
identifier, or by name and link (section 2c).

## 5b. What the Agentic AI panel downloads

The panel's models are not this project's. The visitor's browser fetches their weights from Hugging Face at the pinned
revision in `web.tfjs_models` and `web.webllm_models`, and this project does not redistribute them. The conversions
carry no licence field of their own, so the upstream model's licence is the one given here, read from each upstream
model's card on 2026-11-05.

| Engine | Model on the panel | Weights fetched from | Upstream | Licence |
|---|---|---|---|---|
| Transformers.js | distilGPT-2 | `Xenova/distilgpt2` | `distilbert/distilgpt2` | Apache-2.0 |
| Transformers.js | Qwen2.5-0.5B-Instruct | `onnx-community/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 |
| WebLLM | SmolLM2-360M, q4f16_1 and q4f32_1 | `mlc-ai/SmolLM2-360M-Instruct-*-MLC` | `HuggingFaceTB/SmolLM2-360M-Instruct` | Apache-2.0 |
| WebLLM | Qwen2.5-0.5B, q4f16_1 | `mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 |
| WebLLM | Phi-3-mini-4k, q4f16_1 | `mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC` | `microsoft/Phi-3-mini-4k-instruct` | MIT |

**Removed: Llama-3.2-1B** (2026-11-05, the owner's choice). The panel offered it with none of what its licence asks. The
[Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) allows commercial use below 700 million monthly
active users, and section 1.b asks anyone making the materials available to supply the agreement, show "Built with Llama",
keep an attribution notice and follow the Acceptable Use Policy. The other four models cover the same sizes without
terms of their own.

**The WebLLM model libraries** (`web/public/webllm/`) are the one part this project does distribute. WebLLM's default
fetches prebuilt ones from `raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs`, a repository with no licence file, so
`scripts/build_webllm_libs.sh` builds them from MLC-LLM and TVM, both Apache-2.0, at the commits the upstream libraries
name. `MANIFEST.json` beside them records what each was built from, and `/licenses/webllm-libs.txt` ships the licences
and notices of what they link.

## 6. OSS libraries (commercial-OK)

- **node-seal** (MIT) - Microsoft SEAL -> WASM, real in-browser CKKS. **Polytune** + **Garble**
  (sine-fdn, MIT) - real in-browser garbled-circuit 2PC for the non-linear (argmax) layer, both parties
  loopback in one tab, compiled from `wasm/qsh-mpc` with 69 Rust crates whose licences are all permissive;
  their own licence files are collected at build time into `/licenses/qsh-mpc.txt`, which ships with the
  demo. `mpc-framework` + `emp-wasm-engine` + `summon-ts` filled this role until Round 65.
- **onnxruntime-web** (MIT) - runs the published industry models in the browser.
- **@huggingface/transformers** and **@mlc-ai/web-llm** (both Apache-2.0) - the Agentic AI panel's two engines. Their
  models and the libraries this project builds for one of them: section 5b.
- **MCP Python SDK** (`mcp`, MIT) - the quantum-safe MCP server; the official
  **@modelcontextprotocol/server-filesystem** for external-MCP interop, MIT and Apache-2.0 while its
  project moves from one to the other (its repository's LICENSE explains which contributions are under
  which; npm 2.1.0 declares "SEE LICENSE IN LICENSE" and its tarball has no such file). Installed from a lockfile in the
  full deployment's image (`docker/mcp-fs/`), and started with `npx` elsewhere; not in the demo.
- **PennyLane** (Apache-2.0) - variational quantum layers. The FedAvg and secure-aggregation code is
  this repository's own; **Flower is cited as prior art, not used as a dependency** (the unused `flwr`
  requirement was removed in Round 17).
- **xgboost** (Apache-2.0), **TenSEAL** (Apache-2.0), **OpenFHE** (BSD-2), **MPyC** (MIT),
  **scikit-learn** / **scipy** (BSD-3).
- **QRNG health testing** follows **NIST SP 800-90B section 4** (Repetition Count + Adaptive Proportion on
  raw pre-conditioning samples). SP 800-22 is *not* used for noise-source monitoring: SP 800-90B section 4.1
  states those tests *"will almost always fail, and thus are not useful for monitoring the noise
  source."* The entropy source is a local state-vector measurement simulation. The optional
  **ANU QRNG** API is opt-in, online-only, needs an API key read from QSH_ANU_API_KEY (the current ANU Quantum Numbers API), never used in CI, and subject to the ANU service terms.

### 6b. Conformance test vectors

| Source | Used for | Licence | Note |
|---|---|---|---|
| C2SP/CCTV ML-KEM `modulus/` | FIPS 203 section 7.2 negative vectors, `web/scripts/check_mlkem_modulus_vectors.mjs` | CC0-1.0 | **Vendored** at `web/scripts/vectors/`, 56 KB gzipped, pinned to an upstream commit recorded in `PROVENANCE.txt` alongside per-file SHA-256. Upstream `ML-KEM/README.md`: "All test vectors are made available under the terms of the CC0 1.0." CC0 waives copyright but expressly does NOT grant patent rights. |
| NIST ACVP-Server | FIPS 203 section 7.3 key-check vectors | US Government work, public domain | Not vendored for that check, and not a substitute for the above: its invalid ML-KEM-512 entries are 1216-byte keys against a correct 800, so they test length, not the modulus. |
| NIST ACVP-Server, commit `975de31` | ML-KEM-768 keyGen / encapsulation / decapsulation known-answer test run in the browser (`web/src/lib/netkeys/vectors/mlkem768_acvp.json`, 10 cases) | US Government work, public domain | **Vendored subset**, copied verbatim with per-file SHA-256 in the JSON. Added Round 58. |
| cacophony `vectors/cacophony.txt`, commit `8ee9d41` | the Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s vector the in-browser WireGuard core is checked against (`noise_ikpsk2.json`) | public domain (Unlicense, upstream `LICENSE`) | one vector of the file, with its SHA-256. Added Round 58. |
| NIST CAVP SP 800-135 `ikev2.zip` | IKEv2 prf / prf+ / SKEYSEED / key-stream cases (`ikev2_cavp.json`, 6 cases) | US Government work, public domain | Added Round 58. |
| RFC 5903 section 8.1 | the ECP-256 shared-secret example | IETF Trust; code components under Simplified BSD | one worked example. Added Round 58. |
| boringtun (Cloudflare), commit `6dcc889`, `boringtun/src/noise/handshake.rs` | the two WireGuard initial-state hash constants used as a NON-authoritative cross-check (`wireguard_initial_state.json`) | BSD-3-Clause, Copyright (c) 2019 Cloudflare, Inc. | two 32-byte values that follow from the WireGuard construction and identifier strings; attributed here although they are facts of the protocol rather than code. Added Round 58. |

The two sets above the line are test data rather than code and do not ship in the built bundle -
`web/scripts/` is outside `web/public/` and is not an import target. The Round 58 vector files DO
ship: the Validations page runs them in the visitor's tab, from a chunk loaded only when a check is
run. Full provenance for every one of them: `references/net-key-schedules.md`.

**Why the CC0 set is now redistributed here when it used to be fetched.** The check could not gate
CI while it needed the network, and the only alternative to vendoring would have been to skip on
fetch failure. For this particular check that is worse than not running it at all: it exists to
catch a library that silently accepts malformed encapsulation keys, and it found
`@noble/post-quantum` 0.2.1 accepting all 2,595 of them. A skip-on-error version would have reported
that build clean. CC0 is the one licence class under which vendoring needs no further argument.
`--live` still checks the current published set for anyone who wants it.

### 6a. Git submodules

Two upstream projects are vendored as submodules rather than installed from a package index, and
neither appeared in this document until Round 22 audited them. HEonGPU builds the GPU CKKS backend
(Round 86); hpmpc's matrix multiplication is measured on the CPU and the GPU (Round 86, below). A submodule is a dependency whose
licence and maintenance status travel with the repository, so it belongs here with the rest. This is
the one table of them; `docs/development.md` links here.

| Submodule | Upstream | Used for | Licence | Commercial use | Pinned commit | Upstream default branch | Last push |
|---|---|---|---|---|---|---|---|
| `submodules/HEonGPU` | enCRYPTON-labs/HEonGPU (moved from Alisah-Ozcan/HEonGPU) | the GPU CKKS backend, through `native/heongpu_py` | Apache-2.0 | yes | `837ffc87aa26` (v2.1.1) | `837ffc8`, the pin (read 2026-11-06) | 2026-11-02 |
| `submodules/hpmpc` | chart21/hpmpc (PIGEON) | recorded measurements of its secret-shared matrix multiplication and convolution on the CPU and the GPU (`scripts/bench_hpmpc_gpu.sh`), built in a clone outside the submodule; no stage or page runs it | MIT | yes | `155c93572d74` | `a6a4a6a`, 2026-10-23, one commit past the pin | 2026-10-23 |

Both are permissive and commercial-OK and neither is archived, read from the GitHub API rather than
assumed, since a submodule silently falling years behind is the failure mode this table exists to
catch. **HEonGPU's pin is its upstream head**; Round 86 moved it from `1928a14bb002`, 156 commits
behind, to v2.1.1, the release its CMake and the binding are written against. **hpmpc's is one commit
behind**: `a6a4a6a` removes a 47,035-line `log.txt` that carried local build paths and changes no
code. Until Round 86 this paragraph said both pins were their upstream heads, which had stopped
being true of both. Re-check both on each round; the condition to act on is an archived upstream, a
licence change, or a default branch that has not moved in a year.

**What the HEonGPU build links, and what is distributed.** `scripts/build_heongpu.sh` builds a
Python module from the submodule, its own three submodules (`thirdparty/GPU-FFT`, `GPU-NTT` and
`RNGonGPU`, and `GPU-NTT` again inside `RNGonGPU`, all `Alisah-Ozcan` repositories under
Apache-2.0), RAPIDS RMM v26.08.00 and the CCCL and NVTX it pins (fetched by HEonGPU's CMake,
Apache-2.0), NTL 11.5.1 (LGPL-2.1-or-later, downloaded by the script and checked against a SHA-256,
then linked statically), and the system GMP (LGPL-3.0-or-later or GPL-2.0-or-later, linked
dynamically). The LGPL's conditions attach to conveying the module, and this repository conveys
only the binding's source and the build script: the module stays in `native/heongpu_py/build/`,
which `.gitignore` and `.dockerignore` exclude, and no image or release carries it.

**The hazard is one level down, and it is an ABSENT licence rather than a restrictive one** (checked
2026-08-10). `hpmpc` declares two submodules of its own:

| Path | Upstream | Licence |
|---|---|---|
| `nn/PIGEON` | `chart21/flexNN` | MIT |
| `nn/Pygeon` | `chart21/pygeon` | **none - all rights reserved by default** |

A repository with no licence file grants no permissions at all, which is a stronger restriction than
GPL, not a weaker one. `chart21/pygeon` holds PyTorch export and training scripts and is not needed
by anything here.

**Neither nested submodule is populated in this checkout**, so nothing unlicensed is present today.
The exposure is a future `git submodule update --init --recursive`, which would fetch it silently -
and this project's rule is commercial-OK-only. Recorded because the failure mode is a routine command
rather than a decision, and because absent licences do not appear in any scanner: an automated
licence check looks for a restrictive SPDX identifier and finds nothing, which reads as clean.

**Do not initialise submodules recursively from this repository's root.** `--init` without
`--recursive` is sufficient for hpmpc. `scripts/bench_hpmpc_gpu.sh` needs one of the pair, `nn/PIGEON` (chart21/flexNN,
MIT; hpmpc pins `09ee6ba` of 2025-04-18, behind its `hpmpc` branch's `7a1ca4c` of 2025-09-02, and the repository was
last pushed 2026-10-06, read 2026-11-06), whose headers its benchmark includes: it initialises that one path by name,
in its own clone outside the submodule, and stops if `nn/Pygeon` is populated. It also fetches CUTLASS at a pinned tag
(v4.8.1, BSD-3-Clause, headers only) to compile hpmpc's GEMM and convolution objects. Nothing it builds is distributed.
HEonGPU is the exception, because it does not build without its own submodules:
`scripts/build_heongpu.sh` runs `--init --recursive` inside `submodules/HEonGPU` only, where every
nested repository is Apache-2.0, and never at the root, where it would also fetch `nn/Pygeon`.

## 7. Rejected as NOT commercial-OK

- **DESILO** `desilofhe` - non-commercial licence (removed Round 4).
- **IBM `snapml`** (Graph Feature Preprocessor) - **Proprietary**. This is why the finance model
  reimplements the documented graph typologies instead of using the published GFP, and why it scores
  below the published baseline. The gap is reported rather than closed by taking the dependency.
- **Bank Account Fraud** suite - CC BY-NC-ND; **IEEE-CIS** - restrictive Kaggle terms.
- **UNSW-NB15** - commercial use requires the authors' agreement.
- **NSL-KDD** - research-only, and in any case a 1999-derived corpus that no longer reflects current traffic; listed so the record of what was considered is complete.
- **Zama tfhe-rs / Concrete** - BSD 3-Clause Clear, whose patent grant is expressly withheld (read at the primary source 2026-08-11, `references/REFERENCES.md`).
- **QKDNetSim** - **GPL-2.0** (strong copyleft), so it is not listed as a commercial-OK reference
  implementation.
- **CERBSim `webgpu`** (PyPI `webgpu` 1.8.0, 2026-10-21; github.com/CERBSim/webgpu) - **LGPL-2.1-or-later**, the
  licence its `pyproject.toml` declares, with the LGPL 2.1 text as its `LICENSE` (read 2026-11-06). Declined by the
  owner on its licence (Round 40), by the reasoning QKDNetSim takes one step weaker: the demo conveys its bundle to
  every visitor, section 6 is the permissive set, and an LGPL library inside that bundle would carry its
  conditions, LGPL-2.1 section 6's among them (a user must be able to relink the work with a modified library), to
  every page load. The licence is what declines it, not the design: its documentation says scenes "serialize to
  standalone HTML files with a built-in JavaScript WebGPU engine. No Python runtime needed in the browser", which is
  the property this project's client-side rule asks for. Nor does anything need it: the pages' 3-D views draw with
  three.js (0.186.1, MIT), loaded only where drawn. *Reopens if* it is relicensed under a permissive licence, or a page
  needs what three.js cannot draw and the register records how the bundle would meet LGPL-2.1 section 6 (loading the
  library as its own file, which section 6(b) allows, is one way).

### Removed from this list: 5G-NIDD, whose rejection is withdrawn

**5G-NIDD** was listed above as NOT commercial-OK from 2026-08-05 until Round 47. That entry is
withdrawn and the dataset now appears in **section 1** as usable, CC BY 4.0, `access_type: Open`,
adjudicated at the authors' repository of record (Fairdata/Metax, DOI
10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361). The old reasoning was that secondary sources
reporting CC BY 4.0 cannot carry a commercial-OK policy while IEEE DataPort marks the dataset
subscription-required and names no licence. The principle was right and its application was
inverted: the Metax record is the primary source and the IEEE DataPort listing is the mirror, so the
adjudication rejected the dataset on the weaker of the two pieces of evidence and called that
caution. A rejection is not automatically the safe answer - it is a finding, and it needs a primary
source exactly as a permission does.

This note stays instead of the deleted bullet because a reader who acted on the rejection needs to
meet the correction where they met the error. It is the third correction of this shape in this file -
CIC-IDS2017's "states no licence at all" was too strong (section 1), and the QHE reference PDF was
swapped for its CC BY journal version rather than dropped (section 8) - and all three failed the same
way: a first source was read, no grant was found in it, and "not found here" was recorded as "not
granted anywhere".

## 8. Bundled reference PDFs (committed, redistributed verbatim)

Every **tracked** PDF under `references/` is handed to anyone who clones, so each needs a grant that
permits that. Adjudicated **per file at primary source on 2026-08-15**, not by venue and not by
inference.

This said "every PDF under `references/`" until Round 50, which is false of a working checkout: **7 of
the 11** PDFs present on disk are tracked, and the other four are gitignored and reach no clone. The
distinction matters because it is exactly what the enforcing test encodes -
`tests/test_bundled_reference_pdfs_carry_a_grant.py` derives its corpus from `git ls-files`, so a
sentence about "every PDF in the directory" describes a set the gate does not check and a clone does
not receive.

**Read arXiv licences from OAI-PMH, not from the Atom API.** `export.arxiv.org/api/query` carries no
licence field at all and reports "not present" even for a CC0 paper, so anyone adjudicating from it
reads the whole corpus as unlicensed. `oaipmh.arxiv.org/oai?verb=GetRecord&metadataPrefix=arXiv`
returns a real `<license>` element. **Two of the seven** papers kept below are licensed exactly
because that distinction was checked - the two whose "Verified at" column reads `arXiv OAI-PMH`.

This said "three of the six" until Round 49 and was wrong in both numbers: the table below has seven
rows, and only two of them are arXiv papers at all. Recounted against the table rather than adjusted
by arithmetic, because a count corrected without looking is a count that will drift again.

### Kept - an explicit grant permitting redistribution

| File (`references/`) | Identified as | Licence | Verified at |
|---|---|---|---|
| `Efficient_Simulation_of_Quantum_Secure_Multiparty_Computation.pdf` | arXiv:2501.10083v1 (2025), one author's reworking of Sutradhar and Om, Sci. Rep. 11, 2206 (2021) | **CC0 1.0** on the posting (arXiv), over an article under **CC BY 4.0** (Crossref, read 2026-11-08), which permits a reworking with credit; the credit is at `references/REFERENCES.md` and in the posting's own references | arXiv OAI-PMH `<license>` and Crossref `license` |
| `Quantum_Federated_Learning_A_Comprehensive_Survey.pdf` | arXiv:2508.15998v1 | **CC BY 4.0** | arXiv OAI-PMH `<license>` |
| `Quantum-Secure_Multiparty_Deep_Learning.pdf` | Phys. Rev. X **15**, 041056 (2025), DOI `10.1103/k8wg-qmbh` | **CC BY 4.0** | Crossref `license` (`vor`) and the PDF's own page-1 footer; the DOI is APS's new format, and the old-style `PhysRevX.15.041056` is not registered |
| `Quantum_Secure_Multiparty_Computation_based_on_Secure_Summation_and_QKD.pdf` | Research Square `10.21203/rs.3.rs-6706796/v1` | **CC BY 4.0** | Crossref `license` and the PDF's own cover page |
| `Rate-1_Quantum_Fully_Homomorphic_Encryption.pdf` | IACR ePrint **2020/1454** | **CC BY 4.0** | ePrint licence link; **byte-identical** to the live ePrint PDF, sha256 `75abdb4e...`, 357,090 B |
| `Leveraging_ASIC_AI_Chips_for_Homomorphic_Encryption.pdf` | IACR ePrint **2026/160** | **CC BY 4.0** | ePrint licence link; **byte-identical** to the live ePrint PDF, sha256 `65e6ccae...`, 3,634,311 B |
| `Quantum_Delegated_and_Federated_Learning_via_Quantum_Homomorphic_Encryption.pdf` | *Research Directions: Quantum Technologies* **3**, e3, `10.1017/qut.2025.2` | **CC BY 4.0** | Crossref `license` and the PDF's own page-1 statement |

The last row is a **swap, not a survival**. The arXiv preprint of that paper (arXiv:2409.19359) is
under arXiv's default licence and grants nothing, but the journal version of record is CC BY 4.0, so
the file was replaced rather than dropped and the reference is unchanged; the preprint's copy was
removed from the repository's history on 2026-10-31, with the four files below. Byte-identity for
the two ePrints is recorded because those two PDFs carry no self-identification at all - `Rate-1` has no
arXiv stamp, no DOI, no copyright line and no venue header anywhere in 34 pages, so matching the
live ePrint byte for byte is the only thing that ties the file to the licence being claimed for it.

### Excluded - no grant permitting redistribution

Removed from tracking in Round 45 and excluded by `.gitignore`, and removed from the repository's
whole history on 2026-10-31. GitHub held copies in the repository's read-only pull-request refs, which
a force-push does not rewrite. On 2026-11-02 this repository was created again from a rewritten history
with no pull-request refs, and the repository that holds those refs is private
([`docs/deliverable-verification.md`](docs/deliverable-verification.md), Round 86). All four remain cited
in `references/REFERENCES.md` with their URL.

| File | Licence | Why excluded |
|---|---|---|
| `EncFormer_..._over_Encrypted_Data.pdf` | arXiv `nonexclusive-distrib/1.0` | Grants **arXiv** the right to distribute and grants third parties nothing |
| `Reverse_Delegated_Training_..._Homomorphic_Encryption.pdf` | arXiv `nonexclusive-distrib/1.0` | Same |
| `Experimental_secure_multiparty_computation_..._bit_commitment.pdf` | **CC BY-NC-ND 4.0** (npj Quantum Information, Crossref `vor` + `tdm`) | NonCommercial and NoDerivatives both conflict with this repository's Apache-2.0 posture. Section 7 above already rejects a *dataset* on this exact licence, so bundling this was the inconsistency |
| `Quantum-Safe_Multi-Party_Computation_..._Gopher_Security's_Quantum_Safety_Blog.pdf` | **none found** | A vendor blog page printed to PDF. No grant in the document or on the source page |

The last row is **not** recorded as "all rights reserved". That phrase appears nowhere in the source
and would be an inference; what is true is that no grant was found, and this project's own rule -
`scripts/train/genomics_prs.py:123`, *"SILENCE IS NOT PERMISSION"* - makes an absent grant sufficient
to exclude without needing to characterise it further.

### What this replaced

`.gitignore` had excluded reference PDFs on **size** (`~35 MB total`) plus one copyright call, never
on licence, and the result was an inversion: the two most permissively licensed PDFs in the project,
both CC BY 4.0, were excluded for being ~17 MB each while a CC BY-NC-ND camera-ready shipped.

**This paragraph then said "the rules now state a licence reason per file", and that was false.**
Corrected Round 50 by reading `.gitignore` rather than this file. Four of the eight reference-PDF
exclusion rules state a licence reason; the other four (`.gitignore:28-32`) still exclude on size and
on one copyright call, exactly as before, and `.gitignore`'s own comment block says so in terms - *"the
three rules above never did [exclude on licence]: they exclude on SIZE"*. Two tracked files
contradicting each other about which rule is in force is worse than either being wrong alone, and
`.gitignore` is the one that is right.

**The inversion is also only half resolved.** The CC BY-NC-ND file was removed, which was the acute
half. But `references/Information-Theoretic_Solutions_for_Seedless_QRNG_*.pdf` is **CC BY 4.0** and
**1.8 MB**, and is still excluded by a rule whose stated reason is the other two files' combined
~35 MB. CC BY permits redistribution without compelling it, so this is not a licence violation - it is
a file excluded for a reason that does not apply to it, under a heading claiming the reason-per-file
problem was fixed.

Scope, stated accurately: section 8's "Excluded" table adjudicates four files that were **deleted** in
Round 45 and exist nowhere in the tree or its history. The four PDFs actually untracked today are
governed by the size and copyright rules above, and are not adjudicated in that table at all.

Enforced by `tests/test_bundled_reference_pdfs_carry_a_grant.py`, which derives the corpus from
`git ls-files` rather than a glob - so a newly committed PDF is in scope the moment it is tracked,
and cannot be admitted by editing a list.

## Vendored schemas

`schemas/cyclonedx/` holds the CycloneDX 1.7 JSON schema (`bom-1.7.schema.json`, `spdx.schema.json`, `jsf-0.82.schema.json`, `cryptography-defs.schema.json`) from `CycloneDX/specification` at tag 1.7.2, unchanged, under the Apache License 2.0 (the `LICENSE` file beside them). `tests/test_r86_cbom.py` validates this repository's CBOM against them, offline.
