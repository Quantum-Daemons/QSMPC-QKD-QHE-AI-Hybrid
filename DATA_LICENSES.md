# Datasets and model sources - licences, attribution, and what may be published

Four different things carry four different licences, and conflating them is how a project ends up
redistributing something it may not:

1. **Training corpora** - large, fetched on demand, never committed and never redistributed.
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

`src/qsh/models/registry.py` is the machine-readable version of this file: each model spec carries
its dataset, licence, source URL, the date the licence text was last read, and a `publishable` flag
that `scripts/publish_hf.py` enforces. The script **refuses** to create a public repo when that flag
is false, because the Hub's public grant is *"perpetual, irrevocable, worldwide, royalty-free,
non-exclusive"* - a licensing mistake there cannot be undone.

---

## 1. Training corpora (fetched on demand, never redistributed)

| Vertical | Corpus | Licence | Weights publishable? | Notes |
|---|---|---|---|---|
| **Finance** | [IBM AMLworld HI-Small](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) - 5,078,345 transactions, 5,177 laundering (1 in 981); **synthetic** | **CDLA-Sharing-1.0** | **Yes** | **AMLworld is synthetic, like PaySim.** IBM's own README: *"The data is generated using a multi-agent virtual world model... Everything is synthetic"*, and the NeurIPS 2023 D&B paper is *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. It is the training corpus because it is calibrated to real transaction statistics **and** its licence permits publishing derived weights - not because it is real. There is no commercial-OK corpus of real laundering transactions. CDLA-Sharing section 1.11 defines *Results* as the outputs of Computational Use, and section 3.5 states *"This Agreement imposes no obligations or restrictions on Your Use or Publication of Results."* Trained weights are Results; the ShareAlike reciprocity attaches to redistributing the **Data**, which we do not do. Test-fixture feature rows (`fixtures.npy`) are git-ignored and excluded from every upload for the same reason. |
| **Medical** | [MedMNIST v2 BreastMNIST](https://medmnist.com/) - 780 breast-ultrasound images | **CC BY 4.0 asserted by MedMNIST; upstream grant unverifiable** | **Unresolved** | Attribution: Yang et al., *Nature Scientific Data* 2022. **Checked 2026-08-10 and downgraded.** MedMNIST licenses itself CC BY 4.0, but BreastMNIST is a resize of BUSI (Al-Dhabyani et al., *Data in Brief* 28:104863, 2020), where the CC BY attaches to the ARTICLE - PMC states no separate data licence, and the dataset host asks only "If you use this dataset, please cite". A citation request is not a grant. Same ambiguity class as CIC-IDS2017 (declined) and ImageNet (dropped); unresolved rather than disproven, and recorded because three HF repos ship on it. Separately, BUSI is contaminated - see `registry.py` `DS_BREASTMNIST` for the measured train/test duplicate with contradictory labels. |
| **Medical (second model)** | [MedMNIST v2 BreastMNIST](https://medmnist.com/) | **CC BY 4.0 asserted by MedMNIST; upstream grant unverifiable** | **Unresolved** | **Corrected Round 55.** This cell read a flat "CC BY 4.0 / Yes" while saying in its own next clause that it is "the same corpus as the ResNet-18 row above" - and that row was downgraded on 2026-08-10 with the reasoning spelled out. The downgrade reached the registry and one of the four places that state this licence; the row that names itself a duplicate of the downgraded one was not among them. See the row above for the full argument: the CC BY on BUSI attaches to the ARTICLE, and a citation request is not a grant. The same corpus as the ResNet-18 row above, deliberately: `medical_fl.resnet50` exists to reproduce a published comparison, and a comparison on different data is not one. Both are scored on the identical 156-image official test split, which ships whole in each model's own HF repo. **Camelyon17-WILDS was removed in Round 20**, along with the `medical_fl.densenet` spec that named it: there was no loader, no `wilds` dependency, no training script and no artefact directory, and the card advertised a Hugging Face repo that did not exist. A planned dataset with no code path is a claim, not a plan. |
| **Genomics** | [PGS Catalog](https://www.pgscatalog.org/downloads/) scoring files [PGS000004](https://www.pgscatalog.org/score/PGS000004/) (alias `PRS313_BC`, 313 variants, Mavaddat et al. 2018 - **the active score**) and [PGS000001](https://www.pgscatalog.org/score/PGS000001/) (alias `PRS77_BC`, 77 SNPs, Mavaddat et al. 2015 - retired, kept for the rsID-keyed code path) + [1000 Genomes phase 3](https://www.internationalgenome.org/) via Ensembl REST | **per-score / see below** | **No** | See section 4. PGS Catalog terms require citing the **original publication**, not only the Catalog, so Mavaddat et al. is named here rather than left to a link. |
| **Medical (runtime default)** | [MedMNIST v2 **PneumoniaMNIST**](https://medmnist.com/) (derived from Kermany et al. chest X-ray) | **CC BY 4.0** | **Yes** | The *runtime* default for the medical use case (`src/qsh/data/loaders.py`, `src/qsh/config/defaults.py`, and downloaded at image build in `docker/Dockerfile.api`) - a different subset from BreastMNIST, which is what the trained model uses. Cite Yang et al. **and** the upstream Kermany et al. source. |
| **Medical (pretraining)** | **none** - the published weights are trained from random initialisation | n/a | **Yes** | `scripts/train/medical_breastmnist.py --from-scratch`. `models/metrics.json` records `transfer_learning: none`, so no ImageNet term attaches to the published artefact. Note the script's *default* is still `IMAGENET1K_V1`: re-running without the flag would reintroduce the unresolved redistribution question. |
| **Satellite** | [ESA Anomaly Dataset](https://zenodo.org/records/12528696), Mission1 lightweight channels 41-46 | **CC BY 3.0 IGO** | **Yes** | Missions are anonymised `Mission1/2/3` - they are **not** named Sentinel anything. Multi-year telemetry, ~15.4M samples per channel. (**Unresolved:** this file said 14 years and `src/qsh/models/registry.py` said 17.5 for the same corpus. The ESA-ADB Zenodo record, arXiv:2406.17826 abstract and the kplabs-pl/ESA-ADB README do not state the span, so neither figure was re-verified; the specific number is withheld here rather than asserted.) Only labels whose `anomaly_types.csv` Category is `Anomaly` count as positives; the annotated **rare *nominal* events** are not faults and scoring them as such would inflate the anomaly rate. |
| **Satellite (second model)** | [OPS-SAT-AD](https://doi.org/10.5281/zenodo.12588358) | **CC BY 4.0** | **Yes - including the data itself** | Ruszczak et al., *The OPS-SAT benchmark for detecting anomalies in satellite telemetry*, Scientific Data 12 (2025), `s41597-025-05035-3`; code MIT at `kplabs-pl/OPS-SAT-AD`. 2,123 annotated segments across 9 channels from ESA's OPS-SAT CubeSat, 20.4% anomalous, with 19 precomputed per-segment features and the benchmark's own train/test split. **The only corpus in this project whose DATA may be redistributed, not just derived weights** - CC BY 4.0 and 507 KB, against ESA-ADB's CC BY 3.0 IGO and 3.6 GB. The published test partition ships entire inside the model's own HF repo as `eval_sample.json`, so the browser reproduces the figure the dataset's 30 published baselines are quoted against. A **different task** from ESA-ADB above: segment classification, not forecasting, so the two scores are not comparable and neither is the other's baseline. |
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
| `data/finance/paysim_sample.tsv`, `web/public/data/finance/paysim_sample.json` | **CC BY-SA 4.0 (ShareAlike)** | Kaggle [ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1), Lopez-Rojas et al. 2016. **PaySim is a simulator: these are synthetic transactions** generated from aggregated real mobile-money logs - the Kaggle title is *"Synthetic Financial Datasets For Fraud Detection"*. Earlier revisions of this file called them "real PaySim transactions", which was wrong. Fraud is **oversampled** for demo balance; the simulator's base rate is 0.13%. **ShareAlike applies to these bundled files**: they are Adapted Material and must be redistributed under CC BY-SA 4.0, not under the project's Apache-2.0. |
| `data/genomics/pgs000004_breast_cancer.tsv`, `g1k_dosages_prs313_slice.tsv`, `pgs000001_breast_cancer.tsv`, `g1k_dosages.tsv`, `g1k_panel.tsv` | see section 4 | PGS Catalog PGS000004 (active) and PGS000001 (retired) + 1000 Genomes phase 3 (Ensembl / IGSR). Cell counts are tabulated in section 4. `g1k_dosages_full.tsv` was named in this row until Round 33 and is **not committed**: it is gitignored and regenerated locally by `scripts/data/fetch_g1k_dosages_bypos.py`. Section 4 was rewritten in Round 32, is labelled measured and complete, and correctly excludes it - so this row, in a section headed "committed", contradicted the authoritative table in its own file. |
| `data/satellite/smap_msl_telemetry.tsv`, `web/public/data/satellite/telemetry.json` | **no licence granted by any provider** `[UNVERIFIED - reasoned inference]` | NASA SMAP/MSL via [Telemanom](https://github.com/khundman/telemanom), Hundman et al. 2018 (KDD), NASA JPL. The Telemanom repository is **self-contradictory about its own code**: the README shows an Apache-2.0 badge while the only licence file, `LICENSE.txt`, is a Caltech/JPL BSD-3-style copyright assertion (*"Copyright (c) 2018, California Institute of Technology... All rights reserved"*). Either way it covers the **code**; **the telemetry carries no licence statement anywhere**, and the Kaggle mirror it links to is labelled *"Data files (c) Original Authors"*. The public-domain reading rests on NASA SPD-41a (*"should be released with a Creative Commons Zero license"* - a *should*, not a grant) plus 17 U.S.C. section 105, which removes **US** copyright only and does not clear the work outside the United States. Treated as an inference, not a licence. This is why **ESA-ADB (CC BY 3.0 IGO) is the satellite training corpus** and SMAP/MSL is retained only as a bundled demo sample. |
| `web/public/data/medical/breast_cancer_sample.json` | **CC BY 4.0** | Wisconsin Diagnostic Breast Cancer, [UCI ML dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (Wolberg, Street & Mangasarian), redistributed in scikit-learn. Earlier revisions of this file said "public domain"; the UCI dataset page states *"This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license"*, so **attribution is required** - being bundled inside scikit-learn does not remove it. Commercial use is permitted. |

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

## 2b. Derived artefacts: 2D overlay slices and trained QCNN weights

These are committed and served by the demo, so they are redistribution and belong in this register.
They were added after section 2 was written and are a different category from a demo sample: an
overlay slice is a projection of a corpus, and QCNN weights are a model fitted on one.

| File | Licence | Source and reasoning |
|---|---|---|
| `web/public/data/satellite/real_2d.json` | **CC BY 4.0** | OPS-SAT-AD held-out split, two of its published per-segment features. Ruszczak, Kotowski, Evans & Nalepa, *The OPS-SAT benchmark for detecting anomalies in satellite telemetry*, Scientific Data 12, 710 (2025), DOI 10.1038/s41597-025-05035-3; data Zenodo 10.5281/zenodo.12588358 (concept DOI). Redistributable, and it is the corpus the shipped model trains on. |
| `web/public/data/medical/real_2d.json` | **CC BY 4.0** | Wisconsin Diagnostic Breast Cancer (UCI dataset 17), attribution required - same corpus and terms as `breast_cancer_sample.json` in section 2. Flagged `is_training_corpus: false` in the payload because the published medical model trains on BreastMNIST: this is a **different cohort**, drawn to give the page a redistributable overlay, and the page says so. |
| `web/public/data/finance/real_2d.json` | **CDLA-Sharing-1.0 (SHARE-ALIKE)** | IBM AMLworld HI-Small held-out rows. Unlike trained weights, this slice **is** Data under CDLA-Sharing, so the reciprocity in section 3 of that licence does attach and the payload states the terms in its own `licence` field. |
| `web/public/data/genomics/real_2d.json`, `data/genomics/gnomad_af_prs313.json`, `data/genomics/gnomad_af.json` | **CC0 1.0** | gnomAD v4 allele frequencies. `gnomad_af_prs313.json` is the **active** cache: 307 of the 313 PGS000004 variants, looked up by GRCh38 coordinate because that score publishes no rsIDs, and it is what `real_2d.json` is generated from. `gnomad_af.json` is the retired rsID-keyed cache for PGS000001's 77 variants (Broad Institute). CC0 waives attribution; the payload credits gnomAD anyway because a citation request is a norm even where it is not a condition. Chosen over redistributing 1000 Genomes genotypes, which section 4 does not permit. |
| `models/qcnn_medical/weights.json`, `web/public/data/medical/qcnn_weights.json` | **CC BY 4.0** | Variational QCNN angles fitted on 8 of the 30 WDBC features. Attribution-only, so publishing weights raises no reciprocity question. |
| `models/qcnn_finance/weights.json`, `web/public/data/finance/qcnn_weights.json` | **CC BY-SA 4.0 (ShareAlike), and this is a deliberate exception** | Fitted on `data/finance/paysim_sample.tsv`, which is CC BY-SA. See the note below. |

> **The finance QCNN is trained on PaySim, which section 2 says the published models are not - and
> that is a real tension, recorded rather than smoothed over.** The rule above exists because
> ShareAlike propagation from a training set to model weights is legally unsettled, so AMLworld,
> whose licence answers the question in its own text, is the corpus every **published** finance model
> uses. The QCNN cannot follow that rule without breaking a different one. Its eight inputs are the
> demo pipeline's feature schema - `amount`, `velocity_24h`, `cross_border`, `night_tx`, `new_payee`,
> `amount_zscore`, `device_risk`, `geo_mismatch` - which is what the browser actually feeds it.
> AMLworld reaches the browser as 32 z-scored graph features, a different space entirely. Training on
> AMLworld and serving PaySim-shaped inputs would reintroduce precisely the feature-space mismatch
> that made the first trained QCNN meaningless, where seven of eight features pinned to -1 (checklist
> QC4). Between shipping a model that cannot work and accepting a ShareAlike obligation, the second
> is the smaller problem.
>
> So the obligation is accepted rather than argued away. These weights are offered under **CC BY-SA
> 4.0**, not the project's Apache-2.0, exactly as the bundled PaySim rows in section 2 are. They are
> **deliberately absent from the Hugging Face publication path** - `models/metrics.json` has no
> `qcnn` entry and no publish script references one - so the boundary being held is that PaySim
> derivatives stay inside this repository and its demo, under the source licence, and are never
> published as standalone model artefacts.

## 3. Telecom has no real dataset in the demo, and says so

**The reason stated here was wrong, and it is corrected rather than quietly swapped.** Every earlier
revision opened *"No commercial-OK real network-attack corpus exists"*, resting on UNSW-NB15
requiring the authors' agreement for commercial use and the CIC-IDS family being research-use. Those
two facts hold, but the sentence generalised them into a claim about the whole field, and 5G-NIDD
falsifies it: CC BY 4.0, Open, section 1. The narrower statement that survives is the real reason and
always was - **no commercial-OK corpus of real QKD-layer attacks exists**, because producing one
needs a physical QKD testbed under attack. 5G-NIDD is network-layer traffic and a different problem,
not a substitute, which is why adopting it changes what this project may publish without changing
what the telecom demo runs on. The demo therefore uses **synthetic**
telemetry whose class structure follows Al-Kuwari et al., arXiv:2509.14282, **16 Sep 2025** (later in *IET Quantum Communication* 7(1), e70028) - **8 classes = 7 attack types + normal traffic**, not "eight attack scenarios". The run logs
this as synthetic; it is never presented as measured on real traffic.

## 4. Genomics - the strictest case

- **Published evaluation samples (`models/*/eval_sample.json`), committed and redistributed.** Each
  model's held-out sample ships inside that model's own Hugging Face repo at the same pinned
  revision as the weights, so a sample can never be paired with a different revision of the model it
  scores. They carry their source corpus's licence: AMLworld under CDLA-Sharing-1.0, BreastMNIST
  under CC BY 4.0 **as asserted by MedMNIST, with the upstream BUSI grant unverifiable** (see the
  medical rows in section 2 - this is the surface where that ambiguity actually binds, because the
  sample is REDISTRIBUTED), ESA-ADB under CC BY 3.0 IGO. Each file states its own provenance in a
  `provenance` field. The full corpora are **not** redistributed and are not in this repository.

- **The encrypted-tier students are published too** (Round 19). `finance_aml.student` (33 params),
  `medical_fl.student` (3226) and `satellite_comms.student` (49) are the models that actually run under
  CKKS. They are distilled from the teachers on the same corpora and inherit the same licences;
  before Round 19 they were fit in memory, measured and discarded, so the one model a reader most
  needs in order to check the encryption claim was the only one they could not obtain.

- **ImageNet-derived weights: the question was removed, not answered.** An earlier revision of this
  file analysed whether ImageNet-pretrained weights could be redistributed and concluded it could not
  be settled: ImageNet's terms bind "the Database" to non-commercial research and say nothing about a
  model trained on it, while torchvision is BSD-3 for the software and silent on the weights. Rather
  than publish on an open question under an irrevocable grant, the dependency was dropped -
  `medical_fl.resnet18` is now trained from random initialisation.

  The trade was expected to cost accuracy and did not: AUC went from 0.8876 to **0.9147**, above the
  MedMNIST v2 published ResNet-18 baseline of 0.901, while accuracy and macro F1 each fell about a
  point. The earlier text predicted "the metric will be lower than 0.878 and that is the honest
  trade"; that prediction was wrong and is recorded here rather than quietly replaced. All three
  teacher repos and all three encrypted-student repos are public.

- **PGS Catalog is not blanket CC BY 4.0.** Round 46 correction: scoring files carry **no**
  `#license` header - PGS000004 downloaded today has 14 `#` lines and none is a licence. The
  licence is in the REST record only, as a pass-through to
  the EMBL-EBI Terms of Use, and **individual scores can be more restrictive** - the PGS Catalog
  documentation shows a live CC BY-NC-ND 4.0 example. `scripts/train/genomics_prs.py` re-reads that
  header at run time (`audit_licence()`) and **records** it. Publication is gated separately, by
  `dataset.publishable` in `src/qsh/models/registry.py`, which `scripts/publish_hf.py` enforces --
  the recorded audit itself blocks nothing. For `genomics.prs` the registry sets `publishable=False`.
- **1000 Genomes / IGSR is not public domain.** Verbatim from the
  [IGSR disclaimer](https://www.internationalgenome.org/IGSR_disclaimer): *"restrictions on different
  pieces of data within IGSR and rights claimed on pieces of data vary"* and *"It remains the
  responsibility of users to ensure that their exploitation of the data does not infringe any of the
  rights of third parties."*
- **What per-sample genotype data this repository does redistribute, stated plainly.** An earlier
  revision of this file claimed "no per-sample genotype is ever published", which was false. It then
  became false a second way: Round 31 moved the genomics use case from PGS000001 (77 variants) to
  PGS000004 / PRS313 (313 variants) and added two tracked files without updating this section, so the
  register described roughly half of what was actually being shipped. Measured 2026-08-11, complete:

  | tracked file | individuals | variants | genotype cells |
  |---|---|---|---|
  | `data/genomics/g1k_dosages.tsv` (PRS77, retired score) | 400 | 77 | 30,800 |
  | `data/genomics/g1k_dosages_prs313_slice.tsv` (PRS313, **active**) | 97 | 313 | 30,361 |
  | `data/genomics/pgs000001_prs_retired.json` (retired, **no longer served**) | 30 | 77 | 2,310 |
  | `web/public/data/genomics/pgs000004_prs313.json` (**active** browser payload) | 30 | 313 | 9,390 |
  | **total tracked** | | | **72,861** |
  | of which **served to browsers** | | | **9,390** |

  All four are effect-allele dosages at published PRS loci for 1000 Genomes samples. Note the second row
  of totals: the retired browser payload was moved out of `web/public/` in Round 32 because nothing in
  the browser read it any more, so what the public demo actually serves is 9,390 cells - one payload,
  30 individuals. Tracked-in-git and served-to-the-public are different exposures and this table used to
  conflate them.

  **The budget this section is cited for, now that it exists here.** Three source files -
  `src/qsh/data/loaders.py`, `tests/test_prs_real_genotypes.py` and
  `scripts/data/build_prs_browser_payload.py` - cite "DATA_LICENSES section 4" for a figure of about
  **30,800 genotype cells**, and until now this section stated a shape (400 x 77) and made no
  cell-budget argument at all. The authority they appealed to had not been written. It is written now,
  and the reasoning is the one those files assumed: **the unit that matters for redistribution is
  genotype cells, not rows.** 400 x 77 is the slice this section originally justified, so 30,800 cells
  is the per-file ceiling the code holds itself to - which is why the PRS313 CI slice is 97 x 313
  (30,361) rather than 400 x 313 (123,200, four times the budget for the same argument).

  This is not a privacy escalation - 1000 Genomes participants consented to open, unrestricted data
  release and the full genotypes are downloadable from IGSR/ENA without access control, so a
  few-hundred-variant subset exposes nothing IGSR does not already publish. But it *is* redistribution,
  so the IGSR disclaimer and attribution apply, and it is recorded here rather than glossed.
  The **2,504-individual** matrices used for the ancestry-stratification analysis are deliberately
  **not** committed (git-ignored). Regenerate with `scripts/data/fetch_g1k_dosages.py` for the
  rsID-keyed PRS77 panel and `scripts/data/fetch_g1k_dosages_bypos.py` for PRS313 - PGS000004 publishes
  no rsIDs, so the position-keyed fetcher is the only one that can build its panel. The published output
  is the aggregate per-super-population statistics, and shipping a full matrix would expand the
  redistribution twenty-four-fold for no demo benefit.
- **Model weights carry no genotypes.** Only the published effect weights and aggregate metrics are
  pushed to the Hub.
- **Nothing openSNP-derived is published.** The archive was CC0 and existing copies remain legally
  usable, but the maintainers deleted it in April 2025 specifically over DNA-misuse concerns. That is
  a request worth honouring regardless of what the licence permits.

## 5. Published model weights

Weights live on the Hugging Face Hub under `Daemons-Q`, pinned by 40-character commit sha, and never
in git (`.gitignore` excludes `*.onnx`, `*.safetensors`, `*.pt`, `*.npy`; CI fails the build if one is
staged). Each repo's model card carries the dataset, its licence, the measured metrics and the
published baseline they are measured against - including where the model falls short.

## 6. OSS libraries (commercial-OK)

- **node-seal** (MIT) - Microsoft SEAL -> WASM, real in-browser CKKS. **mpc-framework** +
  **emp-wasm-engine** + **summon-ts** (MIT) - real in-browser garbled-circuit 2PC for the non-linear
  (argmax) layer, both parties loopback in one tab.
- **onnxruntime-web** (MIT) - runs the published industry models in the browser.
- **MCP Python SDK** (`mcp`, MIT) - the quantum-safe MCP server; the official
  **@modelcontextprotocol/server-filesystem** (MIT) for external-MCP interop.
- **PennyLane** (Apache-2.0) - variational quantum layers. The FedAvg and secure-aggregation code is
  this repository's own; **Flower is cited as prior art, not used as a dependency** (the unused `flwr`
  requirement was removed in Round 17).
- **xgboost** (Apache-2.0), **TenSEAL** (Apache-2.0), **OpenFHE** (BSD-2), **MPyC** (MIT),
  **scikit-learn** / **scipy** (BSD-3).
- **QRNG health testing** follows **NIST SP 800-90B section 4** (Repetition Count + Adaptive Proportion on
  raw pre-conditioning samples). SP 800-22 is *not* used for noise-source monitoring: SP 800-90B section 4.1
  states those tests *"will almost always fail, and thus are not useful for monitoring the noise
  source."* The entropy source is a local state-vector measurement simulation. The optional
  **ANU QRNG** API is opt-in, online-only, never used in CI, and subject to the ANU service terms.

### 6b. Conformance test vectors

| Source | Used for | Licence | Note |
|---|---|---|---|
| C2SP/CCTV ML-KEM `modulus/` | FIPS 203 section 7.2 negative vectors, `web/scripts/check_mlkem_modulus_vectors.mjs` | CC0-1.0 | **Vendored** at `web/scripts/vectors/`, 56 KB gzipped, pinned to an upstream commit recorded in `PROVENANCE.txt` alongside per-file SHA-256. Upstream `ML-KEM/README.md`: "All test vectors are made available under the terms of the CC0 1.0." CC0 waives copyright but expressly does NOT grant patent rights. |
| NIST ACVP-Server | FIPS 203 section 7.3 key-check vectors | US Government work, public domain | Not vendored, and not a substitute for the above: its invalid ML-KEM-512 entries are 1216-byte keys against a correct 800, so they test length, not the modulus. |

Both are test data rather than code, and neither ships in the built bundle — `web/scripts/` is
outside `web/public/` and is not an import target, so Vite neither bundles nor serves it.

**Why the CC0 set is now redistributed here when it used to be fetched.** The check could not gate
CI while it needed the network, and the only alternative to vendoring would have been to skip on
fetch failure. For this particular check that is worse than not running it at all: it exists to
catch a library that silently accepts malformed encapsulation keys, and it found
`@noble/post-quantum` 0.2.1 accepting all 2,595 of them. A skip-on-error version would have reported
that build clean. CC0 is the one licence class under which vendoring needs no further argument.
`--live` still checks the current published set for anyone who wants it.

### 6a. Git submodules

Two upstream projects are vendored as submodules rather than installed from a package index, and
neither appeared in this document until Round 22 audited them. A submodule is a dependency whose
licence and maintenance status travel with the repository, so it belongs here with the rest.

| Submodule | Upstream | Licence | Pinned commit | Upstream default branch | Last push |
|---|---|---|---|---|---|
| `submodules/HEonGPU` | Alisah-Ozcan/HEonGPU | Apache-2.0 | `1928a14bb002` | same commit - at HEAD | 2026-05-11 |
| `submodules/hpmpc` | chart21/hpmpc | MIT | `155c93572d74` | same commit - at HEAD | 2026-07-25 |

Both are permissive and commercial-OK, neither is archived, and **both pins are the current head of
their upstream default branch** - checked against the GitHub API rather than assumed, since a
submodule silently falling years behind is the failure mode this table exists to catch.

One nuance worth recording rather than smoothing: hpmpc's default branch has not moved since
2026-03-02, while the repository's last push is 2026-07-25. The activity is on other branches or
tags, so "actively maintained" and "the branch we track is moving" are different statements and only
the first is true here. Re-check both on each round; the condition to act on is an archived upstream,
a licence change, or a default branch that has not moved in a year.

## 7. Rejected as NOT commercial-OK

- **DESILO** `desilofhe` - non-commercial licence (removed Round 4).
- **IBM `snapml`** (Graph Feature Preprocessor) - **Proprietary**. This is why the finance model
  reimplements the documented graph typologies instead of using the published GFP, and why it scores
  below the published baseline. The gap is reported rather than closed by taking the dependency.
- **Bank Account Fraud** suite - CC BY-NC-ND; **IEEE-CIS** - restrictive Kaggle terms.
- **UNSW-NB15** - commercial use requires the authors' agreement.
- **NSL-KDD** - research-only, and in any case a 1999-derived corpus that no longer reflects current traffic; listed so the record of what was considered is complete.
- **Zama tfhe-rs / Concrete** - BSD-3-Clear, research only; commercial use needs a paid patent licence.
- **QKDNetSim** - **GPL-2.0** (strong copyleft), so it is not listed as a commercial-OK reference
  implementation.

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
| `Efficient_Simulation_of_Quantum_Secure_Multiparty_Computation.pdf` | arXiv:2501.10083v1 | **CC0 1.0** | arXiv OAI-PMH `<license>` |
| `Quantum_Federated_Learning_A_Comprehensive_Survey.pdf` | arXiv:2508.15998v1 | **CC BY 4.0** | arXiv OAI-PMH `<license>` |
| `Quantum-Secure_Multiparty_Deep_Learning.pdf` | Phys. Rev. X **15**, 041056 (2025) | **CC BY 4.0** | Crossref `license` (`vor`) and the PDF's own page-1 footer |
| `Quantum_Secure_Multiparty_Computation_based_on_Secure_Summation_and_QKD.pdf` | Research Square `10.21203/rs.3.rs-6706796/v1` | **CC BY 4.0** | Crossref `license` and the PDF's own cover page |
| `Rate-1_Quantum_Fully_Homomorphic_Encryption.pdf` | IACR ePrint **2020/1454** | **CC BY 4.0** | ePrint licence link; **byte-identical** to the live ePrint PDF, sha256 `75abdb4e...`, 357,090 B |
| `Leveraging_ASIC_AI_Chips_for_Homomorphic_Encryption.pdf` | IACR ePrint **2026/160** | **CC BY 4.0** | ePrint licence link; **byte-identical** to the live ePrint PDF, sha256 `65e6ccae...`, 3,634,311 B |
| `Quantum_Delegated_and_Federated_Learning_via_Quantum_Homomorphic_Encryption.pdf` | *Research Directions: Quantum Technologies* **3**, e3, `10.1017/qut.2025.2` | **CC BY 4.0** | Crossref `license` and the PDF's own page-1 statement |

The last row is a **swap, not a survival**. The arXiv preprint of that paper (arXiv:2409.19359) is
under arXiv's default licence and grants nothing, but the journal version of record is CC BY 4.0, so
the file was replaced rather than dropped and the reference is unchanged. Byte-identity for the two
ePrints is recorded because those two PDFs carry no self-identification at all - `Rate-1` has no
arXiv stamp, no DOI, no copyright line and no venue header anywhere in 34 pages, so matching the
live ePrint byte for byte is the only thing that ties the file to the licence being claimed for it.

### Excluded - no grant permitting redistribution

Removed from tracking in Round 45 and excluded by `.gitignore`. All four remain cited in
`references/REFERENCES.md` with their URL; only the bundled copy is gone.

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
Round 45 and exist nowhere in the tree. The four PDFs actually untracked today are governed by the
size and copyright rules above, and are not adjudicated in that table at all.

Enforced by `tests/test_bundled_reference_pdfs_carry_a_grant.py`, which derives the corpus from
`git ls-files` rather than a glob - so a newly committed PDF is in scope the moment it is tracked,
and cannot be admitted by editing a list.

## Vendored submodules, and a transitive licence that is absent rather than restrictive

Checked 2026-08-10. Both direct submodules are commercially permissive and both are at their
upstream tip:

| Submodule | Licence | Upstream tip | Commercial use |
|---|---|---|---|
| `submodules/HEonGPU` | Apache-2.0 | 2026-05-11 | yes |
| `submodules/hpmpc` | MIT | 2026-03-02 | yes |

**The hazard is one level down, and it is an ABSENT licence rather than a restrictive one.** `hpmpc`
declares two submodules of its own:

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

**Do not initialise submodules recursively in this repository.** `--init` without `--recursive` is
sufficient for both direct submodules, and neither builds anything that needs the nested pair.
