# Datasets and model sources — licences, attribution, and what may be published

Three different things carry three different licences, and conflating them is how a project ends up
redistributing something it may not:

1. **Training corpora** — large, fetched on demand, never committed and never redistributed.
2. **Bundled demo samples** — small, committed under `data/` and `web/public/data/`, so tests and the
   public demo run offline.
3. **Trained model weights** — derived artefacts, published to the Hugging Face Hub *only* when the
   training corpus's licence permits it.

The project's own code is Apache-2.0. Everything below keeps its own licence.

`src/qsh/models/registry.py` is the machine-readable version of this file: each model spec carries
its dataset, licence, source URL, the date the licence text was last read, and a `publishable` flag
that `scripts/publish_hf.py` enforces. The script **refuses** to create a public repo when that flag
is false, because the Hub's public grant is *"perpetual, irrevocable, worldwide, royalty-free,
non-exclusive"* — a licensing mistake there cannot be undone.

---

## 1. Training corpora (fetched on demand, never redistributed)

| Vertical | Corpus | Licence | Weights publishable? | Notes |
|---|---|---|---|---|
| **Finance** | [IBM AMLworld HI-Small](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — 5,078,345 transactions, 5,177 laundering (1 in 981); **synthetic** | **CDLA-Sharing-1.0** | **Yes** | **AMLworld is synthetic, like PaySim.** IBM's own README: *"The data is generated using a multi-agent virtual world model... Everything is synthetic"*, and the NeurIPS 2023 D&B paper is *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. It is the training corpus because it is calibrated to real transaction statistics **and** its licence permits publishing derived weights — not because it is real. There is no commercial-OK corpus of real laundering transactions. CDLA-Sharing §1.11 defines *Results* as the outputs of Computational Use, and §3.5 states *"This Agreement imposes no obligations or restrictions on Your Use or Publication of Results."* Trained weights are Results; the ShareAlike reciprocity attaches to redistributing the **Data**, which we do not do. Test-fixture feature rows (`fixtures.npy`) are git-ignored and excluded from every upload for the same reason. |
| **Medical** | [MedMNIST v2 BreastMNIST](https://medmnist.com/) — 780 breast-ultrasound images | **CC BY 4.0** | **Yes** | Attribution: Yang et al., *Nature Scientific Data* 2022. |
| **Medical (planned)** | [Camelyon17-WILDS](https://wilds.stanford.edu/datasets/) | **CC0 1.0** | **Yes** | The official split trains on 3 hospitals and holds 2 out as OOD val/test. Federating across all 5 would destroy the OOD test set, so the federation spans the 3 training centres only. **Planned only:** there is no loader, no `wilds` dependency and no DenseNet training script; the registry's `medical_fl.densenet` entry has no artefact directory. |
| **Genomics** | [PGS Catalog](https://www.pgscatalog.org/downloads/) scoring files + [1000 Genomes phase 3](https://www.internationalgenome.org/) via Ensembl REST | **per-score / see below** | **No** | See §4. |
| **Medical (runtime default)** | [MedMNIST v2 **PneumoniaMNIST**](https://medmnist.com/) (derived from Kermany et al. chest X-ray) | **CC BY 4.0** | **Yes** | The *runtime* default for the medical use case (`src/qsh/data/loaders.py`, `src/qsh/config/defaults.py`, and downloaded at image build in `docker/Dockerfile.api`) — a different subset from BreastMNIST, which is what the trained model uses. Cite Yang et al. **and** the upstream Kermany et al. source. |
| **Medical (pretraining)** | **ImageNet-1k** via `torchvision` `ResNet18_Weights.IMAGENET1K_V1` | See the analysis below | **NO — publication withheld** | `scripts/train/medical_breastmnist.py` initialises from ImageNet-pretrained weights, so the published `qsh-medical-breastmnist-resnet18` weights are ImageNet-derived. ImageNet's own terms are **non-commercial research** for the images; the torchvision weights are BSD-3 but were trained on that corpus. This project has **not** completed a terms analysis for redistributing the derived weights — treat as an open compliance question, not a cleared one. |
| **Satellite** | [ESA Anomaly Dataset](https://zenodo.org/records/12528696), Mission1 lightweight channels 41–46 | **CC BY 3.0 IGO** | **Yes** | Missions are anonymised `Mission1/2/3` — they are **not** named Sentinel anything. Multi-year telemetry, ~15.4M samples per channel. (**Unresolved:** this file said 14 years and `src/qsh/models/registry.py` said 17.5 for the same corpus. The ESA-ADB Zenodo record, arXiv:2406.17826 abstract and the kplabs-pl/ESA-ADB README do not state the span, so neither figure was re-verified; the specific number is withheld here rather than asserted.) Only labels whose `anomaly_types.csv` Category is `Anomaly` count as positives; the annotated **rare *nominal* events** are not faults and scoring them as such would inflate the anomaly rate. |
| **Telecom (declared, NOT implemented)** | [CIC-IDS2017, Engelen-corrected](https://intrusion-detection.distrinet-research.be/WTMC2021/) | **no licence stated by either provider** | **No** | CIC-IDS2017 states no licence — only *"publicly available for researchers"* plus a citation request — and the Engelen redistribution states no terms either. **Absent an affirmative grant, neither is safe to bundle or redistribute.** **No code path loads it:** `cicids2017_fixed_dir()` in `src/qsh/models/datasets.py` is never called and the fetch block is commented out; the shipped telecom model is trained on synthetic telemetry. Listed here because the registry declares it. Cite Engelen, Rimmer & Joosen (WTMC 2021 @ IEEE S&P Workshops) *and* Sharafaldin et al. (2018). |

**ESA-ADB channels are distributed as Python pickles.** Unpickling arbitrary downloaded bytes
executes whatever the stream asks for, so `qsh.models.datasets` loads them through a **restricted
unpickler** with an explicit class allowlist, derived by parsing the opcodes with `pickletools`
(which reads without executing). That allowlist is enforced at `find_class`, so the load is safe by
construction rather than because someone inspected one file once.

## 2. Bundled demo samples (committed, offline)

| File | Licence | Source |
|---|---|---|
| `data/finance/paysim_sample.tsv`, `web/public/data/finance/paysim_sample.json` | **CC BY-SA 4.0 (ShareAlike)** | Kaggle [ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1), Lopez-Rojas et al. 2016. **PaySim is a simulator: these are synthetic transactions** generated from aggregated real mobile-money logs — the Kaggle title is *"Synthetic Financial Datasets For Fraud Detection"*. Earlier revisions of this file called them "real PaySim transactions", which was wrong. Fraud is **oversampled** for demo balance; the simulator's base rate is 0.13%. **ShareAlike applies to these bundled files**: they are Adapted Material and must be redistributed under CC BY-SA 4.0, not under the project's Apache-2.0. |
| `data/genomics/pgs000001_breast_cancer.tsv`, `g1k_dosages.tsv`, `g1k_dosages_full.tsv`, `g1k_panel.tsv` | see §4 | PGS Catalog PGS000001 + 1000 Genomes phase 3 (Ensembl / IGSR). |
| `data/satellite/smap_msl_telemetry.tsv`, `web/public/data/satellite/telemetry.json` | **no licence granted by any provider** `[UNVERIFIED — reasoned inference]` | NASA SMAP/MSL via [Telemanom](https://github.com/khundman/telemanom), Hundman et al. 2018 (KDD), NASA JPL. The Telemanom repository is **self-contradictory about its own code**: the README shows an Apache-2.0 badge while the only licence file, `LICENSE.txt`, is a Caltech/JPL BSD-3-style copyright assertion (*"Copyright (c) 2018, California Institute of Technology ... All rights reserved"*). Either way it covers the **code**; **the telemetry carries no licence statement anywhere**, and the Kaggle mirror it links to is labelled *"Data files © Original Authors"*. The public-domain reading rests on NASA SPD-41a (*"should be released with a Creative Commons Zero license"* — a *should*, not a grant) plus 17 U.S.C. §105, which removes **US** copyright only and does not clear the work outside the United States. Treated as an inference, not a licence. This is why **ESA-ADB (CC BY 3.0 IGO) is the satellite training corpus** and SMAP/MSL is retained only as a bundled demo sample. |
| `web/public/data/medical/breast_cancer_sample.json` | **CC BY 4.0** | Wisconsin Diagnostic Breast Cancer, [UCI ML dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (Wolberg, Street & Mangasarian), redistributed in scikit-learn. Earlier revisions of this file said "public domain"; the UCI dataset page states *"This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license"*, so **attribution is required** — being bundled inside scikit-learn does not remove it. Commercial use is permitted. |

> **One authority, checked by a test.** Every string in this section is generated from
> `BUNDLED_DEMO_SAMPLES` in [`src/qsh/models/registry.py`](src/qsh/models/registry.py) and reaches the
> browser through `frontend_config()` → `web/src/lib/config.defaults.json`.
> [`tests/test_demo_provenance.py`](tests/test_demo_provenance.py) asserts the shipped JSON, the baked
> config and the registry all carry the same text, and fails on any file that restates one of the
> corrected claims. This exists because the demo went on printing *"real PaySim mobile-money
> transactions"* through four separate rounds of correction: the string lived in six places and fixing
> four of them looked like fixing it.

> **Why PaySim is a demo sample and not the finance training corpus.** Whether CC BY-SA's ShareAlike
> propagates from a training set to model weights is **legally unsettled**. AMLworld's CDLA-Sharing
> answers that question explicitly in its own text (§3.5), so it is the corpus the published finance
> model is trained on. PaySim stays as a bundled demo sample, which is plain redistribution under
> CC BY-SA with attribution — a case the licence does cover.

## 3. Telecom has no real dataset in the demo, and says so

No commercial-OK real network-attack corpus exists: UNSW-NB15 requires the authors' agreement for
commercial use, and the CIC-IDS family is research-use. The demo therefore uses **synthetic**
telemetry whose class structure follows Al-Kuwari et al., arXiv:2509.14282, **16 Sep 2025** (later in *IET Quantum Communication* 7(1), e70028) — **8 classes = 7 attack types + normal traffic**, not "eight attack scenarios". The run logs
this as synthetic; it is never presented as measured on real traffic.

## 4. Genomics — the strictest case

- **ImageNet-derived weights: analysed 2026-08-04, and NOT cleared.** `scripts/train/medical_breastmnist.py`
  initialises from `ResNet18_Weights.IMAGENET1K_V1`, so the trained `qsh-medical-breastmnist-resnet18`
  weights are ImageNet-derived. Two primary sources were read, and between them they leave the
  question open rather than answering it:
  - **ImageNet's own terms of access** say *"Researcher shall use the Database only for
    non-commercial research and educational purposes."* That binds use of **the Database**. The
    terms say nothing either way about redistributing a *model trained on it*.
  - **torchvision** is **BSD-3-Clause**, but that licence covers the *software*. The repository
    makes no separate statement about the licence of the pretrained weights it distributes.

  So there is no affirmative grant covering public redistribution of ImageNet-derived weights, and
  no authoritative statement that it is prohibited either. The Hugging Face public grant is
  *"perpetual, irrevocable, worldwide, royalty-free, non-exclusive"*, so an error here cannot be
  undone. **Round 18 therefore published `finance_aml.gbm` (AMLworld, CDLA-Sharing-1.0 §3.5 puts
  trained weights outside the sharing obligation as "Results") and `satellite_comms.telemanom`
  (ESA-ADB, CC BY 3.0 IGO) publicly, and held the medical model back.**

  The clean way to remove the question rather than accept it: retrain BreastMNIST **from scratch**,
  without ImageNet initialisation. The corpus is small (546 training images) so this is cheap; the
  metric will be lower than 0.878 and that is the honest trade. `models/metrics.json` records
  `transfer_learning: "ImageNet1K_V1 ResNet-18, conv1 summed to 1 channel, fresh fc head"`, so the
  provenance is already in the artefact either way.

- **PGS Catalog is not blanket CC BY 4.0.** Scoring files carry a `#license` header that defaults to
  the EMBL-EBI Terms of Use, and **individual scores can be more restrictive** — the PGS Catalog
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
  revision of this file claimed "no per-sample genotype is ever published", which was false: the repo
  tracks `data/genomics/g1k_dosages.tsv` (**400 individuals x 77 SNPs**) and the browser bundle
  `web/public/data/genomics/pgs000001_prs.json` carries **30 individuals**. Both are effect-allele
  dosages at 77 published PRS loci for 1000 Genomes samples.
  This is not a privacy escalation — 1000 Genomes participants consented to open, unrestricted data
  release and the full genotypes are downloadable from IGSR/ENA without access control, so a 77-SNP
  subset exposes nothing IGSR does not already publish. But it *is* redistribution, so the IGSR
  disclaimer and attribution apply, and it is recorded here rather than glossed.
  The **2,504-individual** matrix used for the ancestry-stratification analysis is deliberately **not**
  committed (git-ignored, regenerable via `scripts/data/fetch_g1k_dosages.py`): the published output
  is the aggregate per-super-population statistics, and shipping the full matrix would expand the
  redistribution for no demo benefit.
- **Model weights carry no genotypes.** Only the published effect weights and aggregate metrics are
  pushed to the Hub.
- **Nothing openSNP-derived is published.** The archive was CC0 and existing copies remain legally
  usable, but the maintainers deleted it in April 2025 specifically over DNA-misuse concerns. That is
  a request worth honouring regardless of what the licence permits.

## 5. Published model weights

Weights live on the Hugging Face Hub under `Daemons-Q`, pinned by 40-character commit sha, and never
in git (`.gitignore` excludes `*.onnx`, `*.safetensors`, `*.pt`, `*.npy`; CI fails the build if one is
staged). Each repo's model card carries the dataset, its licence, the measured metrics and the
published baseline they are measured against — including where the model falls short.

## 6. OSS libraries (commercial-OK)

- **node-seal** (MIT) — Microsoft SEAL → WASM, real in-browser CKKS. **mpc-framework** +
  **emp-wasm-engine** + **summon-ts** (MIT) — real in-browser garbled-circuit 2PC for the non-linear
  (argmax) layer, both parties loopback in one tab.
- **onnxruntime-web** (MIT) — runs the published industry models in the browser.
- **MCP Python SDK** (`mcp`, MIT) — the quantum-safe MCP server; the official
  **@modelcontextprotocol/server-filesystem** (MIT) for external-MCP interop.
- **PennyLane** (Apache-2.0) — variational quantum layers. The FedAvg and secure-aggregation code is
  this repository's own; **Flower is cited as prior art, not used as a dependency** (the unused `flwr`
  requirement was removed in Round 17).
- **xgboost** (Apache-2.0), **TenSEAL** (Apache-2.0), **OpenFHE** (BSD-2), **MPyC** (MIT),
  **scikit-learn** / **scipy** (BSD-3).
- **QRNG health testing** follows **NIST SP 800-90B §4** (Repetition Count + Adaptive Proportion on
  raw pre-conditioning samples). SP 800-22 is *not* used for noise-source monitoring: SP 800-90B §4.1
  states those tests *"will almost always fail, and thus are not useful for monitoring the noise
  source."* The entropy source is a local state-vector measurement simulation. The optional
  **ANU QRNG** API is opt-in, online-only, never used in CI, and subject to the ANU service terms.

## 7. Rejected as NOT commercial-OK

- **DESILO** `desilofhe` — non-commercial licence (removed Round 4).
- **IBM `snapml`** (Graph Feature Preprocessor) — **Proprietary**. This is why the finance model
  reimplements the documented graph typologies instead of using the published GFP, and why it scores
  below the published baseline. The gap is reported rather than closed by taking the dependency.
- **Bank Account Fraud** suite — CC BY-NC-ND; **IEEE-CIS** — restrictive Kaggle terms.
- **UNSW-NB15** — commercial use requires the authors' agreement.
- **Zama tfhe-rs / Concrete** — BSD-3-Clear, research only; commercial use needs a paid patent licence.
- **QKDNetSim** — **GPL-2.0** (strong copyleft), so it is not listed as a commercial-OK reference
  implementation.
