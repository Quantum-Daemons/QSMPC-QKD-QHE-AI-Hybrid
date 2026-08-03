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
| **Finance** | [IBM AMLworld HI-Small](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — 5,078,345 transactions, 5,177 laundering (1 in 981) | **CDLA-Sharing-1.0** | **Yes** | CDLA-Sharing §1.11 defines *Results* as the outputs of Computational Use, and §3.5 states *"This Agreement imposes no obligations or restrictions on Your Use or Publication of Results."* Trained weights are Results; the ShareAlike reciprocity attaches to redistributing the **Data**, which we do not do. Test-fixture feature rows (`fixtures.npy`) are git-ignored and excluded from every upload for the same reason. |
| **Medical** | [MedMNIST v2 BreastMNIST](https://medmnist.com/) — 780 breast-ultrasound images | **CC BY 4.0** | **Yes** | Attribution: Yang et al., *Nature Scientific Data* 2022. |
| **Medical (planned)** | [Camelyon17-WILDS](https://wilds.stanford.edu/datasets/) | **CC0 1.0** | **Yes** | The official split trains on 3 hospitals and holds 2 out as OOD val/test. Federating across all 5 would destroy the OOD test set, so the federation spans the 3 training centres only. |
| **Genomics** | [PGS Catalog](https://www.pgscatalog.org/downloads/) scoring files + [1000 Genomes phase 3](https://www.internationalgenome.org/) via Ensembl REST | **per-score / see below** | **No** | See §4. |
| **Satellite** | [ESA Anomaly Dataset](https://zenodo.org/records/12528696), Mission1 lightweight channels 41–46 | **CC BY 3.0 IGO** | **Yes** | Missions are anonymised `Mission1/2/3` — they are **not** named Sentinel anything. 14 years of telemetry, ~15.4M samples per channel. Only labels whose `anomaly_types.csv` Category is `Anomaly` count as positives; the annotated **rare *nominal* events** are not faults and scoring them as such would inflate the anomaly rate. |
| **Telecom** | [CIC-IDS2017, Engelen-corrected](https://intrusion-detection.distrinet-research.be/WTMC2021/) | **no licence stated by either provider** | **No** | CIC-IDS2017 states no licence — only *"publicly available for researchers"* plus a citation request — and the Engelen redistribution states no terms either. **Absent an affirmative grant, neither is safe to bundle or redistribute.** Local training and benchmarking only. Cite Engelen, Rimmer & Joosen (WTMC 2021 @ IEEE S&P Workshops) *and* Sharafaldin et al. (2018). |

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
| `data/satellite/smap_msl_telemetry.tsv`, `web/public/data/satellite/telemetry.json` | **no licence granted by any provider** `[UNVERIFIED — reasoned inference]` | NASA SMAP/MSL via [Telemanom](https://github.com/khundman/telemanom), Hundman et al. 2018 (KDD), NASA JPL. The Telemanom **code** is Apache-2.0; **the telemetry carries no licence statement anywhere**, and the Kaggle mirror it links to is labelled *"Data files © Original Authors"*. The public-domain reading rests on NASA SPD-41a (*"should be released with a Creative Commons Zero license"* — a *should*, not a grant) plus 17 U.S.C. §105, which removes **US** copyright only and does not clear the work outside the United States. Treated as an inference, not a licence. This is why **ESA-ADB (CC BY 3.0 IGO) is the satellite training corpus** and SMAP/MSL is retained only as a bundled demo sample. |
| `web/public/data/medical/breast_cancer_sample.json` | **public domain** | Wisconsin Diagnostic Breast Cancer, UCI ML / scikit-learn (Wolberg, Street & Mangasarian). |

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

- **PGS Catalog is not blanket CC BY 4.0.** Scoring files carry a `#license` header that defaults to
  the EMBL-EBI Terms of Use, and **individual scores can be more restrictive** — the PGS Catalog
  documentation shows a live CC BY-NC-ND 4.0 example. `scripts/train/genomics_prs.py` re-reads that
  header at run time and gates publication on it.
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
