# Bundled datasets & model sources — licenses + attribution

This project bundles **small, commercial-OK** real-data samples so every industry use case runs on a
real (or real-methodology) dataset offline (no runtime download). Each is listed with its license and
attribution. The project's own code is Apache-2.0; bundled data keeps its own license.

| Use case | Bundled file / source | License | Attribution / notes |
|----------|----------------------|---------|---------------------|
| **Genomics** | `data/genomics/pgs000001_breast_cancer.tsv` (real **77-SNP breast-cancer PRS** weights) **+ `data/genomics/g1k_dosages.tsv` (real genotype dosages)** | weights **CC BY 4.0**; genotypes **open / IGSR** | PRS weights: PGS Catalog [PGS000001](https://www.pgscatalog.org/score/PGS000001/) (PRS77_BC; Mavaddat et al.). Genotypes: **real 1000 Genomes phase-3** effect-allele dosages (400 individuals) for exactly these 77 SNPs, fetched once from the **Ensembl REST API** over IGSR / 1000 Genomes (open data-use, commercial-OK, no auth). Both the PRS model **and** the genotypes are now real — the PRS is a genuine score over real human genotypes. (openSNP CC0 was the prior plan but opensnp.org is offline; 1000 Genomes via Ensembl is the live no-auth source.) |
| **Medical** | MedMNIST (PneumoniaMNIST), baked into the full Docker image | **CC BY 4.0** | [MedMNIST v2](https://medmnist.com/) (Yang et al.). Real medical images; the genuine FHE CNN (Round 10). |
| **Finance** | `data/finance/paysim_sample.tsv` — a **real PaySim** transaction sample → the 8 AML features | **CC BY-SA 4.0** | Real PaySim mobile-money transactions (Kaggle [ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1); Lopez-Rojas et al. 2016, the published industry-standard simulator). The 8 features are derived from real PaySim fields (amount / type / step / balances); fraud is oversampled for training balance (real base rate 0.13 %). The **model** is a real **XGBoost** (Apache-2.0) distilled to a CKKS-linear surrogate. |
| **Telecom/Defense** | QKD-attack telemetry signatures (literature-grounded) | grounded in Al-Kuwari et al. 2026 (arXiv:2509.14282) | The **model** is a real classical RF + a **Quantum SVC** (quantum-kernel SVM, the 2026 QML-IDS standard) + a variational QNN. |
| **Satellite** | `data/satellite/smap_msl_telemetry.tsv` — **real NASA SMAP/MSL** telemetry (6 channels + real labeled anomalies) | **CC0 / US-Gov public domain** | Real NASA spacecraft telemetry from the **Telemanom** project (Hundman et al. 2018, KDD; NASA JPL). Per NASA's data policy (17 U.S.C. §105; NASA Science Data Portal CC0), mission data without a restrictive notice is public domain → commercial use permitted. The **model** is a real linear telemetry-anomaly classifier run under CKKS. |

> **Real-data status (web-verified June 2026 — honest).** **Four of the five verticals run on real, commercial-OK
> data**: **medical** (Wisconsin breast-cancer, public domain + MedMNIST CC-BY), **genomics** (PGS-Catalog CC-BY
> weights **+ real 1000 Genomes phase-3 genotypes** via the open Ensembl/IGSR API), **finance** (a real **PaySim**
> transaction sample, Kaggle CC-BY-SA-4.0; Lopez-Rojas 2016), and **satellite** (real **NASA SMAP/MSL** spacecraft
> telemetry, Telemanom / NASA JPL — obtained on NASA's CC0 / US-Gov-public-domain basis per NASA's official data
> policy, *not* relying on the Kaggle wrapper's "© authors" label). Only **telecom/defense** runs on
> **cited-methodology synthetic** telemetry, because **no commercial-OK real network-attack dataset exists**
> (UNSW-NB15 / CICIDS / NSL-KDD are research-only; the model is grounded in Al-Kuwari 2026) — labelled honestly in
> the run, never uncited. In every case the **model/architecture is the real artefact**, and the FHE-encrypted
> inference is the FHE-feasible part (linear / PRS / small CNN / distilled surrogate), not task-SOTA — a
> high-fidelity PoC, not a trained production system.

### Client-side demo (`web/public/data/`, loaded by `web/src/lib/realData.ts`)
The public **client-side demo** ships the same real datasets where small + commercial-OK, so it runs on **real
data with zero backend** (for the four verticals where the full deployment also has real data — medical, genomics,
finance + satellite):
| Demo file | License | Source |
|---|---|---|
| `web/public/data/genomics/pgs000001_prs.json` — the real **PGS000001** 77-SNP PRS weights + real **1000 Genomes** genotype dosages → a real in-browser Polygenic Risk Score | **CC BY 4.0** / IGSR | PGS Catalog [PGS000001](https://www.pgscatalog.org/score/PGS000001/) (Mavaddat et al.) + 1000 Genomes phase-3 (Ensembl/IGSR) — same sources as the bundled `.tsv`s above. |
| `web/public/data/medical/breast_cancer_sample.json` — a real **Wisconsin Diagnostic Breast Cancer** sample (12 rows × 8 features) + a real logistic-regression model | **public domain** | UCI ML Repository / scikit-learn bundled dataset (Wolberg, Street & Mangasarian). |
| `web/public/data/finance/paysim_sample.json` — real **PaySim** transaction rows (8 AML features) + the distilled CKKS-linear fraud model | **CC BY-SA 4.0** | Kaggle [ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) (Lopez-Rojas 2016) — same source as the bundled `.tsv`. |
| `web/public/data/satellite/telemetry.json` — real **NASA SMAP/MSL** telemetry windows + a real linear anomaly model | **CC0 / public domain** | Telemanom (Hundman et al. 2018; NASA JPL) — same source as the bundled `.tsv`. |

Only **telecom** has no shippable commercial-OK real attack dataset → the demo uses representative **synthetic**
features (logged honestly in the run). The demo sets are public / de-identified, so no real *personal* data is
shipped to the browser.

## OSS libraries (commercial-OK)
- **node-seal** (MIT) — Microsoft SEAL → WASM, real in-browser CKKS. **mpc-framework** + **emp-wasm-engine** + **summon-ts** (all MIT; voltrevo / PSE) — real in-browser **garbled-circuit authenticated-garbling 2PC** for the non-linear (argmax) layer, both parties run loopback in one browser (zero VPS).
- **MCP Python SDK** (`mcp`, MIT — bundles **FastMCP**) — the real quantum-safe MCP server (5 per-industry tools) + stdio client; the official **@modelcontextprotocol/server-filesystem** (MIT, Node) for real external-MCP interop. Full deployment only (the demo keeps the client-side mirror → zero VPS).
- **Flower** + **PennyLane** (Apache-2.0) — Quantum Federated Learning. **xgboost** (Apache-2.0). **TenSEAL** (Apache-2.0) / **OpenFHE** (BSD-2) / **MPyC** (MIT).
- **scikit-learn** / **scipy** (BSD-3) — the **OptiQKD** parameter optimizer (GBM/GP surrogate + gradient-free search) + the **QRNG** SP 800-22 health suite. The QRNG entropy source is a local **state-vector quantum-measurement simulation** (no external data, commercial-OK). The optional **ANU QRNG** API (`qrng.anu.edu.au`, real hardware entropy) is **opt-in + online only** — never used in CI/tests — and its use is subject to the ANU Quantum Numbers API terms (attribute the ANU QRNG service). Reference OSS (adoptable, all MIT/Apache/BSD; see `docs/pqc-roadmap.md`): SeQUeNCe, TNO-Quantum qkd_key_rate, oqs-provider, rosenpass, QFL-MLNCP/flower-fhe, dorahacksglobal QRNG. Round-16 primary-literature references (arXiv preprints, cited in `references/REFERENCES.md`, **not bundled**): **OptiQKD / AML-QKD** (arXiv:2603.04192, Mohamed et al. 2026 — AI-QKD parameter tuning) and the **seedless-QRNG + hybrid PQC-QKD combiner** (arXiv:2603.26907, Giestinhas & Spiller 2026 — SP 800-90B min-entropy + Leftover-Hash-Lemma extractor + "secure-if-either" combiner).

## Rejected as NOT commercial-OK (consistent discipline)
- **DESILO** `desilofhe` — non-commercial license (removed Round 4).
- **Bank Account Fraud** suite — CC BY-NC-ND (non-commercial); **IEEE-CIS** — restrictive Kaggle terms.
- **UNSW-NB15** — commercial use requires the authors' agreement (not freely commercial).
- **Zama tfhe-rs / Concrete** — BSD-3-Clear for research only; commercial use needs a paid Zama patent license.

These are excluded to keep the entire stack unambiguously commercial-OK.
