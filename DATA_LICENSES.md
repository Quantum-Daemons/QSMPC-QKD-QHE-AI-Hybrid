# Bundled datasets & model sources — licenses + attribution

This project bundles **small, commercial-OK** real-data samples so every industry use case runs on a
real (or real-methodology) dataset offline (no runtime download). Each is listed with its license and
attribution. The project's own code is Apache-2.0; bundled data keeps its own license.

| Use case | Bundled file / source | License | Attribution / notes |
|----------|----------------------|---------|---------------------|
| **Genomics** | `data/genomics/pgs000001_breast_cancer.tsv` — a real **77-SNP breast-cancer Polygenic Risk Score** | **CC BY 4.0** | PGS Catalog, [PGS000001](https://www.pgscatalog.org/score/PGS000001/) (PRS77_BC; Mavaddat et al.). The PRS *weights* are real; genotype dosages are sampled under Hardy-Weinberg from realistic allele frequencies (real **openSNP** CC0 / **1000 Genomes** genotypes are substitutable). |
| **Medical** | MedMNIST (PneumoniaMNIST), baked into the full Docker image | **CC BY 4.0** | [MedMNIST v2](https://medmnist.com/) (Yang et al.). Real medical images; the genuine FHE CNN (Round 10). |
| **Finance** | `data/finance/` — PaySim/AMLSim-methodology transactions (the model is the real artefact) | data: simulator-methodology; **PaySim** CSV is **CC BY-SA 4.0** | Real customer transaction data is privacy-impossible to share; **PaySim** (Lopez-Rojas et al.) is the published industry-standard simulator. The **model** is a real **XGBoost** (Apache-2.0) distilled to a CKKS-linear surrogate. |
| **Telecom/Defense** | QKD-attack telemetry signatures (literature-grounded) | grounded in Al-Kuwari et al. 2026 (arXiv:2509.14282) | The **model** is a real classical RF + a **Quantum SVC** (quantum-kernel SVM, the 2026 QML-IDS standard) + a variational QNN. |
| **Satellite** | ESA-ADB-style multi-channel telemetry, **generated** at runtime (no bundled file) | real **ESA-ADB** is **CC BY 3.0 IGO** (commercial-OK) and substitutable | [ESA-ADB](https://zenodo.org/records/12528696) (ESA / Airbus / KP Labs) — real ESA satellite telemetry. The **model** is a real linear telemetry-anomaly classifier run under CKKS. |

> **Finance / satellite note:** these run on **generated** synthetic data at runtime (PaySim/AMLSim methodology;
> ESA-ADB-style) — there is no bundled CSV; the *model/architecture* is the real artefact. Real PaySim (CC BY-SA)
> / ESA-ADB (CC BY 3.0 IGO) are commercial-OK and substitutable but not shipped (size).

### Client-side demo (`web/public/data/`, loaded by `web/src/lib/realData.ts`)
The public **client-side demo** ships the same real datasets where small + commercial-OK, so it runs on **real
data with zero backend** (for the use cases where the full deployment also has real data — medical + genomics):
| Demo file | License | Source |
|---|---|---|
| `web/public/data/genomics/pgs000001_prs.json` — the real **PGS000001** 77-SNP PRS weights as JSON → a real in-browser Polygenic Risk Score | **CC BY 4.0** | PGS Catalog [PGS000001](https://www.pgscatalog.org/score/PGS000001/) (Mavaddat et al.) — same source as the bundled `.tsv` above. |
| `web/public/data/medical/breast_cancer_sample.json` — a real **Wisconsin Diagnostic Breast Cancer** sample (12 rows × 8 features) + a real logistic-regression model | **public domain** | UCI ML Repository / scikit-learn bundled dataset (Wolberg, Street & Mangasarian). |

finance / telecom / satellite have no shippable commercial-OK real set → the demo uses representative **synthetic**
features (logged honestly in the run). The demo sets are public / de-identified, so no real *personal* data is
shipped to the browser.

## OSS libraries (commercial-OK)
- **node-seal** (MIT) — Microsoft SEAL → WASM, real in-browser CKKS. **mpc-framework** + **emp-wasm-engine** + **summon-ts** (all MIT; voltrevo / PSE) — real in-browser **garbled-circuit authenticated-garbling 2PC** for the non-linear (argmax) layer, both parties run loopback in one browser (zero VPS).
- **MCP Python SDK** (`mcp`, MIT — bundles **FastMCP**) — the real quantum-safe MCP server (5 per-industry tools) + stdio client; the official **@modelcontextprotocol/server-filesystem** (MIT, Node) for real external-MCP interop. Full deployment only (the demo keeps the client-side mirror → zero VPS).
- **Flower** + **PennyLane** (Apache-2.0) — Quantum Federated Learning. **xgboost** (Apache-2.0). **TenSEAL** (Apache-2.0) / **OpenFHE** (BSD-2) / **MPyC** (MIT).

## Rejected as NOT commercial-OK (consistent discipline)
- **DESILO** `desilofhe` — non-commercial license (removed Round 4).
- **Bank Account Fraud** suite — CC BY-NC-ND (non-commercial); **IEEE-CIS** — restrictive Kaggle terms.
- **UNSW-NB15** — commercial use requires the authors' agreement (not freely commercial).
- **Zama tfhe-rs / Concrete** — BSD-3-Clear for research only; commercial use needs a paid Zama patent license.

These are excluded to keep the entire stack unambiguously commercial-OK.
