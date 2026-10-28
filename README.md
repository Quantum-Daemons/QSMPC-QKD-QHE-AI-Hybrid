# QSMPC, QKD, QHE, AI - Quantum-Safe Hybrid Orchestration

> An end-to-end orchestration **security layer** that chains **QKD -> QSMPC -> QHE/FHE ->
> privacy-preserving AI** on a **PQC-first hybrid** foundation, with industry use cases
> (finance / healthcare / genomics / telecom-defence / **satellite**), a **classical-vs-quantum
> benchmark** (latency, accuracy, a **security posture table**, and an
> **Alice <-> Bob** two-party E2E), and a researcher/business-facing **SPA** that runs **entirely
> client-side** in public-demo mode.

License: **Apache-2.0**, Python **3.12**, GPU: **NVIDIA RTX PRO 6000 Blackwell + CUDA 13**

This is a **research / proof-of-concept**: real implementations where tractable, honest simulation
where not. Claims are cited to primary sources in [`references/`](references/REFERENCES.md), and
every correction a check has made is recorded in [`docs/verification-log.md`](docs/verification-log.md).

**So the claim is the composition, not either half:** PQC key establishment to a simulated QKD link
(it yields a key length, not key bytes) to secure multiparty computation to homomorphic evaluation to
model inference, chained end to end in one browser tab with zero server compute. The two MPC parties
are notional and the quantum channel is simulated, because no browser API for QKD exists. What is
and is not new, the positioning (quantum-safe, not quantum-advantage) and the four workstreams:
[`docs/use-cases.md`](docs/use-cases.md).

---

## At a glance

| Area | What runs | Real or simulated | What it is checked against |
|---|---|---|---|
| Post-quantum cryptography | ML-KEM-768 + X25519 hybrid key agreement, ML-DSA-65 signatures | real, in the browser and in Python | FIPS 203 and FIPS 204; NIST's ACVP cases for ML-KEM-768 |
| QKD on fibre | BB84 decoy-state and E91; the BB84 key is a composable finite key over a privacy-amplification block of 1e7 key-basis bits, E91's stays asymptotic | simulated channel, real bounds | Ma et al. 2005; Rusca et al. 2018 Table I (26 dB row reproduced) |
| QKD from a satellite | Micius downlink; the key is the finite key of one pass; a trusted relay and a constellation | simulated | Liao et al. 2017's pass (detections within 1 percent), 23 measured Micius nights, Lim et al. 2014 |
| QRNG | quantum-measurement simulation with SP 800-90B health tests and seeded or seedless extraction | simulated source, real tests | NIST SP 800-90B |
| Private inference | CKKS linear layers (node-seal) and a garbled-circuit 2PC for the non-linear part (Polytune) | real protocols, notional parties | plaintext against CKKS on the same inputs |
| QSMPC | private set intersection and secure summation; quantum forms in the full deployment | real in the browser, simulated quantum forms | the plaintext result |
| QHE and QML | quantum one-time pad, a Clifford+T QCNN, a QML intrusion detector | simulated circuits | a logistic-regression baseline on the same split |
| Models | eleven published models, run with ONNX Runtime in the browser | real inference | per-model metrics and model cards |
| Agentic AI | a small language model in the browser over a quantum-safe MCP channel | real | signed tool results |

---

## Architecture

```mermaid
flowchart TB
  CFG["config: defaults.py, link profiles, validation sets"]
  UI["Pages: React + Vite SPA"] --> CC["clientCompute: one pipeline, data to signed result"]
  CFG -->|export_config.py| CC
  CC --> BR
  subgraph BR["Runs in the browser: the public demo, no server compute"]
    PQC["PQC: ML-KEM-768 + X25519, ML-DSA-65"]
    QKD["QKD: simulated channel, finite-key bounds"]
    FHE["CKKS: node-seal"]
    MPC["2PC: Polytune, WASM"]
    ONNX["Models: ONNX Runtime Web, weights from the Hugging Face Hub"]
    MCP["Agent: MCP channel, a small LM"]
    PQC ~~~ FHE ~~~ ONNX
    QKD ~~~ MPC ~~~ MCP
  end
  UI -.->|full mode| SRV
  subgraph SRV["Full deployment, optional"]
    API["FastAPI: REST and WebSocket"] --> ORCH["qsh.orchestration: the same pipeline, reading the config directly"]
    ORCH --> MOD["qsh: pqc, qkd, qrng, qsmpc, qhe, fhe, mpc, qml_ids, models, mcp"]
  end
```

Both columns run one pipeline from one config: `scripts/export_config.py` bakes the Python defaults
into the browser, and cross-language tests hold the two implementations to each other. More:
[`docs/architecture.md`](docs/architecture.md).

### How a run flows

```mermaid
flowchart TB
  subgraph S1["1. Set up the session"]
    direction LR
    D["Data"] --> H["PQC handshake: ML-KEM-768 + X25519"] --> K["QKD key: rate and finite key, simulated"]
  end
  subgraph S2["2. Compute without revealing the data"]
    direction LR
    C["QSMPC: PSI or secure sum"] --> I["Private inference: CKKS + 2PC"] --> Q["QSMPC-FHE comparison"]
  end
  subgraph S3["3. Report"]
    direction LR
    R["Result"] --> SG["Signed with ML-DSA-65"]
  end
  S1 --> S2 --> S3
```

| Use case | Industry | Its own stages |
|---|---|---|
| `finance_aml` | finance, AML | PSI over cross-bank blacklists, federated training, FHE fraud inference |
| `medical_fl` | healthcare | federated training with secure aggregation, private diagnosis on MedMNIST images |
| `genomics` | genomics | secure summation of pooled allele counts, FHE |
| `telecom_ids` | telecom, defence | simulated QKD with an optional eavesdropper, a QML intrusion detector |
| `satellite_comms` | satellite | free-space QKD over a Micius downlink, QSDC with an inline eavesdrop check |

### The QKD key

```mermaid
flowchart TD
  CH["Channel: fibre loss, or a satellite pass"] --> G["Gains and errors per intensity"]
  G --> A["Asymptotic decoy bound: the rate per pulse sent"]
  G --> BL["Counts of a block: 1e7 key-basis bits on fibre, one pass from a satellite"]
  BL --> F["Composable finite key: Rusca et al. 2018, or Lim et al. 2014 with a vacuum state"]
  F --> T{"256 bits or more?"}
  T -->|yes| X["Folded into the PQC session key"]
  T -->|no| P["The session rests on PQC alone"]
```

---

## Pages

```mermaid
flowchart LR
  O["Overview"] --- V["Introduction and Value"]
  O --- VA["Validations"]
  O --- E["E2E Orchestration"]
  O --- U["Use cases"]
  U --- F["Finance - AML"]
  U --- M["Medical - FL"]
  U --- G["Genomics"]
  U --- T["Telecom / Defence"]
  U --- ST["Satellite: Downlink and Network tabs"]
```

| Page | Route | What it shows |
|---|---|---|
| Overview | `/` | the architecture at a glance, the four workstreams, what is real and what is simulated |
| Introduction and Value | `/value` | what is and is not new, the prior art checked, and why now: the regulatory clock |
| Validations | `/validations` | the PQC primitives, a BB84 run with an eavesdropper, NIST ACVP cases, network key schedules, an MCP agent call, all in the browser |
| E2E Orchestration | `/orchestration` | the pipeline of any use case, run live: start, pause, step, stop |
| Use cases | `/usecases` | the five industry use cases |
| A use case | `/usecase/<id>` | a two-party simulator, the run's metrics and security facts, the benchmark, the models, an MCP panel; the satellite page adds the Network tab |

<table>
<tr>
<td><img src="docs/images/overview.jpg" alt="Overview page" width="420"><br>Overview</td>
<td><img src="docs/images/value.jpg" alt="Introduction and Value page" width="420"><br>Introduction and Value</td>
</tr>
<tr>
<td><img src="docs/images/validations.jpg" alt="Validations page after an ML-KEM and a BB84 run" width="420"><br>Validations, after an ML-KEM and a BB84 run</td>
<td><img src="docs/images/orchestration.jpg" alt="E2E Orchestration page after a run" width="420"><br>E2E Orchestration, after a run</td>
</tr>
<tr>
<td><img src="docs/images/usecases.jpg" alt="Use cases index" width="420"><br>Use cases</td>
<td><img src="docs/images/usecase-finance.jpg" alt="Finance AML use case after a run" width="420"><br>Finance - AML</td>
</tr>
<tr>
<td><img src="docs/images/usecase-medical.jpg" alt="Medical FL use case after a run" width="420"><br>Medical - FL</td>
<td><img src="docs/images/usecase-genomics.jpg" alt="Genomics use case after a run" width="420"><br>Genomics</td>
</tr>
<tr>
<td><img src="docs/images/usecase-telecom.jpg" alt="Telecom and defence use case after a run" width="420"><br>Telecom / Defence</td>
<td><img src="docs/images/usecase-satellite.jpg" alt="Satellite use case after a run" width="420"><br>Satellite, Downlink tab</td>
</tr>
<tr>
<td><img src="docs/images/usecase-satellite-network.jpg" alt="Satellite Network tab: a pass, the measured nights, the relay" width="420"><br>Satellite, Network tab</td>
<td></td>
</tr>
</table>

The images are taken from a local client-only build by
[`web/scripts/capture_screenshots.mjs`](web/scripts/capture_screenshots.mjs), so they can be
regenerated with each change.

---

## Quickstart

```bash
bash env/setup_venv.sh && source .venv/bin/activate   # Python 3.12 venv and the core stack
make test                                              # the test suite, slow tests deselected
cd web && npm ci && npm run dev                        # the SPA at http://localhost:5173
```

The full deployment adds the optional tiers and the backend:

```bash
pip install -r requirements-ai.txt -r requirements-fhe.txt -r requirements-api.txt
uvicorn api.main:app --port 8000
python -m qsh.orchestration.pipeline --usecase finance_aml --out results/
```

The GPU, FHE, MPC, Rust/WASM and dependency notes, the upstream submodules and the languages each
part is written in: [`docs/development.md`](docs/development.md).

### Development workflow

```mermaid
flowchart TB
  subgraph B1["Before the merge"]
    direction LR
    L["Local: Python 3.12 venv, pytest, scripts/ci_local.sh"] --> PR["Pull request: GitHub Actions CI"] --> DB["The branch on the demo: Docker, deployed from git"] --> LB["Live check in a browser"]
  end
  subgraph B2["After the merge"]
    direction LR
    MG["Merge to main"] --> DM["Main on the demo again"] --> LM["Live check again"]
  end
  B1 --> B2
```

Every round is recorded in [`docs/deliverable-verification.md`](docs/deliverable-verification.md)
(what was checked and found) and [`docs/verification-log.md`](docs/verification-log.md) (what it
taught); `make report` rebuilds the technical report,
[`report/technical-report.pdf`](report/technical-report.pdf).

---

## Repository layout

```
src/qsh/      the qsh package: orchestration, pqc, qkd, qml_ids, qsmpc, qhe, config, models, qrng,
              fhe, mpc, ai, mcp (real MCP server, SDK >=2.0), usecases, benchmark, data, monitoring
api/          FastAPI backend (REST, WebSocket, downloads, /metrics)
web/          React + Vite + TS SPA, and web/scripts/ checks and the screenshot capture
docker/       Dockerfiles, compose (full and client-only builds), Caddyfile
monitoring/   Prometheus config and Grafana notes
references/   primary-source notes and REFERENCES.md
scripts/      bootstrap, benchmark, export_config.py, publish_hf.py, deploy_demo.sh, ci_local.sh,
              train/ (per-vertical training), data/ (dataset provenance)
docs/         the documents below, and images/
report/       the technical report (LaTeX) and its generated tables
tests/        pytest suite
```

---

## Documentation

| Document | What it holds |
|---|---|
| [`docs/use-cases.md`](docs/use-cases.md) | what is and is not new, the positioning, the four workstreams, the industry use cases |
| [`docs/architecture.md`](docs/architecture.md) | the pipeline contract, faithful hybrid inference, the SPA's shape, the compute topologies |
| [`docs/development.md`](docs/development.md) | the quickstart with its notes, the dev environment, GPU, FHE, MPC, Rust/WASM, advisories, languages, submodules |
| [`docs/models-and-benchmarks.md`](docs/models-and-benchmarks.md) | the datasets, the MPC-FHE against QSMPC-FHE benchmark, model tiers, what the pages show |
| [`docs/findings.md`](docs/findings.md) | the WS3 and WS4 challenges, and every concern and recommendation recorded so far |
| [`docs/deployment.md`](docs/deployment.md) | the deployment runbook and tiers |
| [`docs/threat-model.md`](docs/threat-model.md) | the threat model |
| [`docs/crypto-agility.md`](docs/crypto-agility.md) | the crypto-agility registry and its rules |
| [`docs/pqc-roadmap.md`](docs/pqc-roadmap.md) | the PQC roadmap and future tracks |
| [`docs/ai-act-art50-gap-analysis.md`](docs/ai-act-art50-gap-analysis.md) | the EU AI Act Art. 50 gap analysis |
| [`docs/deliverable-verification.md`](docs/deliverable-verification.md) | the verification register, round by round |
| [`docs/verification-log.md`](docs/verification-log.md) | the verification log |
| [`report/technical-report.pdf`](report/technical-report.pdf) | the technical report |
| [`references/REFERENCES.md`](references/REFERENCES.md) | the annotated primary sources |
| [`DATA_LICENSES.md`](DATA_LICENSES.md) | the datasets and their licences |

---

## References & license

- Primary sources: [`references/REFERENCES.md`](references/REFERENCES.md) (annotated, paper -> code -> workstream).
- Standards/threat context: [`docs/pqc-roadmap.md`](docs/pqc-roadmap.md).
- Code license: **Apache-2.0** ([`LICENSE`](LICENSE)). Upstream OSS retains its own licenses
  ([`docs/development.md`](docs/development.md), OSS / submodules).

> Private business notes are kept outside this repository by design and are not published here.
