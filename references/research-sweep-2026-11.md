# Research sweep, August to November 2026 (read on 2026-11-21)

The owner's choice of 2026-11-21, after the live audit: search the work published from August to November 2026 for each
use case, and record anything that changes a figure, a comparison or a claim on the pages, with its source. Each item was
read at its origin (the arXiv API and abstract page, the IACR ePrint record, the publisher's or project's own page) on
2026-11-21. "Cited" means a page or a model card now names it; "read, not cited" says why not.

## Across the pages: the cost of Shor's algorithm

| Source | What it states | Use |
|---|---|---|
| Pappalardo, Goudarzi, Hertz, Jacinto et al., "Shor's algorithm compiled on cat qubits", arXiv:2610.37839 (v1 2026-10-27, v2 2026-11-19), IACR ePrint 2026/2665 (CC BY) | RSA-2048 with about 23,000 cat qubits in 3.14 days, at a 1 us cycle, a 10 us reaction time, phase-flip noise $`p_Z = 5 \times 10^{-4}`$ and no bit flips; about 45,000 with a bit-flip time of about an hour; secp256k1 with at most 30,000 cat qubits in an hour; RSA-2048 in 1.25 hours on about 611,000 | Cited on `/value` |
| Webster, Berent, Chandra, Hockings et al., "The Pinnacle Architecture", arXiv:2602.11457 (v2 2026-05-05) | RSA-2048 with fewer than 100,000 physical qubits on quantum LDPC codes, at a $`10^{-3}`$ error rate, a 1 us cycle and a 10 us reaction time | Cited on `/value`: a 2026 estimate the card had left out while it said "Two 2026 preprints" |
| Xue and Covey, arXiv:2605.03951 (2026-05-05) | RSA-2048 on a half-million-qubit modular atomic processor, in 16 percent more time than one module | Read, not cited: a distribution result, not a lower count |

`/value`'s card now lists the four 2026 estimates, each under its own hardware assumptions, and says they are not on one
scale.

## Telecom: attacks on a QKD link, classified

| Source | What it states | Use |
|---|---|---|
| Isha, Singh, Kumar, Pal, Hambarde and Shukla, arXiv:2608.04047 (2026-08-04) | Seven BB84 eavesdropping attacks and a normal channel, under noise and loss, from 63 temporal QBER features: XGBoost at 88.01 percent accuracy and macro F1 0.8803 over ten runs; a fixed 11 percent QBER threshold at 25.82 percent accuracy as an attack-or-normal detector | Cited in the telecom note: the nearest published task in shape, classical, on another simulator |
| Das, Ahmed, Hussain and Pham, "QF-LOG", arXiv:2610.25203 (QUANCOM 2026) | Macro F1 0.8211 to 0.8243 on its generator, 0.5645 to 0.5669 on the same equations under shifted operating parameters | Cited in the telecom note: an in-distribution synthetic score need not travel |
| Mohamed and Al-Kuwari, arXiv:2603.03502 (2026-03-03) | Intrusion detection in decoy-state QKD as a minimax game, keeping 82 to 92 percent of the honest finite-key rate under adaptive attacks | Read, not cited: a different objective (secret-bit retention) |

The telecom note's earlier "nearest strong 2026 result" was IACR ePrint 2026/1282, on entanglement-based BBM92 and binary
detection. It stays, as the result on a different problem.

## Finance: AML on AMLworld HI-Small

| Source | What it states | Use |
|---|---|---|
| Losavio, Fenoglio, Gjoreski and Langheinrich, "Pricing Cross-Bank Context in Federated Anti-Money Laundering", arXiv:2610.35996 (ICAIF 2026 workshop, non-archival) | On HI-Small, across six GNN backbones and up to 32 silos, the transactions both banks already record give the largest gain; two counts per remote account stay useful under strong privacy, while the noise that protects shared GNN states erases their value | Read, not cited on a page: federated detection, where the finance page's DP protects one published count |
| Shahbazi et al., arXiv:2609.37155 (2026-09-29) | A GATv2 variant on HI-Small at AUPRC 64.85 and "best F1" 68.42 | Read, not cited: the abstract does not say how the threshold for the best F1 was chosen |
| IACR ePrint 2026/2879 (2026-11-12, CC BY) | MPC across institutions for systemic portfolio analysis (1 to 4 minutes) and repair (2.6 to 8.8 hours; 13 to 21 minutes federated) on public portfolios | Read, not cited: cross-institution MPC in finance, not AML |

MEGA v3 (TMLR 08/2026, Multi-FraudGT at 75.81) was already in `docs/findings.md`.

## Medical and genomics: encrypted inference

| Source | What it states | Use |
|---|---|---|
| Biswas et al., "Safhire", arXiv:2509.01253 (v2 2026-10-16, accepted by TMLR) | Linear layers under FHE on a server, non-linearities in plaintext at the client, intermediate values shuffled so the client cannot rebuild the model; up to 7.6x and 7.4x lower latency than Orion and Concrete-ML | Cited on `/value`: the medical CNN student's split is published |
| Galanopoulos, Provatas and Georgakopoulos-Soares, arXiv:2609.16211 (2026-09-14) | A genomic foundation model under client-assisted approximate HE: all released blocks and the head for one input, in 6,683 s and 9,839 MiB on one accelerator | Cited on `/value` beside Safhire |
| Ucar and Tuncer, Zenodo record 23771403 (2026-10-29, CC BY-SA 4.0) | Code and data for segmentation under non-interactive CKKS with the Orion compiler, without bootstrapping: lung fields on chest radiographs and lesions on breast ultrasound | Read, not cited: segmentation on a server; no paper linked yet |
| Backour et al., "Encrypt What Matters", arXiv:2609.09357 | Selective homomorphic inference over a region of interest | Read, not cited |

A search for peer-reviewed encrypted neural-network inference in a browser tab, from August to November 2026, found none,
so `/value`'s "not found" stands.

## Satellite

| Source | What it states | Use |
|---|---|---|
| Nguyen, Rozenberg, Zhu and Hu, arXiv:2607.07335 (2026-07-08) | Supervised (multiscale CNN, GCN, GAT) against unsupervised (Elliptic Envelope, ECOD) detectors on ESA-ADB: supervised detect better, unsupervised cost less | Read, not cited: no figure in the abstract to set beside the page's |

ESA-ADB (arXiv:2406.17826v2), Lyu (arXiv:2607.11969v2) and the 2 km free-space QKD link (arXiv:2607.24118v2) are each still
at the version the pages cite.

## DeFi

| Source | What it states | Use |
|---|---|---|
| null402, its site (read 2026-11-21) | A confidential lending app on Zama's fhEVM on the Ethereum Sepolia testnet, with liquidation triggers through the Gateway's asynchronous decryption; "fhEVM Mainnet Ready" is a label, and no mainnet contract is shown | Recorded in `defi-lending-key-custody.md`: the page's "no confidential liquidation in production" stands |
| IACR ePrint 2026/1728 (2026-08-18, CC BY) | Aave V3, SparkLend, Compound III and Morpho Blue in matched Ethereum fork tests, liquidation outcomes included | Read, not cited: the page re-derives Aave V3 and compares no protocol |

## Standards

- FIPS 206 (FN-DSA) and FIPS 207 (HQC): no draft in NIST's FIPS listing, and none in the PQC project's news after IR 8547
  (2026-10-27). No page names either.
- ISO/IEC 28033-1 and -2 (fully homomorphic encryption): draft standards as indexed; the ISO catalogue could not be read
  past its bot check. No page or test cites them; the CBOM check stays on the HE Standard v1.1 (2018).
