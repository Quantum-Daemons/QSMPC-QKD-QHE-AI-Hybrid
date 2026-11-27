---
license: cc-by-4.0
library_name: onnx
pipeline_tag: tabular-classification
tags:
  - quantum-safe
  - privacy-preserving
  - telecom-ids
  - onnx
  - onnxruntime-web
---

# Federated 5G-NIDD detector

multinomial logistic regression, 9x36 + 9, on 36 standardised and clipped flow features, trained by FedAvg across the corpus's two base stations under pairwise masks

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **federated** model for the `telecom_ids` use case: a linear model the corpus's two base
stations trained together without pooling their flows, by FedAvg under pairwise masks agreed by X25519, the
page's secure aggregation. It is trained on the labels and distilled from no teacher. The telecom page scores
it under CKKS on 2,048 held-out flows that ship with the site, from the site's own copy of these weights,
and replays the last round's masked sum in the tab. The pre-registered comparison with each station alone
and with pooling, which it does not win, is in the notes below.

## Measured performance

| metric | value |
|---|---|
| `ece_station_1` | 0.004095 |
| `ece_station_2` | 0.004994 |
| `flamby_macro_f1` | 0.970821 |
| `l2` | 1e-06 |
| `local_iterations` | 20 |
| `masked_messages` | 270 |
| `n_eval` | 233591 |
| `pooled_flamby_macro_f1` | 0.973941 |
| `rounds` | 40 |
| `station_1_alone_flamby_macro_f1` | 0.984722 |
| `station_1_macro_f1` | 0.959339 |
| `station_2_alone_flamby_macro_f1` | 0.958705 |
| `station_2_macro_f1` | 0.982303 |
| `student_params` | 333 |
| `temperature` | 0.9312519672407261 |
| `union_accuracy` | 0.986078 |
| `union_macro_f1` | 0.97163 |

## Published baselines this is measured against

- **Target metric**: macro F1 across 9 network-attack classes, the mean of the two base stations' test flows (FLamby's metric)
- **Baseline**: No published figure on this split: no peer-reviewed federation of 5G-NIDD by base station was found (2026-11-26). The comparison is the pre-registered one in references/telecom-nidd-federated-preregistration.md, on the same rows and by the same metric: station 1's own model 0.985, pooling every flow in one place (the page's CKKS student) 0.974, station 2's own model 0.959 (docs/measurements/telecom-nidd-federated-2026-11-26.json).
- **Companion metric shown alongside**: `pooled_flamby_macro_f1` - the same procedure on every flow in one place, the page's CKKS student, scored by the same metric.

## Training data

- **Dataset**: 5G-NIDD (BTS1_BTS2_fields_preserved.csv)
- **Licence**: CC-BY-4.0
- **Source**: https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361 (licence read 2026-08-16)
- **Attribution (CC-BY-4.0)**: Y. Siriwardhana et al., 'Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)', in IEEE Data Descriptions, doi:10.1109/IEEEDATA.2025.3592888. Dataset: doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361, CC BY 4.0.

1,215,890 real 5G network flows x 113 columns, counted in this file rather than quoted, 9 classes: Benign 477,737 | UDPFlood 457,340 | HTTPFlood 140,812 | SlowrateDoS 73,124 | TCPConnectScan 20,052 | SYNScan 20,043 | UDPScan 15,906 | SYNFlood 9,721 | ICMPFlood 1,155. 281,529 OF THOSE ROWS ARE THE SAME FLOWS AS 281,529 OTHERS UNDER A CONTRADICTORY LABEL, AND THE MODEL DOES NOT READ THEM. One attacker's UDP flood (10.155.15.7 to the victim) is in the first base station's file labelled Benign and in the second's labelled UDPFlood. The copies agree on every column but the labels and the file-position columns, and the 421,471 packets behind them carry identical timestamps in the two capture files. The loader removes the Benign copy, leaving 934,361 rows, Benign 196,208 of them, and asserts the count. LICENCE VERIFIED AT THE SOURCE, not at a mirror: the Metax record for dataset 9d13ef28-2ca7-44b0-9950-225359afac65 reads access_type Open, license CC-BY-4.0 and restriction_grounds []. The IEEE DataPort listing whose subscription wall this project originally read as a licence restriction is a MIRROR; the Fairdata record is the authors' repository of record. ONE COLUMN MATTERS MORE THAN THE REST. This is `BTS1_BTS2_fields_preserved.csv`, 113 columns. The release also ships `Combined.csv` at 52 columns, which is what this project trained on until Round 50. Same rows, same classes, same distribution - and all 52 of Combined's columns are informationally identical in the wide file, equal distinct-value counts on every one, so Combined is a strict column SUBSET. It omits `IdleTime`, a column worth 22 accuracy points on the rows as published (0.9829 with it, 0.7616 without) because it names the attack session's file, and so tells the two copies of the flood apart: its 17 values are all multiples of 128, one per session and base station, and read as Unix times they fall between 13 and 15 September 2022, though the dataset paper gives the captures as 26 and 29 August. It is dropped, as `StartTime` is. Five other columns Combined lacks (`Dir`, `Flgs`, `TcpOpt`, `sVlan`, `dVlan`) are worth 0.0006 accuracy between them - which is redundancy rather than emptiness, since `Dir` alone separates ICMPFlood at one-vs-rest AUC 0.9811 and `TcpOpt` separates SlowrateDoS at 0.9433. The other 41 columns already carry what they carry. SIXTY-SEVEN OF THE 113 COLUMNS ARE UNUSABLE and the loader drops them with reasons attached (qsh.models.datasets.NIDD_DROPPED, republished in the model's feature_spec.json): 3 labels (`Attack Type`, its binary collapse `Label`, and `Attack Tool` which scores accuracy 0.9615 on its own), 9 row identifiers, 10 host fingerprints (addresses, MACs, ports, IP ids), 5 columns bit-identical to `Dur` on every row (`RunTime`/`Mean`/`Sum`/`Min`/`Max`), the file label `IdleTime`, and 39 that hold exactly one state. 46 features remain. THE 39 SINGLE-STATE COLUMNS ARE WORTH A SEPARATE SENTENCE because their names promise signal: 30 of them are 100% NULL in this release, and that includes every SIntPkt/DIntPkt inter-packet-time column and every jitter column. They are empty, not weak. THE ROW IDENTIFIERS AND THE COPIES ARE ONE FINDING. On the rows as published, inputs equal on the two copies cannot pass accuracy 0.7685, so a higher score rests on a column that differs between them: `Unnamed: 0`, `Rank`, `Seq`, `Offset` or `IdleTime`. The dataset paper's 9-class baselines (Samarakoon et al., arXiv:2212.01298) rank `Seq` and `Offset` among their top-10 inputs, and Wang, Fok & Thing's 28 inputs include `Offset`. See M_NIDD's baseline_source below.

## Notes and limitations

FEDERATED, NOT DISTILLED. The corpus's two base-station files are two clients: each round, each runs 20 L-BFGS-B iterations on its own training flows from the global model, and the aggregator sums their parameters, weighted by their shares of the training flows and masked with pairwise masks whose seeds each pair agrees by X25519 (the page's secure aggregation), for 40 rounds; the penalty and the temperature were chosen from masked sums of the stations' validation losses. The stations exchanged their row counts and each feature's mean and deviation in the clear, to standardise alike. THE PRE-REGISTERED COMPARISON IT DOES NOT WIN. By the mean of the two stations' test macro F1, station 1's own model scores 0.985, pooling 0.974, this model 0.971 and station 2's own 0.959: station 1's model carries to station 2's flows while station 2's does not carry back (SYNFlood 0.83 on station 1's flows), and one linear model trained on both sits between them. It is 0.003 below pooling. Under CKKS in the browser, on the 2,048 held-out flows, the largest logit error is about 2e-7 and no decision changes; all 9 scores are decrypted and the argmax is taken in the clear, as for the other linear students. LIMITS: with two stations each can compute the other's update from the global model and its own, so the masks hide the updates from the aggregator only; X25519 is not post-quantum; one process played both stations and the aggregator, offline and once; no differential privacy is applied; the published model's split is random, so a station's test flows share attack sessions with its training flows; the time split below orders them, and only the cross-station test holds whole capture files out. UNDER A TIME SPLIT IN EACH CAPTURE FILE (Round 86, batch 9, pre-registered before any fit in references/telecom-nidd-clean-protocols-preregistration.md). The same federation, each station on its own time-ordered parts (the first 70 percent of each capture file's rows of each class to train, the last 15 percent to test), scores 0.9860 by the mean of the stations' macro F1, within 0.007 of one linear model trained on both (0.9867); without the six identifier-like columns 0.9698, and without TTL as well 0.8640; every masked message cancelled. The linear model refitted on that split, without the six columns, agrees with its plaintext under CKKS on 2,048 of its test flows (no decision changes; largest logit error 2.2e-7). Trained on station 2 and tested on station 1, the linear model scores a macro F1 of 0.935; on NetsLab-5GORAN-IDD, scored without retraining on the classes both corpora share, 0.26.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
