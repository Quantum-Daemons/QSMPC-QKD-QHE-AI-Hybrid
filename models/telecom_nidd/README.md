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

# 5G-NIDD network-attack classifier (9 classes)

gradient-boosted trees over 46 flow-statistic features

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **plaintext** model for the `telecom_ids` use case. No code path
evaluates this model under encryption; what each use case encrypts, if anything,
is stated in the notes below.

## Measured performance

| metric | value |
|---|---|
| `aggregate_accuracy` | 0.997932 |
| `leak_counterfactual_accuracy` | 0.999987 |
| `leak_counterfactual_accuracy_gain` | 0.002055 |
| `leak_counterfactual_macro_f1` | 0.99977 |
| `macro_f1` | 0.997244 |
| `majority_baseline` | 0.489468 |
| `n_classes` | 9 |
| `n_classes_predicted` | 9 |
| `n_eval` | 233591 |
| `n_features` | 46 |
| `n_samples` | 934361 |
| `wall_clock_s` | 134.3 |

## Published baselines this is measured against

- **Target metric**: macro F1 across 9 network-attack classes
- **Baseline**: Two published figures on 5G-NIDD are usually quoted - Wang, Fok & Thing report 99.61% accuracy for a classical CNN and 99.71% for a hybrid quantum-classical Quan-ConvCNN - and BOTH are accuracies on the BINARY benign-versus-malicious task, while this model is scored on 9 classes, which is why no number appears in `baseline`. The 9-class anchors are the dataset paper's random forest at 99.48% accuracy on its top-10 features (Samarakoon et al., arXiv:2212.01298) and BiTAD (Future Internet 17(11):482) at 90.47%. Caveats. (1) The Wang, Fok & Thing authors' own claim is about generalisation to UNSEEN attacks rather than about the headline; `references/REFERENCES.md` WS2 records this. (2) THOSE FIGURES ARE ON THE ROWS AS PUBLISHED, WHERE ONE ATTACKER'S FLOOD IS PRESENT TWICE UNDER CONTRADICTORY LABELS, and this model's are on the 934,361 rows left when the Benign copy is removed. (3) MEASURED HERE, on the rows as published, with an identical gradient-boosted model: accuracy 0.7616 with every identifier, every host fingerprint and the file label dropped, and no model reading those columns can pass 0.7685; 0.9829 with the file-label column `IdleTime` restored. On the cleaned rows it reaches 0.9979 without them. (4) A score above 0.7685 on the rows as published rests on a column that differs between the two copies, and those are the file-position columns: the dataset paper's top-10 inputs include `Seq` and `Offset`, and of Wang, Fok & Thing's 28 inputs exactly one, `Offset`, differs. Fitted to those 28, this project's booster scores binary accuracy 0.9599 with `Offset` and 0.7643 without it; that is not a rerun of their network. (5) When Wang, Fok & Thing train on one base station and test on the other, their accuracies fall from above 98% to 78.2-78.9% trained on base station 1 and 57.2-81.6% trained on base station 2 (their Tables V and VI), where the classical CNN, at 74.28%, beats four of the five hybrids. With the copies in place this project's booster falls the same way, to 0.4199 and 0.6059, and with them out it scores 0.9973 and 0.9877. These figures are an anchor and not a target that has been beaten or missed.
- **Companion metric shown alongside**: `aggregate_accuracy` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

There is no baseline number to have a gap to, and that is a deliberate registry decision rather than a missing measurement: the two figures usually quoted for this corpus (99.61% for a classical CNN, 99.71% for a hybrid quantum-classical Quan-ConvCNN, Wang/Fok/Thing) are ACCURACIES on the BINARY benign-versus-malicious task, and this model is judged on 9-class macro F1, so putting them in `baseline` would render an accuracy beside a macro F1 as though the two were the same scale. WHAT THE COMPARISON CAN AND CANNOT SAY. The 9-class anchors are the dataset paper's random forest at 99.48% accuracy on its top-10 features (Samarakoon et al., arXiv:2212.01298) and BiTAD (Future Internet 17(11) 482) at 0.9047. This model's 0.9979 is on 934,361 rows and its own stratified random split, and those figures are on the rows as published, so each is an anchor and not a head-to-head. WHAT THE ROWS SAY ABOUT THOSE FIGURES. On the 1,215,890 rows as published, inputs equal on the two copies of the flood cannot pass 0.7685, binary or nine-class, so a score above that rests on a column that differs between the copies. Those are `Unnamed: 0`, `Rank`, `Seq`, `Offset` and `IdleTime`. The dataset paper's top-10 inputs include `Seq` and `Offset`. Of Wang/Fok/Thing's 28 inputs exactly one, `Offset`, differs between the copies, and `published_inputs_check` fits this project's booster to them: on the binary task accuracy 0.9599 with `Offset` and 0.7643 without it. That measures what those columns can do on these rows and does not rerun those papers' models. Wang/Fok/Thing's fall across base stations, from above 98% to 78.23% and 74.28% for the classical CNN, has the same shape here: with the copies in place this booster scores 0.4199 and 0.6059, and with them out 0.9973 and 0.9877.

## Training data

- **Dataset**: 5G-NIDD (BTS1_BTS2_fields_preserved.csv)
- **Licence**: CC-BY-4.0
- **Source**: https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361 (licence read 2026-08-16)
- **Attribution (CC-BY-4.0)**: Y. Siriwardhana et al., 'Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)', in IEEE Data Descriptions, doi:10.1109/IEEEDATA.2025.3592888. Dataset: doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361, CC BY 4.0.

1,215,890 real 5G network flows x 113 columns, counted in this file rather than quoted, 9 classes: Benign 477,737 | UDPFlood 457,340 | HTTPFlood 140,812 | SlowrateDoS 73,124 | TCPConnectScan 20,052 | SYNScan 20,043 | UDPScan 15,906 | SYNFlood 9,721 | ICMPFlood 1,155. 281,529 OF THOSE ROWS ARE THE SAME FLOWS AS 281,529 OTHERS UNDER A CONTRADICTORY LABEL, AND THE MODEL DOES NOT READ THEM. One attacker's UDP flood (10.155.15.7 to the victim) is in the first base station's file labelled Benign and in the second's labelled UDPFlood. The copies agree on every column but the labels and the file-position columns, and the 421,471 packets behind them carry identical timestamps in the two capture files. The loader removes the Benign copy, leaving 934,361 rows, Benign 196,208 of them, and asserts the count. LICENCE VERIFIED AT THE SOURCE, not at a mirror: the Metax record for dataset 9d13ef28-2ca7-44b0-9950-225359afac65 reads access_type Open, license CC-BY-4.0 and restriction_grounds []. The IEEE DataPort listing whose subscription wall this project originally read as a licence restriction is a MIRROR; the Fairdata record is the authors' repository of record. ONE COLUMN MATTERS MORE THAN THE REST. This is `BTS1_BTS2_fields_preserved.csv`, 113 columns. The release also ships `Combined.csv` at 52 columns, which is what this project trained on until Round 50. Same rows, same classes, same distribution - and all 52 of Combined's columns are informationally identical in the wide file, equal distinct-value counts on every one, so Combined is a strict column SUBSET. It omits `IdleTime`, a column worth 22 accuracy points on the rows as published (0.9829 with it, 0.7616 without) because it names the attack session's file, and so tells the two copies of the flood apart: its 17 values are all multiples of 128, one per session and base station, and read as Unix times they fall between 13 and 15 September 2022, though the dataset paper gives the captures as 26 and 29 August. It is dropped, as `StartTime` is. Five other columns Combined lacks (`Dir`, `Flgs`, `TcpOpt`, `sVlan`, `dVlan`) are worth 0.0006 accuracy between them - which is redundancy rather than emptiness, since `Dir` alone separates ICMPFlood at one-vs-rest AUC 0.9811 and `TcpOpt` separates SlowrateDoS at 0.9433. The other 41 columns already carry what they carry. SIXTY-SEVEN OF THE 113 COLUMNS ARE UNUSABLE and the loader drops them with reasons attached (qsh.models.datasets.NIDD_DROPPED, republished in the model's feature_spec.json): 3 labels (`Attack Type`, its binary collapse `Label`, and `Attack Tool` which scores accuracy 0.9615 on its own), 9 row identifiers, 10 host fingerprints (addresses, MACs, ports, IP ids), 5 columns bit-identical to `Dur` on every row (`RunTime`/`Mean`/`Sum`/`Min`/`Max`), the file label `IdleTime`, and 39 that hold exactly one state. 46 features remain. THE 39 SINGLE-STATE COLUMNS ARE WORTH A SEPARATE SENTENCE because their names promise signal: 30 of them are 100% NULL in this release, and that includes every SIntPkt/DIntPkt inter-packet-time column and every jitter column. They are empty, not weak. THE ROW IDENTIFIERS AND THE COPIES ARE ONE FINDING. On the rows as published, inputs equal on the two copies cannot pass accuracy 0.7685, so a higher score rests on a column that differs between them: `Unnamed: 0`, `Rank`, `Seq`, `Offset` or `IdleTime`. The dataset paper's 9-class baselines (Samarakoon et al., arXiv:2212.01298) rank `Seq` and `Offset` among their top-10 inputs, and Wang, Fok & Thing's 28 inputs include `Offset`. See M_NIDD's baseline_source below.

## Notes and limitations

NETWORK-LAYER flow classification on real 5G traffic. This is a DIFFERENT TASK from the QKD-layer attack classifier on this same page, with zero overlap in label space or feature space, so neither score is the other's baseline. REFITTED IN ROUND 86 ON CLEANED ROWS AND WITHOUT THE FILE LABEL, the two facts that explain every number here. The published file holds one attacker's UDP flood twice, the same 281,529 flows labelled Benign in one base station's file and UDPFlood in the other's, and the Benign copies are removed. From Round 50 the model also read `IdleTime`, which names the file a flow came from, not anything the flow did. ACCURACY IS 0.9979 AND MACRO F1 0.9972 on 934,361 flows. On the rows as published the same fit scores accuracy 0.7616, no model reading these 46 columns can pass 0.7685 there, and training on one base station and testing on the other gives 0.42 and 0.61. On the cleaned rows that test gives 0.9973 and 0.9877 (`cross_base_station`). A LINEAR MODEL ON THE NUMERIC COLUMNS REACHES ACCURACY 0.9838, because each attack class was produced by its own tool against its own service. Two weak spots remain: trained on the second base station, SYNFlood on the first scores F1 0.22, and without the TTL and hop columns UDPScan across base stations falls to F1 0.05, since nmap's raw probes carry a TTL of their own. THE LEAK COUNTERFACTUALS ARE RECORDED EVERY RUN. Restoring the corpus's integer row identifiers or the file label takes accuracy to about 1.0000, a shortcut of the kind Engelen et al. (WTMC 2021) found in CIC-IDS2017. See M_NIDD's baseline_source for what that means for the published figures on these rows. NO STUDENT IS DISTILLED FROM THIS MODEL: the page's encrypted tier runs telecom_ids.student, distilled from the QKD-attack random forest (telecom_ids.qml).

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
