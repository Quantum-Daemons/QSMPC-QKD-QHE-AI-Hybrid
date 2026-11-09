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
| `aggregate_accuracy` | 0.761558 |
| `leak_counterfactual_accuracy` | 0.999727 |
| `leak_counterfactual_accuracy_gain` | 0.238169 |
| `leak_counterfactual_macro_f1` | 0.999481 |
| `macro_f1` | 0.927503 |
| `majority_baseline` | 0.392911 |
| `n_classes` | 9 |
| `n_classes_predicted` | 9 |
| `n_eval` | 303973 |
| `n_features` | 46 |
| `n_samples` | 1215890 |
| `wall_clock_s` | 73.8 |

## Published baselines this is measured against

- **Target metric**: macro F1 across 9 network-attack classes
- **Baseline**: Two published figures on 5G-NIDD are usually quoted - Wang, Fok & Thing report 99.61% accuracy for a classical CNN and 99.71% for a hybrid quantum-classical Quan-ConvCNN - and BOTH are accuracies on the BINARY benign-versus-malicious task, while this model is scored on 9 classes, which is why no number appears in `baseline`. The 9-class anchors are the dataset paper's random forest at 99.48% accuracy on its top-10 features (Samarakoon et al., arXiv:2212.01298) and BiTAD (Future Internet 17(11):482) at 90.47%. Caveats. (1) The Wang, Fok & Thing authors' own claim is about generalisation to UNSEEN attacks rather than about the headline; `references/REFERENCES.md` WS2 records this. (2) Accuracy is the wrong comparator on a corpus whose two majority classes are 39.3% and 37.6% of the rows. (3) MEASURED HERE: an identical gradient-boosted model reaches accuracy 0.9997 with the corpus's integer row counters restored, 0.9829 with the capture-time column `IdleTime` restored, and 0.7616 with every identifier, every host fingerprint and the capture time dropped. (4) Whether published figures lean on row identifiers or capture time is OPEN: those top-10 inputs include the Argus sequence number `Seq` and `Offset`, Wang, Fok & Thing's inputs include `Offset`, both columns are dropped here as identifiers, and no published ablation settles it. (5) This model's figures come from a stratified random split, which flatters: when Wang, Fok & Thing train on one base station and test on the other, their accuracies fall from above 98% to 78.2-78.9% trained on base station 1 and 57.2-81.6% trained on base station 2 (their Tables V and VI), where the classical CNN, at 74.28%, beats four of the five hybrids. These figures are an anchor and not a target that has been beaten or missed.
- **Companion metric shown alongside**: `aggregate_accuracy` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

There is no baseline number to have a gap to, and that is a deliberate registry decision rather than a missing measurement: the two figures usually quoted for this corpus (99.61% for a classical CNN, 99.71% for a hybrid quantum-classical Quan-ConvCNN, Wang/Fok/Thing) are ACCURACIES on the BINARY benign-versus-malicious task, and this model is judged on 9-class macro F1, so putting them in `baseline` would render an accuracy beside a macro F1 as though the two were the same scale. WHAT THE COMPARISON CAN AND CANNOT SAY. This model scores about 0.7616 9-class accuracy with every row identifier, every host fingerprint and the capture time dropped. The 9-class anchors are the dataset paper's random forest at 99.48% accuracy on its top-10 features (Samarakoon et al., arXiv:2212.01298) and BiTAD (Future Internet 17(11) 482) at 0.9047; this model sits below them, on its own stratified random split rather than a shared one, so each is an anchor and not a head-to-head. WHAT THE COUNTERFACTUALS SHOW. Restoring the corpus's integer row counters takes accuracy to about 0.9997, recorded every run as `leak_counterfactual_accuracy`, and restoring the capture time to about 0.9829, recorded as `capture_time_counterfactual`: that is what a bookkeeping column is worth on this corpus. They do not show what the published figures used. Whether the published figures lean on such columns is OPEN: the dataset paper's top-10 inputs include the sequence number `Seq` and `Offset`, Wang/Fok/Thing's inputs include `Offset`, both columns are dropped here, and no published ablation settles it.

## Training data

- **Dataset**: 5G-NIDD (BTS1_BTS2_fields_preserved.csv)
- **Licence**: CC-BY-4.0
- **Source**: https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361 (licence read 2026-08-16)
- **Attribution (CC-BY-4.0)**: Y. Siriwardhana et al., 'Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)', in IEEE Data Descriptions, doi:10.1109/IEEEDATA.2025.3592888. Dataset: doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361, CC BY 4.0.

1,215,890 real 5G network flows x 113 columns, counted in this file rather than quoted, 9 classes: Benign 477,737 | UDPFlood 457,340 | HTTPFlood 140,812 | SlowrateDoS 73,124 | TCPConnectScan 20,052 | SYNScan 20,043 | UDPScan 15,906 | SYNFlood 9,721 | ICMPFlood 1,155. LICENCE VERIFIED AT THE SOURCE, not at a mirror: the Metax record for dataset 9d13ef28-2ca7-44b0-9950-225359afac65 reads access_type Open, license CC-BY-4.0 and restriction_grounds []. The IEEE DataPort listing whose subscription wall this project originally read as a licence restriction is a MIRROR; the Fairdata record is the authors' repository of record. THE MOST IMPORTANT LINE HERE IS ABOUT ONE COLUMN. This is `BTS1_BTS2_fields_preserved.csv`, 113 columns. The release also ships `Combined.csv` at 52 columns, which is what this project trained on until Round 50. Same rows, same classes, same distribution - and all 52 of Combined's columns are informationally identical in the wide file, equal distinct-value counts on every one, so Combined is a strict column SUBSET. It omits `IdleTime`, a column worth 22 accuracy points (0.9829 with it, 0.7616 without) because it names the capture window: its 17 values are all multiples of 128 and, as Unix times, fall between 13 and 15 September 2022, each dominated by one attack scenario and benign traffic. It is dropped, as `StartTime` is. Five other columns Combined lacks (`Dir`, `Flgs`, `TcpOpt`, `sVlan`, `dVlan`) are worth 0.0006 accuracy between them - which is redundancy rather than emptiness, since `Dir` alone separates ICMPFlood at one-vs-rest AUC 0.9811 and `TcpOpt` separates SlowrateDoS at 0.9433. The other 41 columns already carry what they carry. SIXTY-SEVEN OF THE 113 COLUMNS ARE UNUSABLE and the loader drops them with reasons attached (qsh.models.datasets.NIDD_DROPPED, republished in the model's feature_spec.json): 3 labels (`Attack Type`, its binary collapse `Label`, and `Attack Tool` which scores accuracy 0.9615 on its own), 9 row identifiers, 10 host fingerprints (addresses, MACs, ports, IP ids), 5 columns bit-identical to `Dur` on every row (`RunTime`/`Mean`/`Sum`/`Min`/`Max`), the capture time, and 39 that hold exactly one state. 46 features remain. THE 39 SINGLE-STATE COLUMNS ARE WORTH A SEPARATE SENTENCE because their names promise signal: 30 of them are 100% NULL in this release, and that includes every SIntPkt/DIntPkt inter-packet-time column and every jitter column. They are empty, not weak. THE ROW IDENTIFIERS REMAIN A FINDING, with a narrower claim than Round 49 made. `Unnamed: 0` is the CSV's own index and restoring it alone takes accuracy to 0.9997. That is what a row counter is worth on this corpus. Whether published figures lean on such columns is an OPEN question: the dataset paper's 9-class baselines (Samarakoon et al., arXiv:2212.01298) rank the Argus sequence number `Seq` and `Offset` among their top-10 inputs, and Wang, Fok & Thing's 28 inputs include `Offset`, both columns this project drops as row identifiers; no published ablation settles it. See M_NIDD's baseline_source below.

## Notes and limitations

NETWORK-LAYER flow classification on real 5G traffic. This is a DIFFERENT TASK from the QKD-layer attack classifier on this same page, with zero overlap in label space or feature space, so neither score is the other's baseline. REFITTED IN ROUND 86 WITHOUT THE CAPTURE TIME, the fact that explains every number here. From Round 50 it also read `IdleTime` and scored accuracy 0.9829. That column holds 17 values, all multiples of 128, which as Unix times fall between 13 and 15 September 2022 and are each dominated by one attack scenario: it names when a flow was captured, as `StartTime` does, which is dropped for that reason. Without it the model scores 0.7616, and `capture_time_counterfactual` re-measures the difference on every run. READ THE TWO NUMBERS THE OTHER WAY ROUND FROM EVERYWHERE ELSE IN THIS PROJECT - STILL. Every other model here reports an aggregate that flatters and a per-class figure that does not. On this corpus macro F1 is the OPTIMISTIC number and aggregate accuracy the pessimistic one: 0.9275 macro F1 against 0.7616 accuracy. Benign and UDPFlood are 39.3% and 37.6% of the rows and are the only pair that confuses, so accuracy is dominated by them while the macro average is carried by seven classes at 0.9875 or better, and the two lowest per-class F1 are Benign at 0.6401 and UDPFlood at 0.7308. THE LEAK COUNTERFACTUAL IS STILL RECORDED EVERY RUN and its claim is now narrower. Restoring the corpus's integer row identifiers takes accuracy to about 0.9997, a shortcut of the kind Engelen et al. (WTMC 2021) found in CIC-IDS2017. That shows what a row counter is worth on this corpus. It does not show what the published figures used; whether those figures lean on identifiers or on capture time stays open -- see M_NIDD's baseline_source. NO STUDENT IS DISTILLED FROM THIS MODEL: the page's encrypted tier runs telecom_ids.student, distilled from the QKD-attack random forest (telecom_ids.qml).

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
