---
license: other
license_name: cc-by-4.0
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

gradient-boosted trees over 39 flow-statistic features

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **plaintext** model for the `telecom_ids` use case; the encrypted
path runs a distilled student, not this model.

## Measured performance

| metric | value |
|---|---|
| `aggregate_accuracy` | 0.760946 |
| `leak_counterfactual_accuracy` | 0.999724 |
| `leak_counterfactual_accuracy_gain` | 0.238778 |
| `leak_counterfactual_macro_f1` | 0.999494 |
| `macro_f1` | 0.926439 |
| `majority_baseline` | 0.392911 |
| `n_classes` | 9 |
| `n_classes_predicted` | 9 |
| `n_eval` | 303973 |
| `n_features` | 39 |
| `n_samples` | 1215890 |
| `wall_clock_s` | 93.9 |

## Published baselines this is measured against

- **Target metric**: macro F1 across 9 network-attack classes
- **Baseline**: Two published figures on 5G-NIDD are usually quoted - Wang, Fok & Thing report 99.61% accuracy for a CNN and 99.71% for a hybrid quantum-classical Quan-ConvCNN - and BOTH are accuracies, which is why no number appears in `baseline`. Three caveats, and the third is new and material. (1) Those are hybrid quantum-classical results with a 0.10-point spread, and the authors' own claim is about generalisation to UNSEEN attacks rather than about the headline; `references/REFERENCES.md` WS2 records this. (2) Accuracy is the wrong comparator on a corpus whose two majority classes are 39.3% and 37.6% of the rows. (3) MEASURED HERE: an identical gradient-boosted model reaches accuracy 0.9997 with three integer row counters restored and 0.7610 without them, so the published 99.6-99.7% range is reproducible from row identifiers alone. That does NOT establish that those papers used them - their feature set is not stated in what was read - so it is recorded as an open question about the comparison rather than as a refutation of the papers. Either way it means these figures are an anchor and not a target that has been beaten or missed.
- **Companion metric shown alongside**: `aggregate_accuracy` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

There is no baseline number to have a gap to, and that is a deliberate registry decision rather than a missing measurement: the two figures usually quoted for this corpus (99.61% for a CNN, 99.71% for a hybrid quantum-classical Quan-ConvCNN, Wang/Fok/Thing) are ACCURACIES, and this model is judged on macro F1, so putting them in `baseline` would render an accuracy beside a macro F1 as though the two were the same scale. What can be said is measured here and recorded above. Fitting this identical configuration with the corpus's three integer row identifiers restored - `Unnamed: 0`, `Seq`, `Offset`, all three of which trip this project's own single-feature leak threshold on their own - reproduces that 99.6-99.7% range. Removing them costs roughly 24 points of accuracy. The published figures are therefore an anchor rather than a target that has been beaten or missed. THE LIMIT OF THAT CLAIM: it shows the range is reachable from row counters on this corpus. It does NOT show the cited papers used them, because their feature set is not stated in what was read. This is an open question about the comparison, not a refutation of the papers.

## Training data

- **Dataset**: 5G-NIDD (Combined.csv)
- **Licence**: CC-BY-4.0
- **Source**: https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361 (licence read 2026-08-16)
- **Attribution (CC-BY-4.0)**: Y. Siriwardhana et al., 'Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)', in IEEE Data Descriptions, doi:10.1109/IEEEDATA.2025.3592888. Dataset: doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361, CC BY 4.0.

1,215,890 real 5G network flows x 52 columns, counted in this file rather than quoted, 9 classes: Benign 477,737 | UDPFlood 457,340 | HTTPFlood 140,812 | SlowrateDoS 73,124 | TCPConnectScan 20,052 | SYNScan 20,043 | UDPScan 15,906 | SYNFlood 9,721 | ICMPFlood 1,155. LICENCE VERIFIED AT THE SOURCE, not at a mirror: the Metax record for dataset 9d13ef28-2ca7-44b0-9950-225359afac65 reads access_type Open, license CC-BY-4.0 and restriction_grounds []. The IEEE DataPort listing whose subscription wall this project originally read as a licence restriction is a MIRROR; the Fairdata record is the authors' repository of record. THIRTEEN OF THE 52 COLUMNS ARE UNUSABLE and the loader drops them with reasons attached (qsh.models.datasets.NIDD_DROPPED, republished in the model's feature_spec.json): 3 labels (`Attack Type`, its binary collapse `Label`, and `Attack Tool` which scores accuracy 0.9615 on its own), 3 row identifiers, 5 columns bit-identical to `Dur` on every row (`RunTime`/`Mean`/`Sum`/`Min`/`Max`), and 2 constants (`sVid`/`dVid`, one distinct value each). 39 features remain. THE ROW IDENTIFIERS ARE THE FINDING. `Unnamed: 0`, `Seq` and `Offset` are counters, and each trips this project's own single-feature leak threshold on its own: one-vs-rest AUC 0.9990 (SYNScan), 0.9275 and 0.9502 (both ICMPFlood). Restoring all three to the same model moves accuracy from 0.7610 to 0.9997. What that does and does NOT establish is set out on the ModelSpec below.

## Notes and limitations

NETWORK-LAYER flow classification on real 5G traffic. This is a DIFFERENT TASK from the QKD-layer attack classifier on this same page, with zero overlap in label space or feature space, so neither score is the other's baseline and 0.93 here is not 'worse' than 0.99 there. READ THE TWO NUMBERS THE OTHER WAY ROUND FROM EVERYWHERE ELSE IN THIS PROJECT. Every other model here reports an aggregate that flatters and a per-class figure that does not. On this corpus the inversion holds: macro F1 is the OPTIMISTIC number and aggregate accuracy the pessimistic one, because seven minority classes score 0.98-1.00 while the two majority classes (39.3% and 37.6% of the rows) mutually confuse. The residual is one confusion pair, Benign <-> UDPFlood, not diffuse weakness. THE REASON THIS MODEL IS WORTH SHIPPING is the counterfactual recorded beside its metrics every run: refitting the identical configuration with the corpus's three integer row identifiers restored moves accuracy from 0.7610 to 0.9997 and macro F1 from 0.9266 to 0.9999. That is the same shape of result Engelen et al. (WTMC 2021) produced for CIC-IDS2017, which this repository already cites as its reason for reporting per-class F1 first. It is an open question about how the published 99.6-99.7% figures were obtained, NOT a demonstration that they were obtained that way -- see M_NIDD's baseline_source, which states the limit of the claim.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
