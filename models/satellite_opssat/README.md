---
license: other
license_name: cc-by-4.0
library_name: onnx
pipeline_tag: tabular-classification
tags:
  - quantum-safe
  - privacy-preserving
  - satellite-comms
  - onnx
  - onnxruntime-web
---

# OPS-SAT segment anomaly detector

gradient-boosted trees over 18 published per-segment features + `sampling`

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **plaintext** model for the `satellite_comms` use case; the encrypted
path runs a distilled student, not this model.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.9659735349716446 |
| `anomaly_rate_test` | 0.213611 |
| `auc_pr` | 0.968107105684616 |
| `auc_pr_baseline` | 0.21361058601134217 |
| `auroc` | 0.9881509530292716 |
| `f1` | 0.918918918918919 |
| `n_features` | 19 |
| `n_test_segments` | 529 |
| `n_train_segments` | 1594 |
| `wall_clock_s` | 7.5 |

## Published baselines this is measured against

- **Target metric**: AUC PR
- **Baseline to beat**: 0.979 - FCNN, best of the 30 algorithms benchmarked in Ruszczak et al., Scientific Data 12 (2025), on this dataset's own test split (AUC PR 0.979, F1 0.946, MCC 0.932). XGBOD reaches 0.975 and RF+ICCS 0.963.
- **Published ceiling**: 0.979 - FCNN, the best of the 30 published baselines - there is no higher number to cite.
- **Companion metric shown alongside**: `f1` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: OPS-SAT-AD
- **Licence**: CC-BY-4.0
- **Source**: https://doi.org/10.5281/zenodo.12588358 (licence read 2026-08-05)
- **Attribution (CC-BY-4.0)**: Ruszczak, Kotowski, Evans & Nalepa, 'The OPS-SAT benchmark for detecting anomalies in satellite telemetry', Scientific Data 12, 710 (2025), doi:10.1038/s41597-025-05035-3, CC BY 4.0.

2,123 annotated telemetry segments across 9 channels from ESA's OPS-SAT CubeSat, 20.4% anomalous, with 18 published per-segment features plus a `sampling` metadata column, and the benchmark's own train/test split. That split is stratified random, not temporal, so scores on it do not measure generalisation over time. Ruszczak et al., Scientific Data 12 (2025). CC BY 4.0, which permits redistributing the data with credit; the per-segment feature table is about 507 KB, the raw segment file much larger. A different task from ESA-ADB: segment classification, not forecasting, so its score is not comparable with the Telemanom model's and neither is the other's baseline.

## Notes and limitations

Segment CLASSIFICATION on real OPS-SAT telemetry, which is a different task from the Telemanom forecaster above: this one is scored against the 30 algorithms published with the dataset, not against persistence, and the two numbers must not be read as a comparison. Chosen because its corpus is CC BY 4.0 and small, so the whole test partition can be redistributed alongside the weights. The benchmark's split is stratified random, so these scores do not measure generalisation over time. The card's metrics are the scikit-learn model's before export; the published ONNX graph scores accuracy 0.9509 on the same 529 test segments against 0.9660 recorded here, so the browser's re-score reproduces the graph, not the card.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
