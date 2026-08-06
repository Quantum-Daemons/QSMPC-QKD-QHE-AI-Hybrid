---
license: other
license_name: cc-by-4.0
library_name: onnx
pipeline_tag: time-series-forecasting
tags:
  - quantum-safe
  - privacy-preserving
  - satellite-comms
  - onnx
  - onnxruntime-web
---

# OPS-SAT segment anomaly detector

gradient-boosted trees over 19 per-segment telemetry features

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo. This is the **plaintext** model for the
`satellite_comms` use case; the encrypted path runs a distilled student, not this model.

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
| `wall_clock_s` | 1.7 |

## Published baselines this is measured against

- **Target metric**: AUC PR
- **Baseline to beat**: 0.979 - FCNN, best of the 30 algorithms benchmarked in Kotowski et al., Scientific Data 12 (2025), on this dataset's own test split (AUC PR 0.979, F1 0.946, MCC 0.932). XGBOD reaches 0.975 and RF+ICCS 0.963.
- **Published ceiling**: 0.979 - FCNN, the best of the 30 published baselines - there is no higher number to cite.
- **Companion metric shown alongside**: `f1` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: OPS-SAT-AD
- **Licence**: CC-BY-4.0
- **Source**: https://zenodo.org/records/12588359 (licence read 2026-08-05)

2,123 annotated telemetry segments across 9 channels from ESA's OPS-SAT CubeSat, 20.4% anomalous, with 19 precomputed per-segment features and the benchmark's own train/test split. Kotowski et al., Scientific Data 12 (2025). CC BY 4.0 and 507 KB, so unlike every other corpus here the DATA can be redistributed as well as the weights. A different task from ESA-ADB: segment classification, not forecasting, so its score is not comparable with the Telemanom model's and neither is the other's baseline.

## Notes and limitations

Segment CLASSIFICATION on real OPS-SAT telemetry, which is a different task from the Telemanom forecaster above: this one is scored against the 30 algorithms published with the dataset, not against persistence, and the two numbers must not be read as a comparison. Chosen because its corpus is CC BY 4.0 and 507 KB - the one case in this project where the data itself can be redistributed alongside the weights.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
