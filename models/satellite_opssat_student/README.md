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

# CKKS-linear OPS-SAT segment student

logistic regression over 18 published per-segment features + `sampling`

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **encrypted-tier** model for the `satellite_comms` use case: it is a
distilled student small enough to evaluate under real CKKS, and the demo runs it
that way. It is not the plaintext teacher, and its numbers should be read against
that teacher rather than against the published literature.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.958412 |
| `auc_pr` | 0.968055 |
| `disagreement_on_positives` | 0.070796 |
| `metric_delta` | 0.014571 |
| `metric_delta_pp` | 1.4571 |
| `n_eval` | 529 |
| `student_metric` | 0.904348 |
| `student_params` | 210 |
| `teacher_metric` | 0.918919 |
| `teacher_params` | 117 |

## Published baselines this is measured against

- **Target metric**: AUC PR
- **Baseline to beat**: 0.979 - FCNN, best of the 30 algorithms benchmarked in Ruszczak et al., Scientific Data 12 (2025), on this dataset's own test split (AUC PR 0.979, F1 0.946, MCC 0.932). XGBOD reaches 0.975 and RF+ICCS 0.963.
- **Published ceiling**: 0.979 - FCNN, the best of the 30 published baselines - there is no higher number to cite.
- **Companion metric shown alongside**: `f1` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: OPS-SAT-AD
- **Licence**: CC-BY-4.0
- **Source**: https://doi.org/10.5281/zenodo.12588358 (licence read 2026-08-05)

2,123 annotated telemetry segments across 9 channels from ESA's OPS-SAT CubeSat, 20.4% anomalous, with 18 published per-segment features plus a `sampling` metadata column, and the benchmark's own train/test split. Ruszczak et al., Scientific Data 12 (2025). CC BY 4.0 and 507 KB, so unlike every other corpus here the DATA can be redistributed as well as the weights. A different task from ESA-ADB: segment classification, not forecasting, so its score is not comparable with the Telemanom model's and neither is the other's baseline.

## Notes and limitations

19 columns is small enough to evaluate under CKKS as a single packed dot product, so the encrypted tier here runs on the published feature set rather than on a reduction of it - and it had room to spare. The student now fits over the QUADRATIC expansion of those features, 209 terms, still one packed dot product at unchanged multiplicative depth, because the expansion is a function of the plaintext input and is computed before encryption. A degree-8192 ring has 4096 slots, so the cipher was never the binding constraint here - the feature map was. Measured: minority-class F1 0.8649 -> 0.9043, AUC-PR 0.9275 -> 0.9681, the latter matching the teacher exactly.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
