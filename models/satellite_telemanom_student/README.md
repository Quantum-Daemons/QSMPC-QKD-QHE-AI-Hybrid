---
license: other
license_name: cc-by-3.0-igo
license_link: https://creativecommons.org/licenses/by/3.0/igo/
library_name: onnx
pipeline_tag: time-series-forecasting
tags:
  - quantum-safe
  - privacy-preserving
  - satellite-comms
  - onnx
  - onnxruntime-web
---

# CKKS-linear telemetry forecaster student

linear next-step forecaster over the last 8 rows x 6 channels (288 weights + 6 biases), evaluated under CKKS; the client thresholds the decrypted forecast with Telemanom's procedure

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **encrypted-tier** model for the `satellite_comms` use case: a
distilled student small enough to evaluate under CKKS. Whether the public demo
encrypts THIS model differs by use case; the notes below say what actually runs.
It is not the plaintext teacher, and its numbers should be read against that
teacher rather than against the published literature. Its agreement with the
teacher was measured offline, in plaintext.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.999624 |
| `disagreement_on_positives` | 0.024323 |
| `floor` | 0.0105 |
| `forecast_mae_student_to_teacher` | 0.126269 |
| `k_rows` | 8 |
| `metric_delta` | 0.010171 |
| `metric_delta_pp` | 1.0171 |
| `n_eval` | 7364160 |
| `n_scoring_units` | 29 |
| `n_unlabelled_windows` | 7208289 |
| `ridge_alpha` | 1000000.0 |
| `student_metric` | 0.93977 |
| `student_params` | 294 |
| `teacher_metric` | 0.949941 |
| `teacher_params` | 84860 |
| `wall_clock_s` | 354.1 |

## Published baselines this is measured against

- **Target metric**: corrected event-wise F0.5 (ESA-ADB)
- **Baseline to beat**: 0.968 - ESA-ADB Supplementary Table 9 (arXiv:2406.17826 v2; Table 18 of the DMLR 2026 version, read 2026-11-07), Mission1 lightweight (ch. 41-46), Telemanom-ESA-Pruned: corrected event-wise F0.5 0.968 (precision 0.999, recall 0.862) for ANOMALIES ALONE, which is what this project scores, so it is the like-for-like row. Earlier rounds cited Table 2's 0.786, which scores all events but communication gaps, anomalies PLUS annotated rare nominal events: a real number, and the lenient row. Since Round 86 the model is trained, thresholded and scored by ESA-ADB's own recipe and rules (scripts/train/satellite_telemanom.py): its Mission1 split, the 30 s zero-order hold, its scaling and training windows, Telemanom's thresholding as ESA-ADB's TelemanomThresholding applies it, and a reimplementation of its ESAScores over the test half's 29 anomalies. Two choices behind the published figure are not copied into this project's own result, and the record carries both: its floor of 0.05/7 is not shown to have been chosen without the test half, so the floor here is chosen on the validation months; and ESA-ADB's prepared series reads the labels to write annotated samples the zero-order hold dropped back onto the grid. Applying both reproduces the published figure (esa_scores.restored_annotated_samples in the record).

## Training data

- **Dataset**: ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46
- **Licence**: CC-BY-3.0-IGO
- **Source**: https://zenodo.org/records/12528696 (licence read 2026-08-03)
- **Attribution (CC-BY-3.0-IGO)**: Kotowski et al., 'European Space Agency Dataset and Benchmark for Real-World Anomaly Detection in Spacecraft Time Series', Journal of Data-centric Machine Learning Research (DMLR), 2026 (earlier version: arXiv:2406.17826). Data: De Canio, Kotowski & Haskamp, 'ESA Anomaly Dataset', Zenodo record 12528696 (2024), CC BY 3.0 IGO.

Missions are anonymised Mission1/2/3 - they are not named Sentinel anything. Mission1 holds 14 years of telemetry and Mission2 3.5 (anonymised durations, arXiv:2406.17826v2 Table 1; the paper sizes the corpus by data points instead, because sampling rates vary and the timeline is anonymised). 1.19% of points annotated across Mission1 and Mission2 (1.80% in Mission1). Mission3 is excluded upstream for triviality. The CC BY 3.0 IGO term appears on the Zenodo landing page but NOT in its REST metadata, which returns no licence at all - an API-only check reads this corpus as unlicensed.

## Notes and limitations

A LINEAR NEXT-STEP FORECASTER SINCE ROUND 86, trained by scripts/train/satellite_student_forecaster.py from the published teacher. The logistic student before it classified the teacher's flags from 8 rows and could not follow the retrained teacher, because a threshold on a forecast error is not linear in the window. A forecast is close to linear in the last rows, and Telemanom's thresholding needs nothing else. So the encrypted part is the forecast: six dot products with the last 8 z-scored rows, ridge-fitted to the teacher's own forecasts on its unlabelled training windows. The client decrypts the forecast and thresholds it with the teacher's procedure at the teacher's floor, in plaintext. The record holds the pipeline's event scores beside the teacher's over the whole test half, the forecast errors of both against persistence, and the same model fitted to the truth instead of the teacher. The page scores this student end to end on its published stretch: its forecasts in plaintext and under CKKS (encryptedLinearBatch), Telemanom's thresholding in a TypeScript port checked flag for flag against the Python, and ESA-ADB's scorer in a port checked against the Python. Its weights.json carries the z-score constants, because the thresholding works in raw units.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
