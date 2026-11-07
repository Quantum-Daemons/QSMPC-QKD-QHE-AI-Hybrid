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

# CKKS-linear telemetry anomaly student

logistic regression over channel x timestep windows

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
| `agreement` | 0.999192 |
| `disagreement_on_positives` | 0.184615 |
| `metric_delta` | 0.961026 |
| `metric_delta_pp` | 96.1026 |
| `n_eval` | 2844581 |
| `n_scoring_units` | 12 |
| `student_metric` | 0.0 |
| `student_params` | 49 |
| `teacher_metric` | 0.961026 |
| `teacher_params` | 84860 |

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

The one published model with no browser evaluation sample, and the reason is the metric rather than the weights: event-wise F0.5 is defined over contiguous runs of a timeline and cannot be computed from a shuffled set of independent rows, which is what the browser evaluator scores. It needs no scaler of its own: scripts/train/satellite_telemanom.py fits it on the forecaster's z-scored windows (the last 8 timesteps x 6 channels), so the absent `input` block in its weights.json is correct rather than the omission it looks like. NOT REPUBLISHED SINCE THE ROUND 86 RETRAIN, and to be replaced. Against the retrained forecaster this logistic student scores far below the teacher on its evaluation slice (the record's student_metric and teacher_metric) and raises far fewer alarms. A linear map over a window cannot reproduce a forecast-error threshold, so the replacement will be a linear next-step forecaster evaluated under CKKS, whose decrypted forecast the client thresholds itself (the owner's choice, 2026-11-07). The Hub copy is the student of the Round 71 model. No code path here evaluates this student under CKKS today: the satellite page has no private-inference stage, and the full deployment's telemetry stage encrypts the OPS-SAT-AD demo regression the browser scores in plaintext.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
