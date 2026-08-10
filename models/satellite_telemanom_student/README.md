---
license: other
license_name: cc-by-3.0-igo
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

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo. This is the **plaintext** model for the
`satellite_comms` use case and it runs under real CKKS encryption.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.998677 |
| `disagreement_on_positives` | 0.142857 |
| `metric_delta` | -0.003234 |
| `metric_delta_pp` | -0.3234 |
| `n_eval` | 307601 |
| `n_scoring_units` | 2 |
| `student_metric` | 0.003234 |
| `student_params` | 49 |
| `teacher_metric` | 0.0 |
| `teacher_params` | 80486 |

## Published baselines this is measured against

- **Target metric**: event-wise F0.5
- **Baseline to beat**: 0.786 - ESA-ADB Table 2, Mission1 lightweight (ch. 41-46), best event-wise F0.5. NOTE the caption: Table 2 scores detection of ALL EVENTS excluding communication gaps - anomalies PLUS annotated rare nominal events - whereas this project scores anomalies only, and ESA-ADB splits each mission into first-half train / second-half test. So 0.786 is an indicative target, not a like-for-like number.
- **Companion metric shown alongside**: `event_wise_f0_5_full_channels` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46
- **Licence**: CC-BY-3.0-IGO
- **Source**: https://zenodo.org/records/12528696 (licence read 2026-08-03)

Missions are anonymised Mission1/2/3 — they are not named Sentinel anything. 17.5 years, 1.19% annotated. Mission3 is excluded upstream for triviality.

## Notes and limitations

The one published model with no browser evaluation sample, and the reason is the metric rather than the weights. Round 21 gave the other three students samples; this one is scored on event-wise F0.5, which is defined over contiguous runs of a timeline and cannot be computed from a shuffled set of independent rows -- which is exactly what the browser evaluator scores. Shipping a per-row metric instead would verify a number nobody published for it. It needs no scaler, unlike the other three: scripts/train/satellite_telemanom.py fits it on raw `flat_te` (the last 8 timesteps x 6 channels), so the absent `input` block in its weights.json is correct rather than the omission it looks like. Worth reading alongside its numbers: the teacher scores 0.0 event-wise F0.5 on this split and the student 0.0032, so agreement of 0.9987 is agreement about finding almost nothing.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
