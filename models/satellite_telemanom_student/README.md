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
a quantum-safe orchestration demo.
This is the **encrypted-tier** model for the `satellite_comms` use case: it is a
distilled student small enough to evaluate under real CKKS, and the demo runs it
that way. It is not the plaintext teacher, and its numbers should be read against
that teacher rather than against the published literature.

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
- **Baseline to beat**: 0.968 - ESA-ADB Supplementary Table 9, Mission1 lightweight (ch. 41-46), Telemanom-ESA-Pruned: corrected event-wise F0.5 0.968 (precision 0.999, recall 0.862) for detection of ANOMALIES ALONE - which is what this project scores, so this is the like-for-like row. Earlier rounds cited Table 2's 0.786 instead. That number is real and correctly attributed, but Table 2 scores ALL EVENTS excluding communication gaps, i.e. anomalies PLUS annotated rare nominal events, so it was the wrong comparator and the lenient one. Verified against the paper (arXiv:2406.17826 v2) in Round 24. THREE protocol differences remain and none of them are closed by changing the number: (1) ESA-ADB's metric multiplies event precision by the TIME-LEVEL true-negative rate, which is why most of its baselines read <0.001 - this project records that corrected value as `corrected_event_wise_f0_5` beside the uncorrected one; (2) ESA-ADB splits each mission in half, this project tests on the last 30 percent; (3) ESA-ADB's Mission1 lightweight test partition holds 65 annotated events of which 29 are anomalies, against 3 in the window scored here. So this remains an indicative target, and the sample-size gap is roughly tenfold.
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
