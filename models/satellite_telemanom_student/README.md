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
This is the **encrypted-tier** model for the `satellite_comms` use case: a
distilled student small enough to evaluate under CKKS. Whether the public demo
encrypts THIS model differs by use case; the notes below say what actually runs.
It is not the plaintext teacher, and its numbers should be read against that
teacher rather than against the published literature. Its agreement with the
teacher was measured offline, in plaintext.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.975641 |
| `disagreement_on_positives` | 0.0 |
| `metric_delta` | 0.0 |
| `metric_delta_pp` | 0.0 |
| `n_eval` | 307601 |
| `n_scoring_units` | 2 |
| `student_metric` | 0.0 |
| `student_params` | 49 |
| `teacher_metric` | 0.0 |
| `teacher_params` | 80486 |

## Published baselines this is measured against

- **Target metric**: event-wise F0.5
- **Baseline to beat**: 0.968 - ESA-ADB Supplementary Table 9, Mission1 lightweight (ch. 41-46), Telemanom-ESA-Pruned: corrected event-wise F0.5 0.968 (precision 0.999, recall 0.862) for detection of ANOMALIES ALONE - which is what this project scores, so this is the like-for-like row. Earlier rounds cited Table 2's 0.786 instead. That number is real and correctly attributed, but Table 2 scores ALL EVENTS excluding communication gaps, i.e. anomalies PLUS annotated rare nominal events, so it was the wrong comparator and the lenient one. Verified against the paper (arXiv:2406.17826 v2) in Round 24. FOUR protocol differences remain and none of them are closed by changing the number: (1) ESA-ADB's metric multiplies event precision by the TIME-LEVEL true-negative rate, which is why most of its baselines read <0.001 - this project records that corrected value as `corrected_event_wise_f0_5` beside the uncorrected one; (2) ESA-ADB splits each mission in half, this project tests on the last 30 percent; (3) ESA-ADB's Mission1 lightweight test partition holds 65 annotated events of which 29 are anomalies, against 3 in the window scored here; (4) ESA-ADB resamples Mission1 at 0.033 Hz, about one sample per 30 s, chosen from the dominant rate of channels 41-46, while this model resamples to 300 s, so its 250-step window spans about 20.8 hours where the benchmark's would span about 2.1, and short events can vanish. So this remains an indicative target, and the sample-size gap is roughly tenfold.
- **Companion metric shown alongside**: `event_wise_f0_5_full_channels` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46
- **Licence**: CC-BY-3.0-IGO
- **Source**: https://zenodo.org/records/12528696 (licence read 2026-08-03)
- **Attribution (CC-BY-3.0-IGO)**: Kotowski et al., 'European Space Agency Dataset and Benchmark for Real-World Anomaly Detection in Spacecraft Time Series', Journal of Data-centric Machine Learning Research (DMLR), 2026 (earlier version: arXiv:2406.17826). Data: De Canio, Kotowski & Haskamp, 'ESA Anomaly Dataset', Zenodo record 12528696 (2024), CC BY 3.0 IGO.

Missions are anonymised Mission1/2/3 - they are not named Sentinel anything. Mission1 holds 14 years of telemetry and Mission2 3.5 (anonymised durations, arXiv:2406.17826v2 Table 1; the paper sizes the corpus by data points instead, because sampling rates vary and the timeline is anonymised). 1.19% of points annotated across Mission1 and Mission2 (1.80% in Mission1). Mission3 is excluded upstream for triviality. The CC BY 3.0 IGO term appears on the Zenodo landing page but NOT in its REST metadata, which returns no licence at all - an API-only check reads this corpus as unlicensed.

## Notes and limitations

The one published model with no browser evaluation sample, and the reason is the metric rather than the weights. Round 21 gave the other three students samples; this one is scored on event-wise F0.5, which is defined over contiguous runs of a timeline and cannot be computed from a shuffled set of independent rows -- which is exactly what the browser evaluator scores. Shipping a per-row metric instead would verify a number nobody published for it. It needs no scaler, unlike the other three: scripts/train/satellite_telemanom.py fits it on raw `flat_te` (the last 8 timesteps x 6 channels), so the absent `input` block in its weights.json is correct rather than the omission it looks like. Worth reading alongside its numbers: since the Round 71 retrain the teacher and the student both score 0.0 event-wise F0.5 on this split, so their agreement is agreement about finding nothing, and there is no fidelity claim to make. No code path here evaluates this student under CKKS today: the satellite page has no private-inference stage, and the full deployment's telemetry stage encrypts its own SMAP/MSL model.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
