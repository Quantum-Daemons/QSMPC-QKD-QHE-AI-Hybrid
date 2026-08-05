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

# Telemanom LSTM forecaster (per channel)

2 x LSTM(80), sequence length 250, dropout 0.3, non-parametric dynamic threshold with pruning

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo. This is the **plaintext** model for the
`satellite_comms` use case; the encrypted path runs a distilled student, not this model.

## Measured performance

| metric | value |
|---|---|
| `anomaly_sample_rate` | 0.0075528 |
| `event_wise_f0_5` | 0.714286 |
| `n_anomaly_intervals` | 320 |
| `n_parameters` | 80486 |
| `point_adjusted_f1` | 0.994475 |
| `published_full_channel_f0_5` | 0.061 |
| `published_lightweight_f0_5` | 0.786 |
| `wall_clock_s` | 60.1 |
| `windowed_iforest_event_wise_f0_5` | 0.000248 |

## Published baselines this is measured against

- **Target metric**: event-wise F0.5
- **Baseline to beat**: 0.786 — ESA-ADB Table 2, Mission1 lightweight (ch. 41-46), best event-wise F0.5. NOTE the caption: Table 2 scores detection of ALL EVENTS excluding communication gaps - anomalies PLUS annotated rare nominal events - whereas this project scores anomalies only, and ESA-ADB splits each mission into first-half train / second-half test. So 0.786 is an indicative target, not a like-for-like number.
- **Companion metric shown alongside**: `event_wise_f0_5_full_channels` — reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46
- **Licence**: CC-BY-3.0-IGO
- **Source**: https://zenodo.org/records/12528696 (licence read 2026-08-03)

Missions are anonymised Mission1/2/3 — they are not named Sentinel anything. 17.5 years, 1.19% annotated. Mission3 is excluded upstream for triviality.

## Notes and limitations

Report event-wise F0.5, never point-adjusted F1. Always run a trivial baseline alongside: windowed isolation forest wins the ESA-ADB Mission2 lightweight subset outright. MEASURED ROUND 18, and it matters for how the headline is read: this forecaster's next-step MAE on 60s-resampled, z-scored Mission1 channels 41-46 is ~0.90 on the evaluation split against ~0.041 for persistence -- i.e. simply repeating the last observed value is about 22x more accurate. Consistent across train, test and the whole series, so it is not a sampling artefact. Telemanom derives its anomaly signal from forecast residuals, so residuals this large are dominated by model error rather than by anomalies; the event-wise F0.5 of 0.714 should be read with that in mind. The demo shows the persistence comparison on the page rather than only the F0.5.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
