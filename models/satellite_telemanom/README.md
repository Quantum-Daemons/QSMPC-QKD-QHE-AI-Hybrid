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
| `event_wise_f0_5` | 0.454545 |
| `n_anomaly_intervals` | 320 |
| `n_parameters` | 80486 |
| `point_adjusted_f1` | 0.993691 |
| `published_full_channel_f0_5` | 0.061 |
| `published_lightweight_f0_5` | 0.786 |
| `wall_clock_s` | 43.4 |
| `windowed_iforest_event_wise_f0_5` | 0.000248 |

## Published baselines this is measured against

- **Target metric**: event-wise F0.5
- **Baseline to beat**: 0.786 - ESA-ADB Table 2, Mission1 lightweight (ch. 41-46), best event-wise F0.5. NOTE the caption: Table 2 scores detection of ALL EVENTS excluding communication gaps - anomalies PLUS annotated rare nominal events - whereas this project scores anomalies only, and ESA-ADB splits each mission into first-half train / second-half test. So 0.786 is an indicative target, not a like-for-like number.
- **Companion metrics actually recorded** in `models/metrics.json`: `point_adjusted_f1` (0.9937) and
  `windowed_iforest_event_wise_f0_5` (0.000248). The first is carried to expose how much point
  adjustment inflates a score - it reads 0.99 against an event-wise 0.4545 on the same run - and the
  second is the classical baseline on the same split. `published_full_channel_f0_5` (0.061) and
  `published_lightweight_f0_5` (0.786) are ESA-ADB's figures, not ours.

## Training data

- **Dataset**: ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46
- **Licence**: CC-BY-3.0-IGO
- **Source**: https://zenodo.org/records/12528696 (licence read 2026-08-03)

Missions are anonymised Mission1/2/3 - they are not named Sentinel anything. 17.5 years, 1.19% annotated. Mission3 is excluded upstream for triviality.

## Notes and limitations

Report event-wise F0.5, never point-adjusted F1. Always run a trivial baseline alongside: windowed isolation forest wins the ESA-ADB Mission2 lightweight subset outright. RETRACTION, Round 19: a Round 18 note here claimed this forecaster was about 22x WORSE than persistence. That was a measurement error of ours, not a property of the model -- the probe resampled at 60s while the model is trained at 300s (see `resample` in the achieved record). Re-measured at the trained cadence, next-step MAE is 0.0314 on the evaluation split against 0.0897 for persistence, i.e. the model is about 2.9x BETTER; 0.40x / 0.35x / 0.38x on train / test / whole series. Preprocessing for any evaluation must be read from this record rather than hardcoded, which is now what scripts/prep_model_inputs.py does. SEPARATE AND STILL OPEN: event-wise F0.5 does not reproduce across identical-config retrains -- 0.4545 here against the 0.714286 recorded -- while point-adjusted F1 does (0.9937 vs 0.9945) and the isolation-forest baseline reproduces exactly (0.000248). Treat the event-wise figure as a single draw from a wide distribution, which is the instability arXiv:2607.11969 (2026-07) describes for overlap-based metrics.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
