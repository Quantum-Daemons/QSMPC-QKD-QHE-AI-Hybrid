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
a quantum-safe orchestration demo.
This is the **plaintext** model for the `satellite_comms` use case; the encrypted
path runs a distilled student, not this model.

## Measured performance

| metric | value |
|---|---|
| `anomaly_sample_rate` | 0.0075528 |
| `event_wise_f0_5` | 0.454545 |
| `event_wise_fn` | 2 |
| `event_wise_fp` | 1 |
| `event_wise_tp` | 1 |
| `n_anomaly_intervals` | 320 |
| `n_events_scored` | 3 |
| `n_parameters` | 80486 |
| `point_adjusted_f1` | 0.993691 |
| `published_full_channel_f0_5` | 0.061 |
| `published_lightweight_f0_5` | 0.786 |
| `wall_clock_s` | 43.2 |
| `windowed_iforest_event_wise_f0_5` | 0.000248 |

## Published baselines this is measured against

- **Target metric**: event-wise F0.5
- **Baseline to beat**: 0.968 - ESA-ADB Supplementary Table 9, Mission1 lightweight (ch. 41-46), Telemanom-ESA-Pruned: corrected event-wise F0.5 0.968 (precision 0.999, recall 0.862) for detection of ANOMALIES ALONE - which is what this project scores, so this is the like-for-like row. Earlier rounds cited Table 2's 0.786 instead. That number is real and correctly attributed, but Table 2 scores ALL EVENTS excluding communication gaps, i.e. anomalies PLUS annotated rare nominal events, so it was the wrong comparator and the lenient one. Verified against the paper (arXiv:2406.17826 v2) in Round 24. THREE protocol differences remain and none of them are closed by changing the number: (1) ESA-ADB's metric multiplies event precision by the TIME-LEVEL true-negative rate, which is why most of its baselines read <0.001 - this project records that corrected value as `corrected_event_wise_f0_5` beside the uncorrected one; (2) ESA-ADB splits each mission in half, this project tests on the last 30 percent; (3) ESA-ADB's Mission1 lightweight test partition holds 65 annotated events of which 29 are anomalies, against 3 in the window scored here. So this remains an indicative target, and the sample-size gap is roughly tenfold.
- **Companion metric shown alongside**: `event_wise_f0_5_full_channels` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

Read n_events_scored before comparing this to the baseline. The scored test window holds THREE labelled anomaly events, not the 320 that n_anomaly_intervals counts over the whole timeline - the annotated anomalies sit earlier in the mission, so a temporal split leaves almost nothing to score. Both recorded values decompose uniquely on three events: 0.714286 is tp=1 fp=0 fn=2 and 0.454545 is tp=1 fp=1 fn=2, so the 0.26 swing between two identical-config runs is ONE extra false-alarm run. The detector flags 2 points out of 441,600. Three protocol gaps to the citation remain open and are stated rather than closed: ESA-ADB multiplies event precision by the time-level TNR (recorded here as corrected_event_wise_f0_5), it splits each mission in half where this tests on the last 30 percent, and its lightweight test partition holds 65 events against 3 here. Workings: references/satellite-event-count-investigation.md.

## Training data

- **Dataset**: ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46
- **Licence**: CC-BY-3.0-IGO
- **Source**: https://zenodo.org/records/12528696 (licence read 2026-08-03)

Missions are anonymised Mission1/2/3 — they are not named Sentinel anything. 17.5 years, 1.19% annotated. Mission3 is excluded upstream for triviality.

## Notes and limitations

Report event-wise F0.5, never point-adjusted F1. Always run a trivial baseline alongside: windowed isolation forest wins the ESA-ADB Mission2 lightweight subset outright. RETRACTION, Round 19: a Round 18 note here claimed this forecaster was about 22x WORSE than persistence. That was a measurement error of ours, not a property of the model -- the probe resampled at 60s while the model is trained at 300s (see `resample` in the achieved record). Re-measured at the trained cadence, next-step MAE is 0.0314 on the evaluation split against 0.0897 for persistence, i.e. the model is about 2.9x BETTER; 0.40x / 0.35x / 0.38x on train / test / whole series. Preprocessing for any evaluation must be read from this record rather than hardcoded, which is now what scripts/prep_model_inputs.py does. SEPARATE AND STILL OPEN: event-wise F0.5 does not reproduce across identical-config retrains -- 0.4545 here against the 0.714286 recorded -- while point-adjusted F1 does (0.9937 vs 0.9945) and the isolation-forest baseline reproduces exactly (0.000248). Treat the event-wise figure as a single draw from a wide distribution, which is the instability arXiv:2607.11969 (2026-07) describes for overlap-based metrics.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
