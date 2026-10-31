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
| `corrected_event_wise_f0_5` | 0.0 |
| `event_wise_f0_5` | 0.0 |
| `event_wise_fn` | 3 |
| `event_wise_fp` | 2 |
| `event_wise_tp` | 0 |
| `n_anomaly_intervals` | 320 |
| `n_events_scored` | 3 |
| `n_parameters` | 80486 |
| `point_adjusted_f1` | 0.0 |
| `published_full_channel_f0_5` | 0.061 |
| `published_lightweight_f0_5` | 0.786 |
| `wall_clock_s` | 41.4 |
| `windowed_iforest_event_wise_f0_5` | 0.000248 |

## Published baselines this is measured against

- **Target metric**: event-wise F0.5
- **Baseline to beat**: 0.968 - ESA-ADB Supplementary Table 9, Mission1 lightweight (ch. 41-46), Telemanom-ESA-Pruned: corrected event-wise F0.5 0.968 (precision 0.999, recall 0.862) for detection of ANOMALIES ALONE - which is what this project scores, so this is the like-for-like row. Earlier rounds cited Table 2's 0.786 instead. That number is real and correctly attributed, but Table 2 scores ALL EVENTS excluding communication gaps, i.e. anomalies PLUS annotated rare nominal events, so it was the wrong comparator and the lenient one. Verified against the paper (arXiv:2406.17826 v2) in Round 24. FOUR protocol differences remain and none of them are closed by changing the number: (1) ESA-ADB's metric multiplies event precision by the TIME-LEVEL true-negative rate, which is why most of its baselines read <0.001 - this project records that corrected value as `corrected_event_wise_f0_5` beside the uncorrected one; (2) ESA-ADB splits each mission in half, this project tests on the last 30 percent; (3) ESA-ADB's Mission1 lightweight test partition holds 65 annotated events of which 29 are anomalies, against 3 in the window scored here; (4) ESA-ADB resamples Mission1 at 0.033 Hz, about one sample per 30 s, chosen from the dominant rate of channels 41-46, while this model resamples to 300 s, so its 250-step window spans about 20.8 hours where the benchmark's would span about 2.1, and short events can vanish. So this remains an indicative target, and the sample-size gap is roughly tenfold.
- **Companion metric shown alongside**: `event_wise_f0_5_full_channels` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

Read n_events_scored before comparing this to the baseline. The scored test window holds 3 labelled anomaly event(s), not the 320 that n_anomaly_intervals counts over the whole timeline - the annotated anomalies sit earlier in the mission, so a temporal split leaves almost nothing to score. This run decomposes as tp=0 fp=2 fn=3, and the detector flags 2 of 441,600 points. The figure does not reproduce across identical-config retrains: two earlier runs recorded 0.714286 (tp=1 fp=0 fn=2) and 0.454545 (tp=1 fp=1 fn=2), one false-alarm run apart, both with the whole-series z-score this script used until Round 34. Three protocol gaps to the citation remain open and are stated rather than closed: ESA-ADB multiplies event precision by the time-level TNR (recorded here as corrected_event_wise_f0_5), it splits each mission in half where this tests on the last 30 percent, and its lightweight test partition holds 65 events against 3 here. Workings: references/satellite-event-count-investigation.md.

## Training data

- **Dataset**: ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46
- **Licence**: CC-BY-3.0-IGO
- **Source**: https://zenodo.org/records/12528696 (licence read 2026-08-03)
- **Attribution (CC-BY-3.0-IGO)**: Kotowski et al., 'European Space Agency Dataset and Benchmark for Real-World Anomaly Detection in Spacecraft Time Series', Journal of Data-centric Machine Learning Research (DMLR), 2026 (earlier version: arXiv:2406.17826). Data: De Canio, Kotowski & Haskamp, 'ESA Anomaly Dataset', Zenodo record 12528696 (2024), CC BY 3.0 IGO.

Missions are anonymised Mission1/2/3 - they are not named Sentinel anything. Mission1 holds 14 years of telemetry and Mission2 3.5 (anonymised durations, arXiv:2406.17826v2 Table 1; the paper sizes the corpus by data points instead, because sampling rates vary and the timeline is anonymised). 1.19% of points annotated across Mission1 and Mission2 (1.80% in Mission1). Mission3 is excluded upstream for triviality. The CC BY 3.0 IGO term appears on the Zenodo landing page but NOT in its REST metadata, which returns no licence at all - an API-only check reads this corpus as unlicensed.

## Notes and limitations

Report event-wise F0.5, never point-adjusted F1. Always run a trivial baseline alongside: a windowed isolation forest scores highest on the ESA-ADB Mission2 lightweight subset in the paper's Table 2 (all events, 0.949), though not in Supplementary Table 9 (anomalies only, 0.000). RETRACTION, Round 19: a Round 18 note here claimed this forecaster was about 22x WORSE than persistence. That was a measurement error of ours, not a property of the model -- the probe resampled at 60s while the model is trained at 300s (see `resample` in the achieved record). Preprocessing for any evaluation must be read from this record rather than hardcoded, which is now what scripts/prep_model_inputs.py does. RETRAINED IN ROUND 71. The weights published until then were trained with a z-score fitted on the whole series, test window included: the trainer was fixed in Round 34 and the model was never retrained, so its feature_spec.json still carried whole-series constants. With the constants fitted on the training window, next-step MAE over all 441,600 test windows is 0.0250 against 0.0764 for persistence, about 3.1x better. The event-wise figure fell to 0 (tp=0 fp=2 fn=3): the detector still flags 2 points of 441,600 and neither lands in one of the 3 scored events, where the two leaky runs had caught one and recorded 0.714286 and 0.454545. Point-adjusted F1 fell with it, from 0.9937 to 0, which is how much one flagged point inside a long event carries under point adjustment; the isolation-forest baseline reproduces exactly (0.000248). Treat the event-wise figure as a single draw from a wide distribution: three events cannot rank detectors. With 0 of 3 events detected, the exact (Clopper-Pearson) 95% interval for recall runs from 0 to 0.71.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
