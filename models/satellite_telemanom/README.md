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

# Telemanom LSTM forecaster (one model over six channels, as Telemanom-ESA)

2 x LSTM(80), sequence length 250, dropout 0.3, a 10-step forecast served by its first step, Telemanom's non-parametric dynamic threshold with pruning per channel

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **plaintext** model for the `satellite_comms` use case. No code path
evaluates this model under encryption; what each use case encrypts, if anything,
is stated in the notes below.

## Measured performance

| metric | value |
|---|---|
| `anomaly_sample_rate` | 0.00755741 |
| `corrected_event_wise_f0_5` | 0.949941 |
| `event_wise_f0_5` | 0.950413 |
| `event_wise_fn` | 6 |
| `event_wise_fp` | 0 |
| `event_wise_tp` | 23 |
| `n_anomaly_events_timeline` | 51 |
| `n_anomaly_intervals` | 320 |
| `n_events_scored` | 29 |
| `n_parameters` | 84860 |
| `point_adjusted_f1` | 0.948868 |
| `published_full_channel_f0_5` | 0.061 |
| `published_lightweight_f0_5` | 0.786 |
| `wall_clock_s` | 1499.2 |
| `windowed_iforest_corrected_event_wise_f0_5` | 5.6e-05 |
| `windowed_iforest_event_wise_f0_5` | 8.6e-05 |

## Published baselines this is measured against

- **Target metric**: corrected event-wise F0.5 (ESA-ADB)
- **Baseline to beat**: 0.968 - ESA-ADB Supplementary Table 9 (arXiv:2406.17826 v2; Table 18 of the DMLR 2026 version, read 2026-11-07), Mission1 lightweight (ch. 41-46), Telemanom-ESA-Pruned: corrected event-wise F0.5 0.968 (precision 0.999, recall 0.862) for ANOMALIES ALONE, which is what this project scores, so it is the like-for-like row. Earlier rounds cited Table 2's 0.786 (Table 4 of the DMLR version), which scores all events but communication gaps, anomalies PLUS annotated rare nominal events: a real number, and the lenient row. Since Round 86 the model is trained, thresholded and scored by ESA-ADB's own recipe and rules (scripts/train/satellite_telemanom.py): its Mission1 split, the 30 s zero-order hold, its scaling and training windows, Telemanom's thresholding as ESA-ADB's TelemanomThresholding applies it, and a reimplementation of its ESAScores over the test half's 29 anomalies. Two choices behind the published figure are not copied into this project's own result, and the record carries both: its floor of 0.05/7 is not shown to have been chosen without the test half, so the floor here is chosen on the validation months; and ESA-ADB's prepared series reads the labels to write annotated samples the zero-order hold dropped back onto the grid. Applying both reproduces the published figure (esa_scores.restored_annotated_samples in the record).
- **Companion metric shown alongside**: `published_full_channel_f0_5` - the same method's published score on the full Mission1 channel set (ESA-ADB Table 2, 0.061), a figure from the paper and not measured here, beside its matched lightweight figure 0.786.

### Gap to the published baseline

Read n_events_scored before comparing this to the baseline. Since Round 86 the forecaster follows ESA-ADB's Telemanom-ESA recipe on its own Mission1 split: a 30 s zero-order hold, training only on unlabelled windows before 2007-01-01 with early stopping on the months from 2006-10-01, a 10-step forecast scored on its first step, and Telemanom's thresholding per channel. The half after the split holds 29 labelled anomalies of the 51 on these channels. This score is the plain zero-order hold at an error floor of 0.0105 raw units, chosen on the validation months, which hold 2 anomalies: tp=23 fp=0 fn=6, 8,069 of 7,364,160 points flagged. The published 0.968 is Telemanom-ESA-Pruned, whose floor of 0.05/7 the paper does not show was chosen without the test half, and it is scored on ESA-ADB's prepared series, which writes annotated samples the zero-order hold dropped back onto the grid - a step that reads the labels. Replicating both gives 0.968 (tp=25 fp=0 fn=4), and 0.791 over all events but communication gaps, where Table 2 prints 0.786 (28 of 65 events here; the published recall 0.424 is no whole number of events out of this count, so ESA-ADB's all-events set differs from this selection by at least one event); esa_scores holds every arm and floor. Remaining differences: PyTorch where ESA-ADB uses Keras, from Keras's initialisers, one seed, and ESA-ADB's per-file resampling at the split. Until Round 86 this model trained at 300 s on a 70/30 cut holding 3 events and flagged 2 of 441,600 points. Workings: references/satellite-event-count-investigation.md.

### ESA-ADB's scores, both input series

Corrected event-wise F0.5 over the test half's anomalies, and over all events but communication gaps (ESA-ADB's Table 2 selection). One model; the two series differ only in ESA-ADB's restoration of annotated samples.

| input series | floor | anomalies | tp / fp / fn | all events |
|---|---|---|---|---|
| plain zero-order hold | floor from the validation months (0.0105) | **0.950** (this project's result) | 23 / 0 / 6 | 0.745 |
| plain zero-order hold | the paper's floor (0.007) | 0.950 | 23 / 0 / 6 | 0.769 |
| plain zero-order hold | no floor (0.0) | 0.094 | 26 / 309 / 3 | 0.185 |
| plain zero-order hold | no floor, channels agreeing (k = 6 of 6, from the validation months) | 0.909 | 22 / 1 / 7 | 0.845 |
| ESA-ADB's prepared series, which reads the labels | floor from the validation months (0.0105) | 0.968 | 25 / 0 / 4 | 0.769 |
| ESA-ADB's prepared series, which reads the labels | the paper's floor (0.007) | **0.968** (replication) | 25 / 0 / 4 | 0.791 |
| ESA-ADB's prepared series, which reads the labels | no floor (0.0) | 0.101 | 28 / 308 / 1 | 0.191 |
| ESA-ADB's prepared series, which reads the labels | no floor, channels agreeing (k = 6 of 6, from the validation months) | 0.930 | 24 / 1 / 5 | 0.859 |
| published Telemanom-ESA-Pruned (not measured here) | 0.05/7 | 0.968 | | 0.786 |

## Training data

- **Dataset**: ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46
- **Licence**: CC-BY-3.0-IGO
- **Source**: https://zenodo.org/records/12528696 (licence read 2026-08-03)
- **Attribution (CC-BY-3.0-IGO)**: Kotowski et al., 'European Space Agency Dataset and Benchmark for Real-World Anomaly Detection in Spacecraft Time Series', Journal of Data-centric Machine Learning Research (DMLR), 2026 (earlier version: arXiv:2406.17826). Data: De Canio, Kotowski & Haskamp, 'ESA Anomaly Dataset', Zenodo record 12528696 (2024), CC BY 3.0 IGO.

Missions are anonymised Mission1/2/3 - they are not named Sentinel anything. Mission1 holds 14 years of telemetry and Mission2 3.5 (anonymised durations, arXiv:2406.17826v2 Table 1; the paper sizes the corpus by data points instead, because sampling rates vary and the timeline is anonymised). 1.19% of points annotated across Mission1 and Mission2 (1.80% in Mission1). Mission3 is excluded upstream for triviality. This project's data is version 1.0 of the Zenodo record (12528696, 25 June 2024), which the benchmark's README links: its ESA-Mission1.zip matches that record's MD5. A version 2 (record 15237121, 17 April 2025, the same licence) has since replaced all three archives with files of other checksums, and neither it nor the README says what changed. The CC BY 3.0 IGO term appears on the Zenodo landing page but NOT in its REST metadata, which returns no licence at all - an API-only check reads this corpus as unlicensed.

## Notes and limitations

Report ESA-ADB's corrected event-wise F0.5, never point-adjusted F1. Always run a trivial baseline alongside: a windowed isolation forest scores highest on the ESA-ADB Mission2 lightweight subset in the paper's Table 2 (Table 4 of the DMLR version; all events, 0.949), though not on Mission1, where it reads below 0.001 in ESA-ADB's own tables and does here too. RETRAINED IN ROUND 86 by ESA-ADB's Telemanom-ESA recipe (see the metric's baseline_source). The record holds two arms from one model. The project's result is the plain zero-order hold at an error floor chosen on the validation months, inputs a deployed detector could compute. The replication arm applies the two steps of ESA-ADB's published figure this project does not take as its own, the label-reading restoration of annotated samples and the paper's floor, and it reproduces that figure. The floor skips the windows whose errors stay small; without one, both arms land near the benchmark's plain Telemanom-ESA (0.090). The record's esa_scores holds both of ESA-ADB's readings, anomalies alone and all events, for every arm and floor, and a third reading that requires k channels to agree in place of a floor (the owner's choice). Its k is chosen on the validation months, where every k from 2 to 6 finds both anomalies with no false alarm and the time-level TNR alone picks the largest: it scores lower than the floor on anomalies and higher on all events, because it keeps rare events the floor drops. Its int8 graph quantises only the output layer. onnxruntime's dynamic quantiser converts an LSTM only when both of its weight tensors are stored constants, and the exported graph computes them, so both LSTM layers stay fp32 and the int8 file is nearly the size of the fp32 one. Earlier models, kept as history: until Round 71 the weights carried a z-score fitted on the whole series; the Round 71 model, trained at 300 s on a 70/30 cut whose test window held 3 events, flagged 2 of 441,600 points and scored 0. A Round 18 note claimed it was 22x worse than persistence, a probe that resampled at 60 s against a model trained at 300 s, and the lesson stands: preprocessing for any evaluation is read from this record, never hardcoded, which scripts/prep_model_inputs.py does.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
