# Satellite: real telemetry, real satellite-QKD data and teacher models (read 2026-11-25)

On the owner's choice of 2026-11-25 (prefer practical large-scale REAL data to synthetic, include calibration and
preprocessing in the verification, and assess anything worth bringing to the encrypted paths), a survey read the
primary sources for the satellite page, and the agent then re-read each record and licence it relies on below at the
record's own interface on 2026-11-25: Zenodo's records API, the Hugging Face model API and arXiv's API. Nothing was
downloaded. Where a figure was read by the survey and not re-read, it says so.

## What the page uses today

- **Telemetry, real:** ESA-ADB (CC BY 3.0 IGO), Mission1 lightweight channels 41 to 46, for the Telemanom forecaster
  and its CKKS linear student; OPS-SAT-AD version 1 (CC BY 4.0, Zenodo 12588359) for the gradient-boosted model and its
  CKKS student over the quadratic expansion of its features; the satellite QCNNs and the tree network on both
  (`DATA_LICENSES.md` section 1). NASA SMAP/MSL was removed in Round 84: no provider grants a licence for it.
- **The QKD downlink and QSDC:** simulated, from published link parameters (the Micius profiles), not from measured
  pass data.
- **Calibration:** Telemanom's non-parametric dynamic threshold, fitted as ESA-ADB's Telemanom-ESA fits it; the
  project's teacher scores a corrected event-wise $`F_{0.5}`$ of 0.950 on the anomalies of channels 41 to 46 and its
  CKKS student, 294 weights, 0.940 (`models/metrics.json`), against the benchmark's 0.968 for Telemanom-ESA-Pruned on
  anomalies alone (DMLR 2026, Table 18, the like-for-like row the student's card cites, read 2026-11-07).

## Real data found

| Record | Read at | Licence | Size | What it is |
|---|---|---|---|---|
| Jinan-1 source data, *Microsatellite-based real-time quantum key distribution* (Li et al., Nature 640, 47-54, 2025) | Zenodo 14732295 (2025-01-24, creator Liao Sheng-Kai) | CC BY 4.0 | three xlsx files, 1,476,081 B in all | the paper's figure data; the survey reports, from the captions, one measured pass (2022-09-25) of range, count rate and QBER per packet. The columns were not opened |
| ESA-ADB, all of Mission1 | Zenodo (the local version-1 zip holds every channel) | CC BY 3.0 IGO | already on this machine | 58 target channels; the survey counts 55 test anomalies there against 29 in channels 41 to 46. Mission2's test sets hold 4 (lightweight) and 9 (full) anomalies, too few for an anomaly-only test |
| ESA-ADB version 2 | Zenodo | CC BY 3.0 IGO | | the survey compared the two versions' zip directories and found only each mission's `channels.csv` changed; the labels and telemetry files are identical. What changed in `channels.csv` is undocumented |

No public in-orbit QKD data newer than Jinan-1 was found: QEYSSat is in development, EAGLE-1 is due in late 2027, and
the SPOQC payload paper (arXiv:2611.07861) reports ground tests only (survey). Simulated key-rate sets on Zenodo are
not calibration data.

## Teacher models, licence read at the Hugging Face API

| Model | Licence | Created | Note |
|---|---|---|---|
| `amazon/chronos-2` | Apache-2.0 | 2025-10-30 | multivariate forecaster |
| `NX-AI/TiRex-2` | Apache-2.0 | 2026-06-16 | arXiv:2607.01204 |
| `Datadog/Toto-2.0-22m` | Apache-2.0 | 2026-04-14 | |
| `ibm-granite/granite-timeseries-tspulse-r1` | Apache-2.0 | 2025-06-03 | about one million parameters, built for anomaly scoring (arXiv:2505.13033) |
| `google/timesfm-2.5-200m-pytorch` | Apache-2.0 | 2025-09-02 | |
| `google/timesfm-3.0-pytorch` | `timesfm-non-commercial-license-v1.0` | 2026-08-24 | **not usable here**: non-commercial, and the survey reads it as excluding distillation and distributing derivatives |

The survey also lists Moirai (CC BY-NC 4.0), timeh2o (non-commercial), Timestral (a revenue cap that reaches
derivatives) and TiRex version 1 (an attribution licence) as not usable. A teacher would be run offline; its weights
would never ship, and only a student fitted to it would.

## Practice the 2025-2026 work follows

- ESA-ADB's own preprocessing: zero-order hold on a fixed grid, each channel z-scored on nominal training points only.
- Thresholds: Telemanom's non-parametric dynamic threshold; extreme-value thresholds (a generalised Pareto fit,
  arXiv:2609.10017, 2026-09-09); adaptive weighted-quantile conformal scoring on top of a frozen foundation model
  (arXiv:2604.20122, 2026-04-22). Telemetry is not exchangeable, so a conformal set is calibrated on the time-ordered
  validation window, not a random split.
- Metrics: corrected event-wise $`F_{0.5}`$ (ESA-ADB's) and VUS-PR, which the survey's sources find hardest to game.
- Leakage: ESA-ADB's split is temporal. OPS-SAT-AD's official split is stratified by class only, so neighbouring
  segments of one channel can sit in both partitions; a time-blocked, channel-grouped re-split is a robustness row,
  beside the official split, not a replacement.

## 2026 research the page should know

- Encrypted analytics for space systems: encrypted collision-avoidance manoeuvres under two-party CKKS on real
  conjunction data messages (arXiv:2610.14501, 2026-10-10), after CKKS collision-risk work of 2025. Nothing found that
  runs FHE or MPC on telemetry anomaly detection, so the page claims no "first"; it applies encrypted inference to a
  public telemetry benchmark.
- Large-scale ESA-ADB benchmarking: arXiv:2607.07335 (2026-07-08).
- Satellite QKD: security of BB84 with passive basis choice and detection-efficiency mismatch for a realistic
  satellite link (arXiv:2608.09793, 2026-08-10), which bears on the page's Micius profiles, whose basis choice is
  passive.

## Options for the owner

1. **Channels 41 to 46, with teachers and calibration in the verified path** (nothing to fetch). Zero-shot forecasts or
   scores from the Apache-2.0 teachers above, offline; the existing CKKS linear student refitted to each; three
   thresholds fitted on the validation window only (the current one, an extreme-value one, adaptive conformal); the
   anomaly-only corrected event-wise $`F_{0.5}`$ over the 29 test anomalies and VUS-PR; the browser's decrypted forecasts
   must reproduce the plaintext alarm flags exactly. A test that every scaler and threshold constant comes from the
   training and validation windows. A fall in agreement under a long-context teacher is a result, not a failure.
2. **The simulated downlink calibrated to Jinan-1** (1.5 MB, CC BY 4.0). A plaintext fit of the page's link budget to
   the measured pass, with residuals, the QBER comparison and the predicted against the reported key. No CKKS.
3. **All of Mission1** (already local; heavy preprocessing). A small precomputed window-feature slice for a
   quadratic or square scorer under CKKS; published full-set scores are low, which the page would say.

Not recommended: Mission2 as the anomaly-only test; Mission3; the non-commercial teachers above; SatNOGS as a corpus
for published weights (CC BY-SA, unlabelled); SMAP/MSL (still no grant); swapping OPS-SAT-AD to version 2 (different
files, nothing gained).

## Open questions

- What changed in ESA-ADB version 2's `channels.csv`.
- What the Jinan-1 xlsx files hold, column by column (read on the owner's approval of the download).
- Whether Toto-2.0's and TiRex-2's pretraining corpora include ESA-ADB (the survey checked only Chronos-2's).
