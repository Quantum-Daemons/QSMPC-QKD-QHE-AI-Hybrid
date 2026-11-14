# VUS-PR beside the event-wise F0.5, and a decision boundary on the satellite overlay

Read on 2026-11-14 for the satellite page's two cards (`scripts/measure_satellite_vus_pr.py`,
`scripts/measure_satellite_boundary.py`, `web/src/components/SatVusPrCard.tsx` and `SatBoundaryCard.tsx`). A research
agent opened the sources named here and read HTML pages and source code only; "read" says how far. Nothing below is a
PDF this project downloaded.

## VUS-PR

- Paparrizos, Boniol, Palpanas, Tsay, Elmore and Franklin, *Volume Under the Surface: A New Accuracy Evaluation Measure
  for Time-Series Anomaly Detection*, PVLDB 15(11), 2022. (not read: PDF only) The definition was read in the journal
  version, arXiv:2502.13318v1, and in the TSB-UAD documentation. (read: html) VUS-PR is the mean, over buffer sizes from
  0 to a maximum, of a range-based average precision: each anomaly range gains a soft margin of half the buffer on each
  side, weighted by $`\sqrt{1 - d/w}`$ at distance $`d`$ from its edge, margin points count only where the detector
  flags them, and recall is multiplied by the share of events hit (the existence reward).
- Liu and Paparrizos, *The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark* (TSB-AD),
  NeurIPS 2024 Datasets and Benchmarks. (read: abstract) It recommends VUS-PR as the most reliable measure.
- TSB-AD 1.5, the code: `get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250)` calls
  `generate_curve`, which computes the curve at 250 thresholds spaced by rank: `RangeAUC_volume_opt` takes
  `score_sorted[i]` for `i` in `np.linspace(0, len(score) - 1, thre)`, so the thresholds sit evenly through the ranking,
  about 29,600 steps apart on this test half; the 'opt' version gives the same values as the original algorithm. (read: code, and the 1.5 wheel itself, installed here with `--no-deps` with the owner's
  approval: Apache-2.0, SHA-256 equal to PyPI's) Its multivariate benchmark script sets the buffer with
  `find_length_rank(data[:,0].reshape(-1, 1), rank=1)`, a period from the autocorrelation of the first channel, falling
  back to 125. (read: `benchmark_exp/Run_Detector_M.py` on GitHub) That function imports statsmodels, which is not
  installed here; `scripts/measure_satellite_vus_pr.py` computes the same autocorrelation with numpy's FFT, and
  `tests/test_r86_satellite_vus_pr.py` checks it against the direct sum.
- TimeEval's `RangePrVUS` uses linear margins and a default maximum buffer of 500, so its numbers are not TSB-AD's.
  (read: code)
- Lyu, *Did We Actually Fix It? An Independent Adversarial Stress-Test of Post-Point-Adjustment Evaluation Metrics for
  Time-Series Anomaly Detection*, arXiv:2607.11969v2, a single-author preprint. (read: abstract and html) Under best-of-N
  reporting of no-skill detectors VUS-ROC inflates while VUS-PR stays near its prevalence floor on most of its six
  benchmarks, and a random-walk score is VUS-PR's strongest no-skill adversary. The card's two no-skill rows follow it.
- ESA-ADB (Kotowski et al., DMLR 2026; arXiv:2406.17826): its Mission1 experiment configuration computes no VUS metric.
  (read: code; the paper's PDF not read) No published VUS-PR for Mission1's lightweight channels 41 to 46 was found; three
  2026 papers that use Mission1's lightweight subset, two of them on channels 41 to 46, report none. (read: html of each)

## Decision boundaries for a model on more than two features

- scikit-learn 1.9.1, `DecisionBoundaryDisplay`, needs an estimator on exactly two features. (read: html)
- Molnar, *Interpretable Machine Learning*, 3rd edition, and Biecek and Burzykowski, *Explanatory Model Analysis*,
  chapter 10: a slice through other features held fixed depends on where it is placed, correlated features make
  impossible points, and a surrogate is only as good as its measured fidelity. (read: html)
- condvis2 (arXiv:2101.06986) takes an observed point, not per-feature medians, as the default anchor of a section, since
  medians of linked features need not form a realisable point. (read: html)
- The rule the card follows, set before any figure was seen: a slice of the published graph through the test
  partition's medoid, drawn only if at least 20 test segments lie near it and the full model gives at least 90% of them
  the slice's class; otherwise a two-feature surrogate with its fidelity stated; otherwise no boundary. The 20 and the
  90% are this project's thresholds, not a standard.
