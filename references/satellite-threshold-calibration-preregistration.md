# Pre-registration: threshold calibration for the satellite forecaster on ESA-ADB (2026-11-25)

Written and committed before any threshold below was fitted and before any test window was scored, on the owner's
choice of 2026-11-25 to compare the page's threshold with calibrated alternatives fitted on the validation window only
(`references/satellite-real-data-2026-11.md`, option 1, without teacher models). Nothing here is changed after a figure
is seen; a change is an addendum with its date and reason.

## What is compared, and what is not

The forecasts are the shipped CKKS student's (`satellite_comms.student`, the linear next-step forecaster of 294
weights that the page evaluates under CKKS), on ESA-ADB's Mission1 lightweight channels 41 to 46 with the preparation
the student's trainer uses (`scripts/train/satellite_student_forecaster.py`: ESA-ADB's zero-order hold on a 30 s grid,
z-scored on the unlabelled training rows). The error of a channel is $`|\hat{x} - x|`$ in the channel's units, smoothed
as Telemanom smooths it (`telemanom_threshold.smooth_errors`, an exponentially weighted moving average of span 105).
Only the rule that turns the smoothed errors into flags differs between the arms. Every threshold is decrypted-side
arithmetic: the encrypted part of the page is unchanged by any of them.

## The arms

1. **NDT, as now.** Telemanom's non-parametric dynamic threshold with ESA-ADB's pruning and the floor the record chose on
   validation (`telemanom_threshold.flag`), the page's present rule.
2. **Extreme value (POT).** Per channel, a generalised Pareto distribution fitted by maximum likelihood to the excesses
   of the validation window's smoothed errors over their 0.98 quantile (at least 30 excesses, else the empirical
   quantile), and the threshold at which the fitted exceedance probability is $`\alpha`$ (Alcarria et al.,
   arXiv:2609.10017, the method; fitted once on validation here, not recomputed monthly).
3. **Weighted conformal.** Per channel, the weighted $`1 - \alpha`$ quantile of the validation window's smoothed errors,
   each weighted by $`2^{-\Delta t / h}`$ with $`\Delta t`$ its age at the window's end and $`h`$ seven days, so that the
   threshold leans on the most recent behaviour (telemetry is not exchangeable; arXiv:2604.20122 calibrates adaptively
   for the same reason).

For arms 2 and 3, $`\alpha = 10^{-4}`$ is the primary setting, and $`10^{-3}`$ and $`10^{-5}`$ are reported beside it.
A timestep is flagged when any channel's smoothed error exceeds that channel's threshold; no buffer and no pruning
are applied, so the arms differ from NDT in both the threshold and the post-processing, which the record says.

## Data windows

ESA-ADB's own split, as the record uses it: fit before 2006-10-01, validation from 2006-10-01 to 2007-01-01, test from
2007-01-01. The validation window holds two anomaly events; the thresholds model its nominal tail, and no threshold is
chosen by its anomalies. Nothing is read from the test window until every threshold is fixed.

## Metrics

On the test window: ESA-ADB's corrected event-wise $`F_{0.5}`$ over the anomalies (29 events, the like-for-like row of
the student's card) as the primary, with its precision, recall and the number of detected events; the same over all
events; and the share of test timesteps flagged. VUS-PR does not depend on a threshold, so it is reported once, for
the score the arms share.

## Not decided here

Whether the page offers another threshold, and which, is the owner's choice after the figures, which go to
`docs/measurements/satellite-thresholds-2026-11-25.json`.
