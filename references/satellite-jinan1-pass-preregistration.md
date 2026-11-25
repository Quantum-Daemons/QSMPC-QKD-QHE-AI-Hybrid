# Pre-registration: the satellite downlink against Jinan-1's measured pass (2026-11-26)

Written and committed before any model figure was compared with the source data, on the owner's approval of the
download (2026-11-25) and of the satellite survey's option 2 (`references/satellite-real-data-2026-11.md`). Nothing here
is changed after a figure is seen; a change is an addendum with its date and reason. Before writing it, each workbook's
sheet names, column headers and first rows were read, to learn their layout, and, to define the windows below, which
rows of sheet `4c` hold a QBER above zero (from the 114th of its 449 data rows to the last, without a gap) and how its
key column moves (it accumulates, rising in four steps). No value was set beside the model.

## Why

The page's `jinan1_2022_jinan` downlink profile has one calibrated constant, the station efficiency 0.1683, fitted to
the peak count rate of the 25 September 2022 pass as read from the preprint's figure (about 40,700 per second at
823 km, less a background of 650), and its test checks the fall with elevation at two ranges read the same way
(`tests/test_the_model_sits_among_micius_nights.py`). The paper's source data (Li et al., Nature 640, 47-54, 2025;
Zenodo 14732295, CC BY 4.0) gives that pass second by second: counts and slant range (Fig. 4b), QBER and secure key
bits (Fig. 4c). The questions: how well the page's link budget follows the whole measured pass, and what the source
data, rather than a figure reading, makes of its one constant.

## Data

- `Data_Figure4.xlsx`, sheet `4b`: time (UTC+8, a spreadsheet serial day) and counts per second; time (as text) and
  slant range in km. Sheet `4c`: time, QBER and secure key bits.
- `Data_Extented_Figure2.xlsx` (the record's spelling), its one sheet: time, elevation and the noise's share of the
  QBER; reported beside the comparison, not fitted.
- Alignment: each serial time is rounded to the nearest second; a second that repeats keeps its first row, since the
  sheets' notes describe repeated and skipped seconds; counts and range are joined on the second, and 4b and 4c
  likewise; the seconds without a match are dropped and counted.
- Link seconds: those with a QBER above zero in 4c, the paper's own record of when quantum signals were measured.
- Background: the median count rate over the seconds before the first link second and after the last, leaving out the
  ten seconds nearest the link on each side (after it there are none in this file). Measured detections are counts
  less the background.

## The comparison

- The model is the profile as it stands: its detections per second at each second's slant range by the page's own
  link budget (the channel's transmittance and the decoy mixture, as the test above computes them), with the elevation
  taken from the range and the orbit and no minimum elevation applied.
- The residual of a link second is $`10 \log_{10}`$ of measured over modelled detections, in dB, over the link seconds
  whose measured detections are above zero.
- The one constant refitted: the station efficiency at which the mean residual over those seconds is zero (least
  squares in dB with one multiplicative constant), found by bisection, and reported beside 0.1683.
- Reported: the residuals' mean, deviation and root mean square before and after the refit; their linear trend
  against air mass; and the fall from the ten link seconds of smallest range to the first ten and to the last ten link
  seconds, measured against modelled, in dB, which no single constant changes.
- QBER: the model's QBER at each link second against the measured one, their means over the link seconds and the mean
  difference; the noise's share from Extended Data Fig. 2 beside them.
- Key: the measured pass's secure key, the last value of 4c's accumulating column, against the model's key for one
  pass at the profile's top elevation, as a ratio.

## Rules for the profile and the page

1. If the source data's fit differs from 0.1683 by more than 0.1 dB, the profile takes the fit, and its quote and the
   test's constants cite the source data; otherwise the constant stays and its note names the source data's check.
2. The test's two figure-read ranges give way to the source data's falls: each must match the model within 0.5 dB, the
   test's present bound. If one does not, the test records the measured miss and the page says so.
3. The satellite page gains a card: the measured detections per second and the model's against slant range, with the
   residuals' figures, the QBER comparison and the key ratio. The per-second figures ship as an aggregate of the source
   data with its credit.

The comparison is plaintext; nothing here touches CKKS.

## Not decided here

Whether the Jinan-1 profile becomes the satellite page's default downlink is the owner's choice after the figures.

## Addendum, 2026-11-26, before any comparison was run

Two definitions the text above left open, fixed while writing `scripts/measure_jinan1_pass.py` and before it ran.
**The model's QBER** at a second is the signal state's observed error rate at that second's range (`signal_qber` in
`qsh.qkd.bb84`), the quantity a receiver measures; the page's analytic baseline (`intrinsic_qber`) is reported beside
it. **The model's key** is the page's own pass key: the asymptotic key that `simulatePass` in
`web/src/lib/satNetwork.ts` sums over one pass at the profile's top elevation, as the test above compares it, run by
the page's code in node; the finite key over the same pass (`finite_key_over`) is reported beside it.

## Addendum 2, 2026-11-26, after the run: the windows did not hold

The run (`docs/measurements/satellite-jinan1-pass-2026-11-26.json`, kept) showed that the two windows above do not
describe the pass. Sheet `4c`'s QBER is one value per key block, held over the block's seconds: it is first recorded
about 75 seconds after the counts leave the background (near 1,660 km, where they jump from about 730 to 5,750 per
second), and it keeps the last block's value for about 50 seconds after the counts have fallen back (from about
1,500 km). So the background window held the link's first 65 seconds or so, and its median, 6,910 per second, is about
ten times the counts outside the link; the link window lost the link's start and kept seconds without it. The figures
on those windows answer nothing the text asked, and no rule above is applied from them: rule 1 would replace the
constant with a fit to a wrong background.

What follows was chosen after those figures were seen, and is labelled post hoc wherever it is shown:

- background: the median count rate over the file's first and its last 30 seconds, both outside the link by the counts
  themselves;
- link seconds: those whose counts exceed five times that background;
- the rest as above, with the QBER compared over the link seconds that carry a block's value, block by block.

Rules 1 to 3 are not applied to the post-hoc figures either. Whether the profile takes a post-hoc fit, and whether the
page shows the post-hoc comparison, are the owner's choices.
