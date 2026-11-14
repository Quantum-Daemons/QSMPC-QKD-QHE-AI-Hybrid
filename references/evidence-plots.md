# Evidence plots under the private-inference scatter and on the medical DP card

Read on 2026-11-14 for `web/src/lib/agreementStats.ts`, the panels under `EncryptedAgreementChart`
(`web/src/components/VerificationChart.tsx`) and `web/src/components/DpPrivacyUtility.tsx`, the owner's choice of that day
among four ways to explain a run. Each source was read at its page or in its code; nothing was downloaded.

## Agreement between the encrypted and the plaintext scores

- **Bland and Altman**, "Statistical methods for assessing agreement between two methods of clinical measurement", The
  Lancet 327(8476):307-310, 1986, doi:10.1016/S0140-6736(86)90837-8 (the citation read in a library record and on the
  first author's citation-history page, https://www-users.york.ac.uk/~mb55/meas/history.htm; read: html). Each pair's
  difference against its mean, the mean difference (the bias) and the limits of agreement, the bias plus and minus 1.96
  standard deviations of the differences. The two methods here are one model in the clear and under CKKS, so the bias
  says whether encryption shifts the scores and the limits how far one row's score moves; nothing in the method assumes
  either one is the truth, which is why it fits.
- **Lattigo**, `schemes/ckks/precision.go`, `GetPrecisionStats` (https://github.com/tuneinsight/lattigo, read: code). A
  value's precision is $`-\log_2 |\text{have} - \text{want}|`$, of the absolute error, and the library reports its
  minimum, maximum, mean, median and spread. A zero error is replaced there by the scale's $`\log_2`$, which the page does
  not know, so the page counts exact pairs apart instead. The caption above the scatter already gives the worst and median
  in whole bits, as OpenFHE and HElib print them; the histogram shows the rows between.

## Privacy against utility with the noise's spread

- The page's DP arm draws its noise from the tab's CSPRNG, so one run's accuracy at an epsilon is one draw. The card's
  band is a recorded sweep of the same function over 20 draws of seeded noise at the page's seed
  (`scripts/measure_fedheart_dp_sweep.py`), the 10th to 90th percentile with the median as a line, and the run's own row
  is marked against it. The accountant and the conversion it rests on are in `references/differential-privacy.md`.
