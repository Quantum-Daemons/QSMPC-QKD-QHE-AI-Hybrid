# Pre-registration: a QCNN for the DeFi page, on Aave v3 liquidations (2026-11-27)

Written and committed before the data below was downloaded (Round 86, batch 8), on the owner's choices of 2026-11-26 and
2026-11-27: look for a labelled task before deciding the DeFi page's QCNN (checklist QC54), then train one on Spadea and
Seneviratne's Aave v3 data. Nothing here is changed after a figure is seen; a change is an addendum with its date and
reason.

## Why

The DeFi page computes Aave V3's own liquidation rule (a health factor below 1) privately, so it has no learned
classifier for a QCNN to reproduce. The survey of 2026-11-27 (QC54) found one public, labelled, openly licensed
per-position source for Aave v3: the dataset of Spadea and Seneviratne, "From Risk to Rescue: An Agentic Survival
Analysis Framework for Liquidation Prevention", arXiv:2604.14583 (2026-04-16), on Figshare as
doi:10.6084/m9.figshare.31058368.v1, CC BY 4.0, one archive `data.zip` of 13,863,498,879 bytes, MD5
d9f43ce5010d10a1f836719e795c6db7. The paper describes it as over 21.8 million records of Aave v3 on the Polygon subgraph
with 90 engineered features (user-centric, market-level, transactional and temporal), built on the FinSurvival
benchmark's index and outcome events (Green et al., arXiv:2507.14160). The question is whether a tree QCNN on a few of
those features carries signal for liquidation after a borrow, read beside classical comparators of the same inputs. It
is not a claim of quantum advantage: a tree QCNN on classical data of this kind is classically simulable (Bermejo et
al., PRX Quantum 7, 020304, 2026), and the page says so.

## The task

- **Index event: a borrow. Outcome: the first liquidation of the same account after it.** For each borrow, the label is
  1 if that liquidation comes within 30 days of the borrow, and 0 if the account is observed for more than 30 days after
  the borrow without one. A borrow observed for 30 days or less without a liquidation has no label and is left out. The
  horizon of 30 days is fixed here, before the data is seen.
- If the archive holds survival tables in FinSurvival's form (a time from the index event and an event indicator per
  row), its borrow-to-liquidation table is used directly by the rule above. Otherwise the borrows and liquidations are
  joined by account from the archive's transaction records.

## Windows

By time, so nothing later is used to predict anything earlier: if the archive carries a training and a test part, its
test part is the test window and the last 15 percent of its training part, by borrow time, is the validation window;
otherwise the borrows are ordered by time, the first 70 percent form the fit window, the next 15 the validation window
and the last 15 the test window. Nothing is chosen on the test window.

## Inputs: k of the archive's features, chosen on the fit window only

- Numeric features of the borrow row, standardised by the fit window's mean and standard deviation and clipped at 10.
  Left out, by name in the addendum: identifiers (accounts, transactions, blocks, row ids), raw timestamps, and any
  feature computed from events after the borrow.
- Each input alone is scored by $`|\mathrm{AUC} - 0.5|`$ against the label on the fit window and the k highest kept, ties
  by name (the finance QCNN's rule). **k = 8** is the primary arm; **k = 4**, the first four, is reported beside it.

## Fit set, validation, test

- Fit: if positives are under a quarter of the fit window, every positive and a seeded sample (seed 0) of four negatives
  for each; otherwise the whole fit window. Either way, at most 20,000 rows, drawn stratified by label at seed 0 if more.
- Validation and test: every labelled borrow of the window, at its own prevalence.

## The arms

1. **QCNN.** The finance QCNN's circuit and recipe (`--arch tree --layers 3 --epochs 120 --encoding-scale 0.5`, one input a
   wire encoded RY$`(c \pi \tanh z)`$ with $`c = 0.5`$), seeds 0 to 19, each run's checkpoint chosen by validation average
   precision (the finance comparison's correction of 2026-11-26), at k = 8 and k = 4.
2. **No entanglement.** Arm 1's circuit without its entangling gates, seeds 0 to 4, at k = 8.
3. **Classical comparators** on the same inputs and fit set: logistic regression; gradient-boosted trees (scikit-learn's
   histogram booster, 300 rounds, learning rate 0.05, depth from $`\{3, 5\}`$ by validation average precision); and the
   classical surrogate on the circuit's own feature map, the $`3^k`$ products of one, the sine and the cosine of each
   wire's angle (Schreiber, Eisert and Meyer, PRL 131, 100803, 2023), by logistic regression.
4. **Reference row:** the booster on every numeric feature and the whole fit window, not like for like.

## Metrics

On the test window: average precision with a 95 percent bootstrap interval (1,000 resamples stratified by label), the
prevalence beside it, AUC, and the minority-class F1 at the threshold best on validation; for arms 1 and 2 the mean
over seeds and the seed ensemble's score; paired bootstrap differences between the arm-1 seed that would ship and each
other arm.

## Criteria for the page

1. **Signal:** the arm-1 seed with the best validation average precision at k = 8 has a test average precision at least
   twice the test window's prevalence.
2. **Export:** the page's clear contraction of that seed equals the state-vector readout within $`10^{-5}`$ on every row
   of the sample the page ships, and no decision differs.

If both hold, the DeFi page shows that QCNN as trained and not encrypted, beside its comparators, with every arm's
figure, its interval and the prevalence, the statement that the circuit is classically simulable with no quantum
advantage claimed, and which comparators score above it. If either fails, nothing is added to the page, the record
keeps every figure, and QC54 states the outcome. Which seed ships, a Hub upload and the page's default view are the
owner's choices after the figures.

## Data handling

- The archive is downloaded once, in one write stream, outside the repository, and its MD5 checked against the record
  above before anything reads it. The repository holds only this project's code, records and what the page ships.
- Nothing shipped carries an account, a transaction hash or any identifier: the page's sample holds the k inputs and the
  label of each row only. Credit, under CC BY 4.0: Spadea, F. and Seneviratne, O., "From Risk to Rescue: An Agentic
  Survival Analysis Framework for Liquidation Prevention Dataset", Figshare, doi:10.6084/m9.figshare.31058368.v1.
- **Addendum before any label:** after the download, and reading only the archive's file listing and the column headers
  of its files, an addendum names the files and columns used and left out, the borrow and liquidation event names, and
  the timestamps' unit. It is committed before any label is computed.

## Compute

Arms 2 and 3 and the selection on the CPU. Arm 1 on the GPU only with the owner's approval of that job, otherwise on the
CPU; the device of every run is recorded and changes no rule above.

## Addendum, 2026-11-27, after the download and before any label

Read so far: the archive's file listing and the column headers (first lines) of `data/Borrow/Liquidated/data.csv`,
`data/transactions.csv` and `data/users.csv`, and nothing else of them. The download: `data.zip`, 13,863,498,879 bytes,
MD5 d9f43ce5010d10a1f836719e795c6db7, equal to the record above; 59 entries, 50.63 GB uncompressed.

- **The table.** The archive holds FinSurvival's survival tables, one for each pair of index and outcome event,
  `data/<index>/<outcome>/data.csv`, over Borrow, Deposit, Liquidated, Repay and Withdraw. The task's table is
  `data/Borrow/Liquidated/data.csv` (1,487,673,105 bytes); no other file is read. It carries no training and test parts,
  so the windows are by its `timestamp`: the first 70, next 15 and last 15 percent of the borrows.
- **The label.** FinSurvival's columns `timeDiff`, the time from the borrow in seconds (FinSurvival's
  `dataPreprocessing.R` divides it by 86,400 to compare it with a cutoff in days), and `status`, 1 for an observed
  liquidation and 0 for censoring. As that code does, a row with `timeDiff` at or below 0 is left out as invalid. Then
  the rule above, as that code writes it: a row censored at or before 30 days ($`\mathtt{timeDiff} \le 30 \times
  86{,}400`$ with `status` 0) is left out; the label is 1 when `timeDiff` is at most $`30 \times 86{,}400`$, else 0.
- **The inputs.** Of the 92 columns, left out: the label (`timeDiff`, `status`); the events' names (`Index Event`,
  `Outcome Event`, `type`); the identifiers (`user`, `pool`, `reserve`); the raw `timestamp`; and the non-numeric
  columns (`coinType`, `userCoinTypeMode`, `userReserveMode`), which FinSurvival's code one-hot encodes and this
  pre-registration, taking numeric features only, leaves out. The other 80 are the candidate inputs, named in the
  record; `isWeekend` and `userIsNew` are read as 0 and 1, and a column that does not parse as numbers on reading is
  left out and named. By FinSurvival's construction (Green et al.) the features describe the account and the market at
  the borrow; nothing in their names says otherwise.
