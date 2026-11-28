# Pre-registration: SatQuMA's published key lengths, recomputed, and this project's bound on the same pass

Written and committed on 2026-11-28, on the owner's choices of that day (download SatQKD_sim_data; pre-register its
check; freeze), and frozen at its SHA-256 in the script before any figure below is computed. The EDA before it read
the files' names, columns and row counts only; before it was frozen, the input columns that fix Check B's
window and rate were read as well (`xi`, `dt`, `fs`, `NoPass`, `minElev`, `maxElev`, `startElev`, `stopElev`,
`shiftElev`, `shift_dt` and `SysLoss` of the four Fig5a files; Fig7's time, elevation and system loss), and no key length,
count or term of Eq. (2).

## What is known, said first

Sidhu et al., "Finite key performance of satellite quantum key distribution under practical constraints"
(arXiv:2301.13209v2, read 2026-11-28), compute their figures with SatQuMA: efficient BB84 with two decoy states (three
intensities, the third vacuum), the X basis for the key and the Z basis for the phase error, and the key length of
their Eq. (2): l = floor(s_X0 + s_X1 (1 - h(phi_X)) - lambda_EC - 6 log2(21/eps_s) - log2(2/eps_c)). The deposit
(Zenodo 8101679, CC BY 4.0) holds the CSVs behind their figures; its Fig5a files carry, per configuration, every term
of Eq. (2) beside the key length, and its Fig7 file a pass's elevation and losses against time. This project's
three-intensity bound (`vacuum_decoy_key_length` in `src/qsh/qkd/finite_key.py`) is Lim et al.'s; Eq. (2) has the form
of Lim et al.'s key length (the paper cites its reference 42 for it, not read here); the satellite page computes its
key with that bound's TypeScript twin (`finiteKey.ts`), its default (`free_space_vacuum_decoy`).

## Check A: Eq. (2) on each row's own terms (exact)

For every row of the four Fig5a CSVs (`Fig5a_LeftAxis_4GB`, `_8GB`, `_16GB` and `Fig5a_RightAxis`), l is recomputed
from the row's own `sX0`, `sX1`, `phiX`, `lambdaEC`, `eps_s` and `eps_c` by Eq. (2), in float64, and compared with the
row's `SKL`. Criterion: on every row the two are equal, or differ by at most one bit where the row's figures are
rounded (the CSV's own precision). Where Eq. (2) gives no key (l at most 0), the row is compared with 0 as well as
with l, since a code may report an absent key as 0, and such rows are counted apart. Any other difference is reported
as found, row by row, and nothing is refitted.

## Check B: this project's bound on the same pass (across implementations, informative)

Fig7 tabulates one pass, from its peak at 90 degrees of elevation (time 0) to the horizon, every half degree. The
Fig5a rows on that pass are the four with `xi` = 0 (maximum elevation 90 degrees), one per file, with half-windows
`dt` of 8, 16 and 32 s (`_4GB`, `_8GB`, `_16GB`) and 2 s (`Fig5a_RightAxis`); every other row is a lower pass whose
losses Fig7 does not give, and is in Check A only. On every row `startElev` equals `stopElev` and `shiftElev` and
`shift_dt` are 0, so each window is symmetric about the peak, and Fig7, one half of the pass, gives both halves.

For each of those four rows, the block is the pass from -dt to +dt. The half from the peak to dt is cut at Fig7's
times into segments, the last ending at dt; each segment's loss L is Fig7's diffraction plus atmospheric loss
interpolated linearly at the segment's middle time, plus the row's `SysLoss` (in place of Fig7's own system loss), and
its pulses are the row's source rate `fs` (500 MHz, Sidhu et al.'s Table 1) times the segment's duration, counted
twice, once per half. The channel's transmittance is 10^(-L/10), so none of this project's free-space model enters.
A row is computed only if Fig7's elevation at dt, interpolated linearly, is within 0.01 degree of the row's
`startElev`, and Fig7's `TotLoss` equals the sum of its three losses on every row to 1e-6 dB; a row failing either is
reported as failing it, with no key length. The device is the row's: `Pec` as this project's dark-count probability
per gate and per detector (its background yield is twice it), `QBERI` as the misalignment error, the three
intensities `mu1`, `mu2`, `mu3` and their probabilities `P1`, `P2`; `Px` as this project's key-basis probability (one
probability for both parties, in both models); after-pulsing and dead time 0 (this project's model has no
after-pulsing; the row's `Pap` is reported beside); SatQuMA's X basis mapped to this project's key basis. The bound
is `vacuum_decoy_key_length`, through `finite_key_over`, with the row's `eps_s` and `eps_c`, this project's
error-correction factor (1.16) and its configured statistics. Reported per row: both key lengths, their ratio, and
this project's key-basis and test-basis counts, single-photon and vacuum bounds, phase error and error-correction bits
beside SatQuMA's `nX`, `nZ`, `sX1`, `sX0`, `phiX` and `lambdaEC`. No agreement is required: the two differ in the
error-correction term, the statistics and after-pulsing, which the record names.

## Not done here

No SatQuMA code is run or installed; no parameter of either bound is tuned to make them meet; the satellite page does
not change.

## Compute

CPU, seconds. Script `scripts/check_satqkd_key_length.py`; record `docs/measurements/satellite-satqkd-key-length-<the
run's date>.json`, with the deposit's DOI and the files' SHA-256.
