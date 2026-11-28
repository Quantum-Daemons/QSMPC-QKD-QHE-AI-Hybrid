# Second addendum to the SatQKD pre-registration: SatQuMA's cap, the satellite's statistics and its time slots, after the results

Written and committed on 2026-11-28, after Check B's and Check B''s results (records
`docs/measurements/satellite-satqkd-key-length-2026-11-28.json` and
`docs/measurements/satellite-satqkd-key-length-b-prime-2026-11-28.json`) and after an exploration of the difference
between the two bounds. The pre-registration (`references/satellite-satqkd-key-length-preregistration.md`), the first
addendum (`references/satellite-satqkd-key-length-preregistration-addendum-loss.md`) and both records are unchanged.

## What was found after Check B', and how

On the four zenith-pass rows of Fig5a, Check B' counted 0.94 to 0.99 of SatQuMA's key-basis detections on the three
left-axis rows and 0.80 on the right-axis row, and kept 0.42 to 0.43 of its key on the left-axis rows and 0.04 on the
right-axis row, with a phase error about three times SatQuMA's. Reading SatQuMA v2.0.0's
code (cnqo-qcomms/SatQuMA, commit 04c205a, MIT licence) found three differences that bear on these rows:

1. The cap. SatQuMA takes the smaller of the decoy bound on the single-photon errors in the test basis (Lim et al.'s
   Eq. (4)) and every error observed in that basis (`key/protocols/key_efficient_BB84.py`, line 371). This project's
   bound took the decoy bound alone, as Lim et al. print it.
2. The statistics. Checks B and B' ran this project's configured default, `finite_key_deviation` ("tighter", the
   fibre page's), because their script passes no statistics; the satellite page runs `free_space_finite_key_deviation`
   ("chernoff"), the Chernoff-type bounds per count of Yin et al. 2020 that SatQuMA also uses.
3. The window. SatQuMA counts $`2\Delta t + 1`$ one-second slots at the whole seconds $`t = -\Delta t, \dots, \Delta t`$,
   each at the loss of its second (`key/protocols/key_efficient_BB84.py`, line 238; `channel/time_dependent_loss.py`,
   lines 95 to 119; one second of pulses per slot, `input/protocol_inputs/input_efficient_BB84.py`, line 516), where
   Checks B and B' integrate $`2\Delta t`$ seconds.

An exploration on 2026-11-28, computed in memory and not recorded, before this addendum was written, found: with the
cap alone this project's key at 0.95, 0.98, 0.99 and 0.70 of SatQuMA's on the four rows (8, 16, 32 and 2 seconds either
side of the peak); with the cap, "chernoff" and SatQuMA's slots, at 1.02, 1.01, 1.01 and 1.08; and, with the cap, the
two statistics' phase errors within 0.003 of each other. The checks below are therefore not blind to their results, and
no criterion is set for them.

The owner chose on 2026-11-28 to keep the papers' bound on the page and to make the cap a named option,
`cap_single_photon_errors` in `src/qsh/qkd/finite_key.py` and `capSinglePhotonErrors` in `web/src/lib/finiteKey.ts`,
off by default; only this comparison names it.

## Check B'', chosen after the results and labelled so

Check B''s four rows, its loss (Fig7's total loss shifted so that its value at the peak is the row's `SysLoss`), its
device, source rate, intensities, probabilities, security parameters and $`f_{\mathrm{EC}}`$, in three variants, each
reported beside Checks B and B', never in their place:

- B''a: Check B' with `cap_single_photon_errors=True`; its statistics stay Check B''s.
- B''b: B''a with the satellite page's statistics, `free_space_finite_key_deviation` as the config ships it.
- B''c: B''b on SatQuMA's window: one segment for each whole second $`t = -\Delta t, \dots, \Delta t`$, each of one
  second of pulses, $`f_s \times 1\,\mathrm{s}`$, at the loss
  $`L(|t|) = L_{\mathrm{tot}}(|t|) - L_{\mathrm{tot}}(0) + L_{\mathrm{sys}}`$, Fig7's total loss interpolated linearly at
  $`|t|`$ as in Check B'.

The script first recomputes Check B' and stops if its keys differ from the Check B' record's. Reported per row and
variant: the key and its ratio to SatQuMA's, the key-basis and test-basis counts and their ratios to SatQuMA's, and the
phase error beside SatQuMA's. The record names what still differs: SatQuMA's after-pulsing, absent here; its
error-correction term as it records it, against $`f_{\mathrm{EC}}\, h(E)\, n`$ here; its errors per intensity at the
basis's average error rate, against each intensity's own error rate here; and its Chernoff lower bounds, not floored at
zero, against floored here.

## Not done here

No SatQuMA code is run, and the page's bound is not changed: the satellite page keeps the papers' Eq. (4) uncapped, and
its card says that SatQuMA caps and the page does not.

Script `scripts/check_satqkd_key_length_b_double_prime.py`, frozen at this addendum's SHA-256; record
`docs/measurements/satellite-satqkd-key-length-b-double-prime-2026-11-28.json`.
