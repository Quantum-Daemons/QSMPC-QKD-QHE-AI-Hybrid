# Addendum to the SatQKD pre-registration: Check B on the deposit's own loss, after the result

Written and committed on 2026-11-28, after the pre-registered run, whose record is
`docs/measurements/satellite-satqkd-key-length-2026-11-28.json`. The pre-registration
(`references/satellite-satqkd-key-length-preregistration.md`) is unchanged, and its Check B stands as recorded.

## What the run found

Check A met its criterion: Eq. (2) on each row's own terms gives the row's key length on every row of the four Fig5a
files. Check B found no key on any of the four zenith-pass rows, where SatQuMA reports 8,399 to 336,358 bits: this
project's key-basis counts were about 76 times fewer than SatQuMA's, about 19 dB of loss too many. The cause is the
pre-registration's reading of `SysLoss`. It took the row's `SysLoss` as a system loss to add to Fig7's diffraction and
atmospheric losses. The deposit's Fig7 shows what it is: at the peak, Fig7's `TotLoss` is 40.0 dB, its `DiffLoss` 19.38,
`AtmLoss` 0.57 and `SysLoss` 20.05, so the 40 dB each zenith-pass row of Fig5a carries as `SysLoss` is the link's total
loss at zenith, and the pre-registered check counted the zenith's diffraction and atmospheric loss, about 20 dB, twice.

## Check B', chosen after Check B's result and labelled so

The same four rows and the same procedure as Check B, with one change: each segment's loss is Fig7's `TotLoss`
interpolated linearly at the segment's middle time, shifted so that its value at the peak is the row's `SysLoss`,
that is $`L(t) = L_{\mathrm{tot}}(t) - L_{\mathrm{tot}}(0) + L_{\mathrm{sys}}`$. For these four rows $`L_{\mathrm{sys}}`$
equals Fig7's $`L_{\mathrm{tot}}(0)`$, 40.0 dB, so the loss is Fig7's own total. Everything else is Check B's: the
window, the segments, the source rate, the device, the bound and its parameters. It is reported beside Check B, never
in its place, and the record says it was chosen after Check B's result. No agreement is required, as for Check B; the
record names the differences between the two bounds as Check B's does.

Script `scripts/check_satqkd_key_length_b_prime.py`, frozen at this addendum's SHA-256; record
`docs/measurements/satellite-satqkd-key-length-b-prime-2026-11-28.json`.
