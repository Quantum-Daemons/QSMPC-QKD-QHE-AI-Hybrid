# Addendum to the square-student pre-registration: parameter set A2

Written and committed on 2026-11-19, after the pre-registered run and before any run of what it describes. The
pre-registration (`references/medical-square-student-preregistration.md`) is unchanged; its results are in
`docs/measurements/medical-square-student-2026-11-19.json`.

## Why there is an A2

Under parameter set A ($`N = 8192`$, 50 + 30 + 30 + 30 + 50 bits, scale $`2^{30}`$) the exported student failed
criterion 3: the largest logit error was $`8.2 \times 10^{-3}`$, every one of the 156 test images was above
$`10^{-3}`$, and although no decision changed, the smallest decision margin on the test split is 0.0077. Parameter
set B passed. The owner chose on 2026-11-19 to pre-register an improved set at the same ring dimension rather than
offer A on the page or offer B alone.

## What A2 is

- $`N = 8192`$, coefficient modulus 48 + 34 + 34 + 34 + 52 bits (202), scale $`2^{34}`$. 202 bits is the
  HomomorphicEncryption.org standard's 128-bit post-quantum bound for a ternary secret at $`N = 8192`$, reached and
  not exceeded. The special prime, 52 bits, is the largest, and the first prime leaves 14 bits above the scale for
  the logits.
- The same exported weights (seed 2, the folded file whose SHA-256 the record gives), the same 156 test images, the
  same encrypted pass and the same TenSEAL build as the pre-registered run.

## The criterion

Criterion 3 of the pre-registration, unchanged: the largest absolute difference between a decrypted logit and the
plaintext logit is at most $`10^{-3}`$ on all 156 test images, and the argmax decision equals the plaintext one on
all 156. If A2 passes, the page offers A2 and B and the record keeps A as failed; if it fails, the page offers B only.
