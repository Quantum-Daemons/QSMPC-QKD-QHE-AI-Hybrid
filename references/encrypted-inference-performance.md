# Encrypted-inference performance, per use case, on an axis that is actually comparable

Round 38. Every number below is measured in a browser tab on the machine stated, or quoted from a
primary source that was fetched and read - not from a search summary. A previous round was burned by
a search result that quoted a paper's training accuracy as if it were validation accuracy, so the
rule here is that a comparator has to be resolved to the artefact that published it.

## The problem with comparing total latency

This project's encrypted tier evaluates between 20 and 3,226 parameters on WASM CPU inside a browser
tab. The 2026 encrypted-inference figures it would sit beside are ResNets and transformers on
datacentre GPUs. Putting 700 ms for a 64-parameter student next to a few hundred milliseconds for a
CNN on an A100 tells a reader nothing about either system, and this repository's own rule against
incomparable numbers forbids it.

Cost per homomorphic rotation is the axis that survives the comparison. A rotation triggers a
key-switch, which is the primitive the CKKS literature benchmarks, and model size drops out of it.

## What was measured, and what had to be fixed to measure it

The obvious derivation - `fhe_linear_ms / fhe_dense_rotations`, both of which the page already
printed side by side - is wrong, and wrong by a lot. `fhe_linear_ms` is the measured wall-clock
multiplied by `benchScale()` for the model size selected in the Parameters panel, while
`fhe_dense_rotations` is a raw count from the loop. On the genomics page the scale factor is 33, so
that division reads **159.9 ms per rotation against a true 3.18**.

`rotateVector` is now timed directly. The page reports `fhe_ms_per_rotation` from that measurement,
and reports `fhe_linear_measured_ms` and `fhe_bench_scale` beside `fhe_linear_ms` so the projected
figure is decomposable rather than a number of unstated provenance sitting among raw counts.

## Per-rotation cost, five measurements

Chrome 149.0.0.0, WASM CPU, Linux x86_64, 20 logical cores, `crossOriginIsolated` true. CKKS ring
degree 8192, coefficient modulus `[60, 40, 40, 60]` (200 bits), one multiplicative level.

| use case | model | rotations | rotate ms | ms per rotation |
|---|---|---|---|---|
| finance | XGBoost-distilled CKKS-linear surrogate, 8 features, 2 classes | 3 | 9.29 | 3.10 |
| finance | (repeat run) | 3 | 9.72 | 3.24 |
| medical | bundled linear head, dense layer | 3 | 13.17 | 4.39 |
| medical | published CNN student, convolution | 16 | 67.81 | 4.24 |
| medical | published CNN student, dense layers | 136 | 421.40 | 3.10 |
| telecom | 8-class softmax student, batched over 512 rows | 24 | 76.0 (derived) | 3.17 |
| genomics | PGS000004 effect weights as `[-w/2, +w/2]` | 6 | 19.07 | 3.18 |

Five independent models, four use cases, dense and convolutional paths, single-row and 512-row
batches: **3.10 to 4.39 ms per rotation**. That is the behaviour a rotation at a fixed
ring degree should show, and it is why this is the portable figure. (The telecom rotate-ms cell is
its reported per-rotation figure times its rotation count rather than a separately displayed total;
the per-rotation value itself is measured.)

At ring degree 16384 with the same coefficient modulus, the same finance measurement gives **6.73 ms
per rotation** - 2.17x, for a doubled ring.

## A claim this round got wrong, and then measured properly

Two comments in `clientFHE.ts` and one in `clientCompute.ts` asserted that key-switching dominates
CKKS cost. The first version of this document said the claim held for the dense layer and failed for
the convolution - rotations 57% against 10% - and concluded that im2col encoding dominated the
convolution instead. **That was wrong, and the original comments were right.**

`fhe_conv_rotation_share` divided the convolution block's rotation time by
`encryptedStudentInference`'s `fheMs`, which is the WHOLE student: the convolution plus its two dense
layers. `encryptedStudentInference` returns `{...conv, fheMs, ...}`, so `rotations` and `rotateMs`
come from the spread and describe the conv block, while `fheMs` is overridden with the accumulated
total. A numerator and a denominator of different scopes, and the metric name `fhe_conv_ms` said
convolution while carrying the student's total.

Measured with the scopes separated, medical page, one run:

| stage | rotations | rotate ms | stage ms | rotation share |
|---|---|---|---|---|
| bundled linear model (`realClientHybrid`) | 3 | 13.17 | 23.2 | **57%** |
| published student, convolution | 16 | 67.81 | 96.60 | **70%** |
| published student, dense layers | 136 | 421.40 | 490.23 | **86%** |
| published student, total | 152 | 489.21 | 586.83 | 83% |

The two stages sum to the total exactly - 96.60 + 490.23 = 586.83 - so nothing is unattributed.

So key-switching dominates everywhere, at 57% to 86% depending on how much non-rotation work the
stage carries, and the per-rotation cost is 3.2-4.2 ms throughout. The page now reports
`fhe_conv_ms` (the convolution), `fhe_encrypted_student_ms` (the total),
`fhe_student_dense_ms` and a rotation share per stage, so the two scopes cannot be divided into each
other again.

**How this was caught matters more than the number.** It was not caught by a test, by the build, or
by reading the page - every figure rendered plausibly. It was caught by an adversarial review of the
round's own diff, asking of each new metric: over what set is this computed, and does the sentence
beside it describe that same set. That is the question the round was created to ask about other
people's numbers, and it had not been asked of its own.

## The published comparator, which does exist

The plan for this round expected to find none and to record the absence as the finding. That was
wrong: one primary source publishes browser-WASM CKKS per-primitive timings at both ring degrees.

**node-seal `BENCHMARK.md`**, https://github.com/s0l0ist/node-seal/blob/main/BENCHMARK.md, last
updated 2025-03-27. Fetched and read directly rather than via a summary. Its header states: MacBook
Pro 14-inch 2021, Apple M1 Pro (8P+2E), 32 GB, macOS Sequoia 15.3.2, arm64; Microsoft SEAL v4.1.2,
node-seal v5.1.5, Chrome 134.0.6998.166, Firefox 136.0.3, Safari 18.3.1. All figures in
microseconds, 10 iterations per benchmark.

CKKS `EvaluateRotate`, quoted verbatim:

```
| n=8192/log(q)=218/CKKS/EvaluateRotate/iterations:10  |  1836 |  4339 |  4577 |  6830 |  5400 |  4700 |
| n=16384/log(q)=438/CKKS/EvaluateRotate/iterations:10 | 11458 | 26477 | 27930 | 41200 | 32200 | 27800 |
```

Columns are Native C++, Node, Bun, Chrome, Firefox, Safari.

### Which of the two rows is a fair comparison, and which is not

**n=8192 is comparable.** Same ring degree, same library, both Chrome, both WASM. The coefficient
moduli differ - 218 bits published against 200 here - which is a similar limb count, so the
comparison is like-for-like enough to state. Published Chrome 6.83 ms against 3.10-3.24 ms measured
here: this host is roughly **2.2x faster** than the published M1 Pro figure. Different CPU and a
slightly smaller modulus both point the same way, and neither is quantified here, so the honest
reading is "the same order, this host faster", not a speedup claim.

**n=16384 is NOT comparable, and quoting it as one would be the error this document exists to
avoid.** The published row is at log(q)=438; this project runs 200 bits at that ring. Key-switching
cost grows with the number of RNS limbs, so most of the gap between the published 41.2 ms and the
6.73 ms measured here is the modulus, not the platform. The two numbers are recorded together only
so the difference in parameters is visible.

### Caveats carried from the source

Ten iterations per benchmark and no error bars. Chrome's timer is quantised to 10 us and
Firefox/Safari to 100 us, which makes the cheap operations in that table unusable (several literal
`0.000` entries) but is negligible against rotation timings in the thousands of microseconds. The
file does not state whether WASM SIMD or threads were enabled. The Chrome encrypt rows are an 11x
outlier that Firefox and Safari do not reproduce and should be treated as a measurement artefact;
the rotation rows agree across all three browsers within about 1.5x, which is why rotation is the
defensible row to cite.

### Native and GPU context

Ozcan and Savas, "GPU Implementations of Three Different Key-Switching Methods for Homomorphic
Encryption Schemes", IACR ePrint 2025/124, https://eprint.iacr.org/2025/124.pdf. Table 5, CKKS
relinearization: single-thread Microsoft SEAL on a Ryzen 9 7950X3D gives 1641 us at ring 8192 and
14563 us at 16384; HEonGPU on an RTX 4090 gives 41.8 us and 84.1 us.

This measures relinearization, not rotation. Both are key-switching, and node-seal's own table shows
the two within 1% of each other at both ring degrees (8192: relin 1832 against rotate 1836; 16384:
11292 against 11458), so it is a sound proxy - but it is cited as relinearization.

The paper's CPU SEAL figure at ring 8192 (1641 us, Ryzen) against node-seal's native column (1832
us, M1 Pro) is a useful independent check that the node-seal numbers are per-operation rather than
totals over ten iterations: had they been totals, the native figure would be 183 us and no CPU
difference would explain a 9x gap.

## End-to-end latency against the operational window

Total encrypted latency is not comparable across deployment classes, but it is checkable against the
requirement each vertical actually has. Fraud scoring, clinical decision support and compliance
checks are interactive-batch work - seconds, not milliseconds.

Neither number the page previously showed answered this. The benchmark panel's `total` is
`fhe_linear_ms + mpc_nonlinear_ms`, both scaled by `benchScale()`, so it is a projection rather than
a time anything took. And the line under it, labelled "client compute (wall-clock)" and reading
0.07 ms, is the plaintext reference inference only - which, sitting directly beneath a 700 ms total,
read as the whole computation costing 0.07 ms in the browser. That label now says what it is.

`encrypted_tier_measured_ms` is the measured sum: CKKS linear, plus the convolution where there is
one, plus the secret-shared non-linear stage.

| use case | measured encrypted inference | inside a 1-10 s window |
|---|---|---|
| finance | 700 ms | yes |
| medical | 1,269 ms (of which 579 ms is the encrypted convolution) | yes |

Both are inside it with room. The page reports the verdict rather than leaving two numbers for a
reader to compare, and a page that leaves the window will say so.

## One-time cost a visitor feels

CKKS key generation runs once per tab, synchronously on the main thread, and `clientFHE` already
recorded that it made the first click feel like a hang. Measured: **242-255 ms at ring 8192**, and
**477 ms at 16384**. That cost is the main reason the 16384 experiment was reverted; see
`src/qsh/config/defaults.py` for the full before/after table and the reasoning.

## What is not claimed

No accuracy comparison appears here. Every published model in this project already carries a cited
baseline and ceiling in its `MetricSpec` and is re-scored row by row in the visitor's browser against
a recorded figure with a stated tolerance and a loud failure. Repeating that here would duplicate a
check that already runs.

No total-latency comparison against GPU encrypted inference appears here either, and that is
deliberate rather than an omission.
