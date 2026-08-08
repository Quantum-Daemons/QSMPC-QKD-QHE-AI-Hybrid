# Security scoring: sources evaluated and declined (Round 22, 2026-08-08)

Round 22 asked whether this project's 0-100 security score is a defensible, publicly grounded,
quantum-resistance-aware method or an invented formula, and named five candidate replacements.

The answer to the first question was: invented. The answer to the second was: adopt none of them.

This file records the verdicts so a later round does not re-evaluate the same five sources and reach
the same conclusion at the same cost. Each was read directly, not summarised from search results -
which mattered, because the search summaries of the first two were considerably more flattering than
the papers.

## Why the existing score had to go

Of 17 fields in `SecurityScoringConfig`, exactly one is traceable to a citation (`leakage_low 0.1`,
Sulimany et al., PRX 15 041056). Two are traceable by name only. Fourteen are unexplained and none
carries a stated rationale. The maxima sum to exactly 30+30+20+10+10 = 100, which is the signature of
a number designed backwards from its total.

The headline 68-to-96 gap was mostly an artefact of the call sites rather than of the cryptography:
+12 for a substring match on a hand-written label, +10 because the classical call passed no QBER
while the quantum call passed a literal, +6 for the same pattern on leakage. Sixteen of the 28 points
were a difference in which arguments two functions chose to supply.

It also barely moved - 2 of 5 selectors reached it, giving 4 distinct values on a 0-100 axis whose
bottom 63 percent was unreachable - and in the Python path both rows were literal constants for every
configuration, because the caller passed neither the KEM nor the QKD key length.

## The five candidates

| Source | What it actually is | Verdict |
|---|---|---|
| QERS, arXiv:2601.13399 | Weights are author-selected across three scenarios; the paper says ML "could" optimise them and gives no calibration methodology. Normalisation bounds come from an ESP32C6 plus Raspberry Pi Wi-Fi testbed. Three of seven inputs are RSSI, packet loss and energy - none exists in a browser. Scope is resource-constrained IoT/IIoT and the paper states applicability beyond it "remains unaddressed". | Decline |
| CARS, IACR ePrint 2026/1467 | No formula published. Weights come from a two-round Delphi process with 12 engineers. "Calibrated against 43 repositories" describes measurement of codebases, with no optimisation procedure and no validation against ground truth. Scores a codebase's migration readiness - version-level and static, not per-run. | Decline |
| quantum-safe library assessment, arXiv:2605.17061 | A ternary Full/Partial/None rubric over eight dimensions, scoped to libraries. Explicitly unsuitable for scoring a single operational system's security configuration. | Decline |
| VIAVI blog | Vendor marketing. No formula, no thresholds, no acceptance criteria; promotes named test products. | Decline |
| fierce-network (sponsored), pqshield, qtonic, qodequay | The same genre - one sponsored article, two product pages, one vendor guide. The most methodological-sounding was sampled and was marketing. | Decline |

Adopting QERS or CARS would have replaced invented weights with different invented weights, and
added a citation implying a rigour that does not transfer to this setting. That is worse than the
prior state, because the prior state at least did not claim external grounding.

## What was adopted instead

One idea from arXiv:2605.17061 transfers, without any of its content: **per-dimension verdicts
sourced from specifications, rather than a synthesised number.** This project already used that shape
successfully in the provenance badge's five-state enum.

So the composite was retired in favour of reported facts, each traceable to a standard or to this
run:

- NIST security category as the headline - ML-KEM-768 is Category 3, ML-KEM-1024 is Category 5,
  FIPS 203. This is the one figure here that is standard, citable, and comparable against another
  system.
- Signature level (ML-DSA-65 / -87, FIPS 204), whether QKD key material is mixed into the session
  key, secure key bits and SKR as measured by the run, and the QBER verdict.
- The performance overheads the run already measures: FHE-linear ms, MPC-non-linear ms,
  communication KB, ciphertext/plaintext ratio, KEM ciphertext bytes, signature bytes.

Every one of these is per-use-case verifiable in the E2E simulator, which a synthesised score is not.
That was the deciding argument.

## Defects fixed regardless of presentation

- `secFromConfig` returned no `qber` field, so the consumer guard `typeof sec?.qber === "number"` was
  always false and a constant was used unconditionally - under a comment claiming the Eve slider had
  been wired through. The slider never reached the verdict; one literal had been replaced by another.
- The Python call sites passed neither `kem` nor `qkd_key_bits`, making both rows constant.
- The configured QBER threshold 0.11 sits **above** this repository's own measured GLLP zero-key
  point of 0.098106, so a channel in [0.098, 0.11) scored full marks while producing zero key.

## Related

The claim in `README.md` that the security score was "the production-relevant number" was withdrawn
in the same round.
