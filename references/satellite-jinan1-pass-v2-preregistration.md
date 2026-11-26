# Pre-registration: the Jinan-1 pass, second analysis (2026-11-26)

Written and committed before any figure below was computed, on the owner's choice of 2026-11-26 to show the post-hoc
comparison of `references/satellite-jinan1-pass-preregistration.md` (its addendum 2) and to pre-register a second
analysis. Nothing here is changed after a figure is seen; a change is an addendum with its date and reason. What is
already known when writing it: the first analysis's figures (the record
`docs/measurements/satellite-jinan1-pass-posthoc-2026-11-26.json`), and two facts read in the preprint's Methods
(arXiv:2408.10994v1): an 800-ps gate that keeps "a ~80% gating efficiency for quantum photons", and ten percent of the
detected quantum photons sampled to evaluate the QBER, the other ninety percent used for the key.

## Windows

Addendum 2's, now fixed in advance: the background is the median count rate over the file's first and last 30 seconds,
the link seconds are those whose counts exceed five times it, and a second's measured detections are its counts less
the background. The model is the `jinan1_2022_jinan` profile through the page's link budget, as in the first analysis.

## The questions

1. **The atmosphere.** The station efficiency and the profile's zenith atmospheric loss are fitted together, by least
   squares on the per-second residual in dB over the link seconds, the atmospheric loss scaled by the model's air
   mass (Kasten and Young) as the profile scales it. No third elevation-dependent term is fitted: over this pass the
   range and the air mass are nearly collinear. Reported: both fitted values with their standard errors and correlation,
   the residual deviation, and the change from the first analysis's one-parameter fit.
2. **The sifted bits.** The model's sifted bits over the pass are its signal-state detections per second, times 0.8 for
   the gate and 0.9 for the share not sampled, times one half for the basis match, summed over the link seconds. They
   are set beside the measured 2,000,000 (the 20 packets of 100 kbit that sheet `4c`'s QBER changes mark). Reported:
   the ratio, and the time lag that best aligns the model's cumulative sifted bits with the 20 packet boundaries.
3. **The keys by block.** The measured key of each 500-kbit block is the step of sheet `4c`'s accumulating column (four
   steps). The model's finite key for the same block is the page's bound (`finite_key_over`) on the model's counts over
   that block's seconds, with the gate and sampling factors of question 2. Reported: the four ratios and their mean.
4. **Out of sample.** For the three other passes of the preprint's Table 1 (20, 28 and 29 September 2022: maximum
   elevations 41.8, 77.7 and 52.7 degrees; 17, 31 and 20 packets; 330,240, 571,392 and 436,096 bits), the model's
   packets (its sifted bits over 100 kbit) and finite key on a pass simulated at each maximum elevation, with the same
   factors and either set of constants, against the published figures.

## Rules

1. The profile takes the fitted pair of question 1 only if the residual deviation falls by at least 0.05 dB and the
   fitted zenith loss is between 0 and 3 dB; otherwise it keeps its values, and the record says which held.
2. The page's key model gains the gate and sampling factors only if, with them, the model's sifted bits on all four
   passes come within a factor of 1.5 of the published packets; otherwise the factors are reported and the page keeps
   its model, with the card saying so.
3. Whatever the outcome, the satellite card states the second analysis's figures beside the first's.

The comparison is plaintext; nothing here touches CKKS.
