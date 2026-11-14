# Differential privacy on the demo: what was read, and what each source supports

Read at the sources on 2026-11-13 for the owner's question of whether differential privacy is recommended for each use
case and, where it is, how to implement it faithfully in a browser. "Agent-read" marks a figure a research agent read
and this document did not read again; every other statement was read here at the source named.

## Guidance

- **NIST SP 800-226, "Guidelines for Evaluating Differential Privacy Guarantees"**, final, 6 March 2025; the page lists no
  later revision (https://csrc.nist.gov/pubs/sp/800/226/final). Read in the PDF: a study it cites finds
  $`\epsilon = 0.1`$ strong and values below 1 reasonable, deployments have used $`1 < \epsilon \le 20`$, and larger
  values call for an assessment of the release's risk; for $`\delta`$ the common recommendation is $`\delta \le 1/n^2`$,
  its key takeaway repeats that rule for conversions, it says neither that rule nor $`1/(n \log n)`$ suits small
  $`n`$, and it calls values above $`10^{-5}`$ suspicious unless carefully justified; floating-point arithmetic is
  listed as a privacy hazard of implementations.

## Mechanisms and accounting

- **Kairouz, Liu and Steinke, "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure
  Aggregation"**, ICML 2021 (arXiv:2102.06387). Read in the PDF: Theorem 1 and Proposition 14 bound the sum of $`n`$
  parties' discrete Gaussians as $`\tfrac{1}{2}e^2`$-concentrated DP with the two-term minimum and the $`\tau`$
  correction that `web/src/lib/clientDp.ts` implements; the guarantee is for addition or removal of one individual,
  with $`n`$ the number of trustworthy clients, and assumes the secure aggregation reveals only the sum; concentrated
  DP converts to $`(\epsilon, \delta)`$-DP as in Canonne, Kamath and Steinke (below).
- **The conversion (from 2026-11-14).** $`\rho`$-zCDP is $`(a, a\rho)`$-Renyi DP at every order $`a > 1`$, and
  Canonne, Kamath and Steinke's Proposition 12 (arXiv:2004.00010, the form Google's dp-accounting 0.7.0 applies)
  gives $`\epsilon = \min_{a>1}\, a\rho + \log(1 - 1/a) - \frac{\log\delta + \log a}{a-1}`$, which the pages
  minimise by golden-section search on $`\log(a-1)`$, capped by the simpler
  $`\epsilon \le \rho + 2\sqrt{\rho \log(1/\delta)}`$ (Bun and Steinke, TCC 2016, Proposition 1.3) they used
  until then. At the same $`\epsilon`$ the noise is 7 to 18 percent smaller; $`\rho = 1.0491`$ at
  $`\delta = 10^{-5}`$ is 7.28 rather than 8, the figure a separate numpy and scipy accountant gave for the murmur
  study.
- **Canonne, Kamath and Steinke, "The Discrete Gaussian for Differential Privacy"**, NeurIPS 2020 (arXiv:2004.00010):
  Algorithms 1 to 3, exact sampling of Bernoulli$`(e^{-\gamma})`$, the discrete Laplace and the discrete Gaussian from
  uniform integers in rational arithmetic, which the page implements in BigInt and Python in integers.
- **Bun and Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds"**, TCC 2016
  (arXiv:1605.02065): zero-concentrated DP composes by adding $`\rho`$, which is how the pages count several releases
  about the same people.
- **Mironov, "On Significance of the Least Significant Bits for Differential Privacy"**, CCS 2012: textbook
  floating-point Laplace noise leaks the value it hides through the low-order bits; the reason the page samples in
  integers.
- **Libraries, versions read on PyPI on 2026-11-13**: Opacus 1.6.0 (5 May 2026, Apache-2.0), dp-accounting 0.7.0
  (10 November 2026, Apache-2.0), OpenDP 0.16.0 (6 October 2026, MIT; Rust with Python and R bindings, no JavaScript).
  No maintained JavaScript library was found (agent-read), so the page has its own sampler and accountant, held to a
  Python twin.

## Federated learning

- **Record level, not hospital level.** With four hospitals, a guarantee per hospital would need noise as large as a
  hospital's whole update; DP-FedAvg's client-level guarantee was demonstrated over 763,430 users (McMahan et al.,
  arXiv:1710.06963; agent-read). The page's unit is one patient.
- **FedAvg's local steps.** FedAvg's 100 local SGD steps a round have no ready per-patient bound (agent's reading), so
  the private arm takes one full-batch step per secure aggregation; its no-noise row isolates what the noise costs.
- **FLamby's own DP results on Fed-Heart-Disease** (du Terrail et al., NeurIPS 2022, appendix; read off its plots by
  the agent): about 0.71 without DP, about 0.685 at $`\epsilon = 1`$ and about 0.61 at 0.1 ($`\delta = 10^{-4}`$),
  with per-hospital DP-SGD through Opacus. The page's arm differs in design (distributed noise in the secure sum,
  full-batch steps), so the figures are context, not a reproduction.

## Measured here

- Medical, seed 1, 20 draws of private noise (`qsh.ai.fed_heart.run_dp`), with the conversion of 2026-11-14: 0.721
  with no noise; $`0.724 \pm 0.008`$ at $`\epsilon = 8`$, $`0.725 \pm 0.010`$ at 4, $`0.723 \pm 0.017`$ at 2,
  $`0.718 \pm 0.024`$ at 1 (range 0.667 to 0.767); FedAvg without DP 0.7005. Total noise standard deviation in
  gradient units: 4.87, 8.97, 16.84 and 32.04 (5.2, 9.9, 19.2 and 37.8 under the simpler bound, when $`\epsilon = 1`$
  gave $`0.706 \pm 0.032`$).
- Genomics, one count of at most 60 alleles over 30 people, four cohorts: total noise standard deviation 15.3, 8.1, 4.3,
  2.3 and 1.3 alleles at $`\epsilon`$ = 0.5, 1, 2, 4 and 8 ($`\delta = 10^{-5}`$; 19.4, 9.8, 5.0, 2.6 and 1.4 before).
- Finance (from 2026-11-13), the count of shared accounts as it would be published: one party, sensitivity 1,
  $`\epsilon = 1`$, $`\delta = 10^{-8}`$, so $`V = 30`$ and a standard deviation of 5.5 accounts (the accountant
  gives $`\epsilon = 0.983`$; $`V = 38`$ and 6.2 before), on a count of 60 at the default volume.
- Composition: the medical table's four noised rows published together are $`\epsilon = 9.75`$ at
  $`\delta = 10^{-6}`$ ($`\rho`$ = 1.052, 0.311, 0.088 and 0.024, summed), not 8; the page prints the figure from
  the run. It was 9.56 under the simpler bound: each row now meets its own target with less noise, so carries more
  $`\rho`$, and the sum converts to a slightly larger total.

## Where it does not apply

- **Finance (the PSI's answer).** Which flagged accounts the banks share is the answer the PSI exists to give; noise
  would add false matches and misses (a DP intersection is a different protocol, for non-sensitive intersections;
  agent-read). Since 2026-11-13 the count, as it would be published, carries DP (above); the answer to the banks does
  not.
- **Telecom.** The detector's data are synthetic, generated in the tab; they describe no person.
- **Satellite.** Spacecraft telemetry (ESA-ADB, OPS-SAT) describes no person, and both are public.
- **DeFi.** The released bit is the decision itself; randomized response at $`\epsilon = \ln 3`$ answers wrongly one
  time in four (Dwork and Roth, "The Algorithmic Foundations of Differential Privacy", 2014).
- **The genomics page's own polygenic score.** It goes back to the person it describes; DP protects people inside a
  released dataset.

## The other pages: DP and threshold cryptography, checked 2026-11-13

The owner asked whether DP or further cryptography has research value for the telecom/defense, satellite, DeFi and
PQC-QKD use cases, and to leave the pages as they are where it is not recommended. A research agent read the sources;
the rows marked "read here" were read again at the source for this file, the rest are agent-read.

| page | candidate | evidence | verdict |
|---|---|---|---|
| telecom | DP on the detector or its published models | the tab's data are synthetic; 5G-NIDD's benign flows come from real devices but are published (arXiv:2212.01298, section III-B) | not recommended: it would protect no one whose data are not already public |
| telecom | federated IDS across operators with record-level DP | Tory, Ali, Layek and Hasan, arXiv:2609.03420 (v1, 3 Sep 2026; no venue on arXiv; read here, abstract): DP-SGD with coordinate-wise median on UNSW-NB15, rare-attack detection degraded; the PDF's F1 0.954 without DP and 0.738 at $`\epsilon = 1`$, Worms 20 percent and Shellcode 52.5 percent, are agent-read. 3GPP TS 23.288 V18.5.0, clause 5.3, names neither secure aggregation nor DP for federated learning among NWDAFs (agent-read) | not recommended: simulation only, the rare attacks degrade most, and the page has no federation |
| telecom | DP synthesis of network traces | NetDPSyn, IMC 2024 (arXiv:2409.05249; agent-read) | not applicable: the page releases no traces |
| satellite | DP on telemetry | ESA-ADB and OPS-SAT describe no person | not recommended |
| satellite | MPC conjunction screening across operators | ShareSat, an ESA prototype on Sharemind with three servers, about 7 minutes a conjunction (agent-read, from a 2026 Tartu thesis citing project documents) | not for this page: a different computation, so a different use case |
| DeFi | DP on the bit | the bit is the decision | not recommended |
| DeFi | a t-of-n threshold CKKS committee in place of the one key holder | Zama's KMS documentation (read here): 13 MPC nodes run by different organisations, robust while at most a third act maliciously, every decryption output signed by each node, user decryption re-encrypted for the user. Lattigo v6.2.0 (read here, in the module cache): `Thresholdizer`, `Combiner` and `PublicKeySwitchProtocol` exist; the protocols are secure against passive adversaries only, and a party must not make a second share for the same ciphertext (`SECURITY.md`, "On the insecurity of retries") | implement as an option, the default key holder unchanged (the owner's choice of 2026-11-13) |
| DeFi | threshold ML-DSA for the key holder's signature | NIST's threshold call (read here; updated 2026-11-05): full packages expected 2027-03-01 and not before | not recommended: not standardised; separate per-member signatures match Zama's practice |
| PQC-QKD | threshold ML-DSA or ML-KEM | the same call names threshold ML-DSA schemes and no ML-KEM; the page makes a fresh ML-DSA key each run, so there is no long-lived key to split | not recommended |
| PQC-QKD | the key split over node-disjoint relay paths | Clermont, Gansel and Struck, "Key Transport over Untrusted QKD Relay Networks", ePrint 2026/1095 (read here, abstract; "Major revision. SCN 2026", so acceptance is not confirmed): proactive secret sharing with one-time pads over pairwise QKD links, information-theoretically secure against a semi-honest adversary corrupting up to $`t-1`$ nodes in any one layer, and a cross-layer attack on a prior construction. A routing study at OFC 2025 (W1J.6) and an expired individual IETF draft (draft-li-ipsecme-qkd-multipath-secret-sharing) are agent-read; no deployment was found | add, as a two-path XOR split in the SDN panel (the owner's choice of 2026-11-13): the only candidate that weakens an assumption the page states, that the relay sees the key |

What rests on searches rather than proof: that no cross-operator DP IDS, no DP for QKD-network monitoring or lending,
and no fielded multipath relay was found.

