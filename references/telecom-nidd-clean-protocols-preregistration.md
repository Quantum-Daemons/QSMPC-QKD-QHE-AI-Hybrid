# Pre-registration: the 5G-NIDD models under leakage-aware protocols (2026-11-27)

Written and committed before any model is fitted under these protocols (Round 86, batch 9), on the owner's choices of
2026-11-27 (three panels: the caveat and the re-run; the re-run's protocols and checks; its federation split, models,
validation and the cross-testbed data). Nothing here is changed after a figure is seen; a change is an addendum with
its date and reason.

## Why

Every 5G-NIDD model this project ships (the teacher, the CKKS student, the two-station federation and the eight-client
comparison) is trained and scored on a random stratified split of flows, and six of its inputs are columns Hameed and
Ibrahim, "LEAF", J. Cybersecur. Priv. 6(6):205 (15 November 2026), find stand in for the label: `sHops`, `dHops`,
`sVid`, `dVid`, `SrcTCPBase`, `DstTCPBase`. They report that three published models lose 2.96 to 7.39 points of
accuracy on their stricter protocol. The two columns two other 2026 studies find leak the most, `Seq` and `Offset`
(Ozyilmaz and Okay, ISCTurkiye 2026; Abraheem et al., Wadi Alshatti Univ. J. Pure Appl. Sci. 4(2), 2026-07-18), are
not inputs here, and the loader removes the 281,529 contradictory Benign copies. The cards state the limits
(checklist MS32). This measures what the shipped recipes give under protocols that keep a capture session, or later
traffic, out of training. It changes nothing the page ships until the owner chooses.

## What the file allows (checked on 2026-11-27, no model fitted)

`BTS1_BTS2_fields_preserved.csv`: 17 capture files, contiguous, each Benign and one attack at one base station; `Seq`
restarts in most files and is monotone in none; `StartTime` holds minutes and seconds only; within every file the row
order is chronological (97.8 to 99.97 percent of consecutive rows move forward in time).

## Data

`D.load_5g_nidd(with_identifiers=True, with_file_label=True)`: the shipped rows (identifiers and the file label not
inputs; the contradictory copies removed; the file's order kept). The trailing `Seq` and `IdleTime` columns are used
only to order and group rows (protocol P3; the capture file), never as inputs. A row's station is the federation's:
before `D.NIDD_SECOND_FILE_STARTS_AT_CLEANED`, station 1; from it, station 2.

## Protocols

- **P0, the shipped split (the reference).** 75/25 stratified at seed 42, then a stratified 10 percent of the
  training part at seed 0 as validation.
- **P1, cross-station.** Train on one station, test on every row of the other; both directions, each reported.
  Validation: per class, the last 15 percent of the training station's rows in file order; training on the rest.
- **P2, a time split in each capture file.** Within each (capture file, class) group, in file order: the first 70
  percent training, the next 15 percent validation, the last 15 percent test.
- **P3, LEAF's rule on this export (secondary).** Per class over the whole file, a stable sort by `Seq` (ties keep the
  file's order): the first 70 percent training, the next 15 percent validation, the last 15 percent test. LEAF reads a
  different export (`Combined.csv`), so P3 is comparable with its Table 1 in rule, not in rows.

In every ordered group of $`n`$ rows the cuts fall at $`\lfloor 0.70\,n \rfloor`$ and $`\lfloor 0.85\,n \rfloor`$
(P1's validation: the rows from $`\lfloor 0.85\,n \rfloor`$ on, per class of the training station). A class absent from
a test part is left out of that part's macro-F1, and the record says which.

## Input sets

- For the linear models (36 numeric inputs): **F36** as shipped; **F30**, F36 without the six; **F28**, F30 without
  `sTtl` and `dTtl` (TTL is a known shortcut, Jacobs et al., CCS 2022; LEAF keeps it).
- For the teacher (46 inputs, the ten categorical ones ordinal-encoded as shipped): **T46**, **T40**, **T38**, the same
  removals.

## The single-feature check, before any model here is fitted

On each protocol's training part, for each input and each class one against the rest, one threshold fitted on the
training part and scored by F1 on the validation part. An input is flagged for a class where that F1 reaches 0.99. All
values are reported; no input set above is changed by them.

## The arms

1. **Teacher.** The shipped XGBoost recipe (300 trees, depth 8, learning rate 0.1, subsample 0.9, column sample 0.9,
   `hist`, no early stopping, as shipped), on T46, T40 and T38 under P0, P1 (both directions), P2 and P3; seeds 0 to 4.
2. **CKKS student.** The shipped student's procedure (standardised on the training part, clipped to $`[-20, 20]`$, the
   penalty from $`10^{-6}`$ to $`10^{-1}`$ by decades chosen by validation log loss, a temperature fitted on
   validation), on F36, F30 and F28 under P0, P1, P2 and P3; deterministic, one run each. Under P2 on F30, its CKKS
   agreement is re-checked as the page's (2,048 held-out rows, decisions changed, the largest logit error).
3. **Two-station federation.** The shipped masked FedAvg (40 rounds of 20 local L-BFGS-B iterations, each station
   weighted by its training rows, every masked message checked to cancel exactly), each station on its own P2 parts,
   on F36, F30 and F28; one run each.
4. **Square MLP (new).** 36 or 30 or 28 inputs, one hidden layer of 64 units with $`x^2`$ as its activation, nine
   outputs; Adam at $`10^{-3}`$, batch 512, unweighted cross-entropy, 30 epochs, the epoch reported chosen by validation
   macro-F1; under P1 and P2 on F30; seeds 0 to 4. On the GPU only on the owner's approval of that job, else on the CPU;
   the device is recorded and changes no rule.
5. **Cross-testbed (NetsLab-5GORAN-IDD, Zenodo 18923275, CC BY 4.0).** Its flow CSV is Zeek's connection log, with none
   of 5G-NIDD's Argus inputs by name. This project's map, fixed here: `Dur` from `duration`; `SrcPkts`, `DstPkts` from
   `src_pkts`, `dst_pkts`; `SrcBytes`, `DstBytes` from `src_ip_bytes`, `dst_ip_bytes`; `TotPkts`, `TotBytes` their
   sums; `Proto` from `proto`; and the rates and loads by Argus's definitions, as checked on 5G-NIDD's own rows on
   2026-11-27 (no model fitted): $`\mathrm{Rate} = (\mathrm{TotPkts} - 1) / \mathrm{Dur}`$ and likewise `SrcRate`
   and `DstRate` from their own counts (exact on every row checked);
   $`\mathrm{SrcLoad} = 8\,\mathrm{SrcBytes}\,(\mathrm{SrcPkts} - 1) / (\mathrm{SrcPkts}\,\mathrm{Dur})`$ and likewise
   `DstLoad` (within $`10^{-3}`$ on 97.9 percent of the rows checked); `Load` their sum (exact); every rate and load 0
   where the duration is 0 or the count is one packet, as in the file. A teacher and a student are trained on
   5G-NIDD's P2 training part with these inputs only and scored on NetsLab without retraining, on the classes both
   share (`syn` SYNFlood, `http_flood` HTTPFlood, `udp` UDPFlood, `icmp` ICMPFlood, `slowloris` SlowrateDoS,
   `portscan_udp` UDPScan, `portscan_tcp` as one TCP-scan class, merged at scoring: a prediction of SYNScan or
   TCPConnectScan counts as it, the models being the nine-class ones,
   `benign` Benign) and as Benign against any attack. If the map's distributions are judged too far apart to score,
   the PCAPs (16.4 GB) and Argus re-extraction go to the owner first.

## Metrics

Macro-F1 over the classes present in the test part, weighted F1 and accuracy; under P1 each direction and their mean;
under P2 and P3 each station's macro-F1 and their mean (FLamby's metric); for seeded arms, the mean and standard
deviation over seeds.

## Stated before any figure

- **G1, the protocol gap.** For each model on its shipped inputs, P0 minus P1 (the mean of the directions) and P0
  minus P2, with a 95 percent t interval over seeds for the seeded arms; stated as measured.
- **G2, the six columns.** F36 against F30 (T46 against T40) under P0, P1 and P2, as measured.
- **G3, TTL.** F30 against F28 (T40 against T38) under P1 and P2, as measured.
- **G4, across testbeds.** The cross-testbed figures, as measured.

None decides what ships by itself. The page shows the clean protocols' figures beside the current ones; a Hub
republish waits for the owner's approval. Every figure is kept in
`docs/measurements/telecom-nidd-clean-protocols-<the run's date>.json`, with the single-feature check's values.

## Not done here

No hyperparameter is retuned beyond each recipe's own selection; no oversampling; no new features; no network is
evaluated under CKKS except the student's re-check in arm 2.

## Compute

The teacher trains on the GPU and scores on the CPU, as shipped, on the owner's approval of that job, else on the CPU;
the square MLP likewise; the linear arms and the federation on the CPU. The device is recorded and changes no rule
above.
