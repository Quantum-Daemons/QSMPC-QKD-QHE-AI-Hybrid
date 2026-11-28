# Pre-registration: the PFCP held-out-day check's labels audited, its application-layer flows, and normal-only detectors

Written and committed on 2026-11-28, on the owner's choices of that day (a label and endpoint audit first, the
application-layer features, normal-only detectors), after the held-out-day check
(`references/telecom-pfcp-held-out-day-preregistration.md`, record
`docs/measurements/telecom-pfcp-held-out-day-2026-11-28.json`), and frozen at its SHA-256 in the script before any
model below is fitted and before the audit runs.

## What is known, said first

On the CICFlowMeter UPF flows, the held-out-day check found the session-modification days recognised (AUC 0.99 to
1.00 for the student and the teacher) and the session-establishment day not (the student's AUC 0.01, the teacher's
0.65 on average, with about a third of that day's Normal flows above the threshold); the deletion day lay between. A
comment on the dataset's IEEE DataPort page questions the deletion run's labels. No published study tests this dataset
on an unseen attack day. Read before this was written, and nothing else: each run's application-layer PFCP flow files
at the 120-second timeout (`PFCP Flows/120s-timeout/`), their columns and their label counts; the UPF file holds 234
(establishment), 245 (deletion), 259 (modification, DROP) and 244 (modification, DUPL) flows, of which 120, 115, 130
and 122 are Malicious. Its columns are an unnamed index, the flow ID, the source and destination addresses and ports,
the protocol, the duration, the forward and backward packet counts, 26 counters of PFCP message types, and three
direction counters.

## The questions

Are the labels consistent with the addresses and with each attack's message type? Does a detector read from the PFCP
messages themselves recognise an attack day it never saw? Does a detector fitted on Normal flows only?

## The data

The four runs of the held-out-day check, read in place from the same archives. Two inputs per run:

- CIC: the held-out-day check's UPF CICFlowMeter flows and its inputs, exactly as that pre-registration states.
- APP: the run's `PFCP Flows/120s-timeout/UPF-120s-FLOW.csv`, labelled Normal or Malicious. Inputs, by name with
  surrounding spaces removed: the duration, the forward and backward packet counts, the 26 message counters and the
  three direction counters that every run's file has, without any column constant over the training days; never the
  index, the flow ID, the addresses, the ports or the protocol. Values as the held-out-day check reads them: empty,
  infinite or NaN is 0, and a row whose field count differs from its header's is dropped and counted.

Each run is the held-out day in turn, the other three the training days, split as the held-out-day check splits them:
each training day's first 80 percent of flows in file order fitted on, its last 20 percent the validation part.

## The arms

- **A, the audit.** No model, run first, changing no arm. For each run, on the CIC and the APP UPF files: the flows
  per label; the number of distinct source addresses, destination addresses and address pairs among each label's
  flows, and how many of each appear under both labels; and the share of flows whose label is their address pair's
  majority label. On the APP file only: for each label, the share of flows with a non-zero counter of the run's attack
  message (session establishment: `PFCPSessionEstablishmentRequest_counter`; deletion:
  `PFCPSessionDeletionRequest_counter`; both modification runs: `PFCPSessionModificationRequest_counter`). Counts and
  shares only; no address is recorded.
- **P, the application layer.** The held-out-day check's two arms on APP, with nothing else changed: the 5G-NIDD
  teacher's recipe, its hyperparameters unchanged, two classes, seeds 0 to 4, on the CPU; and the linear student's
  procedure, its L2 from its grid on the validation part, a temperature fitted there.
- **O, normal-only.** On CIC and on APP. Each detector is fitted on the fitting part's Normal flows only, and its
  threshold is set at a false-positive rate of 1 percent on the validation part's Normal flows; no Malicious flow is
  read before the held-out day is scored. Inputs standardised by the fitted Normal flows' mean and standard deviation
  (a zero deviation taken as 1) and clipped at 20. Higher scores are more anomalous. Hyperparameters, fixed here:
  - IForest: scikit-learn's `IsolationForest(n_estimators=100, max_samples="auto", random_state=seed)` for seeds 0 to
    4, the score the negative of `score_samples`.
  - kNN: the mean Euclidean distance to the 5 nearest fitted Normal flows.
  - HBOS: for each input, 10 equal-width bins over the fitted Normal flows' range; a value's bin share is
    $`(c + 1)/(n + 10)`$, with $`c`$ the fitted flows in its bin and $`n`$ all of them, and a value outside the range
    takes $`1/(n + 10)`$; the score is $`-\sum_j \log p_j(x_j)`$ over the inputs.

## What is measured

As in the held-out-day check, on each held-out day and for each model or detector: the AUC of the Malicious class; the
detection rate at the threshold fixed on validation; the false-positive rate that threshold gives on the held-out
day's Normal flows; each with a 95 percent interval from 2,000 resamples of the held-out day's flows
(`numpy.random.default_rng(0)`). For the teacher and IForest, the mean, minimum and maximum over seeds.

## Stated before any figure

- No criterion decides anything: the telecom page and its models do not change, and whether a card shows a result is
  the owner's choice after it.
- The days are small, so the intervals will be wide. A day with no Malicious or no Normal held-out flow is reported as
  not scorable.
- Expected, without a prediction of size: the application-layer counters name the message type, so arm P may do
  better than CIC on the modification and deletion days; a normal-only detector may flag a whole day whose Normal
  traffic differs from the training days', as the establishment day's did for the supervised arms.

## Not done here

No pcap is parsed, no other timeout or entity's file is used, and no hyperparameter is tuned on a held-out day. A
second 5G-core dataset is not tested here: neither candidate downloaded on 2026-11-28 ships a label per record in a
form these detectors can read (5G-Attacks labels packets by the attacker's address in its own preparation code;
CTD5G ships packet captures and attack logs timed to the second), so testing either needs labels reconstructed first,
which the owner is asked about.

## Compute

CPU, minutes. Script `scripts/check_pfcp_audit_app_layer_normal_only.py`; record
`docs/measurements/telecom-pfcp-audit-app-layer-normal-only-<the run's date>.json`, aggregate figures only, with
the deposit's DOI and the archives' SHA-256.
