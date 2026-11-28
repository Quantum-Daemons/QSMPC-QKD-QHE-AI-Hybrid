# Pre-registration: a detector of PFCP attacks, tested on an attack day it never saw

Written and committed on 2026-11-28, on the owner's choices of that day (download the 5GC PFCP Intrusion Detection
Dataset; pre-register its check; freeze it), and frozen at its SHA-256 in the script before any model below is
fitted. The EDA before it read the archives' file lists, the CSVs' columns and the label counts of a few files.

## What is known, said first

The dataset (Zenodo 7888347, CC BY 4.0; K3Y, the SANCUS project under Horizon 2020 grant 952672) records four PFCP
denial-of-service attacks between a malicious SMF and the UPF of an emulated 5G core, each a 4-hour run on its own
date: session establishment on 2022-10-05, session deletion on 2022-10-13, session modification with the DROP flag on
2022-11-01 and with the DUPL flag on 2022-11-22. Each run ships CICFlowMeter flow statistics per network function,
labelled Normal or Malicious, at several flow timeouts, and pcaps. The files are small: the deletion run's UPF file at
the 120-second timeout holds 403 flows (274 Normal, 129 Malicious). The telecom page's 5G-NIDD models read Argus
flows of another testbed, so none of them can score these; this measures a new detector on new inputs.

## The question

Does a detector trained on three PFCP attack days recognise the fourth's attack?

## The data

Each run's UPF flows at the 120-second timeout (`120-sec-CSV/UPF.csv` under the run's CICFlowMeter folder, whose
name the archives spell three ways), labelled Normal or Malicious; anything but four runs found stops the check.
Inputs: CICFlowMeter's numeric features that every run's file has, by name, without the identifiers (the unnamed
index, Flow ID, the source and destination addresses and ports, the timestamp) and without any column constant over
the training days; a value written as empty, infinite or NaN is taken as 0, and a row whose field count differs from
its header's is dropped and counted. Each run is one held-out day in turn: the other three are fitted on, their last
20 percent of flows in file order the validation part. The student's inputs are standardised by the fitted part's
moments and clipped at 20, as the telecom students' are; the teacher reads them raw, as the 5G-NIDD teacher does.

## The arms

- **Teacher:** the 5G-NIDD teacher's recipe (`scripts/train/telecom_nidd.py`), its hyperparameters unchanged, two
  classes, five seeds (0 to 4), on the CPU (the recipe names the GPU; its `device` is set to the CPU here).
- **Student:** the 5G-NIDD linear student's procedure (`scripts/train/telecom_nidd_student.py`: multinomial logistic
  regression, its L2 chosen on the validation part from its grid, a temperature fitted on validation), deterministic.

## What is measured

On each held-out day: AUC of the Malicious class; the detection rate of its Malicious flows at the score threshold
that gives a false-positive rate of 1 percent on the validation part; the false-positive rate that threshold gives on
the held-out day's Normal flows; each with a 95 percent bootstrap interval over the held-out day's flows (2,000
resamples, `numpy.random.default_rng(0)`; a resample without a class adds nothing to the figures that need it),
for the student and for each of the teacher's seeds. For the teacher, the mean, minimum and maximum over seeds.

## Stated before any figure

- No criterion decides anything: the telecom page and its models do not change; whether a card shows the result is
  the owner's choice after it.
- The days are small, so the intervals will be wide, and are reported as they come.
- A day whose held-out flows hold no Malicious or no Normal flow is reported as not scorable.

## Not done here

No pcap is parsed; no other timeout, entity or the balanced CSVs are used; no hyperparameter is tuned on a held-out
day.

## Compute

CPU, minutes. Script `scripts/check_pfcp_held_out_day.py`; record `docs/measurements/telecom-pfcp-held-out-day-<the
run's date>.json`, aggregate figures only, with the deposit's DOI and the archives' SHA-256.
