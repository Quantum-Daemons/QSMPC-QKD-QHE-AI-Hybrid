# CKKS-linear intrusion-detection student - telecom

> **Nothing in this repository reads this file.** It is written by
> `scripts/train/telecom_ids.py::distil_linear_student`, and the same function writes the identical
> byte string to `web/public/data/telecom/student_weights.json`, which is the copy the browser
> actually fetches. Searched across `src/`, `web/src/`, `api/`, `scripts/` and the generated config:
> the only path that opens a student weight file at run time is
> `web/src/lib/clientCompute.ts::runEncryptedTelecomStudent`, and it fetches the `web/public` copy.
> This directory is the archival copy of a served artifact, not the served artifact. That is worth
> knowing before anyone edits it and expects the demo to change.

NOT PUBLISHED TO HUGGING FACE. `telecom_ids.student` carries no `hf_repo` and no pinned revision:
64 floats do not need a pinned download, and bundling them means the encrypted tier works on a fresh
clone with no network. The registry entry exists (`ModelSpec` key `telecom_ids.student`) and a
`models/metrics.json` record exists, but this directory is **not** in `registry.ARTIFACT_DIRS`, so
nothing resolves the key to this path either.

## What it is

The T1 student for the `telecom_ids` use case, distilled from the plaintext RandomForest teacher
(`telecom_ids.qml`, in `models/telecom_qml_ids`) so that this vertical has an encrypted tier at all.
`kind` is `logistic-linear-multiclass`: what runs under CKKS is 8 dot products over 7 standardised
features, `W` being 8 rows of 7 and `b` being 8.

**Distilled from the teacher's PROBABILITIES, not its argmax.** On 8 classes the argmax throws away
exactly the minority-class signal that a macro F1 is computed from, so the soft targets are what the
student is fitted to.

**The argmax is NOT encrypted, and the card says so because the artifact does.** CKKS has no native
comparison. The client decrypts all 8 class scores and takes the maximum in the clear. That is the
mainstream construction and it is honest in a single-tab demo where the client is the user, but it
must never be described as deciding the class under encryption.

Preprocessing ships with the weights, in the same file: subtract `input.mean`, divide by
`input.scale`. Both were fitted on the training split and are published rather than left for a
caller to guess, which is the same discipline the other students follow.

## What this file records

| quantity | value |
|---|---|
| `n_features` | 7 |
| `n_parameters` | 64 |
| entries in `input.mean`, and in `input.scale` | 7 |

`n_parameters` = 64 is `W` (8 classes x 7 features) plus `b` (8 intercepts). The eight class names
are carried in `classes`, in the row order of `W`, and the seven feature names in `features`, in the
column order of `W` and of the two standardisation vectors - so a caller can align both axes without
guessing. `teacher_key` and `student_key` name the pair this artifact belongs to.

**There are no metrics in this file.** It records the model, not its evaluation. Any figure below
that is not in the table above came from `models/metrics.json`, and is labelled as such.

## Evaluation figures, which live in models/metrics.json and not here

Recorded under the key `telecom_ids.student` by the same training run, on a held-out split of 600
rows over 8 balanced classes:

| figure | value |
|---|---|
| macro F1 (the headline) | 0.988329 |
| aggregate accuracy (companion) | 0.988333 |
| agreement with the teacher | 0.995 |
| classes predicted, of 8 | 8 |
| rows evaluated | 600 |

Per class, from the same record: `combined`, `detector_blinding`, `normal` and
`photon_number_splitting` at 1.0; `intercept_resend` 0.986842; `rng_manipulation` 0.986486;
`trojan_horse` 0.96732; `wavelength_trojan` 0.965986.

**Read the predicted-class count before reading the macro F1.** A linear student that stops emitting
a class scores 0 on it by construction, and the macro average then absorbs a zero that looks like
poor discrimination rather than an absent output. This repository's own variational QNN carries
exactly that defect on this use case and is documented as doing so, so the check is not
hypothetical. All 8 classes are predicted here.

**Agreement is not evidence that the minority classes survived.** Agreement on the finance student
rose from 0.884 to 0.947 across a change that left minority-class F1 unmoved. Per-class F1 is the
headline; the aggregate is a companion.

## Where the classical baseline sits

The teacher is a RandomForest, not a quantum model, and it is the number to beat on this corpus:
macro F1 0.986629 on its own split, recorded in `models/metrics.json` under `telecom_ids.qml`. On the
same use case the variational quantum models lose to it - the quantum-kernel SVC at 0.831611 macro F1
and the variational QNN at 0.466667 - and logistic regression is ahead of the QCNN circuits in
`models/qcnn_telecom` by a wide margin. Nothing here is a claim of quantum advantage; this student
exists to make the encrypted tier real, not to win a benchmark. Those figures are the teacher's
record, not this artifact's, which is why they are under this heading rather than in the table above.

## Reproducibility

    python scripts/train/telecom_ids.py

This rewrites `weights.json` here, `web/public/data/telecom/student_weights.json`, and the
`telecom_ids.student` record in `models/metrics.json`. The two weight files are written from ONE
serialised string, so they cannot drift from each other.

## Data and licence

**SYNTHETIC.** QKD-attack telemetry generated by `qsh.qml_ids.attacks`: 8 classes = 7 attack types
plus normal traffic, with the class structure following Al-Kuwari et al. (arXiv:2509.14282; now
peer-reviewed as IET Quantum Communication 7(1), e70028, 2026). No commercial-OK corpus of real QKD
attack captures exists, which is why this is generated rather than collected. Generated by this
repository - no third-party terms attach.

## Scope

A research proof of concept, not a production intrusion detector. The figures above are what this
repository measured on the split described, with the code in `scripts/train/`. The encrypted
evaluation is real CKKS on 64 parameters; the class decision is taken in plaintext on the client.
