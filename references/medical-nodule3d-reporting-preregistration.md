# Pre-registration: the 3D nodule student reported at each of its steps (2026-11-27)

Written and committed before any figure below was computed (Round 86, batch 8), on the owner's choice of 2026-11-27 to
take both suggestions of the model review of 2026-11-26 (`references/model-choice-vs-2026-practice.md`), each
pre-registered. Nothing here is changed after a figure is seen; a change is an addendum with its date and reason.

## Why

Lee et al., "Development of Privacy-preserving Deep Learning Model with Homomorphic Encryption: A Technical Feasibility
Study in Kidney CT Imaging", Radiology: Artificial Intelligence 7(6):e240798 (November 2025), doi:10.1148/ryai.240798,
report AUC and AUPRC at each of three steps: a reference network, its approximation for encryption (ReLU replaced by a
polynomial, max pooling by average pooling), and the network under CKKS. The record of the medical page's 3D square
student (`docs/measurements/medical-nodule3d-square-student.json`, under
`references/medical-nodule3d-square-student-preregistration.md`) gives the teacher's AUC out of fold only, the plaintext
students' test AUC, and the encrypted student's agreement with its plaintext self (the largest logit error and the
decisions changed). It gives no AUPRC, no teacher figure on the test part, and no AUC of the decrypted logits. This adds
them. It changes neither the shipped model, its threshold, nor the pre-registered record.

## What is computed, on the record's test part, on the CPU

The test part is the record's: 347 nodules of 148 patients, 109 malignant (`split.counts.test`), so a model that ranks at
random has an AUPRC of about 0.314. Every figure below is computed on exactly those crops.

1. **The teacher.** The teacher is the 3D ResNet-18 candidate (`teacher: resnet3d18`). Its only networks are the five
   fold networks of the record's `teachers` stage, each trained on three fifths of the training patients and stopped on a
   fourth. Each scores every test crop, and the teacher's test probability of malignancy is the mean of the five. Nothing
   is retrained, and the record says that this teacher is an average of fold networks, not one network fitted on every
   training patient.
2. **The plaintext students.** For each of the five seeds of the square student and of the ReLU control, the test AUPRC
   beside the test AUC, both from the seed's checkpoint (`square_seed{s}`, `control_seed{s}`). Each recomputed AUC must
   equal the record's `test_auc` to $`10^{-6}`$, or the run stops, since the checkpoint would not be the one recorded.
   Also the shipped seed's folded float64 network, as the record's `export_bootstrap` scores it.
3. **The encrypted student.** The shipped seed (5), through the page's own module (`web/src/lib/clientFHESquare3d.ts`,
   node-seal), under each pre-registered parameter set (A2, B), on every test crop, exactly as the record's `ckks` stage
   runs it, now keeping each crop's decrypted logits: the AUC and AUPRC of the decrypted logits, and of the plaintext
   logits of the same run beside them. CKKS noise differs between runs, so these are one run's; its largest logit error
   is recorded beside them, as the `ckks` stage records its own.

Each score is malignant against benign. The students are scored by their logit difference (class 1 minus class 0), the
teacher by its mean probability; AUC and AUPRC depend only on the order, so the Platt map changes neither. AUPRC is
average precision, as scikit-learn's `average_precision_score` computes it. Each figure carries a 95 percent percentile
interval over 2,000 resamples of the test patients, numpy's default generator seeded 0, as the record's
`export_bootstrap`.

## What is reported

All of it, whatever it shows, in `docs/measurements/medical-nodule3d-reporting-2026-11-27.json` and on the 3D card: the
teacher, the plaintext student (with the ReLU control beside it) and the encrypted student, each with AUC and AUPRC, as
one table. No figure decides anything: nothing is adopted or withdrawn by it. Per-crop scores stay in the work directory
outside the repository; the record holds the figures only.

## How these steps differ from Lee et al.'s

Their reference network is their encrypted network's own architecture with ReLU and max pooling. The nearest here is the
ReLU control, the same design with ReLU in place of the square; the teacher is a different network (a 3D ResNet-18) that
the students were distilled from. Both are reported, each named for what it is.

## Addendum, 2026-11-27, before any figure of this report was read

The first run stopped where step 2 says it must: the square student's seed 3 gave a test AUC of 0.905867 on the CPU
against the record's 0.905905. Its checkpoint's SHA-256 equals the record's `checkpoint_sha256`, so it is the recorded
checkpoint; the record's test AUCs were computed from logits on the GPU, and the difference, $`3.8 \times 10^{-5}`$, is
one of the test part's $`109 \times 238 = 25{,}942`$ malignant-benign pairs ordered the other way ($`1 / 25{,}942 \approx
3.85 \times 10^{-5}`$), as the two devices' rounding of a near tie can. Nothing had been read but the seeds' AUCs the
record already holds. So step 2's check becomes: each checkpoint's SHA-256 equals the record's, and each recomputed AUC
is within two such pairs, $`2 / (109 \times 238)`$, of the record's; a larger difference still stops the run. The record
gives both AUCs for every seed. The folded float64 network's check, the same arithmetic on the same device as the
record's, stays at $`10^{-6}`$. Nothing else changes.
