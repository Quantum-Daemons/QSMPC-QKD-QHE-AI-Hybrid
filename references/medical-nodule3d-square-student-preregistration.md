# Pre-registration: a square-activation 3D nodule student for a fully encrypted pass

Written and committed on 2026-11-24, after the crops script was tried on three patients and before any label was
counted or any model trained. Its SHA-256 is recorded in `docs/verification-log.md` with the results. Nothing below is
changed after a run. Anything the runs show that this file did not anticipate is reported as found afterwards, not
folded in here.

## The question

The medical page's square student (`medical_fl.square_student`, BreastMNIST at 14 x 14) runs end to end under CKKS in
the tab, with no layer decrypted in between. Can a network of the same shape in three dimensions, on a 28 x 28 x 28
crop of a chest-CT nodule, separate the nodules radiologists rated likely malignant from those they rated likely
benign, on patients it never saw? And does CKKS change any of its decisions? The owner chose on 2026-11-24 to answer
this on the whole LIDC-IDRI collection, with the crop encrypted at 28 x 28 x 28.

## The data

- **Source.** LIDC-IDRI from The Cancer Imaging Archive, under CC BY 3.0 (Armato III et al., 2015,
  doi:10.7937/K9/TCIA.2015.LO9QL9SX; Armato et al., Medical Physics 38(2):915-931, 2011, doi:10.1118/1.3528204). All
  1,018 CT series are used, and every file is checked against the MD5 that TCIA's API lists for it.
- **Nodules.** pylidc 0.2.3 holds the readers' marks of nodules of 3 mm or more, the ones they rated for malignancy.
  Each of its clusters of those marks (`Scan.cluster_annotations()`, default tolerance) gives one crop.
- **Crop.** 28 points a side at 1 mm, NoduleMNIST3D's size and spacing (MedMNIST v2, Yang et al., Scientific Data
  10:41, 2023). It is centred on the mean of the readers' centroids and sampled by trilinear interpolation in
  millimetres, with air (-1000 HU) outside the scan. It is clipped to [-1000, 400] HU, a lung window chosen here, and
  scaled to 0 to 255. Each network reads $`(u / 255 - 0.5) / 0.5`$, as the 2D student does.
- **Label.** The median of the readers' malignancy ratings, 1 to 5. A median of exactly 3 is dropped. Above 3 is
  malignant, and below 3 is benign. MedMNIST v2 also ignores level 3 but does not say how it combines the readers.
  The earlier ACS paper by its first author (Yang et al., IEEE JBHI 2021, arXiv:1911.10477) takes the mode, which can
  tie. The median is defined for any set of ratings, so these labels differ from NoduleMNIST3D's. They are
  radiologists' suspicion, not pathology, and every figure says so.

## The split

- By patient, never by nodule. Only patients with at least one labelled nodule take part.
- Each patient's majority label sets its stratum, and a tie counts as malignant. Within each stratum, benign first,
  the patients sorted by ID are permuted by NumPy's default generator seeded 0. The first
  $`\mathrm{round}(0.7 n)`$ go to training, the next $`\mathrm{round}(0.1 n)`$ to validation, and the rest to test.
- A patient with two CT series is one patient.
- The split is written once, with its SHA-256, and never redrawn. Before training, two checks are run and reported:
  no patient is in two parts, and the counts of nodules and labels in each part.

## The arms

- **Teachers.** By the owner's choice, the outside models touch only training patients, since CT-FM and
  TotalSegmentator may have seen LIDC-IDRI scans in their own training. So no teacher touches a validation or test
  crop. Each candidate is scored by five-fold cross-validation over the training patients, and the candidate with the
  best out-of-fold AUC teaches. Its out-of-fold probabilities are the soft labels. There are three candidates:
  - a ResNet-18 built with 3D convolutions (basic blocks of 64, 128, 256 and 512 channels), from random
    initialisation, trained on the radiologists' labels by the recipe below;
  - an L2-regularised logistic regression on the frozen features of CT-FM's feature extractor (Apache-2.0,
    `project-lighter/ct_fm_feature_extractor` at revision `527b0d1c`), from a cube around each nodule prepared as the
    model card prescribes, with its strength chosen by cross-validation inside each training fold;
  - a size-only candidate: a logistic regression on the logarithm of the volume of TotalSegmentator's lung-nodule
    mask nearest the crop's centre (Apache-2.0, release `v2.5.0-weights`, `Dataset913_lung_nodules`, run on each
    training scan). It also shows how much of the label nodule size alone explains.
- **Candidate arm, "square".** Convolution 1 -> 4 channels, 3 x 3 x 3, padding 1; batch normalisation, folded into the
  convolution for inference, which is exact; $`x^2`$; average pooling 2 x 2 x 2; dense $`4 \cdot 14^3 \to 2`$. Under
  CKKS that is multiplicative depth 3, as for the 2D student.
- **Control arm.** The same network with a ReLU in place of $`x^2`$.
- **Recipe, both arms.** The loss is the cross-entropy between the network's two-class softmax and the teacher's pair
  $`(1 - p, p)`$ on each training crop. The settings are those of `qsh.models.train.train_torch` for the 2D student:
  AdamW with weight decay $`10^{-4}`$, learning rate $`2 \times 10^{-3}`$ with a 5% warm-up, gradient clipping at
  1.0, mixed precision on the GPU, batch 32, at most 120 epochs, early stopping after 25 without improvement, and the
  checkpoint with the best validation AUC on the radiologists' labels. Training crops get random flips along each
  axis and a random multiple of 90 degrees in the axial plane. Seeds 1 to 5. The AUC is `qsh.models.metrics.auroc`.
- **Calibration and threshold.** Platt scaling on the validation crops maps the logit difference to a probability.
  The decision threshold is the one that maximises balanced accuracy on the validation crops, the lowest of any tie.
- **CKKS parameter sets, both evaluated.** A2: $`N = 8192`$, coefficient modulus 48 + 34 + 34 + 34 + 52 bits (202),
  scale $`2^{34}`$. B: $`N = 16384`$, 60 + 40 + 40 + 40 + 60 bits (240), scale $`2^{40}`$. Both are within the
  HomomorphicEncryption.org standard's 128-bit post-quantum bounds for a ternary secret (202 bits at $`N = 8192`$,
  411 at $`N = 16384`$).
- **The encrypted pass.** It is the page's own code, `web/src/lib/clientFHESquare3d.ts`, under node-seal. The client
  encrypts the crop's 27 convolution taps, each split into blocks of $`N / 2`$ slots: 6 blocks under A2 and 3 under B.
  The evaluator forms each channel from the tap blocks and that channel's kernel taps, adds the bias, squares, and
  takes one inner product per class with the dense weights averaged over each pooling window. A rotate-and-add over
  all $`N / 2`$ slots then puts each logit in every slot, so the client decrypts two logits and no partial sum.

## The criteria, all fixed before the first run

1. **Accuracy.** The candidate's mean test AUC over seeds 1 to 5 is at least the control's mean minus 0.03. The
   margin is about one standard error of an AUC near 0.85 (Hanley and McNeil, 1982) with the roughly 350 test
   nodules, a quarter of them malignant, that a 20% patient split is expected to hold. That figure was fixed before
   any label was counted.
2. **No collapse.** Every candidate seed's test AUC is at least 0.60, and each seed predicts both classes on the test
   nodules at its validation threshold.
3. **Fidelity under CKKS.** The exported seed is the candidate seed with the best validation AUC. On every test crop
   and under each parameter set, the largest absolute difference between a decrypted logit and the plaintext logit
   is at most $`10^{-3}`$. The decision at the validation threshold must also equal the plaintext one on every crop.
4. **Reported whatever the outcome.**
   - Per seed, both arms: test and validation AUC, and balanced accuracy at the validation threshold.
   - For the exported seed: a 95% bootstrap interval over test patients (2,000 resamples, seed 0).
   - The malignant share and the majority-class rate in each part.
   - Each teacher candidate's out-of-fold AUC.
   - Per parameter set: the largest logit error, the decisions that differ, the median and largest time per crop,
     and the ciphertexts and bytes in and out.
   - NoduleMNIST3D's official-split figures (test AUC: ResNet-18 3D 0.863, ResNet-50 3D 0.875, auto-sklearn 0.914),
     cited for context, not compared: that split shares scans across its parts, and its labels differ.

If criterion 1 or 2 fails, the candidate is not adopted and the page does not change. The result is then recorded as
a negative finding. If criterion 3 fails under one parameter set only, that set is not offered on the page.
