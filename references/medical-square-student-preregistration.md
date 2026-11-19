# Pre-registration: a square-activation medical student for a fully encrypted pass

Written and committed on 2026-11-19, before any run of the experiment it describes; its SHA-256 is recorded in
`docs/verification-log.md` with the results. Nothing below is changed after a run. Anything the runs show that this
file did not anticipate is reported as found afterwards, not folded in here.

## The question

The medical page's CKKS student (`medical_fl.student`, BreastMNIST at 14 x 14, test AUC 0.788) runs its layers under
CKKS but decrypts between them: ReLU and pooling run in the clear in the tab. Can a student run end to end under CKKS
in the tab, with no decryption between layers, without losing the accuracy that makes it worth running? The owner
chose on 2026-11-19 to answer this offline first, then in the browser, under two parameter sets.

## The design

- **Data.** MedMNIST v2 BreastMNIST, the official splits (546 train, 78 validation, 156 test), each image as the
  published student reads it: pixel / 255, then (x - 0.5) / 0.5, then area-downsampled to 14 x 14.
- **Teacher.** The published `medical_fl.resnet18`, run from its ONNX graph (the pinned revision `3ee6e722`'s graph
  differs from the local file only by the node metadata stripped on 2026-11-18, with bit-identical outputs). Its
  argmax on the training and validation rows, taken in order, is the label both arms learn.
- **Control arm.** The published student's architecture: convolution 1 -> 4 channels, 3 x 3, padding 1; ReLU;
  average pooling 2 x 2; dense 196 -> 16; ReLU; dense 16 -> 2.
- **Candidate arm, "square".** Convolution 1 -> 4 channels, 3 x 3, padding 1; batch normalisation (folded into the
  convolution's weights and bias for inference, which is exact); $`x^2`$; average pooling 2 x 2; dense 196 -> 2.
  Under CKKS that is multiplicative depth 3: the convolution, the square, and the dense layer with the pooling folded
  into its weights.
- **Recipe, both arms.** Class-weighted cross-entropy on the teacher's labels, Adam at learning rate $`2 \times
  10^{-3}`$, batch 32, at most 120 epochs, early stopping after 25 without improvement, the checkpoint with the best
  validation AUC; seeds 1 to 5. The AUC is `qsh.models.metrics.auroc`, the function that gave the published 0.788.
- **CKKS parameter sets, both evaluated.** A: $`N = 8192`$, coefficient modulus 50 + 30 + 30 + 30 + 50 bits (190),
  scale $`2^{30}`$. B: $`N = 16384`$, 60 + 40 + 40 + 40 + 60 bits (240), scale $`2^{40}`$. Both are within the
  HomomorphicEncryption.org standard's 128-bit post-quantum bounds for a ternary secret (202 bits at $`N = 8192`$,
  411 at $`N = 16384`$). The encrypted pass: the client encrypts the image's nine shifted copies (the convolution's
  taps), replicated across the four channels; the evaluator multiplies by the kernel taps and adds the bias, squares,
  and takes one inner product per class with the pooled dense weights; the client decrypts two logits.

## The criteria, all fixed before the first run

1. **Accuracy.** The candidate's mean test AUC over seeds 1 to 5 is at least the control's mean minus 0.05. The
   margin is about one standard error of an AUC near 0.8 with 42 malignant and 114 benign test images (Hanley and
   McNeil, 1982: about 0.044).
2. **No collapse.** Every candidate seed's test AUC is at least 0.60, and it predicts both classes on the test split.
3. **Fidelity under CKKS.** For the exported seed (the candidate seed with the best validation AUC), on all 156 test
   images and under each parameter set: the largest absolute difference between a decrypted logit and the plaintext
   logit is at most $`10^{-3}`$, and the argmax decision equals the plaintext one on all 156.
4. **Reported whatever the outcome.** Per-seed test and validation AUC for both arms; per parameter set the largest
   logit error, the decisions that differ, the median and largest time per image (Python, TenSEAL), and the sizes
   of the encrypted input and output.

If criterion 1 or 2 fails, the candidate is not adopted, the page does not change, and the result is recorded as a
negative finding. If criterion 3 fails under one parameter set only, that set is not offered on the page.
