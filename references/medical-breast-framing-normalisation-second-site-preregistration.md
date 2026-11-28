# Pre-registration: the BreastMNIST models on lesion-framed images, after label-free normalisation, and on a second site

Written and committed on 2026-11-28, on the owner's choices of that day (a framing arm on BUS-BRA's boxes, a second
site, BrEaST-Lesions-USG, and a label-free normalisation arm), after the first external test
(`references/medical-busbra-external-test-preregistration.md`, record
`docs/measurements/medical-busbra-external-2026-11-28.json`), and frozen at its SHA-256 in the script before any of
the models below scores any image of these arms.

## What is known, said first

On BUS-BRA, whole images resized to 28x28, the four published BreastMNIST models scored a malignant-class AUC of 0.59
to 0.66, against 0.79 to 0.92 for the same graphs on BreastMNIST's test split. Published work finds the same fall for
whole-image models trained on BUSI and tested on BUS-BRA and on BrEaST-Lesions-USG, and less of it when models are
trained and tested on lesion crops (Wang et al. 2026, Diagnostics 16:1537: whole image about 0.65 to 0.71, crops with a
12 percent margin 0.74 to 0.81). Read before this was written, and nothing else:

- BrEaST-Lesions-USG's clinical sheet and image headers: 256 images, one per patient (`CaseID`); `Classification`
  benign 154, malignant 98, normal 4; 252 tumour masks; widths 363 to 1,024 pixels (median 545) and heights 289 to 618
  (median 537), every image stored as RGBA.
- BUS-BRA's `BBOX` column against its masks on its first 400 images: it is [x, y, width, height], within 2 pixels in
  total of each mask's extent.
- The two ResNets' unfused training checkpoints (`model.safetensors` in each model's directory of the main checkout,
  never published; SHA-256 `6c1499d4...` for ResNet-18, `b17d6262...` for ResNet-50): on BreastMNIST's published
  156-image evaluation sample they reproduce their published fp32 graphs to $`1.5 \times 10^{-5}`$ and
  $`3.8 \times 10^{-5}`$ in the largest logit difference, with every decision the same. The published graphs have
  their batch normalisation folded into the convolutions, so only these checkpoints can have it re-estimated.

## The questions

On a second site, do the models fall as on BUS-BRA? Does framing each image on its lesion change their scores? Does
adapting the input statistics, or the ResNets' batch-normalisation statistics, to the new site without its labels
change them?

## The data

- BUS-BRA, every image and its `bus_data.csv` row, read as in the first pre-registration.
- BrEaST-Lesions-USG, version 1 (The Cancer Imaging Archive, doi:10.7937/9WKK-Q141, CC BY 4.0 under TCIA's Data Usage
  Policy; descriptor Pawlowska et al., Scientific Data 11:148, 2024): the images-and-masks archive (SHA-256
  `c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355`) and the clinical sheet (SHA-256
  `89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf`). Each image is read with Pillow's
  `convert("L")`, which drops the alpha channel; the script reports how many images have any alpha value below 255.
  The label is `Classification`: malignant is BreastMNIST's class 0, benign and normal its class 1 ("normal or
  benign"). No image is excluded.

Every image, whole or cropped, is then resized to 28x28, rounded and clipped, and preprocessed as the first
pre-registration states (cubic spline resize with anti-aliasing, pixel / 255, then (x - 0.5) / 0.5, the students'
area resize to 14x14).

## The models

The first pre-registration's five graphs, each at the SHA-256 the shipped config pins, or nothing runs: ResNet-18's
int8 and fp32 graphs, ResNet-50's fp32 graph and the two students' fp32 graphs. For arm N2 only, the two ResNets'
unfused checkpoints above, at those SHA-256s; the script first re-checks them against their graphs on the published
sample and runs N2 only if the largest logit difference is at most $`10^{-4}`$ with no decision changed.

## The arms

- **S, a second site.** BrEaST-Lesions-USG's whole images through the first pre-registration's protocol in full: the
  five graphs, each beside the same graph's BreastMNIST test figures; the students' float64 networks against their
  graphs; and both students' CKKS paths through the page's own modules in node, under the page's parameter sets, held
  to the page's rule. BrEaST names no device, so there is no per-device figure.
- **F, framing.** On BUS-BRA and on BrEaST, each image cropped to its lesion's box widened on each side by $`m`$ times
  the box's width (left and right) and height (top and bottom), clipped to the image, then treated as a whole image.
  BUS-BRA's box is its `BBOX`; BrEaST's is the extent of the image's tumour mask (pixels above 127 after
  `convert("L")`), and its four normal images, which have none, are scored whole. Three margins, fixed here:
  $`m = 0.12`$ (Wang et al.'s), $`m = 0`$ (the box alone) and $`m = 0.5`$. The five graphs; no CKKS stage.
- **N, label-free normalisation.** On BUS-BRA and on BrEaST, whole images. Each dataset's patient IDs (BUS-BRA's `Case`,
  BrEaST's `CaseID`), in numeric order, are shuffled by `numpy.random.default_rng(0).permutation`; the first half (rounded down) is the adaptation half and the
  rest the test half. No label is read by any adaptation, and every arm-N figure is on the test half.
  - N0: no adaptation, the five graphs on the test half, the baseline for the same images.
  - N1, input statistics: with $`\mu_A, \sigma_A`$ the mean and standard deviation over every pixel of the
    adaptation half's preprocessed 28x28 inputs, and $`\mu_B, \sigma_B`$ those of BreastMNIST's training split
    (MedMNIST v2's `breastmnist.npz`, `train_images`, through the same preprocessing), each test-half input $`x`$
    becomes $`(x - \mu_A)\,\sigma_B/\sigma_A + \mu_B`$, before the students' resize. The five graphs.
  - N2, batch-normalisation statistics: each unfused ResNet with every `BatchNorm2d`'s running mean and variance
    re-estimated on the adaptation half (train mode with `momentum=None`, the cumulative average over batches of 64 in
    the half's order, no gradient, every weight fixed), then scored in eval mode on the test half; the unadapted
    checkpoint is scored on the test half beside it.

## What is measured, per model and arm

The first pre-registration's figures: the AUC and the AUPRC of the malignant class from the model's own score, each
with a 95 percent interval from 2,000 patient resamples (`numpy.random.default_rng(0)`), and the accuracy and the
confusion matrix at the model's own decision rule. Beside each arm's figure, the same model's figure on the same images
unframed (arm F) or unadapted (arm N), and for arm S the same graph's BreastMNIST test figure.

## Stated before any figure

- No criterion decides adoption: nothing on the page or the Hub changes because of a result, and no adapted network is
  published. Whether a card shows a result is the owner's choice after it.
- Expected: on BrEaST, lower figures than on BreastMNIST; framing, not predicted in either direction, since these
  models were trained on BUSI's whole frames, not on crops; normalisation, small changes, the direction not predicted.
- The CKKS agreement of arm S is held to the page's rule: the largest logit difference at most $`10^{-3}`$ and no
  decision changed on an image whose plaintext margin exceeds twice it. A failure is reported as found.

## Not done here

No fine-tuning, no retraining, no threshold moved, no label read by an adaptation, no image or per-patient figure
published. Retraining on lesion crops is pre-registered separately.

## Compute

CPU throughout: ONNX Runtime for the graphs, PyTorch for arm N2's checkpoints, NumPy for the students' float64
networks, node for the CKKS stage. Script `scripts/check_breast_framing_normalisation_second_site.py`; the record is
`docs/measurements/medical-breast-framing-normalisation-second-site-<the run's date>.json`, aggregate figures only,
with each archive's and each graph's and checkpoint's SHA-256.
