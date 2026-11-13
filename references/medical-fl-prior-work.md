# Federated learning in medicine: the published cases, what reproduces here, and how

Read at the sources on 2026-11-13, for the owner's question of whether the medical page's federated learning matches
the published cases (COVID-19 chest X-ray CNNs across hospitals, FeTS, NVIDIA FLARE, federated fine-tuning of
foundation models), and how each could be reproduced faithfully in a browser. "Agent-read" marks a figure a research
agent read and this document did not read again; every other figure was read here at the source named. Nothing below is
a licence opinion: where a dataset's own pages disagree, the disagreement is the finding.

Heart sounds with ECG (the NEJM AI auscultation study and the public data that copy its design) and ConvNeXt trained
from scratch are in [heart-sound-and-ecg-prior-work.md](heart-sound-and-ecg-prior-work.md).

## What the project runs

| where | what | measured |
|---|---|---|
| browser, medical page (since 2026-11-13) | FLamby's Fed-Heart-Disease on four real hospitals: each alone, FedAvg under the pairwise masks, all four pooled (`web/src/lib/clientFedHeart.ts`, twin `qsh.ai.fed_heart`) | five seeds: alone 0.735, 0.636, 0.601, 0.644; federated $`0.712 \pm 0.017`$; pooled 0.715 (FLamby's metric, sample standard deviation); on its own test rows a hospital's own model is above the federated one at three or four of the four hospitals in every seed |
| full deployment, `{"dataset": "medmnist", "model": "cnn"}` | FedAvg of a 3,226-parameter CNN over 409 BreastMNIST images of 14x14 in four label-skewed hospitals | until 2026-11-13 one class for every image (0.7282, the majority); class-weighted Adam now, seed 0: balanced accuracy 0.619, AUC 0.685 |
| published models | ResNet-18, ResNet-50 and the CKKS student on BreastMNIST | trained centrally |

Built since: CKKS-encrypted aggregation and threshold CKKS and BGV beside the masks, run on request (2026-11-13).
Built since: record-level differential privacy beside the federation (2026-11-13; `references/differential-privacy.md`).
Planned, in the owner's order of 2026-11-13: chest
X-ray federation on NIH ChestX-ray14 with a view-position bias analysis and CAM; EfficientNet-B0, DeiT-Tiny and VGG-11
trained federated offline; a 2D U-Net on breast ultrasound; FLARE, FeTS-style and federated-LoRA experiments offline.

## The published cases

| case | what it showed | here |
|---|---|---|
| EXAM (Dayan et al., Nature Medicine, 2021; PMC9157510) | 20 institutes, 16,148 cases; mean AUC on each site's own test 0.795 local and 0.920 federated, across all sites' tests 0.667 and 0.920, at 24 hours; a ResNet-34 with spatial attention pre-trained on CheXpert and frozen during FL, fused with EMR features; 224 x 224; FedAvg for 200 rounds of one local epoch (the training details agent-read from the author manuscript) | protocol only: the data is closed and the released model is for research use |
| Peng et al. (JAMIA 2023), 42 hospitals | DenseNet121; internal AUROC 0.94 local and 0.951 FedAvg, not significant (P = .49); external 0.557 to 0.601 and 0.667 to 0.713 (agent-read) | protocol only |
| Swarm Learning (Warnat-Herresthal et al., Nature, 2021; PMC8189907) | swarm above every single node, including NIH chest X-rays at 128 px (agent-read) | not reproduced: the HPE software is for non-commercial use (agent-read) |
| Diniz et al. (npj Digital Medicine, 14 Feb 2026; PMC12916833), systematic review | 160 articles, 710 decentralised models (557 federated), 8,149 comparisons: decentralised learning above local in 83 percent of accuracy and 82 percent of AUROC comparisons; centralised above decentralised in 78 percent of accuracy and of Dice comparisons and 51 percent of AUROC ones | the framing every claim on the page follows |
| FLamby Fed-Heart-Disease (du Terrail et al., NeurIPS 2022 Datasets and Benchmarks) | four UCI hospitals; logistic regression; averaged here from its five published result files over the four client test sets: alone 0.722, 0.629, 0.567, 0.637; FedAvg $`0.685 \pm 0.028`$; FedYogi 0.731; pooled 0.735 | reproduced in the browser, preparation step for step |
| FeTS (Pati et al., Nature Communications, 2022; PMC9722782) | 71 sites, 6,314 patients, a 3D U-Net with residual connections (agent-read) | not in a browser: 4-channel MRI of $`128^3`$ voxels, and the BraTS terms differ by copy |
| NVIDIA FLARE 2.9.1 (PyPI, 12 Nov 2026, Apache-2.0) | homomorphic aggregation with TenSEAL: CKKS, N = 8192, coefficient moduli [60, 40, 40], scale $`2^{40}`$, the secret key in each client's context and none in the server's (`nvflare/lighter/impl/he.py` at 2.9.1); the HE recipe does not run in the simulator (its documentation) | offline only |
| federated LoRA on BiomedCLIP (arXiv:2609.02101, 2 Sep 2026) | four chest X-ray cohorts, mean AUC 0.802 federated and 0.812 pooled; naive averaging of the LoRA factors costs 0.097 | offline only; BiomedCLIP's card puts "any deployed use case" out of scope |

## Data that may ship, and data that may not

| corpus | read at the source | for |
|---|---|---|
| UCI Heart Disease | CC BY 4.0 (UCI page); the four files' MD5s are FLamby's; Cleveland 199/104, Hungary 172/89, Switzerland 30/16, VA Long Beach 85/45 train/test after FLamby's preparation | the medical page's federation (shipped) |
| NIH ChestX-ray14 | NIH's FAQ, Q04: "The usage of the data set is unrestricted", with a link to the download site, an acknowledgement of the NIH Clinical Center and a citation of the CVPR 2017 paper. NIH's change log of 2026-10-09 lists pixel-identical images, four pairs of them across the official training and test lists | the chest X-ray federation, built from the original PNGs (42 GB), the duplicates removed |
| ChestMNIST (MedMNIST) | `info.py` says it uses the official split, with a test split of 22,433 images; NIH's `test_list.txt` holds about 25,596 (424.9 KB at 17 bytes a line); no patient IDs or view positions | not used for the bias analysis |
| Kermany et al. (PneumoniaMNIST's source), Mendeley v3 | CC BY 4.0 in the licence field, and the description reads "made available for use in research only" | not shipped |
| BUS-UCLM v3 (Mendeley, 27 Jan 2025) | CC BY 4.0 (v1 and v2 were non-commercial); 683 images, 38 patients, RGB masks | the 2D U-Net, hospitals by patient |
| Breast-Lesions-USG (TCIA) | CC BY 4.0; 256 patients, 66.6 MB | the U-Net's external test hospital |
| BUSI (BreastMNIST's source) | the article is CC BY and the authors state no data licence; 155 duplicates and 70 axilla images (Pawlowska et al., Data in Brief 48:109247, 2023) | already shipped as BreastMNIST, under MedMNIST's assertion |
| COVIDx, COVID-QU-Ex, BIMCV-COVID19+, CheXpert, MIMIC-CXR | non-commercial, compiled from sources with their own terms, or credentialed (agent-read) | not shipped |

## How each reproduces in a browser

- **The chest X-ray federation.** EXAM froze its pre-trained backbone and federated the rest, so the faithful small
  version is a frozen backbone and a federated head; ImageNet weights carry ImageNet's terms, which this project's
  models avoid. Hospitals split NIH's official training list by patient, the shared test set is NIH's test list less
  the duplicates, and the page reports EXAM's pair of figures beside pooled training. Pleural effusion is the finding
  the agent recommended (13,317 positives, the best label precision of the common findings in Oakden-Rayner's visual
  check; agent-read).
- **View position.** About 40 percent of NIH's images are AP (agent-read); models learn AP, portable markers and site
  as shortcuts (Zech et al., 2018; DeGrave et al., 2021; agent-read). The page will report AUROC by view, sensitivity
  and specificity by view at one threshold, a probe that predicts the view from the model's features, and a PA-only
  model beside the mixed one.
- **CAM.** For a backbone ending in global average pooling and a linear head, Grad-CAM reduces to CAM (Selvaraju et
  al., ICCV 2017, section 3.1; agent-read), so the page can draw exact maps from a forward graph that also returns the
  last feature maps; ONNX Runtime Web 1.31.1 runs no gradient graph. Saporta et al. (Nature Machine Intelligence,
  2022) found Grad-CAM the best of seven saliency methods on chest X-rays and all seven significantly worse than
  radiologists at localisation (abstract read here; the percentages agent-read), so the page will say a map shows where
  this model's score came from, not where a lesion is, and measure the overlap with NIH's 984 boxes.
- **Training in a tab.** No maintained library: ONNX Runtime deprecated its training packages at 1.20.0, and jax-js
  0.1.27 (18 Oct 2026, MIT) calls itself early. Models of a few thousand parameters train with hand-written
  backpropagation; EfficientNet-B0 (5.3M parameters), VGG-11 and DeiT-Tiny train federated offline, without ImageNet
  weights, and the page verifies them as it verifies the other published models.
- **Encryption beside FedAvg (built 2026-11-13; `web/src/lib/clientFedAgg.ts`, `wasm/qsh-fedthresh`).** The pairwise masks are exact and let the aggregator read the sum. CKKS aggregation as
  FLARE does it is the second design: the clients hold the key and the aggregator only adds ciphertexts. Threshold
  CKKS, in which no party holds the whole key, is the third (Lattigo v6, Apache-2.0, which states WASM builds for
  browser clients and implements multiparty CKKS). Decryption shares need smudging noise, and at a security parameter
  of 128 bits that noise swamps a scale of $`2^{40}`$, so the page measures the settings side by side. Measured: CKKS
  cannot reach 128 bits within Lattigo's scale cap of $`2^{128}`$ (error about $`2.8 \times 10^{6}`$ there), so the
  page shows threshold CKKS at 16 bits (Lattigo's examples' $`\sigma = 2^{30}`$, a demonstration) and 40 bits, and
  threshold BGV at 128 bits, which is exact and equals the masks to the bit. A ring of 16,384 does not help: the
  requirement is on the scale against the deviation, not on the ring.

## Offline experiments planned for the development GPU

1. FLARE 2.9.1: MedNIST DenseNet-121 with a real per-client split, then the spleen 3D UNet bundle; TenSEAL aggregation
   against plain FedAvg in its PoC environment.
2. FeTS at small scale: Fed-IXITiny's three London sites (IXI, CC BY-SA 3.0; Guy's 249/62, HH 145/36, IOP 59/15 per
   FLamby) with a 3D UNet, alone against federated against pooled.
3. Federated LoRA on chest X-rays with weights whose terms allow publication, four patient-disjoint clients, rank 8,
   naive against SVD aggregation as in arXiv:2609.02101; BiomedCLIP, RAD-DINO and the HAI-DEF models only as
   research-flagged runs.
