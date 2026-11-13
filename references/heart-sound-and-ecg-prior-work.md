# Heart sounds and ECG, and modern CNNs: what was read, and what it supports here

Read on 2026-11-13 for the owner's question of whether AI auscultation (heart sound with ECG) and modern CNNs such as
ConvNeXt suit the medical page, as an extension and as prior work for federated learning, MPC, FHE and differential
privacy. A research agent read the sources; "read here" marks what was read again at the source for this file, and the
rest is agent-read.

## The NEJM AI study

- **Sato, Izumi, Saito, ..., Kagiyama, "Artificial Intelligence-Enhanced Auscultation to Identify Left Ventricular
  Dysfunction - Development and External Validation in Prospective Multicenter Studies"**, NEJM AI 3(11):AIoa2600258,
  online 15 October 2026 (DOI 10.1056/AIoa2600258, PMID 42932469; read here, the PubMed abstract). Adults referred for
  echocardiography, recorded at four standard auscultation sites with a simultaneous single-lead ECG; left ventricular
  dysfunction defined as LVEF of at most 40 percent on concurrent echocardiography. The model is an ensemble of a
  transformer-based network and a one-dimensional CNN. Training cohort 2,359 patients (8.9 percent reduced LVEF),
  AUROC 0.926 (0.911 to 0.939); external validation 964 patients from an independent registry (7.6 percent), AUROC
  0.904 (0.876 to 0.930), sensitivity 86.3 percent, specificity 74.2 percent, accuracy 75.1 percent; a two-site
  version 0.903. In a separate 100-patient reader substudy the AI's accuracy was 82.0 percent against 62.0 percent for
  the consensus of seven cardiologists (P = 0.004). Funded by AMI Inc. and JSPS KAKENHI 25K19371.
- **Two corrections to how it was first summarised.** The AUROC of 0.904 is the external validation, not the comparison
  with cardiologists, which is the accuracy in the 100-patient substudy. And the model is an ensemble of two networks
  whose scores are combined, not one CNN-Transformer hybrid, and not ConvNeXt: the released code (agent-read,
  `ami-co-ltd/ai-auscultation-lvd`, MIT, runs on dummy data only) has a wav2vec 2.0-style transformer over sound and
  ECG and an EfficientNet-style 1D CNN over the sound. The data are not shared (consent) and the weights are not
  released, so the result can be cited as a protocol and not reproduced.
- **Context, agent-read.** Attia et al., Nature Medicine 2019: a 12-lead ECG CNN for EF of at most 35 percent, AUC
  0.93. Bachtiger et al., Lancet Digital Health 2022: single-lead ECG from a stethoscope, AUROC 0.85 at one position
  and 0.91 at two. TRICORDER (Lancet 2026), a cluster trial in 205 practices, found no rise in heart-failure detection
  (IRR 0.94, 0.86 to 1.02): a page must not claim a clinical benefit.

## Public data

| dataset | content | licence | read here |
|---|---|---|---|
| Ma, Zhang, ..., Zhang, "Multi-Channel ECG and PCG Dataset for HFrEF Detection", Zenodo 16934966 | 2,480 synchronised ECG and PCG pairs from 620 subjects, four sites each, Eko DUO, Tongji and Liyuan Hospitals; HFrEF when LVEF is at most 40 percent on echocardiography; 362.8 MB | CC BY 4.0 | yes |
| PhysioNet/CinC Challenge 2016, v1.0.0 | heart-sound recordings at 2 kHz from several source databases, one folder with single-lead ECG; normal or abnormal; 1.1 GB | ODC-By 1.0 (the CinC papers are CC BY 3.0) | yes |
| CirCor DigiScope, v1.0.3 | 942 patients, 3,163 recordings at 4 kHz, sound only; murmur present, absent or unknown; patient IDs; 558.9 MB | ODC-By 1.0 | yes |
| EchoNext; MIMIC-IV-ECHO | ECGs with echo LVEF | signed agreement; credentialed | not usable here (agent-read) |

The Zenodo set copies the NEJM AI task design on public data; by the agent's count its second hospital has 5 positive
subjects, too few for a cross-hospital evaluation. ODC-By 1.0 licenses the database and says it does not cover the
individual contents, a gap to record in `DATA_LICENSES.md` before anything ships from the PhysioNet sets.

## Secure computation on ECG and heart sounds (agent-read)

Federated ECG across US and Japanese centres (Goto et al., Circulation 2022): AUC 0.90 to 0.96 at every site, where
single-site models transferred poorly. Federated ECG with DP-SGD over 1.57 million ECGs (Agrawal et al., arXiv
2405.00725): AUROC near 0.5 below epsilon 1. CKKS inference of a 1D CNN on ECG (UniHENN, arXiv 2402.03060): seven
multiplicative levels, 96.87 against 98.90 percent, 5.1 s an inference, beyond the page's two levels. No MPC work on
ECG from 2022 to 2026 was found.

## ConvNeXt

- **Liu et al., "A ConvNet for the 2020s"**, CVPR 2022 (arXiv:2201.03545); **Woo et al., ConvNeXt V2**, CVPR 2023
  (arXiv:2301.00808). Code MIT; ConvNeXt V2's ImageNet weights CC BY-NC 4.0, so only from-scratch training fits the
  project (agent-read). Part of ConvNeXt's ImageNet gain is its training recipe, which alone moved ResNet-50 from 76.1
  to 78.8 percent, so backbones are compared under one recipe.
- **From scratch on NIH ChestX-ray14** (agent-read): no published ConvNeXt result was found. From-scratch ResNet-50
  matched ImageNet fine-tuning in Baltruschat et al., Scientific Reports 2019 (Effusion 87.6 against 87.1), and
  ResNet-50 from scratch reached 80.31 against 81.70 with ImageNet in Hosseinzadeh Taher et al. 2021.

## What follows for the project

- A ConvNeXt backbone for the chest X-ray federation: optional, as one offline run of a small variant under the same
  recipe as the baselines; the browser trains the head only, so the backbone matters less there.
- Heart sounds: a federation across the PhysioNet 2016 sources with each source held out in turn, and CirCor's murmur
  label with per-patient DP, are the variants public data support; the HFrEF set would follow the NEJM AI design and
  could never be compared with 0.904. All after the chest X-ray work, in the owner's order, and each download needs
  the owner's approval first.
