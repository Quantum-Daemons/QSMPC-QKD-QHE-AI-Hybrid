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

## The federation on the Challenge 2016 data (2026-11-14)

- Clifford, Liu, Moody, Springer, Silva, Li and Mark, *Classification of Normal/Abnormal Heart Sound Recordings: the
  PhysioNet/Computing in Cardiology Challenge 2016* (read here, the paper in the released zip): eight databases from
  seven teams in seven countries; normal recordings from healthy subjects and abnormal ones mostly from patients with
  heart valve defects or coronary artery disease; recordings resampled to 2,000 Hz; signal quality hand-corrected, the
  poor ones labelled unsure. Their example algorithm, scored leaving one training database out, ranged from 0.47 on b to
  0.86 on c in the mean of sensitivity and specificity, $`0.59 \pm 0.15`$ across the databases.
- The training-recording count differs by where it is read: 3,240 in the released REFERENCE files, which this project
  used; 3,126 in the PhysioNet page's introduction and 3,125 in its announcement; 3,153 in the CinC paper's Table 1, which
  also gives the patients per database (121 for a's 409 recordings, 356 for e's 2,054). PhysioNet asks for Liu et al.,
  Physiol Meas 37(12):2181-2213, 2016, doi:10.1088/0967-3334/37/12/2181, and its platform citation, Pollard et al.,
  Nature Health, 2026 (read here, the PhysioNet page). The files are under ODC-By 1.0.
- This project's federation (`scripts/train/medical_pcg_fl.py`, three seeds): with the recordings split at random, the
  network reaches AUROC 0.961; held out source by source it reaches 0.321 to 0.728 pooled and 0.242 to 0.740 under
  FedAvg, below 0.5 on some sources in every seed. The data name no patients, so within-source splits cannot be made
  patient-disjoint, which is why the held-out sources are the honest figure.
- Liu et al. 2016, section 2.6 (read here, the author version at riunet.upv.es, handle 10251/84129): database e's 174
  healthy volunteers (172 male, aged $`25 \pm 3`$) were recorded with an MLT201 microphone or a piezoelectric sensor in
  the Biomedical Engineering Lab at Dalian University of Technology, its 335 patients with coronary artery disease
  ($`60 \pm 12`$ years) in the Second Hospital of Dalian Medical University with a 3M Littmann electronic stethoscope
  at the mitral position. Sensor, setting and age go with the label inside e, which no normalisation undoes.
- Koike, Qian, Schuller and Yamamoto, EMBC 2021, doi:10.1109/EMBC46164.2021.9629714 (read in full by an agent; Table II
  checked here): leaving each database out with a PANNs CNN14 pretrained on AudioSet, standardising the spectrogram raised
  the unweighted average recall on all six databases (the mean from 56.0 to 59.3; e 49.3 to 55.8), frequency masking to
  a mean of 60.4. This project's baseline already masks frequency and standardises each window; removing each band's
  mean as well raised its average but not every source.
- Humayun et al., IEEE JBHI 24(8):2189-2198, 2020 (read in full by an agent): batches balanced over domain and class
  helped on the databases seen in training and not on an unseen one (mean of sensitivity and specificity 46.37 to
  47.95). Azam et al., arXiv:2106.01865 (read in full by an agent): a stethoscope's response is additive in the log
  filterbank domain and mean normalisation removes it, the reasoning behind `--band-mean`.
- Kutlu and Colak, J Clin Med 15(20):8058, 2026 (read by an agent): models trained on CirCor and tested on the Challenge
  2016 recordings reached AUROC 0.31 to 0.74, two classical models below 0.5; label shift (the Challenge's abnormal
  includes conditions without a murmur) as well as the change of device.

