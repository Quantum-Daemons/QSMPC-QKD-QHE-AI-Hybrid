# Chest X-ray federation on NIH ChestX-ray14: what was read, and what the page's card rests on

Read on 2026-11-14 for the medical page's chest X-ray card (`scripts/train/medical_cxr_fl.py`,
`web/src/components/CxrFlCard.tsx`). A research agent opened the sources named here; "read" says how far: the full text,
an abstract, a table, or the NIH files in the Box viewer (nothing downloaded from it). Figures this project computed from
`Data_Entry_2017_v2020.csv` and the official lists are marked [here].

## The dataset

- Wang, Peng, Lu, Lu, Bagheri and Summers, *ChestX-ray8*, CVPR 2017, arXiv:1705.02315 (v5, the latest): 112,120 frontal
  images in the ChestX-ray14 appendix; labels mined from the reports with DNorm and MetaMap and negation rules on
  dependency graphs. The paper's accuracies compare the mining with people reading the reports, not the images: on 900 NIH
  reports, Effusion precision 94.8, recall 99.2 (Table 18). Effusion AUROC on the published split, ResNet-50 from ImageNet
  with only the last layers trained: 0.7585 (Table 17). (read: full text)
- The NIH README gives 30,805 patients; the FAQ says an uncertain mention is labelled No Finding, and "The usage of the
  data set is unrestricted" with a link, an acknowledgement of the NIH Clinical Center and the CVPR citation asked for.
  (read: Box viewer) The optional MAPLEZ labels in the same folder are CC BY-NC 4.0 and are not used here.
- The released `BBox_List_2017.csv` holds 984 boxes on 880 images, not the 1,600 the paper describes; Effusion has 153 boxes
  on 153 images. (read: Box viewer) All 153 are in `test_list.txt` [here]. Coordinates are on the 1024 px PNGs.
- NIH's change log of 2026-10-09 lists pixel-identical images, four pairs across the official split. (read: Box viewer)
  This project found all ten listed pairs among 25 pixel-identical groups and drops 26 images [here].
- View and prevalence [here]: train_val AP 30,310 images (Effusion 12.6%), PA 56,214 (8.6%); test AP 14,500 (20.0%), PA
  11,096 (15.8%).

## The labels

- Oakden-Rayner, *Exploring large-scale public medical image datasets*, Academic Radiology 27(1):106-112, 2020,
  doi:10.1016/j.acra.2019.10.006: a radiologist re-read 50 images per label on the images themselves; Effusion's PPV was 74%
  (permissive) and 70% (clinical). (read: full text of arXiv:1907.12720v1; the journal abstract reports PPVs mostly 10 to
  30% below the documentation)
- Majkowska et al., Radiology 2020: adjudicated labels for pneumothorax, nodule or mass, airspace opacity and fracture; not
  effusion. (read: abstract)

## The view as a confounder

- Baltruschat et al., Sci Rep 9:6381, 2019, doi:10.1038/s41598-019-42294-8 (arXiv:1803.02315v2): frozen features of a
  from-scratch ResNet-50 predict AP against PA with AUC 0.9983; on the official split, Effusion AUROC 0.822, 0.818 and 0.818
  for ResNet-38, -50 and -101 at 448 px with age, sex and view (Table 5), about 0.05 below their random patient-level splits
  (Table 3). (read: full text)
- Zech et al., PLOS Medicine 2018: a CNN tells the hospital system (99.95% on NIH) and portable from not (100% at Mount
  Sinai) from the image; it does not analyse AP against PA. (read: full text)
- Farquhar, medRxiv 2026-01-27 (not peer reviewed): on NIH, sensitivity gaps between AP and PA of 22 to 50 points at Youden
  thresholds across four models, for an infiltration proxy; no effusion figures. (read: full text, two tables did not render)
- No paper was found that reports one model's Effusion AUROC by view on the official test split, which the card does.

## Federated learning on chest X-rays

- Tayebi Arasteh et al., Sci Rep 13:22576, 2023, doi:10.1038/s41598-023-49956-8: five datasets as sites, ResNet-50 from
  ImageNet-21K, FedAvg with one local epoch a round; on ChestX-ray14's official split, pleural effusion 81.5 alone against
  82.6 federated, with no pooled baseline (Table 2). (read: full text)
- Kulkarni et al., arXiv:2301.06683v6: NIH and CheXpert as two clients, a random NIH split; pooled 0.82, FedAvg 0.81 on
  NIH over seven labels. (read: full text)
- No paper was found that splits sites by view position, which the card's hospitals do.

## Where the maps point

- Zhang et al., ECCV 2016 (arXiv:1608.00507): the pointing game counts a hit when a map's maximum falls on the object,
  widened by a 15-pixel tolerance, and reports a centre baseline that always points at the image's centre. (read: full text)
- Saporta et al., Nature Machine Intelligence 4:867-878, 2022, doi:10.1038/s42256-022-00536-x: on CheXpert's radiologist
  segmentations, every saliency method fell short of a human benchmark; for pleural effusion with DenseNet-121 ensembles,
  Grad-CAM's hit rate on true positives was 0.507 (Table 1). (read: full text)
- Wang et al. score their maps on NIH's boxes by intersection over the detected box: Effusion 0.7974 at a threshold of
  0.1 and 0.4183 at 0.5 (Table 19), with heatmaps of 32 by 32. The card's maps are 7 by 7 before upsampling.

## The second backbone

- ConvNeXt-atto is a timm configuration (`timm/models/convnext.py`): depths 2, 2, 6, 2; widths 40, 80, 160, 320; 3.70 M
  parameters and 0.55 GMACs at 224 px; ImageNet-1k top-1 75.672. torchvision ships only tiny to large, so the card's is
  built from torchvision's `ConvNeXt` class with that block setting: the same function, not module for module the same.
  With a one-channel stem and one logit it has 3,373,561 parameters. (read: code and the results tables)
