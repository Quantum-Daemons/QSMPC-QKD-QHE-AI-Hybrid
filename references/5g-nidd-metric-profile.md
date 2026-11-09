# Why this project's 5G-NIDD figures invert against the published ones

Measured 2026-08-17, Round 50. Written because Round 49 published a claim about this corpus without
consulting work that already existed, and because the shape of the disagreement rules out the
explanation Round 49 offered.

## Correction, Round 86 (2026-11-09): `IdleTime` is a per-file label

This file argued from Round 50 to Round 86 that `IdleTime` is a flow statistic and that the published 0.90 to 0.98 figures are reachable without leakage. The first is wrong and the second does not follow. Reading the column's values settles it:

- It holds **17 distinct values over 1,215,890 rows, all exact multiples of 128**, which is the spacing of a 32-bit float at this magnitude.
- Read as Unix times they fall between 2022-09-13 07:49 and 2022-09-15 21:09 UTC, while the dataset paper gives the captures as 26 and 29 August 2022 (Samarakoon et al., arXiv:2212.01298, section V-A). So they are neither an idle duration nor the capture time.
- **The values go with the attack sessions and the two base stations.** The paper records each attack in its own session and each session at two base stations, into separate files. In the CSV, 1663055360 holds 10,019 SYNScan and 1,431 Benign flows and its counterpart in the other base station's file, 1663275008, holds 10,024 SYNScan and 1,502 Benign; 1663272960 holds 175,521 UDPFlood and 53,066 Benign; 1663273088 holds 238,840 Benign and 290 UDPFlood.

So the column names the file a flow came from, and each file is one attack. "17 distinct values cannot address a row" was true and beside the point: they address the file. The model that read it was recognising the file.

**What changed.** `telecom_ids.nidd` is refitted on 46 features without it: **accuracy 0.7616, macro F1 0.9275** on the same 303,973-row test split, with per-class F1 Benign 0.6401 and UDPFlood 0.7308 and every other class at 0.9875 or above. A linear model on the numeric columns reaches accuracy 0.7549. Restoring the column gives 0.9829 and 0.9950, recorded on every run as `file_label_counterfactual`, beside the row-counter one at 0.9997.

**Is 0.7616 the model's fault?** No model reading these 46 columns can exceed accuracy **0.7685** on this corpus (`flow_level_ceiling`): the rows reduce to 374,933 distinct feature vectors, 33,712 of them occur under more than one class, almost all under both Benign and UDPFlood, and those vectors hold 52% of the rows. This is the "near-duplicate flows" smell of Flood et al. (EuroS&P 2024), taken to exact identity across classes. Tuning cannot move it; more information per flow could.

**How far the random split can be believed.** Training on one base station's file and testing on the other's gives **accuracy 0.4199 and 0.6059, macro F1 0.8108 and 0.7950** (`cross_base_station`), and the UDPFlood F1 falls to 0.0 in the first direction. The headline is a stratified random split, which puts flows of one attack session on both sides. Arp et al. (USENIX Security 2022) name this family of problem as data snooping and spurious correlations.

What follows is the document as Round 50 wrote it, corrected where it depended on that reading.

## The disagreement

| source | task | accuracy | macro F1 |
|---|---|---|---|
| **this project** (`telecom_ids.nidd`) | 9-class | **0.7608** | **0.9263** |
| BiTAD (Future Internet **17**(11) 482) | 9-class | 0.9047 | 0.91 (weighted) |
| BiTAD | binary | 0.9333 | 0.91 |
| AdaBoost baseline, same corpus | 9-class | 0.91 | **0.67** |

Ours is inverted on **both** axes at once: a much higher macro F1 and a much lower accuracy than
published multiclass work on the same nine classes of the same corpus. That is the opposite of the
usual imbalanced-corpus signature, where the aggregate flatters and the macro exposes.

## What it is not

**The corpus load is correct.** All eight attack classes reproduce the published distribution to
within 0.08 percentage points - UDPFlood 61.96% of malicious traffic against 61.90% published,
HTTPFlood 19.08% against 19.00%, down to ICMPFlood 0.16% against 0.15%. Whatever differs, it is not
which rows we read.

**It is not binary-versus-multiclass, and this is the measurement that settles it.** Refitting the
identical configuration as a two-class benign-vs-attack problem:

    9-class, 39 features    accuracy 0.7608   macro F1 0.9263   weighted F1 0.7548
    binary,  39 features    accuracy 0.7630   macro F1 0.7342   weighted F1 0.7530

**Binary accuracy is 0.7630 against the 9-class 0.7608** - a difference of 0.2 points. Collapsing
eight attack classes into one buys essentially nothing, so the multiclass structure is not what costs
the accuracy. Published binary figures on this corpus are around 0.93.

**It is not macro-versus-weighted reporting.** Weighted F1 is 0.7548, close to the accuracy, so a
reader comparing our headline against a weighted number would see the same gap.

## What it is: one class pair, and it is a two-class problem

The entire error budget is a single confusion:

| true -> predicted | rows | share of test set |
|---|---|---|
| Benign -> UDPFlood | 55,591 | 18.29% |
| UDPFlood -> Benign | 16,417 | 5.40% |
| all seven other classes, combined | 699 | 0.23% |
| **total error** | | **23.92%** |

Per-class F1: `ICMPFlood` 1.0000, `SYNFlood` 1.0000, `UDPScan` 0.9984, `TCPConnectScan` 0.9983,
`SYNScan` 0.9985, `HTTPFlood` 0.9902, `SlowrateDoS` 0.9810 - then `UDPFlood` 0.7312 and `Benign`
0.6394. Benign recall is **0.5345**: the model finds barely half of benign traffic.

So the macro F1 of 0.9263 is arithmetically honest and substantively misleading on its own. Seven of
nine classes are near-perfectly separable in our feature set and the two that carry 77% of the rows
are not separable at all.

**Why the seven are easy is itself worth stating**: each attack class was generated by a distinct tool
against a distinct service, so protocol and service fingerprints identify them almost trivially. The
project's own leak check flags **21 of 39** features at one-vs-rest AUC above 0.90 (`Rate` 0.9941,
`AckDat` 0.9928, `sTtl` 0.9920). That is not a defect in the features; it is what this corpus is.

## What Round 50 measured, and what it read wrongly

The release also offers `BTS1_BTS2_fields_preserved.zip` - 113 columns against `Combined.csv`'s 52,
the same 1,215,890 rows. Fitting the identical configuration on it, with row identifiers, host
fingerprints (`SrcAddr`, `SrcMac`, `Sport`, ...) and the five bit-identical duration duplicates all
dropped:

| configuration | features | accuracy | macro F1 |
|---|---|---|---|
| wide export, identifiers and hosts dropped | 47 | 0.982897 | 0.995019 |
| the same, minus the six columns `Combined.csv` lacks | 41 | 0.760949 | 0.926455 |
| the same, **minus `IdleTime` alone** | 46 | **0.761558** | 0.927503 |
| shipped narrow (`Combined.csv`) | 39 | 0.7608 | 0.9263 |

**`IdleTime` accounts for the entire gap.** Removing that one column from the wide export reproduces
the narrow result to within 0.001 accuracy. Round 50 took this for a dominant flow statistic. It is
a per-file label, so the 0.98 row measures how well a model recognises the file each flow came from.

**A claim made here in an earlier revision was wrong and is corrected rather than quietly dropped.**
It said the other five columns absent from `Combined.csv` "change nothing, each scoring one-vs-rest
AUC 0.5000". Two of them do not: **`Dir` scores 0.9811** on ICMPFlood and **`TcpOpt` 0.9433** on
SlowrateDoS, both above this project's own 0.90 leak-screen threshold (`sVlan` 0.6207, `Flgs` 0.6136,
`dVlan` 0.5020). What is true is that they are **redundant, not empty**: dropping all six costs 0.2219
accuracy against 0.2213 for `IdleTime` alone, so the five together are worth 0.0006. Individually
informative and jointly superfluous is a different statement from uninformative, and the first
version asserted the second on the strength of a measurement taken in a different feature context.

Also verified: all 52 columns `Combined.csv` shares with the wide export are informationally
identical between the two files - equal distinct-value counts on every one - so `Combined.csv` is a
strict column subset, not a degraded copy.

### A caution earned the hard way, in this file's own measurements

The first attempt at the wide fit returned **0.9997 / 0.9988** and was recorded, briefly, as
confirmation. It was leakage: the drop list named `SrcId`, `Rank`, `Seq` and `Offset` but not
`Unnamed: 0`, so the row index survived. Reproduced deliberately as a control - keeping `Unnamed: 0`
gives exactly 0.9997, the same figure Round 49 recorded as its identifier counterfactual on the
narrow export.

That is worth stating plainly: the identifier leak is easy enough to reintroduce that it happened
here, in the pass written to investigate it, on the second attempt. The 0.9997 in Round 49's register
and the 0.9997 above are the same defect found twice.

## What Round 49 got wrong

Round 49's register says the published 99.6-99.7% figures are "an anchor rather than a target" and
that "it is not established that those papers dropped `Unnamed: 0`". Both sentences were written
without checking, and leakage-aware work on this corpus exists:

- *An empirical assessment of ML models for 5G network intrusion detection: a data leakage-free
  approach*, Computers and Security / ScienceDirect S2772671124001700.
- *TinyML Driven Intrusion Detection for 5G Network Slices with Leakage-Free Validation*, ICCK 2026.
- *Supervised Machine Learning-Based Intrusion Detection for 5G Networks: Evaluation on the 5G-NIDD
  Dataset*, Computers **15**(6), 362 (2026).

The identifier finding itself stands - restoring `Unnamed: 0`, `Seq` and `Offset` moves this
configuration from 0.7608 to 0.9997, and three integer row counters should not be able to do that.

Whether the published figures lean on such columns stays open. Round 50 said they could not, because 0.982897 was reachable with every identifier removed; that figure reads the file label, so it says nothing either way. The dataset paper's own multiclass feature ranking lists `Seq` and `Offset` among its top ten.

**What changed.** Round 50 refitted the model on the wide export with `IdleTime` kept and published it at 0.982897 and 0.995019. Round 86 removed the column; see the correction at the top.
