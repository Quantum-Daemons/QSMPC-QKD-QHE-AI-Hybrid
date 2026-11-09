# What the 5G-NIDD file holds, and what this project's figures mean

Measured 2026-11-09 (Round 86) on `BTS1_BTS2_fields_preserved.csv` and the capture files of the same release. This replaces the accounts that Rounds 49 and 50 and the first half of Round 86 gave of this corpus; what each got wrong is at the end, because the numbers they produced were published.

## The finding

The file holds 1,215,890 flow records. **281,529 of them are the same flows as 281,529 others, under a contradictory label.** One attacker's UDP flood, from `10.155.15.7` to the victim `10.41.150.68`, is in the file twice:

| | first base station | second base station |
|---|---|---|
| per-attack file | `UDPFlood1.csv`, 467,717 rows | `UDPFlood2.csv`, 286,197 rows |
| flows from `10.155.15.7` to the victim | 281,529 | 281,529 |
| flows from `10.155.15.4` to the victim | 175,811 | 0 |
| `Label`, `Attack Type`, `Attack Tool` of the `.7` flows | Benign, Benign, Benign | Malicious, UDPFlood, Hping3 |

**The evidence, each item checkable from the release:**

1. **The rows.** The two sets agree row for row, in file order, on `StartTime`, `LastTime`, `Dur`, `SrcAddr`, `DstAddr`, `Proto`, `Sport`, `Dport`, `TotPkts` and `TotBytes`. Of all 113 columns, the only ones that differ are the three labels and the file-position columns `Unnamed: 0`, `Rank`, `Seq`, `Offset` and `IdleTime`, and `Cause` and `State` on a single row.
2. **The captures.** `UDPflood_BS1_nogtp.pcapng` (1,214,969 packets) and `UDPflood_BS2_nogtp.pcapng` (656,389 packets) both hold the 421,471 packets from `.7` to the victim, in the same order and with the same length and ports. **All 421,471 timestamps are identical.** Two capture points cannot stamp a packet at the same instant, so one capture is in both files. The packets run from 2022-08-26 16:03:48 to 16:13:32 EEST.
3. **The schedule.** The dataset's descriptor runs each attacker for 10 minutes of a 30-minute capture, overlapping by 5. In `UDPFlood1.csv` the `.4` flows run from minute 58:18 to 08:43 and the `.7` flows from 03:49 to 13:32, which is that overlap.
4. **The arithmetic.** Rows with one feature vector get one prediction, so at best one copy of each pair is scored right:

$$
\text{accuracy} \;\le\; 1 - \frac{281{,}529}{1{,}215{,}890} = 0.76846
$$

The ceiling measured on the 46 columns is 0.768455. For the binary task on 27 of Wang, Fok and Thing's inputs it is 0.768458. No other cause is needed to reach that figure.

**Why the labels disagree can be read from the files, and that it is an error cannot be proved from them.** The descriptor says the data were labelled and each attack given its own label, without saying how, and the authors' published scripts have no labelling step. In all 18 per-attack files exactly one source and destination pair carries the attack label, and no flow of that pair is labelled anything else, so labels go by address pair and session. In `UDPFlood1.csv` that pair is `.4` to the victim, so the `.7` flows it also holds came out Benign. Why that capture holds them is not documented.

## What the loader does

`load_5g_nidd` removes the 281,529 rows labelled Benign, keeping the UDPFlood copy, which is what the traffic was. The rule is on the labels and the flow key (`nidd_contradictory_copies`), and the loader asserts the count, the address pair and that every copy is in the first file, so a re-release that changes any of them stops the build instead of removing rows. Benign falls from 477,737 to 196,208 and the rows to 934,361. The fit on the rows as published is kept beside the result in `contradictory_copies`, on every run.

## What the copies did to every number

The same booster, the same 46 columns, seed 42.

| | rows as published | copies removed |
|---|---|---|
| rows | 1,215,890 | 934,361 |
| accuracy | 0.7616 | **0.9979** |
| macro F1 | 0.9275 | **0.9972** |
| ceiling of these columns | 0.7685 | 0.999996 |
| accuracy with `IdleTime` restored | 0.9829 | 0.99999 |
| first base station to second: accuracy (UDPFlood F1) | 0.4199 (0.000) | 0.9973 (1.000) |
| second base station to first: accuracy (UDPFlood F1) | 0.6059 (0.555) | 0.9877 (1.000) |
| linear model on the numeric columns: accuracy | 0.7549 | 0.9838 |

On the cleaned rows the per-class F1 is 0.9873 for SlowrateDoS, 0.9934 for HTTPFlood and at least 0.9983 for the other seven. On the rows as published the two classes involved scored 0.6401 (Benign) and 0.7308 (UDPFlood); both are now at 1.0000.

## `IdleTime` names the file

It holds 17 values over the file's rows, all exact multiples of 128, the spacing of a 32-bit float at this magnitude, one for each attack session and base station. Read as Unix times they fall between 2022-09-13 and 2022-09-15, while the captures are from 26 and 29 August 2022, so it is neither a duration nor the capture time. It is one of the columns that tells the two copies apart, which is why it was worth 22 accuracy points on the rows as published. It stays out, along with the four row counters; `file_label_counterfactual` and `leak_counterfactual_*` re-measure them, and on the cleaned rows each still takes accuracy from 0.9979 to about 1.0000.

## What this says about published figures

On the rows as published, inputs equal on the two copies cannot pass 0.7685, on the binary task or on nine classes. A score above that rests on a column that differs between the copies.

- The dataset paper's top-10 inputs for its 9-class models include `Seq` and `Offset` (Samarakoon et al., arXiv:2212.01298), and its random forest reports 99.48%.
- Of Wang, Fok and Thing's 28 inputs (arXiv:2504.20436, Table III) exactly one, `Offset`, differs between the copies. Fitted to those 28, this project's booster scores binary accuracy **0.9599 with `Offset` and 0.7643 without it** (`published_inputs_check`). Their network is not rerun here.
- Their cross-base-station fall, from above 98% to 78.23% and 74.28% for the classical CNN, was read as different traffic patterns at the two stations. On this project's booster the same test gives 0.4199 and 0.6059 with the copies and 0.9973 and 0.9877 without them. The direction trained on the first station learns the flood as Benign and is then tested on it as UDPFlood.
- Neither the descriptor nor Wang, Fok and Thing mention the copies.

## What remains weak

- **SYNFlood across base stations.** Trained on the second station and tested on the first, its F1 is 0.22, while the reverse is 0.997.
- **The scan classes lean on TTL.** Without `sTtl`, `dTtl`, `sHops` and `dHops` (`without_ttl_and_hops`), UDPScan's F1 across base stations is 0.05 and 0.04. Nmap's raw probes carry a TTL of their own, so this is a property of the tool and a real signal, but it is one signal.
- Window statistics computed from the raw captures take UDPScan to 0.99 across base stations without TTL, and do not change SYNFlood: [`5g-nidd-window-features.md`](5g-nidd-window-features.md).
- **A random split is the easier test**, and 64% of test rows have a feature vector also present in the training rows. On the remaining rows accuracy is 0.9942 and macro F1 0.9812 (`test_rows_unseen_in_training`).

## What earlier rounds got wrong

- **Round 49** published 0.7608 on the narrow export and attributed the gap to published figures to row identifiers, without checking.
- **Round 50** found that one column, `IdleTime`, closed the gap, took it for a flow statistic because 17 values cannot address a row, and published 0.9829. The column addresses the file. Round 50 also corrected a claim of its own that the other columns absent from `Combined.csv` score AUC 0.5: `Dir` and `TcpOpt` are informative and redundant.
- **Round 86, first half** removed the column, republished the model at 0.7616 on the Hub before any page used it, and read that figure and its ceiling 0.7685 as a shortage of information per flow. The ceiling was the arithmetic of the copies, found by looking at the addresses behind the two classes that confused.
- An earlier wide fit briefly recorded 0.9997 as confirmation, with `Unnamed: 0` still in the columns. It is the same leak the counterfactual now measures.

## Sources

- Y. Siriwardhana et al., "Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)", *IEEE Data Descriptions* 2, 358-369, 2025, doi:10.1109/IEEEDATA.2025.3592888. Read in full; open access, CC BY.
- S. Samarakoon et al., "5G-NIDD: A Comprehensive Network Intrusion Detection Dataset Generated over 5G Wireless Network", arXiv:2212.01298. The schedule of the sessions and the tools' command lines.
- The dataset's record, doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361, CC BY 4.0: `BTS1_BTS2_fields_preserved.zip`, `BS1_each_attack_csv.zip`, `BS2_each_attack_csv.zip`, `BS1_GTP_removed.zip` and `BS2 GTP removed.zip`, each checked against the sha256 in the record.
- The authors' scripts, https://github.com/yushan1986/5g-nidd, read 2026-11-09.
- Z. Wang, K.-W. Fok and V. L. L. Thing, arXiv:2504.20436, Tables II to VI.
- M. Flood et al., "Bad Design Smells in Benchmark NIDS Datasets", IEEE EuroS&P 2024, on near-duplicate flows, and D. Arp et al., "Dos and Don'ts of Machine Learning in Computer Security", USENIX Security 2022, on data snooping and spurious correlations.
