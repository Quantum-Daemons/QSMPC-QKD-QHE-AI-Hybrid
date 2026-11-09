# Window statistics from the 5G-NIDD captures

Measured 2026-11-09 (Round 86) by `scripts/research/nidd_window_features.py`; the numbers are in [`data/5g-nidd-window-features.json`](data/5g-nidd-window-features.json). This is research, and nothing on the demo reads it.

## The question

The flow model `telecom_ids.nidd` scores accuracy 0.9979 on 46 Argus columns, and across base stations 0.9973 and 0.9877, once the flows the file holds twice are out ([`5g-nidd-metric-profile.md`](5g-nidd-metric-profile.md)). Two weak spots remain:

- SYNFlood, trained on the second station and tested on the first, has F1 0.22.
- UDPScan across stations has F1 0.05 and 0.04 once the TTL and hop columns are removed, so it leans on one field a tool sets.

Do statistics computed from the packets themselves fix either?

## The method

**The unit** is a conversation, an unordered pair of addresses, at a one-second tick. A row is written for each second in which the pair exchanged a packet.

**It is causal.** A row reads only packets at or before its tick, over windows of 1 and 10 seconds, so a detector could compute it live. Direction is relative to the address that sent the pair's first packet.

**The 26 statistics per window**, 52 in all: packet and byte counts each way, mean and spread of length, distinct destination and source ports, the share of TCP, UDP and ICMP, the shares of TCP flag patterns each way, the inter-arrival mean and spread, the mean TCP window, and three TTL statistics. Keeping several time windows per channel is the idea of Kitsune's AfterImage (Mirsky et al., NDSS 2018), which uses damped windows with decay $`d_\lambda(t) = 2^{-\lambda t}`$ and 115 features; these are plain sliding windows.

**The labels** follow the dataset's rule, which the script reads and checks from its per-attack files: in each attack session exactly one address pair is the attack, and everything else is Benign. The second station's attacker in the first station's UDP-flood capture is left out.

**The test** is one base station against the other, with the same booster as the flow model, over five booster seeds. There are 104,605 rows, 65,275 from the first station. The classes are far apart in size: 98,434 Benign rows, 72 SYNScan and 79 TCPConnectScan, because a scan lasts about half a minute in these captures.

## The results

Accuracy and macro F1, mean of five seeds (the macro F1 standard deviation is at most 0.0031):

| window features | first station to second | second station to first |
|---|---|---|
| all 52 | 0.9995, 0.9758 | 0.9932, 0.8927 |
| without the 6 TTL statistics | 0.9996, 0.9830 | 0.9975, 0.8987 |

F1 of the classes in question. A flow row and a window row are different units, so this compares what each can do, not one test set.

| class, direction | flow model, 46 columns | flow model without TTL and hops | windows without TTL |
|---|---|---|---|
| UDPScan, first to second | 0.9979 | 0.0493 | 0.9899 |
| UDPScan, second to first | 0.9975 | 0.0416 | 0.9947 |
| SYNFlood, first to second | 0.9969 | 0.9971 | 0.9996 |
| SYNFlood, second to first | 0.2229 | 0.8297 | 0.8433 |
| SYNScan, second to first | 0.9984 | 0.9363 | 0.3260 |

## What it says

1. **Windows remove UDPScan's dependence on TTL**, from 0.05 to 0.99 across base stations. Nmap's raw probes carry a TTL of their own, spread evenly over 23 values from 36 to 58 at the capture point, so TTL is a real signature of the tool, but a detector that loses that one field loses the class. What a window sees instead is many destination ports from one source and the victim's replies.
2. **SYNFlood's weakness is TTL itself, and windows do not change that.** In the first station's SYN flood about a third of the packets arrive with TTL 64 and the rest with 63; in the second station all arrive with 63. A model trained on the second treats 64 as something else: F1 0.22 for the flow model and 0.23 for the window model with TTL. Removing TTL gives 0.83 and 0.84.
3. **Windows are not a replacement.** Without TTL, SYNScan from the second station to the first falls to 0.33, on 34 windows of that class in the test set and 38 in training.

Nothing here beats the flow model across the board, so the demo keeps it. If the demo ever needs UDPScan without TTL, a window model answers that, and its input is a second of a conversation and not one flow record.

## Sources

- Y. Mirsky, T. Doitshman, Y. Elovici and A. Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS 2018, arXiv:1802.09089. Damped statistics per host, channel and socket; 115 features. Read in full for the feature definitions.
- Y. Siriwardhana et al., "Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)", *IEEE Data Descriptions* 2, 358-369, 2025. The captures and the session schedule.
- The dataset's record, doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361, CC BY 4.0. The script reads `BS1_GTP_removed.zip`, `BS2 GTP removed.zip`, `BS1_each_attack_csv.zip` and `BS2_each_attack_csv.zip`.
