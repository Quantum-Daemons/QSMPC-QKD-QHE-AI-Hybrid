# Real anti-money-laundering data: what exists, under what licence (read 2026-11-22)

The finance models train on IBM's AMLworld HI-Small, which is synthetic, and their cards said that no commercial-OK
corpus of real laundering transactions *exists*. On the owner's choice of 2026-11-22 the claim was checked against what
is published before it was reworded. A survey found the candidates; each licence below was then read at the
dataset's own record on 2026-11-22 (Kaggle's dataset API, UCI's page, Figshare's and Zenodo's APIs), and nothing was
downloaded. "Commercial-OK" is `DATA_LICENSES.md` section 6's sense: a permissive licence.

## Real data with laundering or illicit labels

| Corpus | Read at | Licence | What its record says |
|---|---|---|---|
| Elliptic (Weber et al., 2019) | Kaggle `ellipticco/elliptic-data-set` (0.70 GB, updated 2019-07-31) | CC BY-NC-ND 4.0 | *"This anonymized data set is a transaction graph collected from the Bitcoin blockchain"*; illicit means scams, malware, terrorist organisations, ransomware and Ponzi schemes, not laundering as such |
| Elliptic2 (Bellei et al., arXiv:2404.19109) | Kaggle `ellipticco/elliptic2-data-set` (88.25 GB, updated 2024-04-23) | CC BY-NC-ND 4.0 | 122K subgraphs; *"Suspicious subgraphs are those which are indicative of money laundering"*. The code repository (`MITIBMxGraph/Elliptic2`, Apache-2.0, a licence of the code only) links `elliptic.co/elliptic2`, which redirected to the provider's home page |
| BABD-13 | Kaggle `lemonx/babd13` (updated 2024-01-16) | CC BY-NC-ND 4.0 | Bitcoin address behaviour, one class money laundering |
| Ethereum phishing transaction network (XBlock; Chen et al. 2019) | Kaggle `xblock/ethereum-phishing-transaction-network` (updated 2020-03-23) | "Unknown" | Phishing labels. It is the real-data set of Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs* (AAAI 2024, arXiv:2306.11586), the Multi-GNN the finance page runs: *"Since banks do not release their data, we turn to cryptocurrencies for a real-world dataset"* |
| Corporate bank accounts (Gueneau, Celisse and Delange, arXiv:2507.08835, 2025) | Kaggle `haroldg/time-series-of-transactions-money-laundering` (2.32 GB, updated 2025-06-10) | GPL 3 | *"anonymised 3-month transaction time-series (corporate bank accounts), labelled for anti-money-laundering (AML) research"*, accounts tagged by *"an in-house monitoring engine"*, privacy-preserved with a differential-privacy mechanism at $`\epsilon = 1`$ and a PCA projection; it calls itself *"the first publicly available, real-life AML time-series dataset at account level"* |

So a real AML-labelled corpus does exist, and the strongest wording a search supports is that none was found under a
permissive licence. The cards, the registry note, `DATA_LICENSES.md` and `docs/models-and-benchmarks.md` now say so,
in the owner's chosen words.

## Real data under CC BY 4.0, with related labels

| Corpus | Read at | Size | Labels |
|---|---|---|---|
| BitcoinHeist | UCI dataset 526, DOI 10.24432/C5BG8V | 110.7 MB download, 2,916,697 rows | ransomware families against "white" addresses (not verified clean) |
| BitcoinTemporalGraph | Figshare article 26305093, version 3, DOI 10.6084/m9.figshare.26305093.v3 | `dataset.tar.gz` 17.39 GB, `addresses.csv` 0.01 GB | *"a graph of value transfers between Bitcoin users"* (a PostgreSQL dump) and about 100,000 labelled addresses |
| Ransomwhere | Zenodo record 6512122, version 1.1.0 | small | crowdsourced ransomware payments, positives only |
| BitFraud | Zenodo record 24170215, published 2026-11-12 | files restricted during review | Bitcoin fraud labels with a transaction graph |

## The owner's choices of 2026-11-22

- **External checks, planned, not replacements.** A BitcoinHeist sanity check of the tabular pipeline, a GNN check on
  BitcoinTemporalGraph (reproduce its published baselines, then adapt IBM's Multi-GNN, leaving out entity names and
  any forum text the archive carries), and BitFraud watched until its files open. Each download waits for its own
  approval. None has laundering labels, so none can stand in for AMLworld's task.
- **The GPL-3.0 bank set**: a local comparison, never published, after its own download approval. GPL-3.0 permits
  commercial use, but it is copyleft, which this repository's permissive set excludes, and whether it reaches trained
  weights is unsettled.

## The checks as run

- **BitcoinHeist, 2026-11-22** (`scripts/measure_bitcoinheist_check.py`, record
  `docs/measurements/bitcoinheist-check-2026-11-22.json`). The finance booster's settings, a threshold chosen on 2015,
  the soft logistic student and its linear score under CKKS, fitted on 2011-2014 (1,468,876 rows,
  18,592 positives, 6 families) and tested on 2016-2017 (749,117 rows,
  19,117 positives, 26 families, most of them never seen in the fit window). The booster
  reaches minority-class F1 0.069 and AUPRC 0.039 against a base rate of
  0.026 (AUROC 0.65); the linear student is at chance (AUROC 0.52, AUPRC
  0.024). Under CKKS the student's 64 scored rows match its clear score within
  $`1.1e-06`$ and all 64 decisions agree. So the pipeline and its encrypted path run on
  real rows, and a detector fitted on earlier ransomware families does not carry to new ones; it is not a laundering
  result and not a detector.
- **BitcoinTemporalGraph**: downloaded (MD5s match Figshare's). The archive is a PostgreSQL directory-format dump, two
  table files of 46.61 GB and 26.99 GB, 73.61 GB unpacked; the GNN check streams both tables out of the archive, since
  the extracted edge table read at 4 to 32 MB/s through the NTFS driver, and runs on the GPU only on the owner's
  approval of each job.

  **The GNN check, 2026-11-23** (`scripts/measure_btg_check.py`, record `docs/measurements/btg-check-2026-11-23.json`).
  The dataset's paper (Schnoering and Vazirgiannis, Scientific Data 12, 404 (2025),
  https://www.nature.com/articles/s41597-025-04684-8) predicts seven entity types from the node features on a random
  40/30/30 split. Its Table 6 gives scikit-learn's gradient boosting macro-F1 0.57, though that row's seven per-class F1
  average 0.599, and four GNNs on a two-hop sample 0.56 to 0.64 (GraphSage 0.62). Its labels come from BitcoinTalk posts
  and profiles, exchanges' published wallets, the Padua and Montreal ransomware sets, the US Treasury's SDN list and
  miners' coinbase messages. The dump holds the paper's 252,219,007 nodes, 785,954,737 edges and 34,098 labelled nodes,
  33,708 of them in the seven classes, and the edges give every one of those as many distinct neighbours as its stored
  degree. On a stratified 40/30/30 split over 5 seeds, with the 27 of the paper's 35 features that need no BTC/USD price
  series and each model selected on the validation split's macro-F1, test macro-F1 (mean and standard deviation over the
  seeds) is $`0.547 \pm 0.016`$ for the paper's GBC rerun, $`0.614 \pm 0.014`$ for a class-balanced HistGradientBoosting
  booster, $`0.470 \pm 0.003`$ for the GNN's network without neighbours and $`0.570 \pm 0.007`$ for two GraphSAGE layers
  over a one-hop sample of 10 neighbours. The selection rule was changed once, after a look at seed 0's validation
  curves: training first kept the epoch of lowest validation loss, as the authors' code does, which stopped the GNN
  about 0.07 macro-F1 short of its best validation score and the network without neighbours about 0.02, so every model
  now keeps its best validation macro-F1. Seed 0's test scores under the first rule had been seen before the change;
  the record keeps every run's lowest-loss epoch and its validation score. The neighbours add +0.099 to the same
  network on all five seeds, most on
  RANSOMWARE (F1 0.309 against 0.108), but the booster on the node's own features stays ahead of the one-hop GNN on
  every seed. The rerun is within 0.05 of the paper's GBC on five classes and short on EXCHANGE (0.466 against 0.54) and
  above all RANSOMWARE (0.139 against 0.50), where the missing USD amounts may matter, ransoms being priced in dollars;
  that is not tested here. The split is by node, as the paper's, so one entity's clusters can sit on both sides; it is
  not a laundering result and nothing here is a detector.
- **The GPL-3.0 corporate-account set**: downloaded for a local comparison; nothing derived from it is committed.

## Read by the survey and not verified here

These were reported by the survey and are not relied on: BybitML (FlowShield, arXiv:2608.17355) as CC BY-NC 4.0 on a
private OSF project; the EVM dataset of arXiv:2607.18869, whose Figshare link returned no page; DIAM, T-Finance,
EthereumHeist and ICIJ's FinCEN Files transaction data, each reported without a licence. BitcoinTemporalGraph's
published baselines and how its labels were made, listed here on 2026-11-22, were read in its paper on 2026-11-23 (the
GNN check above). The systematic review arXiv:2405.19383 (v4) classes datasets as proprietary, open-source or synthetic
and gives no licences.
