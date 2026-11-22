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

## Read by the survey and not verified here

These were reported by the survey and are not relied on: BybitML (FlowShield, arXiv:2608.17355) as CC BY-NC 4.0 on a
private OSF project; the EVM dataset of arXiv:2607.18869, whose Figshare link returned no page; DIAM, T-Finance,
EthereumHeist and ICIJ's FinCEN Files transaction data, each reported without a licence; and BitcoinTemporalGraph's
published baselines and how its labels were made. The systematic review arXiv:2405.19383 (v4) classes datasets as
proprietary, open-source or synthetic and gives no licences.
