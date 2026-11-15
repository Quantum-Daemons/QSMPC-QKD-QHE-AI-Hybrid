# IBM's Multi-GNN on this project's windows (Round 86)

The finance page compares its compact GNN with IBM's official Multi-GIN+EU and Multi-PNA+EU, run on the same temporal windows, and
the compact GNN is distilled from that PNA. These files reproduce both from IBM's code (`IBM/Multi-GNN`, Apache-2.0) without
vendoring it.

| File | What it does |
|---|---|
| `format_amlworld.py` | Writes IBM's `formatted_transactions.csv` from this repository's AMLworld loader, with the official encodings and the rows in `load_sorted()` order |
| `project_split.patch` | Two changes to IBM's code at commit `252b0252afca109d1d216c411c59ff70753b25fc`: with `QSH_SPLIT=project` the fit, validation and test windows are this project's 56 / 14 / 30 percent by position, and `AddEgoIds` defines `forward`, which PyTorch Geometric 2.8 requires of a transform |
| `dump_logits.py` | Loads a saved checkpoint and writes the laundering probability of every edge of the three windows, keyed by edge id, and counts the seed edges the neighbour sampling never returned |
| `causal.patch` | On top of `project_split.patch`, the causal protocol under `QSH_CAUSAL=1` (Round 86, the owner's choice of 2026-11-15): each mini-batch is one batch of 2,048 transactions in time order, sampled with IBM's own shared sampler, with every sampled edge later than that batch dropped, and every window scaled by the training window's statistics; with `QSH_PATIENCE=n` training stops after n epochs without a better validation F1, and `QSH_LOADER_WORKERS=n` samples in n processes |
| `export_for_browser.py` | Exports a checkpoint as one ONNX file at opset 18 (by `torch.export`; ONNX Runtime within $`10^{-4}`$ of PyTorch on the seeds) and writes the 16 held-out transactions the published finance MLP and compact GNN are checked on, with the neighbourhood the protocol samples for them, for the finance page to re-score |
| `neighbourhood.py` | What the model read for each of the 16 (the owner's choice of 2026-11-15): the transfers at the payer and the payee, those within two hops, and how many came after the scored one, in its batch of 2,048 or beyond it. The export writes these counts into the sample with each edge's row and time; the page derives them again and reports a difference (`web/src/lib/graphNeighbourhood.ts`, held with this file to one hand-counted graph by `web/scripts/check_graph_neighbourhood.mjs` and `tests/test_r86_ibm_neighbourhood.py`) |

## Steps

```bash
git clone https://github.com/IBM/Multi-GNN && cd Multi-GNN && git checkout 252b0252afca109d1d216c411c59ff70753b25fc
git apply <this repository>/scripts/train/ibm_multignn/project_split.patch
cp <this repository>/scripts/train/ibm_multignn/dump_logits.py .
# data_config.json: point "aml_data", "model_to_save" and "model_to_load" at local folders
python <this repository>/scripts/train/ibm_multignn/format_amlworld.py <aml_data>/Small_HI
QSH_SPLIT=project WANDB_MODE=disabled python main.py --data Small_HI --model pna --emlps --reverse_mp --ego --ports --testing --n_epochs 100 --seed 1 --save_model
QSH_SPLIT=project WANDB_MODE=disabled python dump_logits.py --data Small_HI --model pna --emlps --reverse_mp --ego --ports --testing --seed 1 --inference_out <logits dir>
python scripts/train/finance_aml_gnn.py --teacher-probs <logits dir>/train.npz
```

The checkpoint is the epoch with the best validation F1; IBM's code saves it as `checkpoint_False.tar` because its `--unique_name` is a
flag, and `dump_logits.py` loads the same name.

## What was measured (2026-11-10 and 11)

| Model, seed 1 | Best validation F1, epoch | Test F1 there | Test AUPRC |
|---|---|---|---|
| Multi-GIN+EU, the paper's split | 0.5833, 79 | 0.6205 (paper: $`64.79 \pm 1.22`$ over five seeds) | - |
| Multi-GIN+EU, this project's windows | 0.5988, 33 | 0.6024 | - |
| Multi-PNA+EU, this project's windows | 0.6182, 25 | 0.6878 (paper: $`68.16 \pm 2.65`$) | - |
| Multi-PNA+EU, the same command with `--save_model` | 0.6177, 36 | 0.6616 | 0.6415 |

The two PNA runs share the seed and differ by 0.026, which is GPU non-determinism and of the order the paper's spread shows. From the
dumped probabilities of the saved run: the test window's 2,321 laundering edges were all scored, and the 875 seed edges the sampling
never returned (0.06 percent) are all legitimate, so IBM's evaluation, which drops unsampled seed edges on HI-Small, scores the same
F1 as the whole window would; at a threshold tuned on the validation window, as this project's models are, the test F1 is 0.6782.
The test graph holds every edge of the test window, so these models' neighbourhoods include transactions later than the one scored.

## The causal protocol (Round 86)

IBM's test graph holds every transaction of the test window, so the neighbourhood its loader samples for a transaction
includes later ones; the page's own models see transactions up to the end of their batch of 2,048 in time order. The
owner asked for both: IBM's protocol, labelled, and a causal one comparable with the page's models.

PyG's own temporal sampling (`time_attr`, `edge_label_time`) was tried first and set aside. It samples each seed in its
own subgraph, and for a link it grows one tree from each end, so the seed transaction appears as two copies, each joining
one endpoint to a copy of the other, and no edge joins the two endpoints IBM's model scores; at a fan-out of 10, five of
128 seeds were not sampled in their own subgraph at all. `causal.patch` keeps IBM's shared sampler and makes each
mini-batch one batch of 2,048 seeds, so a single cut serves the whole mini-batch, then drops the sampled edges later than
it. A node with more than the fan-out of edges keeps the earlier ones among those sampled from all of its edges, so its
causal neighbourhood can be smaller than a sampler restricted to earlier edges would draw; below the fan-out it is exact.
IBM's port numbers rank a node's neighbours by their first transaction, so they do not depend on later transactions. IBM
scales each window by its own statistics, the test window's including later transactions; the causal protocol scales all
three by the training window's.

Measured on this machine's GPU (96 GB): one IBM run holds 42 to 58 GB and samples about 2.1M edges per batch of 8,192
seeds; per-seed subgraphs would sample about 6.6M at that batch size. The causal runs use batches of 2,048 (one time
batch), early stopping after 20 epochs without a better validation F1, and four loader processes, all stated on the
model's card. The queue (`run_pna_seeds.sh` and `run_queue_after_pna.sh` in the working folder) trains five seeds of each
model under each protocol, one run at a time.
