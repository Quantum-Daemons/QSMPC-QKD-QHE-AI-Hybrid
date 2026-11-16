# IBM's Multi-GNN on this project's windows (Round 86)

The finance page compares its compact GNN with IBM's official Multi-GIN+EU and Multi-PNA+EU, run on the same temporal windows, and
the compact GNN is distilled from that PNA. These files reproduce both from IBM's code (`IBM/Multi-GNN`, Apache-2.0) without
vendoring it.

| File | What it does |
|---|---|
| `format_amlworld.py` | Writes IBM's `formatted_transactions.csv` from this repository's AMLworld loader, with the official encodings and the rows in `load_sorted()` order |
| `project_split.patch` | Two changes to IBM's code at commit `252b0252afca109d1d216c411c59ff70753b25fc`: with `QSH_SPLIT=project` the fit, validation and test windows are this project's 56 / 14 / 30 percent by position, and `AddEgoIds` defines `forward`, which PyTorch Geometric 2.8 requires of a transform |
| `dump_logits.py` | Loads a saved checkpoint and writes the laundering probability of every edge of the three windows, keyed by edge id, and counts the seed edges the neighbour sampling never returned |
| `precise_bn.py` | Recomputes a saved checkpoint's BatchNorm statistics on the fit window and writes that checkpoint, with the validation and test windows (and with `--with-train` the fit window) scored under both statistics on the same sampled batches; the export takes only its checkpoint (the owner's choice of 2026-11-16, below) |
| `causal.patch` | On top of `project_split.patch`, the causal protocol under `QSH_CAUSAL=1` (Round 86, the owner's choices of 2026-11-15): each mini-batch is one batch of transactions in time order (2,048, split where it spans more than 17 minutes; `QSH_CAUSAL_CUT=batch` keeps the plain 2,048), sampled with IBM's own shared sampler, with every sampled edge later than that batch dropped, and every window scaled by the training window's statistics; with `QSH_PATIENCE=n` training stops after n epochs without a better validation F1, and `QSH_LOADER_WORKERS=n` samples in n processes |
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
# For the export: the statistics recomputed on the fit window, then the ONNX file and the page's sample
cp <this repository>/scripts/train/ibm_multignn/precise_bn.py .
QSH_SPLIT=project WANDB_MODE=disabled python precise_bn.py --data Small_HI --model pna --emlps --reverse_mp --ego --ports --testing --seed 1 --checkpoint <model_to_save>/checkpoint_False.tar --inference_out <fit-bn dir>
QSH_SPLIT=project python <this repository>/scripts/train/ibm_multignn/export_for_browser.py --checkpoint <fit-bn dir>/checkpoint_fit_bn.tar --val-probs <fit-bn dir>/val.npz --protocol ibm --fixtures <this repository>/models/finance_aml_mlp_large/weights.json --out <artifact dir>
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

## Five seeds under IBM's protocol (2026-11-15 and 16)

Multi-PNA+EU on this project's windows, seeds 1 to 5, each IBM's 100 epochs with `--save_model`; seed 1's row is its
saved run above. IBM's loop keeps an epoch when its validation F1 is higher than every earlier epoch's, never the first
epoch; the epoch shown is the one its `save_model` stores, the loop's index plus one.

| Seed | Best validation F1, epoch | Test F1 there |
|---|---|---|
| 1 | 0.6177, 36 | 0.6616 |
| 2 | 0.6089, 38 | 0.6506 |
| 3 | 0.5928, 54 | 0.6254 |
| 4 | 0.6208, 33 | 0.6751 |
| 5 | 0.5732, 39 | 0.6309 |

Over the five, the test F1 has mean $`0.6487`$ and sample standard deviation $`0.0208`$; the paper reports
$`68.16 \pm 2.65`$ over five runs on its own split. Seed 4 has the best validation F1, so it is the seed exported.
Seeds 1 to 3 trained on kernel 5.14.0-687.66.1.el9_8 and seeds 4 and 5 on 5.14.0-687.66.2.el9_8, all with driver
620.70.06, torch 2.13.0+cu130, PyTorch Geometric 2.8.1 and cuDNN 9.20.0. The causal run started from the same
interpreter and environment on 2026-11-16 maps cuBLAS 13.10.0.20 of the CUDA 13.5 toolkit (through `LD_LIBRARY_PATH`)
and the CUDA 12 cuBLAS of the `nvidia-cublas-cu12` 12.9.3.13 wheel.

## IBM's loop never switches to evaluation mode (2026-11-16)

IBM's code at `252b025` calls neither `model.eval()` nor `model.train()` anywhere, so the model stays in training mode
throughout. Two things follow.
- The validation and test F1 its loop logs, the five-seed figures above among them, are computed with the head's dropout
  on (0.288 for PNA) and BatchNorm normalising each sampled subgraph by its own statistics. This is IBM's protocol, and
  the paper's figures come from the same loop.
- Every validation and test pass updates BatchNorm's running statistics, and the checkpoint is saved after the test
  pass, so a saved checkpoint's statistics are set by the test window's last batches. Its `num_batches_tracked` is the
  epochs times the 348 fit, 87 validation and 186 test batches of an epoch: 22,356 for seed 1 after 36 epochs and
  20,493 for seed 4 after 33. Any eval-mode use of a checkpoint, as `dump_logits.py`, the browser export and the finance
  page all make, normalises with test-window statistics.

`precise_bn.py` recomputes the statistics as a plain average over one pass of the protocol's fit-window loader
(precise BatchNorm, Wu and Johnson, arXiv:2105.07576; Ioffe and Szegedy, arXiv:1502.03167, normalise at inference by
the training set's statistics), and scores the windows with both statistics on the same sampled batches. On seed 4,
the checkpoint the tab exports:

| BatchNorm statistics | Test F1 at 0.5 | Test AUPRC | Validation-tuned threshold | Test F1 there |
|---|---|---|---|---|
| As saved, set by the test pass | 0.6801 | 0.6299 | 0.536 | 0.6804 |
| Recomputed on the fit window | 0.6723 | 0.6318 | 0.637 | 0.6822 |

178 of the 1,522,629 test decisions at 0.5 change. Over the five seeds, with the recomputed statistics, the test F1 at
the validation-tuned threshold has mean $`0.6577`$ and sample standard deviation $`0.0226`$ (with the saved statistics
$`0.6597 \pm 0.0233`$) and the test AUPRC $`0.6197 \pm 0.0304`$; the thresholds absorb the shift, and the ranking does
not move. The owner's choice of 2026-11-16: every exported checkpoint, under
both protocols, has its statistics recomputed on the fit window, and `export_for_browser.py` refuses one that has not;
the card gives the figure IBM's loop logs beside the exported model's. Under the causal protocol this keeps the
normalisation, too, from reading past a transaction's horizon.

## The causal protocol (Round 86)

IBM's test graph holds every transaction of the test window, so the neighbourhood its loader samples for a transaction
includes later ones; the page's own models see a transaction's neighbours up to its horizon, the end of its batch of
2,048 in time order and at most 17 minutes after it (`finance_aml_fan.batch_reach`, the furthest any fit-window row sees
to the end of its batch). The owner asked for both: IBM's protocol, labelled, and a causal one comparable with the
page's models.

PyG's own temporal sampling (`time_attr`, `edge_label_time`) was tried first and set aside. It samples each seed in its
own subgraph, and for a link it grows one tree from each end, so the seed transaction appears as two copies, each joining
one endpoint to a copy of the other, and no edge joins the two endpoints IBM's model scores; at a fan-out of 10, five of
128 seeds were not sampled in their own subgraph at all. `causal.patch` keeps IBM's shared sampler and makes each
mini-batch one batch of seeds, so a single cut serves the whole mini-batch, then drops the sampled edges later than it.
A batch is 2,048 rows, split further where it spans more than 17 minutes (`causal_batches` in the patched
`data_loading.py`), so no seed reads past its horizon. The split changes no fit or validation batch (1,389 and 348 on
HI-Small) and only the test window's sparse last days (745 batches become 999): there a batch of 2,048 spans up to days,
and without the split 654 of the 2,321 test laundering transactions (28%) could read transactions more than 17 minutes
later, up to 7.7 days (measured 2026-11-15). The owner chose the split for the tab and the card, and the test F1 of the
same checkpoints under the plain batches of 2,048 (`QSH_CAUSAL_CUT=batch`) recorded beside it. A node with more than the fan-out of edges keeps the earlier ones among those sampled from all of its edges, so its
causal neighbourhood can be smaller than a sampler restricted to earlier edges would draw; below the fan-out it is exact.
IBM's port numbers rank a node's neighbours by their first transaction, so they do not depend on later transactions. IBM
scales each window by its own statistics, the test window's including later transactions; the causal protocol scales all
three by the training window's.

Measured on this machine's GPU (96 GB): one IBM run holds 42 to 58 GB and samples about 2.1M edges per batch of 8,192
seeds; per-seed subgraphs would sample about 6.6M at that batch size. The causal runs use batches of 2,048 (one time
batch), early stopping after 20 epochs without a better validation F1, and four loader processes, all stated on the
model's card. The queue (`run_pna_seeds.sh` and `run_queue_after_pna.sh` in the working folder) trains five seeds of each
model under each protocol, one run at a time; each checkpoint then goes through `precise_bn.py` (its causal loader
under `QSH_CAUSAL=1`) before an export.
