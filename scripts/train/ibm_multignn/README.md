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
| `causal.patch` | On top of `project_split.patch`, the causal protocol under `QSH_CAUSAL=1` (Round 86, the owner's choices of 2026-11-15): each mini-batch is one batch of transactions in time order (2,048, split where it spans more than 17 minutes; `QSH_CAUSAL_CUT=batch` keeps the plain 2,048), sampled with IBM's own shared sampler, with every sampled edge later than that batch dropped, and every window scaled by the training window's statistics; with `QSH_PATIENCE=n` training stops after n epochs without a better validation F1, and `QSH_LOADER_WORKERS=n` samples in n processes; `QSH_CAUSAL_TIME=drop` or `lag` replaces the absolute time input (below; `absolute`, the default, keeps IBM's) |
| `export_for_browser.py` | Exports a checkpoint as one ONNX file at opset 18 (by `torch.export`; ONNX Runtime within $`10^{-4}`$ of PyTorch on the seeds) and writes the 16 held-out transactions the published finance MLP and compact GNN are checked on, with the neighbourhood the protocol samples for them, for the finance page to re-score. Under `QSH_CAUSAL_TIME=lag` a seed's time input is measured from the end of its batch, as the test loader measures it; the report compares the 16 with the recorded scoring and gives how far each window's transfers lie before their batch's end. Every node's metadata is dropped from the graph, since `torch.export` writes each node's stack trace, with this machine's file paths, into it (2026-11-18), and a graph that still holds the home directory is refused |
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

### The time input under the causal protocol (2026-11-16)

IBM's model reads the transaction's absolute `Timestamp` as an edge feature. IBM scales each window by its own
statistics, so every window's times span about the same range. Scaled by the fit window's statistics, as the causal
protocol scales them, they do not: the fit window's times run from $`-1.14`$ to $`2.0`$ standard deviations, the
validation window's from $`2.0`$ to $`2.9`$, and the test window's from $`2.9`$ to $`9.7`$ ($`5.0`$ to $`9.7`$ in its last
1,108 rows). No other edge column moves outside the fit window's range (amount, currency, payment format and both
ports; measured 2026-11-16). Three seeds trained that way kept validation F1 at $`0.62`$ and logged a test F1 of
0.4125, 0.4307 and 0.5544, against $`0.6487 \pm 0.0208`$ under IBM's protocol; they were stopped and are kept here as
the record of the defect, not used.

Split at the test window's sparse last 1,108 rows (2022-09-11 to 18, 655 of the 2,321 laundering transfers), IBM's
protocol scores 0.54 to 0.60 on the rest of the window and 0.81 to 0.86 there, where its neighbourhoods include the
transfers that complete a pattern; the three causal seeds scored 0.35 to 0.51 and 0.37 to 0.77.

`QSH_CAUSAL_TIME` (the owner's choice of 2026-11-16: five seeds of each, the export the better on validation) sets the
input where the loader cuts each batch, `_DropLater` in the patched `train_util.py`:
- `drop`: the time input is zero; time only orders and cuts the batches.
- `lag`: each edge's time before its batch ends, divided by the fit window's span and clipped at it, so it lies in
  $`[-1, 0]`$ in every window. Temporal graph networks encode time relative to the event in the same way (TGAT,
  arXiv:2002.07962; GraphMixer, arXiv:2302.11636; DyGFormer, arXiv:2303.13047), and a linear encoder does as well as a
  sinusoidal one (arXiv:2504.08129). A reverse edge, which carries no timestamp of its own, takes its forward edge's
  by its id.

#### The ten runs (2026-11-16 and 17)

Five seeds of each time input, alternating seed by seed, from the queue `run_queue_time_modes.sh` in the working
folder (2026-11-16 09:37 to 2026-11-17 06:21 UTC; kernel 5.14.0-687.66.2, driver 620.70.06). Each run is one line of
`logs/runs_20261116.jsonl` there: its checkpoint and sha256, the figures, the kernel and driver, and the seconds spent
training, in `precise_bn.py` and on the plain batches; the model's record in `models/metrics.json` carries all ten.
The exported way is eval mode with the statistics recomputed on the fit window, at the threshold that maximises
validation F1; the last column is the same checkpoint under the plain batches of 2,048 at that threshold.

| Time input | Seed | Kept epoch | Logged validation F1 | Logged test F1 | Test F1, exported way | Test AUPRC | Test F1, plain batches |
|---|---|---|---|---|---|---|---|
| `lag` | 1 | 16 | 0.6258 | 0.5474 | 0.6252 | 0.5789 | 0.4581 |
| `lag` | 2 | 28 | 0.6178 | 0.6025 | 0.5786 | 0.4984 | 0.4003 |
| `lag` | 3 | 34 | 0.6171 | 0.5737 | 0.5973 | 0.5230 | 0.4096 |
| `lag` | 4 | 33 | 0.6115 | 0.5718 | 0.6589 | 0.5907 | 0.4638 |
| `lag` | 5 | 14 | 0.6152 | 0.5188 | 0.6440 | 0.5844 | 0.4512 |
| `lag`, mean | | | $`0.6175 \pm 0.0053`$ | $`0.5628 \pm 0.0314`$ | $`0.6208 \pm 0.0330`$ | $`0.5551 \pm 0.0416`$ | $`0.4366 \pm 0.0294`$ |
| `drop` | 1 | 30 | 0.5989 | 0.5403 | 0.5255 | 0.5313 | 0.4874 |
| `drop` | 2 | 34 | 0.6109 | 0.5930 | 0.6156 | 0.5904 | 0.5469 |
| `drop` | 3 | 25 | 0.5842 | 0.5156 | 0.5517 | 0.5260 | 0.5088 |
| `drop` | 4 | 23 | 0.5920 | 0.4966 | 0.5469 | 0.5377 | 0.4627 |
| `drop` | 5 | 46 | 0.6025 | 0.4875 | 0.5665 | 0.5162 | 0.5009 |
| `drop`, mean | | | $`0.5977 \pm 0.0102`$ | $`0.5266 \pm 0.0423`$ | $`0.5613 \pm 0.0338`$ | $`0.5403 \pm 0.0291`$ | $`0.5013 \pm 0.0309`$ |
|---|---|---|---|---|---|---|---|---|

`lag` has the better mean logged validation F1, $`0.6175`$ against $`0.5977`$, and its seed 1 the best of its five,
$`0.6258`$, so that is the export (the owner's rule): test F1 0.6252 and AUPRC 0.5789 the exported way, at a
validation-tuned threshold of 0.4311. Over five seeds `lag` also scores higher under the protocol's batches and lower
under the plain batches of 2,048. Under the protocol's batches every scored transfer lies at most 1,020 s before its
batch's end, in every window. Under the plain batches 1,352 test transfers, 654 of them laundering (28% of the
window's 2,321), lie further back, up to 663,600 s (7.7 days), so their `lag` input falls below anything training
gave, down to the clip at $`-1`$ (`seconds_before_batch_end` in the export's report). The three runs on the absolute
timestamp scored 0.3995, 0.3546 and 0.5888 the exported way, and 0.3520, 0.2990 and 0.5207 under the plain batches.
A run took 1.3 to 2.6 hours of training and about 11 minutes in `precise_bn.py`, both batchings.

The export samples each of the 16 transfers alone (they lie in different batches) but measures its `lag` input back
from the last time among the test-window rows of its batch, as the test loader measures it. Its 16 probabilities lie
within 0.021 of those the recorded scoring gave the same transfers and decide the same at the threshold
(`against_the_recorded_scoring` in the report; under IBM's protocol, with the 16 drawn together, 0.097 and the same
16 decisions).

### IBM's reverse edges share their forward edges' ports (2026-11-16)

IBM's `create_hetero_obj` (`data_util.py` at `252b025`) assigns one tensor to both edge types' `edge_attr` and then
swaps the last two columns, the in- and out-port, "for the reverse edges", in place. The swap reaches the forward
edges through the shared tensor, so both directions carry the swapped pair: with ports $`(1, 2)`$ in, both read
$`(2, 1)`$ (reproduced on a two-edge graph). It is part of IBM's protocol as the paper's code runs it, and this
reproduction keeps it under both protocols.

## Multi-GIN+EU, five seeds under each protocol (2026-11-17 and 18)

Not exported or published. Five seeds under IBM's protocol, each IBM's 100 epochs, and five under the causal protocol
with each time input: `lag`, the one PNA's runs chose, from the same queue (2026-11-17 06:21 to 2026-11-18 05:26 UTC),
then `drop` on the owner's choice of 2026-11-18 (`run_queue_gin_drop.sh`, 06:29 to 11:41 UTC), with the same kernel
and driver. GIN's head dropout is 0.105, PNA's 0.288 (`model_settings.json`).

| Protocol | Seed | Kept epoch of run | Logged validation F1 | Logged test F1 | Test F1, eval mode with the fit-window statistics | Test AUPRC | Test F1, the statistics IBM's loop saved | Test F1, plain batches |
|---|---|---|---|---|---|---|---|---|
| IBM's | 1 | 57 of 100 | 0.6083 | 0.6251 | 0.5848 | 0.5684 | 0.6369 | - |
| IBM's | 2 | 14 of 100 | 0.5633 | 0.5633 | 0.5652 | 0.5808 | 0.5841 | - |
| IBM's | 3 | 46 of 100 | 0.5682 | 0.5886 | 0.5709 | 0.6035 | 0.5587 | - |
| IBM's | 4 | 91 of 100 | 0.6119 | 0.6094 | 0.5381 | 0.5609 | 0.6057 | - |
| IBM's | 5 | 67 of 100 | 0.6139 | 0.6228 | 0.6093 | 0.6094 | 0.5846 | - |
| IBM's, mean | | | $`0.5931 \pm 0.0251`$ | $`0.6018 \pm 0.0260`$ | $`0.5737 \pm 0.0262`$ | $`0.5846 \pm 0.0213`$ | $`0.5940 \pm 0.0292`$ | - |
| causal, `lag` | 1 | 6 of 26 | 0.3771 | 0.3285 | 0.3547 | 0.2916 | 0.3780 | 0.2267 |
| causal, `lag` | 2 | 50 of 70 | 0.5757 | 0.4298 | 0.4986 | 0.4867 | 0.2307 | 0.3343 |
| causal, `lag` | 3 | 16 of 36 | 0.4494 | 0.4260 | 0.4205 | 0.3443 | 0.4913 | 0.2913 |
| causal, `lag` | 4 | 25 of 45 | 0.5710 | 0.4788 | 0.5005 | 0.4553 | 0.4619 | 0.3424 |
| causal, `lag` | 5 | 11 of 31 | 0.4664 | 0.3843 | 0.5056 | 0.4232 | 0.1885 | 0.3433 |
| causal, `lag`, mean | | | $`0.4879 \pm 0.0849`$ | $`0.4095 \pm 0.0563`$ | $`0.4560 \pm 0.0667`$ | $`0.4002 \pm 0.0806`$ | $`0.3501 \pm 0.1356`$ | $`0.3076 \pm 0.0500`$ |
| causal, `drop` | 1 | 7 of 27 | 0.3420 | 0.3312 | 0.3462 | 0.2623 | 0.4062 | 0.3312 |
| causal, `drop` | 2 | 4 of 24 | 0.2866 | 0.2699 | 0.3363 | 0.2751 | 0.3673 | 0.2874 |
| causal, `drop` | 3 | 25 of 45 | 0.4316 | 0.3422 | 0.3941 | 0.2430 | 0.3267 | 0.3474 |
| causal, `drop` | 4 | 9 of 29 | 0.3029 | 0.2967 | 0.2802 | 0.2041 | 0.3183 | 0.2564 |
| causal, `drop` | 5 | 17 of 37 | 0.0099 | 0.0034 | 0.0576 | 0.0261 | 0.0560 | 0.0530 |
| causal, `drop`, mean | | | $`0.2746 \pm 0.1583`$ | $`0.2487 \pm 0.1401`$ | $`0.2829 \pm 0.1323`$ | $`0.2021 \pm 0.1020`$ | $`0.2949 \pm 0.1381`$ | $`0.2551 \pm 0.1185`$ |

Under IBM's protocol the logged test F1 is $`0.6018 \pm 0.0260`$, against the paper's $`64.79 \pm 1.22`$ on its own
split. With the statistics recomputed on the fit window the exported way gives $`0.5737 \pm 0.0262`$, against
$`0.5940 \pm 0.0292`$ with those IBM's loop saved; for PNA the two were $`0.6577`$ and $`0.6597`$. Under the causal
protocol GIN logs a validation F1 of $`0.4879 \pm 0.0849`$ with `lag`, against PNA's $`0.6175 \pm 0.0053`$, and seeds
1, 3 and 5 kept epochs 6, 16 and 11 and stopped 20 epochs later. With `drop` it logs $`0.2746 \pm 0.1583`$, below
`lag` as for PNA, and its seed 5 never rose above 0.01. A run took 3.3 hours under IBM's protocol and 0.65 to 1.7
hours of training under the causal one.
