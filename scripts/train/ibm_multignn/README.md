# IBM's Multi-GNN on this project's windows (Round 86)

The finance page compares its compact GNN with IBM's official Multi-GIN+EU and Multi-PNA+EU, run on the same temporal windows, and
the compact GNN is distilled from that PNA. These files reproduce both from IBM's code (`IBM/Multi-GNN`, Apache-2.0) without
vendoring it.

| File | What it does |
|---|---|
| `format_amlworld.py` | Writes IBM's `formatted_transactions.csv` from this repository's AMLworld loader, with the official encodings and the rows in `load_sorted()` order |
| `project_split.patch` | Two changes to IBM's code at commit `252b0252afca109d1d216c411c59ff70753b25fc`: with `QSH_SPLIT=project` the fit, validation and test windows are this project's 56 / 14 / 30 percent by position, and `AddEgoIds` defines `forward`, which PyTorch Geometric 2.8 requires of a transform |
| `dump_logits.py` | Loads a saved checkpoint and writes the laundering probability of every edge of the three windows, keyed by edge id, and counts the seed edges the neighbour sampling never returned |

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
