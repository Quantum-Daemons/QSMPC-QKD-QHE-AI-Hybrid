"""A saved Multi-GNN checkpoint with its BatchNorm statistics recomputed on the fit window (local, not IBM's; Round 86).

IBM's training loop never calls `model.eval()`, so every validation and test pass runs in training mode and updates
BatchNorm's running statistics. The checkpoint it saves after the test pass therefore carries statistics set by the
test window's last batches: its `num_batches_tracked` is the epochs times the fit, validation and test batches of an
epoch. An eval-mode model built from it, as `dump_logits.py`, the browser export and the finance page all build,
normalises with test-window statistics. The owner's choice of 2026-11-16: every exported checkpoint, under both
protocols, has them recomputed on the fit window.

This resets the statistics and recomputes them as a plain average over one pass of the protocol's fit-window loader in
training mode, without gradients (precise BatchNorm, Wu and Johnson, arXiv:2105.07576; the population statistics of
Ioffe and Szegedy, arXiv:1502.03167, are the training set's). It then scores the validation and test windows, and with
`--with-train` the fit window in order, twice on the same sampled batches: with the saved statistics and with the
recomputed ones, so the two differ only in the statistics.

Run from IBM's checkout with the training run's flags (for the causal protocol, from the causal checkout with
QSH_CAUSAL=1), plus --checkpoint and --inference_out:
    QSH_SPLIT=project python precise_bn.py --data Small_HI --model pna --emlps --reverse_mp --ego --ports --testing \
        --seed 4 --checkpoint <checkpoint .tar> --inference_out <dir> [--with-train]

It writes `checkpoint_fit_bn.tar` (the weights unchanged, the statistics recomputed, with their provenance), `val.npz`
and `test.npz` (and `train.npz`) keyed `ids`, `probs` (the recomputed statistics: the exported model's), `probs_saved_bn`
and `labels`, and `bn_report.json`. `export_for_browser.py` takes this checkpoint and this `val.npz`.
"""

import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from data_loading import get_data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
from train_util import AddEgoIds, add_arange_ids, extract_param, get_loaders
from training import get_model
from util import create_parser, logger_setup, set_seed

try:  # the causal protocol's loaders and seed rule (causal.patch), as dump_logits.py takes them
    from train_util import _causal, _workers, seed_mask
except ImportError:  # IBM's code with the split patch only
    def _causal(*_args, **_kwargs):
        return None

    def _workers():
        return {}

    seed_mask = None


def take(flag: str, has_value: bool = True):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    value = sys.argv[i + 1] if has_value else True
    del sys.argv[i:i + (2 if has_value else 1)]
    return value


out_dir = Path(take("--inference_out"))
ckpt_file = take("--checkpoint")
with_train = bool(take("--with-train", has_value=False))
args = create_parser().parse_args()
data_config = json.loads(Path("data_config.json").read_text())
logger_setup()
set_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
wandb.init(mode="disabled", config={
    "epochs": args.n_epochs, "batch_size": args.batch_size, "model": args.model, "data": args.data, "num_neighbors": args.num_neighs,
    "lr": extract_param("lr", args), "n_hidden": extract_param("n_hidden", args), "n_gnn_layers": extract_param("n_gnn_layers", args),
    "loss": "ce", "w_ce1": extract_param("w_ce1", args), "w_ce2": extract_param("w_ce2", args), "dropout": extract_param("dropout", args),
    "final_dropout": extract_param("final_dropout", args), "n_heads": None})
config = wandb.config
transform = AddEgoIds() if args.ego else None
add_arange_ids([tr_data, val_data, te_data])
tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)
model = get_model(next(iter(tr_loader)), config, args)
if args.reverse_mp:
    model = to_hetero(model, te_data.metadata(), aggr="mean")
ckpt_file = ckpt_file or f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar'
ckpt = torch.load(ckpt_file, weights_only=False)
if ckpt.get("bn_statistics"):
    raise SystemExit(f"{ckpt_file}: its BatchNorm statistics were already recomputed ({ckpt['bn_statistics']})")
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
logging.info(f"loaded {ckpt_file}, epoch {ckpt['epoch']}")

et = ("node", "to", "node")
rt = ("node", "rev_to", "node")


def batch_norms(m):
    return [x for x in m.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]


saved_tracked = sorted({int(b.num_batches_tracked) for b in batch_norms(model)})
logging.info(f"saved BatchNorm layers {len(batch_norms(model))}, num_batches_tracked {saved_tracked}")

recomputed = copy.deepcopy(model)
for b in batch_norms(recomputed):
    b.reset_running_stats()
    b.momentum = None                     # a plain cumulative average over the pass
recomputed.train()
t0 = time.time()
fit_batches = 0
with torch.no_grad():
    for batch in tr_loader:
        batch[et].edge_attr = batch[et].edge_attr[:, 1:]
        batch[rt].edge_attr = batch[rt].edge_attr[:, 1:]
        batch.to(device)
        recomputed(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        fit_batches += 1
logging.info(f"recomputed BatchNorm statistics over {fit_batches} fit-window batches in {time.time() - t0:.0f} s")
model.eval()
recomputed.eval()

layers = {}
for (name, a), (_, b) in zip(model.named_modules(), recomputed.named_modules(), strict=True):
    if isinstance(a, torch.nn.modules.batchnorm._BatchNorm):
        ratio = (a.running_var / b.running_var).cpu()
        layers[name] = {"max_abs_mean_shift": (a.running_mean - b.running_mean).abs().max().item(),
                        "saved_over_fit_variance_min": ratio.min().item(), "saved_over_fit_variance_max": ratio.max().item()}
        logging.info(f"{name}: {layers[name]}")


@torch.no_grad()
def dump(loader, inds):
    ids_out, p_fit, p_saved, labels_out = [], [], [], []
    inds = inds.detach().cpu()
    for batch in loader:
        batch_edge_inds = inds[batch[et].input_id.detach().cpu()]
        batch_edge_ids = loader.data[et].edge_attr.detach().cpu()[batch_edge_inds, 0]
        edge_ids = batch[et].edge_attr[:, 0].detach().cpu()
        mask = seed_mask(batch, batch_edge_ids) if seed_mask else torch.isin(edge_ids, batch_edge_ids)
        batch[et].edge_attr = batch[et].edge_attr[:, 1:]
        batch[rt].edge_attr = batch[rt].edge_attr[:, 1:]
        batch.to(device)
        a = recomputed(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[et][mask]
        b = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[et][mask]
        ids_out.append(edge_ids[mask].long())
        p_fit.append(torch.softmax(a, -1)[:, 1].cpu())
        p_saved.append(torch.softmax(b, -1)[:, 1].cpu())
        labels_out.append(batch[et].y[mask].cpu())
    return (torch.cat(ids_out).numpy(), torch.cat(p_fit).numpy(), torch.cat(p_saved).numpy(), torch.cat(labels_out).numpy())


out_dir.mkdir(parents=True, exist_ok=True)
windows = [("val", val_loader, val_inds), ("test", te_loader, te_inds)]
if with_train:
    # The fit window in order, as dump_logits.py scores it (the training loader shuffles); under the causal protocol its
    # time batches in order, later edges dropped.
    tr_ordered = LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, edge_label_index=(et, tr_data[et].edge_index),
                                    edge_label=tr_data[et].y,
                                    **(_causal(tr_data, None, transform, False)
                                       or dict(batch_size=args.batch_size, shuffle=False, transform=transform)), **_workers())
    windows.insert(0, ("train", tr_ordered, torch.arange(tr_data[et].edge_index.shape[1])))
summary = {}
for name, loader, inds in windows:
    ids, pf, ps, labels = dump(loader, inds)
    np.savez_compressed(out_dir / f"{name}.npz", ids=ids, probs=pf, probs_saved_bn=ps, labels=labels)
    flips = int(((pf > 0.5) != (ps > 0.5)).sum())
    summary[name] = {"predictions": int(len(ids)), "max_abs_probability_change": float(np.abs(pf - ps).max()),
                     "decisions_at_0.5_changed": flips}
    logging.info(f"{name}: {len(ids):,} predictions; max |p_fit - p_saved| {np.abs(pf - ps).max():.4g}; {flips:,} decisions at 0.5 change")

source_sha = hashlib.sha256(Path(ckpt_file).read_bytes()).hexdigest()
report = {"source_checkpoint": Path(ckpt_file).name, "source_checkpoint_sha256": source_sha, "epoch": ckpt["epoch"],
          "saved_num_batches_tracked": saved_tracked, "fit_batches": fit_batches, "layers": layers, "windows": summary}
(out_dir / "bn_report.json").write_text(json.dumps(report, indent=1))
torch.save({"epoch": ckpt["epoch"], "model_state_dict": recomputed.state_dict(),
            "bn_statistics": f"fit window: a plain average over {fit_batches} fit-window batches",
            "source_checkpoint_sha256": source_sha, "saved_num_batches_tracked": saved_tracked},
           out_dir / "checkpoint_fit_bn.tar")
logging.info(f"wrote {out_dir / 'checkpoint_fit_bn.tar'}")
