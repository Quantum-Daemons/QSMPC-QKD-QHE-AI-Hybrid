"""Dump a saved Multi-GNN checkpoint's laundering probability for every edge of the train, validation and test windows (local, not IBM's).

Mirrors `training.train_gnn` and `train_util.evaluate_hetero` exactly (same config, transform, loaders and seed-edge masking), with
two changes: the train edges are scored in order, without shuffling, and each prediction is kept with its edge id rather than
reduced to an F1, so it can be aligned to the corpus rows. It also counts the seed edges the neighbour sampling never returned,
which `evaluate_hetero` drops for Small_HI: their share is the part of the window the published F1 does not score.

Run from Multi-GNN/ with the same flags as the training run, plus --inference_out DIR:
    QSH_SPLIT=project python dump_logits.py --data Small_HI --model pna --emlps --reverse_mp --ego --ports --testing --seed 1 --inference_out ../pna_logits
"""

import json
import logging
import sys
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

out_dir = None
if "--inference_out" in sys.argv:
    i = sys.argv.index("--inference_out")
    out_dir = Path(sys.argv[i + 1])
    del sys.argv[i:i + 2]
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
ckpt = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar', weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()
logging.info(f"loaded checkpoint from epoch {ckpt['epoch']}")

# The train edges in order: the training loader shuffles, and its seeds are every edge of the train graph.
et = ("node", "to", "node")
tr_ordered = LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, edge_label_index=(et, tr_data[et].edge_index),
                                edge_label=tr_data[et].y, batch_size=args.batch_size, shuffle=False, transform=transform)


@torch.no_grad()
def dump(loader, inds):
    ids_out, probs_out, labels_out = [], [], []
    missing_total = 0
    inds = inds.detach().cpu()
    for batch in loader:
        batch_edge_inds = inds[batch[et].input_id.detach().cpu()]
        batch_edge_ids = loader.data[et].edge_attr.detach().cpu()[batch_edge_inds, 0]
        edge_ids = batch[et].edge_attr[:, 0].detach().cpu()
        mask = torch.isin(edge_ids, batch_edge_ids)
        missing_total += int((~torch.isin(batch_edge_ids, edge_ids)).sum())
        batch[et].edge_attr = batch[et].edge_attr[:, 1:]
        batch["node", "rev_to", "node"].edge_attr = batch["node", "rev_to", "node"].edge_attr[:, 1:]
        batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[et][mask]
        ids_out.append(edge_ids[mask].long())
        probs_out.append(torch.softmax(out, -1)[:, 1].cpu())
        labels_out.append(batch[et].y[mask].cpu())
    ids, probs, labels = torch.cat(ids_out).numpy(), torch.cat(probs_out).numpy(), torch.cat(labels_out).numpy()
    return ids, probs, labels, missing_total


out_dir.mkdir(parents=True, exist_ok=True)
for name, loader, inds in (("train", tr_ordered, torch.arange(tr_data[et].edge_index.shape[1])), ("val", val_loader, val_inds), ("test", te_loader, te_inds)):
    ids, probs, labels, missing = dump(loader, inds)
    uniq = np.unique(ids)
    np.savez_compressed(out_dir / f"{name}.npz", ids=ids, probs=probs, labels=labels)
    logging.info(f"{name}: {len(ids):,} predictions over {len(uniq):,} distinct edges of {len(inds):,} seeds; {missing:,} seed edges never sampled")
