#!/usr/bin/env python3
"""IBM's Multi-PNA+EU as a graph a browser can run, with the 16 held-out transactions it is checked on (Round 86).

The owner's choice of 2026-11-15: IBM's official model (`IBM/Multi-GNN`, Apache-2.0, at 252b025 with this project's
split patch, `project_split.patch`, and for the causal protocol `causal.patch`) published on the Hub and re-scored on
the finance page, under two neighbourhood protocols:

- `ibm`: neighbours sampled from the whole test graph as IBM's loader draws them, so a transaction's neighbourhood
  includes later transactions; IBM's measured setting.
- `causal`: neighbours only from the scored transaction's batch or earlier ones, a batch being 2,048 rows split where it
  spans more than 17 minutes, so nothing read lies past the horizon of the page's own models (the end of the
  transaction's batch of 2,048, and at most 17 minutes after it; the owner's choice of 2026-11-15).

Run from IBM's checkout (the one patched for the protocol), in its environment, with the CPU only, on the checkpoint
`precise_bn.py` writes (BatchNorm statistics recomputed on the fit window, the owner's choice of 2026-11-16; a
checkpoint as IBM's loop saved it is refused) and that script's validation probabilities:

    QSH_SPLIT=project [QSH_CAUSAL=1] python <this repository>/scripts/train/ibm_multignn/export_for_browser.py \\
        --checkpoint <precise_bn.py's checkpoint_fit_bn.tar> --val-probs <precise_bn.py's val.npz> --protocol ibm|causal \\
        --fixtures <this repository>/models/finance_aml_mlp_large/weights.json --out <artifact dir>

What it writes, after checking each against the PyTorch model:
- `onnx/model.onnx`: the hetero model, exported by `torch.export` at opset 18 and saved as one file (a browser cannot
  mount ONNX external data), its inputs the node features, both edge directions' indices and features;
- `eval_sample.json`: the 16 transactions the published MLP and compact GNN are checked on, their sampled
  neighbourhood (one batch, seeded), the seed edges' positions in it, their labels and the logits PyTorch gives, and
  for the page's view each edge's row and time, the reach, and the counts of `neighbourhood.py` (a causal sample that
  reads a transfer past its seed's horizon is refused);
- `export_report.json`: ONNX Runtime against PyTorch on that neighbourhood, the decisions at the threshold tuned on
  the validation window, and the subgraph's size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SAMPLE_SEED = 2026
TOL = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-probs", required=True,
                    help="precise_bn.py's val.npz for this checkpoint: probabilities with the recomputed statistics")
    ap.add_argument("--protocol", choices=("ibm", "causal"), required=True)
    ap.add_argument("--fixtures", required=True, help="the published finance MLP's weights.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="pna", choices=("pna", "gin"))
    a = ap.parse_args()
    if os.environ.get("QSH_SPLIT") != "project":
        raise SystemExit("set QSH_SPLIT=project: the fixtures are positions in this project's test window")
    if (a.protocol == "causal") != (os.environ.get("QSH_CAUSAL") == "1"):
        raise SystemExit("QSH_CAUSAL=1 goes with --protocol causal, and only with it")

    import numpy as np
    import torch
    sys.path.insert(0, os.getcwd())
    sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "src"))  # qsh.models.sample_per_row
    sys.argv = [sys.argv[0], "--data", "Small_HI", "--model", a.model, "--emlps", "--reverse_mp", "--ego", "--ports", "--testing"]
    from data_loading import get_data
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.nn import to_hetero
    from train_util import AddEgoIds, add_arange_ids, get_loaders
    from training import get_model
    from util import create_parser

    args = create_parser().parse_args()
    data_config = json.load(open("data_config.json"))
    settings = json.load(open("model_settings.json"))[a.model]["params"]
    config = SimpleNamespace(**settings)
    tr, va, te, tr_i, va_i, te_i = get_data(args, data_config)
    add_arange_ids([tr, va, te])
    transform = AddEgoIds()
    tr_loader, _, _ = get_loaders(tr, va, te, tr_i, va_i, te_i, transform, args)
    model = to_hetero(get_model(next(iter(tr_loader)), config, args), te.metadata(), aggr="mean")
    ckpt = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    # The owner's choice of 2026-11-16: IBM's loop never calls model.eval(), so a checkpoint it saves carries BatchNorm
    # statistics last set by the test pass; only one whose statistics precise_bn.py recomputed on the fit window ships.
    if not str(ckpt.get("bn_statistics", "")).startswith("fit window"):
        raise SystemExit(f"{a.checkpoint}: its BatchNorm statistics are the ones IBM's loop saved, last set by the test "
                         "pass; recompute them on the fit window with precise_bn.py and export its checkpoint_fit_bn.tar")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    et, rt = ("node", "to", "node"), ("node", "rev_to", "node")
    fx = json.loads(Path(a.fixtures).read_text())["fixtures"]
    rows = torch.tensor(fx["row_index_in_test_window"], dtype=torch.long)
    pos = te_i[0] + rows
    labels = te[et].y[pos]
    if labels.tolist() != fx["labels"]:
        raise SystemExit("the fixtures' labels differ from IBM's data at those positions: the windows do not line up")

    if a.protocol == "ibm":
        # IBM's loader: the 16 seeds in one batch, neighbours from the whole test graph, shared between them.
        torch.manual_seed(SAMPLE_SEED)
        loader = LinkNeighborLoader(te, num_neighbors=args.num_neighs, edge_label_index=(et, te[et].edge_index[:, pos]),
                                    edge_label=labels, batch_size=len(pos), shuffle=False, transform=transform)
        parts = [(pos.tolist(), next(iter(loader)))]
    else:
        # The causal protocol cuts each seed at its own batch of 2,048, and the 16 lie in different batches, so each is
        # sampled alone, with the later edges dropped, and the subgraphs are joined without edges between them.
        from train_util import _DropLater
        parts = []
        for i, p in enumerate(pos.tolist()):
            torch.manual_seed(SAMPLE_SEED + i)
            cut = te[et].edge_time[torch.tensor([p])]
            loader = LinkNeighborLoader(te, num_neighbors=args.num_neighs, edge_label_index=(et, te[et].edge_index[:, [p]]),
                                        edge_label=labels[i:i + 1], batch_size=1, shuffle=False,
                                        transform=_DropLater(cut, transform))
            parts.append(([p], next(iter(loader))))
    xs, eis, eas, ris, ras, seed_at, poss, secs = [], [], [], [], [], [], [], []
    off_n = off_e = 0
    for seeds_here, batch in parts:
        ids = batch[et].edge_attr[:, 0].long()
        for p in seeds_here:
            hits = (ids == p).nonzero().flatten()
            if len(hits) != 1:
                raise SystemExit(f"position {p} appears {len(hits)} times in its batch where it should appear once")
            seed_at.append(off_e + int(hits[0]))
        poss.append(ids)
        secs.append(batch[et].timestamps.double())
        xs.append(batch["node"].x.float())
        eis.append(batch[et].edge_index + off_n)
        ris.append(batch[rt].edge_index + off_n)
        eas.append(batch[et].edge_attr[:, 1:].float())
        ras.append(batch[rt].edge_attr[:, 1:].float())
        off_n += batch["node"].x.shape[0]
        off_e += batch[et].edge_index.shape[1]
    x, ei, ri = torch.cat(xs), torch.cat(eis, 1), torch.cat(ris, 1)
    ea, ra = torch.cat(eas), torch.cat(ras)
    # The IBM tab's view (the owner's choice of 2026-11-15): each forward edge's row in the time-ordered file and its
    # time, and per seed what the model read, counted by neighbourhood.py; the page re-derives the counts.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from neighbourhood import BATCH_ROWS, fit_reach, seed_fields
    reach = fit_reach(te[et].timestamps.double().numpy(), tr[et].edge_index.shape[1])
    if a.protocol == "causal":
        from data_loading import causal_batches
        if causal_batches(te[et].timestamps.double().numpy(), tr[et].edge_index.shape[1])[1] != reach:
            raise SystemExit("the causal patch's reach differs from neighbourhood.fit_reach")
    view_pos, view_sec = torch.cat(poss).numpy(), torch.cat(secs).numpy()
    fields = seed_fields(ei.numpy(), view_pos, view_sec, seed_at, reach)
    if a.protocol == "causal" and any(f["past_horizon"] for f in fields):
        raise SystemExit("a causal neighbourhood reads a transfer past its seed's horizon")
    whole = bool(np.all(view_sec == np.round(view_sec)))

    class Flat(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, edge_index, edge_attr, rev_edge_index, rev_edge_attr):
            return self.m({"node": x}, {et: edge_index, rt: rev_edge_index}, {et: edge_attr, rt: rev_edge_attr})[et]

    flat = Flat(model).eval()
    with torch.no_grad():
        logits = flat(x, ei, ea, ri, ra)
    seed_logits = logits[seed_at].numpy().astype(np.float64)

    out = Path(a.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    graph_path = out / "onnx" / "model.onnx"
    t0 = time.time()
    import tempfile

    import onnx
    import onnxruntime as ort
    # torch.export writes the weights beside the graph; a browser cannot mount that file, so the graph is re-saved whole.
    with tempfile.TemporaryDirectory(prefix="qsh-ibm-export-") as tmpdir:
        tmp = Path(tmpdir) / "model.onnx"
        torch.onnx.export(flat, (x, ei, ea, ri, ra), str(tmp), opset_version=18, dynamo=True,
                          input_names=["x", "edge_index", "edge_attr", "rev_edge_index", "rev_edge_attr"],
                          output_names=["logits"],
                          # The two directions are sampled separately, so their counts differ (7,212 and 7,214 here).
                          dynamic_axes={"x": {0: "N"}, "edge_index": {1: "E"}, "edge_attr": {0: "E"},
                                        "rev_edge_index": {1: "Er"}, "rev_edge_attr": {0: "Er"}, "logits": {0: "E"}})
        onnx.save_model(onnx.load(str(tmp), load_external_data=True), str(graph_path), save_as_external_data=False)
    onnx.checker.check_model(str(graph_path))
    s = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    o = s.run(None, {"x": x.numpy(), "edge_index": ei.numpy(), "edge_attr": ea.numpy(),
                     "rev_edge_index": ri.numpy(), "rev_edge_attr": ra.numpy()})[0]
    diff = float(np.abs(o[seed_at] - seed_logits).max())
    if diff > TOL:
        raise SystemExit(f"ONNX Runtime misses PyTorch on the seeds by {diff:.3g} (> {TOL})")

    val = np.load(a.val_probs)
    if "probs_saved_bn" not in val.files:
        raise SystemExit(f"{a.val_probs}: not precise_bn.py's output, so its probabilities may use the saved statistics")
    order = np.argsort(-val["probs"])
    tp = np.cumsum(val["labels"][order] == 1)
    fp = np.cumsum(val["labels"][order] == 0)
    f1 = 2 * tp / (tp + fp + tp[-1])
    k = int(np.argmax(f1))
    thr = float(val["probs"][order][k])
    # The record the page compares with, from ONNX Runtime's own outputs by the page's rule (qsh.models.sample_per_row):
    # a shifted softmax of the logits, laundering at or above the threshold, the margin p(laundering) - threshold.
    ort_logits = np.asarray(o[seed_at], dtype=np.float64)
    e = np.exp(ort_logits - ort_logits.max(axis=1, keepdims=True))
    prob = e[:, 1] / e.sum(axis=1)
    pred = (prob - thr >= 0).astype(int)
    from qsh.models.sample_per_row import TOLERANCE
    sample = {
        "_about": ("IBM's Multi-PNA+EU on the 16 held-out transactions the published finance MLP and compact GNN are checked "
                   "on, with the neighbourhood its loader sampled for them, written by "
                   "scripts/train/ibm_multignn/export_for_browser.py."),
        "kind": "graph", "protocol": a.protocol,
        "protocol_note": ("Neighbours sampled from the whole test graph, as IBM's loader draws them: a neighbourhood includes "
                          "transactions later than the one scored." if a.protocol == "ibm" else
                          "Neighbours only from the scored transaction's batch or earlier ones, in time order, a batch being "
                          f"2,048 rows split where it spans more than {reach // 60} minutes: nothing read lies past the "
                          "horizon this page's own models give a transaction, the end of its batch of 2,048 and at most "
                          f"{reach // 60} minutes after it."),
        "inputs": {"x": x.numpy().ravel().tolist(), "x_shape": list(x.shape),
                   "edge_index": ei.numpy().ravel().tolist(), "edge_attr": ea.numpy().ravel().tolist(), "edge_attr_dim": int(ea.shape[1]),
                   "rev_edge_index": ri.numpy().ravel().tolist(), "rev_edge_attr": ra.numpy().ravel().tolist(), "n_edges": int(ei.shape[1]),
                   "n_rev_edges": int(ri.shape[1])},
        "seeds": {"positions_in_batch": seed_at, "row_index_in_test_window": rows.tolist(), "labels": labels.tolist()},
        "view": {"about": ("Each forward edge's row in the time-ordered file (its batch of 2,048 is the row divided by "
                           "2,048) and its time in seconds from the corpus's first transfer, the reach (the furthest a "
                           "fit-window row sees to the end of its batch), and per seed what the model read, as "
                           "scripts/train/ibm_multignn/neighbourhood.py defines it."),
                 "batch_rows": BATCH_ROWS, "reach_s": reach, "edge_position": view_pos.tolist(),
                 "edge_seconds": view_sec.astype(np.int64).tolist() if whole else view_sec.tolist(),
                 "per_seed": fields},
        "expected": {"logits": ort_logits.tolist(), "logits_pytorch": seed_logits.tolist(), "decision_threshold": thr,
                     "decision_threshold_note": ("the laundering probability at which F1 peaks on the validation window, from "
                                                 "precise_bn.py's probabilities (BatchNorm statistics recomputed on the fit window)"),
                     "graph": "model.onnx", "accuracy": float((pred == labels.numpy()).mean()),
                     "note": "Accuracy on 16 rows, half of them laundering, at the validation-tuned threshold: it shows the model running "
                             "on the record's inputs, not how well it ranks.",
                     "per_row": {"rows": len(seed_at), "read_from": "logits", "tolerance": TOLERANCE, "graph": "model.onnx",
                                 "predicted": pred.tolist(), "margin": [float(v) for v in prob - thr],
                                 "margin_is": "p(class 1) minus the decision threshold; class 1 at or above 0",
                                 "note": "From ONNX Runtime's outputs on this file's inputs, by the page's rule."}},
        "classes": {"0": "clean", "1": "laundering"},
        "licence": "CDLA-Sharing-1.0", "licence_url": "https://cdla.dev/sharing-1-0/",
        "attribution": ("IBM, Transactions for Anti Money Laundering (AML), HI-Small (Altman et al., NeurIPS 2023 Datasets "
                        "and Benchmarks); model: IBM/Multi-GNN (Egressy et al., AAAI 2024), Apache-2.0"),
        "sample_seed": SAMPLE_SEED,
    }
    (out / "eval_sample.json").write_text(json.dumps(sample, separators=(",", ":")) + "\n")
    report = {
        "protocol": a.protocol, "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "bn_statistics": ckpt["bn_statistics"], "source_checkpoint_sha256": ckpt.get("source_checkpoint_sha256"),
        "source_num_batches_tracked": ckpt.get("saved_num_batches_tracked"),
        "graph_bytes": graph_path.stat().st_size, "opset": 18,
        "subgraph": {"nodes": int(x.shape[0]), "edges": int(ei.shape[1]), "reverse_edges": int(ri.shape[1]), "parts": len(parts)},
        "max_abs_logit_difference_onnx_vs_torch_on_seeds": diff,
        "decision_threshold": thr, "seed_accuracy_at_threshold": sample["expected"]["accuracy"],
        "export_seconds": round(time.time() - t0, 1),
    }
    (out / "export_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
