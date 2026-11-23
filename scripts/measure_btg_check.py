#!/usr/bin/env python3
"""BitcoinTemporalGraph's entity types: does a GNN over a node's neighbours beat the node's own features?

BitcoinTemporalGraph (Schnoering and Vazirgiannis; Figshare article 26305093 version 3, DOI
10.6084/m9.figshare.26305093.v3, CC BY 4.0) is a graph of value transfers between Bitcoin entities: a node is a cluster
of addresses, an edge the transfers from one node to another, in a PostgreSQL directory-format dump. Its paper
(Schnoering and Vazirgiannis, Bitcoin research with a transaction graph dataset, Scientific Data 12, 404 (2025),
doi:10.1038/s41597-025-04684-8) predicts a node's entity type, seven of them, from the node features: a gradient boosting
classifier reaches macro-F1 0.57 and four GNNs 0.56 to 0.64 on a random 40/30/30 split of the labelled nodes (its Table 6).

On the owner's approval of 2026-11-23 (a GPU job; references/aml-real-data-2026-11.md) this asks the same question on the
same labels, with four models on the same splits:

- the paper's baseline rerun: scikit-learn's GradientBoostingClassifier with default settings, fitted on the training and
  validation nodes together, on the paper's pre-processing, as the authors' published code does;
- a features-only booster: HistGradientBoostingClassifier, class-balanced, grown to 1,000 iterations, of which the one
  with the best validation macro-F1 is kept;
- the GNN's own network with the neighbours taken out (a two-layer perceptron), trained exactly as the GNN is, so that the
  difference between the two is what the neighbours add;
- the GNN: two GraphSAGE layers (mean aggregation and a root weight; Hamilton, Ying and Leskovec, NeurIPS 2017) in plain
  PyTorch, on each labelled node's one-hop neighbourhood, the paper's 10 neighbours per draw and 12 draws, with the test
  probabilities averaged over the draws as the paper does.

The inputs are the paper's 27 node features that need no outside data; its eight USD conversions need a BTC/USD price
series the dump does not carry. The networks follow the paper's training: 256 hidden units, ReLU, dropout 0.1, Adam at
1e-4, batches of 32, every class resampled to between 300 and 1,500 training rows and the cross-entropy weighted
inversely to the class counts, for the paper's 2,000 epochs. The epoch kept is the one with the best validation macro-F1,
the figure reported. The authors' code keeps the lowest validation loss instead; on seed 0's validation curves that loss
bottomed out hundreds of epochs before the macro-F1 stopped rising, and the GNN gained more after it than the network
without neighbours, so stopping there would understate the GNN most. Every run records both epochs and the validation
macro-F1 at each. The split is stratified 40/30/30, over five seeds; macro-F1 (the mean of the seven
per-class F1) and the per-class F1 are recorded for every seed, with their mean, standard deviation, minimum and maximum.
The paper's split is random rather than by time, so this one is too; the nodes of one entity can fall on both sides, and
the dump carries no map from addresses to nodes that would let a split group them.

Both tables are streamed out of the published archive `dataset.tar.gz`, whose SHA-256 is checked against the download
manifest's as it is read: the extracted copy of the edge table read too slowly through the NTFS driver for two passes
(4 to 32 MB/s on 2026-11-23, against the archive's page-cache speed). The archive's `toc.dat` is read first and must map
each table to the member and the columns read here. Nothing is held whole: the edge table (46.6 GB of text) is reduced
on the fly to the edges with a labelled endpoint, and the node table (27.0 GB) to the labelled nodes and the sampled
neighbours. Two passes over the archive, cached in --work (outside the repository); --fresh ignores the cache. The record
holds aggregates only: no address, entity name or node alias.

    PYTHONPATH=src python scripts/measure_btg_check.py --archive <dir>/dataset.tar.gz \
        --addresses <dir>/addresses.csv --work <dir>/work --fresh -o docs/measurements/btg-check-<date>.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import queue
import re
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: dataset.tar.gz and addresses.csv as Figshare serves them (files 48121651 and 52159721); Figshare's MD5s matched both.
ARCHIVE_SHA256 = "6fe7af7dee0e9178145d8bdcfcc09526e44e3be2368b151c2e923fc3f5ac9f99"
ARCHIVE_BYTES = 17_387_128_375
ADDRESSES_SHA256 = "a451de6d9974594b0794284769d76c64ad92224ca010485d9265c9102feda283"
TOC, EDGES, NODES = "dataset/toc.dat", "dataset/3617.dat", "dataset/3618.dat"
NODE_COLUMNS = ("alias", "degree", "degree_in", "degree_out", "total_transactions_in", "total_transactions_out",
                "min_sent", "max_sent", "total_sent", "min_received", "max_received", "total_received", "cluster_size",
                "first_transaction_in", "last_transaction_in", "first_transaction_out", "last_transaction_out",
                "cluster_num_edges", "cluster_num_cc", "cluster_num_nodes_in_cc", "label")
NUMERIC = NODE_COLUMNS[1:-1]
EDGE_COLUMNS = ("a", "b", "reveal", "last_seen", "total", "min_sent", "max_sent", "total_sent")
#: The paper's seven entity types, in its Table 6 order.
CLASSES = ("EXCHANGE", "MINING", "GAMBLING", "PONZI", "INDIVIDUAL", "RANSOMWARE", "BET")
FANOUT, DRAWS, SAMPLER_SEED = 10, 12, 20261123
HIDDEN, DROPOUT, LR, BATCH, MIN_ROWS, MAX_ROWS = 256, 0.1, 1e-4, 32, 300, 1500
MAX_EPOCHS, HGB_ITER = 2000, 1000
STAGE_VERSION = 1

COUNT, VALUE, SHARE = "count", "value", "share"
#: The paper's features without its USD conversions (section "Additional features" and its authors' code), with the
#: kind that sets the pre-processing: counts are scaled from their minimum, amounts from their 5th percentile.
FEATURES = (("degree", COUNT), ("degree_in", COUNT), ("degree_out", COUNT), ("total_transactions_in", COUNT),
            ("total_transactions_out", COUNT), ("min_sent", VALUE), ("total_sent", VALUE), ("max_sent", VALUE),
            ("min_received", VALUE), ("total_received", VALUE), ("max_received", VALUE), ("cluster_size", COUNT),
            ("cluster_num_edges", COUNT), ("cluster_num_nodes_in_cc", COUNT), ("avg_sent", VALUE),
            ("avg_received", VALUE), ("age", COUNT), ("degree_age_ratio", COUNT), ("degree_in_age_ratio", COUNT),
            ("degree_out_age_ratio", COUNT), ("transactions_in_age_ratio", COUNT), ("transactions_out_age_ratio", COUNT),
            ("cluster_size_age_ratio", COUNT), ("cluster_num_edges_age_ratio", COUNT), ("proportion_nodes_in_cc", SHARE),
            ("time_before_first_transaction", COUNT), ("degree_out_in_ratio", COUNT))

PAPER_URL = "https://www.nature.com/articles/s41597-025-04684-8"
PAPER = {
    "citation": "Schnoering and Vazirgiannis, Bitcoin research with a transaction graph dataset, Scientific Data 12, 404 "
                "(2025), doi:10.1038/s41597-025-04684-8 (preprint arXiv:2411.10325)",
    "url": PAPER_URL, "table_url": PAPER_URL + "/tables/6",
    "table": "Table 6, performance comparison of the trained models on test split",
    "split": "the labelled nodes divided at random into 40% training, 30% validation and 30% test",
    "graph": "undirected; GNNs see a two-hop sample, 10 then 5 neighbours, 12 samples per node",
    "nodes": 252_219_007, "edges": 785_954_737, "labelled_nodes": 34_098,
    "macro_f1": {"GCN": 0.56, "GraphSage": 0.62, "GAT": 0.64, "GIN": 0.63, "GBC": 0.57},
    "per_class_f1": {"GCN": [0.43, 0.45, 0.56, 0.46, 0.88, 0.19, 0.98], "GraphSage": [0.51, 0.57, 0.58, 0.55, 0.93, 0.25, 0.98],
                     "GAT": [0.49, 0.57, 0.62, 0.57, 0.93, 0.32, 0.98], "GIN": [0.54, 0.54, 0.62, 0.60, 0.94, 0.21, 0.98],
                     "GBC": [0.54, 0.45, 0.57, 0.30, 0.94, 0.50, 0.89]},
    "addresses_by_category": {"INDIVIDUAL": 63_189, "BET": 16_366, "RANSOMWARE": 12_539, "GAMBLING": 4_718,
                              "EXCHANGE": 2_738, "MINING": 1_608, "PONZI": 1_012, "MARKETPLACE": 406, "FAUCET": 401,
                              "BRIDGE": 370, "MIXER": 234},
}
MODELS = ("gbc_paper_setting", "hgb_features", "mlp_features", "gnn_one_hop")


# ----------------------------------------------------------------------------------------------- reading the archive


class _Hashing:
    """A read-only file whose every byte passes through a SHA-256."""

    def __init__(self, f):
        self.f, self.sha, self.n = f, hashlib.sha256(), 0

    def read(self, size=-1):
        b = self.f.read(size)
        self.sha.update(b)
        self.n += len(b)
        return b


def _produce(archive: Path, wanted: set[str], out: queue.Queue, block: int) -> None:
    """Decompress the archive front to back; put (member, chunk of whole lines) for the wanted members, then the digest.

    A member's chunk is None at its end. The COPY end marker (a line holding a backslash and a dot) must close every
    member and is removed; a member that does not end at a line end is refused.
    """
    try:
        with open(archive, "rb") as raw:
            src = _Hashing(raw)
            with tarfile.open(fileobj=src, mode="r|gz", bufsize=4 << 20) as tar:
                for member in tar:
                    if member.name == TOC:
                        out.put((TOC, tar.extractfile(member).read()))
                        continue
                    if member.name not in wanted:
                        continue
                    f = tar.extractfile(member)
                    carry, pending, seen = b"", b"", 0
                    while piece := f.read(block):
                        seen += len(piece)
                        piece = carry + piece
                        cut = piece.rfind(b"\n") + 1
                        carry, piece = piece[cut:], piece[:cut]
                        if len(piece) < (1 << 20):      # a short tail joins the chunk before it, end marker and all
                            pending += piece
                            continue
                        if pending:
                            out.put((member.name, pending))
                        pending = piece
                    body = pending.rstrip(b"\n")
                    if carry or seen != member.size or not (body == b"\\." or body.endswith(b"\n\\.")):
                        raise ValueError(f"{member.name}: does not end with the COPY end marker at a line end")
                    if body[:-2]:
                        out.put((member.name, body[:-2]))
                    out.put((member.name, None))
            while src.read(block):
                pass
        out.put(("", (src.sha.hexdigest(), src.n)))
    except BaseException as e:  # noqa: BLE001 - handed to the reading thread, which raises it
        out.put(("!", e))


def _stream(archive: Path, wanted: set[str], block: int = 64 << 20):
    """Yield (member, chunk) as _produce puts them, decompressing ahead in a thread; check the digest at the end."""
    q: queue.Queue = queue.Queue(maxsize=3)
    t = threading.Thread(target=_produce, args=(archive, wanted, q, block), daemon=True)
    t.start()
    while True:
        name, payload = q.get()
        if name == "!":
            raise payload
        if name == "":
            digest, n = payload
            break
        yield name, payload
    t.join()
    if digest != ARCHIVE_SHA256 or n != ARCHIVE_BYTES:
        raise SystemExit(f"{archive}: SHA-256 {digest} over {n} bytes is not the archive this check was written for "
                         f"({ARCHIVE_SHA256}, {ARCHIVE_BYTES} bytes)")


def _check_toc(toc: bytes) -> dict:
    """The dump's own table of contents: which member holds which table, in which column order."""
    found = {m.group(1).decode(): (m.group(3).decode() + ".dat", tuple(c.strip() for c in m.group(2).decode().split(",")))
             for m in re.finditer(rb"COPY public\.(\w+) \(([^)]*)\) FROM stdin;.{0,256}?(\d+)\.dat", toc, re.S)}
    want = {"node_features": (NODES.split("/")[-1], NODE_COLUMNS), "transaction_edges": (EDGES.split("/")[-1], EDGE_COLUMNS)}
    if found != want:
        raise SystemExit(f"toc.dat maps the tables as {found}, not as this check reads them ({want})")
    return {table: member for table, (member, _) in found.items()}


def _read_copy(chunk: bytes, columns, types: dict, include=None):
    """Parse whole lines of PostgreSQL COPY text: tab-separated, NULL written as backslash-N, no quoting."""
    import pyarrow as pa
    from pyarrow import csv

    return csv.read_csv(pa.BufferReader(chunk), read_options=csv.ReadOptions(column_names=list(columns)),
                        parse_options=csv.ParseOptions(delimiter="\t", quote_char=False, double_quote=False,
                                                       escape_char=False),
                        convert_options=csv.ConvertOptions(column_types=types, include_columns=include,
                                                           null_values=["\\N"], strings_can_be_null=True))


def _node_types() -> dict:
    import pyarrow as pa

    return {c: (pa.string() if c == "label" else pa.int64()) for c in NODE_COLUMNS}


def _numeric(table) -> np.ndarray:
    """The 19 numeric node columns as float64, NULL as NaN (every value is below 2**53, so exact)."""
    return np.column_stack([table[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in NUMERIC])


# ----------------------------------------------------------------------------------------------------- the two passes


def _cached(work: Path, name: str, params: dict, fresh: bool):
    meta, data = work / f"{name}.json", work / f"{name}.npz"
    if fresh or not meta.exists() or not data.exists():
        return None
    m = json.loads(meta.read_text())
    if m.get("params") != params:
        return None
    with np.load(data, allow_pickle=False) as z:
        return m, {k: z[k] for k in z.files}


def _save(work: Path, name: str, params: dict, summary: dict, arrays: dict) -> None:
    np.savez(work / f"{name}.npz", **arrays)
    (work / f"{name}.json").write_text(json.dumps({"params": params, "summary": summary}, indent=1) + "\n")


def labelled_nodes(archive: Path, work: Path, fresh: bool) -> tuple[dict, dict]:
    """Pass 1: every node row is counted; the rows with a label are kept, whatever the label."""
    import pyarrow as pa
    import pyarrow.compute as pc

    params = {"stage": "labelled_nodes", "version": STAGE_VERSION, "archive_sha256": ARCHIVE_SHA256}
    if (hit := _cached(work, "btg-labelled-nodes", params, fresh)) is not None:
        return hit[0]["summary"], hit[1]
    t0, rows, max_alias, kept, toc = time.time(), 0, -1, [], None
    for name, chunk in _stream(archive, {NODES}):
        if name == TOC:
            toc = _check_toc(chunk)
            continue
        if chunk is None:
            continue
        t = _read_copy(chunk, NODE_COLUMNS, _node_types())
        rows += t.num_rows
        max_alias = max(max_alias, pc.max(t["alias"]).as_py())
        kept.append(t.filter(pc.is_valid(t["label"])))
    lab = pa.concat_tables(kept)
    order = np.argsort(lab["alias"].to_numpy())
    aliases = lab["alias"].to_numpy()[order]
    labels = lab["label"].to_pylist()
    labels = [labels[i] for i in order]
    if len(np.unique(aliases)) != len(aliases):
        raise SystemExit("a labelled alias appears twice in node_features, whose primary key is the alias")
    names = sorted(set(labels))
    summary = {"toc": toc, "node_rows": rows, "max_alias": int(max_alias), "labelled_nodes": int(len(aliases)),
               "labels": {k: int(v) for k, v in sorted(Counter(labels).items(), key=lambda kv: -kv[1])},
               "archive_verified": True, "seconds": round(time.time() - t0, 1)}
    arrays = {"aliases": aliases, "label_names": np.array(names),
              "label_codes": np.array([names.index(v) for v in labels], dtype=np.int64), "raw": _numeric(lab)[order]}
    _save(work, "btg-labelled-nodes", params, summary, arrays)
    return summary, arrays


def one_hop(archive: Path, work: Path, fresh: bool, stage1: tuple[dict, dict]) -> tuple[dict, dict]:
    """Pass 2: the edges with a labelled endpoint, the neighbourhood draws, then the sampled neighbours' features.

    The graph is read as undirected, as the paper does: u is a neighbour of v if either sent to the other. Each draw
    takes all of a node's distinct neighbours when it has at most FANOUT of them, else FANOUT without replacement.
    """
    import pyarrow as pa

    s1, a1 = stage1
    params = {"stage": "one_hop", "version": STAGE_VERSION, "archive_sha256": ARCHIVE_SHA256, "fanout": FANOUT,
              "draws": DRAWS, "sampler_seed": SAMPLER_SEED, "classes": list(CLASSES)}
    if (hit := _cached(work, "btg-one-hop", params, fresh)) is not None:
        return hit[0]["summary"], hit[1]
    names = [str(n) for n in a1["label_names"]]
    if missing := [c for c in CLASSES if c not in names]:
        raise SystemExit(f"node_features carries no node labelled {missing}; its labels are {names}")
    lut = np.array([CLASSES.index(n) if n in CLASSES else -1 for n in names])
    codes = lut[a1["label_codes"]]
    in_task = codes >= 0
    codes = codes[in_task]
    task = a1["aliases"][in_task]                           # sorted
    mask = np.zeros(s1["max_alias"] + 1, dtype=bool)
    mask[task] = True
    t0, st, own, nbr = time.time(), Counter(), [], []
    need = table = deg = None
    got_alias, got_raw = [], []
    for name, chunk in _stream(archive, {EDGES, NODES}):
        if name == TOC:
            _check_toc(chunk)
            continue
        if name == EDGES and chunk is not None:
            t = _read_copy(chunk, EDGE_COLUMNS, {"a": pa.int32(), "b": pa.int32()}, include=["a", "b"])
            if t["a"].null_count or t["b"].null_count:
                raise SystemExit("an edge without an endpoint")
            a, b = t["a"].to_numpy(), t["b"].to_numpy()
            if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > s1["max_alias"]:
                raise SystemExit("an edge endpoint outside node_features' aliases")
            la, lb, loop = mask[a], mask[b], a == b
            st["edges"] += len(a)
            st["edges_with_a_labelled_endpoint"] += int((la | lb).sum())
            st["edges_between_two_labelled_nodes"] += int((la & lb).sum())
            st["self_loops"] += int(loop.sum())
            ka, kb = la & ~loop, lb & ~loop
            own += [a[ka], b[kb]]
            nbr += [b[ka], a[kb]]
        elif name == EDGES:                                  # the edge table has ended: draw the neighbourhoods
            li = np.searchsorted(task, np.concatenate(own).astype(np.int64))
            key = np.unique((li << 32) | np.concatenate(nbr).astype(np.int64))
            own = nbr = None
            li, nb = key >> 32, key & 0xFFFFFFFF
            key = None
            indptr = np.searchsorted(li, np.arange(len(task) + 1))
            deg = np.diff(indptr)
            st["distinct_neighbour_pairs"] = int(len(nb))
            rng = np.random.default_rng(SAMPLER_SEED)
            table = np.full((DRAWS, len(task), FANOUT), -1, dtype=np.int64)
            for i in np.flatnonzero(deg <= FANOUT):
                table[:, i, :deg[i]] = nb[indptr[i]:indptr[i + 1]]
            big = np.flatnonzero(deg > FANOUT)
            for r in range(DRAWS):
                for i in big:
                    table[r, i] = nb[indptr[i] + rng.choice(deg[i], FANOUT, replace=False)]
            li = nb = None
            need = np.zeros(s1["max_alias"] + 1, dtype=bool)
            need[task] = True
            need[table[table >= 0]] = True
        elif name == NODES and chunk is not None:
            t = _read_copy(chunk, NODE_COLUMNS, _node_types())
            keep = need[t["alias"].to_numpy()]
            if keep.any():
                t = t.filter(pa.array(keep))
                got_alias.append(t["alias"].to_numpy())
                got_raw.append(_numeric(t))
    union = np.concatenate(got_alias)
    order = np.argsort(union)
    union, raw = union[order], np.concatenate(got_raw)[order]
    if len(union) != int(need.sum()) or len(np.unique(union)) != len(union):
        raise SystemExit("the sampled neighbourhoods name nodes node_features does not hold exactly once")
    lab_rows = np.searchsorted(union, task)
    if not np.array_equal(np.nan_to_num(raw[lab_rows], nan=-1), np.nan_to_num(a1["raw"][in_task], nan=-1)):
        raise SystemExit("the two passes read different features for a labelled node")
    rows = np.where(table >= 0, np.searchsorted(union, np.where(table >= 0, table, 0)), -1)
    per_class = {}
    for c, cls in enumerate(CLASSES):
        d = deg[codes == c]
        per_class[cls] = {"nodes": int(len(d)), "no_neighbour": int((d == 0).sum()),
                          "median_neighbours": float(np.median(d)), "p90_neighbours": float(np.quantile(d, 0.9))}
    summary = {**{k: int(v) for k, v in st.items()}, "task_nodes": int(len(task)), "nodes_read": int(len(union)),
               "sampled_neighbours_unlabelled_or_other": int(len(union) - len(task)),
               "neighbours_per_class": per_class, "archive_verified": True, "seconds": round(time.time() - t0, 1)}
    arrays = {"raw": raw, "lab_rows": lab_rows, "codes": codes.astype(np.int64), "draws": rows.astype(np.int64),
              "degree": deg.astype(np.int64)}
    _save(work, "btg-one-hop", params, summary, arrays)
    return summary, arrays


# ------------------------------------------------------------------------------------------------ features and models


def node_features(raw: np.ndarray) -> np.ndarray:
    """The paper's 27 non-USD features from the 19 numeric columns (NaN where a column is NULL)."""
    c = {name: raw[:, j] for j, name in enumerate(NUMERIC)}
    with np.errstate(divide="ignore", invalid="ignore"):
        age = np.fmax(c["last_transaction_out"], c["last_transaction_in"]) - np.fmin(c["first_transaction_out"],
                                                                                   c["first_transaction_in"])
        per_age = np.clip(np.nan_to_num(age, nan=1.0), 1.0, np.inf)
        f = {k: c[k] for k in ("degree", "degree_in", "degree_out", "total_transactions_in", "total_transactions_out",
                               "min_sent", "total_sent", "max_sent", "min_received", "total_received", "max_received",
                               "cluster_size", "cluster_num_edges", "cluster_num_nodes_in_cc")}
        f["avg_sent"] = c["total_sent"] / np.nan_to_num(c["total_transactions_out"], nan=1.0)
        f["avg_received"] = c["total_received"] / np.nan_to_num(c["total_transactions_in"], nan=1.0)
        f["age"] = age
        f["degree_age_ratio"] = c["degree"] / per_age
        f["degree_in_age_ratio"] = c["degree_in"] / per_age
        f["degree_out_age_ratio"] = c["degree_out"] / per_age
        f["transactions_in_age_ratio"] = c["total_transactions_in"] / per_age
        f["transactions_out_age_ratio"] = c["total_transactions_out"] / per_age
        f["cluster_size_age_ratio"] = c["cluster_size"] / per_age
        f["cluster_num_edges_age_ratio"] = c["cluster_num_edges"] / per_age
        f["proportion_nodes_in_cc"] = c["cluster_num_nodes_in_cc"] / c["cluster_size"]
        f["time_before_first_transaction"] = c["first_transaction_out"] - c["first_transaction_in"]
        f["degree_out_in_ratio"] = c["degree_out"] / c["degree_in"]
    return np.column_stack([f[name] for name, _ in FEATURES])


class PaperScaling:
    """The paper's pre-processing: zeros to missing, log, (log x - q_lo) / (q_95 - q_lo), clip to [0, 1], missing to 0.

    q_lo is the minimum for counts and the 5th percentile for amounts, both over the rows it is fitted on; the share of
    non-isolated addresses is passed through with a missing value set to 1, as the authors' code does.
    """

    def fit(self, f: np.ndarray) -> PaperScaling:
        self.lo, self.hi = np.zeros(f.shape[1]), np.ones(f.shape[1])
        for j, (_, kind) in enumerate(FEATURES):
            if kind != SHARE:
                v = f[:, j]
                v = np.log(v[v > 0])
                self.lo[j], self.hi[j] = np.quantile(v, 0.05 if kind == VALUE else 0.0), np.quantile(v, 0.95)
        return self

    def transform(self, f: np.ndarray) -> np.ndarray:
        out = np.empty(f.shape, dtype=np.float32)
        for j, (_, kind) in enumerate(FEATURES):
            v = f[:, j]
            if kind == SHARE:
                out[:, j] = np.nan_to_num(v, nan=1.0)
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (np.log(np.where(v > 0, v, np.nan)) - self.lo[j]) / max(self.hi[j] - self.lo[j], 1e-12)
            out[:, j] = np.nan_to_num(np.clip(z, 0.0, 1.0), nan=0.0)
        return out


def log_features(f: np.ndarray) -> np.ndarray:
    """The booster's inputs: counts and amounts log-scaled and not clipped; zero, missing or infinite left missing."""
    out = f.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for j, (_, kind) in enumerate(FEATURES):
            if kind != SHARE:
                out[:, j] = np.log(np.where(f[:, j] > 0, f[:, j], np.nan))
    out[~np.isfinite(out)] = np.nan
    return out


def split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 40/30/30 over the labelled nodes."""
    from sklearn.model_selection import train_test_split

    rest, test = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    train, val = train_test_split(rest, test_size=3 / 7, stratify=y[rest], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    per = f1_score(y, pred, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    return {"macro_f1": float(np.mean(per)), "per_class_f1": {c: float(v) for c, v in zip(CLASSES, per, strict=True)},
            "accuracy": float(accuracy_score(y, pred))}


def gbc_paper_setting(f, y, train, val, test, seed) -> tuple[np.ndarray, dict]:
    from sklearn.ensemble import GradientBoostingClassifier

    fit = np.concatenate([train, val])
    sc = PaperScaling().fit(f[fit])
    clf = GradientBoostingClassifier(random_state=seed).fit(sc.transform(f[fit]), y[fit])
    return clf.predict(sc.transform(f[test])), {"trees_per_class": int(clf.n_estimators_)}


def _val_macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0))


def hgb_features(f, y, train, val, test, seed) -> tuple[np.ndarray, dict]:
    """Grown to HGB_ITER iterations on the training split; the iteration with the best validation macro-F1 is kept."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    x = log_features(f)
    clf = HistGradientBoostingClassifier(max_iter=HGB_ITER, early_stopping=False, class_weight="balanced",
                                         random_state=seed).fit(x[train], y[train])
    val_f1 = [_val_macro_f1(y[val], p) for p in clf.staged_predict(x[val])]
    k = int(np.argmax(val_f1))
    pred = next(p for i, p in enumerate(clf.staged_predict(x[test])) if i == k)
    return pred, {"iterations_grown": int(clf.n_iter_), "chosen_iteration": k + 1, "val_macro_f1": round(val_f1[k], 6)}


def make_net(d_in: int, graph: bool):
    """Two GraphSAGE layers on a labelled node's sampled star, or (graph=False) the same layers without neighbours.

    On the star each sampled neighbour's only neighbour is the labelled node, so layer one gives the labelled node
    relu(W1 x_v + U1 mean(x_u)) and each neighbour relu(W1 x_u + U1 x_v), and layer two reads
    W2 h_v + U2 mean(h_u): PyTorch Geometric's SAGEConv (mean aggregation, root weight) written out.
    """
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph = graph
            self.root1, self.root2 = torch.nn.Linear(d_in, HIDDEN), torch.nn.Linear(HIDDEN, len(CLASSES))
            if graph:
                self.nbr1 = torch.nn.Linear(d_in, HIDDEN, bias=False)
                self.nbr2 = torch.nn.Linear(HIDDEN, len(CLASSES), bias=False)
            self.drop = torch.nn.Dropout(DROPOUT)

        def forward(self, xv, xu=None, m=None):
            if not self.graph:
                return self.root2(self.drop(torch.relu(self.root1(xv))))
            w = m / m.sum(1, keepdim=True).clamp(min=1.0)               # mean over the sampled neighbours, 0 if none
            hv = torch.relu(self.root1(xv) + self.nbr1(torch.einsum("bk,bkf->bf", w, xu)))
            hu = torch.relu(self.root1(xu) + self.nbr1(xv)[:, None, :])
            hv, hu = self.drop(hv), self.drop(hu)
            return self.root2(hv) + self.nbr2(torch.einsum("bk,bkf->bf", w, hu))

    return Net()


def _resample(train: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Every class brought to between MIN_ROWS and MAX_ROWS rows: down without replacement, up with."""
    out = []
    for c in range(len(CLASSES)):
        rows = train[y[train] == c]
        if len(rows) > MAX_ROWS:
            rows = rng.choice(rows, MAX_ROWS, replace=False)
        elif len(rows) < MIN_ROWS:
            rows = np.concatenate([rows, rng.choice(rows, MIN_ROWS - len(rows), replace=True)])
        out.append(rows)
    return np.concatenate(out)


def train_net(raw, lab_rows, draws, y, train, val, test, seed, graph: bool, device: str) -> tuple[np.ndarray, dict]:
    """The paper's training on the GPU for its MAX_EPOCHS epochs; the epoch with the best validation macro-F1 is kept.

    The epoch where the class-balanced validation cross-entropy is lowest is recorded beside it, with the validation
    macro-F1 there: stopping on that loss, as the authors' code stops on its validation loss, keeps an earlier epoch.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pad = len(raw)
    # Scaling constants from the training nodes and their sampled neighbours, as the paper fits them on a training split.
    seen = np.concatenate([lab_rows[train], draws[:, train, :].ravel()])
    sc = PaperScaling().fit(node_features(raw[np.unique(seen[seen >= 0])]))
    x = torch.zeros((pad + 1, len(FEATURES)), dtype=torch.float32, device=device)
    for s in range(0, pad, 1 << 20):
        block = torch.from_numpy(sc.transform(node_features(raw[s:s + (1 << 20)])))
        x[s:s + len(block)] = block                                 # row `pad` stays zero: the missing neighbour
    rows = torch.as_tensor(lab_rows, device=device)
    nbr = torch.as_tensor(np.where(draws >= 0, draws, pad), device=device)
    yt = torch.as_tensor(y, device=device)
    counts = np.clip(np.bincount(y[train], minlength=len(CLASSES)), MIN_ROWS, MAX_ROWS)
    weight = counts.sum() / (len(CLASSES) * counts)
    weight = torch.as_tensor(weight / weight.mean(), dtype=torch.float32, device=device)
    net = make_net(len(FEATURES), graph).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    def logits(idx, r):
        if not graph:
            return net(x[rows[idx]])
        u = nbr[r, idx]
        return net(x[rows[idx]], x[u], (u != pad).float())

    def proba(idx: np.ndarray):
        net.eval()
        out = torch.zeros((len(idx), len(CLASSES)), device=device)
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                b = torch.as_tensor(idx[s:s + 8192], device=device)
                for r in range(DRAWS if graph else 1):
                    out[s:s + len(b)] += torch.softmax(logits(b, torch.full_like(b, r)), 1)
        return out / (DRAWS if graph else 1)

    yv = yt[torch.as_tensor(val, device=device)]
    best_f1, best_state, best_epoch = -1.0, None, -1
    min_ce, min_ce_epoch, f1_at_min_ce = np.inf, -1, -1.0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        order = rng.permutation(_resample(train, y, rng))
        r_all = torch.as_tensor(rng.integers(0, DRAWS, size=len(order)), device=device)
        o_all = torch.as_tensor(order, device=device)
        for s in range(0, len(order) - BATCH + 1, BATCH):           # the last short batch is dropped, as the paper's
            b = o_all[s:s + BATCH]
            loss = F.cross_entropy(logits(b, r_all[s:s + BATCH]), yt[b], weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        p = proba(val)
        ce = -torch.log(p[torch.arange(len(val), device=device), yv].clamp(min=1e-12))
        v = float(torch.stack([ce[yv == c].mean() for c in range(len(CLASSES))]).mean())
        f1 = _val_macro_f1(y[val], p.argmax(1).cpu().numpy())
        if f1 > best_f1:
            best_f1, best_epoch, best_state = f1, epoch, copy.deepcopy(net.state_dict())
        if v < min_ce:
            min_ce, min_ce_epoch, f1_at_min_ce = v, epoch, f1
    net.load_state_dict(best_state)
    pred = proba(test).argmax(1).cpu().numpy()
    return pred, {"epochs_run": MAX_EPOCHS, "chosen_epoch": best_epoch + 1, "val_macro_f1": round(best_f1, 6),
                  "lowest_val_balanced_ce_epoch": min_ce_epoch + 1, "val_macro_f1_there": round(f1_at_min_ce, 6),
                  "parameters": int(sum(p.numel() for p in net.parameters())), "train_seconds": round(time.time() - t0, 1)}


def address_list(path: Path) -> dict:
    """addresses.csv's categories, counted, against the paper's Table 4; no address or entity leaves this function."""
    import pandas as pd

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ADDRESSES_SHA256:
        raise SystemExit(f"{path}: SHA-256 {digest} is not the file this check was written for ({ADDRESSES_SHA256})")
    df = pd.read_csv(path, usecols=["address", "category"])
    counts = {k: int(v) for k, v in df["category"].value_counts().items()}
    return {"file_sha256": digest, "rows": int(len(df)), "distinct_addresses": int(df["address"].nunique()),
            "rows_without_category": int(df["category"].isna().sum()), "by_category": counts,
            "by_category_equals_paper_table_4": counts == PAPER["addresses_by_category"],
            "used": "not for the labels: the dump has no map from addresses to nodes, and node_features carries each "
                    "labelled node's type, which is what the paper predicts"}


def summarise(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "min": float(a.min()),
            "max": float(a.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--addresses", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path, help="intermediate files; outside the repository")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fresh", action="store_true", help="recompute both passes rather than reuse --work")
    args = ap.parse_args()
    if args.work.resolve().is_relative_to(REPO):
        raise SystemExit("--work must lie outside the repository")
    args.work.mkdir(parents=True, exist_ok=True)

    import pyarrow
    import sklearn
    import torch
    from threadpoolctl import threadpool_limits

    t_all = time.time()
    addresses = address_list(args.addresses)
    stage1 = labelled_nodes(args.archive, args.work, args.fresh)
    s1 = stage1[0]
    s2, a2 = one_hop(args.archive, args.work, args.fresh, stage1)
    raw, lab_rows, draws, y = a2["raw"], a2["lab_rows"], a2["draws"], a2["codes"]
    f_lab = node_features(raw[lab_rows])
    # The edges read must give every labelled node the distinct-neighbour count node_features stores as its degree.
    degree_col = raw[lab_rows][:, NUMERIC.index("degree")]
    degree_check = {"nodes": int(len(y)), "mismatches": int((np.nan_to_num(degree_col) != a2["degree"]).sum()),
                    "null_degree_and_no_edge": int((np.isnan(degree_col) & (a2["degree"] == 0)).sum())}
    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed, splits, details = {m: [] for m in MODELS}, {}, {m: [] for m in MODELS}
    for seed in seeds:
        train, val, test = split(y, seed)
        splits[str(seed)] = {name: {c: int((y[idx] == k).sum()) for k, c in enumerate(CLASSES)}
                             for name, idx in (("train", train), ("validation", val), ("test", test))}
        with threadpool_limits(limits=args.threads):
            for name, fit in (("gbc_paper_setting", gbc_paper_setting), ("hgb_features", hgb_features)):
                t0 = time.time()
                pred, info = fit(f_lab, y, train, val, test, seed)
                per_seed[name].append({"seed": seed, **scores(y[test], pred)})
                details[name].append({"seed": seed, **info, "seconds": round(time.time() - t0, 1)})
            for name, graph in (("mlp_features", False), ("gnn_one_hop", True)):
                pred, info = train_net(raw, lab_rows, draws, y, train, val, test, seed, graph, args.device)
                per_seed[name].append({"seed": seed, **scores(y[test], pred)})
                details[name].append({"seed": seed, **info})
        print(json.dumps({"seed": seed, **{m: round(per_seed[m][-1]["macro_f1"], 4) for m in MODELS}}), flush=True)

    results = {}
    for m in MODELS:
        results[m] = {"macro_f1": summarise([r["macro_f1"] for r in per_seed[m]]),
                      "per_class_f1": {c: summarise([r["per_class_f1"][c] for r in per_seed[m]]) for c in CLASSES},
                      "accuracy": summarise([r["accuracy"] for r in per_seed[m]]),
                      "per_seed": per_seed[m], "runs": details[m]}
    # Table 6's macro-F1 against the mean of its own seven per-class F1: the GNN rows agree to 0.005, the GBC row does not.
    paper_means = {k: round(float(np.mean(v)), 4) for k, v in PAPER["per_class_f1"].items()}
    paper = PAPER | {"per_class_mean": paper_means,
                     "gbc_row": f"the GBC row's seven per-class F1 average {paper_means['GBC']:.3f}, not the "
                                f"{PAPER['macro_f1']['GBC']:.2f} printed as its macro-F1; each GNN row's average is within "
                                f"{max(abs(paper_means[k] - PAPER['macro_f1'][k]) for k in PAPER['macro_f1'] if k != 'GBC'):.4f} "
                                "of its printed macro-F1"}
    gain = [g["macro_f1"] - b["macro_f1"] for g, b in zip(per_seed["gnn_one_hop"], per_seed["mlp_features"], strict=True)]
    gain_hgb = [g["macro_f1"] - b["macro_f1"] for g, b in zip(per_seed["gnn_one_hop"], per_seed["hgb_features"], strict=True)]
    task_counts = {c: int((y == k).sum()) for k, c in enumerate(CLASSES)}
    record = {
        "what": "BitcoinTemporalGraph's entity types (the paper's seven) predicted from a labelled node's own features and, "
                "by a GNN, from its one-hop neighbourhood too: a real-data check of whether the graph adds to the features",
        "not": "a laundering result, a replacement for AMLworld's task, a reproduction of the paper's two-hop GNNs, or a "
               "detector of any entity type",
        "data": {"name": "BitcoinTemporalGraph", "source": "Figshare article 26305093, version 3",
                 "doi": "10.6084/m9.figshare.26305093.v3", "licence": "CC BY 4.0",
                 "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                 "citation": "Schnoering and Vazirgiannis, BitcoinTemporalGraph, figshare (2024), "
                             "doi:10.6084/m9.figshare.26305093.v3",
                 "archive": {"file": "dataset.tar.gz", "sha256": ARCHIVE_SHA256, "bytes": ARCHIVE_BYTES,
                             "members_read": [TOC, EDGES, NODES], "tables": s1["toc"],
                             "verified_on_both_passes": bool(s1["archive_verified"] and s2["archive_verified"])},
                 "node_rows": s1["node_rows"], "edge_rows": s2["edges"],
                 "rows_match_the_paper": s1["node_rows"] == PAPER["nodes"] and s2["edges"] == PAPER["edges"],
                 "labelled_nodes": s1["labelled_nodes"], "labels": s1["labels"],
                 "labelled_nodes_match_the_paper": s1["labelled_nodes"] == PAPER["labelled_nodes"],
                 "task_classes": list(CLASSES), "task_nodes": task_counts,
                 "addresses_csv": addresses},
        "paper": paper,
        "one_hop": {k: s2[k] for k in ("edges", "edges_with_a_labelled_endpoint", "edges_between_two_labelled_nodes",
                                       "self_loops", "distinct_neighbour_pairs", "nodes_read",
                                       "sampled_neighbours_unlabelled_or_other", "neighbours_per_class")}
                   | {"degree_column_against_the_edges": degree_check,
                      "fanout": FANOUT, "draws": DRAWS, "sampler_seed": SAMPLER_SEED,
                      "direction": "undirected: u is a neighbour of v if either sent to the other; self-loops dropped"},
        "features": {"names": [n for n, _ in FEATURES],
                     "dropped": "the paper's eight USD conversions, which need a BTC/USD price series the dump does not carry",
                     "networks_and_paper_gbc": "the paper's pre-processing (zeros to missing, log, scaled by the minimum "
                                               "or 5th percentile and the 95th percentile of the training rows, clipped to "
                                               "[0, 1], missing to 0; the share of non-isolated addresses unscaled, missing "
                                               "to 1); for the networks fitted on the training nodes and their sampled "
                                               "neighbours, for the paper's GBC on its training and validation nodes",
                     "hgb": "the same features log-scaled and not clipped; zero, missing or infinite left missing"},
        "split": {"kind": "stratified by class, 40% training, 30% validation, 30% test",
                  "why": "the paper's split is random (not by time) over the labelled nodes; a stratified split keeps its "
                         "proportions in every class on every seed",
                  "seeds": seeds, "counts": splits,
                  "caveat": "the split is by node, as the paper's: nodes of one entity can fall on both sides, and the dump "
                            "has no map from addresses to nodes to group them"},
        "models": {
            "gbc_paper_setting": "sklearn GradientBoostingClassifier, default settings, fitted on training and validation "
                                 "together on the natural class mix (the paper's GBC)",
            "hgb_features": f"sklearn HistGradientBoostingClassifier, class_weight balanced, grown to {HGB_ITER} iterations "
                            "on the training split; the iteration with the best validation macro-F1 is kept",
            "mlp_features": f"the GNN's layers without neighbours: {len(FEATURES)}-{HIDDEN}-{len(CLASSES)}, trained as the GNN",
            "gnn_one_hop": f"two GraphSAGE layers (mean aggregation, root weight) on the one-hop star, {HIDDEN} hidden, plain "
                           f"PyTorch; test probabilities averaged over the {DRAWS} draws",
            "training": f"ReLU, dropout {DROPOUT}, Adam lr {LR}, batch {BATCH}, every class resampled each epoch to between "
                        f"{MIN_ROWS} and {MAX_ROWS} rows, cross-entropy weighted inversely to the resampled counts (mean "
                        f"weight 1), a random draw per row per epoch, {MAX_EPOCHS} epochs, the epoch with the best "
                        "validation macro-F1 kept; each run also records the epoch with the lowest class-balanced "
                        "validation cross-entropy and the validation macro-F1 there"},
        "results": results,
        "gnn_minus_mlp_macro_f1": summarise(gain) | {"per_seed": gain, "seeds_gnn_higher": int(sum(g > 0 for g in gain))},
        "gnn_minus_hgb_macro_f1": summarise(gain_hgb) | {"per_seed": gain_hgb,
                                                         "seeds_gnn_higher": int(sum(g > 0 for g in gain_hgb))},
        "deviations_from_the_paper": [
            "one hop of neighbours, 10 per draw, where the paper samples two hops (10 then 5) and its code builds one "
            "more layer than hops; the owner's scope for this check was one hop",
            "no USD features (27 features instead of 35)",
            "a stratified split over five seeds instead of one random split",
            "resampling and the neighbourhood draw are redrawn every epoch (the paper: every 100 epochs)",
            "the networks keep the epoch with the best validation macro-F1 over 2000 epochs, where the authors' code "
            "keeps the lowest validation loss and stops 50 epochs after it; the lowest-loss epoch is recorded per run",
            "GraphSAGE only of the paper's four GNNs; the features-only booster and network are additions"],
        "runtime_seconds": {"pass_1_labelled_nodes": s1["seconds"], "pass_2_one_hop": s2["seconds"],
                            "total": round(time.time() - t_all, 1)},
        "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                        "script_as_committed": all(subprocess.run(["git", "-C", str(REPO), *cmd, "scripts/measure_btg_check.py"],
                                                                  capture_output=True).returncode == 0
                                                   for cmd in (["ls-files", "--error-unmatch"], ["diff", "--quiet", "HEAD", "--"])),
                        "fresh_passes": bool(args.fresh), "python": platform.python_version(), "numpy": np.__version__,
                        "pyarrow": pyarrow.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
                        "cpu_threads": args.threads, "date": time.strftime("%Y-%m-%d")},
    }
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({m: {"macro_f1": results[m]["macro_f1"]} for m in MODELS}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
