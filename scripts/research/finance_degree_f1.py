#!/usr/bin/env python
"""F1 by the receiving account's in-degree, in SALT-GNN's bins, for IBM's two models on this project's windows.

The owner's choice of 2026-11-15 (plan `.claude/plan-finance-degree-f1-20261115.md`): stratify the finance page's IBM
models by the receiver's in-degree, as Losavio, Sovrano, Fenoglio, Gjoreski and Langheinrich do (SALT-GNN,
arXiv:2607.10131, Algorithm 1: $|N_{in}(v)|$ of the transaction's receiver, logarithmic bins 1-4, 5-9, 10-19, 20-49, 50-99
and 100+, F1 at 0.5). The stratification is post hoc; nothing is retrained.

Two windows, because the paper does not state its own:
- `all`: every transfer of the file, which IBM's test graph holds (Multi-GNN `data_loading.py`, `te_edge_index =
  edge_index`), future test transfers included;
- `batch`: the transfers up to the end of the scored one's batch under the causal protocol (Multi-GNN-causal
  `data_loading.causal_batches`: 2,048 rows in time order, split again wherever a batch spans more than the fit window's
  reach, 17 minutes on HI-Small), which is all that protocol lets a transfer read.

Two thresholds: 0.5, the paper's, and the one that maximises F1 on each run's own validation scores, this project's.
Each run scored all but 859 to 895 of the 1,523,504 test transfers (its sampler's draw); an unscored transfer counts as
not flagged, so a positive among them is a miss. Per bin: the seed mean and standard deviation over five runs, and a 95%
bootstrap interval of the seed mean over the bin's rows (the same resample for every run).

    python scripts/research/finance_degree_f1.py --transactions .../formatted_transactions.csv --dumps .../qsh-gnn-work \
        --out docs/measurements/finance-degree-f1-2026-11-20.json --web-copy web/public/data/finance/degree-f1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BIN_EDGES = (5, 10, 20, 50, 100)                       # digitize: [1,5) [5,10) [10,20) [20,50) [50,100) [100,inf)
BIN_NAMES = ("1-4", "5-9", "10-19", "20-49", "50-99", "100+")
ROWS_PER_BATCH = 2048
#: The dumps of IBM's checkpoints with fit-window BatchNorm statistics, the ones the finance page publishes.
FAMILIES = {
    "pna_ibm": ("pna_project_fitbn", "Multi-PNA+EU, IBM's protocol", "all"),
    "pna_causal": ("pna_causal_lag_fitbn", "Multi-PNA+EU, the causal protocol (time as lag)", "batch"),
    "gin_ibm": ("gin_project_fitbn", "Multi-GIN+EU, IBM's protocol", "all"),
    "gin_causal": ("gin_causal_lag_fitbn", "Multi-GIN+EU, the causal protocol (time as lag)", "batch"),
}
#: SALT-GNN, Table 2, HI-Small: F1 at 0.5, mean and standard deviation over 5 seeds, per in-degree bin and overall (the
#: paper has no 100+ bin for HI-Small; * in the paper marks a gain over PNAGMDA at p < 0.05, Holm-corrected). Read in its
#: HTML full text (v1) on 2026-11-15 and again on 2026-11-20, when the plain PNA and GIN rows were added: the plan had
#: said the paper gives no per-bin figures for them, which was wrong.
PUBLISHED = {
    "source": "Losavio et al., SALT-GNN, arXiv:2607.10131 (v1, 2026-07-11), Table 2, HI-Small, F1 at 0.5",
    "caveat": "its split is strictly temporal with no ratios stated and its degree window is unstated, so no ranking "
              "against these rows is claimed",
    "bins": ["1-4", "5-9", "10-19", "20-49", "50-99", "overall"],
    "rows": [
        {"model": "GIN", "f1": [[0.528, 0.076], [0.362, 0.059], [0.479, 0.124], [0.362, 0.127], [0.509, 0.106], [0.418, 0.107]]},
        {"model": "PNA", "f1": [[0.679, 0.069], [0.516, 0.021], [0.662, 0.015], [0.643, 0.030], [0.515, 0.099], [0.629, 0.032]]},
        {"model": "FraudGT", "f1": [[0.733, 0.012], [0.539, 0.022], [0.674, 0.031], [0.696, 0.013], [0.706, 0.044], [0.679, 0.018]]},
        {"model": "PNAGMDA", "f1": [[0.757, 0.039], [0.529, 0.033], [0.680, 0.014], [0.682, 0.020], [0.678, 0.019], [0.676, 0.019]]},
        {"model": "SALT-GAT", "f1": [[0.765, 0.027], [0.557, 0.021], [0.683, 0.016], [0.695, 0.023], [0.751, 0.027], [0.694, 0.007]]},
        {"model": "SALT-Trans", "f1": [[0.795, 0.028], [0.573, 0.042], [0.688, 0.016], [0.723, 0.013], [0.741, 0.019], [0.716, 0.018]]},
    ],
}


def load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path, usecols=["EdgeID", "to_id", "Timestamp", "Is Laundering"])
    assert (df["EdgeID"].to_numpy() == np.arange(len(df))).all(), "EdgeID is not the row number"
    ts = df["Timestamp"].to_numpy(np.float64)
    assert (np.diff(ts) >= 0).all(), "the rows are not in time order"
    return df["to_id"].to_numpy(np.int64), ts, df["Is Laundering"].to_numpy(np.int64)


def windows(n: int) -> tuple[int, int]:
    """This project's windows (Multi-GNN `QSH_SPLIT=project`): fit [0, val_cut), validation [val_cut, cut), test [cut, n)."""
    cut = int(n * 0.7)
    return cut, int(cut * 0.8)


def batch_end(ts: np.ndarray, n_fit: int, rows: int = ROWS_PER_BATCH) -> tuple[np.ndarray, int]:
    """Each row's last row of its causal batch (Multi-GNN-causal `causal_batches`, the same rule), and the reach."""
    n = len(ts)
    end = np.minimum(n - 1, (np.arange(n) // rows + 1) * rows - 1)
    reach = 60 * int((ts[end] - ts)[:n_fit].max() // 60)
    ids, b, t0 = np.empty(n, dtype=np.int64), -1, 0.0
    for i, t in enumerate(ts.tolist()):
        if i % rows == 0 or t - t0 > reach:
            b, t0 = b + 1, t
        ids[i] = b
    last = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    np.maximum.at(last, ids, np.arange(n))
    return last[ids], reach


def in_degrees(to: np.ndarray, rows: np.ndarray, upto: np.ndarray | None) -> np.ndarray:
    """In-degree of each row's receiver over every row (upto None) or over rows with index at most `upto`."""
    n = len(to)
    if upto is None:
        return np.bincount(to)[to[rows]]
    keys = np.sort(to * (n + 1) + np.arange(n))
    return (np.searchsorted(keys, to[rows] * (n + 1) + upto, side="right")
            - np.searchsorted(keys, to[rows] * (n + 1), side="left"))


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """The score that maximises F1 when every row scoring at least it is flagged."""
    order = np.argsort(-probs, kind="stable")
    p, y = probs[order], labels[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    f1 = 2 * tp / (tp + fp + y.sum())
    return float(p[int(np.argmax(f1))])


def f1(y: np.ndarray, flag: np.ndarray) -> float:
    tp = int(np.count_nonzero(flag & (y == 1)))
    if not tp:
        return 0.0
    return 2 * tp / (2 * tp + int(np.count_nonzero(flag & (y == 0))) + int(np.count_nonzero(~flag & (y == 1))))


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def per_run_and_bootstrap(y: np.ndarray, flags: np.ndarray, idx: np.ndarray, bootstrap: int,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """F1 of each run on the rows `idx`, and the bootstrap distribution of the runs' mean F1 over those rows.

    F1 depends on a row only through its label and the five runs' flags, 64 patterns at most, so a resample of the rows
    is a multinomial draw of the pattern counts: the same distribution as resampling rows, at a cost independent of the
    bin's size (the dense bins hold a million rows)."""
    runs = flags.shape[0]
    pattern = y[idx].astype(np.int64) << runs
    for r in range(runs):
        pattern |= flags[r, idx].astype(np.int64) << r
    counts = np.bincount(pattern, minlength=2 << runs).astype(np.float64)
    codes = np.arange(2 << runs)
    pos = (codes >> runs) & 1
    hit = np.array([(codes >> r) & 1 for r in range(runs)])            # runs x patterns
    m_tp, m_fp, m_fn = hit * pos, hit * (1 - pos), (1 - hit) * pos
    point = f1_from_counts(m_tp @ counts, m_fp @ counts, m_fn @ counts)
    if not len(idx):
        return point, np.zeros(1)
    draws = rng.multinomial(len(idx), counts / counts.sum(), size=bootstrap).astype(np.float64)   # bootstrap x patterns
    boot = f1_from_counts(draws @ m_tp.T, draws @ m_fp.T, draws @ m_fn.T)                         # bootstrap x runs
    return point, boot.mean(axis=1)


def run(transactions: Path, dumps: Path, bootstrap: int, seed: int) -> dict:
    to, ts, y = load_rows(transactions)
    n = len(to)
    cut, val_cut = windows(n)
    test = np.arange(cut, n)
    upto, reach = batch_end(ts, val_cut)
    deg = {"all": in_degrees(to, test, None), "batch": in_degrees(to, test, upto[test])}
    bins = {w: np.digitize(d, BIN_EDGES) for w, d in deg.items()}
    yt = y[test]
    rng = np.random.default_rng(seed)
    out: dict = {
        "what": "F1 by the receiving account's in-degree, in SALT-GNN's bins, for IBM's Multi-GNN models on this "
                "project's windows; computed offline from each run's test and validation scores, nothing retrained",
        "plan": ".claude/plan-finance-degree-f1-20261115.md (the owner's choice of 2026-11-15)",
        "licence": "CDLA-Sharing-1.0",
        "licence_url": "https://cdla.dev/sharing-1-0/",
        "attribution": ("Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', "
                        "NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), "
                        "CDLA-Sharing-1.0."),
        "modifications": ("Not the published rows: counts of HI-Small's test-window transfers and laundering labels by "
                          "the receiver's in-degree, and F1 of model scores on them. No transfer or account is included."),
        "method": {
            "rows": f"AMLworld HI-Small in IBM's format, {n} transfers in time order; test window rows [{cut}, {n}), "
                    f"{n - cut} transfers, {int(yt.sum())} laundering",
            "degree": "in-degree of the transfer's receiver, SALT-GNN Algorithm 1",
            "windows": {
                "all": "every transfer of the file, as IBM's test graph holds them (future test transfers included)",
                "batch": f"transfers up to the end of the scored one's causal batch ({ROWS_PER_BATCH} rows, split where "
                         f"a batch spans more than {reach} s, the fit window's reach)",
            },
            "bins": dict(zip(BIN_NAMES, ["[1, 4]", "[5, 9]", "[10, 19]", "[20, 49]", "[50, 99]", "[100, inf)"], strict=True)),
            "thresholds": {"0.5": "SALT-GNN's (its Appendix F)",
                           "validation": "the score maximising F1 on each run's own validation scores"},
            "unscored": "a test transfer a run did not score counts as not flagged",
            "interval": f"95% percentile bootstrap of the five-run mean over the bin's rows, {bootstrap} resamples, "
                        f"seed {seed}",
        },
        "test_rows": int(n - cut),
        "test_positives": int(yt.sum()),
        "bins_by_window": {
            w: [{"bin": BIN_NAMES[k], "rows": int((b == k).sum()), "positives": int(yt[b == k].sum())} for k in range(6)]
            for w, b in bins.items()
        },
        "families": {},
        "published": PUBLISHED,
    }
    for key, (prefix, label, native) in FAMILIES.items():
        runs = sorted(p for p in dumps.glob(f"{prefix}_seed[0-9]") if p.is_dir())
        assert len(runs) == 5, f"{prefix}: {len(runs)} runs, five expected"
        flags = {"0.5": np.zeros((5, n - cut), dtype=bool), "validation": np.zeros((5, n - cut), dtype=bool)}
        meta = []
        for r, path in enumerate(runs):
            te, va = np.load(path / "test.npz"), np.load(path / "val.npz")
            ids = te["ids"].astype(np.int64)
            assert len(np.unique(ids)) == len(ids) and ids.min() >= cut and ids.max() < n, path.name
            assert (te["labels"] == y[ids]).all(), f"{path.name}: labels disagree with the file"
            thr = best_threshold(va["probs"].astype(np.float64), va["labels"].astype(np.int64))
            for name, t in (("0.5", 0.5), ("validation", thr)):
                flags[name][r, ids - cut] = te["probs"] >= t
            unscored = np.setdiff1d(test, ids)
            meta.append({"run": path.name, "scored": int(len(ids)), "unscored": int(len(unscored)),
                         "unscored_positives": int(y[unscored].sum()), "validation_threshold": thr})
        fam: dict = {"label": label, "native_window": native, "runs": meta, "by_window": {}}
        for w, b in bins.items():
            fam["by_window"][w] = {}
            for name, fl in flags.items():
                overall = np.array([f1(yt, fl[r]) for r in range(5)])
                rows_out = []
                for k in range(6):
                    idx = np.flatnonzero(b == k)
                    per_run, boot = per_run_and_bootstrap(yt, fl, idx, bootstrap, rng)
                    direct = np.array([f1(yt[idx], fl[r, idx]) for r in range(5)])
                    assert np.allclose(per_run, direct), (key, w, name, BIN_NAMES[k], per_run, direct)
                    rows_out.append({"bin": BIN_NAMES[k], "f1_mean": round(float(per_run.mean()), 4),
                                     "f1_sd": round(float(per_run.std(ddof=1)), 4),
                                     "f1_runs": [round(float(v), 4) for v in per_run],
                                     "interval_95": [round(float(np.percentile(boot, 2.5)), 4),
                                                     round(float(np.percentile(boot, 97.5)), 4)]})
                fam["by_window"][w][name] = {"overall_mean": round(float(overall.mean()), 4),
                                             "overall_sd": round(float(overall.std(ddof=1)), 4), "bins": rows_out}
        out["families"][key] = fam
    return out


def check(record: dict) -> list[str]:
    """What the record must hold together, without the data. An empty list means it does."""
    problems: list[str] = []
    for w, rows in record["bins_by_window"].items():
        if [r["bin"] for r in rows] != list(BIN_NAMES):
            problems.append(f"{w}: the bins are not SALT-GNN's")
        if sum(r["rows"] for r in rows) != record["test_rows"]:
            problems.append(f"{w}: the bins' rows do not add up to the test window")
        if sum(r["positives"] for r in rows) != record["test_positives"]:
            problems.append(f"{w}: the bins' positives do not add up to the test window's")
    if set(record["families"]) != set(FAMILIES):
        problems.append("the families are not the four this script scores")
    for key, fam in record["families"].items():
        if len(fam["runs"]) != 5:
            problems.append(f"{key}: {len(fam['runs'])} runs, five expected")
        for w, by_thr in fam["by_window"].items():
            for name, res in by_thr.items():
                for b in res["bins"]:
                    runs = b["f1_runs"]
                    where = f"{key} {w} {name} {b['bin']}"
                    if len(runs) != 5 or not all(0.0 <= v <= 1.0 for v in runs):
                        problems.append(f"{where}: five F1 values in [0, 1] expected")
                    # Each run and the mean are rounded to 4 places apart, so they can differ by up to 1e-4.
                    elif abs(sum(runs) / 5 - b["f1_mean"]) > 1.0001e-4:
                        problems.append(f"{where}: the mean is not the runs' mean")
                    lo, hi = b["interval_95"]
                    if not 0.0 <= lo <= hi <= 1.0:
                        problems.append(f"{where}: the interval is not ordered within [0, 1]")
    if record["published"] != PUBLISHED:
        problems.append("the published rows are not SALT-GNN's Table 2 as recorded in this script")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--transactions", type=Path, required=True)
    ap.add_argument("--dumps", type=Path, required=True, help="the folder holding the runs' <family>_seed<k>/ dumps")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--web-copy", type=Path, help="also write the same bytes here, for the page (web/public/data/finance/degree-f1.json)")
    args = ap.parse_args()
    record = run(args.transactions, args.dumps, args.bootstrap, args.seed)
    problems = check(record)
    assert not problems, problems
    text = json.dumps(record, indent=1) + "\n"
    args.out.write_text(text)
    if args.web_copy:
        args.web_copy.write_text(text)
        print(f"wrote {args.web_copy}")
    print(f"wrote {args.out}")
    for key, fam in record["families"].items():
        w = fam["native_window"]
        for name in ("0.5", "validation"):
            row = fam["by_window"][w][name]
            print(f"  {key:10s} {w:5s} thr={name:10s} overall {row['overall_mean']:.3f} | "
                  + " ".join(f"{b['bin']} {b['f1_mean']:.3f}" for b in row["bins"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
