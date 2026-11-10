#!/usr/bin/env python
"""The transaction graphs the finance page draws (Round 86): laundering patterns from the held-out window, and the neighbourhood of each row the published MLP can score.

WHY A GRAPH. The AMLworld paper presents its data this way (Altman et al., NeurIPS 2023, Figure 1, transactions in tabular and graph form; Figure 2, the eight laundering patterns modelled), because laundering is a pattern across accounts and no single row shows it.
The features the MLP reads are counts over such neighbourhoods, so this is also what it looks at.

WHAT IS DRAWN. (1) One exemplar per shape, a connected component of laundering-labelled edges in the held-out test window, chosen by a fixed rule: 4 to 12 edges, the earliest component of that shape. (2) For each fixture row of `finance_aml.mlp_large`, the edges at its two accounts
in the three days up to the end of its batch, at most 30, with its own edge and the laundering labels marked.

Account identifiers are replaced by letters, per graph. The file carries the notices CDLA-Sharing-1.0 asks for.

    QSH_DATA_ROOT=... python scripts/data/build_transaction_graphs.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "train")]

from finance_aml import load_sorted, temporal_windows  # noqa: E402
from finance_aml_fan import BATCH_WINDOW  # noqa: E402

from qsh.models import registry  # noqa: E402

OUT = REPO / "web" / "public" / "data" / "finance" / "transaction_graphs.json"
SHAPES = ("fan-out", "fan-in", "gather-scatter", "bipartite", "cycle", "chain")
WINDOW_MINUTES = 3 * 24 * 60
MAX_EDGES = 30


def letters(n: int) -> str:
    """A, B, ..., Z, AA, AB, ...: short names for the accounts of one graph."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def shape_of(edges) -> str:
    """The shape of a connected component of laundering edges, from its in- and out-degrees."""
    out = edges.groupby("from_account").size()
    into = edges.groupby("to_account").size()
    both = set(edges.from_account) & set(edges.to_account)
    n = len(edges)
    if n >= 3 and out.max() == n:
        return "fan-out"
    if n >= 3 and into.max() == n:
        return "fan-in"
    if out.max() == 1 and into.max() == 1:
        return "cycle" if set(edges.from_account) == set(edges.to_account) else "chain"
    if len(both) == 1 and out.max() > 1 and into.max() > 1:
        return "gather-scatter"
    return "bipartite" if not both else "mixed"


def graph_of(edges, origin) -> dict:
    """Named accounts and edges in time order, minutes from `origin`."""
    names: dict[str, str] = {}

    def name(a: str) -> str:
        return names.setdefault(a, letters(len(names)))

    out = []
    for e in edges.sort_values("timestamp", kind="stable").itertuples():
        out.append({"from": name(e.from_account), "to": name(e.to_account), "minute": int((e.timestamp - origin).total_seconds() / 60),
                    "amount": round(float(e.amount_paid), 2), "currency": str(e.pay_currency), "format": str(e.payment_format), "laundering": int(e.label)})
    return {"nodes": list(names.values()), "edges": out}


def main() -> int:
    df = load_sorted()
    _, _, te, _ = temporal_windows(df)
    pos = te[te["label"] == 1]
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pos.from_account, pos.to_account, strict=True):
        parent[find(a)] = find(b)
    components: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(pos.from_account):
        components[find(a)].append(i)
    candidates: dict[str, list[tuple]] = defaultdict(list)
    for idx in components.values():
        if 4 <= len(idx) <= 12:
            edges = pos.iloc[idx]
            candidates[shape_of(edges)].append((edges.timestamp.min(), len(idx), idx))
    exemplars = []
    for shape in SHAPES:
        if not candidates[shape]:
            raise SystemExit(f"no component of shape {shape!r} has 4 to 12 edges")
        first, _n, idx = min(candidates[shape], key=lambda c: (c[0], c[1]))
        exemplars.append({"shape": shape, **graph_of(pos.iloc[idx], first)})

    weights = json.loads((REPO / registry.ARTIFACT_DIRS["finance_aml.mlp_large"] / "weights.json").read_text())
    n_before_test = len(df) - len(te)
    frames = {"from": df.groupby("from_account").indices, "to": df.groupby("to_account").indices}
    minutes = ((df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60).to_numpy().astype(np.int64)
    scored = []
    for row_in_test, label in zip(weights["fixtures"]["row_index_in_test_window"], weights["fixtures"]["labels"], strict=True):
        pos_row = n_before_test + int(row_in_test)
        row = df.iloc[pos_row]
        end = min(len(df) - 1, (pos_row // BATCH_WINDOW + 1) * BATCH_WINDOW - 1)
        t_end, t_start = minutes[end], minutes[end] - WINDOW_MINUTES
        near = np.unique(np.concatenate([frames["from"][row.from_account], frames["to"][row.from_account] if row.from_account in frames["to"] else [],
                                         frames["to"][row.to_account], frames["from"][row.to_account] if row.to_account in frames["from"] else []]).astype(np.int64))
        near = near[(minutes[near] >= t_start) & (minutes[near] <= t_end)]
        keep = np.concatenate([[pos_row], near[near != pos_row]])
        # The row itself, then laundering-labelled edges, then the most recent, up to the cap.
        ordered = sorted(keep[1:], key=lambda i: (-int(df["label"].iloc[int(i)]), -int(minutes[int(i)])))
        chosen = [pos_row, *[int(i) for i in ordered[:MAX_EDGES - 1]]]
        sub = df.iloc[sorted(chosen)]
        origin = sub.timestamp.min()
        g = graph_of(sub, origin)
        scored_edge = [i for i, e in enumerate(sub.sort_values("timestamp", kind="stable").itertuples()) if e.Index == pos_row]
        names_from = g["edges"][scored_edge[0]]["from"]
        names_to = g["edges"][scored_edge[0]]["to"]
        scored.append({"row_index_in_test_window": int(row_in_test), "label": int(label), "from": names_from, "to": names_to, "scored_edge": scored_edge[0], "edges_in_window": int(len(near)), **g})
    ds = registry.DS_AMLWORLD
    payload = {
        "provenance": "IBM AMLworld HI-Small, transactions from the held-out temporal test window and the days before it. The corpus is entirely synthetic by IBM's own statement.",
        "licence": ds.licence, "licence_url": "https://cdla.dev/sharing-1-0/", "attribution": ds.attribution,
        "modifications": ("Not the published rows: the laundering-labelled edges of six connected components of the test window, chosen by a fixed rule, and up to 30 edges around each of the 16 fixture rows of finance_aml.mlp_large. "
                          "Account identifiers are replaced by letters per graph and timestamps by minutes from each graph's first edge."),
        "batch_window": BATCH_WINDOW, "window_minutes": WINDOW_MINUTES,
        "exemplars": exemplars, "scored_rows": scored,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: {OUT.stat().st_size / 1024:.1f} KB, {[(e['shape'], len(e['edges'])) for e in exemplars]}, {len(scored)} scored rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
