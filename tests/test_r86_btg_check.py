"""The BitcoinTemporalGraph GNN check is recorded as it came out, and the note quotes the record (Round 86, 2026-11-23).

WHY THIS EXISTS

On the owner's approval of 2026-11-23 a GNN over each labelled node's neighbours was set against models on the node's own
features, on BitcoinTemporalGraph's entity types (scripts/measure_btg_check.py, references/aml-real-data-2026-11.md).
The record is a handful of aggregates distilled from 74 GB of dump that is never kept, so CI cannot re-derive it. What can
be held is that it agrees with itself, quotes the dataset paper's figures as the paper prints them and says where, names
nothing that leads back to an address or a node, and is what the note quotes.

WHAT IS ASSERTED

- Counts add up: the labels to the labelled nodes, the seven classes to the task nodes, each seed's three splits to the
  task nodes class by class, and the one-hop edge counts within the edge rows.
- Every model is reported for every seed; every score lies in [0, 1]; each macro-F1 is the mean of its seven per-class
  F1; each mean, spread, minimum and maximum is that of the per-seed values; the GNN's gains are the recorded differences.
- The paper's Table 6 is quoted as printed, with its URL, and the archive's SHA-256 is the one the script checks; the
  record comes from fresh passes over the archive by the script as committed.
- No Bitcoin address and no long list of numbers (an alias list) is in the record.
- The note quotes the record's figures at the precision it prints them, and each comparison it states (the neighbours
  add on every seed and most on RANSOMWARE, the booster leads on every seed, the rerun is near the paper on five
  classes) is the record's.
- Seeded: a record missing a seed for one model, a macro-F1 that is not its classes' mean, and a note that rounds the
  GNN up are refused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "btg-check-2026-11-23.json"
NOTE = REPO / "references" / "aml-real-data-2026-11.md"
SCRIPT = REPO / "scripts" / "measure_btg_check.py"
CLASSES = ["EXCHANGE", "MINING", "GAMBLING", "PONZI", "INDIVIDUAL", "RANSOMWARE", "BET"]
MODELS = ["gbc_paper_setting", "hgb_features", "mlp_features", "gnn_one_hop"]
#: Table 6 of Schnoering and Vazirgiannis, Scientific Data 12, 404 (2025), as printed.
TABLE_6 = {"GCN": 0.56, "GraphSage": 0.62, "GAT": 0.64, "GIN": 0.63, "GBC": 0.57}
PAPER_URL = "https://www.nature.com/articles/s41597-025-04684-8"
_ADDRESS = re.compile(r"\b(?:[13][1-9A-HJ-NP-Za-km-z]{25,34}|bc1[02-9ac-hj-np-z]{11,71})\b")


def _record() -> dict:
    if not RECORD.exists():
        pytest.skip(f"the record {RECORD.name} is absent: scripts/measure_btg_check.py writes it from the "
                    "BitcoinTemporalGraph archive, which this repository does not hold")
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _summary_problems(where: str, s: dict, values: list[float]) -> list[str]:
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    want = {"mean": mean, "std": std, "min": min(values), "max": max(values)}
    return [f"{where}: {k} is {s[k]}, the per-seed values give {v}" for k, v in want.items() if abs(s[k] - v) > 1e-9]


def record_problems(r: dict) -> list[str]:
    problems = []
    d = r["data"]
    if d["licence"] != "CC BY 4.0" or d["doi"] != "10.6084/m9.figshare.26305093.v3" or "Schnoering" not in d["citation"]:
        problems.append("the record does not carry the licence, the DOI and the credit")
    if "laundering" not in r["not"] or "detector" not in r["not"]:
        problems.append("the record does not say what it is not")
    if sum(d["labels"].values()) != d["labelled_nodes"]:
        problems.append(f"the labels add to {sum(d['labels'].values())}, not {d['labelled_nodes']}")
    if d["task_classes"] != CLASSES or d["task_nodes"] != {c: d["labels"][c] for c in CLASSES}:
        problems.append("the task is not the paper's seven classes with every labelled node of them")
    a = d["addresses_csv"]
    if sum(a["by_category"].values()) + a["rows_without_category"] != a["rows"]:
        problems.append("addresses.csv's categories do not add to its rows")
    h = r["one_hop"]
    if not (h["edges"] == d["edge_rows"] and h["edges_between_two_labelled_nodes"] <= h["edges_with_a_labelled_endpoint"]
            <= h["edges"] and h["self_loops"] <= h["edges"]):
        problems.append("the one-hop edge counts do not nest within the edge rows")
    if {c: v["nodes"] for c, v in h["neighbours_per_class"].items()} != d["task_nodes"]:
        problems.append("the one-hop per-class counts are not the task nodes")
    seeds = r["split"]["seeds"]
    if len(seeds) < 3 or sorted(r["split"]["counts"]) != sorted(str(s) for s in seeds):
        problems.append("fewer than three seeds, or a seed without its split counts")
    total = sum(d["task_nodes"].values())
    for seed, parts in r["split"]["counts"].items():
        for c in CLASSES:
            if sum(parts[p][c] for p in ("train", "validation", "test")) != d["task_nodes"][c]:
                problems.append(f"seed {seed}: the splits of {c} do not add to its nodes")
        for p, share in (("train", 0.4), ("validation", 0.3), ("test", 0.3)):
            if abs(sum(parts[p].values()) - share * total) > 2:
                problems.append(f"seed {seed}: {p} holds {sum(parts[p].values())} of {total}, not {share:.0%}")
    res = r["results"]
    if sorted(res) != sorted(MODELS):
        problems.append(f"the models are {sorted(res)}, not {sorted(MODELS)}")
        return problems
    missing = [m for m in MODELS if [row["seed"] for row in res[m]["per_seed"]] != seeds
               or [row["seed"] for row in res[m]["runs"]] != seeds]
    if missing:
        return problems + [f"{m} is not reported for every seed" for m in missing]
    for m in MODELS:
        rows = res[m]["per_seed"]
        for row in rows:
            per = [row["per_class_f1"][c] for c in CLASSES]
            if not all(0.0 <= v <= 1.0 for v in [*per, row["macro_f1"], row["accuracy"]]):
                problems.append(f"{m} seed {row['seed']}: a score outside [0, 1]")
            if abs(row["macro_f1"] - sum(per) / len(per)) > 1e-9:
                problems.append(f"{m} seed {row['seed']}: macro-F1 {row['macro_f1']} is not the mean of its classes")
        problems += _summary_problems(f"{m} macro-F1", res[m]["macro_f1"], [row["macro_f1"] for row in rows])
        for c in CLASSES:
            problems += _summary_problems(f"{m} {c}", res[m]["per_class_f1"][c], [row["per_class_f1"][c] for row in rows])
    for base, key in (("mlp_features", "gnn_minus_mlp_macro_f1"), ("hgb_features", "gnn_minus_hgb_macro_f1")):
        want = [g["macro_f1"] - b["macro_f1"] for g, b in zip(res["gnn_one_hop"]["per_seed"], res[base]["per_seed"],
                                                               strict=True)]
        got = r[key]
        if any(abs(x - y) > 1e-12 for x, y in zip(got["per_seed"], want, strict=True)) or \
                got["seeds_gnn_higher"] != sum(w > 0 for w in want):
            problems.append(f"{key} is not the GNN's macro-F1 less {base}'s, seed by seed")
        problems += _summary_problems(key, got, want)
    p = r["paper"]
    if p["macro_f1"] != TABLE_6 or p["url"] != PAPER_URL or "Table 6" not in p["table"]:
        problems.append("the paper's Table 6 is not quoted as printed, with its URL")
    return problems


def note_problems(note: str, r: dict) -> list[str]:
    """The note's figures are the record's at the precision it prints, and its claims are the record's to make."""
    flat = re.sub(r"\s+", " ", note)
    res = r["results"]

    def f1(m: str, c: str) -> float:
        return res[m]["per_class_f1"][c]["mean"]

    want = ["`docs/measurements/btg-check-2026-11-23.json`", PAPER_URL, f"{r['data']['labelled_nodes']:,}",
            f"{sum(r['data']['task_nodes'].values()):,}", f"{len(r['split']['seeds'])} seeds",
            f"average {r['paper']['per_class_mean']['GBC']:.3f}",
            f"F1 {f1('gnn_one_hop', 'RANSOMWARE'):.3f} against {f1('mlp_features', 'RANSOMWARE'):.3f}",
            f"EXCHANGE ({f1('gbc_paper_setting', 'EXCHANGE'):.3f} against",
            f"RANSOMWARE ({f1('gbc_paper_setting', 'RANSOMWARE'):.3f} against"]
    for m in MODELS:
        s = res[m]["macro_f1"]
        want.append(f"$`{s['mean']:.3f} \\pm {s['std']:.3f}`$")
    want.append(f"{r['gnn_minus_mlp_macro_f1']['mean']:+.3f}")
    problems = [f"the note does not quote {w!r}" for w in want if w not in flat]
    seeds = len(r["split"]["seeds"])
    if r["gnn_minus_mlp_macro_f1"]["seeds_gnn_higher"] != seeds or "on all five seeds" not in flat or seeds != 5:
        problems.append("the note says the neighbours add on all five seeds, and the record does not")
    if r["gnn_minus_hgb_macro_f1"]["seeds_gnn_higher"] != 0 or "ahead of the one-hop GNN on every seed" not in flat:
        problems.append("the note says the booster leads the GNN on every seed, and the record does not")
    gains = {c: f1("gnn_one_hop", c) - f1("mlp_features", c) for c in CLASSES}
    if max(gains, key=gains.get) != "RANSOMWARE" or "most on RANSOMWARE" not in flat:
        problems.append("the note says the neighbours add most on RANSOMWARE, and the record does not")
    close = sum(abs(f1("gbc_paper_setting", c) - p) <= 0.05 for c, p in zip(CLASSES, r["paper"]["per_class_f1"]["GBC"],
                                                                         strict=True))
    if close != 5 or "within 0.05 of the paper's GBC on five classes" not in flat:
        problems.append(f"the rerun is within 0.05 of the paper's GBC on {close} classes, not the note's five")
    return problems


def test_the_record_is_what_it_says() -> None:
    r = _record()
    assert record_problems(r) == []
    assert f'ARCHIVE_SHA256 = "{r["data"]["archive"]["sha256"]}"' in SCRIPT.read_text(encoding="utf-8")
    assert r["data"]["archive"]["verified_on_both_passes"] and r["data"]["addresses_csv"]["by_category_equals_paper_table_4"]
    env = r["environment"]
    assert env["script_as_committed"] and env["fresh_passes"] and env["git"], "not a fresh run of the committed script"


def test_the_record_names_no_address_and_no_node() -> None:
    r = _record()
    text = RECORD.read_text(encoding="utf-8")
    assert not _ADDRESS.findall(text), "the record holds something shaped like a Bitcoin address"

    def long_lists(x, path="") -> list[str]:
        if isinstance(x, dict):
            return [p for k, v in x.items() for p in long_lists(v, f"{path}.{k}")]
        if isinstance(x, list):
            return ([path] if len(x) > 64 else []) + [p for i, v in enumerate(x) for p in long_lists(v, f"{path}[{i}]")]
        return []

    assert long_lists(r) == [], "a list long enough to be node data"


def test_the_note_quotes_the_record() -> None:
    r = _record()
    assert note_problems(NOTE.read_text(encoding="utf-8"), r) == []


def test_the_checks_refuse_a_missing_seed_a_wrong_mean_and_a_rounded_note() -> None:
    r = _record()
    missing = json.loads(json.dumps(r))
    missing["results"]["gnn_one_hop"]["per_seed"].pop()
    assert record_problems(missing)
    wrong = json.loads(json.dumps(r))
    wrong["results"]["hgb_features"]["per_seed"][0]["macro_f1"] += 0.01
    assert record_problems(wrong)
    s = r["results"]["gnn_one_hop"]["macro_f1"]
    note = NOTE.read_text(encoding="utf-8")
    rounded = note.replace(f"{s['mean']:.3f} \\pm", f"{s['mean'] + 0.05:.3f} \\pm")
    assert note_problems(rounded, r)
