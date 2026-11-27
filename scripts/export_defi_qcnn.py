#!/usr/bin/env python3
"""The DeFi page's QCNN card's data, once both of its pre-registered criteria hold (Round 86, batch 8).

`references/defi-qcnn-aave-v3-preregistration.md`: if the arm-1 seed best on validation has a test average precision of
at least twice the prevalence (criterion 1) and the page's clear contraction equals the state vector on every row the
page ships (criterion 2, `scripts/check_defi_qcnn_export.py`), the page shows that circuit as trained and not
encrypted, beside its comparators. This writes `web/public/data/defi/defi_qcnn.json`: the seed's weights as trained, the
tree network's stage maps (`qsh.qhe.qcnn_tree.network`), a sample of test-window rows (each its eight standardised
inputs and its label, nothing else: no account, transaction or other identifier), the sample's average precision as the
trainer's circuit gives it, which the tab's re-score must reproduce, and the record's figures. It refuses to write when
either criterion fails, and says which.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/export_defi_qcnn.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SITE = REPO / "web" / "public" / "data" / "defi" / "defi_qcnn.json"
SAMPLE_ROWS, SAMPLE_SEED = 4096, 0


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build() -> dict:
    from sklearn.metrics import average_precision_score as ap

    from qsh.qhe import qcnn_tree

    C = _module("defi_qcnn_comparison", REPO / "scripts" / "train" / "defi_qcnn_comparison.py")
    C.check_prereg()
    record = json.loads(C.RECORD.read_text())
    crit = record["criteria"]
    if not crit["met"]:
        raise SystemExit(f"the criteria do not hold (signal {crit['1_signal']}, export {crit['2_export_check']}): "
                         "nothing is added to the page")
    ship = record["arms"]["qcnn8"]["shipped_seed"]
    wpath = C.WORK / "qcnn8" / f"seed{ship['seed']}" / "weights.json"
    w = json.loads(wpath.read_text())
    d = C.load_inputs()
    te = d["te"]
    pick = np.sort(np.random.default_rng(SAMPLE_SEED).choice(len(te), size=SAMPLE_ROWS, replace=False))
    # The rows as the page ships them (nine decimals), so the figure the tab must reproduce is computed on the same numbers.
    rows = np.round(d["X"][te][pick].astype(np.float64), 9)
    labels = d["y"][te][pick].astype(int)
    readout = qcnn_tree.readout(w, rows)
    logits = readout * float(w["scale"]) + float(w["bias"])
    net = qcnn_tree.network(w)
    site_cmp = json.loads(C.SITE.read_text())
    return {
        "_generated_by": "scripts/export_defi_qcnn.py; do not edit by hand",
        "preregistration": record["preregistration"]["file"], "record": str(C.RECORD.relative_to(REPO)),
        "task": site_cmp["task"], "features": d["meta"]["features"], "seed": ship["seed"],
        "weights": w, "tree": {"maps": [m.tolist() for m in net["maps"]], "readout": net["readout"].tolist()},
        "rows_about": (f"{SAMPLE_ROWS:,} of the test window's {len(te):,} borrows, drawn uniformly at seed {SAMPLE_SEED}, "
                       "each its eight standardised inputs and its label"),
        "rows": [[float(v) for v in r] for r in rows], "labels": labels.tolist(),
        "sample": {"rows": SAMPLE_ROWS, "positive": int(labels.sum()),
                   # Unrounded: the tab's re-score is held to it within 1e-9.
                   "average_precision": float(ap(labels, logits))},
        "figures": {"test_ap": ship["test_ap"], "test_auc": ship["test_auc"], "minority_f1": ship["minority_f1"],
                    "prevalence": record["test_prevalence"], "twice_prevalence": record["twice_prevalence"],
                    "test_rows": int(len(te)), "test_positive": int(d["y"][te].sum())},
        "export_check": {k: crit["2_export_check"][k] for k in ("test_rows", "page_vs_state_vector", "criterion_met")},
        "sources": {str(C.RECORD.relative_to(REPO)): hashlib.sha256(C.RECORD.read_bytes()).hexdigest()},
        "licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": ("Spadea, F. and Seneviratne, O., From Risk to Rescue: An Agentic Survival Analysis Framework for "
                        "Liquidation Prevention Dataset, Figshare, doi:10.6084/m9.figshare.31058368.v1, CC BY 4.0."),
        "modifications": (f"{SAMPLE_ROWS:,} test-window borrows of the archive's Borrow-to-Liquidated table, each reduced to "
                          "eight of its numeric inputs, standardised by the fit window and clipped at 10, and a label "
                          "computed here (liquidated within 30 days); no account, transaction or identifier"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = json.dumps(build(), separators=(",", ":")) + "\n"
    if args.check:
        if not SITE.exists() or SITE.read_text() != text:
            print(f"{SITE.relative_to(REPO)} is not what scripts/export_defi_qcnn.py writes", file=sys.stderr)
            return 1
        return 0
    SITE.parent.mkdir(parents=True, exist_ok=True)
    SITE.write_text(text)
    print(f"wrote {SITE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
