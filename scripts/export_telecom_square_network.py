#!/usr/bin/env python3
"""The telecom page's file for its square network (Round 86, batch 12; the owner's choice of 2026-11-28, after the
pre-registered run: the network replaces the linear student on the telecom page's encrypted 5G-NIDD path).

Writes `web/public/data/telecom_nidd/square_network.json`: the exported seed's weights (from the run's work directory,
checked against the SHA-256 the record holds), the parameter sets the criteria offered, the record's figures, and the
2,048 test flows the run's encrypted stages read (P2's test part, drawn as the shipped student's held-out rows were),
raw, with their labels, for the page to scale, encrypt and score as the browser stage did. Not part of the
pre-registration: it changes no figure, and `--check` rebuilds the file and compares.

    QSH_DATA_ROOT=... python scripts/export_telecom_square_network.py --work W [--check]
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
sys.path.insert(0, str(REPO / "scripts" / "train"))

spec = importlib.util.spec_from_file_location("telecom_nidd_square_student",
                                              REPO / "scripts" / "train" / "telecom_nidd_square_student.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

PAGE_FILE = REPO / "web" / "public" / "data" / "telecom_nidd" / "square_network.json"
KEY = "telecom_ids.square_network"
#: The page's set: B, the owner's choice of 2026-11-28 after A2's largest logit error was found to vary from run to
#: run across the page's 1e-3 rule (references/telecom-nidd-square-student-preregistration-addendum-spread.md).
DEFAULT_SET = "B"
ATTRIBUTION = ("Y. Siriwardhana et al., 'Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)', in IEEE "
               "Data Descriptions, doi:10.1109/IEEEDATA.2025.3592888. Dataset: doi:10.23729/e80ac9df-d9fb-47e7-8d0d-"
               "01384a415361, CC BY 4.0.")


def page_file(work: Path) -> dict:
    record = json.loads(T.RECORD.read_text())
    c = record["criteria"]
    if not c.get("adopted"):
        raise SystemExit("the square network was not adopted, so the page does not change")
    weights_bytes = (work / "weights.json").read_bytes()
    if hashlib.sha256(weights_bytes).hexdigest() != record["weights_sha256"]:
        raise SystemExit("the work directory's weights are not the record's exported seed")
    w = json.loads(weights_bytes)
    X, y, classes, cols, _w, rows = T.flows(work)
    raw = X[np.ix_(rows, cols)].astype(np.float64)
    labels = y[rows].astype(int)
    arms, seed = record["arms"]["square"], str(record["export_seed"])
    test = [arms[s]["test"]["flamby_metric"] for s in sorted(arms)]
    offered = c["sets_offered"]
    if DEFAULT_SET not in offered:
        raise SystemExit(f"the page's set {DEFAULT_SET} is not one the criteria offered ({offered})")
    return {
        "_about": ("The telecom page's square network (Round 86, batch 12): references/telecom-nidd-square-student-"
                   "preregistration.md, run as pre-registered (docs/measurements/telecom-nidd-square-student-"
                   "2026-11-28.json), and on the owner's choice of 2026-11-28 in place of the linear student on the "
                   "page's encrypted 5G-NIDD path. Written by scripts/export_telecom_square_network.py."),
        "key": KEY,
        **{k: w[k] for k in ("kind", "n_in", "width", "centre", "scale", "clip", "w1", "b1", "w2", "b2", "classes",
                             "decision", "inputs", "export_seed", "batch_norm")},
        "ckks": {name: w["ckks"][name] for name in offered},
        "default_set": DEFAULT_SET,
        "default_set_note": ("B, the owner's choice of 2026-11-28: under A2 the largest logit error varies from run to "
                             "run and passed the page's 1e-3 rule in two of 27 runs, under B it stayed near 2e-5 "
                             "(references/telecom-nidd-square-student-preregistration-addendum-spread.md)"),
        "record": str(T.RECORD.relative_to(REPO)),
        "figures": {
            "control_test_flamby": record["control"]["test"]["flamby_metric"],
            "test_flamby_mean": round(float(np.mean(test)), 6), "test_flamby_min": min(test),
            "export_seed_test": arms[seed]["test"], "export_seed_epoch": arms[seed]["epoch"],
            "export_bootstrap": [record["export_bootstrap"]["low"], record["export_bootstrap"]["high"]],
            "ckks": {name: {k: s[k] for k in ("max_abs_error", "decisions_differ", "encrypted_accuracy")}
                     for name, s in record["ckks"]["sets"].items()},
            "browser": {name: {"slowest_total_ms": s["slowest_total_ms"],
                               "largest_wasm_memory_bytes": s["largest_wasm_memory_bytes"]}
                        for name, s in record["browser"]["sets"].items()},
            "browser_name": record["browser"]["browser"],
            "test_part_rows": record["data"]["parts"][2],
        },
        "corpus": "5G-NIDD (real traffic from a 5G test network, University of Oulu)",
        "split": ("P2's test part: in each (capture file, class) group in file order, the last 15 percent "
                  "(references/telecom-nidd-clean-protocols-preregistration.md)"),
        "fitNote": "fitted on P2's training part, the epoch kept by validation macro-F1",
        "licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": ATTRIBUTION,
        "modifications": ("The flows published twice under contradictory labels lose their Benign copy; the 30 numeric "
                          "inputs without LEAF's six columns (the re-run's F30); 2,048 flows of P2's test part, each "
                          "class in its share (numpy.random.default_rng(0)), raw, for the page to standardise by the "
                          "training part's moments and clip at 20, as the network's client does."),
        "held_out": {"raw": raw.tolist(), "labels": labels.tolist(), "feature_names": w["inputs"],
                     "class_counts": {classes[k]: int(np.sum(labels == k)) for k in range(len(classes))}},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the committed file; write nothing")
    args = ap.parse_args()
    text = json.dumps(page_file(args.work.resolve()), separators=(",", ":")) + "\n"
    if args.check:
        same = PAGE_FILE.exists() and PAGE_FILE.read_text() == text
        print(f"{PAGE_FILE.relative_to(REPO)} {'is' if same else 'is NOT'} what the run and its work directory give")
        return 0 if same else 1
    PAGE_FILE.write_text(text)
    print(f"wrote {PAGE_FILE.relative_to(REPO)} ({PAGE_FILE.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
