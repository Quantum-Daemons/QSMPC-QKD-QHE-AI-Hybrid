#!/usr/bin/env python
"""Add each published evaluation sample's per-row record (`expected.per_row`), computed as the page computes it.

The samples are the PUBLISHED ones: each file is checked against the SHA-256 recorded for its pin before anything is
read, so the record describes exactly the rows the Hub serves, and no sample is regenerated (a regeneration redraws
rows from the raw corpora, and a record of a different draw would describe nothing the page fetched). The record is
`qsh.models.sample_per_row.per_row_record`, the browser's rule in Python; before a file is written, the metric that
record implies is compared with the metric the file already states, and a difference stops the run.

    python scripts/add_per_row_records.py            # report what would be written, write nothing
    python scripts/add_per_row_records.py --write    # write, and demote each rewritten model's pin until republished

After `--write`: `scripts/publish_hf.py --key <key> --public` for each, then `scripts/pin_hub_hashes.py` and
`scripts/export_config.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qsh.models import registry  # noqa: E402
from qsh.models.sample_per_row import implied_metrics, per_row_record  # noqa: E402

HASHES = REPO / "models" / "hub_file_hashes.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the records (default: report only)")
    ap.add_argument("--key", action="append", help="only these model keys (repeatable)")
    ap.add_argument("--replace", action="store_true",
                    help="recompute a record a sample already carries; the sample without it must be the pinned bytes")
    args = ap.parse_args()
    from prep_model_inputs import _invalidate_pin_if_published

    pins = json.loads(HASHES.read_text())
    bad = 0
    for key, rec in sorted(pins.items()):
        if args.key and key not in args.key:
            continue
        directory = REPO / registry.ARTIFACT_DIRS[key]
        path = directory / "eval_sample.json"
        if "eval_sample.json" not in rec["files"] or not path.exists():
            continue
        raw = path.read_bytes()
        sample = json.loads(raw)
        task = sample.get("task", "classification")
        graph_name = sample.get("expected", {}).get("graph")
        if task not in ("classification", "forecast") or not graph_name:
            print(f"{key}: {task} sample with {'graph ' + graph_name if graph_name else 'no graph'} - verified by its own path, skipped")
            continue
        if "per_row" in sample["expected"]:
            if not args.replace:
                print(f"{key}: already carries a per-row record")
                continue
            # Without its record the file must be byte for byte the pinned one, or the record describes other rows.
            del sample["expected"]["per_row"]
            raw = (json.dumps(sample, indent=1) + "\n").encode()
        if hashlib.sha256(raw).hexdigest() != rec["files"]["eval_sample.json"]:
            print(f"{key}: the local sample is not the one pinned at {rec['revision'][:8]} - refusing")
            bad += 1
            continue
        graph = directory / "onnx" / graph_name
        if hashlib.sha256(graph.read_bytes()).hexdigest() != rec["files"].get(f"onnx/{graph_name}"):
            print(f"{key}: the local {graph_name} is not the pinned graph - refusing")
            bad += 1
            continue
        record = per_row_record(sample, graph)
        implied = implied_metrics(sample, record)
        diffs = {k: abs(v - float(sample["expected"][k])) for k, v in implied.items()}
        ok = all(d <= (1e-6 if k == "mae" else 1e-12) for k, d in diffs.items()) and bool(implied)
        near = sum(1 for m in record.get("margin", []) if abs(m) <= record["tolerance"])
        print(f"{key}: {record['rows']} rows from '{record['read_from']}' of {graph_name}; implied "
              + ", ".join(f"{k} {v:.6f} (recorded {float(sample['expected'][k]):.6f})" for k, v in implied.items())
              + (f"; {near} rows within the tolerance of the boundary" if "margin" in record else "")
              + ("" if ok else "  <-- DIFFERS from the recorded figure, not written"))
        if not ok:
            bad += 1
            continue
        if args.write:
            sample["expected"]["per_row"] = record
            path.write_text(json.dumps(sample, indent=1) + "\n")
            _invalidate_pin_if_published(key)
            print(f"    wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KB)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
