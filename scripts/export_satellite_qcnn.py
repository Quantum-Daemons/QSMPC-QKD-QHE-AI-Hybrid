#!/usr/bin/env python3
"""The satellite QCNN card's data: one file per corpus, from the shipped artifacts and the records.

On the owner's choice of 2026-11-24 (both circuits on the satellite page, OPS-SAT first; the tree network run under
CKKS there). Each `web/public/data/satellite/satellite_qcnn_<corpus>.json` holds the artifact's weights as shipped in
`models/qcnn_satellite_<corpus>/weights.json`, the tree network's stage maps (`qsh.qhe.qcnn_tree.network`), the rows the
page runs (OPS-SAT: all 529 test rows; ESA-ADB: the first 8,192 test windows from 2007-01-01, one CKKS block) with their
labels, and the figures the card quotes, copied from the records with their SHA-256. Named `satellite_qcnn_*` and not
`qcnn_weights*`: the pipeline's QCNN backend is not offered on the satellite page.

    python scripts/export_satellite_qcnn.py            # writes both files
    python scripts/export_satellite_qcnn.py --check    # exits 1 if either is not what this writes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

OUT = REPO / "web" / "public" / "data" / "satellite"
SCORES = REPO / "docs" / "measurements" / "satellite-qcnn-2026-11-24.json"
CKKS = REPO / "docs" / "measurements" / "satellite-qcnn-tree-ckks-2026-11-24.json"
CORPORA = {
    "opssat": {"usecase": "satellite_opssat", "rows": None, "title": "OPS-SAT-AD",
               "rows_about": "all 529 test rows of the benchmark's own partition",
               "attribution": {"dataset": "OPS-SAT-AD", "citation": "Ruszczak, Kotowski, Evans and Nalepa, The OPS-SAT benchmark "
                               "for detecting anomalies in satellite telemetry, Scientific Data 12, 710 (2025), "
                               "doi:10.1038/s41597-025-05035-3; dataset Zenodo 12588359", "licence": "CC-BY-4.0",
                               "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                               "modifications": "eight of nineteen features, the test partition only"}},
    "esa": {"usecase": "satellite_esa", "rows": 8192, "title": "ESA-ADB Mission1",
            "rows_about": "the first 8,192 test windows from 2007-01-01 (one CKKS block)",
            "attribution": {"dataset": "ESA Anomalies Dataset, Mission1", "citation": "Kotowski et al., European Space "
                            "Agency Benchmark for Anomaly Detection in Satellite Telemetry (ESA-ADB); dataset Zenodo 12528696",
                            "licence": "CC-BY-3.0-IGO", "licence_url": "https://creativecommons.org/licenses/by/3.0/igo/",
                            "modifications": "channels 41 to 46 on a 30 s grid, z-scored, cut into 64-step windows, "
                                             "eight statistics a window (scripts/data/esa_qcnn_windows.py)"}},
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(short: str) -> dict:
    from qcnn_variational import load, split_for

    from qsh.qhe import qcnn_tree

    spec = CORPORA[short]
    wpath = REPO / "models" / f"qcnn_satellite_{short}" / "weights.json"
    w = json.loads(wpath.read_text())
    X, y, cols, _ = load(spec["usecase"])
    order, _, vcut = split_for(spec["usecase"], y, 0)
    te = order[vcut:]
    if spec["rows"]:
        te = te[:spec["rows"]]
    net = qcnn_tree.network(w)
    scores = json.loads(SCORES.read_text())[short]
    ckks = next(c for c in json.loads(CKKS.read_text())["cases"] if c["name"] == short)
    onnx_rec = json.loads((REPO / "models" / f"qcnn_satellite_{short}" / "tree.json").read_text())
    pick = lambda d: {k: d[k] for k in ("average_precision", "minority_f1", "balanced_accuracy")}  # noqa: E731
    return {
        "_generated_by": "scripts/export_satellite_qcnn.py; do not edit by hand",
        "corpus": short, "title": spec["title"], "features": cols, "weights": w,
        "tree": {"maps": [m.tolist() for m in net["maps"]], "readout": net["readout"].tolist()},
        "rows_about": spec["rows_about"], "rows": X[te].tolist(), "labels": y[te].astype(int).tolist(),
        "figures": {"qcnn_seed0": pick(scores["qcnn_by_seed"]["0"]), "qcnn_spread": scores["qcnn_spread"],
                    "logistic_regression": pick(scores["logistic_regression_z"]),
                    "classical_surrogate": pick(scores["classical_surrogate"]),
                    "reference_rows": scores.get("reference_rows"), "note": scores.get("note"),
                    "test_rows": scores["rows"]["test"], "test_positive": scores["rows"]["test_positive"],
                    "trainer_rule_minority_f1": scores["qcnn_by_seed"]["0"]["trainer_rule_minority_f1"]},
        "ckks_record": {"rows": ckks["rows"], "blocks": ckks["blocks"], "per_row": ckks["per_row"],
                        "tree_max_abs_readout_error": ckks["tree"]["max_abs_readout_error"],
                        "tree_decisions_differing": ckks["tree"]["decisions_differing"],
                        "pauli_rows": ckks["pauli"]["rows"], "pauli_max_error": ckks["pauli"]["maxError"]},
        "onnx_record": {"sha256": onnx_rec["sha256"], "bytes": onnx_rec["bytes"],
                        "max_abs_readout_difference": onnx_rec["max_abs_readout_difference"],
                        "decisions_differing": onnx_rec["decisions_differing"], "test_rows": onnx_rec["test_rows"]},
        "sources": {"docs/measurements/satellite-qcnn-2026-11-24.json": _sha(SCORES),
                    "docs/measurements/satellite-qcnn-tree-ckks-2026-11-24.json": _sha(CKKS),
                    f"models/qcnn_satellite_{short}/weights.json": _sha(wpath)},
        # The register's row fields (tests/test_r86_every_shipped_data_file_is_licensed.py): the licence, its link, the
        # credit and the note of the changes, at the top level as every shipped row file carries them.
        "licence": spec["attribution"]["licence"], "licence_url": spec["attribution"]["licence_url"],
        "attribution": spec["attribution"]["citation"], "modifications": spec["attribution"]["modifications"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    bad = 0
    for short in CORPORA:
        text = json.dumps(build(short), separators=(",", ":")) + "\n"
        f = OUT / f"satellite_qcnn_{short}.json"
        if args.check:
            if not f.exists() or f.read_text(encoding="utf-8") != text:
                print(f"{f.relative_to(REPO)} is not what scripts/export_satellite_qcnn.py writes", file=sys.stderr)
                bad += 1
        else:
            f.write_text(text, encoding="utf-8")
            print(f"wrote {f.relative_to(REPO)} ({len(text) / 1e3:.0f} kB)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
