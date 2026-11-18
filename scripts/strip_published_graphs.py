#!/usr/bin/env python3
"""Drop the node metadata from published models' ONNX graphs and re-record their sizes (Round 86, 2026-11-18).

torch.export (torch 2.9 and later) writes each node's Python stack trace into the node's metadata, with the exporting
machine's absolute file paths, and seven published graphs carried them (`qsh.models.export.strip_node_metadata`, which
the exporter now applies itself). For each model given, this rewrites every graph in its artefact folder's `onnx/`
in place, after checking the rewritten graph against the original two ways: ONNX Runtime gives the same outputs bit
for bit on random inputs, and the graph is byte-identical once the original's node metadata is set aside. It then
writes the new sizes where the export recorded them, `export_report.json` (and `features_report.json`, the chest X-ray
model's features graph) and the record's `export` block in `models/metrics.json`, with a note of what was dropped.

The graphs are git-ignored and live in the folder they were published from. Publish them afterwards with
`scripts/publish_hf.py`, which pins the new revision, then run `scripts/pin_hub_hashes.py` and `scripts/export_config.py`.

    python scripts/strip_published_graphs.py --key medical_fl.resnet18 --key medical_fl.resnet50 --key medical_fl.cxr_fedavg
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import registry, train  # noqa: E402
from qsh.models.export import strip_node_metadata  # noqa: E402

#: The report files an export writes, each naming its graphs in `fp32_path` and `int8_path`.
REPORTS = ("export_report.json", "features_report.json")


def same_graph(original: Path, stripped: Path) -> tuple[bool, bool]:
    """(outputs bit-identical on random inputs, graph byte-identical with the original's node metadata set aside)."""
    import onnx
    import onnxruntime as ort

    rng = np.random.default_rng(2026)
    s0 = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(str(stripped), providers=["CPUExecutionProvider"])
    feeds = {i.name: rng.standard_normal([d if isinstance(d, int) else 4 for d in i.shape]).astype(np.float32)
             for i in s0.get_inputs()}
    outputs = all(np.array_equal(a, b) for a, b in zip(s0.run(None, feeds), s1.run(None, feeds), strict=True))
    g0, g1 = onnx.load(str(original)), onnx.load(str(stripped))
    for node in g0.graph.node:
        del node.metadata_props[:]
        node.doc_string = ""
    return outputs, g0.SerializeToString() == g1.SerializeToString()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="append", required=True, help="a registry key; repeat for several")
    a = ap.parse_args()
    records = json.loads((REPO / "models" / "metrics.json").read_text())
    for key in a.key:
        folder = REPO / registry.ARTIFACT_DIRS[key]
        graphs = sorted((folder / "onnx").glob("*.onnx"))
        if not graphs:
            raise SystemExit(f"{key}: no graph in {folder / 'onnx'}; put the published files there first")
        sizes, dropped = {}, {}
        for g in graphs:
            with tempfile.TemporaryDirectory(prefix="qsh-strip-") as tmp:
                copy = Path(tmp) / g.name
                shutil.copyfile(g, copy)
                n = strip_node_metadata(copy)
                if n:
                    outputs, graph = same_graph(g, copy)
                    if not (outputs and graph):
                        raise SystemExit(f"{g}: outputs identical {outputs}, graph identical {graph}; left unchanged")
                    shutil.copyfile(copy, g)
            rel = str(g.relative_to(folder))
            sizes[rel], dropped[rel] = g.stat().st_size, n
            print(f"  {key} {rel}: {n} metadata entries dropped, {sizes[rel]:,} bytes")
        for name in REPORTS:
            path = folder / name
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            for field, which in (("fp32_bytes", "fp32_path"), ("int8_bytes", "int8_path")):
                if report.get(which) in sizes:
                    report[field] = sizes[report[which]]
            path.write_text(json.dumps(report, indent=2) + "\n")
        record = records.get(key)
        export = (record or {}).get("export")
        if export is not None:
            report = json.loads((folder / "export_report.json").read_text())
            export["fp32_bytes"], export["int8_bytes"] = report["fp32_bytes"], report["int8_bytes"]
            export["node_metadata_dropped"] = {
                "date": "2026-11-18", "git_sha": train.git_sha(), "entries": dropped,
                "check": "ONNX Runtime outputs bit-identical on random inputs; the graph otherwise byte-identical",
                "why": "torch.export had written each node's stack trace, with the exporting machine's file paths"}
            train.record_metrics(key, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
