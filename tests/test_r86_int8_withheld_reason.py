"""A card with no int8 graph says why (Round 86, fact-check C8).

The ResNet-50 card printed `export.int8_agreement 0.9615` beside `export.int8_bytes 0`. That reads as an int8 graph
scoring 0.96 and then vanishing. Its export report says the graph was built, measured and withheld, because the
confident agreement, 0.9865, missed the 0.99 gate. Tree ensembles and ai.onnx.ml linear models have no int8 graph for
another reason. The record keeps the figures and the export report keeps the reason, so the config carries the
report's reason to the card for every model whose record serves no int8 graph.
"""

from __future__ import annotations

import json
from pathlib import Path

from qsh.config import frontend_config

REPO = Path(__file__).resolve().parents[1]


def _models() -> list[dict]:
    out: list[dict] = []

    def walk(o: object) -> None:
        if isinstance(o, dict):
            if "key" in o and "int8_withheld_reason" in o:
                out.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(frontend_config())
    return out


def test_every_model_without_an_int8_graph_says_why() -> None:
    metrics = json.loads((REPO / "models" / "metrics.json").read_text())
    models = {m["key"]: m for m in _models()}
    assert models, "the config carries no models"
    missing = [k for k, rec in metrics.items()
               if isinstance(rec, dict) and (rec.get("export") or {}).get("int8_bytes") == 0
               and k in models and not models[k]["int8_withheld_reason"]]
    assert not missing, f"no int8 graph and no reason on the card: {missing}"
    assert "withheld" in models["medical_fl.resnet50"]["int8_withheld_reason"]


def test_the_card_prints_the_reason() -> None:
    panel = (REPO / "web" / "src" / "components" / "ModelPanel.tsx").read_text()
    assert "{spec.int8_withheld_reason}" in panel and "No int8 graph:" in panel
