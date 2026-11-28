"""Every published graph on this machine is the file its pin names (Round 86, batch 12).

On 2026-11-28 two local fp32 graphs were found not to be the files the shipped config pins (medical ResNet-18's
`onnx/model.onnx` and the satellite forecaster student's), while the per-row re-score test passed: it checks only the
graph each sample's record names, ResNet-18's int8 build among them, and the page reads neither stale file. A check run
on a stale local copy measures a model nobody published, so every graph present here, or in the checkout
`QSH_PUBLISHED_GRAPHS` names, is held to its pin. The graphs are gitignored: where none is present (CI), nothing is
compared, and `QSH_REQUIRE_PUBLISHED_GRAPHS=1` turns that into a failure, as the re-score test's does.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _pins() -> list[tuple[str, str, str]]:
    """(key, models directory, onnx file) -> pinned SHA-256, from the shipped config and the registry's artefact map."""
    cfg = json.loads((REPO / "web" / "src" / "lib" / "config.defaults.json").read_text())
    dirs = dict(re.findall(r'"([a-z_]+\.[a-z_0-9]+)": "models/([a-z_0-9]+)"', (REPO / "src" / "qsh" / "models" / "registry.py").read_text()))
    out = []

    def walk(o) -> None:
        if isinstance(o, dict):
            if "file_sha256" in o and o.get("key") in dirs:
                for f, h in o["file_sha256"].items():
                    if f.startswith("onnx/"):
                        out.append((o["key"], dirs[o["key"]], f, h))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(cfg)
    return out


def test_every_local_graph_is_its_pinned_file() -> None:
    stores = [REPO] + ([Path(os.environ["QSH_PUBLISHED_GRAPHS"])] if os.environ.get("QSH_PUBLISHED_GRAPHS") else [])
    compared, wrong = 0, []
    for key, d, f, pin in _pins():
        for store in stores:
            p = store / "models" / d / f
            if p.exists():
                compared += 1
                if hashlib.sha256(p.read_bytes()).hexdigest() != pin:
                    wrong.append(f"{key}: {p} is not the pinned {f} ({pin[:12]})")
                break
    assert not wrong, wrong
    if compared == 0:
        msg = "no published graph is on this machine, so none was compared"
        if os.environ.get("QSH_REQUIRE_PUBLISHED_GRAPHS") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
