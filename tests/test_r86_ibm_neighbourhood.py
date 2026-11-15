"""What IBM's Multi-GNN read for a scored transfer, counted by hand on a small graph (Round 86).

`scripts/train/ibm_multignn/neighbourhood.py` writes these counts into the IBM models' browser samples, and the finance
page re-derives them from the sample (`web/src/lib/graphNeighbourhood.ts`, held to the same graph by
`web/scripts/check_graph_neighbourhood.mjs`). The graph below has each case the definition names: a transfer the sample
holds twice, a self-transfer, one later in the scored transfer's batch of 2,048, two beyond it, one two hops out, and
one three hops out that two rounds of message passing do not reach.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "web" / "scripts" / "vectors" / "graph_neighbourhood.json"

_spec = importlib.util.spec_from_file_location("ibm_neighbourhood", REPO / "scripts" / "train" / "ibm_multignn" / "neighbourhood.py")
nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_the_hand_counts_hold() -> None:
    f = _fixture()
    got = nb.seed_fields(f["edge_index"], f["edge_position"], f["edge_seconds"], f["seed_at"])
    assert got == f["expected"]


def test_the_fixture_has_each_case() -> None:
    f = _fixture()
    pos, src, dst = f["edge_position"], f["edge_index"][0], f["edge_index"][1]
    assert len(pos) != len(set(pos)), "a transfer held twice"
    assert any(a == b for a, b in zip(src, dst)), "a self-transfer"
    e = f["expected"][0]
    assert e["later_in_batch"] >= 1 and e["beyond_batch"] >= 2 and e["read"] < len(set(pos)) - 1


def test_a_seed_with_nothing_later_reads_zero_ahead() -> None:
    got = nb.seed_fields([[0, 1], [1, 2]], [10, 3], [120.0, 60.0], [0])
    assert got[0]["later_in_batch"] == 0 and got[0]["beyond_batch"] == 0 and got[0]["furthest_later_s"] == 0.0
    assert got[0]["read"] == 1 and got[0]["payee_out"] == 1


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError):
        nb.seed_fields([[0], [1]], [1, 2], [0.0], [0])
