"""What IBM's Multi-GNN read for a scored transfer, counted by hand on a small graph (Round 86).

`scripts/train/ibm_multignn/neighbourhood.py` writes these counts into the IBM models' browser samples, and the finance
page re-derives them from the sample (`web/src/lib/graphNeighbourhood.ts`, held to the same graph by
`web/scripts/check_graph_neighbourhood.mjs`). The graph below has each case the definition names: a transfer the sample
holds twice, a self-transfer, one later and within the scored transfer's horizon, one in its batch of 2,048 but past the
reach, two in a later batch, one exactly the reach later (within), one two hops out, and one three hops out that two
rounds of message passing do not reach. The reach the export reads off the fit rows is held to the finance page's own
`finance_aml_fan.batch_reach` on the same rows.
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
    got = nb.seed_fields(f["edge_index"], f["edge_position"], f["edge_seconds"], f["seed_at"], f["reach_s"])
    assert got == f["expected"]


def test_the_fixture_has_each_case() -> None:
    f = _fixture()
    pos, src, dst = f["edge_position"], f["edge_index"][0], f["edge_index"][1]
    assert len(pos) != len(set(pos)), "a transfer held twice"
    assert any(a == b for a, b in zip(src, dst)), "a self-transfer"
    a, b = f["expected"]
    assert a["later_within_horizon"] >= 1 and a["past_horizon"] >= 3 and a["read"] < len(set(pos)) - 1
    assert b["past_horizon"] == 0 and b["furthest_later_s"] == f["reach_s"], "a transfer exactly the reach later is within"


def test_a_seed_with_nothing_later_reads_zero_ahead() -> None:
    got = nb.seed_fields([[0, 1], [1, 2]], [10, 3], [120.0, 60.0], [0], 1020)
    assert got[0]["later_within_horizon"] == 0 and got[0]["past_horizon"] == 0 and got[0]["furthest_later_s"] == 0.0
    assert got[0]["read"] == 1 and got[0]["payee_out"] == 1


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError):
        nb.seed_fields([[0], [1]], [1, 2], [0.0], [0], 1020)


def test_the_reach_is_read_off_the_first_rows() -> None:
    # Three batches of 2 rows; the first two (the "fit" rows) see at most 120 s to their batch's end, the third 600 s.
    secs = [0, 120, 180, 240, 300, 900]
    assert nb.fit_reach(secs, n_fit=4, rows=2) == 120
    assert nb.fit_reach(secs, n_fit=6, rows=2) == 600


def test_the_reach_is_the_pages_batch_reach() -> None:
    import numpy as np
    import pandas as pd
    spec = importlib.util.spec_from_file_location("finance_aml_fan", REPO / "scripts" / "train" / "finance_aml_fan.py")
    fan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fan)
    rng = np.random.default_rng(2026)
    minutes = np.cumsum(rng.choice([0, 0, 0, 1, 2, 60], size=9000))  # bursts within a minute, and quiet stretches
    df = pd.DataFrame({"timestamp": pd.Timestamp("2022-09-01") + pd.to_timedelta(minutes, unit="m")})
    for n_fit in (2048, 5000, 9000):
        assert nb.fit_reach(minutes * 60.0, n_fit) == 60 * fan.batch_reach(df, n_fit)
