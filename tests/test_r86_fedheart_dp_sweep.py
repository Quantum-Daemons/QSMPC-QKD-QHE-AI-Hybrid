"""The medical DP card's band is the page's own code over many draws, and the card says what it is (Round 86).

`scripts/measure_fedheart_dp_sweep.py` runs `fedHeartDp` (web/src/lib/clientFedHeart.ts) 20 times at the page's seed with
seeded noise and records each epsilon's spread; `DpPrivacyUtility` draws it beside the run's own row. Held here:

1. two of the recorded draws are what the page's code gives again, to the last digit, so the record is the code's and
   not a figure typed in;
2. the card's file carries the record's figures, each percentile band contains its median, and the no-noise row, which
   draws no noise, is the same in every draw;
3. the card reads that file and is drawn under the DP table.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
RECORD = REPO / "docs" / "measurements" / "fedheart-dp-sweep-2026-11-14.json"
CARD = WEB / "public" / "data" / "medical" / "fedheart_dp_sweep.json"


def _script():
    spec = importlib.util.spec_from_file_location("sweep", REPO / "scripts" / "measure_fedheart_dp_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_two_recorded_draws_are_what_the_code_gives_again(tmp_path: Path) -> None:
    if not (WEB / "node_modules" / ".bin" / "esbuild").exists() or shutil.which("node") is None:
        pytest.skip("node or esbuild absent, so the page's code cannot run here. This is a SKIP, not a pass.")
    mod = _script()
    record = json.loads(RECORD.read_text())
    (tmp_path / "h.mjs").write_text(mod.HARNESS)
    r = subprocess.run(["node", str(tmp_path / "h.mjs"), str(WEB), "2", record["seeded_label"], str(tmp_path / "o.json")],
                       cwd=WEB, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    again = [[x["mean"] for x in run] for run in json.loads((tmp_path / "o.json").read_text())["runs"]]
    assert again == record["per_draw"][:2]


def test_the_card_carries_the_record_and_its_bands_hold_their_medians() -> None:
    record, card = json.loads(RECORD.read_text()), json.loads(CARD.read_text())
    assert card["rows"] == record["rows"] and card["draws"] == record["draws"] == len(record["per_draw"]) == 20
    per = np.array(record["per_draw"])
    for k, row in enumerate(record["rows"]):
        assert row["p10"] <= row["p50"] <= row["p90"], row
        assert abs(row["mean"] - per[:, k].mean()) < 1e-6 and abs(row["p50"] - np.percentile(per[:, k], 50)) < 1e-6
        if row["epsilon"] is None:
            assert row["sd"] == 0 and len(set(per[:, k])) == 1, "the no-noise row draws no noise"
        else:
            assert row["sd"] > 0, "a noised row varies across draws"


def test_the_card_reads_the_file_and_sits_under_the_dp_table() -> None:
    comp = (WEB / "src" / "components" / "DpPrivacyUtility.tsx").read_text()
    assert "data/medical/fedheart_dp_sweep.json" in comp
    panel = (WEB / "src" / "components" / "FedHeartPanel.tsx").read_text()
    table = panel[panel.index("function DpTable"):]
    assert "<DpPrivacyUtility dp={dp} />" in table[:table.index("\n}\n")]
