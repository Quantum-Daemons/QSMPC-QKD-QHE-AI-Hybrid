"""The network-bench figures on /validations are the recorded ones.

WHY THIS EXISTS

Round 64 measured the secure argmax under emp-wasm and Polytune between two tabs over netem-shaped
WebSocket relays. The record is `prototypes/net-bench/results/` (both parties' runs and the relays'
TCP counters) and `summary.json` derived from it by `analyze.py`. The Validations page cannot import
from prototypes/ - a separate gate forbids web/ referring to it, so that nothing measured is shipped by
accident - so the page reads a trimmed COPY, `web/src/lib/mpcNetBench.json`.

A copy can drift from what it copies. This regenerates the copy from the record with the same function
that wrote it and requires the two to be equal, and checks the record's own invariant that every run
returned the right index, so the page can never show a figure the measurement did not produce.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "prototypes" / "net-bench"
WEB_COPY = REPO / "web" / "src" / "lib" / "mpcNetBench.json"


def _analyze():
    spec = importlib.util.spec_from_file_location("net_bench_analyze", BENCH / "analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_page_copy_equals_the_record(tmp_path: Path) -> None:
    regenerated = tmp_path / "copy.json"
    _analyze().web_copy(BENCH / "results" / "summary.json", regenerated)
    assert json.loads(WEB_COPY.read_text()) == json.loads(regenerated.read_text()), (
        "web/src/lib/mpcNetBench.json differs from what prototypes/net-bench/analyze.py --web derives "
        "from results/summary.json; regenerate it rather than editing it")


def test_every_recorded_run_was_correct() -> None:
    summary = json.loads((BENCH / "results" / "summary.json").read_text())
    wrong = [(c["tag"], name) for c in summary["configs"] for name, e in c["engines"].items()
             if e["correct"] != e["runs"]]
    assert not wrong, f"recorded runs with a wrong index: {wrong}"
    assert len(summary["configs"]) == 20, "the grid has 20 configurations (18 at 2 classes, 2 at 10)"
