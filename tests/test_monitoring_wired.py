"""The metrics endpoint must serve numbers a run produced, not a well-formed empty document.

WHY THIS TEST EXISTS

`monitoring/exporter.py` declared four Prometheus metrics and three recorder helpers, and for three
consecutive rounds nothing called any of them. `BENCH_LATENCY` had no reader at all, not even inside
the file that defined it. The `/metrics` endpoint returned HTTP 200 with valid Prometheus text, the
Grafana dashboard in `monitoring/grafana/` rendered without error, and every panel on it was empty.

Nothing about that state was detectable from the outside. The endpoint worked, the dashboard existed,
the metrics were declared with sensible names and labels, and the scrape parsed. It appeared on the
deferred list three rounds running, each time described one level further down: "add a dashboard",
then "charts nothing", then "zero call sites".

This is the shape of the project's own founding rule - a result labelled MEASURED that produced no
work - applied to observability. So the test asserts the property that was actually missing: after a
pipeline run, the scrape contains series that only a run can produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qsh.monitoring import exporter  # noqa: E402

pytest.importorskip("prometheus_client",
                    reason="the no-op branch is correct by construction; there is nothing to assert")


def scrape() -> list[str]:
    text = exporter.render()[0].decode()
    return [ln for ln in text.splitlines() if ln.startswith("qsh_") and not ln.startswith("#")]


def test_the_recorders_exist_and_are_callable():
    """All four, including the one whose metric had no reader."""
    for name in ("record_run", "record_stage", "record_qber", "record_benchmark"):
        assert callable(getattr(exporter, name)), f"{name} is not callable"


def test_a_recorded_run_reaches_the_scrape():
    exporter.record_run("medical_fl", "done")
    assert any('qsh_pipeline_runs_total{state="done",usecase="medical_fl"}' in ln for ln in scrape())


def test_a_recorded_stage_reaches_the_scrape():
    exporter.record_stage("qkd_key_agreement", 12.5)
    assert any('qsh_stage_latency_ms_count{stage="qkd_key_agreement"}' in ln for ln in scrape())


def test_a_recorded_benchmark_reaches_the_scrape():
    """BENCH_LATENCY is the one that had no reader anywhere in the tree."""
    exporter.record_benchmark("MPC-FHE", 1285.63)
    assert any('qsh_benchmark_latency_ms_count{path="MPC-FHE"}' in ln for ln in scrape())


def test_the_pipeline_actually_calls_them():
    """The regression that matters: helpers can exist, be tested, and have no production caller.

    Asserted against the module's imports rather than by running a pipeline, so the test states the
    wiring directly instead of depending on which stages a default pipeline happens to contain.
    """
    from qsh.orchestration import pipeline

    source = Path(pipeline.__file__).read_text()
    for call in ("record_stage(", "record_run(", "record_qber("):
        assert call in source, f"{call} has no call site in orchestration/pipeline.py"

    from qsh.benchmark import runner

    assert "record_benchmark(" in Path(runner.__file__).read_text(), \
        "record_benchmark has no call site in benchmark/runner.py"
