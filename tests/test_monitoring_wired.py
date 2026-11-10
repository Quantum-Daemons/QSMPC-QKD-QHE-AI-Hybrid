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
    exporter.record_benchmark("MPC-FHE", 1285.63, "tenseal-ckks-cpu + mpyc-shamir")
    assert any('qsh_benchmark_latency_ms_count{backend="tenseal-ckks-cpu",path="MPC-FHE"}' in ln for ln in scrape())


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


def test_latency_buckets_are_in_milliseconds():
    """Round 86, found on the full stack's Grafana walk: the latencies are milliseconds and the default buckets are seconds,
    so every stage above 10 ms fell into +Inf and the dashboard's p95 read exactly 10. A 557 ms observation must land in a
    finite bucket, and the top finite bound must cover a ten-minute stage."""
    finite = [b for b in exporter.LATENCY_BUCKETS_MS if b != float("inf")]
    assert min(finite) <= 1 and max(finite) >= 600_000
    exporter.record_benchmark("MPC-FHE", 557.0, "bucket-probe")
    lines = [ln for ln in scrape()
             if ln.startswith('qsh_benchmark_latency_ms_bucket{backend="bucket-probe",le="1000.0",path="MPC-FHE"}')]
    assert lines and float(lines[0].split()[-1]) >= 1, "a 557 ms run is not counted at or below 1,000 ms"


def test_every_stage_series_exists_before_the_first_run():
    """So the first run after the API starts is an increase `rate()` can see."""
    text = "\n".join(scrape())
    for stage in exporter.STAGES:
        assert f'qsh_stage_latency_ms_count{{stage="{stage}"}}' in text, stage


def test_every_stage_a_pipeline_can_run_is_pre_initialised():
    """The list above is a literal, so a stage class added later would bring the first-run gap back for that stage alone.
    Walk every Stage subclass the use cases define (conditional ones such as qkd_optimize included) and require each name."""
    import importlib
    import pkgutil

    import qsh.usecases as usecases
    from qsh.orchestration.stages import Stage

    for mod in pkgutil.iter_modules(usecases.__path__):
        importlib.import_module(f"qsh.usecases.{mod.name}")

    def walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from walk(sub)

    names = {sub.name for sub in walk(Stage) if isinstance(getattr(sub, "name", None), str)}
    assert len(names) >= 14, f"the walk found only {sorted(names)}"
    missing = sorted(names - set(exporter.STAGES))
    assert not missing, f"stages without a pre-initialised latency series: {missing}"


def test_an_extrapolated_benchmark_row_is_not_recorded_as_a_latency():
    """The 'large' tier's row is projected from 'medium'. A latency histogram cannot carry that flag, so the row stays out."""
    from qsh.benchmark.metrics import BenchResult
    from qsh.benchmark.runner import record_measured_rows

    record_measured_rows([
        BenchResult(path="MPC-FHE", backend="probe-measured + mpyc-shamir", workload="linear", latency_ms=420.0),
        BenchResult(path="MPC-FHE", backend="probe-projected (projected)", workload="linear", latency_ms=9_000.0,
                    projected=True),
    ])
    text = "\n".join(scrape())
    assert 'backend="probe-measured"' in text
    assert 'backend="probe-projected"' not in text, "a projected latency reached the histogram"


def test_qber_reads_not_a_number_until_a_qkd_stage_reports_one():
    """A gauge starts at 0, and the dashboard read "QBER 0" before any key agreement had run. A fresh interpreter, because
    other tests in the session may have run a pipeline that set it."""
    import subprocess

    src = Path(__file__).resolve().parents[1] / "src"
    out = subprocess.run([sys.executable, "-c", "from qsh.monitoring import exporter; print(exporter.render()[0].decode())"],
                         capture_output=True, text=True, check=True, env={"PYTHONPATH": str(src), "PATH": ""}).stdout
    line = next(ln for ln in out.splitlines() if ln.startswith("qsh_qber "))
    assert line.split()[-1].lower() == "nan", line


def test_the_dashboard_hides_what_did_not_run_and_splits_the_benchmark_by_backend():
    import json

    board = json.loads((Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "qsh-pipeline.json").read_text())
    exprs = {p["title"]: p["targets"][0]["expr"] for p in board["panels"]}
    p95 = next(e for t, e in exprs.items() if "p95" in t)
    assert p95.rstrip().endswith("> 0"), "the p95 panel lists every pre-initialised stage, the ones that never ran included"
    bench = next(e for t, e in exprs.items() if t.startswith("Benchmark"))
    assert "backend" in bench and "_sum" in bench and "_count" in bench, bench
