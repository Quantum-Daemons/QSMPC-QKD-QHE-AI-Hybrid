"""Prometheus metrics for the orchestration layer.

Exposes counters/histograms the local Grafana dashboards chart: pipeline runs,
per-stage latency, QBER, and benchmark latency. Mount via the FastAPI app's
``/metrics`` endpoint (the API can call :func:`metrics_asgi_app`). Kept optional so
the core library has no hard dependency on prometheus-client.
"""

from __future__ import annotations

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False


if _HAVE:
    PIPELINE_RUNS = Counter("qsh_pipeline_runs_total", "Pipeline runs", ["usecase", "state"])
    STAGE_LATENCY = Histogram("qsh_stage_latency_ms", "Stage latency (ms)", ["stage"])
    QBER = Gauge("qsh_qber", "Last QBER observed")
    BENCH_LATENCY = Histogram("qsh_benchmark_latency_ms", "Benchmark latency (ms)", ["path"])

    def record_run(usecase: str, state: str) -> None:
        PIPELINE_RUNS.labels(usecase, state).inc()

    def record_stage(stage: str, ms: float) -> None:
        STAGE_LATENCY.labels(stage).observe(ms)

    def record_qber(value: float) -> None:
        QBER.set(value)

    def record_benchmark(path: str, ms: float) -> None:
        BENCH_LATENCY.labels(path).observe(ms)

    def render() -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST

else:  # graceful no-ops
    def record_run(*_a, **_k) -> None: ...
    def record_stage(*_a, **_k) -> None: ...
    def record_qber(*_a, **_k) -> None: ...
    def record_benchmark(*_a, **_k) -> None: ...
    def render() -> tuple[bytes, str]:
        return b"# prometheus-client not installed\n", "text/plain"
