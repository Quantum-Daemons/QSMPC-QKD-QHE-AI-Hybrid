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


#: Boundaries in MILLISECONDS, from 1 ms to 10 minutes, about 2.5x apart (Round 86). The latencies are recorded in milliseconds,
#: and prometheus-client's default buckets are for seconds, 0.005 to 10: every stage above 10 ms fell into +Inf, so the
#: dashboard's p95 read exactly 10 for CKKS, MPC and QKD stages that take hundreds or thousands, and the benchmark's 557 ms
#: MPC-FHE run read 10 too. Found on the full stack's Grafana walk (checklist FD1).
LATENCY_BUCKETS_MS = (1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 60_000, 120_000, 300_000, 600_000)

#: The stages a run can report. Their series exist from the start at zero, so the first run after the API starts is an
#: increase that `rate()` can see; a labelled series that appears already counted reads as no increase, and the dashboard's
#: p95 panel stayed empty until a second run. The list covers every stage class any use case's pipeline can run, the
#: use-case-specific ones included; `tests/test_monitoring_wired.py` fails when a new stage class is missing from it.
STAGES = ("pqc_handshake", "qkd_key_agreement", "qkd_optimize", "qsmpc_compute", "federated_training", "private_inference",
          "explainability", "quantum_comparison", "sign_result", "result",
          "qml_ids", "satellite_qkd", "qsdc_telemetry", "telemetry_anomaly")


def short_backend(backend: str) -> str:
    """The benchmark's backend label: the first word of the row's backend string, `tenseal-ckks-cpu` from
    `tenseal-ckks-cpu + mpyc-shamir`, `qotp+qcnn` from the QSMPC-FHE row's longer description."""
    return backend.split()[0] if backend and backend.split() else "unknown"


if _HAVE:
    PIPELINE_RUNS = Counter("qsh_pipeline_runs_total", "Pipeline runs", ["usecase", "state"])
    STAGE_LATENCY = Histogram("qsh_stage_latency_ms", "Stage latency (ms)", ["stage"], buckets=LATENCY_BUCKETS_MS)
    QBER = Gauge("qsh_qber", "Last QBER observed")
    #: Not a number until a QKD stage reports one. A gauge starts at 0, and the dashboard read "QBER 0" before any key
    #: agreement had run, and again after each API restart.
    QBER.set(float("nan"))
    #: Per path AND backend: MPC-FHE runs on TenSEAL, OpenFHE and HEonGPU where built, and one series per path mixed them,
    #: so the dashboard's median of two rows read a bucket bound that was neither. Measured rows only (benchmark/runner.py).
    BENCH_LATENCY = Histogram("qsh_benchmark_latency_ms", "Benchmark latency (ms), measured rows only", ["path", "backend"],
                              buckets=LATENCY_BUCKETS_MS)
    for _stage in STAGES:
        STAGE_LATENCY.labels(_stage)

    def record_run(usecase: str, state: str) -> None:
        PIPELINE_RUNS.labels(usecase, state).inc()

    def record_stage(stage: str, ms: float) -> None:
        STAGE_LATENCY.labels(stage).observe(ms)

    def record_qber(value: float) -> None:
        QBER.set(value)

    def record_benchmark(path: str, ms: float, backend: str = "") -> None:
        BENCH_LATENCY.labels(path, short_backend(backend)).observe(ms)

    def render() -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST

else:  # graceful no-ops
    def record_run(*_a, **_k) -> None: ...
    def record_stage(*_a, **_k) -> None: ...
    def record_qber(*_a, **_k) -> None: ...
    def record_benchmark(*_a, **_k) -> None: ...
    def render() -> tuple[bytes, str]:
        return b"# prometheus-client not installed\n", "text/plain"
