# Monitoring

Monitoring for the **full local deployment**.

> **The public demo has no monitoring, by design.** It is client-only: `docker/docker-compose.demo.yml`
> runs a single `web` container with no API and no Prometheus, and `docker/env/Caddyfile.client-only`
> answers `/metrics` with 404. Nothing below applies to it. See *Concerns* for what that costs.

## Full local deployment (GPU box)

Run Prometheus + Grafana via `docker compose -f docker/docker-compose.yml up`:

- **Prometheus** scrapes the API `/metrics` endpoint (`qsh.monitoring.exporter`) and `node-exporter`.
- **Grafana** (anonymous viewer enabled) at `http://localhost:3000`. The dashboard charting
  `qsh_pipeline_runs_total`, `qsh_stage_latency_ms`, `qsh_qber` and `qsh_benchmark_latency_ms` is
  **already shipped** at `monitoring/grafana/qsh-pipeline.json` and is auto-provisioned - you do not
  need to add a datasource or import anything by hand.
- **GPU metrics**: run the NVIDIA **DCGM exporter** and add it to `prometheus.yml`
  (commented stub provided) to chart GPU utilization / memory / power for FHE & training.
- For a light-footprint host, ship metrics off-box (`remote_write` stub in `prometheus.yml`) rather
  than running Grafana locally.

**Zabbix** was evaluated and dropped: Prometheus + Grafana already cover metrics and
dashboards here, and Zabbix's agent/server footprint is unjustified at this scale.

## Concerns

- **The four pipeline metrics are recorded, as of Round 21.** For the three rounds before that they
  were declared and never called: `src/qsh/monitoring/exporter.py` defined `qsh_pipeline_runs_total`,
  `qsh_stage_latency_ms` and `qsh_qber` with no caller outside that file, `qsh_benchmark_latency_ms`
  had no recorder helper at all, and `/metrics` therefore returned a valid, permanently empty
  response while every dashboard panel charted nothing. Nothing about that was visible from outside:
  the endpoint worked, the scrape parsed, the dashboard rendered.

  The call sites are `src/qsh/orchestration/pipeline.py` - `record_stage` per stage, `record_run` and
  `record_qber` per run in a `finally`, so a run that raised is counted in the state it ended in -
  and `src/qsh/benchmark/runner.py`, `record_benchmark` per measured row, which is where the `path`
  and `backend` labels come from. Measured in Round 21: the scrape went from 1 series to 39 after one run.
  `tests/test_monitoring_wired.py` asserts that production code CALLS them, rather than that the
  helpers exist; the helpers always existed.
- **Until Round 86 the two latency histograms had bucket bounds in seconds and values in
  milliseconds.** prometheus-client's default bounds run from 0.005 to 10, so every stage slower than
  10 ms landed in `+Inf` and the dashboard's p95 read exactly 10 for CKKS, MPC and QKD stages that
  take hundreds or thousands of milliseconds; the benchmark's 557 ms MPC-FHE row read 10 as well.
  Found on the full stack's Grafana walk. Both histograms now use `LATENCY_BUCKETS_MS`, 1 ms to
  10 minutes about 2.5x apart. A second, smaller gap: a labelled series that first appears with a
  count of 1 is no increase to `rate()`, so the p95 panel stayed empty until a second run. Every
  stage class a use case can run now has its series at zero from start-up, and the panel hides the
  stages that did not run in its window; the tests fail when a 557 ms observation lands in `+Inf` or
  a new stage class is missing from the list.
- **Three more readings on the same walk were wrong, and are fixed in Round 86.** The QBER gauge
  started at 0, so the panel read "QBER 0" before any key agreement had run and after each API
  restart; it is now not a number until a QKD stage reports one. The benchmark histogram had a
  `path` label only, so MPC-FHE's TenSEAL and OpenFHE rows shared one series and the median of the
  two read a bucket bound that was neither; it now carries a `backend` label and the panel shows the
  exact mean per path and backend (sum over count, since the API started). And the extrapolated
  `large` tier's latency went into that histogram as if measured; only measured rows are recorded.
- **Grafana provisioning was silently broken until it was fixed here.** `docker-compose.yml` mounted
  the bare dashboard directory onto `/etc/grafana/provisioning/dashboards`. Grafana reads *provider
  YAML* from that path and follows each provider's own `options.path`, so a directory containing only
  `qsh-pipeline.json` loaded nothing. The stack came up healthy with an empty dashboard list, which is
  exactly how this went unnoticed. `monitoring/grafana/provisioning/` now supplies the dashboard
  provider and the Prometheus datasource; the JSON is mounted at `/var/lib/grafana/dashboards`.
- **No operator visibility into the public demo.** Because the client-only build serves no `/metrics`
  and logs nothing server-side, a demo that breaks in a visitor's browser is invisible to the operator.
  There is currently no error reporting path at all.

## Recommendations

1. Verify provisioning after any compose change by asserting a non-empty dashboard list
   (`curl -s localhost:3000/api/search`) rather than by the container reporting healthy.
2. Read each quantile panel against a run whose time you know. A quantile that equals a bucket bound
   for every series, as the p95 of 10 did, says the bounds do not cover the values.
3. The public demo's absence of telemetry is recorded in `docs/threat-model.md` as an accepted
   residual risk. A minimal client-side error beacon remains the alternative if that choice changes.
