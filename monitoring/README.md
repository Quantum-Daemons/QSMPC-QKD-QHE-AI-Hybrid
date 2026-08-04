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
  **already shipped** at `monitoring/grafana/qsh-pipeline.json` and is auto-provisioned — you do not
  need to add a datasource or import anything by hand.
- **GPU metrics**: run the NVIDIA **DCGM exporter** and add it to `prometheus.yml`
  (commented stub provided) to chart GPU utilization / memory / power for FHE & training.
- For a light-footprint host, ship metrics off-box (`remote_write` stub in `prometheus.yml`) rather
  than running Grafana locally.

**Zabbix** was evaluated and dropped: Prometheus + Grafana already cover metrics and
dashboards here, and Zabbix's agent/server footprint is unjustified at this scale.

## Concerns

- **The four pipeline metrics are declared but never recorded.** `src/qsh/monitoring/exporter.py`
  defines `qsh_pipeline_runs_total`, `qsh_stage_latency_ms` and `qsh_qber`, and the API serves
  `/metrics` — but a repo-wide search finds no call site that records any of them, and
  `qsh_benchmark_latency_ms` has no recorder helper at all. `/metrics` therefore returns a valid,
  permanently empty response and every dashboard panel charts nothing. **Instrumenting the pipeline
  stages is the prerequisite for this directory being useful**; until then, treat the dashboard as
  scaffolding, not as working observability.
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

1. Instrument the stages: call the exporter helpers from `src/qsh/orchestration/pipeline.py` so the
   metrics carry data, and add a recorder for `qsh_benchmark_latency_ms`.
2. Verify provisioning after any compose change by asserting a non-empty dashboard list
   (`curl -s localhost:3000/api/search`) rather than by the container reporting healthy.
3. For the public demo, decide deliberately between "no telemetry, by design" (state it in
   `docs/threat-model.md` as an accepted residual risk) and adding a minimal client-side error beacon.
   Right now the absence is undocumented rather than chosen.
