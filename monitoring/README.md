# Monitoring

Two profiles, by environment.

## Local full deployment (RTX PRO 6000 box)
Run Prometheus + Grafana via `docker compose -f docker/docker-compose.yml up`:
- **Prometheus** scrapes the API `/metrics` endpoint (`qsh.monitoring.exporter`) and `node-exporter`.
- **Grafana** (anonymous viewer enabled) at `http://localhost:3000`. Add a Prometheus
  data source pointing at `http://prometheus:9090`, then import a dashboard charting
  `qsh_pipeline_runs_total`, `qsh_stage_latency_ms`, `qsh_qber`, `qsh_benchmark_latency_ms`.
- **GPU metrics**: run the NVIDIA **DCGM exporter** and add it to `prometheus.yml`
  (commented stub provided) to chart GPU utilization / memory / power for FHE & training.

## Public demo VPS (<vps-provider> 6 vCPU / 12 GB)
Do **not** run Prometheus + Grafana on-box — they compete with the app for the 12 GB.
Options, lightest first:
1. Structured JSON container logs (`json-file`, rotated) + the API `/metrics` endpoint
   scraped externally.
2. **Prometheus `remote_write` → Grafana Cloud free tier** (stub in `prometheus.yml`)
   so dashboards live off-box.

**Zabbix** was evaluated and dropped: Prometheus + Grafana already cover metrics and
dashboards here, and Zabbix's agent/server footprint is unjustified at this scale.
