# Monitoring

Monitoring for the **full local deployment**.

## Full local deployment (GPU box)
Run Prometheus + Grafana via `docker compose -f docker/docker-compose.yml up`:
- **Prometheus** scrapes the API `/metrics` endpoint (`qsh.monitoring.exporter`) and `node-exporter`.
- **Grafana** (anonymous viewer enabled) at `http://localhost:3000`. Add a Prometheus
  data source pointing at `http://prometheus:9090`, then import a dashboard charting
  `qsh_pipeline_runs_total`, `qsh_stage_latency_ms`, `qsh_qber`, `qsh_benchmark_latency_ms`.
- **GPU metrics**: run the NVIDIA **DCGM exporter** and add it to `prometheus.yml`
  (commented stub provided) to chart GPU utilization / memory / power for FHE & training.
- For a light-footprint host, ship metrics off-box (`remote_write` stub in `prometheus.yml`) rather
  than running Grafana locally.

**Zabbix** was evaluated and dropped: Prometheus + Grafana already cover metrics and
dashboards here, and Zabbix's agent/server footprint is unjustified at this scale.
