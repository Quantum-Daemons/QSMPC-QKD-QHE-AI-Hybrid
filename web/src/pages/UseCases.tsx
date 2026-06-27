import { Link } from "react-router-dom";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";
import { USECASES } from "../lib/useCases";

const DEMO = isClientDemo();

// Use Cases index — each card links to a dedicated per-use-case page (a live two-party
// simulator + that use case's benchmark + XAI). The Benchmark + per-case run UIs are unified there.
export default function UseCases() {
  return (
    <div>
      <h1 className="page-title">Industry Use Cases</h1>
      <p className="page-sub">
        Each use case has its own page: a live <b>two-party (Alice↔Bob) simulator</b> you Start / Pause /
        Step / Stop / Restart, with that industry's benchmark (latency · accuracy · security score · quantum
        advantage) and, where applicable, a signed XAI explanation. Finance / Medical / Genomics are deep
        (private inference); Telecom and Satellite are QKD/QSDC scaffolds.
      </p>
      <p className="page-sub" style={{ fontSize: 13 }}>
        <b>Agentic AI · quantum-safe MCP is a cross-cutting layer integrated into every use case below</b> —
        each page embeds an in-browser agent whose request/response is sealed + signed over a client-side
        quantum-safe MCP channel (ML-KEM-768 + ML-DSA-65 + AES-256-GCM) as that industry's own MCP tool
        (<code>score_transaction</code> / <code>query_cohort</code> / <code>pool_allele_counts</code> /
        <code>inspect_qkd_telemetry</code> / <code>relay_mission_telemetry</code>), fed by that run's live metrics.
      </p>
      {DEMO && <div className="note" style={{ marginBottom: 12 }}>🖥️ {DEMO_BANNER}</div>}

      <div className="grid cols-2">
        {USECASES.map((u) => (
          <Link key={u.id} to={`/usecase/${u.id}`} className="card" style={{ textDecoration: "none", color: "inherit", display: "block" }}>
            <h3>{u.title}</h3>
            <div className="pill-row">
              <span className={`tag ${u.inference ? "deep" : "scaffold"}`}>{u.inference ? "deep" : "scaffold"}</span>
              <span className="tag">{u.industry}</span>
            </div>
            <p style={{ fontSize: 13 }}>{u.blurb}</p>
            <p className="muted" style={{ fontSize: 11 }}>{u.market}</p>
            <span className="btn" style={{ marginTop: 6, display: "inline-block" }}>Open simulator →</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
