import { Link } from "react-router-dom";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";
import { USECASES } from "../lib/useCases";
import { modelsFor, studentModel, teacherModel } from "../lib/config";

const DEMO = isClientDemo();

// Use Cases index — each card links to a dedicated per-use-case page (a live two-party
// simulator + that use case's benchmark + XAI). The Benchmark + per-case run UIs are unified there.
export default function UseCases() {
  return (
    <div>
      <h1 className="page-title">Industry Use Cases</h1>
      <p className="page-sub">
        Each use case has its own page: a live <b>two-party (Alice↔Bob) simulator</b> you Start / Pause /
        Step / Stop / Restart, with that industry's benchmark (latency · accuracy · security posture) and, where applicable, a signed XAI explanation. Finance / Medical / Genomics run a
        private inference; Telecom and Satellite are QKD/QSDC paths. Each card names the model that
        page actually runs, the corpus it was trained on and that corpus's licence, read from the
        registry rather than written here.
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
        {USECASES.map((u) => {
          // Everything below is read from the registry. A card cannot claim a model, a metric or a
          // licence this repository did not record, and it cannot go stale when one changes.
          const teacher = teacherModel(u.id);
          const student = studentModel(u.id);
          const pinned = modelsFor(u.id).filter((m) => m.hf_repo_id && m.hf_revision);
          const achieved = teacher?.achieved as Record<string, number> | null | undefined;
          const headline = teacher?.metric?.name;
          const score = headline && achieved ? achieved[headline] : undefined;
          const studentParams = (student?.achieved as { student_params?: number } | undefined)?.student_params;
          // "scaffold" used to be hardcoded to !inference, which understated telecom and satellite:
          // both now carry trained, published, pinned models. Earn it from registry state instead.
          const depth = pinned.length > 0 ? "published" : teacher?.achieved ? "trained" : "scaffold";
          return (
            <Link key={u.id} to={`/usecase/${u.id}`} className="card" style={{ textDecoration: "none", color: "inherit", display: "block" }}>
              <h3>{u.title}</h3>
              <div className="pill-row">
                <span className={`tag ${depth === "scaffold" ? "scaffold" : "deep"}`}>{depth}</span>
                <span className="tag">{u.industry}</span>
              </div>
              <p style={{ fontSize: 13 }}>{u.blurb}</p>
              {teacher && (
                <p className="muted" style={{ fontSize: 11, marginBottom: 2 }}>
                  <b>Model</b> {teacher.label}
                  {score != null && headline && <> · {teacher.metric.display} <b>{score.toFixed(4)}</b>
                    {teacher.metric.baseline != null && <> vs published {teacher.metric.baseline}</>}</>}
                  {!teacher.achieved && <> · <i>not trained in this build</i></>}
                </p>
              )}
              {teacher && (
                <p className="muted" style={{ fontSize: 11, marginBottom: 2 }}>
                  <b>Data</b> {teacher.dataset.name} · {teacher.dataset.licence}
                  {studentParams != null && <> · encrypted student {studentParams} params</>}
                </p>
              )}
              <p className="muted" style={{ fontSize: 11 }}>
                {pinned.length > 0
                  ? <>{pinned.length} model{pinned.length > 1 ? "s" : ""} published and pinned - verifiable in your browser</>
                  : <>weights not published: the source data's licence does not clearly permit redistributing them</>}
              </p>
              <p className="muted" style={{ fontSize: 11 }}>{u.market}</p>
              <span className="btn" style={{ marginTop: 6, display: "inline-block" }}>Open simulator →</span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
