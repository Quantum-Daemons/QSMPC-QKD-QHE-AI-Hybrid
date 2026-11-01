import { Link } from "react-router";
import KeyPoints from "../components/KeyPoints";
import StageBadge from "../components/StageBadge";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";
import { USECASES } from "../lib/useCases";
import { modelsFor, studentModel, teacherModel } from "../lib/config";

const DEMO = isClientDemo();

/**
 * What a card prints beside a headline that shares a metric NAME with its published figure and not a
 * measurement (the registry sets `metric.comparison_caveat` for exactly those). Keyed by registry key.
 *
 * The satellite card printed "event-wise F0.5 0.0000 vs published 0.968" and nothing else, while its
 * own registry entry says the figure is over 3 scored events, under another protocol, and does not
 * reproduce across retrains. Every number below is one that entry states - ESA-ADB's 29 anomalies
 * among 65 annotated events of the Mission1 lightweight test set in `baseline_source` (the paper's
 * Supplementary Table 11; 0.968 is its Table 9, anomalies alone), and the earlier runs' 0.714286 and
 * 0.454545 in `baseline_gap_note` - and tests/test_r84_pages_claims.py holds each one to the
 * entry, so this sentence cannot outlive the record it summarises. The event count itself is read
 * from the record (`n_events_scored`), not written here.
 */
const NOT_LIKE_FOR_LIKE: Record<string, string> = {
  "satellite_comms.telemanom":
    "Not a like-for-like pair: ESA-ADB's 0.968 is scored on 29 anomaly events (65 annotated events "
    + "in all), with another split, resampling and precision correction. With this few events one "
    + "detection moves the figure by tenths: two earlier, identically configured trainings, before a "
    + "scaler fix, scored 0.714 and 0.455.",
};

// Use Cases index — each card links to a dedicated per-use-case page (a live architecture
// simulator + that use case's benchmark + XAI). The Benchmark + per-case run UIs are unified there.
export default function UseCases() {
  return (
    <div>
      <h1 className="page-title">Industry Use Cases</h1>
      {/* Two paragraphs (123 words) became four points and a table drawn from the use-case data,
          so the per-industry MCP tool and channel cannot drift from what each page runs. */}
      <KeyPoints items={[
        <>Each use case has its own page with a live <b>architecture simulator</b>: each party's components, a lane per stage between them, animated as the run goes. Start / Pause / Step / Stop / Restart drive it.</>,
        <>Each page shows that industry's benchmark (latency · communication · security posture) and, where applicable, a signed XAI explanation.</>,
        <>Every card's model, corpus and licence is read from the registry, not written here.</>,
        <><b>Agentic AI - quantum-safe MCP</b> runs on every page: an in-browser agent&apos;s request and response are sealed and signed (ML-KEM-768 + X25519, ML-DSA-65, AES-256-GCM) in a JSON-RPC envelope named after that industry&apos;s MCP tool, fed by the run&apos;s live metrics. No tool is executed.</>,
      ]} />
      <table className="facts-table small" style={{ margin: "10px 0 16px", maxWidth: 860 }}>
        <thead><tr><th>Use case</th><th>Agent's MCP tool</th><th>Private inference</th><th>Quantum channel</th></tr></thead>
        <tbody>
          {USECASES.map((u) => (
            <tr key={u.id}>
              <td>{u.nav}</td>
              <td><code>{u.agenticTool}</code></td>
              <td>{u.inference ? "CKKS + garbled-circuit 2PC" : "-"}</td>
              {/* The attack classifier is not a quantum channel, and by default it is not quantum
                  either (an in-tab softmax and the published random forest), so "+ QML-IDS" left
                  this column. Telecom's page names its classifier itself. */}
              <td>{u.id === "satellite_comms" ? "free-space BB84/E91 + QSDC test" : "fibre BB84/E91"}{" "}
                <StageBadge kind="simulated" /></td>
            </tr>
          ))}
        </tbody>
      </table>
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
          // How many events an event-wise headline was divided by, and whether the published figure
          // is a different measurement. Both are the registry's, as on the model panel.
          const scoredOver = typeof achieved?.n_events_scored === "number" ? achieved.n_events_scored : null;
          const notComparable = !!teacher?.metric?.comparison_caveat;
          // "scaffold" used to be hardcoded to !inference, which understated telecom and satellite:
          // both now carry trained, published, pinned models. Earn it from registry state instead.
          // "measured", not "trained": the one unpublished model with recorded results is genomics'
          // PRS, whose weights are PGS000072's published ones, measured here and trained by no one.
          const depth = pinned.length > 0 ? "published" : teacher?.achieved ? "measured" : "scaffold";
          return (
            <Link key={u.id} to={`/usecase/${u.id}`} className="card" style={{ textDecoration: "none", color: "inherit", display: "block" }}>
              {/* h2: each card is a section of this page, and an h3 here left the outline going
                  from the page title straight to h3 (SC 1.3.1, Round 74). Styled as before. */}
              <h2>{u.title}</h2>
              <div className="pill-row">
                <span className={`tag ${depth === "scaffold" ? "scaffold" : "deep"}`}>{depth}</span>
                <span className="tag">{u.industry}</span>
              </div>
              <p style={{ fontSize: 13 }}>{u.blurb}</p>
              {teacher && (
                <p className="muted" style={{ fontSize: 11, marginBottom: 2 }}>
                  <b>Model</b> {teacher.label}
                  {score != null && headline && <> · {teacher.metric.display} <b>{score.toFixed(4)}</b>
                    {scoredOver != null && <> over {scoredOver} scored event{scoredOver === 1 ? "" : "s"}</>}
                    {teacher.metric.baseline != null && (notComparable
                      ? <> · published {teacher.metric.baseline}, a different measurement</>
                      : <> vs published {teacher.metric.baseline}</>)}</>}
                  {!teacher.achieved && <> · <i>not trained in this build</i></>}
                </p>
              )}
              {teacher && score != null && notComparable && (
                <p className="muted" style={{ fontSize: 11, marginBottom: 2 }}>
                  {NOT_LIKE_FOR_LIKE[teacher.key]
                    ?? "Not a like-for-like pair: the published figure is measured under another protocol; the use-case page says how."}
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
                  : u.id === "genomics"
                    // Served, not withheld: the payload ships the score's published weights, under
                    // the licence the Data line above prints.
                    ? <>not on the Hugging Face Hub: the score's published effect weights are served with this page, under the licence above</>
                    : <>weights not published: the source data's licence does not clearly permit redistributing them</>}
              </p>
              <span className="btn" style={{ marginTop: 6, display: "inline-block" }}>Open simulator →</span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
