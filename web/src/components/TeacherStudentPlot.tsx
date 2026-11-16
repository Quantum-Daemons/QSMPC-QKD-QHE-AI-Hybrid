// IBM's Multi-PNA+EU against the two models distilled from it (Round 86, the owner's choice of 2026-11-16), on the 16
// held-out transfers all three are checked on. The teacher's logits are the pipeline run's own (its verification of
// the published graph sample); the students' are computed here in the clear from their pinned weights, after each is
// checked against the logits its trainer recorded (`lib/teacherStudent.ts`).
import { useEffect, useId, useMemo, useState } from "react";
import { CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { ChartReadout, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import type { GnnWeights } from "../lib/clientHybridGnn";
import type { MlpWeights } from "../lib/clientHybridMlp";
import { modelsFor } from "../lib/config";
import { publishedWeightsOf } from "../lib/onnxInference";
import { asJson } from "../lib/pinnedFetch";
import {
  clipLogOdds, gnnStudent, LOG_ODDS_CLIP, mlpStudent, type Pairing, pairUp, type StudentRows, type TeacherRows, thresholdLogOdds,
} from "../lib/teacherStudent";

const STUDENTS = [
  { key: "finance_aml.gnn_compact", name: "compact GNN" },
  { key: "finance_aml.mlp_large", name: "MLP" },
] as const;

const tick = { fill: "var(--muted)", fontSize: 11 };
const LAUNDERING = "var(--danger)";
const GNN_CLEAN = "var(--accent)";
const MLP_CLEAN = "var(--accent-2)";
const axisLabel = (value: string) => ({ value, position: "insideBottom" as const, offset: -8, fill: "var(--muted)", fontSize: 11 });
const yLabel = (value: string) => ({ value, angle: -90, position: "insideLeft" as const, fill: "var(--muted)", fontSize: 11 });
const f2 = (v: number) => v.toFixed(2);

async function studentRows(key: string): Promise<StudentRows> {
  const spec = modelsFor("finance_aml").find((m) => m.key === key);
  if (!spec) throw new Error(`${key} is not in this page's registry`);
  const bytes = await publishedWeightsOf(spec);       // checked against its pin before it is read
  return key === "finance_aml.gnn_compact" ? gnnStudent(key, asJson<GnnWeights>(bytes)) : mlpStudent(key, asJson<MlpWeights>(bytes));
}

/** Which of the 16 the run decrypted under MPC, per student, as the run reports it: an index into the student's fixtures. */
export interface DecryptedRows { "finance_aml.gnn_compact"?: number; "finance_aml.mlp_large"?: number }

export default function TeacherStudentPlot({ teacher, teacherName, decrypted, taughtBy }:
  { teacher: TeacherRows | null; teacherName: string; decrypted?: DecryptedRows; taughtBy?: Record<string, string> }) {
  const captionId = useId();
  const [students, setStudents] = useState<Record<string, StudentRows | string>>({});
  useEffect(() => {
    if (!teacher) return;
    let live = true;
    for (const s of STUDENTS) {
      studentRows(s.key).then((r) => { if (live) setStudents((o) => ({ ...o, [s.key]: r })); },
                              (e: Error) => { if (live) setStudents((o) => ({ ...o, [s.key]: e.message })); });
    }
    return () => { live = false; };
  }, [teacher]);

  const pairings = useMemo(() => {
    const out: Record<string, Pairing | string> = {};
    if (!teacher) return out;
    for (const s of STUDENTS) {
      const got = students[s.key];
      out[s.key] = got === undefined ? "loading" : typeof got === "string" ? got : pairUp(teacher, got);
    }
    return out;
  }, [teacher, students]);

  if (!teacher) {
    return <p className="muted small">Press <strong>Start</strong>: the pipeline run scores {teacherName} on its 16 held-out
      transfers from the published graph, and this view then draws them against the compact GNN and the MLP distilled from it.</p>;
  }
  const ready = STUDENTS.map((s) => [s, pairings[s.key]] as const);
  const drawn = ready.filter(([, p]) => typeof p === "object" && !p.refused) as [typeof STUDENTS[number], Pairing][];
  // One series per student and label, each with its own shape (WCAG 1.4.1: never colour alone), named in the legend.
  const pts = (key: string, label: number) => {
    const p = drawn.find(([s]) => s.key === key)?.[1];
    return p ? p.points.filter((q) => q.label === label).map((q) => ({ x: clipLogOdds(q.teacher), y: clipLogOdds(q.student), row: q.row })) : null;
  };
  const gnnL = pts("finance_aml.gnn_compact", 1), gnnC = pts("finance_aml.gnn_compact", 0);
  const mlpL = pts("finance_aml.mlp_large", 1), mlpC = pts("finance_aml.mlp_large", 0);
  const tz = thresholdLogOdds(teacher.threshold);
  const lim = LOG_ODDS_CLIP + 1;
  const byRow = new Map<number, Record<string, Pairing["points"][number]>>();
  for (const [s, p] of drawn) for (const q of p.points) byRow.set(q.row, { ...(byRow.get(q.row) ?? {}), [s.key]: q });
  const says = (v: number) => (v ? "laundering" : "clean");

  return (
    <div>
      <p className="small" style={{ margin: "6px 0 8px" }}>
        {ready.map(([s, p]) => (
          <span key={s.key} style={{ marginRight: 14 }}>
            <b>{s.name}</b>: {p === "loading" ? "fetching its pinned weights..."
              : typeof p === "string" ? `not drawn (${p})`
              : p.refused ? `not drawn: ${p.refused}`
              : `${p.agree} of ${p.n} decisions agree with ${teacherName}`}{taughtBy?.[s.key] ? ` (${taughtBy[s.key]})` : ""}
          </span>
        ))}
      </p>
      {drawn.length > 0 && (
        <figure className="chart-figure">
          <figcaption id={captionId} className="small muted">
            <b>{teacherName} (the teacher, across) against each student (up), log-odds of laundering,</b> one point per held-out
            transfer and student. The teacher&apos;s log-odds are this run&apos;s, from the published graph; each student&apos;s are
            computed here in the clear from its pinned weights, within{" "}
            {Math.max(...drawn.map(([, p]) => p.maxFixtureDiff)).toExponential(1)} of the logits its trainer recorded. Dashed: the
            teacher&apos;s threshold ({teacher.threshold.toFixed(3)} on its probability) across and the students&apos; up, at zero, where
            each one&apos;s threshold is folded into its last bias; dotted: equal log-odds. Drawn clipped at {"\u00b1"}{LOG_ODDS_CLIP}.
          </figcaption>
          <ChartReadout hint={HINT_POINTER}>
            <MarksHidden>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart role="img" title={`Log-odds of laundering, ${teacherName} against the compact GNN and the MLP, per held-out transfer`}
                              aria-describedby={captionId} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" name="teacher log-odds" domain={[-lim, lim]} tick={tick} label={axisLabel(`${teacherName}, log-odds`)} />
                  <YAxis type="number" dataKey="y" name="student log-odds" domain={[-lim, lim]} tick={tick} label={yLabel("student, log-odds")} />
                  <Tooltip content={<ReadoutTooltip />} formatter={(v: number) => f2(v)} />
                  <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine segment={[{ x: -lim, y: -lim }, { x: lim, y: lim }]} stroke="var(--muted)" strokeDasharray="1 3" />
                  <ReferenceLine x={tz} stroke="var(--muted)" strokeDasharray="4 4" />
                  <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="4 4" />
                  {gnnL && <Scatter name="compact GNN, laundering (cross)" data={gnnL} fill={LAUNDERING} shape="cross" isAnimationActive={false} />}
                  {gnnC && <Scatter name="compact GNN, clean (circle)" data={gnnC} fill={GNN_CLEAN} shape="circle" isAnimationActive={false} />}
                  {mlpL && <Scatter name="MLP, laundering (star)" data={mlpL} fill={LAUNDERING} shape="star" isAnimationActive={false} />}
                  {mlpC && <Scatter name="MLP, clean (triangle)" data={mlpC} fill={MLP_CLEAN} shape="triangle" isAnimationActive={false} />}
                </ScatterChart>
              </ResponsiveContainer>
            </MarksHidden>
          </ChartReadout>
        </figure>
      )}
      {byRow.size > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table className="table small" style={{ marginTop: 8 }}>
            <thead><tr><th>Test-window row</th><th>Label</th><th>{teacherName}</th>
              {drawn.map(([s]) => <th key={s.key}>{s.name}</th>)}</tr></thead>
            <tbody>
              {[...byRow.entries()].map(([row, per]) => {
                const any = Object.values(per)[0];
                return (
                  <tr key={row}>
                    <td>{row.toLocaleString("en-US")}</td><td>{says(any.label)}</td>
                    <td>{f2(any.teacher)}, {says(any.teacherSays)}</td>
                    {drawn.map(([s]) => {
                      const q = per[s.key];
                      const at = decrypted?.[s.key], st = students[s.key];
                      const mark = at !== undefined && typeof st === "object" && st.rows[at] === row ? " (decrypted under MPC this run)" : "";
                      return <td key={s.key}>{q ? `${f2(q.student)}, ${says(q.studentSays)}${q.studentSays === q.teacherSays ? "" : " (differs)"}${mark}` : "-"}</td>;
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
