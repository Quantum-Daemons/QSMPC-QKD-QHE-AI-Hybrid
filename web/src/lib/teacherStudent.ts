// The IBM tab's teacher-student view (Round 86, the owner's choice of 2026-11-16): IBM's Multi-PNA+EU against the two
// models distilled from it, the compact GNN and the MLP, on the 16 held-out transfers all three are checked on. The
// teacher's logits are the ones the pipeline run computed from the published ONNX graph (its verification of the
// graph sample); each student's are computed here in the clear from its pinned weights file, and checked against the
// logits the trainer recorded for the same rows before any point is drawn. The run decrypts one of these rows under
// MPC per student; the view marks that row.
//
// It shows what distillation keeps and what it loses, transfer by transfer, as a fidelity report beside the task
// metric (Distillation Loss Statement, arXiv:2604.25110): agreement of the decisions, and where on the log-odds scale
// the two models put each transfer.

import { gnnForward, gnnRow, type GnnWeights } from "./clientHybridGnn";
import { mlpForward, type MlpWeights } from "./clientHybridMlp";

/** Log-odds are drawn clipped to this, so a saturated logit stays on the plot; the table prints the unclipped value. */
export const LOG_ODDS_CLIP = 12;
/** A clear forward that misses the trainer's recorded logits by more than this is not drawn. */
export const FIXTURE_TOLERANCE = 1e-4;

export interface TeacherRows { rows: number[]; labels: number[]; logits: number[][]; threshold: number }
export interface StudentRows { key: string; rows: number[]; labels: number[]; logits: number[][]; maxFixtureDiff: number }

export interface Pairing {
  key: string;
  points: { row: number; label: number; teacher: number; student: number; teacherSays: number; studentSays: number }[];
  agree: number; n: number; maxFixtureDiff: number;
  /** Why nothing is drawn, when the two cannot be paired honestly. */
  refused?: string;
}

/** log p(1)/p(0) from two logits: exact, and finite where a softmax would round to 0 or 1. */
export const logOdds = (l: number[]) => l[1] - l[0];
export const clipLogOdds = (z: number) => Math.max(-LOG_ODDS_CLIP, Math.min(LOG_ODDS_CLIP, z));
/** The teacher's threshold on p(laundering), as log-odds. */
export const thresholdLogOdds = (t: number) => Math.log(t / (1 - t));

const maxDiff = (a: number[], b: number[]) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

/** The MLP's 16 fixture rows in the clear, keyed by their row in the test window. */
export function mlpStudent(key: string, w: MlpWeights): StudentRows {
  const f = w.fixtures;
  const logits = f.rows.map((r) => mlpForward(w, r));
  return { key, rows: f.row_index_in_test_window ?? [], labels: f.labels, logits,
           maxFixtureDiff: logits.reduce((m, l, i) => Math.max(m, maxDiff(l, f.logits[i])), 0) };
}

/** The compact GNN's 16 fixture rows in the clear, keyed by their row in the test window. */
export function gnnStudent(key: string, w: GnnWeights): StudentRows {
  const f = w.fixtures;
  const logits = f.labels.map((_, i) => gnnForward(w, gnnRow(w, i)));
  return { key, rows: f.row_index_in_test_window, labels: f.labels, logits,
           maxFixtureDiff: logits.reduce((m, l, i) => Math.max(m, maxDiff(l, f.logits[i])), 0) };
}

/** Teacher and student on the rows both hold. Each decides by its own rule: the teacher at its validation-tuned
 *  threshold on p(laundering), as its sample records; a student by argmax, its threshold folded into its last bias. */
export function pairUp(t: TeacherRows, s: StudentRows): Pairing {
  const base = { key: s.key, points: [], agree: 0, n: 0, maxFixtureDiff: s.maxFixtureDiff };
  if (!(s.maxFixtureDiff <= FIXTURE_TOLERANCE)) {
    return { ...base, refused: `its clear forward misses the logits its trainer recorded by ${s.maxFixtureDiff.toExponential(2)}` };
  }
  if (s.rows.length !== s.labels.length || t.rows.length !== t.labels.length) {
    return { ...base, refused: "a file does not say which test-window row each of its fixtures is" };
  }
  const at = new Map(s.rows.map((r, i) => [r, i]));
  const tz = thresholdLogOdds(t.threshold);
  const points: Pairing["points"] = [];
  for (let k = 0; k < t.rows.length; k++) {
    const i = at.get(t.rows[k]);
    if (i === undefined) continue;
    if (s.labels[i] !== t.labels[k]) {
      return { ...base, refused: `row ${t.rows[k]} is labelled ${t.labels[k]} by the teacher's sample and ${s.labels[i]} by the student's` };
    }
    const teacher = logOdds(t.logits[k]), student = logOdds(s.logits[i]);
    const teacherSays = teacher >= tz ? 1 : 0;
    const studentSays = s.logits[i][1] > s.logits[i][0] ? 1 : 0;
    points.push({ row: t.rows[k], label: t.labels[k], teacher, student, teacherSays, studentSays });
  }
  if (points.length === 0) return { ...base, refused: "the teacher's sample and the student's fixtures share no row" };
  return { ...base, points, n: points.length, agree: points.filter((p) => p.teacherSays === p.studentSays).length };
}
