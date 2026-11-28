// A PFCP attack detector tested on attack days it never saw (Round 86, batches 12 and 13; the owner's choice of
// 2026-11-28 to show it). Pre-registered in references/telecom-pfcp-held-out-day-preregistration.md and run offline by
// scripts/check_pfcp_held_out_day.py; scripts/export_outside_data_cards.py writes the file this reads, and every figure
// below is read from it.
import { useEffect, useState } from "react";
import CardExport from "./CardExport";
import StageBadge from "./StageBadge";

interface Labels {
  follow_address_pairs: boolean; attack_message: string; attack_message_in_malicious: number;
  attack_message_in_normal: number; consistent_with_the_attack: boolean;
}
interface Day {
  attack: string; date: string; scorable: boolean; labels: Labels; flows?: number; malicious?: number;
  teacher_auc?: { mean: number; min: number; max: number }; teacher_detection_rate?: number; teacher_false_positive_rate?: number;
  student_auc?: number; student_auc_ci95?: [number, number]; student_detection_rate?: number; student_false_positive_rate?: number;
}
interface Pfcp {
  preregistration: string; record: string; days: Day[]; attribution: string;
  audit: { preregistration: string; record: string };
}

const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

export default function PfcpHeldOutDayCard() {
  const [d, setD] = useState<Pfcp | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/telecom/pfcp_held_out_day.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: Pfcp | null) => { if (live && j?.days) setD(j); })
      .catch(() => undefined);
    return () => { live = false; };
  }, []);
  if (!d) return null;
  const missed = d.days.filter((x) => x.scorable && (x.teacher_auc?.mean ?? 1) < 0.7 && (x.student_auc ?? 1) < 0.7);
  const doubtful = d.days.filter((x) => !x.labels.consistent_with_the_attack);
  const allFollow = d.days.every((x) => x.labels.follow_address_pairs);
  return (
    <div className="card" id="telecom-pfcp" style={{ marginTop: 14 }}>
      <h3>A 5G-core PFCP attack detector on attack days it never saw <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        The 5GC PFCP Intrusion Detection Dataset records four denial-of-service attacks on an emulated 5G core&apos;s PFCP
        sessions, each on its own day. A detector was fitted on three days&apos; flows at the user-plane function and scored on
        the fourth, each day in turn: the 5G-NIDD teacher&apos;s recipe (five seeds) and the linear student&apos;s procedure, its
        threshold set for 1% false positives on the training days&apos; last flows. This is a new detector on new inputs: the
        page&apos;s 5G-NIDD models read another testbed&apos;s flows and cannot score these.
        {missed.length ? <> Neither model recognises the {missed.map((x) => x.attack).join(" and ")} day&apos;s attack.</> : null}
      </p>
      {doubtful.length ? (
        <p className="small" style={{ marginTop: 6 }}>
          <strong>Read these labels with care.</strong> An audit run afterwards (pre-registered) found that
          {allFollow ? <> every day&apos;s labels follow its endpoints, each address pair carrying one label, and that</> : null} on
          the {doubtful.length} days marked doubtful in the last column the flows labelled malicious carry the day&apos;s attack
          message no more often than the normal ones. On those days these figures may measure the labelled endpoint rather
          than the attack; the dataset&apos;s creators can settle which.
        </p>
      ) : null}
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>Day held out</th><th>Flows (malicious)</th><th>Teacher AUC, 5 seeds</th><th>Student AUC (95%)</th>
            <th>Student: attack flows flagged / normal flows flagged</th>
            <th>Flows carrying the attack message: malicious / normal</th></tr></thead>
          <tbody>
            {d.days.map((x) => x.scorable ? (
              <tr key={x.date}>
                <td>{x.attack}, {x.date}</td><td>{x.flows} ({x.malicious})</td>
                <td>{f3(x.teacher_auc!.mean)}{x.teacher_auc!.max - x.teacher_auc!.min > 5e-4 ? <> ({f3(x.teacher_auc!.min)}-{f3(x.teacher_auc!.max)})</> : null}</td>
                <td><b>{f3(x.student_auc!)}</b> ({f3(x.student_auc_ci95![0])}-{f3(x.student_auc_ci95![1])})</td>
                <td>{pct(x.student_detection_rate!)} / {pct(x.student_false_positive_rate!)}</td>
                <td>{pct(x.labels.attack_message_in_malicious)} / {pct(x.labels.attack_message_in_normal)}{x.labels.consistent_with_the_attack ? null : <> (doubtful)</>}</td>
              </tr>
            ) : <tr key={x.date}><td>{x.attack}, {x.date}</td><td colSpan={5}>not scorable</td></tr>)}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ marginTop: 6 }}>
        The days are small, so the intervals are wide. Pre-registered (<code>{d.preregistration}</code>); record{" "}
        <code>{d.record}</code>; the audit <code>{d.audit.preregistration}</code>, record <code>{d.audit.record}</code>.
        Data: {d.attribution}.
      </p>
      <CardExport name="telecom_ids-pfcp-held-out-day" label="the PFCP figures" rows={d.days.map((x) => ({ ...x,
        teacher_auc: x.teacher_auc?.mean ?? null, student_auc_ci95: x.student_auc_ci95?.join(" to ") ?? null,
        labels: undefined, attack_message_in_malicious: x.labels.attack_message_in_malicious,
        attack_message_in_normal: x.labels.attack_message_in_normal, labels_consistent_with_the_attack: x.labels.consistent_with_the_attack }))} json={d} />
    </div>
  );
}
