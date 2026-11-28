// The 5G-NIDD models re-run under leakage-aware protocols (Round 86, batch 9; the owner's choices of 2026-11-27).
// Pre-registered in references/telecom-nidd-clean-protocols-preregistration.md and run offline by
// scripts/train/telecom_nidd_clean_protocols.py; scripts/export_nidd_clean_protocols.py writes the file this reads.
// Every comparison sentence is computed here from that file; nothing renders until it exists.
import { useEffect, useState } from "react";

type Cell = { mean: number; sd: number | null; runs: number };
type Grid = Record<string, Record<string, Cell>>;
interface Clean {
  preregistration: { file: string }; record: string;
  splits: Record<string, string>; inputs: Record<string, string>;
  arms: { teacher?: Grid; student?: Grid; mlp?: Grid; federation?: Grid };
  single_feature_check: { flagged: Record<string, Record<string, unknown>>;
    highest: Record<string, { input: string; class: string; f1: number }[]> };
  crosstestbed?: { teacher: { macro_f1: Cell; binary_attack_f1: Cell }; student: { macro_f1: Cell; binary_attack_f1: Cell };
    shared_rows: number; dataset: string; licence: string; source: string; citation: string };
  ckks_recheck?: { rows: number; ciphertexts: number; decisions_differ: number; max_abs_logit_error: number;
    encrypted_accuracy: number; plaintext_accuracy: number };
  attribution: { dataset: string; licence: string; licence_url: string; citation: string };
}

const f4 = (v: number) => v.toFixed(4);
const show = (c?: Cell) => (!c ? "-" : c.sd == null ? f4(c.mean) : `${f4(c.mean)} (${f4(c.sd)})`);
const signed = (v: number) => `${v >= 0 ? "+" : ""}${f4(v)}`;
const ARMS: { key: keyof Clean["arms"]; label: string }[] = [
  { key: "teacher", label: "Teacher (XGBoost)" }, { key: "student", label: "Linear student (batch 6)" },
  { key: "mlp", label: "Square network" }, { key: "federation", label: "Two-station federation" },
];

export default function NiddCleanProtocolsCard() {
  const [d, setD] = useState<Clean | null>(null);
  useEffect(() => {
    let on = true;
    fetch(`${import.meta.env.BASE_URL}data/telecom_nidd/clean_protocols.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: Clean | null) => { if (on && j?.arms) setD(j); })
      .catch(() => undefined);
    return () => { on = false; };
  }, []);
  if (!d) return null;
  const splits = Object.keys(d.splits);
  const t = d.arms.teacher, s = d.arms.student, fed = d.arms.federation;
  const fedSets = fed && s ? (["F36", "F30", "F28"] as const).filter((k) => fed[k]?.P2 && s[k]?.P2) : [];
  const p1 = (g: Record<string, Cell>) => (g.P1_train_1_test_2.mean + g.P1_train_2_test_1.mean) / 2;
  const flagged = Object.values(d.single_feature_check.flagged).reduce((n, v) => n + Object.keys(v).length, 0);
  const top = d.single_feature_check.highest.P2 ?? [];
  const ct = d.crosstestbed;
  return (
    <div className="card">
      <h3 style={{ margin: "0 0 4px" }}>The 5G-NIDD models under leakage-aware protocols, pre-registered</h3>
      <p className="small" style={{ margin: 0 }}>
        The figures above come from a random split of flows. A 2026 study (Hameed and Ibrahim, J. Cybersecur. Priv.
        6(6):205) finds such splits and identifier-like columns inflate 5G-NIDD scores, so the shipped recipes were run
        again, offline and as pre-registered, under three stricter splits: training on one base station and testing on
        the other, a time split within each capture file, and LEAF&apos;s own rule; and with LEAF&apos;s six flagged
        columns removed, then TTL as well. Each cell is FLamby&apos;s metric (or the macro-F1 where the test part is one
        station), the mean over five seeds with the standard deviation in brackets where the arm has seeds.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small" style={{ marginTop: 6 }}>
          <thead>
            <tr>
              <th scope="col" style={{ textAlign: "left" }}>Model, inputs</th>
              {splits.map((p) => <th scope="col" key={p}>{d.splits[p]}</th>)}
            </tr>
          </thead>
          <tbody>
            {ARMS.flatMap(({ key, label }) => Object.entries(d.arms[key] ?? {}).map(([inp, row]) => (
              <tr key={`${key}/${inp}`}>
                <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{label}, {d.inputs[inp] ?? inp}</th>
                {splits.map((p) => <td key={p} style={{ textAlign: "center" }}>{show(row[p])}</td>)}
              </tr>
            )))}
          </tbody>
        </table>
      </div>
      <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {t && s ? (
          <li>Against the shipped split, the stricter splits move the teacher by {signed(p1(t.T46) - t.T46.P0.mean)} (across
            stations, the mean of both directions) and {signed(t.T46.P2.mean - t.T46.P0.mean)} (the time split), and the
            student by {signed(p1(s.F36) - s.F36.P0.mean)} and {signed(s.F36.P2.mean - s.F36.P0.mean)}.</li>
        ) : null}
        {t && s ? (
          <li>Without LEAF&apos;s six columns the teacher moves by {signed(t.T40.P2.mean - t.T46.P2.mean)} and the student by{" "}
            {signed(s.F30.P2.mean - s.F36.P2.mean)} on the time split; without TTL as well, by{" "}
            {signed(t.T38.P2.mean - t.T40.P2.mean)} and {signed(s.F28.P2.mean - s.F30.P2.mean)} more.</li>
        ) : null}
        {fed && s && fedSets.length ? (
          <li>On the time split, the two-station federation, each station on its own parts and every masked message
            cancelling exactly, scores {fedSets.map((k) => f4(fed[k].P2.mean)).join(", ")} against the pooled
            student&apos;s {fedSets.map((k) => f4(s[k].P2.mean)).join(", ")} ({fedSets.map((k) => d.inputs[k] ?? k).join("; ")}).</li>
        ) : null}
        {d.ckks_recheck ? (
          <li>Under CKKS, the student refitted on the time split without LEAF&apos;s six, on{" "}
            {d.ckks_recheck.rows.toLocaleString("en-US")} of its test flows drawn as the shipped student&apos;s were, in{" "}
            {d.ckks_recheck.ciphertexts} ciphertexts: {d.ckks_recheck.decisions_differ} decisions differ from the
            plaintext student&apos;s, the largest logit error {d.ckks_recheck.max_abs_logit_error.toExponential(2)}.</li>
        ) : null}
        <li>Before any model, a single threshold on one input reaches an F1 of 0.99 for no class on any split
          ({flagged} flagged){top.length ? <>; the highest on the time split: {top.map((x) => `${x.input} for ${x.class} ${x.f1.toFixed(2)}`).join(", ")}</> : null}.</li>
        {ct ? (
          <li>Across testbeds ({ct.dataset}, {ct.licence}, without retraining, on the classes both share and {ct.shared_rows.toLocaleString("en-US")} of its
            flows): the teacher on the mapped inputs scores a macro-F1 of {show(ct.teacher.macro_f1)} and the student{" "}
            {show(ct.student.macro_f1)}; as Benign against any attack, {show(ct.teacher.binary_attack_f1)} and{" "}
            {show(ct.student.binary_attack_f1)}.</li>
        ) : null}
      </ul>
      <p className="small muted" style={{ marginTop: 6 }}>
        Stated as measured; none decides what ships. The page&apos;s models and the published ones are unchanged by this
        card: the square network the page has run for 5G-NIDD since batch 12, in place of the linear student, is this
        recipe on new seeds, adopted by a pre-registration of its own
        (<code>references/telecom-nidd-square-student-preregistration.md</code>). Pre-registration: <code>{d.preregistration.file}</code>; record: <code>{d.record}</code>. Data:{" "}
        {d.attribution.dataset}, <a href={d.attribution.licence_url} target="_blank" rel="noreferrer">{d.attribution.licence}</a>;{" "}
        {d.attribution.citation}{ct ? <> Across testbeds: <a href={ct.source} target="_blank" rel="noreferrer">{ct.citation}</a>.</> : null}
      </p>
    </div>
  );
}
