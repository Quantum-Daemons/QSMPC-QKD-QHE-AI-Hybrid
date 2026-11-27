// 5G-NIDD federated over eight Dirichlet-split clients, with a 1D-CNN baseline (Round 86, batch 8; the owner's choice
// of 2026-11-27). Pre-registered in references/telecom-nidd-dirichlet-preregistration.md and run offline by
// scripts/train/telecom_nidd_dirichlet.py, whose web/public/data/telecom_nidd/dirichlet.json this reads; rendered inside
// the two-station card, and nothing until that file exists.
import { useEffect, useState } from "react";

type Stat = { mean: number; sd: number | null };
type Summary = Record<"flamby_metric" | "union_macro_f1" | "union_weighted_f1" | "union_accuracy", Stat>;
interface Dirichlet {
  preregistration: { file: string }; record: string; clients: number; alpha: number; device_cnn: string;
  summary: Record<"linear_fedavg_masked" | "cnn_fedavg" | "cnn_pooled", Summary>;
  C1: { linear_eight_clients_mean_flamby: number; pooled_flamby: number; gap: number; within_0_01: boolean };
  C2: { cnn_minus_linear_mean_flamby: number; ci95: [number, number] | null; seeds: number };
  references: { two_station_federated_flamby: number; pooled_flamby: number; alone_flamby: number[] };
}

// Four decimals: the pooled 1D-CNN's deviation over the five seeds is 0.0001, which three would print as zero.
const f4 = (v: number) => v.toFixed(4);
const cell = (s: Stat) => (s.sd == null ? f4(s.mean) : `${f4(s.mean)} (${f4(s.sd)})`);
const signed = (v: number) => `${v >= 0 ? "+" : ""}${f4(v)}`;

export default function NiddDirichletSection() {
  const [d, setD] = useState<Dirichlet | null>(null);
  useEffect(() => {
    let on = true;
    fetch(`${import.meta.env.BASE_URL}data/telecom_nidd/dirichlet.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: Dirichlet | null) => { if (on && j?.summary) setD(j); })
      .catch(() => undefined);
    return () => { on = false; };
  }, []);
  if (!d) return null;
  const s = d.summary;
  const rows: { label: string; v: Summary }[] = [
    { label: `Linear, masked FedAvg, ${d.clients} clients`, v: s.linear_fedavg_masked },
    { label: `1D-CNN, FedAvg in the clear, ${d.clients} clients`, v: s.cnn_fedavg },
    { label: "1D-CNN, pooled (centralised reference)", v: s.cnn_pooled },
  ];
  const c2 = d.C2;
  return (
    <div style={{ marginTop: 10 }}>
      <h4 style={{ margin: "6px 0 2px" }}>{d.clients} clients split by a Dirichlet draw, and a 1D-CNN beside the linear model</h4>
      <p className="small" style={{ margin: 0 }}>
        A pre-registered comparison, run offline: both stations&apos; fit rows divided among {d.clients} clients by class
        proportions drawn from a Dirichlet distribution at alpha {d.alpha} (Yurochkin et al.&apos;s partition, the
        set-up of Jeong et al., CMC 87(1), 2026, on 5G-NIDD), five partitions; the linear model and masked FedAvg above with{" "}
        {d.clients} clients; and a 1D-CNN after Jeong et al.&apos;s, as this project reads it (one fully connected layer of 128,
        where their section 5.2 has two, of 128 and 64), under FedAvg in the clear and pooled.
        Each cell is the mean over the five seeds, with the standard deviation in brackets, on the same test flows.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="small" style={{ width: "100%", borderCollapse: "collapse", marginTop: 4 }}>
          <thead><tr><th scope="col" style={{ textAlign: "left" }}>Model</th><th scope="col">FLamby metric</th>
            <th scope="col">macro-F1, all test flows</th><th scope="col">weighted F1</th><th scope="col">accuracy</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}><td>{r.label}</td><td style={{ textAlign: "center" }}>{cell(r.v.flamby_metric)}</td>
                <td style={{ textAlign: "center" }}>{cell(r.v.union_macro_f1)}</td>
                <td style={{ textAlign: "center" }}>{cell(r.v.union_weighted_f1)}</td>
                <td style={{ textAlign: "center" }}>{cell(r.v.union_accuracy)}</td></tr>
            ))}
            <tr className="muted"><td>Linear, masked FedAvg, the two stations (above)</td>
              <td style={{ textAlign: "center" }}>{f4(d.references.two_station_federated_flamby)}</td><td colSpan={3} /></tr>
            <tr className="muted"><td>Linear, pooled (above)</td>
              <td style={{ textAlign: "center" }}>{f4(d.references.pooled_flamby)}</td><td colSpan={3} /></tr>
          </tbody>
        </table>
      </div>
      <p className="small" style={{ marginTop: 6 }}>
        Over {d.clients} clients the linear model {d.C1.within_0_01
          ? <>comes within 0.01 of pooling ({signed(-d.C1.gap)}), as the pre-registration&apos;s first comparison asked.</>
          : <>is {f4(d.C1.gap)} below pooling, outside the 0.01 the pre-registration&apos;s first comparison asked.</>}{" "}
        The 1D-CNN under FedAvg scores {signed(c2.cnn_minus_linear_mean_flamby)} against the linear model on the same
        clients{c2.ci95 ? <> (95% interval {signed(c2.ci95[0])} to {signed(c2.ci95[1])} over the {c2.seeds} paired seeds)</> : null},
        the second comparison, stated as measured; neither decides what ships.
      </p>
      <p className="small muted" style={{ marginTop: 4 }}>
        With {d.clients} clients a client learns from the sum only the sum of the other {d.clients - 1}, so the masks hide one
        client&apos;s update from each other client unless the {d.clients - 2} others collude; the 1D-CNN arms are baselines
        in the clear, neither masked nor encrypted, and the page&apos;s encrypted model stays the two-station linear one.
        Like the two stations&apos; figures, these come from the student&apos;s random split and its six flagged inputs
        (the card&apos;s note below).
        The 1D-CNN ran on {d.device_cnn === "cpu" ? "the CPU" : "the GPU"}. Pre-registration:{" "}
        <code>{d.preregistration.file}</code>; record: <code>{d.record}</code>.
      </p>
    </div>
  );
}
