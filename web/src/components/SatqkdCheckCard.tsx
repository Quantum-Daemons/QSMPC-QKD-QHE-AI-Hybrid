// SatQuMA's published key lengths recomputed, and this project's bound on the same pass (Round 86, batches 12 and 13;
// the owner's choice of 2026-11-28 to show it). Pre-registered in references/satellite-satqkd-key-length-preregistration.md,
// with an addendum written after its check B's result; scripts/export_outside_data_cards.py writes the file this reads,
// and every figure below is read from it.
import { useEffect, useState } from "react";
import CardExport from "./CardExport";
import StageBadge from "./StageBadge";

interface Row {
  file: string; half_window_s: number; satquma_key: number; this_project_key: number; key_ratio: number;
  key_basis_count_ratio: number; satquma_phase_error: number; this_project_phase_error: number;
}
interface Satqkd {
  preregistration: string; addendum: string; records: string[];
  check_a: { rows: number; rows_with_a_key: number; keyed_equal: number; rows_without_a_key: number; meets: boolean };
  check_b_found_a_key: boolean; check_b_prime: Row[]; attribution: string;
}

const n0 = (v: number) => Math.round(v).toLocaleString("en-US");

export default function SatqkdCheckCard() {
  const [d, setD] = useState<Satqkd | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/satellite/satqkd_check.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: Satqkd | null) => { if (live && j?.check_b_prime) setD(j); })
      .catch(() => undefined);
    return () => { live = false; };
  }, []);
  if (!d) return null;
  const a = d.check_a;
  const rows = [...d.check_b_prime].sort((p, q) => p.half_window_s - q.half_window_s);
  return (
    <div className="card" id="sat-satqkd" style={{ marginTop: 14 }}>
      <h3>SatQuMA&apos;s published key lengths, recomputed, and this page&apos;s bound on the same pass <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        Sidhu et al. computed their satellite finite-key figures with SatQuMA and published the terms behind them. Recomputed from
        each row&apos;s own terms by their Eq. (2), {a.keyed_equal} of the {a.rows_with_a_key} rows with a key give exactly the key
        length they record{a.rows_without_a_key ? <>, and the other {a.rows_without_a_key} rows have none</> : null}. This
        page&apos;s three-intensity bound on the same zenith pass, with the same channel, counts close to SatQuMA (the ratio is
        in the table) and keeps less key, because its phase-error bound is about three times SatQuMA&apos;s; which part of that
        bound differs is not resolved here. The pre-registered comparison read the deposit&apos;s system loss as a loss to add and counted the
        zenith&apos;s loss twice{d.check_b_found_a_key ? null : <>, so it found no key</>}; the one below, chosen after that result
        and labelled so in an addendum, takes the deposit&apos;s total loss.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>Window: either side of the peak</th><th>SatQuMA&apos;s key (bits)</th><th>This page&apos;s bound (bits)</th>
            <th>Ratio</th><th>Key-basis counts, ratio</th><th>Phase error: SatQuMA / here</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.file}>
                <td>&plusmn;{r.half_window_s} s</td><td>{n0(r.satquma_key)}</td><td>{n0(r.this_project_key)}</td><td>{r.key_ratio.toFixed(2)}</td>
                <td>{r.key_basis_count_ratio.toFixed(3)}</td><td>{r.satquma_phase_error.toFixed(3)} / {r.this_project_phase_error.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ marginTop: 6 }}>
        Pre-registered (<code>{d.preregistration}</code>, addendum <code>{d.addendum}</code>); records{" "}
        {d.records.map((f, i) => <span key={f}>{i ? ", " : ""}<code>{f}</code></span>)}. No SatQuMA code was run. Data: {d.attribution}.
      </p>
      <CardExport name="satellite_comms-satqkd-check" label="the SatQKD figures" rows={rows.map((r) => ({ ...r }))} json={d} />
    </div>
  );
}
