// The QKD control plane, drawn as its own plane rather than folded into the agent panel.
//
// ETSI GS QKD 015 V2.1.1 clause 1: "Application layer interfaces and quantum-channel interfaces are
// out of scope." So the MCP agent above is a SAE — an application-plane key consumer — and the
// controller is a separate, deterministic element. Showing them as one box would be the same class
// of error this project keeps finding: a label asserting something the architecture does not do.
import { useMemo } from "react";
import {
  allocate, buildTopology, kmeStatus, linkEskr, linkTrustNote, type SdnTopology,
} from "../lib/clientSdnController";

/** One wording, used by both rate cells, so the caveat cannot drift between two columns. */
const RATE_CELL_TITLE =
  "bit/s - this run's two-intensity decoy-state bound per pulse (Ma, Qi, Zhao & Lo, Phys. Rev. A 72, "
  + "012326, 2005) multiplied by an ASSUMED source repetition rate. No QKD hardware exists in this "
  + "project and nothing here measures a clock, so the rate scales exactly with the Source rate "
  + "(MHz) control in the Parameters panel. A modelled upper bound, not a measurement: Yuan et al. "
  + "(IEEE/OSA JLT 36, 3427, 2018) report over 1 Mb/s at 1 GHz over 50 km of fibre with a protocol "
  + "that sifts about twice as efficiently as the standard BB84 modelled here.";

export default function SdnControlPlanePanel({
  usecase, distanceKm, keyBudgetBits, qkdEnabled, maxSpanKm,
  secretKeyRatePerPulse, sourceRateHz, pulses,
}: {
  usecase: string;
  distanceKm: number;
  /** Longest single quantum span before a trusted relay is required. Infinity for a downlink,
   *  which is one line of sight however long it is; undefined keeps the terrestrial default. */
  maxSpanKm?: number;
  /** Key material this run produced (metrics.secure_key_bits) — the allocation budget. */
  keyBudgetBits: number;
  /** False under the PQC-only profile — there is then no key association to manage. */
  qkdEnabled: boolean;
  /** Bits of secret key per PULSE, as the run reported it (`metrics.skr_per_pulse`). */
  secretKeyRatePerPulse: number;
  /** The ASSUMED source repetition rate in Hz that the run used. Comes from the run's own metric,
   *  not from the params, for the same reason `keyBudgetBits` does: the panel must show what RAN. */
  sourceRateHz: number;
  /** Channel uses this run made. */
  pulses: number;
}) {
  const topo: SdnTopology = useMemo(() => {
    const t = buildTopology({ usecase, distanceKm, keyBudgetBits, qkdEnabled, maxSpanKm,
                              secretKeyRatePerPulse, sourceRateHz, pulses });
    // Draw down a realistic slice so the remaining budget is visibly finite. This said "so eskr
    // differs from skr" while both fields held the same session total - the sentence described a
    // difference between two copies of one number.
    if (t.apps.length) allocate(t, t.apps[0].app_id, Math.round(keyBudgetBits * 0.25));
    return t;
    // Every value the topology reads must be listed, or the panel keeps a stale one. This is
    // the dependency-array class the DAG's node labels already hit: threaded in, used, and
    // omitted here, so React never rebuilt and the label stayed frozen.
  }, [usecase, distanceKm, keyBudgetBits, qkdEnabled, maxSpanKm,
      secretKeyRatePerPulse, sourceRateHz, pulses]);

  const kme = kmeStatus(topo);
  const ext = (kme.status_extension ?? {}) as Record<string, string>;

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>QKD control plane · ETSI GS QKD 015</h3>
      {/* THE ASSUMPTION, ADJACENT TO THE NUMBERS IT GOVERNS. A bit/s figure invites a comparison
          with published systems whether or not one is offered, and the nearest measured figure this
          repository cites belongs to a different channel. Saying so beside the table is cheaper than
          hoping nobody makes the comparison - and this project's own rule is that a figure a reader
          would quote must carry its caveat in the same visual block, not thirty lines below. */}
      {topo.source_rate_hz != null && (
        <p className="muted small" style={{ marginTop: 0 }}>
          <b>skr and eskr are bit/s, and they rest on an assumption.</b>{" "}
          They are computed as this run&apos;s <b>two-intensity decoy-state bound</b> per pulse
          (Ma, Qi, Zhao &amp; Lo, <i>Phys. Rev. A</i> <b>72</b>, 012326, 2005) multiplied by a source
          repetition rate of{" "}
          <b>{(topo.source_rate_hz / 1e6).toLocaleString()} MHz</b>, over a channel time of{" "}
          {topo.channel_seconds != null ? `${(topo.channel_seconds * 1e3).toFixed(3)} ms` : "n/a"}{" "}
          (pulses divided by that rate). <b>No QKD device exists in this project</b> and nothing here
          measures a clock: the rate is the <b>Source rate (MHz)</b> control in the Parameters panel,
          and every figure in these two columns scales with it. Its default is the 100 MHz reported
          by arXiv:2607.24118 (preprint, 2026-07-27, CC BY 4.0) for a 2 km free-space link;{" "}
          <b>that paper&apos;s own 164.8 kbit/s at QBER ~3.3% is its measurement over its own
          channel, not this one&apos;s, and is neither reproduced nor claimed here.</b>{" "}
          Read the result as a <b>modelled upper bound</b>: Yuan et al. (<i>IEEE/OSA J. Lightwave
          Technol.</i> <b>36</b>, 3427, 2018) report over 1 Mb/s at 1 GHz across 50 km of fibre using
          a protocol that sifts about twice as efficiently as the standard BB84 modelled here.
        </p>
      )}
      <p className="muted small" style={{ marginTop: 0 }}>
        Three planes, kept apart on purpose. GS QKD 015 clause 1 puts application interfaces
        <b> out of scope</b> of the control interface, so the Agentic-AI panel is a{" "}
        <b>SAE (Secure Application Entity)</b> — it requests keys and issues intents, and does{" "}
        <b>not</b> control the quantum layer.
      </p>

      <div className="grid cols-3" style={{ marginTop: 12 }}>
        {[
          ["Application", "SAE · MCP agent", "out of scope of QKD 015", "requests keys, issues intents"],
          ["Control", "SDN controller + KME", "GS QKD 015 / 018 · ITU-T Y.3804, Y.3805",
            "topology, path computation, key relay, resource allocation"],
          ["Quantum / link", "QKD modules + SDN agent", "GS QKD 004 / 014 / 020",
            "key generation and delivery"],
        ].map(([plane, who, spec, does]) => (
          <div key={plane} className="card" style={{ padding: 12 }}>
            <div style={{ fontWeight: 700, fontSize: 13 }}>{plane}</div>
            <div style={{ fontSize: 12, marginTop: 2 }}>{who}</div>
            <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>{spec}</div>
            <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>{does}</div>
          </div>
        ))}
      </div>

      <h4 style={{ marginTop: 18, marginBottom: 6, fontSize: 13 }}>
        Key association links <span className="muted" style={{ fontWeight: 400 }}>(clause 4.5)</span>
      </h4>
      {topo.links.length === 0 ? (
        <p className="muted small">No key association — the PQC-only profile carries the session
          on hybrid PQC key establishment alone.</p>
      ) : (
        <table className="mini-table">
          <tbody>
            {/* `skr` and `eskr` now carry the quantity the spec's name promises - bit/s - because
                the run supplies the one factor that was missing: an ASSUMED source repetition rate.
                They held a session total until Round 42, which nulled them; Round 44 fills them from
                (bits per pulse) x (pulses per second) rather than leaving a spec field permanently
                empty because its one input was never asked for. `n/a` still shows where a run
                produced no rate - PQC-only, or before a run - because 0 bit/s is a claim. */}
            <tr><th>qkdl_id</th><th>qkdl_type</th><th>path</th>
              <th className="num">skr (bit/s, assumed clock)</th>
              <th className="num">eskr (bit/s, assumed clock)</th>
              <th className="num">key budget</th><th>trust</th></tr>
            {topo.links.map((l) => {
              const eskr = linkEskr(l, topo);
              return (
              <tr key={l.qkdl_id} className={l.qkdl_type === "VIRT" ? "metric-companion" : undefined}>
                <td><code>{l.qkdl_id}</code></td>
                <td><code>{l.qkdl_type}</code></td>
                <td className="muted small">
                  {l.qkdl_local_qkdn_id}
                  {l.virt_next_hop.length ? ` → ${l.virt_next_hop.join(" → ")} → ` : " → "}
                  {l.qkdl_remote_qkdn_id}
                </td>
                <td className="num" title={RATE_CELL_TITLE}>
                  {l.skr == null ? <span className="muted">n/a</span> : Math.round(l.skr).toLocaleString()}
                </td>
                <td className="num" title={RATE_CELL_TITLE}>
                  {eskr == null ? <span className="muted">n/a</span> : Math.round(eskr).toLocaleString()}
                </td>
                <td className="num">{l.key_budget_bits.toLocaleString()} bit</td>
                <td className="muted small">{linkTrustNote(l)}</td>
              </tr>);
            })}
          </tbody>
        </table>
      )}

      {topo.links.some((l) => l.qkdl_type === "VIRT") && (
        <div className="note" style={{ marginTop: 10, borderColor: "var(--warn)", color: "var(--warn)" }}>
          ⚠ This association is relayed through a <b>trusted node</b>, which holds the key in
          plaintext. End-to-end secrecy therefore rests on that node's physical security, not on
          the quantum channel. This is the standing critique of trusted-relay QKD networks
          (ANSSI/BSI joint position paper; QCNC 2026, arXiv:2604.05599) and it is drawn here rather
          than hidden.
        </div>
      )}

      <h4 style={{ marginTop: 18, marginBottom: 6, fontSize: 13 }}>
        Key delivery <span className="muted" style={{ fontWeight: 400 }}>(GS QKD 014)</span>
      </h4>
      <table className="mini-table">
        <tbody>
          {[["source_KME_ID", kme.source_KME_ID], ["master_SAE_ID", kme.master_SAE_ID],
            ["slave_SAE_ID", kme.slave_SAE_ID], ["key_size", kme.key_size],
            ["stored_key_count", kme.stored_key_count]].map(([k, v]) => (
            <tr key={String(k)}><td><code>{String(k)}</code></td>
              <td className="num">{String(v)}</td></tr>
          ))}
        </tbody>
      </table>

      <p className="muted" style={{ fontSize: 11, marginTop: 10 }}>
        Targets <b>{ext.spec}</b> — still the only published edition; ed2 remains a stable draft.
        The object model, workflows and field names follow the published ETSI YANG; the{" "}
        <b>quantum layer is simulated</b>, so every SKR figure comes from the analytic channel model,
        not from hardware. {ext.authentication}
      </p>
      {topo.notifications.map((n, i) => (
        <div key={i} className="muted" style={{ fontSize: 11, marginTop: 4 }}>· {n}</div>
      ))}
    </div>
  );
}
