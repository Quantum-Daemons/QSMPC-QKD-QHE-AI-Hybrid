// The QKD control plane, drawn as its own plane rather than folded into the agent panel.
//
// ETSI GS QKD 015 V2.1.1 clause 1: "Application layer interfaces and quantum-channel interfaces are
// out of scope." So the MCP agent above is a SAE — an application-plane key consumer — and the
// controller is a separate, deterministic element. Showing them as one box would be the same class
// of error this project keeps finding: a label asserting something the architecture does not do.
import { useMemo } from "react";
import {
  allocate, buildTopology, kmeStatus, linkTrustNote, type SdnTopology,
} from "../lib/clientSdnController";

export default function SdnControlPlanePanel({
  usecase, distanceKm, keyBudgetBits, qkdEnabled, maxSpanKm,
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
}) {
  const topo: SdnTopology = useMemo(() => {
    const t = buildTopology({ usecase, distanceKm, keyBudgetBits, qkdEnabled, maxSpanKm });
    // Draw down a realistic slice so the remaining budget is visibly finite. This said "so eskr
    // differs from skr" while both fields held the same session total - the sentence described a
    // difference between two copies of one number.
    if (t.apps.length) allocate(t, t.apps[0].app_id, Math.round(keyBudgetBits * 0.25));
    return t;
  }, [usecase, distanceKm, keyBudgetBits, qkdEnabled, maxSpanKm]);

  const kme = kmeStatus(topo);
  const ext = (kme.status_extension ?? {}) as Record<string, string>;

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>QKD control plane · ETSI GS QKD 015</h3>
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
            {/* `skr` keeps the spec's name and is empty; the column beside it says what this run
                actually produced, with its unit. Renaming the ETSI field would break the rule that
                field names come from the published YANG; filling it with a total broke a bigger
                one. */}
            <tr><th>qkdl_id</th><th>qkdl_type</th><th>path</th><th className="num">skr (bit/s)</th>
              <th className="num">key budget</th><th>trust</th></tr>
            {topo.links.map((l) => (
              <tr key={l.qkdl_id} className={l.qkdl_type === "VIRT" ? "metric-companion" : undefined}>
                <td><code>{l.qkdl_id}</code></td>
                <td><code>{l.qkdl_type}</code></td>
                <td className="muted small">
                  {l.qkdl_local_qkdn_id}
                  {l.virt_next_hop.length ? ` → ${l.virt_next_hop.join(" → ")} → ` : " → "}
                  {l.qkdl_remote_qkdn_id}
                </td>
                {/* The spec's rate fields, empty because this simulation has no clock to
                    produce one. They held a session total until Round 42. */}
                <td className="num muted" title="ETSI skr is a rate in bit/s; this simulation has no wall-clock model of the quantum channel, so no rate is reported">n/a</td>
                <td className="num">{l.key_budget_bits.toLocaleString()} bit</td>
                <td className="muted small">{linkTrustNote(l)}</td>
              </tr>
            ))}
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
