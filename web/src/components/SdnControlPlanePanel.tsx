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
import Details from "./Details";

/** The bound a run's rate rests on, from the decoy states it reported (`qkd_decoy_states`, Round 78).
 *  The table named the one-decoy Eq. (41) on every run, including the satellite's Vacuum+Weak runs
 *  since Round 77 and E91 runs, which have no decoy states at all. */
function boundOf(decoyStates: string | undefined): { short: string; paper: string; source: string } {
  const decoyPaper = "Ma, Qi, Zhao & Lo, Phys. Rev. A 72, 012326 (2005)";
  if (decoyStates === "vacuum + one decoy") {
    return { short: "vacuum + weak decoy-state bound", paper: decoyPaper,
      source: "section 3.4, Eqs. (34) and (37): a signal, one decoy and the vacuum state, whose yield is measured" };
  }
  if (decoyStates?.startsWith("none")) {
    // Not the decoy paper: an entangled source's analysis is Ma, Fung & Lo's, as the facts row cites it.
    return { short: "E91 secure fraction (device-dependent; no decoy states)",
      paper: "Ma, Fung & Lo, Phys. Rev. A 76, 012307 (2007)",
      source: "an entangled source has no decoy states: 1 - h(e) - f h(e) on the model error rate" };
  }
  return { short: "one-decoy bound", paper: decoyPaper,
    source: "section 3.5, Eq. (41): a signal and one decoy, vacuum yield not assumed" };
}

/** One wording, used by both rate cells, so the caveat cannot drift between two columns. The ratio
 *  to Li et al.'s measured rate is computed: it was a literal, "3.8x", which no run had matched since
 *  Round 76's detector set. */
function rateCellTitle(perPulse: number, decoyStates: string | undefined): string {
  // Before any run there is no bound to name: `decoyStates` is the run's own metric.
  if (decoyStates === undefined) return "bit/s - not yet run";
  const ratio = perPulse > 0 ? `${(8.88e-3 / perPulse).toFixed(1)}x this run's figure per pulse` : "above this run's figure";
  const b = boundOf(decoyStates);
  return `bit/s - this run's ${b.short} per pulse sent (${b.paper}) multiplied by an ASSUMED source `
    + "repetition rate. ASYMPTOTIC: no finite-key "
    + "correction is applied to this rate (the satellite's key budget is a block's finite key since "
    + "Round 79; the rate is not). It is the infinite-key limit of a LOWER bound, "
    + "not a ceiling on real hardware - Li et al. (Nat. Photon. 17, 416-421, 2023) measured 22.2 Mb/s "
    + `at 50 km on a 2.5 GHz clock, 8.88e-3 bit/pulse, ${ratio}, with full composable finite-key `
    + "security. No QKD hardware exists in this project and nothing here measures a clock: the rate "
    + "scales exactly with the Source rate (MHz) control in the Parameters panel.";
}
export default function SdnControlPlanePanel({
  usecase, distanceKm, keyBudgetBits, qkdEnabled, maxSpanKm,
  secretKeyRatePerPulse, sourceRateHz, pulses, decoyStates,
}: {
  /** The decoy states the run's rate rests on (`metrics.qkd_decoy_states`). */
  decoyStates?: string;
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
        <div style={{ marginTop: 0 }}>
          {/* 214 words became a table of how the number is built, with the comparison to published
              systems and the correction history in a fold. What stays visible is the assumption itself,
              because a bit/s figure invites comparison whether or not one is offered. */}
          <p className="muted small" style={{ marginTop: 0 }}>
            <b>skr and eskr are bit/s, and they rest on an assumption.</b> <b>No QKD device exists in this
            project</b> and nothing here measures a clock: the rate is the <b>Source rate (MHz)</b>{" "}
            control in the Parameters panel, and every figure in these two columns scales with it.
          </p>
          <table className="facts-table small">
            <thead><tr><th>Factor</th><th>Value in this run</th><th>Where it comes from</th></tr></thead>
            <tbody>
              {/* BEFORE A RUN there is no bound to name. `decoyStates` is the run's own metric, and the
                  row read "0 bit/pulse, this run's one-decoy bound" on a page that had not run and
                  whose default source sends the vacuum (found live in Round 79). */}
              {decoyStates === undefined
                ? <tr><td>secure fraction per pulse sent</td><td>not yet run</td><td>the decoy-state bound the run&apos;s source supports, named once a run has chosen it</td></tr>
                : <tr><td>secure fraction per pulse sent</td><td><b>{secretKeyRatePerPulse > 0 ? secretKeyRatePerPulse.toExponential(3) : "0"}</b> bit/pulse, this run&apos;s {boundOf(decoyStates).short}{usecase === "satellite_comms" ? null : <>, on gains corrected for the detectors&apos; dead time</>}{decoyStates.startsWith("none") ? <>; every pair counts</> : <>; only the signal pulses carry key</>}</td><td>{boundOf(decoyStates).paper}, {boundOf(decoyStates).source}; <b>asymptotic</b> - no finite-key correction, so the infinite-key limit of a lower bound{usecase === "satellite_comms" ? null : <>. The dead-time factor is Rusca et al. 2018&apos;s Eq. (B3), counting every click</>}</td></tr>}
              <tr><td>x source repetition rate</td><td><b>{(topo.source_rate_hz / 1e6).toLocaleString()} MHz</b></td><td>{usecase === "satellite_comms"
                ? <>an assumption; default 100 MHz, the Micius source&apos;s (Liao et al., <i>Nature</i> <b>549</b>, 43, 2017), whose measured rate is not reproduced or claimed here</>
                : <>an assumption; default 1 GHz, the clock Rusca et al. (<i>Appl. Phys. Lett.</i> <b>112</b>, 171104, 2018) state for this page&apos;s detector set - a simulation set, not a measured link. With their 100 ns dead time the per-pulse rate depends on it too</>}</td></tr>
              <tr><td>channel time</td><td>{topo.channel_seconds == null ? "n/a"
                : topo.channel_seconds >= 1 ? `${topo.channel_seconds.toFixed(1)} s` : `${(topo.channel_seconds * 1e3).toFixed(3)} ms`}</td><td>{usecase === "satellite_comms"
                ? <>the key block&apos;s pulses divided by that rate - one pass by default (Round 79), so the key budget below, the block&apos;s composable finite key, is spent over the pass&apos;s link time</>
                : <>the key block&apos;s pulses divided by that rate (Round 80), so the key budget below, the block&apos;s composable finite key, is spent over the block&apos;s link time</>}</td></tr>
            </tbody>
          </table>
          <Details summary="How this compares with published systems">
            <p>
              arXiv:2607.24118 (preprint, v2 2026-10-16, CC BY 4.0) reports an average finite-key rate of
              110 kbit/s at QBER 3.28% (v1 said 164.8 kbit/s) - its measurement over its own channel, not
              this one&apos;s. Li et al. (<i>Nat. Photon.</i>{" "}
              <b>17</b>, 416-421, 2023) measured <b>22.2 Mb/s at 50 km</b> of fibre on a 2.5 GHz clock,{" "}
              <b>8.88 x 10<sup>-3</sup> bit/pulse</b> with full composable finite-key security
              {secretKeyRatePerPulse > 0
                ? <> - {(8.88e-3 / secretKeyRatePerPulse).toFixed(1)}x this run&apos;s figure per pulse
                    {usecase === "satellite_comms" ? <> (a fibre link against this satellite downlink)</> : null}</>
                : null}, so this bound is not a ceiling on what hardware can do.
            </p>
            <p>
              A round of this project previously called the same number &quot;a modelled upper bound, ~2.4x a
              real deployed link&quot;; that was wrong in both directions and is corrected here.
            </p>
          </Details>
        </div>
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
                <td className="num" title={rateCellTitle(secretKeyRatePerPulse, decoyStates)}>
                  {l.skr == null ? <span className="muted">n/a</span> : Math.round(l.skr).toLocaleString()}
                </td>
                <td className="num" title={rateCellTitle(secretKeyRatePerPulse, decoyStates)}>
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
