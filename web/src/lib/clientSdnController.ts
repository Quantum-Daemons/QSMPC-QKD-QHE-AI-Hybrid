// Browser mirror of src/qsh/qkd/sdn_controller.py + kme.py — the ETSI GS QKD 015 V2.1.1 control
// plane and the GS QKD 014 V1.1.1 key-delivery surface.
//
// Why this exists: the demo talks about QKD and had no control plane, while an "Agentic AI" panel
// sat next to it. GS QKD 015 clause 1 is explicit — "Application layer interfaces and
// quantum-channel interfaces are out of scope" — so an MCP agent is a **SAE** (Secure Application
// Entity), an application-plane key consumer, never the controller. This module supplies the
// missing middle so the page can draw three honest planes instead of implying two.
//
// Field names (qkdn_id, qkdl_id, qkdl_type, virt_next_hop, phys_channel_att, skr/eskr …) are taken
// verbatim from the published YANG at forge.etsi.org/rep/qkd/gs015-ctrl-int-sdn, revision
// 2022-01-30, BSD-3-Clause. Nothing is renamed to read more nicely.
//
// The quantum layer is SIMULATED — there is no QKD hardware anywhere in this project and every
// SKR/QBER value comes from the analytic channel models in clientQKD. Standards-faithful control
// plane over a simulated quantum layer is an established pattern, not a shortcut: arXiv:2607.06602
// (2026-07) validated a production VPN against a simulated KME before moving to hardware.

/** identity QKD-LINK-TYPES: PHYS = direct quantum channel; VIRT = multi-hop via trusted relays. */
export type QkdLinkType = "PHYS" | "VIRT";

export interface SdQkdNode {
  qkdn_id: string;
  qkdn_location_id: string;
  qkdn_status: "NEW" | "OPERATING" | "DOWN" | "FAILURE" | "OUT";
  key_relay_mode_enable: boolean;
}

export interface QkdLink {
  qkdl_id: string;
  qkdl_type: QkdLinkType;
  qkdl_local_qkdn_id: string;
  qkdl_remote_qkdn_id: string;
  /** Ordered intermediate nodes for a VIRT link. Each one handles the key in plaintext. */
  virt_next_hop: string[];
  phys_channel_att: number | null;
  skr: number;
  eskr: number;
  qkdl_applications: string[];
}

export interface QkdApplication {
  app_id: string;
  server_app_id: string;
  client_app_id: string[];
  backing_qkdl_id: string[];
  consumed_bits: number;
}

export interface SdnTopology {
  spec: string;
  nodes: SdQkdNode[];
  links: QkdLink[];
  apps: QkdApplication[];
  notifications: string[];
  simulated: true;
}

const trustNote = (l: QkdLink): string =>
  l.qkdl_type === "VIRT" && l.virt_next_hop.length
    ? `relayed through ${l.virt_next_hop.length} trusted node(s) [${l.virt_next_hop.join(" → ")}] — `
      + "each sees the key in the clear"
    : "direct quantum channel — no intermediate node handles the key";

export const linkTrustNote = trustNote;

/**
 * Build the control-plane view for a run.
 *
 * Two topologies, chosen by what the run actually did rather than by a switch: a direct PHYS link
 * when the two parties are adjacent, and a VIRT association through one relay when the configured
 * distance exceeds a single span. The relay is what makes the trusted-node caveat visible, which is
 * the part of QKD networking that a demo usually hides and that ANSSI/BSI and QCNC 2026
 * (arXiv:2604.05599) both press on.
 */
export function buildTopology(opts: {
  usecase: string;
  distanceKm: number;
  skrBitsPerSec: number;
  qkdEnabled: boolean;
  saeId?: string;
  /** Longest single quantum span before a trusted relay is required, km. */
  maxSpanKm?: number;
}): SdnTopology {
  const { usecase, distanceKm, skrBitsPerSec, qkdEnabled } = opts;
  const maxSpan = opts.maxSpanKm ?? 80;
  const saeId = opts.saeId ?? "sae-mcp-agent";

  const a: SdQkdNode = { qkdn_id: "qkdn-a", qkdn_location_id: `${usecase} · party A`,
    qkdn_status: qkdEnabled ? "OPERATING" : "OUT", key_relay_mode_enable: true };
  const b: SdQkdNode = { qkdn_id: "qkdn-b", qkdn_location_id: `${usecase} · party B`,
    qkdn_status: qkdEnabled ? "OPERATING" : "OUT", key_relay_mode_enable: true };

  const notifications: string[] = [];
  const nodes: SdQkdNode[] = [a, b];
  const links: QkdLink[] = [];

  if (!qkdEnabled) {
    notifications.push("PQC-only profile: no quantum channel, so the controller has no key "
      + "association to manage. PQC hybrid key establishment carries the session on its own.");
    return { spec: "ETSI GS QKD 015 V2.1.1 (2022-04)", nodes, links, apps: [], notifications,
      simulated: true };
  }

  if (distanceKm > maxSpan) {
    const relay: SdQkdNode = { qkdn_id: "qkdn-relay", qkdn_location_id: "trusted relay · mid-span",
      qkdn_status: "OPERATING", key_relay_mode_enable: true };
    nodes.splice(1, 0, relay);
    const half = distanceKm / 2;
    links.push(
      { qkdl_id: "qkdl-a-r", qkdl_type: "PHYS", qkdl_local_qkdn_id: "qkdn-a",
        qkdl_remote_qkdn_id: "qkdn-relay", virt_next_hop: [], phys_channel_att: half * 0.2,
        skr: skrBitsPerSec, eskr: skrBitsPerSec, qkdl_applications: [] },
      { qkdl_id: "qkdl-r-b", qkdl_type: "PHYS", qkdl_local_qkdn_id: "qkdn-relay",
        qkdl_remote_qkdn_id: "qkdn-b", virt_next_hop: [], phys_channel_att: half * 0.2,
        skr: skrBitsPerSec, eskr: skrBitsPerSec, qkdl_applications: [] },
      // Clause 5.4. A relayed association is never better than its weakest hop.
      { qkdl_id: "qkdl-a-b", qkdl_type: "VIRT", qkdl_local_qkdn_id: "qkdn-a",
        qkdl_remote_qkdn_id: "qkdn-b", virt_next_hop: ["qkdn-relay"], phys_channel_att: null,
        skr: skrBitsPerSec, eskr: skrBitsPerSec, qkdl_applications: [saeId] },
    );
    notifications.push(`${distanceKm} km exceeds the ${maxSpan} km single-span limit, so the `
      + "controller created a VIRT key association through a trusted relay (clause 5.4). "
      + "That relay sees the key in the clear — end-to-end secrecy now rests on its physical security.");
  } else {
    links.push({ qkdl_id: "qkdl-a-b", qkdl_type: "PHYS", qkdl_local_qkdn_id: "qkdn-a",
      qkdl_remote_qkdn_id: "qkdn-b", virt_next_hop: [], phys_channel_att: distanceKm * 0.2,
      skr: skrBitsPerSec, eskr: skrBitsPerSec, qkdl_applications: [saeId] });
    notifications.push(`PHYS link created over a single ${distanceKm} km span (clause 5.3) — `
      + "no intermediate node handles the key.");
  }

  const apps: QkdApplication[] = [{
    app_id: saeId, server_app_id: "qsh-orchestrator", client_app_id: ["browser-agent"],
    backing_qkdl_id: [links[links.length - 1].qkdl_id], consumed_bits: 0,
  }];
  notifications.push(`SAE ${saeId} registered against ${apps[0].backing_qkdl_id[0]} (clause 5.2). `
    + "It requests keys and issues intents; it does not control the quantum layer.");

  return { spec: "ETSI GS QKD 015 V2.1.1 (2022-04)", nodes, links, apps, notifications,
    simulated: true };
}

/**
 * Charge key material to a registered SAE, bounded by the association's remaining budget.
 * Mirrors SdnController.allocate: the controller allocates, the KME delivers. Returns the bits
 * actually granted, which may be fewer than requested — that is the honest answer, not a padded one.
 */
export function allocate(topo: SdnTopology, appId: string, bits: number): number {
  const app = topo.apps.find((x) => x.app_id === appId);
  if (!app) return 0;
  const backing = topo.links.filter((l) => app.backing_qkdl_id.includes(l.qkdl_id));
  const budget = backing.reduce((s, l) => s + l.eskr, 0);
  const granted = Math.max(0, Math.floor(Math.min(bits, budget)));
  app.consumed_bits += granted;
  for (const l of backing) l.eskr = Math.max(0, l.eskr - granted);
  return granted;
}

/**
 * GS QKD 014 V1.1.1 Status data format value, quoted from the specification's own example block:
 * "key_size": 352, described there as "Default size of key the KME can deliver to the SAE (in bit)".
 * It was previously written twice as a bare literal, once for the field and once as the divisor for
 * `stored_key_count`, so editing one and not the other would have made the two silently disagree.
 */
const QKD014_KEY_SIZE_BITS = 352;

/**
 * How many keys this SIMULATOR will hand out per request. The specification's example advertises 128;
 * this is deliberately lower because the demo derives key material in the browser rather than from a
 * QKD device. It is a property each KME advertises for itself, not a normative constant - but the
 * divergence from the published example is stated rather than left as an unexplained literal.
 */
const SIM_MAX_KEY_PER_REQUEST = 8;

/** GS QKD 014 V1.1.1 Status data format — the only published edition; ed2 is still a draft. */
export function kmeStatus(topo: SdnTopology, slaveSaeId = "sae-party-b"): Record<string, unknown> {
  const link = topo.links[topo.links.length - 1];
  return {
    source_KME_ID: "kme-a", target_KME_ID: "kme-b",
    master_SAE_ID: topo.apps[0]?.app_id ?? "sae-mcp-agent", slave_SAE_ID: slaveSaeId,
    key_size: QKD014_KEY_SIZE_BITS,
    stored_key_count: Math.floor((link?.eskr ?? 0) / QKD014_KEY_SIZE_BITS),
    max_key_per_request: SIM_MAX_KEY_PER_REQUEST,
    status_extension: {
      spec: "ETSI GS QKD 014 V1.1.1 (2019-02)",
      simulated: true,
      note: "Keys come from this project's QRNG + extractor path, not from a QKD device.",
      authentication: "GS QKD 014 requires mutual TLS with the SAE identity taken from the client "
        + "certificate; the in-browser model has no transport to attach that to.",
    },
  };
}
