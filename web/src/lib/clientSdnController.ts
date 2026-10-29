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
  /** ETSI GS QKD 015's secret key RATE, bit/s. Derived as (bits per pulse) x (assumed source rate).
   *
   *  It held a session TOTAL until Round 42, which set it to `null` because bit/s needs a clock and
   *  none existed. Round 44 supplies the clock as an explicit, labelled assumption rather than
   *  leaving a spec field permanently empty, so the field now carries the quantity its name
   *  promises. `null` still means "this run produced no rate" - PQC-only, or before a run.
   *
   *  ASSIGNED ONCE, at link construction, from a value whose identifier says BitPerSec. Never from
   *  anything named `bits`. */
  skr: number | null;
  /** Bits handed out on this link. The LEDGER, and the only thing `allocate()` writes besides the
   *  budget. Per link, because consumption is per link. */
  consumed_bits: number;
  /** What this run actually produced, in bits - the quantity the allocation budget is denominated
   *  in, and the one the panel renders with its unit. */
  key_budget_bits: number;
  qkdl_applications: string[];
}

/**
 * `eskr` - effective secret key rate after the applications' consumption, bit/s.
 *
 * HAS NO STORAGE, AND THAT IS THE POINT. Round 42's defect was that `allocate()` subtracted granted
 * BITS from `eskr`, a field documented as bit/s: the rate was doubling as the budget ledger. A
 * comment asking people not to do that again is weaker than removing the thing to be written, so
 * `eskr` is computed on read and there is no field to decrement.
 *
 * The conversion is the one place a total legitimately meets a rate: bits consumed over the channel
 * time the run implies. Returns `null` - never 0 - when there is no rate or no channel time, because
 * `0 bit/s` is a claim about a link and absence is not.
 */
export function linkEskr(link: QkdLink, topo: SdnTopology): number | null {
  if (link.skr == null) return null;
  const secs = topo.channel_seconds;
  if (secs == null || !(secs > 0)) return null;
  const consumptionRate = link.consumed_bits / secs;     // bit/s
  const eskr = link.skr - consumptionRate;
  return Number.isFinite(eskr) ? Math.max(0, eskr) : null;
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
  /** Seconds of quantum channel this run implies: pulses / source rate. `null` when no clock was
   *  supplied or no quantum channel ran. `linkEskr` needs it and refuses to guess, which is why it
   *  lives on the topology rather than being recomputed per link by a caller that may not have it. */
  channel_seconds: number | null;
  /** The ASSUMED source repetition rate, Hz, carried so every consumer can state what the bit/s
   *  figures rest on without reaching back into config. */
  source_rate_hz: number | null;
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
  /** Key material this run PRODUCED, in BITS. Not a rate, and the name says so.
   *
   *  It was `skrBitsPerSec`, and the caller passes `secure_key_bits` - a total, 1915 - which then
   *  landed in ETSI GS QKD 015's `skr` and `eskr`. `src/qsh/qkd/sdn_controller.py` documents those
   *  fields as `secret key rate, bit/s` and logs them that way, so a reader who knows the spec read
   *  a session total as bits per second. The checklist already carries the sentence that condemns
   *  it, about these same two quantities: "`secure_key_bits` must equal `skr` times the pulse count,
   *  NOT `skr` itself ... Dimensional mismatch is the check."
   *
   *  This stays the ALLOCATION LEDGER and stays in bits. `allocate()` draws down from it. The rate
   *  fields below are derived and are never decremented - letting the rate become the ledger again
   *  is the defect Round 42 removed, and it would come back wearing a correct unit.
   */
  keyBudgetBits: number;
  /** Bits of secret key per PULSE, as the BB84/E91 models compute it (`res.secretKeyRate`).
   *
   *  Dimensionless per-pulse fraction. Multiplied by `sourceRateHz` below it gives the bit/s the
   *  ETSI fields are defined to carry. Passed in rather than divided out of `keyBudgetBits`, because
   *  `secure_key_bits` is floored and rounded on the way here and dividing it back would recover a
   *  slightly different number than the model produced.
   */
  secretKeyRatePerPulse: number;
  /** Source repetition rate in Hz - THE ONE ASSUMED QUANTITY, and the reason a bit/s exists at all.
   *
   *  Not measured and not measurable here: `docs/findings.md` records that no QKD device exists in this
   *  project, "a boundary of the deliverable, not a gap in it". A repetition rate is a property of a
   *  transmitter, stated in a paper's experimental section rather than derived from one, so the only
   *  honest ways to have a bit/s are to assume a clock openly or to publish none.
   *
   *  Round 42 published none, and that was defensible. It also left the spec's rate fields carrying
   *  `null` while the panel showed `n/a`, and - because nothing looked for READERS of the emptied
   *  field - left `stored_key_count` computing `(eskr ?? 0) / 352` and advertising zero stored keys
   *  on every run for two rounds.
   *
   *  Default 100 MHz, from arXiv:2607.24118 (CC BY 4.0), whose v2 (2026-10-16) reports an average
   *  finite-key 110 kbps at QBER 3.28% over a 2 km free-space channel (v1: 164.8 kbps). THE CLOCK IS BORROWED AND THE EXPERIMENT IS NOT: that paper's
   *  channel is 2 km horizontal and this model's default is 50 km of fibre, so a derived bit/s here
   *  is this simulation's own number under an assumed clock, never a reproduction of that result.
   */
  sourceRateHz: number;
  /** Channel uses this run made. With `sourceRateHz` it gives the channel time; on its own it is
   *  the count the per-pulse models already work in. */
  pulses: number;
  qkdEnabled: boolean;
  saeId?: string;
  /** Longest single quantum span before a trusted relay is required, km. */
  maxSpanKm?: number;
}): SdnTopology {
  const { usecase, distanceKm, keyBudgetBits, qkdEnabled, secretKeyRatePerPulse, sourceRateHz } = opts;
  // THE ONE MULTIPLICATION THIS ROUND IS ABOUT: bits/pulse x pulses/s = bit/s, which is what ETSI
  // GS QKD 015 defines `skr` to be. `eskr` is the same rate after the applications' expected
  // consumption; both are DERIVED and neither is ever decremented. `key_budget_bits` remains the
  // ledger. A rate that falls as bits are spent would be Round 42's defect returning in a correct
  // unit, which is the single most likely way this change ships its own bug.
  const skr = qkdEnabled && sourceRateHz > 0 ? secretKeyRatePerPulse * sourceRateHz : null;
  // pulses / (pulses per second) = seconds. The channel time this run implies, and the denominator
  // that turns consumed BITS into a consumption RATE inside `linkEskr`. Guarded so a zero or absent
  // clock yields null rather than Infinity - an infinite rate would render, and would be believed.
  const channelSeconds = qkdEnabled && sourceRateHz > 0 ? opts.pulses / sourceRateHz : null;
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
    // Nulls, not zeros. There is no quantum channel on this profile, so there is no rate and no
    // channel time - and `0 bit/s` would be a claim about a link that does not exist.
    return { spec: "ETSI GS QKD 015 V2.1.1 (2022-04)", nodes, links, apps: [], notifications,
      channel_seconds: null, source_rate_hz: null, simulated: true };
  }

  if (distanceKm > maxSpan) {
    const relay: SdQkdNode = { qkdn_id: "qkdn-relay", qkdn_location_id: "trusted relay · mid-span",
      qkdn_status: "OPERATING", key_relay_mode_enable: true };
    nodes.splice(1, 0, relay);
    const half = distanceKm / 2;
    links.push(
      { qkdl_id: "qkdl-a-r", qkdl_type: "PHYS", qkdl_local_qkdn_id: "qkdn-a",
        qkdl_remote_qkdn_id: "qkdn-relay", virt_next_hop: [], phys_channel_att: half * 0.2,
        skr, consumed_bits: 0, key_budget_bits: keyBudgetBits, qkdl_applications: [] },
      { qkdl_id: "qkdl-r-b", qkdl_type: "PHYS", qkdl_local_qkdn_id: "qkdn-relay",
        qkdl_remote_qkdn_id: "qkdn-b", virt_next_hop: [], phys_channel_att: half * 0.2,
        skr, consumed_bits: 0, key_budget_bits: keyBudgetBits, qkdl_applications: [] },
      // Clause 5.4. A relayed association is never better than its weakest hop.
      { qkdl_id: "qkdl-a-b", qkdl_type: "VIRT", qkdl_local_qkdn_id: "qkdn-a",
        qkdl_remote_qkdn_id: "qkdn-b", virt_next_hop: ["qkdn-relay"], phys_channel_att: null,
        skr, consumed_bits: 0, key_budget_bits: keyBudgetBits, qkdl_applications: [saeId] },
    );
    notifications.push(`${distanceKm} km exceeds the ${maxSpan} km single-span limit, so the `
      + "controller created a VIRT key association through a trusted relay (clause 5.4). "
      + "That relay sees the key in the clear — end-to-end secrecy now rests on its physical security.");
  } else {
    links.push({ qkdl_id: "qkdl-a-b", qkdl_type: "PHYS", qkdl_local_qkdn_id: "qkdn-a",
      qkdl_remote_qkdn_id: "qkdn-b", virt_next_hop: [], phys_channel_att: distanceKm * 0.2,
      skr, consumed_bits: 0, key_budget_bits: keyBudgetBits, qkdl_applications: [saeId] });
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
    channel_seconds: channelSeconds, source_rate_hz: sourceRateHz, simulated: true };
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
  // Denominated in BITS, which is what an allocation is. This drew down `eskr` until Round 42 -
  // the ETSI rate field was doubling as the budget ledger, which is why a session total ended up
  // in it in the first place. Subtracting granted bits from a bit/s field is the dimensional error
  // one layer below the one the panel displayed.
  const budget = backing.reduce((s, l) => s + l.key_budget_bits, 0);
  const granted = Math.max(0, Math.floor(Math.min(bits, budget)));
  app.consumed_bits += granted;
  for (const l of backing) {
    l.key_budget_bits = Math.max(0, l.key_budget_bits - granted);
    l.consumed_bits += granted;
  }
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
    // FROM THE BIT COUNT, not from a rate. This read `(link?.eskr ?? 0) / 352`, and when Round 42
    // set `eskr` to null the `?? 0` turned an absence into a measurement: the table advertised zero
    // stored keys on every run for two rounds and nothing failed. A key COUNT is bits divided by a
    // key size; `key_budget_bits` is the bits this run actually has left to give out.
    stored_key_count: Math.floor((link?.key_budget_bits ?? 0) / QKD014_KEY_SIZE_BITS),
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
