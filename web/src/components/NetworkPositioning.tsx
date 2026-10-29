/**
 * Where this demo sits against a network-layer PQC-QKD deployment - the user's companion network demo,
 * which runs WireGuard and strongSwan IKEv2 between real containers.
 *
 * WHAT IS AND IS NOT CLAIMED. The two projects apply the same hybrid idea at two layers; this page
 * neither connects to the network demo nor imports any figure from it - that repository commits no
 * latency or throughput benchmark, so there is none to quote. Its QKD is simulated too (a BB84
 * simulator behind ETSI GS QKD 014), which the caption says. The key schedules this page computes in
 * the browser are the network layer's MATH, not its tunnels: no packet leaves the tab.
 */
const NETWORK_DEMO = "https://pqc-qkd-hybrid.daemons.jp/";

export default function NetworkPositioning() {
  return (
    <figure className="chart-figure" style={{ margin: 0 }}>
      <svg viewBox="0 0 960 250" role="img" aria-labelledby="netpos-title netpos-desc"
        style={{ width: "100%", minWidth: 640, height: "auto", display: "block" }}>
        <title id="netpos-title">Application layer (this demo) against network layer (the companion network demo)</title>
        <desc id="netpos-desc">
          Top: this demo, in one browser tab: a PQC hybrid session key protects an MCP agent channel and
          the private computation. Bottom: the companion network demo, between containers: WireGuard tunnels
          keyed from QKD plus a PQC agreement, and an IKEv2 lane with ML-KEM and a post-quantum preshared
          key. Both simulate QKD. This page computes the bottom layer's key schedules in the browser.
        </desc>
        {/* application layer */}
        <rect x={10} y={14} width={940} height={96} rx={10} fill="var(--bg-2)" stroke="var(--accent-2)" />
        <text x={24} y={36} fontSize={13} fontWeight={700} fill="var(--text)">Application layer - this demo, in your browser tab</text>
        {[["PQC hybrid session key", "ML-KEM-768 + X25519, HKDF"], ["MCP agent channel", "ML-DSA-65 + AES-256-GCM"],
          ["private computation", "CKKS + garbled-circuit 2PC"], ["QKD", "simulated, key length only"]].map(([a, b], i) => (
          <g key={a}>
            <rect x={24 + i * 230} y={48} width={214} height={50} rx={8} fill="var(--panel)"
              stroke={i === 3 ? "var(--quantum)" : "var(--border)"} strokeDasharray={i === 3 ? "5 4" : undefined} />
            <text x={34 + i * 230} y={68} fontSize={12} fontWeight={600} fill="var(--text)">{a}</text>
            <text x={34 + i * 230} y={86} fontSize={11} fill="var(--muted)">{b}</text>
          </g>
        ))}
        {/* network layer */}
        <rect x={10} y={140} width={940} height={96} rx={10} fill="var(--bg-2)" stroke="var(--accent)" />
        <text x={24} y={162} fontSize={13} fontWeight={700} fill="var(--text)">Network layer - the companion network demo, between containers</text>
        {[["WireGuard hop tunnel", "PSK = HKDF(QKD || PQC-HPKE)"], ["WireGuard data tunnel", "PSK from Rosenpass"],
          ["IPsec / IKEv2 lane", "ML-KEM-768 + PPK (RFC 9370, 8784)"], ["QKD key servers", "simulated BB84, ETSI GS QKD 014"]].map(([a, b], i) => (
          <g key={a}>
            <rect x={24 + i * 230} y={174} width={214} height={50} rx={8} fill="var(--panel)"
              stroke={i === 3 ? "var(--quantum)" : "var(--border)"} strokeDasharray={i === 3 ? "5 4" : undefined} />
            <text x={34 + i * 230} y={194} fontSize={12} fontWeight={600} fill="var(--text)">{a}</text>
            {/* 10 units (Round 80): at 11 the IPsec line, then "ML-KEM-768 (RFC 9370) + PPK (RFC
                8784)", ran out of its 214-unit box. */}
            <text x={34 + i * 230} y={212} fontSize={10} fill="var(--muted)">{b}</text>
          </g>
        ))}
        <text x={480} y={128} fontSize={11} textAnchor="middle" fill="var(--muted)">
          same hybrid idea, two layers - this page computes the lower layer&apos;s key schedules in the browser; no packet leaves the tab
        </text>
      </svg>
      <figcaption className="figure-note">
        The network layer runs for real in the companion demo at{" "}
        <a href={NETWORK_DEMO} target="_blank" rel="noopener noreferrer">pqc-qkd-hybrid.daemons.jp</a>{" "}
        (WireGuard and strongSwan between containers; its QKD key servers are simulated too). Nothing on this
        page connects to it, and no figure from it is quoted here.
      </figcaption>
    </figure>
  );
}
