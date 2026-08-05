// Per-use-case TWO-PARTY (Alice↔Bob) MPC-FHE secret-sharing sequence diagram — driven LIVE by the
// running pipeline. Each party sits on its own LAN; they collaborate over a WAN without a trusted
// third party and without sharing raw data. A step lights up when its stageId is the pipeline's
// active stage and is marked done once the stage completes — so Start/Stop/Restart/Pause/Step on the
// page drive this diagram as a simulator. The MPC-FHE private inference is shown as its two halves:
// FHE-linear (a CKKS ciphertext crosses the WAN) + MPC-non-linear (secret-shares exchanged for
// ReLU/argmax). A final full-width "QSMPC-FHE comparison" bar shows the quantum-secure path that the
// benchmark evaluates against (the `quantum_comparison` stage), so the advantage/security numbers are
// clearly produced by an actual run. Grounded in references/ (Zhang QOT npj 2026, Li&Deng QHE-FL,
// Sutradhar QSMS, EncFormer linear-FHE/non-linear-MPC split). PoC runs LAN/local.

type Kind = "local-a" | "local-b" | "q-arrow" | "c-arrow" | "c-bidir" | "fhe-arrow" | "mpc-bidir" | "compare";
type M = Record<string, any>;
/** The crypto-agility selections for the run being drawn. */
export interface DiagramConfig { kem?: string; qkd?: string; sig?: string }
interface Step {
  stageId: string; kind: Kind;
  label: (m: M, c: DiagramConfig) => string;
  /** Drawn only when this returns false. Used to drop the quantum arrow under a PQC-only profile. */
  hidden?: (c: DiagramConfig) => boolean;
}
interface Spec { a: { name: string; sub: string }; b: { name: string; sub: string }; steps: Step[] }

// The protocol names used to be hardcoded here, so selecting ml-kem-1024 still drew "ML-KEM-768",
// selecting e91 still drew "BB84", and selecting pqc-only still drew a quantum arrow with an
// em-dash QBER. The diagram is the most-looked-at part of the page; it must show what actually ran.
const KEM_LABEL: Record<string, string> = {
  "ml-kem-768+x25519": "ML-KEM-768 + X25519",
  "ml-kem-1024+x25519": "ML-KEM-1024 + X25519",
};
const kemName = (c: DiagramConfig) =>
  KEM_LABEL[String(c.kem ?? "")] ?? (c.kem ? String(c.kem).toUpperCase() : "hybrid KEM");
const sigName = (c: DiagramConfig) => (c.sig ? String(c.sig).toUpperCase() : "ML-DSA");
const qkdName = (c: DiagramConfig) => {
  const q = String(c.qkd ?? "");
  if (q.startsWith("e91")) return "E91 (entanglement)";
  if (q.startsWith("bb84")) return q.includes("decoy") ? "BB84 decoy-state" : "BB84";
  return q || "QKD";
};
const pqcOnly = (c: DiagramConfig) => String(c.qkd ?? "") === "pqc-only";

const f = (m: M, k: string, d = "—") => {
  const v = m[k];
  if (v == null) return d;
  if (typeof v === "number")
    return Math.abs(v) >= 1e-3 || v === 0 ? String(Number(v.toFixed(5))) : v.toExponential(2);
  return String(v);
};

// The MPC-FHE private inference, drawn as its two halves (the 2026 hybrid: linear→FHE, non-linear→MPC).
// Both map to the backend `private_inference` stage, so they light up together when it runs and the
// labels fill from the live `fhe_linear_ms` / `mpc_nonlinear_ms` / `mpc_comm_kb` metrics.
const HYBRID_INFERENCE: Step[] = [
  { stageId: "private_inference", kind: "fhe-arrow",
    label: (m) => `FHE-linear · CKKS ct · ${f(m, "fhe_linear_ms")} ms` },
  { stageId: "private_inference", kind: "mpc-bidir",
    label: (m) => `MPC-non-linear · shares · ${f(m, "mpc_nonlinear_ms")} ms · ${f(m, "mpc_comm_kb")} KB` },
];

// The quantum-secure ALTERNATIVE the benchmark scores against (the `quantum_comparison` stage runs a
// real QOTP+QCNN private inference). Rendered as a full-width bar so it's clear QSMPC-FHE is executed
// and the quantum-advantage / security-score are produced by an actual run, not asserted.
const QSMPC_COMPARE: Step = {
  stageId: "quantum_comparison", kind: "compare",
  label: (m) => `QSMPC-FHE comparison · QOTP + QCNN (T-gates) · advantage ${f(m, "quantum_advantage_score")}/10`,
};

export const USECASE_SPECS: Record<string, Spec> = {
  finance_aml: {
    a: { name: "Bank A", sub: "blacklist + fraud model · LAN" },
    b: { name: "Bank B", sub: "candidate accounts · LAN" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "qkd_key_agreement", kind: "q-arrow", label: (m, c) => `${qkdName(c)} · QBER ${f(m, "qber")}`, hidden: pqcOnly },
      { stageId: "qsmpc_compute", kind: "c-bidir", label: (m) => `keyed-hash PSI · shared flagged = ${f(m, "psi_intersection_size")}` },
      ...HYBRID_INFERENCE,
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record + explanation` },
    ],
  },
  medical_fl: {
    a: { name: "Hospital", sub: "patient data (never leaves) · LAN" },
    b: { name: "Aggregator", sub: "global model · LAN" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "qkd_key_agreement", kind: "q-arrow", label: (m, c) => `${qkdName(c)} · QBER ${f(m, "qber")}`, hidden: pqcOnly },
      { stageId: "federated_training", kind: "c-bidir", label: (m) => `federated train + pairwise masking (demo keys) · acc ${f(m, "federated_accuracy")}` },
      ...HYBRID_INFERENCE,
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record + explanation` },
    ],
  },
  genomics: {
    a: { name: "Cohort 1", sub: "allele counts · LAN" },
    b: { name: "Cohort 2", sub: "allele counts · LAN" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "qkd_key_agreement", kind: "q-arrow", label: (m, c) => `${qkdName(c)} · QBER ${f(m, "qber")}`, hidden: pqcOnly },
      { stageId: "qsmpc_compute", kind: "q-arrow", label: (m) => `QSMS secure sum (d-qudit GHZ) · pooled = ${f(m, "secure_sum")}` },
      ...HYBRID_INFERENCE,
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record` },
    ],
  },
  telecom_ids: {
    a: { name: "Carrier", sub: "QKD Tx + IDS plane · LAN" },
    b: { name: "Remote site / Eve", sub: "QKD Rx (eavesdropper)" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "qkd_key_agreement", kind: "q-arrow", label: (m, c) => `${qkdName(c)} (w/ eavesdropper) · QBER ${f(m, "qber")}`, hidden: pqcOnly },
      { stageId: "qml_ids", kind: "local-a", label: (m) => `QML-IDS attack detection · macro-F1 ${f(m, "ids_macro_f1")}` },
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record` },
    ],
  },
  satellite_comms: {
    a: { name: "Ground station", sub: "telemetry uplink · LAN" },
    b: { name: "LEO satellite", sub: "free-space downlink" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "satellite_qkd", kind: "q-arrow", label: (m, c) => `free-space ${qkdName(c)} (${f(m, "channel_loss_db")} dB) · QBER ${f(m, "qber")}`,
        hidden: pqcOnly },
      { stageId: "qsdc_telemetry", kind: "q-arrow", label: (m) => `QSDC direct telemetry · secrecy ${f(m, "qsdc_secrecy_capacity")} bit/carrier` },
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record` },
    ],
  },
};

export function hasTwoPartySpec(usecase: string): boolean {
  return usecase in USECASE_SPECS;
}

const ALICE_X = 172;
const BOB_X = 612;
const MID = (ALICE_X + BOB_X) / 2;
const ROW0 = 100;
const ROW_H = 58;

export default function TwoPartyDiagram({
  usecase, activeStage, doneStages, metrics, config = {},
}: {
  usecase: string;
  activeStage: string | null;
  doneStages: Set<string>;
  metrics: Record<string, any>;
  /** Crypto-agility selections, so the arrows name what actually ran. */
  config?: DiagramConfig;
}) {
  const spec = USECASE_SPECS[usecase];
  if (!spec) return null;
  // Drop steps the selected profile does not run, rather than drawing them with an em-dash value.
  const steps = spec.steps.filter((st) => !st.hidden?.(config));
  const idle = activeStage == null && doneStages.size === 0;
  const opacity = (s: Step) =>
    idle ? 1 : s.stageId === activeStage ? 1 : doneStages.has(s.stageId) ? 0.95 : 0.26;
  const isActive = (s: Step) => !idle && s.stageId === activeStage;
  const isDone = (s: Step) => doneStages.has(s.stageId);
  const bottomY = ROW0 + steps.length * ROW_H + 6;
  const rowY = (i: number) => ROW0 + i * ROW_H;

  return (
    <svg viewBox={`0 0 780 ${bottomY + 78}`} width="100%" xmlns="http://www.w3.org/2000/svg"
      style={{ fontFamily: "inherit" }} role="img" aria-label={`${usecase} two-party diagram`}>
      <defs>
        <marker id="tp-c" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
          <path d="M0,0 L8,3.2 L0,6.4 Z" fill="var(--accent)" /></marker>
        <marker id="tp-cl" markerWidth="9" markerHeight="9" refX="1" refY="3.2" orient="auto">
          <path d="M8,0 L0,3.2 L8,6.4 Z" fill="var(--accent)" /></marker>
        <marker id="tp-q" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
          <path d="M0,0 L8,3.2 L0,6.4 Z" fill="var(--quantum)" /></marker>
        <marker id="tp-m" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
          <path d="M0,0 L8,3.2 L0,6.4 Z" fill="#f5a623" /></marker>
        <marker id="tp-ml" markerWidth="9" markerHeight="9" refX="1" refY="3.2" orient="auto">
          <path d="M8,0 L0,3.2 L8,6.4 Z" fill="#f5a623" /></marker>
      </defs>

      {/* WAN band between the two LANs (kept above + clear of the party boxes) */}
      <text x={MID} y={18} textAnchor="middle" fontSize={10.5} fill="#6b7689">⟷ WAN ⟷</text>

      {/* lane headers */}
      {[[spec.a, ALICE_X, "var(--accent)"], [spec.b, BOB_X, "var(--quantum)"]].map(([p, x, col]: any) => (
        <g key={p.name}>
          <rect x={x - 96} y={26} width={192} height={44} rx={9} fill="var(--panel)" stroke={col} strokeWidth={1.5} />
          <text x={x} y={48} textAnchor="middle" fontSize={16} fontWeight={700} fill={col}>{p.name}</text>
          <text x={x} y={63} textAnchor="middle" fontSize={10.5} fill="#97a3b6">{p.sub}</text>
        </g>
      ))}

      {/* lifelines */}
      {[ALICE_X, BOB_X].map((x) => (
        <line key={x} x1={x} y1={72} x2={x} y2={bottomY} stroke="#2a3650" strokeWidth={1.5} strokeDasharray="3 4" />
      ))}

      {/* protocol steps */}
      {steps.map((s, i) => {
        const y = rowY(i);
        const op = opacity(s);
        const mark = isDone(s) ? " ✓" : isActive(s) ? " ▶" : "";
        const label = s.label(metrics, config) + mark;

        // full-width QSMPC-FHE comparison bar (the quantum-secure alternative the benchmark scores)
        if (s.kind === "compare") {
          return (
            <g key={i} opacity={op}>
              <rect x={64} y={y - 17} width={652} height={34} rx={8}
                fill={isActive(s) ? "#231a3a" : "var(--panel)"} stroke="var(--quantum)"
                strokeWidth={isActive(s) ? 2.4 : 1.4} strokeDasharray="6 3" />
              <text x={MID} y={y + 4} textAnchor="middle" fontSize={11} fill="var(--quantum)" fontWeight={600}>{label}</text>
            </g>
          );
        }
        // local computation box at one party
        if (s.kind === "local-a" || s.kind === "local-b") {
          const cx = s.kind === "local-a" ? ALICE_X : BOB_X;
          const col = s.kind === "local-a" ? "var(--accent)" : "var(--quantum)";
          return (
            <g key={i} opacity={op}>
              <rect x={cx - 150} y={y - 16} width={300} height={32} rx={7}
                fill={isActive(s) ? "#1c2740" : "var(--panel)"} stroke={col} strokeWidth={isActive(s) ? 2.4 : 1.3} />
              <text x={cx} y={y + 4} textAnchor="middle" fontSize={11} fill="#dfe7f3">{label}</text>
            </g>
          );
        }
        // message arrow between the two parties (classical / quantum / FHE ciphertext / MPC shares)
        const quantum = s.kind === "q-arrow";
        const mpc = s.kind === "mpc-bidir";
        const bidir = s.kind === "c-bidir" || s.kind === "mpc-bidir";
        const col = quantum ? "var(--quantum)" : mpc ? "#f5a623" : "var(--accent)";  // fhe-arrow = accent
        const dash = quantum ? "5 3" : mpc ? "2 2" : undefined;
        const mEnd = quantum ? "url(#tp-q)" : mpc ? "url(#tp-m)" : "url(#tp-c)";
        const mStart = bidir ? (mpc ? "url(#tp-ml)" : "url(#tp-cl)") : undefined;
        return (
          <g key={i} opacity={op}>
            <text x={MID} y={y - 9} textAnchor="middle" fontSize={11}
              fill={isActive(s) ? "#ffffff" : "#c4cee0"}>{label}</text>
            <line x1={ALICE_X + 6} y1={y} x2={BOB_X - 6} y2={y} stroke={col}
              strokeWidth={isActive(s) ? 2.6 : 1.8} strokeDasharray={dash}
              markerEnd={mEnd} markerStart={mStart} />
          </g>
        );
      })}

      {/* channel legend — moved to the BOTTOM so it never overlaps the party boxes */}
      <g transform={`translate(70, ${bottomY + 18})`} fontSize={10} fill="#97a3b6">
        <line x1={0} y1={-3} x2={22} y2={-3} stroke="var(--accent)" strokeWidth={2} markerEnd="url(#tp-c)" />
        <text x={27} y={1}>classical / FHE ct</text>
        {/* The legend has to agree with the arrows: under a PQC-only profile no quantum channel is
            drawn, so advertising one in the key was the same hardcoding one level down. */}
        {!pqcOnly(config) && (<>
          <line x1={200} y1={-3} x2={222} y2={-3} stroke="var(--quantum)" strokeWidth={2} strokeDasharray="4 3" markerEnd="url(#tp-q)" />
          <text x={227} y={1}>quantum ({qkdName(config)})</text>
        </>)}
        <line x1={430} y1={-3} x2={452} y2={-3} stroke="#f5a623" strokeWidth={2} strokeDasharray="2 2" markerEnd="url(#tp-m)" markerStart="url(#tp-ml)" />
        <text x={457} y={1}>MPC secret-shares</text>
      </g>

      <text x={MID} y={bottomY + 46} textAnchor="middle" fontSize={11} fill="#97a3b6">
        no trusted third party · no raw data shared · {spec.a.name} (LAN) ↔ WAN ↔ {spec.b.name}
      </text>
      <text x={MID} y={bottomY + 62} textAnchor="middle" fontSize={10} fill="#6b7689">
        MPC-FHE hybrid: linear → FHE (CKKS ciphertext) · non-linear → MPC (secret-shares) · vs QSMPC-FHE · PoC runs LAN/local
      </text>
    </svg>
  );
}
