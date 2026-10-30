import StageBadge, { type StageKind } from "./StageBadge";

/**
 * The end-to-end chain as it actually runs in the public demo, one box per stage, each badged with
 * what executed. It replaces a row of text pills that drew "QRNG entropy" as its own stage (it runs
 * inside the PQC handshake), coloured the CKKS stage with the warning style used for scaffolds (CKKS
 * is real), and drew the QKD step on the key path - where nothing it produces ever goes.
 *
 * THE QKD BOX IS DRAWN OFF THE KEY PATH ON PURPOSE. The browser's `bb84()` / `e91()` return a key
 * LENGTH and no key bytes (checklist SC2), and the only write to the session key is the PQC hybrid's
 * (`clientCompute.ts`, `pqc_handshake`). A diagram that routed the session key through QKD would draw
 * a mixing step the code does not perform.
 *
 * Badges follow the truth table checked against the code in the Round 58 survey: QRNG is a simulated
 * state-vector source whose SP 800-90B health tests are really computed; PSI, secure sum and the
 * federated masking are classical, real protocols whose parties all run in this one tab, so they are
 * badged "notional" like the garbled-circuit 2PC; the quantum one-time pad uses a seeded mask. Every
 * federated mask is derived from the one PQC session key, which the aggregator also holds, so the
 * masks cancel exactly and hide nothing from it; the PSI's HMAC key is held by both banks.
 */
type Box = { x: number; y: number; w: number; title: string; lines: string[]; kind: StageKind; id: string };

const H = 92;
const ROW1 = 34;
const ROW2 = 212;

// Widths with room for the widest common sans-serif (Round 80). The page font is the visitor's
// `system-ui`; in DejaVu Sans, the Linux default, "bundled sample rows" ran out of a 130-unit box and
// two titles ran into their badge dots, with every box fitting in the narrower font this was drawn in.
// The QKD box sits at 670 so "count only - not mixed in" clears its border; at 640 it touched it.

const BOXES: Box[] = [
  { id: "data", x: 10, y: ROW1, w: 150, title: "Your data", lines: ["bundled sample rows", "never leave the tab"], kind: "real" },
  { id: "pqc", x: 180, y: ROW1, w: 220, title: "PQC handshake",
    lines: ["ML-KEM-768 + X25519", "HKDF combiner (SP 800-227)", "-> 256-bit session key"], kind: "real" },
  { id: "qkd", x: 670, y: ROW1, w: 250, title: "QKD link (BB84 decoy / E91)",
    lines: ["analytic channel model", "yields a key LENGTH,", "not key bytes"], kind: "simulated" },
  { id: "qsmpc", x: 10, y: ROW2, w: 170, title: "QSMPC",
    lines: ["PSI: HMAC under a key", "both sides hold; secure", "sum: additive shares"], kind: "notional" },
  { id: "fl", x: 200, y: ROW2, w: 176, title: "Federated training",
    lines: ["finance, medical:", "FedAvg; every mask from", "the one session key"], kind: "notional" },
  { id: "pi", x: 396, y: ROW2, w: 180, title: "Private inference",
    lines: ["CKKS linear (node-seal)", "+ garbled-circuit 2PC", "argmax (Polytune)"], kind: "notional" },
  { id: "qhe", x: 596, y: ROW2, w: 180, title: "QHE comparison",
    lines: ["QOTP + Clifford+T", "circuit, untrained,", "state-vector simulation"], kind: "seeded" },
  { id: "sig", x: 796, y: ROW2, w: 170, title: "Signed result",
    lines: ["ML-DSA-65 signature", "over the result"], kind: "real" },
];

const COLOR: Record<StageKind, string> = {
  real: "var(--accent-2)", notional: "var(--accent)", simulated: "var(--quantum)", seeded: "var(--warn)",
};

function SvgBox({ b }: { b: Box }) {
  return (
    <g>
      <rect x={b.x} y={b.y} width={b.w} height={H} rx={10} fill="var(--bg-2)" stroke={COLOR[b.kind]}
        strokeWidth={1.6} strokeDasharray={b.kind === "simulated" ? "6 4" : undefined} />
      <text x={b.x + 10} y={b.y + 20} fontSize={13} fontWeight={700} fill="var(--text)">{b.title}</text>
      {b.lines.map((l, i) => (
        <text key={i} x={b.x + 10} y={b.y + 39 + i * 15} fontSize={11.5} fill="var(--muted)">{l}</text>
      ))}
      <circle cx={b.x + b.w - 12} cy={b.y + 15} r={5} fill={COLOR[b.kind]} />
    </g>
  );
}

export default function ArchitectureDiagram() {
  const arrow = "url(#arch-arrow)";
  return (
    <figure className="chart-figure" style={{ margin: 0 }}>
      <svg viewBox="0 0 980 330" role="img" aria-labelledby="arch-title arch-desc"
        style={{ width: "100%", minWidth: 640, height: "auto", display: "block" }}>
        <title id="arch-title">End-to-end chain as it runs in this demo</title>
        <desc id="arch-desc">
          Your data feeds a real PQC handshake (ML-KEM-768 plus X25519) that produces the session key.
          A simulated QKD link sits off the key path: it yields a key length and no key bytes, and
          nothing from it is mixed into the session key. The session key feeds QSMPC and federated
          training on the finance and medical pages, whose parties all run in this tab; private
          inference is real CKKS plus a garbled-circuit argmax between notional parties; an untrained
          QOTP plus Clifford+T comparison circuit with a seeded pad, simulated, runs beside it; the
          result is signed with ML-DSA-65.
        </desc>
        <defs>
          <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
          </marker>
        </defs>
        {/* main key path, row 1 */}
        <line x1={160} y1={ROW1 + H / 2} x2={178} y2={ROW1 + H / 2} stroke="var(--muted)" strokeWidth={1.4} markerEnd={arrow} />
        {/* session key down to row 2 */}
        <path d={`M400,${ROW1 + H / 2} L470,${ROW1 + H / 2} L470,${ROW2 - 40} L95,${ROW2 - 40} L95,${ROW2 - 2}`}
          fill="none" stroke="var(--accent-2)" strokeWidth={1.6} markerEnd={arrow} />
        <text x={478} y={ROW1 + H / 2 + 50} fontSize={11} fill="var(--accent-2)">session key</text>
        {/* QKD: dashed, ends in a cross before the key path */}
        <line x1={668} y1={ROW1 + H / 2} x2={520} y2={ROW1 + H / 2} stroke="var(--quantum)" strokeWidth={1.4} strokeDasharray="5 4" />
        <text x={500} y={ROW1 + H / 2 + 5} fontSize={16} fontWeight={700} fill="var(--quantum)">x</text>
        <text x={520} y={ROW1 + H / 2 - 10} fontSize={10.5} fill="var(--quantum)">count only - not mixed in</text>
        {/* row 2 chain */}
        {[[180, 198], [376, 394], [576, 594], [776, 794]].map(([a, b], i) => (
          <line key={i} x1={a} y1={ROW2 + H / 2} x2={b} y2={ROW2 + H / 2} stroke="var(--muted)" strokeWidth={1.4} markerEnd={arrow} />
        ))}
        {BOXES.map((b) => <SvgBox key={b.id} b={b} />)}
      </svg>
      <figcaption className="figure-note">
        Dashed outline = simulated. The QKD link is drawn off the key path because nothing it produces
        reaches the session key in the browser; in the full local deployment a run that clears 256
        secure bits folds simulated-QKD key material in.
      </figcaption>
      <div className="pill-row" aria-label="Legend">
        <StageBadge kind="real" /> <StageBadge kind="notional" /> <StageBadge kind="simulated" /> <StageBadge kind="seeded" />
      </div>
      <ol className="sr-only">
        {BOXES.map((b) => <li key={b.id}>{b.title}: {b.lines.join(" ")} ({b.kind})</li>)}
      </ol>
    </figure>
  );
}
