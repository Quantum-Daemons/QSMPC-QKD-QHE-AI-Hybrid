// A run's messages as a sequence diagram (Round 86, the owner's fourth choice of 2026-11-14), up to the timeline's
// position, drawn here as SVG from the run's own `message` and `holds` events. The same diagram as Mermaid text
// (`toMermaidSequence`) can be copied or downloaded and drawn by Mermaid anywhere. Mermaid itself is not loaded: a lazy
// import of Mermaid 12.2.0 added 57 chunk files, 5.1 MB, to the deployed site, among them every diagram type this page
// never draws and the ELK layout (elkjs, EPL-2.0 or GPL-3.0), measured on a client-only build on 2026-11-14.
import { useMemo, useRef } from "react";
import type { PipelineEvent } from "../lib/api";
import { downloadText } from "../lib/export";
import { sideName, type HoldsData, type MessageData } from "../lib/runTimeline";
import { toMermaidSequence } from "../lib/sequenceDiagram";
import type { DiagramConfig } from "../lib/useCaseFlows";

const COL = 260, LEFT = 14, HEAD = 46, MSG = 54, NOTE = 26, STAGE = 24, CH = 6.4;
const SIMULATED = new Set(["photons (simulated)", "classical (simulated)"]);
const STROKE: Record<string, string> = {
  public: "var(--muted)", "session key": "var(--accent)", CKKS: "var(--accent-2)", BGV: "var(--accent-2)",
  "garbled circuit": "var(--warn)", "additive shares": "var(--warn)", "pairwise masks": "var(--warn)",
};
const cut = (s: string, n: number) => (s.length > n ? `${s.slice(0, Math.max(1, n - 3))}...` : s);
/** Words into at most `lines` lines of `n` characters, the last cut with "..." when the text runs on. */
function wrap(s: string, n: number, lines = 2): string[] {
  const out: string[] = [];
  let cur = "";
  for (const w of s.split(" ")) {
    if (!cur) cur = w;
    else if (cur.length + 1 + w.length <= n) cur += ` ${w}`;
    else { out.push(cur); cur = w; }
  }
  if (cur) out.push(cur);
  if (out.length <= lines) return out.map((l) => cut(l, n));
  return [...out.slice(0, lines - 1).map((l) => cut(l, n)), cut(out.slice(lines - 1).join(" "), n)];
}
const bytes = (b: number | null, n?: number) =>
  b == null ? "size not measured" : `${b >= 1e6 ? `${(b / 1e6).toFixed(2)} MB` : `${b.toLocaleString("en-US")} B`}${n && n > 1 ? ` over ${n.toLocaleString("en-US")}` : ""}`;

type Row =
  | { kind: "stage"; stage: string; y: number }
  | { kind: "msg"; n: number; d: MessageData; y: number }
  | { kind: "note"; d: HoldsData; y: number };

export default function SequenceView({ usecase, events, config }: {
  usecase: string; events: PipelineEvent[]; config: DiagramConfig;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const { sides, rows, height } = useMemo(() => {
    const shown = events.filter((e) => e.type === "message" || (e.type === "holds" && (e.data as unknown as HoldsData).kept));
    const found: string[] = [];
    for (const e of shown) {
      const d = e.data as unknown as MessageData & HoldsData;
      for (const s of e.type === "message" ? [d.from, d.to] : [d.party]) if (!found.includes(s)) found.push(s);
    }
    const rank = (s: string) => (s === "a" ? 0 : s === "b" ? 1 : s === "many" ? 2 : 3);
    found.sort((x, y) => rank(x) - rank(y));
    const out: Row[] = [];
    let y = HEAD + 10, stage: string | null = null, n = 0;
    for (const e of shown) {
      if (e.stage !== stage) { stage = e.stage; out.push({ kind: "stage", stage, y }); y += STAGE; }
      if (e.type === "message") { n += 1; out.push({ kind: "msg", n, d: e.data as unknown as MessageData, y }); y += MSG; }
      else { out.push({ kind: "note", d: e.data as unknown as HoldsData, y }); y += NOTE; }
    }
    return { sides: found, rows: out, height: y + 10 };
  }, [events]);
  const mermaid = useMemo(() => toMermaidSequence(usecase, events, config), [usecase, events, config]);
  if (sides.length === 0) return null;
  const width = LEFT * 2 + COL * sides.length;
  const x = (s: string) => LEFT + COL * sides.indexOf(s) + COL / 2;
  const name = (s: string) => sideName(usecase, s, config);
  // Where a lifeline must not run: through a note on it, or behind the labels of a message that skips its column, whose
  // labels are centred over it (the page's text-layout gate measures a line through text, whatever is painted over it).
  const gaps = new Map<string, [number, number][]>(sides.map((s) => [s, []]));
  for (const r of rows) {
    if (r.kind === "note") gaps.get(r.d.party)!.push([r.y, r.y + NOTE]);
    if (r.kind === "msg") {
      const [i, j] = [sides.indexOf(r.d.from), sides.indexOf(r.d.to)].sort((a, b) => a - b);
      for (let k = i + 1; k < j; k++) gaps.get(sides[k])!.push([r.y, r.y + MSG]);
    }
  }
  /** A lifeline from the heads to the bottom, in segments around its gaps. */
  const segments = (s: string): [number, number][] => {
    const out: [number, number][] = [];
    let y = HEAD;
    for (const [a, b] of [...(gaps.get(s) ?? [])].sort((p, q) => p[0] - q[0])) {
      if (a > y) out.push([y, a]);
      y = Math.max(y, b);
    }
    if (y < height - 6) out.push([y, height - 6]);
    return out;
  };
  // Stage blocks: from each stage's heading to the next one's.
  const stageRows = rows.filter((r): r is Extract<Row, { kind: "stage" }> => r.kind === "stage");
  // The page's colours are CSS variables, which a file opened on its own cannot resolve, so they are written in.
  const downloadSvg = () => {
    const node = svgRef.current;
    if (!node) return;
    const css = getComputedStyle(document.documentElement);
    const text = new XMLSerializer().serializeToString(node)
      .replace(/var\((--[\w-]+)\)/g, (_m, v: string) => css.getPropertyValue(v).trim() || "#888");
    downloadText(text, `${usecase}-sequence.svg`, "image/svg+xml");
  };
  return (
    <details className="sequence-view">
      <summary>The same messages as a sequence diagram</summary>
      <p className="muted small" style={{ margin: "6px 0" }}>
        Numbered in the order the run sent them, each stage a shaded block. A dotted arrow is a simulated channel, a solid one
        carried the bytes its label gives; a note is something a party keeps. It follows the timeline above; hover a label for
        its whole text.
      </p>
      <div className="sequence-svg">
        <svg ref={svgRef} xmlns="http://www.w3.org/2000/svg" viewBox={`0 0 ${width} ${height}`} width={width} height={height}
          role="img" aria-label={`Sequence of ${rows.filter((r) => r.kind === "msg").length} messages between ${sides.map(name).join(", ")}`}
          style={{ fontFamily: "system-ui, sans-serif" }}>
          <defs>
            {Object.entries({ plain: "var(--muted)", ...STROKE }).map(([k, c]) => (
              <marker key={k} id={`seq-head-${k.replace(/\W+/g, "-")}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={c} />
              </marker>
            ))}
          </defs>
          <rect x={0} y={0} width={width} height={height} fill="var(--panel)" />
          {stageRows.map((r, i) => {
            const end = stageRows[i + 1]?.y ?? height - 6;
            return (
              <g key={`s${i}`}>
                <rect x={4} y={r.y} width={width - 8} height={end - r.y - 2} rx={6} fill="var(--accent)" fillOpacity={i % 2 ? 0.05 : 0.09} />
                <text x={10} y={r.y + 15} fontSize={11} fill="var(--muted)">{r.stage}</text>
              </g>
            );
          })}
          {sides.map((s) => (
            <g key={s}>
              {segments(s).map(([y1, y2], k) => (
                <line key={k} x1={x(s)} y1={y1} x2={x(s)} y2={y2} stroke="var(--border)" strokeDasharray="3 4" />
              ))}
              <rect x={x(s) - COL / 2 + 8} y={6} width={COL - 16} height={HEAD - 14} rx={6} fill="var(--bg-2)" stroke="var(--border)" />
              <text x={x(s)} y={6 + (HEAD - 14) / 2 + 4} fontSize={12} textAnchor="middle" fill="var(--text)">
                <title>{name(s)}</title>{cut(name(s), 30)}
              </text>
            </g>
          ))}
          {rows.map((r, i) => {
            if (r.kind === "stage") return null;
            if (r.kind === "note") {
              const label = `keeps ${r.d.what}`;
              return (
                <g key={i}>
                  <rect x={x(r.d.party) - COL / 2 + 12} y={r.y + 2} width={COL - 24} height={NOTE - 6} rx={4} fill="var(--bg-2)" stroke="var(--border)" />
                  <text x={x(r.d.party)} y={r.y + NOTE / 2 + 3} fontSize={10} textAnchor="middle" fill="var(--muted)">
                    <title>{label}</title>{cut(label, Math.floor((COL - 30) / 5.6))}
                  </text>
                </g>
              );
            }
            const { d } = r;
            const [x1, x2] = [x(d.from), x(d.to)];
            const color = STROKE[d.protection] ?? "var(--muted)";
            const head = `url(#seq-head-${(STROKE[d.protection] ? d.protection : "plain").replace(/\W+/g, "-")})`;
            const span = Math.max(Math.abs(x2 - x1), COL) - 16;
            const mid = (x1 + x2) / 2, yl = r.y + 36;
            const top = `${r.n}. ${d.what}`, under = `${bytes(d.bytes, d.count)}, ${d.protection}`;
            const label = wrap(top, Math.floor(span / CH));
            return (
              <g key={i}>
                <line x1={x1} y1={yl} x2={x2 + (x2 > x1 ? -2 : 2)} y2={yl} stroke={color} strokeWidth={1.6}
                  strokeDasharray={SIMULATED.has(d.protection) ? "5 4" : undefined} markerEnd={head} />
                <text x={mid} y={yl - 6 - 13 * (label.length - 1)} fontSize={11} textAnchor="middle" fill="var(--text)">
                  <title>{`${top} (${under})`}</title>
                  {label.map((l, k) => <tspan key={k} x={mid} dy={k === 0 ? 0 : 13}>{l}</tspan>)}
                </text>
                <text x={mid} y={yl + 12} fontSize={10} textAnchor="middle" fill="var(--muted)">{cut(under, Math.floor(span / 5.6))}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="btn-row" style={{ marginTop: 6 }}>
        <button className="btn ghost" onClick={downloadSvg}>Download the diagram (SVG)</button>
        <button className="btn ghost" onClick={() => { void navigator.clipboard?.writeText(mermaid); }}
          title="The same diagram as Mermaid text, for mermaid.live or a Markdown file">Copy it as Mermaid text</button>
        <button className="btn ghost" onClick={() => downloadText(mermaid, `${usecase}-sequence.mmd`)}>Download the Mermaid text (.mmd)</button>
      </div>
    </details>
  );
}
