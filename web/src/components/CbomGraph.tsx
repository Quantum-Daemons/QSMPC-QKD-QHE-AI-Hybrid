// The CBOM's dependency graph, drawn from the document's own `dependencies` (Round 86): the application uses every
// asset, and an asset that combines others (the hybrid combiner) depends on them. Each dot is coloured by NIST IR
// 8547's class of the asset, which the table above states in words. The labels are measured once laid out, so the
// edges start after the longest label in whichever font the browser picked.
import { useLayoutEffect, useRef, useState } from "react";
import type { CbomDoc } from "../lib/cbomReport";
import { ruleFor, type Ir8547Class } from "../lib/ir8547";

const ROW = 21;
const TOP = 38;
const TRUNK_X = 12;
const DOT_X = 30;
const LABEL_X = 40;
const LANE_GAP = 9;

export const CLASS_COLOUR: Record<Ir8547Class | "none", string> = {
  "post-quantum": "var(--accent-2)", symmetric: "var(--accent)", "quantum-vulnerable": "var(--warn)", hybrid: "var(--quantum)",
  outside: "var(--muted)", none: "var(--muted)",
};

export const CLASS_NAME: Record<Ir8547Class, string> = {
  "post-quantum": "post-quantum", symmetric: "symmetric", "quantum-vulnerable": "quantum-vulnerable", hybrid: "hybrid",
  outside: "outside IR 8547's tables",
};

export default function CbomGraph({ doc }: { doc: CbomDoc }) {
  const assets = doc.components;
  const app = doc.metadata?.component;
  // The application's own entry says which assets it uses; one it does not list gets no branch.
  const used = new Set(doc.dependencies?.find((d) => d.ref === app?.["bom-ref"])?.dependsOn ?? []);
  const index = new Map(assets.map((c, i) => [c["bom-ref"], i]));
  // Every edge between two assets: the root's edges are the trunk and its branches.
  const edges: { from: number; to: number }[] = [];
  for (const d of doc.dependencies ?? []) {
    const from = index.get(d.ref);
    if (from === undefined) continue;
    for (const ref of d.dependsOn ?? []) {
      const to = index.get(ref);
      if (to !== undefined) edges.push({ from, to });
    }
  }
  // The nearest dependency gets the innermost lane, so no edge's end crosses another's lane.
  edges.sort((a, b) => Math.abs(a.from - a.to) - Math.abs(b.from - b.to));
  const labels = useRef<(SVGTextElement | null)[]>([]);
  const [widths, setWidths] = useState<number[]>(() => assets.map((c) => c.name.length * 7));
  useLayoutEffect(() => {
    const w = labels.current.map((t, i) => (t ? t.getComputedTextLength() : assets[i].name.length * 7));
    if (w.some((x, i) => Math.abs(x - widths[i]) > 0.5)) setWidths(w);
  }, [assets, widths]);
  const ends = widths.map((w) => LABEL_X + w + 5);
  const laneStart = Math.max(...ends) + 10;
  const width = Math.ceil(laneStart + Math.max(edges.length - 1, 0) * LANE_GAP + 12);
  const height = TOP + (assets.length - 1) * ROW + 14;
  const y = (i: number) => TOP + i * ROW;
  const summary = `${app?.name ?? "The application"} uses ${assets.length} cryptographic assets`
    + (edges.length ? `; ${[...new Set(edges.map((e) => assets[e.from].name))].join(", ")} depends on ${edges.map((e) => assets[e.to].name).join(", ")}` : "");
  return (
    <svg role="img" aria-label={summary} viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width, display: "block" }}>
      <title>{summary}</title>
      <defs>
        <marker id="cbom-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" style={{ fill: "var(--quantum)" }} />
        </marker>
      </defs>
      <text x={2} y={16} style={{ fill: "var(--text)", fontSize: 12.5, fontWeight: 600 }}>
        {app ? `${app.name} ${app.version ?? ""}`.trim() : "the application"}{used.size ? "" : " (no dependency entry)"}
      </text>
      <line x1={TRUNK_X} y1={22} x2={TRUNK_X} y2={y(assets.length - 1)} style={{ stroke: "var(--border)", strokeWidth: 1.5 }} />
      {assets.map((c, i) => {
        const cls = ruleFor(c["bom-ref"])?.cls ?? "none";
        return (
          <g key={c["bom-ref"]}>
            {used.has(c["bom-ref"]) && <line x1={TRUNK_X} y1={y(i)} x2={DOT_X - 5} y2={y(i)} style={{ stroke: "var(--border)", strokeWidth: 1.5 }} />}
            <circle cx={DOT_X} cy={y(i)} r={4.5} style={{ fill: CLASS_COLOUR[cls] }} />
            <text ref={(el) => { labels.current[i] = el; }} x={LABEL_X} y={y(i) + 4} style={{ fill: "var(--text)", fontSize: 12 }}>{c.name}</text>
          </g>
        );
      })}
      {edges.map((e, k) => {
        const lane = laneStart + k * LANE_GAP;
        return (
          <path key={`${e.from}-${e.to}`} d={`M${ends[e.from]},${y(e.from)} H${lane} V${y(e.to)} H${ends[e.to] + 1}`}
            style={{ fill: "none", stroke: "var(--quantum)", strokeWidth: 1.3 }} markerEnd="url(#cbom-arrow)" />
        );
      })}
    </svg>
  );
}
