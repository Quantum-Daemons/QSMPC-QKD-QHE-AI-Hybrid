// Figures a reviewer expects beside a model of each kind (Round 86, the owner's request): where a CNN looks, where a polygenic score's variants lie, and which classes an intrusion detector confuses.
// Each reads shipped data or the run's own result, in the clear, and says so.
import { useEffect, useMemo, useState } from "react";
import type { OcclusionMap } from "../lib/cnnSaliency";
import { confusion } from "../lib/studentConfusion";

const ink = "var(--text)", muted = "var(--muted)", edge = "var(--border)", accent = "var(--accent)", danger = "var(--danger)";

/** Occlusion sensitivity on the image the page has just run under MPC. */
export function CnnOcclusion({ e, className }: { e: OcclusionMap; className: string }) {
  const size = e.image.length, cell = 6, flat = e.map.flat();
  const low = Math.min(...e.image.flat()), high = Math.max(...e.image.flat());
  const scale = Math.max(...flat.map(Math.abs)) || 1;
  const grey = (v: number) => { const g = Math.round((255 * (v - low)) / (high - low || 1)); return `rgb(${g},${g},${g})`; };
  const heat = (v: number) => (v >= 0 ? `rgba(245,101,101,${(v / scale).toFixed(3)})` : `rgba(79,157,255,${(-v / scale).toFixed(3)})`);
  const grid = (fill: (y: number, x: number) => string, label: string) => (
    <svg role="img" aria-label={label} width={size * cell} height={size * cell} style={{ display: "block", background: "#000" }}>
      {Array.from({ length: size }, (_, y) => Array.from({ length: size }, (_, x) => <rect key={`${y}-${x}`} x={x * cell} y={y * cell} width={cell} height={cell} fill={fill(y, x)} />))}
    </svg>
  );
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
        <figure style={{ margin: 0 }}>{grid((y, x) => grey(e.image[y][x]), "The held-out image")}<figcaption className="small">The held-out image</figcaption></figure>
        <figure style={{ margin: 0 }}>{grid((y, x) => heat(e.map[y][x]), "Occlusion sensitivity")}<figcaption className="small">Occlusion sensitivity</figcaption></figure>
      </div>
      <p className="muted small" style={{ marginTop: 6 }}>
        Red marks pixels whose hiding, in {e.patch}x{e.patch} patches, lowers the student&apos;s margin for <b>{className}</b> the most; blue raises it. It is computed in the clear on the same image, so it describes the model and not the run under MPC. Occlusion is a coarse explanation (Zeiler and Fergus, ECCV 2014), and the image is not for clinical use.
      </p>
    </div>
  );
}

interface Locus { rsid: string; chr: string; pos: number; effect_allele: string; beta: number }

/** A locus plot of a polygenic score: each variant at its place on the genome, at its effect size. */
export function PrsLocusCard() {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<{ name: string; build: string; licence: string; licence_url: string; loci: Locus[] } | null>(null);
  useEffect(() => {
    if (!open || data) return;
    fetch(`${import.meta.env.BASE_URL}data/genomics/pgs000072_loci.json`).then((r) => r.json()).then(setData).catch(() => undefined);
  }, [open, data]);
  const plot = useMemo(() => {
    if (!data) return null;
    const chroms = [...new Set(data.loci.map((l) => l.chr))].sort((a, b) => Number(a) - Number(b));
    const extent = Object.fromEntries(chroms.map((c) => [c, Math.max(...data.loci.filter((l) => l.chr === c).map((l) => l.pos))]));
    let cumulative = 0;
    const start: Record<string, number> = {};
    for (const c of chroms) { start[c] = cumulative; cumulative += extent[c] * 1.03; }
    const W = 760, H = 240, left = 50, right = 12, top = 14, bottom = 34;
    const max = Math.max(...data.loci.map((l) => Math.abs(l.beta))) * 1.1;
    const X = (l: Locus) => left + ((start[l.chr] + l.pos) / cumulative) * (W - left - right);
    const Y = (b: number) => top + ((max - b) / (2 * max)) * (H - top - bottom);
    // Stagger the five labels: two strong variants can sit close together on the genome, and their names must not overprint.
    const top5 = [...data.loci].sort((a, b) => Math.abs(b.beta) - Math.abs(a.beta)).slice(0, 5).sort((a, b) => X(a) - X(b))
      .reduce<{ l: Locus; dy: number }[]>((acc, l) => {
        const clash = acc.filter((p) => Math.abs(X(p.l) - X(l)) < 90 && Math.abs(Y(p.l.beta) - Y(l.beta)) < 14 + Math.abs(p.dy));
        return [...acc, { l, dy: clash.length ? (clash[clash.length - 1].dy <= 0 ? 14 : -14) * (clash.length) : 0 }];
      }, []);
    return { chroms, start, cumulative, W, H, left, right, top, bottom, max, X, Y, top5 };
  }, [data]);
  return (
    <details className="card details" style={{ marginTop: 14 }} id="prs-locus" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Where the score&apos;s variants lie on the genome</summary>
      <div className="details-body">
        {data && plot && (
          <>
            <p>Each point is one of the {data.loci.length} variants {data.name} scores, at its GRCh37 position, with its published effect size: positive raises the score, negative lowers it. The five largest by magnitude are named.</p>
            <div style={{ overflowX: "auto" }}>
              <svg role="img" aria-label="Effect size of each scored variant by genomic position" viewBox={`0 0 ${plot.W} ${plot.H}`} style={{ width: "100%", minWidth: 560, height: "auto", display: "block" }}>
                {plot.chroms.map((c, i) => (
                  <g key={c}>
                    {i % 2 === 0 && <rect x={plot.left + (plot.start[c] / plot.cumulative) * (plot.W - plot.left - plot.right)} y={plot.top} width={(((plot.start[plot.chroms[i + 1] ?? ""] ?? plot.cumulative) - plot.start[c]) / plot.cumulative) * (plot.W - plot.left - plot.right)} height={plot.H - plot.top - plot.bottom} fill={edge} opacity={0.25} />}
                    <text x={plot.left + ((plot.start[c] + ((plot.start[plot.chroms[i + 1] ?? ""] ?? plot.cumulative) - plot.start[c]) / 2) / plot.cumulative) * (plot.W - plot.left - plot.right)} y={plot.H - plot.bottom + 14} textAnchor="middle" fontSize={10} fill={muted}>{c}</text>
                  </g>
                ))}
                <line x1={plot.left} x2={plot.W - plot.right} y1={plot.Y(0)} y2={plot.Y(0)} stroke={muted} strokeDasharray="3 3" />
                {[plot.max, -plot.max].map((b) => <text key={b} x={plot.left - 6} y={plot.Y(b) + 8} textAnchor="end" fontSize={10} fill={muted}>{b.toFixed(2)}</text>)}
                <text x={12} y={plot.Y(0) - 6} fontSize={10} fill={muted}>effect</text>
                <text x={12} y={plot.Y(0) + 8} fontSize={10} fill={muted}>ln(OR)</text>
                {data.loci.map((l) => <circle key={l.rsid} cx={plot.X(l)} cy={plot.Y(l.beta)} r={3} fill={l.beta >= 0 ? danger : accent} opacity={0.85}><title>{`${l.rsid}, chromosome ${l.chr}, position ${l.pos.toLocaleString()}, effect allele ${l.effect_allele}, ln(OR) ${l.beta.toFixed(4)}`}</title></circle>)}
                {plot.top5.map(({ l, dy }) => <text key={l.rsid} x={plot.X(l) + 5} y={plot.Y(l.beta) - 4 + dy} fontSize={10} fill={ink} style={{ paintOrder: "stroke", stroke: "var(--panel)", strokeWidth: 6 }}>{l.rsid}</text>)}
              </svg>
            </div>
            <p className="small muted">Weights: PGS Catalog score PGS000072, Graff et al., Nat Commun 12:970 (2021), <a href={data.licence_url}>{data.licence}</a>. Eight of the positions are the UK Biobank coordinates the scoring file gives in its variant description.</p>
          </>
        )}
      </div>
    </details>
  );
}

/** Which attack classes the published intrusion-detection student confuses, on its 600 held-out rows. */
export function ConfusionCard() {
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState<{ names: string[]; matrix: number[][]; total: number; correct: number } | null>(null);
  useEffect(() => {
    if (!open || model) return;
    Promise.all([fetch(`${import.meta.env.BASE_URL}data/telecom/student_heldout.json`).then((r) => r.json()), fetch(`${import.meta.env.BASE_URL}data/telecom/student_weights.json`).then((r) => r.json())]).then(([held, w]) => {
      setModel(confusion(held, w));
    }).catch(() => undefined);
  }, [open, model]);
  const short = (s: string) => s.replace(/_/g, " ");
  return (
    <details className="card details" style={{ marginTop: 14 }} id="confusion" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Which attacks the detector confuses</summary>
      <div className="details-body">
        {model && (
          <>
            <p>The published 64-parameter student on the {model.total} held-out rows of synthetic telemetry, computed in the clear in this tab: {model.correct} of {model.total} correct. Rows are the true class and columns the predicted one.</p>
            <div style={{ overflowX: "auto" }}>
              <table className="table small" aria-label="Confusion matrix">
                <thead><tr><th scope="col" /><th scope="col" colSpan={model.names.length} style={{ textAlign: "center" }}>predicted</th></tr><tr><th scope="col">true</th>{model.names.map((n) => <th scope="col" key={n} style={{ fontSize: 11 }}>{short(n)}</th>)}</tr></thead>
                <tbody>
                  {model.matrix.map((row, i) => {
                    const sum = row.reduce((a, b) => a + b, 0) || 1;
                    return <tr key={model.names[i]}><th scope="row" style={{ fontSize: 11, textAlign: "left" }}>{short(model.names[i])}</th>{row.map((v, j) => <td key={j} style={{ textAlign: "center", background: v === 0 ? undefined : i === j ? `rgba(79,157,255,${(v / sum).toFixed(2)})` : `rgba(245,101,101,${Math.min(1, 4 * v / sum).toFixed(2)})` }}>{v || ""}</td>)}</tr>;
                  })}
                </tbody>
              </table>
            </div>
            <p className="small muted">The telemetry is synthetic, generated by this project with the attack classes of Al-Kuwari et al. (arXiv:2509.14282), so this describes the model on its own corpus and is not a claim about real links.</p>
          </>
        )}
      </div>
    </details>
  );
}
