import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { useReadoutLine } from "./ChartReadout";

/**
 * A scatter's points drawn on one canvas inside a recharts plot area (Round 85, by the user's decision).
 *
 * WHY. Finance's held-out split is 8,192 rows, and three charts drew every row as an SVG symbol: after
 * one run they were 49,300 of the page's 52,700 elements, and the tab's JS heap reached 0.4 to 1 GB.
 * Every point is still drawn, where recharts' own scales put it and in the shape recharts would have
 * drawn (d3-shape's symbols at recharts' default area); what changes is that they are one element. The
 * axes, grid, reference lines and legend stay recharts' SVG.
 *
 * HOW IT FITS. `<Customized component={<CanvasMarks ... />} />` inside the chart: recharts clones the
 * element with the chart's props and state, so the axis scales (`xAxisMap`, `yAxisMap`) and the plot
 * area (`offset`) arrive as props. Each series keeps its `<Scatter>` with no data, so the legend keys
 * it as before, and where an axis domain is automatic a two-point `<Scatter>` of the data's corners
 * (`corners`) gives recharts the domain all the points did: an automatic domain depends on the
 * extremes alone.
 *
 * READOUT. A canvas has no per-point hover, so the layer finds the point nearest the pointer, within
 * 8 px, and writes it into the chart's `ChartReadout` line the way `ReadoutTooltip` does for the SVG
 * charts; these charts carry no recharts `<Tooltip>`, which would have had nothing left to hover. The
 * canvas is `aria-hidden`: the chart's `role="img"`, its title and its caption are its text.
 */

export type MarkShape = "circle" | "cross" | "diamond" | "square" | "triangle";

export interface MarkSeries {
  /** As the series' `<Scatter name>` reads, shape included: the readout names it. */
  name: string;
  fill: string;
  shape: MarkShape;
  /** The `<Scatter fillOpacity>` it replaces; each point is filled on its own, so overlaps darken. */
  opacity?: number;
  points: readonly { x: number; y: number }[];
}

/** recharts' default symbol area in px^2: `ZAxis`'s range is [64, 64]. */
const AREA = 64;
const SQRT3 = Math.sqrt(3);
const TAN30 = Math.sqrt(1 / 3);
const CROSS: readonly (readonly [number, number])[] = [[-3, -1], [-1, -1], [-1, -3], [1, -3], [1, -1], [3, -1],
  [3, 1], [1, 1], [1, 3], [-1, 3], [-1, 1], [-3, 1]];
/** Pointer distance, squared, within which a point is read. */
const PICK2 = 8 * 8;
/** How far past the plot area the canvas reaches, in px, so a point on an axis is drawn whole, as an
 *  SVG symbol is (recharts clips scatter symbols only when an axis allows data overflow). */
const PAD = 8;

/** One symbol at (cx, cy) as d3-shape draws it (d3-shape/src/symbol/*.js), which is what recharts'
 *  `<Scatter shape>` renders, at the same area. Exported for tests/test_r85_canvas_marks.py, which
 *  compares its path with d3-shape's. */
export function drawSymbol(ctx: Pick<CanvasRenderingContext2D, "moveTo" | "lineTo" | "arc" | "closePath" | "rect">,
                           shape: MarkShape, cx: number, cy: number): void {
  if (shape === "circle") {
    const r = Math.sqrt(AREA / Math.PI);
    ctx.moveTo(cx + r, cy);
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  } else if (shape === "cross") {
    const r = Math.sqrt(AREA / 5) / 2;
    CROSS.forEach(([x, y], k) => (k ? ctx.lineTo(cx + x * r, cy + y * r) : ctx.moveTo(cx + x * r, cy + y * r)));
    ctx.closePath();
  } else if (shape === "diamond") {
    const y = Math.sqrt(AREA / (TAN30 * 2)), x = y * TAN30;
    ctx.moveTo(cx, cy - y);
    ctx.lineTo(cx + x, cy);
    ctx.lineTo(cx, cy + y);
    ctx.lineTo(cx - x, cy);
    ctx.closePath();
  } else if (shape === "square") {
    const w = Math.sqrt(AREA);
    ctx.rect(cx - w / 2, cy - w / 2, w, w);
  } else {
    const y = -Math.sqrt(AREA / (SQRT3 * 3));
    ctx.moveTo(cx, cy + y * 2);
    ctx.lineTo(cx - SQRT3 * y, cy - y);
    ctx.lineTo(cx + SQRT3 * y, cy - y);
    ctx.closePath();
  }
}

/** The two corners of every series' points, keyed for the axes, for a domain-only `<Scatter>`. */
export function corners(series: readonly MarkSeries[], xKey = "x", yKey = "y"): Record<string, number>[] {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of series) {
    for (const p of s.points) {
      if (p.x < x0) x0 = p.x;
      if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.y > y1) y1 = p.y;
    }
  }
  return Number.isFinite(x0) ? [{ [xKey]: x0, [yKey]: y0 }, { [xKey]: x1, [yKey]: y1 }] : [];
}

/** The domain-only `<Scatter>`'s shape: an empty group, which recharts' types require over null. */
export const NO_MARK = () => <g />;

type Scale = (v: number) => number;

export function CanvasMarks({
  series, xName, yName, format, xAxisMap, yAxisMap, offset,
}: {
  series: readonly MarkSeries[];
  /** The axes' names, as the readout prints them. */
  xName: string;
  yName: string;
  format: (v: number) => string;
  // Injected by recharts' `Customized`.
  xAxisMap?: Record<string, { scale?: Scale; allowDataOverflow?: boolean }>;
  yAxisMap?: Record<string, { scale?: Scale; allowDataOverflow?: boolean }>;
  offset?: { left?: number; top?: number; width?: number; height?: number };
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const placed = useRef<{ px: Float32Array; py: Float32Array; s: Int32Array; i: Int32Array } | null>(null);
  const [active, setActive] = useState<{ s: number; i: number } | null>(null);
  const readout = useReadoutLine();
  const xAxis = xAxisMap ? Object.values(xAxisMap)[0] : undefined;
  const yAxis = yAxisMap ? Object.values(yAxisMap)[0] : undefined;
  const xScale = xAxis?.scale, yScale = yAxis?.scale;
  // Clipped to the plot area exactly where recharts clips a scatter: when an axis allows overflow.
  const pad = xAxis?.allowDataOverflow || yAxis?.allowDataOverflow ? 0 : PAD;
  const plotW = Math.max(0, offset?.width ?? 0), plotH = Math.max(0, offset?.height ?? 0);
  const left = (offset?.left ?? 0) - pad, top = (offset?.top ?? 0) - pad;
  const w = plotW ? plotW + 2 * pad : 0, h = plotH ? plotH + 2 * pad : 0;

  useLayoutEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !xScale || !yScale || !w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const total = series.reduce((a, s) => a + s.points.length, 0);
    const px = new Float32Array(total), py = new Float32Array(total);
    const si = new Int32Array(total), ii = new Int32Array(total);
    let n = 0;
    series.forEach((s, k) => {
      ctx.fillStyle = s.fill;
      ctx.globalAlpha = s.opacity ?? 1;
      s.points.forEach((p, j) => {
        const x = xScale(p.x) - left, y = yScale(p.y) - top;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        ctx.beginPath();
        drawSymbol(ctx, s.shape, x, y);
        ctx.fill();
        px[n] = x; py[n] = y; si[n] = k; ii[n] = j; n++;
      });
    });
    ctx.globalAlpha = 1;
    placed.current = { px: px.subarray(0, n), py: py.subarray(0, n), s: si.subarray(0, n), i: ii.subarray(0, n) };
  }, [series, xScale, yScale, left, top, w, h]);

  const pick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const at = placed.current;
    if (!at) return;
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    // The last drawn wins a tie, as the topmost SVG symbol took the hover.
    let best = -1, bestD = PICK2;
    for (let k = 0; k < at.px.length; k++) {
      const dx = at.px[k] - mx, dy = at.py[k] - my, d = dx * dx + dy * dy;
      if (d <= bestD) { bestD = d; best = k; }
    }
    if (best >= 0) setActive({ s: at.s[best], i: at.i[best] });
  };

  let line: ReactNode = null;
  const s = active ? series[active.s] : undefined;
  const p = active && s ? s.points[active.i] : undefined;
  if (s && p) line = <><b>{s.name}: </b>{`${xName} ${format(p.x)} · ${yName} ${format(p.y)}`}</>;
  if (line && readout) readout.last.current = line;

  return (
    <>
      <foreignObject x={left} y={top} width={w} height={h}>
        <canvas ref={ref} aria-hidden="true" onPointerMove={pick}
                style={{ width: w, height: h, display: "block" }} />
      </foreignObject>
      {readout?.el
        ? createPortal(readout.last.current ?? <span className="muted">{readout.hint}</span>, readout.el)
        : null}
    </>
  );
}
