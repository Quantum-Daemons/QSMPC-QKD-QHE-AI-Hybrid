// Three orthogonal planes of a 28 x 28 x 28 chest-CT crop, drawn on 2D canvases (batch 4; the owner's choice of
// 2026-11-24: canvas 2D, three.js optional). Each plane is the crop's own 28 x 28 voxels, enlarged without smoothing,
// so nothing is drawn that the crop does not hold. The contribution map, where given, is laid over the grey: each
// voxel's share of the logit difference, warm towards "malignant" and cool towards "benign". It is exact at the
// convolution's output (clientFHESquare3d.ts, `contributionMap`), so it is the model's arithmetic, not a saliency guess.
//
// "Maximum along the axis" replaces each plane by the largest value along the axis it would slice, the maximum
// intensity projection, with the contribution summed along the same axis.
//
// THE PAGE'S TIME. Every redraw of the three planes is timed, and the median of the last 30 is on the root element
// as `data-redraw-ms`, which the browser gate reads after moving the sliders (the plan's limit: one frame, 16 ms).
import { useEffect, useMemo, useRef, useState } from "react";

export const SIDE = 28;
const ZOOM = 6;                                   // 168 CSS pixels a plane
// One colour per statement: the chart checks read a mark colour from a `const NAME = "#hex"` line.
const WARM = "#ff6b6b";
const COOL = "#4f9dff";
const CROSS = "#f6b73c";

type Plane = "axial" | "coronal" | "sagittal";
type Mode = "slice" | "mip";
const PLANES: Plane[] = ["axial", "coronal", "sagittal"];

const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const WARM_RGB = rgb(WARM), COOL_RGB = rgb(COOL);
const at = (z: number, y: number, x: number) => (z * SIDE + y) * SIDE + x;

/** The voxel a plane's row r and column c show: axial rows are y and columns x, coronal rows z and columns x,
 *  sagittal rows z and columns y, all as the crop stores them, with z drawn from the top. */
function voxel(plane: Plane, k: number, r: number, c: number): number {
  if (plane === "axial") return at(k, r, c);
  if (plane === "coronal") return at(r, k, c);
  return at(r, c, k);
}

/** One plane as 28 x 28 values (the grey) and contributions: a slice at index k, or the projection along its axis. */
function planeValues(volume: ArrayLike<number>, map: ArrayLike<number> | null, plane: Plane, k: number, mode: Mode) {
  const grey = new Float64Array(SIDE * SIDE), share = new Float64Array(SIDE * SIDE);
  for (let r = 0; r < SIDE; r++) {
    for (let c = 0; c < SIDE; c++) {
      if (mode === "slice") {
        const i = voxel(plane, k, r, c);
        grey[r * SIDE + c] = volume[i];
        share[r * SIDE + c] = map ? map[i] : 0;
      } else {
        let top = -Infinity, sum = 0;
        for (let j = 0; j < SIDE; j++) {
          const i = voxel(plane, j, r, c);
          if (volume[i] > top) top = volume[i];
          if (map) sum += map[i];
        }
        grey[r * SIDE + c] = top;
        share[r * SIDE + c] = sum;
      }
    }
  }
  return { grey, share };
}

/** The 99th percentile of |share| over all three planes' possible values: the overlay's full strength. A single
 *  extreme voxel then does not wash out the rest. */
function overlayScale(volume: ArrayLike<number>, map: ArrayLike<number>, mode: Mode): number {
  const vals: number[] = [];
  if (mode === "slice") for (let i = 0; i < map.length; i++) vals.push(Math.abs(map[i]));
  else for (const p of PLANES) {
    const { share } = planeValues(volume, map, p, 0, "mip");
    for (const s of share) vals.push(Math.abs(s));
  }
  vals.sort((a, b) => a - b);
  return vals[Math.floor(0.99 * (vals.length - 1))] || 1;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : 0;
}

export interface Nodule3dViewerProps {
  /** 28^3 voxels, z, y, x order, 0 to 255. */
  volume: ArrayLike<number>;
  /** 28^3 contributions in the same order, or null before a run. */
  contribution?: ArrayLike<number> | null;
  /** What the crop is, for the planes' accessible names. */
  label: string;
}

export default function Nodule3dViewer({ volume, contribution = null, label }: Nodule3dViewerProps) {
  const [pos, setPos] = useState({ z: SIDE / 2, y: SIDE / 2, x: SIDE / 2 });
  const [mode, setMode] = useState<Mode>("slice");
  const [overlay, setOverlay] = useState(true);
  const canvases = useRef<Record<Plane, HTMLCanvasElement | null>>({ axial: null, coronal: null, sagittal: null });
  const root = useRef<HTMLDivElement>(null);
  const times = useRef<number[]>([]);
  const small = useRef<HTMLCanvasElement | null>(null);
  const scale = useMemo(() => (contribution ? overlayScale(volume, contribution, mode) : 1), [volume, contribution, mode]);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      const t0 = performance.now();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (!small.current) {
        small.current = document.createElement("canvas");
        small.current.width = small.current.height = SIDE;
      }
      const sctx = small.current.getContext("2d");
      if (!sctx) return;
      const img = sctx.createImageData(SIDE, SIDE);
      for (const p of PLANES) {
        const c = canvases.current[p];
        const ctx = c?.getContext("2d");
        if (!c || !ctx) continue;
        const k = p === "axial" ? pos.z : p === "coronal" ? pos.y : pos.x;
        const { grey, share } = planeValues(volume, overlay ? contribution : null, p, k, mode);
        for (let i = 0; i < SIDE * SIDE; i++) {
          const g = grey[i];
          const a = overlay && contribution ? Math.min(1, Math.abs(share[i]) / scale) * 0.65 : 0;
          const col = share[i] >= 0 ? WARM_RGB : COOL_RGB;
          img.data[4 * i] = g * (1 - a) + col[0] * a;
          img.data[4 * i + 1] = g * (1 - a) + col[1] * a;
          img.data[4 * i + 2] = g * (1 - a) + col[2] * a;
          img.data[4 * i + 3] = 255;
        }
        sctx.putImageData(img, 0, 0);
        const px = SIDE * ZOOM * dpr;
        if (c.width !== px) { c.width = px; c.height = px; }
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(small.current, 0, 0, px, px);
        if (mode === "slice") {
          // The other two planes' positions, through the centre of their voxels.
          const [row, col] = p === "axial" ? [pos.y, pos.x] : p === "coronal" ? [pos.z, pos.x] : [pos.z, pos.y];
          const u = ZOOM * dpr;
          ctx.strokeStyle = CROSS;
          ctx.globalAlpha = 0.7;
          ctx.lineWidth = dpr;
          ctx.beginPath();
          ctx.moveTo(0, (row + 0.5) * u); ctx.lineTo(px, (row + 0.5) * u);
          ctx.moveTo((col + 0.5) * u, 0); ctx.lineTo((col + 0.5) * u, px);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
      const ms = performance.now() - t0;
      times.current = [...times.current.slice(-29), ms];
      root.current?.setAttribute("data-redraw-ms", median(times.current).toFixed(2));
      root.current?.setAttribute("data-redraws", String(times.current.length));
    });
    return () => cancelAnimationFrame(frame);
  }, [volume, contribution, pos, mode, overlay, scale]);

  const pick = (p: Plane, e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode !== "slice") return;
    const box = e.currentTarget.getBoundingClientRect();
    const r = Math.min(SIDE - 1, Math.max(0, Math.floor(((e.clientY - box.top) / box.height) * SIDE)));
    const c = Math.min(SIDE - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * SIDE)));
    setPos((q) => (p === "axial" ? { ...q, y: r, x: c } : p === "coronal" ? { ...q, z: r, x: c } : { ...q, z: r, y: c }));
  };
  const axisOf: Record<Plane, "z" | "y" | "x"> = { axial: "z", coronal: "y", sagittal: "x" };
  const caption = (p: Plane) => (mode === "slice" ? `${p}, ${axisOf[p]} = ${pos[axisOf[p]]} of 0 to 27`
    : `${p}, the maximum along ${axisOf[p]}`);

  return (
    <div ref={root} className="nodule3d-viewer" data-mode={mode}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
        {PLANES.map((p) => (
          <figure key={p} style={{ margin: 0 }}>
            <canvas ref={(el) => { canvases.current[p] = el; }} role="img" onClick={(e) => pick(p, e)}
              aria-label={`${label}: ${caption(p)}${overlay && contribution ? ", with the contribution map" : ""}`}
              style={{ width: SIDE * ZOOM, height: SIDE * ZOOM, display: "block", borderRadius: 4,
                       cursor: mode === "slice" ? "crosshair" : "default" }} />
            <figcaption className="muted" style={{ fontSize: 12, marginTop: 4 }}>{caption(p)}</figcaption>
          </figure>
        ))}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginTop: 8, alignItems: "center" }}>
        {(["z", "y", "x"] as const).map((a) => (
          <label key={a} style={{ fontSize: 12 }}>
            {a}{" "}
            <input type="range" min={0} max={SIDE - 1} value={pos[a]} disabled={mode !== "slice"}
              aria-label={`the ${a} index of the planes`}
              onChange={(e) => setPos((q) => ({ ...q, [a]: Number(e.target.value) }))} />
          </label>
        ))}
        <label style={{ fontSize: 12 }}>
          <input type="checkbox" checked={mode === "mip"} onChange={(e) => setMode(e.target.checked ? "mip" : "slice")} />
          {" "}Maximum along the axis
        </label>
        {contribution ? (
          <label style={{ fontSize: 12 }}>
            <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
            {" "}Contribution map (warm: towards malignant, cool: towards benign)
          </label>
        ) : null}
      </div>
    </div>
  );
}
