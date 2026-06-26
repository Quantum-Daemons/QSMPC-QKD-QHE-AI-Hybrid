// Client-side download helpers: CSV, JSON, high-DPI PNG (DOM + SVG), and a real
// animated GIF of the orchestration run. All export runs in the browser (no server),
// so it works identically on the public demo VPS.
import { toPng, toCanvas } from "html-to-image";
import Papa from "papaparse";

// gif.js encodes discrete frames -> a real animated GIF. The worker is bundled by Vite.
import GIF from "gif.js";
import gifWorkerUrl from "gif.js/dist/gif.worker.js?url";

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadJSON(data: unknown, filename = "qsh-export.json"): void {
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), filename);
}

export function downloadCSV(rows: Record<string, unknown>[], filename = "qsh-export.csv"): void {
  const csv = Papa.unparse(rows);
  downloadBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), filename);
}

export function downloadText(text: string, filename: string, mime = "text/plain"): void {
  downloadBlob(new Blob([text], { type: mime }), filename);
}

// High-DPI factor: 2x on standard displays, more on Retina/high-DPI.
const hiDPI = () => Math.max(2, Math.ceil((window.devicePixelRatio || 1) * 2));

// Capture any DOM node (e.g. the React-Flow DAG or a chart card) to a crisp PNG.
export async function downloadNodePNG(node: HTMLElement, filename = "qsh.png"): Promise<void> {
  const dataUrl = await toPng(node, {
    pixelRatio: hiDPI(),
    backgroundColor: "#0a0e1a",
    cacheBust: true,
  });
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

// Rasterize a standalone <svg> to a high-DPI PNG (chart export fallback).
export async function downloadSVGasPNG(svg: SVGSVGElement, filename = "chart.png"): Promise<void> {
  const scale = hiDPI();
  const xml = new XMLSerializer().serializeToString(svg);
  const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml;charset=utf-8" }));
  const img = new Image();
  const rect = svg.getBoundingClientRect();
  await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = rej; img.src = url; });
  const canvas = document.createElement("canvas");
  canvas.width = (rect.width || 640) * scale;
  canvas.height = (rect.height || 360) * scale;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  URL.revokeObjectURL(url);
  canvas.toBlob((b) => b && downloadBlob(b, filename), "image/png");
}

// Records frames of a DOM node and encodes a high-quality animated GIF.
// Used to capture the orchestration DAG animating through its stages.
export class FrameRecorder {
  private frames: HTMLCanvasElement[] = [];
  constructor(private node: HTMLElement) {}

  async capture(): Promise<void> {
    // pixelRatio 1.5 keeps GIF size sane while staying crisp
    const canvas = await toCanvas(this.node, { pixelRatio: 1.5, backgroundColor: "#0a0e1a", cacheBust: true });
    this.frames.push(canvas);
  }

  get count(): number {
    return this.frames.length;
  }

  async encodeGIF(filename = "orchestration.gif", delayMs = 600): Promise<void> {
    if (this.frames.length === 0) return;
    const w = this.frames[0].width;
    const h = this.frames[0].height;
    const gif = new GIF({ workers: 2, quality: 1, width: w, height: h, workerScript: gifWorkerUrl });
    for (const f of this.frames) gif.addFrame(f, { delay: delayMs, copy: true });
    await new Promise<void>((resolve) => {
      gif.on("finished", (blob: Blob) => { downloadBlob(blob, filename); resolve(); });
      gif.render();
    });
  }
}
