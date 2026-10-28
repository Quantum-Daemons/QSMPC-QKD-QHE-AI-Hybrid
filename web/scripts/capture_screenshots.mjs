// Screenshots of every page for the README, from a local client-only build (Round 80, the user's
// request). Dependency-free: it drives a headless Chrome over the DevTools protocol with Node's own
// fetch and WebSocket, so the images can be regenerated each round instead of going stale.
//
//   cd web
//   VITE_CLIENT_ONLY=1 npx vite build --outDir /tmp/qsh-shots && npx vite preview --outDir /tmp/qsh-shots --port 4180 &
//   node scripts/capture_screenshots.mjs http://localhost:4180 ../docs/images
//
// Always against localhost: a headless capture has no address bar, and the build it serves is the
// one in this checkout, so an image shows exactly what the commit beside it renders.

import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const BASE = (process.argv[2] ?? "http://localhost:4180").replace(/\/$/, "");
const OUT = resolve(process.argv[3] ?? "../docs/images");
const CHROME = process.env.CHROME ?? "google-chrome";
const PORT = 9333;
const WIDTH = 1366, HEIGHT = 1500;

/** What each image shows. `run` starts the scenario and waits for it; `tab` opens a tab first. */
const SHOTS = [
  { name: "overview", path: "/" },
  { name: "value", path: "/value" },
  { name: "validations", path: "/validations", click: ["Run ML-KEM", "Run BB84"] },
  { name: "orchestration", path: "/orchestration", run: true },
  { name: "usecases", path: "/usecases" },
  { name: "usecase-finance", path: "/usecase/finance_aml", run: true },
  { name: "usecase-medical", path: "/usecase/medical_fl", run: true },
  { name: "usecase-genomics", path: "/usecase/genomics", run: true },
  { name: "usecase-telecom", path: "/usecase/telecom_ids", run: true },
  { name: "usecase-satellite", path: "/usecase/satellite_comms", run: true },
  { name: "usecase-satellite-network", path: "/usecase/satellite_comms", tab: "Network", scrollTo: "One pass over one station" },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function cdpSession() {
  for (let i = 0; i < 50; i++) {
    try {
      const t = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" })).json();
      const ws = new WebSocket(t.webSocketDebuggerUrl);
      await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
      let id = 0;
      const pending = new Map();
      const events = [];
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data);
        if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
        else if (msg.method) events.push(msg.method);
      };
      const send = (method, params = {}) => new Promise((ok) => {
        const n = ++id; pending.set(n, ok); ws.send(JSON.stringify({ id: n, method, params }));
      });
      return { send, events, close: () => ws.close() };
    } catch { await sleep(200); }
  }
  throw new Error("Chrome's DevTools endpoint never came up");
}

const evaluate = async (s, expression) =>
  (await s.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;

async function shoot(s, shot) {
  await s.send("Page.navigate", { url: `${BASE}${shot.path}` });
  await sleep(4000);
  if (shot.tab) {
    await evaluate(s, `[...document.querySelectorAll('[role=tab]')].find(b => b.textContent.trim() === ${JSON.stringify(shot.tab)})?.click()`);
    await sleep(2500);
  }
  for (const label of shot.click ?? []) {
    await evaluate(s, `[...document.querySelectorAll('button')].find(b => b.textContent.trim() === ${JSON.stringify(label)})?.click()`);
    await sleep(3000);
  }
  if (shot.run) {
    await evaluate(s, `[...document.querySelectorAll('button')].find(b => /^\\u25b6 Start$/.test(b.textContent.trim()))?.click()`);
    // A run is done when no stage is still pending; capped so a stalled run still yields an image.
    for (let i = 0; i < 60; i++) {
      await sleep(1000);
      const busy = await evaluate(s, `document.body.innerText.includes('running') || [...document.querySelectorAll('button')].some(b => b.textContent.trim() === '\\u23f8 Pause' && !b.disabled)`);
      if (!busy && i > 5) break;
    }
    await sleep(1500);
  }
  // No fit-view press for the pipeline graph: since Round 80 it sets its own scale from its frame's
  // width, and pressing the control would swap that for React Flow's default fit.
  if (shot.scrollTo) {
    await evaluate(s, `(() => { const h = [...document.querySelectorAll('h2,h3,h4')].find(e => e.textContent.trim().startsWith(${JSON.stringify(shot.scrollTo)}));
                               if (h) window.scrollTo(0, Math.max(0, h.getBoundingClientRect().top + window.scrollY - 24)); })()`);
    await sleep(3500);
  }
  // THE VIEWPORT AS IT IS, not `captureBeyondViewport`: that resizes the page for the capture, the
  // charts' ResizeObservers fire, and the image catches them mid-redraw as empty frames.
  const shotRes = await s.send("Page.captureScreenshot", { format: "jpeg", quality: 80 });
  const file = join(OUT, `${shot.name}.jpg`);
  writeFileSync(file, Buffer.from(shotRes.result.data, "base64"));
  console.log(`  ${shot.name}.jpg  ${shot.path}${shot.tab ? ` [${shot.tab}]` : ""}`);
}

mkdirSync(OUT, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), "qsh-shots-"));
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  `--window-size=${WIDTH},${HEIGHT}`, "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
  "about:blank"], { stdio: "ignore" });
try {
  const s = await cdpSession();
  await s.send("Page.enable");
  await s.send("Runtime.enable");
  await s.send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false });
  console.log(`capturing ${SHOTS.length} pages from ${BASE} into ${OUT}`);
  for (const shot of SHOTS) await shoot(s, shot);
  s.close();
} finally {
  chrome.kill("SIGTERM");
  await sleep(500);
  rmSync(profile, { recursive: true, force: true });
}
