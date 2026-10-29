// Text drawn over text, or out of its box, on every page after a run (Round 80, the user's review).
//
// The user found labels running out of their boxes and over each other in the E2E diagrams during a
// run; the README's screenshots then showed an axis title drawn over its tick labels. Each was
// visible to anyone who looked and to no test. This drives a headless Chrome over the DevTools
// protocol (no dependency, the same way capture_screenshots.mjs does), runs each page's pipeline,
// scrolls through so lazily drawn charts render, and then measures, in the page:
//   - inside every SVG, any two text elements whose boxes overlap;
//   - any text that runs past its own SVG, so the SVG clips it;
//   - any text that runs out of the box it labels (a <rect> and a <text> in one <g>), or into a dot
//     drawn in that box;
//   - any pipeline-graph node whose content overflows it;
//   - any HTML text or table wider than a box that does not scroll, and a page that scrolls sideways.
//
//   cd web && VITE_CLIENT_ONLY=1 npx vite build --outDir /tmp/qsh-check && npx vite preview --outDir /tmp/qsh-check --port 4181 &
//   node scripts/check_text_overlap.mjs http://localhost:4181 [width height]
//
// Exit 1 lists every finding. Run against the deployed site as well: the fonts there are the
// visitor's, and a label that fits one font can overflow another. The page font is `system-ui`,
// so the scan measures whatever this machine resolves that to; GitHub's Ubuntu runners resolve it
// to DejaVu Sans, which is about a tenth wider than the font the pages were drawn in, and CI runs
// the scan there for that reason. To try DejaVu locally, point FONTCONFIG_FILE at a fontconfig
// file that prepends "DejaVu Sans" to `system-ui`.
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = (process.argv[2] ?? "http://localhost:4181").replace(/\/$/, "");
const WIDTH = Number(process.argv[3] ?? 1280), HEIGHT = Number(process.argv[4] ?? 720);
const CHROME = process.env.CHROME ?? "google-chrome";
const PORT = Number(process.env.CDP_PORT ?? 9355);   // set it to run two scans at once
const PAGES = [
  { path: "/" }, { path: "/value" },
  { path: "/validations", click: ["Run ML-KEM", "Run ML-DSA", "Run handshake", "Run BB84"] },
  { path: "/orchestration", run: true }, { path: "/usecases" },
  { path: "/usecase/finance_aml", run: true }, { path: "/usecase/medical_fl", run: true },
  { path: "/usecase/genomics", run: true }, { path: "/usecase/telecom_ids", run: true },
  { path: "/usecase/satellite_comms", run: true },
  { path: "/usecase/satellite_comms", tab: "Network" },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function cdp() {
  for (let i = 0; i < 50; i++) {
    try {
      const t = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" })).json();
      const ws = new WebSocket(t.webSocketDebuggerUrl);
      await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
      let id = 0; const pending = new Map();
      ws.onmessage = (m) => { const msg = JSON.parse(m.data); if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); } };
      const send = (method, params = {}) => new Promise((ok) => { const n = ++id; pending.set(n, ok); ws.send(JSON.stringify({ id: n, method, params })); });
      return { send, close: () => ws.close() };
    } catch { await sleep(200); }
  }
  throw new Error("Chrome's DevTools endpoint never came up");
}
const ev = async (s, expression) =>
  (await s.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;

/** Runs in the page. Boxes are shrunk a little before comparing, so glyph bounding boxes that merely
 *  touch (adjacent tick labels, a label beside its leader line) are not reported. */
const MEASURE = `(() => {
  const out = [];
  const visible = (el) => { const cs = getComputedStyle(el); return cs.visibility !== "hidden" && cs.display !== "none" && +cs.opacity > 0; };
  const shrink = (r, fx, fy) => ({ l: r.left + r.width * fx, r: r.right - r.width * fx, t: r.top + r.height * fy, b: r.bottom - r.height * fy });
  const hit = (a, b) => a.l < b.r && b.l < a.r && a.t < b.b && b.t < a.b;
  const name = (t) => JSON.stringify(t.textContent.trim().slice(0, 48));
  // A rotated label's screen box is far larger than its glyphs, so two parallel tick labels at an
  // angle always "overlap" by box. Rotated text is compared by its true outline instead: the four
  // corners of its own bounding box carried through its transform, tested by separating axes.
  const corners = (t, f = 0.06) => {
    const b = t.getBBox(), m = t.getScreenCTM();
    const dx = b.width * f, dy = b.height * 0.2;
    return [[b.x + dx, b.y + dy], [b.x + b.width - dx, b.y + dy], [b.x + b.width - dx, b.y + b.height - dy], [b.x + dx, b.y + b.height - dy]]
      .map(([x, y]) => [m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f]);
  };
  const rotated = (t) => { const m = t.getScreenCTM(); return Math.abs(m.b) > 1e-3 || Math.abs(m.c) > 1e-3; };
  const separated = (P, Q) => {
    for (const poly of [P, Q]) for (let i = 0; i < 4; i++) {
      const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % 4], nx = y2 - y1, ny = x1 - x2;
      const pr = (R) => R.map(([x, y]) => x * nx + y * ny);
      const a = pr(P), b = pr(Q);
      if (Math.max(...a) < Math.min(...b) || Math.max(...b) < Math.min(...a)) return true;
    }
    return false;
  };
  for (const svg of document.querySelectorAll("svg")) {
    const sr = svg.getBoundingClientRect();
    if (sr.width < 60 || sr.height < 40 || !visible(svg)) continue;          // icons, sparklines
    const where = svg.getAttribute("aria-label") || svg.closest("[aria-label]")?.getAttribute("aria-label")
      || svg.closest("figure")?.querySelector("figcaption")?.textContent?.trim().slice(0, 50) || "an SVG";
    const texts = [...svg.querySelectorAll("text")].filter((t) => t.textContent.trim() && visible(t))
      .map((t) => ({ t, r: t.getBoundingClientRect() })).filter((b) => b.r.width > 0 && b.r.height > 0);
    for (let i = 0; i < texts.length; i++) {
      const a = texts[i];
      // Any edge: a long rotated tick label runs out of the BOTTOM of a chart, which cut the first
      // letter off "photon_number_splitting" and was missed while only the sides were checked.
      if (a.r.left < sr.left - 2 || a.r.right > sr.right + 2 || a.r.top < sr.top - 2 || a.r.bottom > sr.bottom + 2)
        out.push(\`[\${where}] text \${name(a.t)} runs past its SVG (clipped)\`);
      for (let j = i + 1; j < texts.length; j++) {
        const b = texts[j];
        const overlap = rotated(a.t) || rotated(b.t)
          ? !separated(corners(a.t), corners(b.t))
          : hit(shrink(a.r, 0.04, 0.2), shrink(b.r, 0.04, 0.2));
        if (overlap) out.push(\`[\${where}] text \${name(a.t)} overlaps text \${name(b.t)}\`);
      }
    }
    // Text across the edge of a drawn box it does not belong to: partly inside, partly out. A label
    // wholly inside a box or wholly outside it is fine; one on the border is drawn through. On the
    // Overview, "count only - not mixed in" ran into the QKD box's border, and the text-to-text and
    // own-box tests could not see it because the label sits in no box. Only stroked boxes count:
    // a chart's bars and backgrounds carry labels by design.
    const boxes = [...svg.querySelectorAll("rect")].filter((r) => {
      const cs = getComputedStyle(r), b = r.getBoundingClientRect();
      return cs.stroke !== "none" && parseFloat(cs.strokeWidth) > 0 && b.width > 30 && b.height > 20 && visible(r)
        && !(b.width >= sr.width - 4 && b.height >= sr.height - 4);
    }).map((r) => r.getBoundingClientRect());
    // A clearance, not an overlap: that label touched the border by 1 px in one font and crossed it by
    // ten in DejaVu Sans, and a label touching a border already reads as drawn through it.
    for (const { t, r } of texts) {
      const a = { l: r.left, r: r.right, t: r.top + r.height * 0.2, b: r.bottom - r.height * 0.2 };
      for (const b of boxes) {
        const near = hit(a, { l: b.left - 2, r: b.right + 2, t: b.top - 2, b: b.bottom + 2 });
        const within = a.l >= b.left && a.r <= b.right && a.t >= b.top && a.b <= b.bottom;
        if (near && !within) { out.push(\`[\${where}] text \${name(t)} touches or crosses the edge of a box\`); break; }
      }
    }
    for (const g of svg.querySelectorAll("g")) {
      const rects = [...g.children].filter((c) => c.tagName === "rect");
      const labels = [...g.children].filter((c) => c.tagName === "text" && c.textContent.trim());
      if (rects.length !== 1 || labels.length === 0) continue;
      const box = rects[0].getBoundingClientRect();
      // A dot drawn in the box (a status badge) is not text, so the text-on-text test cannot see a
      // title running into it; in DejaVu Sans two titles on the Overview did.
      const dots = [...g.children].filter((c) => c.tagName === "circle").map((c) => c.getBoundingClientRect());
      for (const t of labels) {
        const r = t.getBoundingClientRect();
        if (r.width > 0 && (r.left < box.left - 1 || r.right > box.right + 1))
          out.push(\`[\${where}] text \${name(t)} runs out of its box\`);
        const s = shrink(r, 0.01, 0.2);
        if (dots.some((d) => hit(s, { l: d.left, r: d.right, t: d.top, b: d.bottom })))
          out.push(\`[\${where}] text \${name(t)} runs into the dot in its box\`);
      }
    }
  }
  // A node's TEXT against its box. Not scrollWidth/scrollHeight: the connection handles sit half
  // outside the node by design and count toward both, so every node read as overflowing.
  for (const n of document.querySelectorAll(".react-flow__node")) {
    const box = n.getBoundingClientRect();
    const walk = document.createTreeWalker(n, NodeFilter.SHOW_TEXT);
    const range = document.createRange();
    for (let tn = walk.nextNode(); tn; tn = walk.nextNode()) {
      if (!tn.textContent.trim()) continue;
      range.selectNodeContents(tn);
      for (const r of range.getClientRects()) {
        if (r.left < box.left - 1 || r.right > box.right + 1 || r.top < box.top - 1 || r.bottom > box.bottom + 1)
          out.push(\`[pipeline graph] text \${JSON.stringify(tn.textContent.trim().slice(0, 40))} runs out of its node\`);
      }
    }
  }
  // HTML wider than its box. A scroller may scroll - a wide table or diagram on a phone scrolls in its
  // card by design - but by a few pixels it is a margin or a rounding, not content, and the reader
  // gets a scrollbar for nothing: one unbroken metric name made the Metrics card scroll by 5.
  // Content wider than a box that does not scroll runs over its neighbour, or is cut off where an
  // ancestor clips it; content a scrolling ancestor carries is that scroller's to judge. Screen-
  // reader-only text is clipped by design, and an ellipsis is a deliberate cut.
  const W = document.documentElement.clientWidth;
  if (document.documentElement.scrollWidth > W + 1)
    out.push(\`[page] scrolls sideways: \${document.documentElement.scrollWidth} px of content in \${W}\`);
  const scroller = (cs) => cs.overflowX === "auto" || cs.overflowX === "scroll";
  const tag = (el) => \`[\${el.tagName.toLowerCase()}] \${name(el)}\`;
  for (const el of (document.querySelector("main") ?? document.body).querySelectorAll("*")) {
    if (el.closest("svg, .react-flow, .sr-only")) continue;
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.display === "inline" || cs.visibility === "hidden" || el.clientWidth === 0) continue;
    const spill = el.scrollWidth - el.clientWidth;
    if (spill <= 1) continue;
    if (scroller(cs)) {
      // Not on a phone, where every table scrolls in its own box by design, and not for a wide data
      // table, which scrolls on a desktop too once the window is narrower than it: there the amount
      // is the window's, not a fault. A key-value card that scrolls by a few pixels on a desktop has
      // a word that will not break or a margin too many (the Metrics card, 5 px).
      const wideTable = [...el.querySelectorAll("tr")].some((tr) => tr.children.length >= 7);
      if (spill <= 16 && W > 900 && !wideTable) out.push(\`\${tag(el)} scrolls sideways by \${spill} px\`);
      continue;
    }
    if (cs.textOverflow === "ellipsis") continue;
    if (cs.overflowX !== "visible") { out.push(\`\${tag(el)} is \${spill} px wider than its box, and cut off\`); continue; }
    let up = el.parentElement, fate = "runs over its neighbour";
    for (; up && up !== document.body; up = up.parentElement) {
      const ucs = getComputedStyle(up);
      if (scroller(ucs)) { fate = null; break; }
      if (ucs.overflowX !== "visible") { fate = "is cut off"; break; }
    }
    if (fate) out.push(\`\${tag(el)} is \${spill} px wider than its box and \${fate}\`);
  }
  return [...new Set(out)];
})()`;

/** Runs in the page: the family whose width matches `system-ui` on three strings, or the width. */
const FONT_PROBE = `(() => {
  const c = document.createElement("canvas").getContext("2d");
  const probe = ["The quick brown fox 2200 ms", "encrypted_tier_within_operational_window", "WMWMWM iiiill 0123456789"];
  const widths = (f) => probe.map((t) => { c.font = "13px " + f; return c.measureText(t).width; });
  const want = widths("system-ui");
  const families = ["DejaVu Sans", "Noto Sans", "Noto Sans CJK JP", "Liberation Sans", "Ubuntu", "Cantarell",
    "Segoe UI", "Roboto", "Helvetica", "Arial", "Verdana", "FreeSans", "Nimbus Sans"];
  const eq = (a, b) => a.every((w, i) => Math.abs(w - b[i]) < 0.01);
  // A family that is not installed falls back, and would match whatever the fallback matches; it is
  // installed when it measures the same with a monospace fallback as with a serif one.
  const installed = (f) => eq(widths(\`"\${f}", monospace\`), widths(\`"\${f}", serif\`));
  const same = families.filter((f) => installed(f) && eq(widths(\`"\${f}"\`), want));
  return same.length ? same.join(" / ") : "an unlisted family (" + want.map((w) => w.toFixed(1)).join(", ") + " px)";
})()`;

async function check(s, page) {
  await s.send("Page.navigate", { url: `${BASE}${page.path}` });
  await sleep(3500);
  if (page.tab) {
    await ev(s, `[...document.querySelectorAll('[role=tab]')].find(b => b.textContent.trim() === ${JSON.stringify(page.tab)})?.click()`);
    await sleep(2500);
  }
  for (const label of page.click ?? []) {
    await ev(s, `[...document.querySelectorAll('button')].find(b => b.textContent.trim() === ${JSON.stringify(label)})?.click()`);
    await sleep(2500);
  }
  if (page.run) {
    await ev(s, `[...document.querySelectorAll('button')].find(b => /^\\u25b6 Start$/.test(b.textContent.trim()))?.click()`);
    for (let i = 0; i < 90; i++) {
      await sleep(1000);
      const done = await ev(s, `[...document.querySelectorAll('.badge-state')].some(b => /^(done|error|stopped)$/.test(b.textContent.trim()))`);
      if (done && i > 3) break;
    }
  }
  // Through the page a screen at a time, so lazily drawn charts render before they are measured.
  const height = await ev(s, "document.documentElement.scrollHeight");
  for (let y = 0; y < height; y += HEIGHT * 0.8) { await ev(s, `window.scrollTo(0, ${y})`); await sleep(350); }
  await sleep(1200);
  return (await ev(s, MEASURE)) ?? [];
}

const profile = mkdtempSync(join(tmpdir(), "qsh-overlap-"));
// On a CI runner Chrome's sandbox can be refused by the kernel's user-namespace policy; the scan loads
// only this checkout's own build from localhost, so it runs unsandboxed there.
// SCAN_OFFLINE=1 lets the page reach localhost and nothing else. CI is offline by design (ci.yml's
// header), and the one host a run fetches from is the Hugging Face Hub, for the published models:
// offline, that stage fails and the rest of the run goes on, so the scan measures every page with
// the model's results absent. The local and deploy scans measure them with the results.
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  `--window-size=${WIDTH},${HEIGHT}`, "--hide-scrollbars", "--no-first-run", ...(process.env.CI ? ["--no-sandbox"] : []),
  ...(process.env.SCAN_OFFLINE ? ["--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost"] : []),
  "about:blank"], { stdio: "ignore" });
let failures = 0;
try {
  const s = await cdp();
  await s.send("Page.enable"); await s.send("Runtime.enable");
  await s.send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false });
  console.log(`checking text layout on ${PAGES.length} views of ${BASE} at ${WIDTH} x ${HEIGHT}`
    + (process.env.SCAN_OFFLINE ? ", offline (localhost only)" : ""));
  // Which font the pages are measured in. The page asks for `system-ui`, and what that resolves to is
  // the machine's choice, so the scan says which installed family it measured as wide as: a claim
  // about "the runner's font" is then read from the run, not assumed.
  console.log(`page font: system-ui measures as ${await ev(s, FONT_PROBE)}`);
  for (const page of PAGES) {
    const found = await check(s, page);
    const label = `${page.path}${page.tab ? ` [${page.tab}]` : ""}`;
    if (found.length) { failures += found.length; console.log(`  FAIL ${label}`); for (const f of found) console.log(`       ${f}`); }
    else console.log(`  ok   ${label}`);
  }
  s.close();
} finally {
  chrome.kill("SIGTERM");
  await sleep(400);
  rmSync(profile, { recursive: true, force: true });
}
if (failures) { console.log(`\n${failures} text-layout finding(s)`); process.exit(1); }
console.log("PASS  no text over text, out of its box, or clipped, and no table wider than its box, on any page");
