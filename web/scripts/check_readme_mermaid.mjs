// The README's Mermaid diagrams, measured for the faults the report and page gates look for (Round 80).
//
// The user found text over lines in the report's Figure 1 and a label at the edge of its box in
// Figure 2, and asked for every figure in the documents to be checked for the same. The report's
// figures are checked by `report/build_figures.py` and `make report`, the pages by
// check_text_overlap.mjs; this covers the README, whose diagrams GitHub draws with Mermaid. Each
// block is rendered with Mermaid in a headless Chrome and measured:
//   - a node's text outside its node (a decision by its diamond, not the diamond's bounding box);
//   - any label over another label, node or edge;
//   - an edge label over a node;
//   - an edge drawn through a node's text.
//
// Each diagram must also draw at full size in GitHub's README column, about 830 px beside the
// repository sidebar: a wider one is scaled down, text and all (checklist DC14).
//
// A CI gate since Round 80, at the owner's choice: Mermaid is a dev dependency for this check alone
// (it is never imported by the app), and the web job and scripts/ci_local.sh step 8 run
//   npm run check:readme-mermaid
// which is `node scripts/check_readme_mermaid.mjs [README.md] [mermaid.min.js]`, both defaulted.
//
// Exit 1 lists every finding. Mermaid sizes a node to its text in the font it measures with, which
// is not GitHub's, so look at each diagram on GitHub as well.
import { spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import http from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const readme = process.argv[2] ?? join(WEB, "..", "README.md");
const bundle = process.argv[3] ?? join(WEB, "node_modules", "mermaid", "dist", "mermaid.min.js");
const COLUMN_PX = 830;
if (!existsSync(bundle)) {
  console.error(`no Mermaid bundle at ${bundle}: run \`npm ci\` in web/ (mermaid is a dev dependency)`);
  process.exit(2);
}
const CHROME = process.env.CHROME ?? "google-chrome";
const PORT = 9366, HTTP_PORT = 4191;
const blocks = [...readFileSync(readme, "utf8").matchAll(/(?:<!-- figure: ([\w-]+) -->\n)?```mermaid\n([\s\S]*?)```/g)]
  .map((m, i) => [m[1] ?? `block ${i + 1}`, m[2]]);
const dir = mkdtempSync(join(tmpdir(), "qsh-mermaid-"));
copyFileSync(bundle, join(dir, "mermaid.min.js"));
// 830 px: the README column beside the repository sidebar (checklist DC14).
writeFileSync(join(dir, "index.html"), `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="background:#fff;margin:8px;width:830px">${blocks.map((_, i) => `<div id="m${i}"></div>`).join("")}
<script src="/mermaid.min.js"></script><script>
window.__failed = [];
mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });
(async () => {
  const blocks = ${JSON.stringify(blocks)};
  for (let i = 0; i < blocks.length; i++) {
    try { document.getElementById("m" + i).innerHTML = (await mermaid.render("c" + i, blocks[i][1])).svg; }
    catch (e) { window.__failed.push(blocks[i][0] + ": Mermaid could not render it: " + e); }
  }
  window.__done = true;
})();
</script></body></html>`);
const server = http.createServer((req, res) => {
  try { res.end(readFileSync(join(dir, req.url === "/" ? "index.html" : req.url.slice(1)))); }
  catch { res.statusCode = 404; res.end(); }
}).listen(HTTP_PORT);

/** Runs in the page. Text is measured by its rendered line boxes (a Range over each text node), so a
 *  label's padding inside its node does not count as text. */
const MEASURE = `(() => {
  const out = [], sizes = [];
  const names = ${JSON.stringify(blocks.map((b) => b[0]))};
  const textRects = (root) => {
    const rs = [], w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), rg = document.createRange();
    for (let n = w.nextNode(); n; n = w.nextNode()) {
      if (!n.textContent.trim()) continue;
      rg.selectNodeContents(n);
      rs.push(...[...rg.getClientRects()].filter((r) => r.width > 0));
    }
    return rs;
  };
  const inside = (r, b) => r.left >= b.left - 0.5 && r.right <= b.right + 0.5 && r.top >= b.top - 0.5 && r.bottom <= b.bottom + 0.5;
  const hit = (a, b) => a.left + 1 < b.right && b.left + 1 < a.right && a.top + 1 < b.bottom && b.top + 1 < a.bottom;
  document.querySelectorAll("svg[id^=c]").forEach((svg, i) => {
    const name = names[i];
    const nodes = [...svg.querySelectorAll("g.node")].map((g) => {
      const shape = g.querySelector(":scope > rect, :scope > polygon, :scope > path, :scope > g > rect, :scope > g > polygon");
      return { shape, box: shape.getBoundingClientRect(), label: g.textContent.trim().slice(0, 40), rects: textRects(g) };
    });
    for (const n of nodes) {
      const diamond = n.shape.tagName === "polygon" && n.shape.points.numberOfItems === 4;
      const b = n.box, cx = (b.left + b.right) / 2, cy = (b.top + b.bottom) / 2;
      for (const r of n.rects) {
        if (!inside(r, b)) { out.push(name + ": the text of " + JSON.stringify(n.label) + " runs out of its node"); break; }
        // Inside a diamond when |dx| / half-width + |dy| / half-height <= 1 at every corner.
        if (diamond && [[r.left, r.top], [r.right, r.top], [r.left, r.bottom], [r.right, r.bottom]]
              .some(([x, y]) => Math.abs(x - cx) / (b.width / 2) + Math.abs(y - cy) / (b.height / 2) > 1)) {
          out.push(name + ": the text of " + JSON.stringify(n.label) + " runs out of its diamond"); break;
        }
      }
    }
    const edgeLabels = [...svg.querySelectorAll("g.edgeLabel")].filter((g) => g.textContent.trim())
      .map((g) => ({ label: g.textContent.trim(), rects: textRects(g) }));
    for (const e of edgeLabels) for (const n of nodes)
      if (e.rects.some((r) => hit(r, n.box))) out.push(name + ": edge label " + JSON.stringify(e.label) + " is over node " + JSON.stringify(n.label));
    const all = [...nodes, ...edgeLabels];
    for (let a = 0; a < all.length; a++) for (let b = a + 1; b < all.length; b++)
      if (all[a].rects.some((ra) => all[b].rects.some((rb) => hit(ra, rb))))
        out.push(name + ": " + JSON.stringify(all[a].label) + " overlaps " + JSON.stringify(all[b].label));
    // Each edge sampled every 2 units along its path, in screen coordinates.
    for (const p of svg.querySelectorAll("g.edgePaths path, path.flowchart-link")) {
      const m = p.getScreenCTM(), pts = [];
      for (let s = 0, L = p.getTotalLength(); s <= L; s += 2) {
        const q = p.getPointAtLength(s);
        pts.push([m.a * q.x + m.c * q.y + m.e, m.b * q.x + m.d * q.y + m.f]);
      }
      for (const n of nodes)
        if (n.rects.some((r) => pts.some(([x, y]) => x > r.left + 1 && x < r.right - 1 && y > r.top + 1 && y < r.bottom - 1)))
          out.push(name + ": an edge runs through the text of " + JSON.stringify(n.label));
    }
    const vb = svg.viewBox.baseVal;
    if (vb.width > ${COLUMN_PX})
      out.push(name + ": " + Math.round(vb.width) + " px wide, so GitHub's ${COLUMN_PX} px column draws it scaled down");
    sizes.push(name + ": " + Math.round(vb.width) + " x " + Math.round(vb.height) + ", " + nodes.length + " nodes");
  });
  return { out: [...new Set(out)], sizes };
})()`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const profile = mkdtempSync(join(tmpdir(), "qsh-mermaid-chrome-"));
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--window-size=900,900", "--no-first-run", ...(process.env.CI ? ["--no-sandbox"] : []), "about:blank"],
  { stdio: "ignore" });   // unsandboxed on a CI runner, as check_text_overlap.mjs explains
let result = { out: [], sizes: [] }, failed = [];
try {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    try { target = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" })).json(); }
    catch { await sleep(200); }
  }
  if (!target) throw new Error("Chrome's DevTools endpoint never came up");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
  let id = 0; const pending = new Map();
  ws.onmessage = (m) => { const msg = JSON.parse(m.data); if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); } };
  const send = (method, params = {}) => new Promise((ok) => { const n = ++id; pending.set(n, ok); ws.send(JSON.stringify({ id: n, method, params })); });
  const ev = async (expression) => (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;
  await send("Page.enable");
  await send("Page.navigate", { url: `http://127.0.0.1:${HTTP_PORT}/` });
  for (let i = 0; i < 60 && !(await ev("window.__done === true")); i++) await sleep(500);
  failed = (await ev("window.__failed")) ?? ["the page never finished rendering"];
  result = (await ev(MEASURE)) ?? result;
  ws.close();
} finally {
  chrome.kill("SIGTERM");
  server.close();
  await sleep(400);
  rmSync(profile, { recursive: true, force: true });
  rmSync(dir, { recursive: true, force: true });
}
console.log(`checking ${blocks.length} Mermaid diagrams in ${readme}`);
for (const s of result.sizes) console.log(`  ${s}`);
const findings = [...failed, ...result.out];
if (findings.length || result.sizes.length !== blocks.length) {
  for (const f of findings) console.log(`  FAIL ${f}`);
  if (result.sizes.length !== blocks.length) console.log(`  FAIL measured ${result.sizes.length} of ${blocks.length} diagrams`);
  process.exit(1);
}
console.log(`PASS  no label out of its node, over another, or crossed by an edge; every diagram fits ${COLUMN_PX} px`);
