// The site under its Content-Security-Policy: every page, every model tab, the pipeline runs and the walkthrough, in
// headless Chrome, with every dedicated worker attached, and any CSP violation is a failure (Round 86, 2026-11-20).
//
//   node scripts/check_csp.mjs http://localhost:4183     a local Caddy serving a build with docker/env/Caddyfile.client-only
//   node scripts/check_csp.mjs https://<public name>      the deployed site
//
// WHY. The owner chose an enforced policy rather than a report-only one, so a source the policy misses breaks a
// feature instead of logging a warning. `vite preview`, which the layout scan reads, sends none of Caddy's headers, so
// that scan cannot see a violation; this reads the pages through the Caddyfile itself. A violation is counted from
// three places, since none sees all of them: the document's `securitypolicyviolation` events, the console of the page
// and of each worker (a worker reports its own), and exceptions whose text names the policy (an EvalError from a
// blocked `new Function`, for one). The page must also say it is cross-origin isolated, which COEP and COOP provide.
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = (process.argv[2] ?? "http://localhost:4183").replace(/\/$/, "");
const CHROME = process.env.CHROME ?? "google-chrome";
const PORT = Number(process.env.CDP_PORT ?? 9356);
const USE_CASES = ["finance_aml", "medical_fl", "genomics", "telecom_ids", "satellite_comms", "defi_lending"];
const PAGES = [
  { path: "/" }, { path: "/value" }, { path: "/usecases" },
  { path: "/validations", click: ["Run ML-KEM", "Run ML-DSA", "Run handshake", "Run BB84"] },
  { path: "/orchestration", run: true, tabs: true },
  ...USE_CASES.map((uc) => ({ path: `/usecase/${uc}`, run: true, tabs: true, actions: true })),
  { path: "/walkthrough/", settle: 45000 },
  { path: "/no-such-page" },
].filter((p) => !process.env.CSP_PAGES || process.env.CSP_PAGES.split(",").includes(p.path));
// CSP_PAGES=/a,/b checks only those paths: for a seeded run, which must fail.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const POLICY = /Content Security Policy|Content-Security-Policy|EvalError|unsafe-eval|wasm-unsafe-eval/i;

async function connect() {
  for (let i = 0; i < 300; i++) {
    try {
      const ver = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json();
      const ws = new WebSocket(ver.webSocketDebuggerUrl);
      await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
      let id = 0; const pending = new Map(); const listeners = [];
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data);
        if (msg.id && pending.has(msg.id)) { const { ok, bad } = pending.get(msg.id); pending.delete(msg.id); msg.error ? bad(new Error(msg.error.message)) : ok(msg.result); }
        else if (msg.method) for (const l of listeners) l(msg);
      };
      const send = (method, params = {}, sessionId) => new Promise((ok, bad) => {
        const n = ++id; pending.set(n, { ok, bad }); ws.send(JSON.stringify({ id: n, method, params, sessionId }));
      });
      return { send, on: (f) => listeners.push(f), close: () => ws.close() };
    } catch { await sleep(200); }
  }
  throw new Error("Chrome's DevTools endpoint never came up");
}

const profile = mkdtempSync(join(tmpdir(), "qsh-csp-"));
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--window-size=1280,900", "--no-first-run", ...(process.env.CI ? ["--no-sandbox"] : []),
  // CSP_OFFLINE=1 (CI, which is offline by design): the page reaches localhost and nothing else, so the published models'
  // fetches fail and every page is still walked under its policy.
  ...(process.env.CSP_OFFLINE ? ["--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost"] : []),
  "about:blank"], { stdio: "ignore" });
let failures = 0;
try {
  const b = await connect();
  const { targetId } = await b.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await b.send("Target.attachToTarget", { targetId, flatten: true });
  const S = (m, p) => b.send(m, p, sessionId);
  let found = [];
  const wouldBlock = new Map();   // Content-Security-Policy-Report-Only findings, by text: what a stricter candidate would block
  const record = (where, text) => {
    if (!POLICY.test(text)) return;
    const line = `${where}: ${text.replace(/\s+/g, " ").slice(0, 300)}`;
    // Read on the whole text, before it is cut to 300 characters: Chrome's console names a Report-Only violation at the
    // end ("The policy is report-only, so the violation has been logged but no further action has been taken."), and the
    // document listener below prefixes its own lines with "[Report Only]".
    // The page's own console repeats what its document listener reports, in longer words, so only the listener's line and
    // a worker's console line (a worker has no document) are listed.
    if (/\[Report Only\]|policy is report-only/i.test(text)) {
      if (where !== "page") wouldBlock.set(line, (wouldBlock.get(line) ?? 0) + 1);
    } else found.push(line);
  };
  const workers = new Set();
  b.on(async (msg) => {
    const p = msg.params ?? {};
    if (msg.method === "Target.attachedToTarget") {
      // Every auto-attached target starts paused (waitForDebuggerOnStart), so each is resumed, listened to or not.
      if (p.targetInfo?.type === "worker") {
        workers.add(p.sessionId);
        for (const m of ["Runtime.enable", "Log.enable"]) await b.send(m, {}, p.sessionId).catch(() => {});
      }
      await b.send("Runtime.runIfWaitingForDebugger", {}, p.sessionId).catch(() => {});
      return;
    }
    const where = msg.sessionId === sessionId ? "page" : workers.has(msg.sessionId) ? "worker" : null;
    if (!where) return;
    if (msg.method === "Runtime.consoleAPICalled") record(where, p.args.map((a) => a.value ?? a.description ?? "").join(" "));
    if (msg.method === "Runtime.exceptionThrown") record(where, p.exceptionDetails.exception?.description ?? p.exceptionDetails.text ?? "");
    if (msg.method === "Log.entryAdded") record(where, `${p.entry.text} ${p.entry.url ?? ""}`);
  });
  await S("Page.enable"); await S("Runtime.enable"); await S("Log.enable");
  await S("Target.setAutoAttach", { autoAttach: true, waitForDebuggerOnStart: true, flatten: true });
  await S("Page.addScriptToEvaluateOnNewDocument", { source: `window.__csp = [];
    document.addEventListener("securitypolicyviolation", (e) => window.__csp.push(
      (e.disposition === "report" ? "[Report Only] " : "") + e.violatedDirective + " blocked " + (e.blockedURI || "inline")
      + " at " + (e.sourceFile || "") + ":" + e.lineNumber));` });
  const ev = async (expression) => {
    const r = await S("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result.value;
  };
  const header = await (await fetch(`${BASE}/`)).headers.get("content-security-policy");
  console.log(`checking ${PAGES.length} pages of ${BASE} under its policy: ${header ? header.slice(0, 120) + "..." : "NONE"}`);
  if (!header) { console.log("  FAIL the site sends no Content-Security-Policy"); failures++; }
  for (const page of PAGES) {
    found = [];
    await S("Page.navigate", { url: BASE + page.path });
    await sleep(page.settle ?? 4000);
    const notes = [];
    for (const label of page.click ?? []) {
      await ev(`(() => { const b = [...document.querySelectorAll("button")].find((x) => x.textContent.trim() === ${JSON.stringify(label)}); if (b) b.click(); })()`);
      await sleep(2500);
    }
    if (page.tabs) {
      const n = await ev(`document.querySelectorAll('[role=tab]').length`);
      for (let i = 0; i < n; i++) { await ev(`document.querySelectorAll('[role=tab]')[${i}].click()`); await sleep(1200); }
      notes.push(`${n} tabs`);
    }
    if (page.run) {
      const started = await ev(`(() => { const b = [...document.querySelectorAll("button")].find((x) => x.textContent.trim() === "▶ Start"); if (!b || b.disabled) return false; b.click(); return true; })()`);
      let done = false;
      for (let t = 0; started && t < 240 && !done; t++) {
        await sleep(1000);
        done = await ev(`(() => { const b = [...document.querySelectorAll("button")].find((x) => x.textContent.trim() === "▶ Start"); return !!b && !b.disabled; })()`);
      }
      notes.push(started ? (done ? "run finished" : "run TIMED OUT") : "run NOT STARTED");
      if (started && !done) failures++;
    }
    if (page.actions) {
      // Each enabled button in <main> that runs or loads something, once (Load leads to Run, so re-query each time).
      const seen = [];
      for (let k = 0; k < 30; k++) {
        const label = await ev(`(() => {
          const ok = /^(Run|Load|Compute|Check|Score|Aggregate|Train|Simulate|Fetch|Verify|Federate)\\b/i, no = /download|reset|stop|pause|cancel|clear/i;
          const seen = ${JSON.stringify(seen)};
          const b = [...document.querySelectorAll("main button")].find((x) => !x.disabled && ok.test(x.textContent.trim()) && !no.test(x.textContent)
            && !seen.includes(x.textContent.trim().slice(0, 80)));
          if (!b) return null; const t = b.textContent.trim().slice(0, 80); b.scrollIntoView(); b.click(); return t; })()`);
        if (!label) break;
        seen.push(label);
        for (let t = 0; t < 120; t++) {
          await sleep(1000);
          if (!(await ev(`[...document.querySelectorAll("main button")].some((x) => x.disabled && /\\.\\.\\.|…/.test(x.textContent))`))) break;
        }
      }
      notes.push(`${seen.length} actions`);
    }
    const state = await ev(`({ csp: window.__csp || [], isolated: window.crossOriginIsolated })`);
    for (const v of state.csp) {
      record("document", v.startsWith("[Report Only] ") ? `[Report Only] Content Security Policy: ${v.slice(14)}` : `Content Security Policy: ${v}`);
    }
    const unique = [...new Set(found)];
    const label = `${page.path}${notes.length ? ` (${notes.join(", ")})` : ""}`;
    if (!state.isolated && page.path !== "/no-such-page") unique.push("the page is not cross-origin isolated");
    if (unique.length) { failures += unique.length; console.log(`  FAIL ${label}`); for (const f of unique) console.log(`       ${f}`); }
    else console.log(`  ok   ${label}`);
  }
  if (wouldBlock.size) {
    console.log(`the Report-Only candidate would block ${wouldBlock.size} distinct thing(s) (not failures):`);
    for (const [line, n] of [...wouldBlock.entries()].sort()) console.log(`       x${n} ${line}`);
  }
  // Closing through DevTools exits Chrome cleanly in a fraction of a second (137 ms here); SIGTERM, below, took from
  // 60 ms to 7.6 s on this machine (2026-11-20), and a profile removed before Chrome has exited is left half-written.
  await Promise.race([b.send("Browser.close").catch(() => {}), sleep(2000)]);
  b.close();
} finally {
  // Wait for Chrome to exit before removing its profile: removing it at once raced Chrome's last writes (ENOTEMPTY),
  // and the check then exited 1 after every page had passed.
  const exited = new Promise((r) => (chrome.exitCode !== null || chrome.signalCode !== null ? r() : chrome.once("exit", r)));
  const within = (ms) => Promise.race([exited.then(() => true), sleep(ms).then(() => false)]);
  if (!(await within(3000))) {
    chrome.kill("SIGTERM");
    if (!(await within(15000))) { chrome.kill("SIGKILL"); await within(5000); }
  }
  // Chrome's helper processes can still be writing to the profile after the main one exits, so a failed removal is a
  // note, never the check's verdict: the profile is a throwaway directory under the system's temporary folder.
  try { rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); }
  catch (e) { console.log(`note: the Chrome profile ${profile} was left behind (${e.code})`); }
}
console.log(failures ? `${failures} CSP problem(s)` : "no CSP violation on any page, worker or run");
process.exit(failures ? 1 : 0);
