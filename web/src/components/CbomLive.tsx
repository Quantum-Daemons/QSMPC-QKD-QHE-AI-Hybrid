// The CBOM graph, live (Round 86, the owner's request of 2026-11-23: the graph visualised and verified dynamically by
// the page's own controls, as the E2E simulator's is, before reaching for WebGL or WebGPU).
//
// - "Run the checks" fetches the file again, hashes it, checks it against the schema and reads every asset against
//   NIST IR 8547 on the panel's date, one event at a time: an asset stays dim until it is read, and the combiner's edges
//   appear when its parts are read. The slider replays the run; nothing runs again.
// - "Play the years" moves the panel's date through 1 January of each year from 2026 to 2037, so the table, the counts
//   and the dots change as IR 8547's dates arrive (deprecated after 2030, disallowed after 2035).
// - Clicking an asset, or Enter or Space on it, shows what the CBOM says of it and its standing year by year.
// - "Check this copy" damages a copy of the document in one of five ways and checks it with the in-tab validator, which
//   must refuse each one and say where; the damaged asset is marked on the graph.
// The events, the replay states, the years and the broken copies are pure functions in lib/cbomRun.ts, which
// tests/test_r86_cbom_run.py runs in node.
import { useEffect, useMemo, useRef, useState } from "react";
import CbomGraph, { STANDING_COLOUR } from "./CbomGraph";
import { sha256Hex, type CbomDoc } from "../lib/cbomReport";
import { checkCbom, type SchemaCheck } from "../lib/cbomSchemaCheck";
import { BROKEN_COPIES, cbomRunEvents, graphStateAt, standingsByYear, type CbomEvent } from "../lib/cbomRun";
import { ruleFor, standingOn, type Standing } from "../lib/ir8547";

const FIRST_YEAR = 2026;
const LAST_YEAR = 2037;
const STEP_MS = 110;
const YEAR_MS = 800;

const LEGEND: { s: Standing; label: string }[] = [
  { s: "post-quantum", label: "post-quantum" }, { s: "symmetric", label: "symmetric" }, { s: "quantum-vulnerable", label: "quantum-vulnerable" },
  { s: "deprecated", label: "deprecated" }, { s: "disallowed", label: "disallowed" }, { s: "allowed-composite", label: "hybrid, allowed as a composite" },
  { s: "not-classified", label: "outside IR 8547's tables" },
];

/** "2026 to 2030: Listed as approved; deprecated after 2030. 2031 to 2035: Deprecated. From 2036: Disallowed", from the labels. */
export function yearsInWords(cells: { year: number; label: string }[]): string {
  const runs: { from: number; to: number; label: string }[] = [];
  for (const c of cells) {
    const last = runs[runs.length - 1];
    if (last && last.label === c.label) last.to = c.year; else runs.push({ from: c.year, to: c.year, label: c.label });
  }
  const span = (r: { from: number; to: number }, last: boolean) =>
    last && r.from !== FIRST_YEAR && r.to === LAST_YEAR ? `From ${r.from}` : r.from === r.to ? `${r.from}` : `${r.from} to ${r.to}`;
  return runs.map((r, i) => `${span(r, i === runs.length - 1)}: ${r.label}`).join(". ");
}

export default function CbomLive({ doc, url, date, setDate, builtSha256 }: {
  doc: CbomDoc; url: string; date: string; setDate: (d: string) => void; builtSha256: string;
}) {
  const [events, setEvents] = useState<CbomEvent[] | null>(null);
  const [cursor, setCursor] = useState(0);
  const [auto, setAuto] = useState(false);
  const [runDate, setRunDate] = useState(date);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [playingYears, setPlayingYears] = useState(false);
  const [broken, setBroken] = useState(BROKEN_COPIES[0].id);
  const [brokenResult, setBrokenResult] = useState<{ id: string; result: SchemaCheck } | null>(null);
  const log = useRef<HTMLOListElement | null>(null);
  const reduced = useMemo(() => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches, []);

  // The run, one event at a time, while it plays; a reader moving the slider takes over.
  useEffect(() => {
    if (!events || !auto) return;
    if (cursor >= events.length) { setAuto(false); return; }
    const t = setTimeout(() => setCursor((c) => c + 1), reduced ? 0 : STEP_MS);
    return () => clearTimeout(t);
  }, [events, cursor, auto, reduced]);
  useEffect(() => { if (log.current) log.current.scrollTop = log.current.scrollHeight; }, [cursor]);
  // The years, one a step, from 2026 to 2037.
  useEffect(() => {
    if (!playingYears) return;
    const year = Number(date.slice(0, 4));
    if (!(year < LAST_YEAR)) { setPlayingYears(false); return; }
    const t = setTimeout(() => setDate(`${year + 1}-01-01`), YEAR_MS);
    return () => clearTimeout(t);
  }, [playingYears, date, setDate]);

  const run = async () => {
    setBusy(true);
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = await res.arrayBuffer();
      const fresh = JSON.parse(new TextDecoder().decode(buf)) as CbomDoc;
      const [sha, schema] = await Promise.all([sha256Hex(buf), checkCbom(fresh)]);
      setEvents(cbomRunEvents({ doc: fresh, bytes: buf.byteLength, sha256: sha, builtSha256, schema, isoDate: date }));
    } catch (e) {
      setEvents([{ kind: "fetch", ok: false, text: `The file could not be fetched (${(e as Error).message})` }]);
    } finally {
      setRunDate(date); setCursor(0); setAuto(true); setBusy(false);
    }
  };
  const playYears = () => {
    if (playingYears) { setPlayingYears(false); return; }
    setDate(`${FIRST_YEAR}-01-01`);
    setPlayingYears(true);
  };
  const checkBroken = async () => {
    const b = BROKEN_COPIES.find((x) => x.id === broken)!;
    setBrokenResult({ id: b.id, result: await checkCbom(b.apply(doc)) });
  };

  const replaying = events !== null && cursor < events.length;
  const state = replaying ? graphStateAt(events, cursor, runDate) : null;
  const brokenCopy = brokenResult ? BROKEN_COPIES.find((x) => x.id === brokenResult.id) ?? null : null;
  const sel = selected ? doc.components.find((c) => c["bom-ref"] === selected) ?? null : null;
  const selRule = sel ? ruleFor(sel["bom-ref"]) : undefined;
  const years = sel ? standingsByYear([sel["bom-ref"]], FIRST_YEAR, LAST_YEAR).map((y) => ({
    year: y.year, standing: y.standings[sel["bom-ref"]], label: selRule ? standingOn(selRule, `${y.year}-07-01`).label : "No rule" })) : [];
  const done = events !== null && cursor >= events.length;
  return (
    <div>
      <div className="btn-row" style={{ alignItems: "center", flexWrap: "wrap" }}>
        <button className="btn" onClick={run} disabled={busy}>{busy ? "Fetching…" : events ? "Run the checks again" : "Run the checks"}</button>
        {events && <button className="btn ghost" onClick={() => setAuto((a) => !a)} disabled={cursor >= events.length}>{auto ? "Pause" : "Resume"}</button>}
        {events && (
          <input type="range" min={0} max={events.length} step={1} value={cursor} style={{ flex: "1 1 160px", minWidth: 120 }}
            onChange={(e) => { setAuto(false); setCursor(Number(e.target.value)); }}
            aria-label="Position in the run of the checks" aria-valuetext={`event ${cursor} of ${events.length}`} />
        )}
        {events && <span className="small muted">event {cursor} of {events.length}</span>}
      </div>
      <div className="btn-row" style={{ alignItems: "center", flexWrap: "wrap", marginTop: 8 }}>
        <button className="btn ghost" onClick={playYears}>
          {playingYears ? "Stop the years" : `Play the years, ${FIRST_YEAR} to ${LAST_YEAR}`}
        </button>
        <span className="small">Standing on <strong>{date}</strong>{replaying ? ` (the run reads on ${runDate})` : ""}</span>
      </div>
      <div style={{ marginTop: 10 }}>
        <CbomGraph doc={doc} isoDate={date} run={state} selected={selected} onSelect={(ref) => setSelected((s) => (s === ref ? null : ref))}
          marked={brokenCopy?.ref ?? null} />
      </div>
      <p className="small" style={{ marginTop: 6 }}>
        {LEGEND.map((l) => (
          <span key={l.s} style={{ marginRight: 14, whiteSpace: "nowrap" }}>
            <span aria-hidden="true" style={{ display: "inline-block", width: 9, height: 9, borderRadius: "50%", background: STANDING_COLOUR[l.s], marginRight: 5 }} />{l.label}
          </span>
        ))}
      </p>
      {events && (
        <ol className="cbom-log" ref={log} aria-label="The run's log" style={{ overflowWrap: "anywhere" }}>
          {events.slice(0, cursor).map((e, i) => (
            <li key={i} className={i === cursor - 1 ? "latest" : undefined}>
              <span className={`mark ${e.ok === true ? "ok" : e.ok === false ? "bad" : ""}`} aria-hidden="true">{e.ok === true ? "✓" : e.ok === false ? "✗" : "·"}</span>
              {e.ok === false && <span className="sr-only">Failed: </span>}{e.text}
            </li>
          ))}
        </ol>
      )}
      <p className="small" aria-live="polite" style={{ minHeight: "1.5em" }}>{done ? events![events!.length - 1].text : ""}</p>
      {sel && (
        <div className="card" style={{ background: "var(--bg-2)", marginTop: 8 }}>
          <h4 style={{ margin: "0 0 4px" }}>{sel.name}</h4>
          <p style={{ margin: "4px 0" }}>{selRule ? <>{selRule.basis !== "none" && <><span className="tag">{selRule.basis}</span>{" "}</>}{selRule.reading}</> : "This report has no rule for this asset."}</p>
          <p className="small" style={{ margin: "4px 0" }}>Year by year. {yearsInWords(years)}.</p>
          <div className="cbom-years" aria-hidden="true">
            {years.map((y) => <span key={y.year} title={`${y.year}: ${y.label}`} style={{ borderTop: `4px solid ${STANDING_COLOUR[y.standing]}` }}>{String(y.year).slice(2)}</span>)}
          </div>
          <p className="small muted" style={{ margin: "6px 0 0" }}>
            Where the CBOM says the code calls it: {sel.evidence?.occurrences?.map((o) => o.location).join(", ") || "not stated"}.
          </p>
        </div>
      )}
      <div className="btn-row" style={{ alignItems: "center", flexWrap: "wrap", marginTop: 10 }}>
        <label className="small" htmlFor="cbom-broken">Try a broken copy</label>
        <select id="cbom-broken" value={broken} onChange={(e) => { setBroken(e.target.value); setBrokenResult(null); }} style={{ maxWidth: "100%", flex: "1 1 220px" }}>
          {BROKEN_COPIES.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
        </select>
        <button className="btn ghost" onClick={checkBroken}>Check this copy</button>
      </div>
      {brokenResult && (
        <p className="small" aria-live="polite">
          {brokenResult.result.valid
            ? <><strong>Accepted</strong>, which it should not have been: the in-tab check missed this damage.</>
            : <><strong>Refused</strong>, as it should be: <span className="mono" style={{ overflowWrap: "anywhere" }}>{brokenResult.result.error}</span>.{brokenCopy?.ref ? " The damaged asset is marked on the graph." : ""}</>}
        </p>
      )}
    </div>
  );
}
