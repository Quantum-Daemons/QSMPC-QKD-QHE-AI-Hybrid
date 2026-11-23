// The cryptography this repository's code calls, from the CycloneDX document the repository generates (Round 86),
// read against NIST IR 8547 on a date the reader picks, with the served file's digest and version as its provenance,
// the document's dependency graph, and the report as Markdown or CSV. The owner's choice of 2026-11-23.
import { useEffect, useMemo, useState } from "react";
import CbomLive from "./CbomLive";
import { downloadCSV, downloadText } from "../lib/export";
import { IR8547, localIsoDate, type Standing } from "../lib/ir8547";
import { reportCounts, reportCsvRows, reportMarkdown, reportRows, sha256Hex, type CbomDoc } from "../lib/cbomReport";
import { checkCbom, SCHEMA_CHECK, type SchemaCheck } from "../lib/cbomSchemaCheck";

const URL_ = `${import.meta.env.BASE_URL}cbom/qsh.cdx.json`;
const FILE = "qsh.cdx.json";

const STANDING_CLASS: Record<Standing | "none", string> = {
  "post-quantum": "pq", symmetric: "sym", "approved-quantum-vulnerable": "qv", "quantum-vulnerable": "qv", deprecated: "dep", disallowed: "dis",
  "allowed-composite": "hyb", "not-classified": "out", none: "out",
};

/** IR 8547's transitions, as days a reader can jump to: the last day before each, and the first day of each. */
const DAYS: { day: string; label: string }[] = [
  { day: "2030-12-31", label: "2030-12-31, the last day before deprecation" },
  { day: "2031-01-01", label: "2031-01-01, deprecated" },
  { day: "2036-01-01", label: "2036-01-01, disallowed" },
];

export default function CbomPanel() {
  const [doc, setDoc] = useState<CbomDoc | null>(null);
  const [sha, setSha] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [schema, setSchema] = useState<SchemaCheck | null>(null);
  const [date, setDate] = useState(() => localIsoDate());
  useEffect(() => {
    let live = true;
    fetch(URL_)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); })
      .then(async (buf) => {
        const parsed = JSON.parse(new TextDecoder().decode(buf)) as CbomDoc;
        const digest = await sha256Hex(buf);
        if (live) { setDoc(parsed); setSha(digest); }
        const checked = await checkCbom(parsed).catch((e: Error) => ({ valid: false, error: `the check did not run (${e.message})` }));
        if (live) setSchema(checked);
      })
      .catch((e: Error) => { if (live) setFailed(e.message); });
    return () => { live = false; };
  }, []);
  const day = /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : localIsoDate();
  const rows = useMemo(() => (doc ? reportRows(doc, day) : []), [doc, day]);
  if (failed) return <p className="muted small">The inventory could not be loaded ({failed}).</p>;
  if (!doc || !sha) return <p className="muted small">Loading the inventory…</p>;
  const components = doc.components;
  const exposed = components.filter((c) => c.cryptoProperties?.algorithmProperties?.nistQuantumSecurityLevel === 0);
  const app = doc.metadata?.component;
  const built = __CBOM_SHA256__;
  const counts = reportCounts(rows);
  const namesWith = (s: Standing | "none") => rows.filter((r) => !r.partOf && r.standing === s).map((r) => r.name).join(", ");
  const provenance = { file: FILE, sha256: sha, builtSha256: built };
  return (
    <div className="card" id="cbom">
      <p style={{ marginTop: 0 }}>
        {components.length} cryptographic algorithms, from this repository&apos;s own list of what its code calls, as a CycloneDX 1.7 document. Each entry is checked against the source on every
        build, and a new cryptographic import with no entry fails it. This is not the output of a scanner, it does not list what the libraries do inside, and nothing here is FIPS 140-3 validated.
        {" "}{exposed.length} of them would not survive a large quantum computer ({exposed.map((c) => c.name).join(", ")}).
      </p>
      <p className="small">
        The file read: {FILE}, {doc.bomFormat} {doc.specVersion}, serial <span className="mono" style={{ overflowWrap: "anywhere" }}>{doc.serialNumber}</span>, version {doc.version}, describing {app?.name} {app?.version}.
        {" "}Its SHA-256, computed in this tab from the bytes fetched: <span className="mono" style={{ wordBreak: "break-all" }}>{sha}</span>
        {built === sha ? ", the file this build was made from." : <>, which is <strong>not</strong> the file this build was made from (<span className="mono" style={{ wordBreak: "break-all" }}>{built}</span>).</>}
        {" "}Checked in this tab against {SCHEMA_CHECK}:{" "}
        {schema === null ? "checking…" : schema.valid ? <strong>valid</strong> : <><strong>not valid</strong> ({schema.error})</>}.
      </p>

      <h3 style={{ marginTop: 16 }}>Read against NIST IR 8547, on a date</h3>
      <p>
        Each asset with the table or section of <a href={IR8547.url} target="_blank" rel="noreferrer">NIST IR 8547</a> (final, October 2026) it rests on, read on {IR8547.read}, and
        what IR 8547 says of it on the date below. This is a classification by IR 8547&apos;s tables, not a certification. An asset IR 8547 does not name is outside its tables, with what it is.
      </p>
      <div className="btn-row" style={{ alignItems: "center", flexWrap: "wrap" }}>
        <label className="small" htmlFor="cbom-date">Standing on</label>
        <input id="cbom-date" type="date" value={date} min="2024-01-01" max="2099-12-31" onChange={(e) => setDate(e.target.value)} />
        <button className="btn ghost" onClick={() => setDate(localIsoDate())}>Today</button>
        {DAYS.map((d) => <button key={d.day} className="btn ghost" onClick={() => setDate(d.day)}>{d.label}</button>)}
      </div>
      <ul className="small" style={{ marginTop: 10, listStyle: "none", paddingLeft: 0 }}>
        {counts.map((c) => (
          <li key={c.standing}><span className={`standing ${STANDING_CLASS[c.standing]}`}>{c.label}</span>: {c.count} ({namesWith(c.standing)})</li>
        ))}
      </ul>
      <div style={{ overflowX: "auto" }}>
        <table className="table">
          <thead><tr><th scope="col">Algorithm</th><th scope="col" style={{ minWidth: 230 }}>NIST IR 8547</th><th scope="col" style={{ minWidth: 140 }}>On {day}</th><th scope="col">Standard</th><th scope="col" style={{ minWidth: 150 }}>Used for</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.ref}>
                <th scope="row" style={{ textAlign: "left", fontWeight: 500, paddingLeft: r.partOf ? 22 : undefined }}>
                  {r.partOf ? `↳ ${r.name}` : r.name}
                  <div className="mono small muted">{r.primitive}</div>
                </th>
                <td>{r.basis !== "none" && <><span className="tag">{r.basis}</span>{" "}</>}{r.reading}</td>
                <td><span className={`standing ${STANDING_CLASS[r.standing]}`}>{r.standingLabel}</span></td>
                <td style={{ overflowWrap: "anywhere" }}>{r.standard}</td>
                <td>{r.usedFor}<div className="muted small">{r.scope}</div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 style={{ marginTop: 16 }}>What depends on what, checked in the tab</h3>
      <p>
        Drawn from the document&apos;s own dependencies: the application uses every asset, and the hybrid combiner depends on the assets it combines. What a library uses inside
        (ML-KEM&apos;s SHA-3, for one) is not listed, so an asset with no edge of its own has unknown dependencies, not none. Run the checks to watch the file fetched, hashed,
        checked against the schema and read asset by asset, then replay the run; play the years to watch IR 8547&apos;s dates arrive; pick an asset for its details; or check a
        broken copy, which the schema check must refuse.
      </p>
      <CbomLive doc={doc} url={URL_} date={day} setDate={setDate} builtSha256={built} />

      <div className="btn-row" style={{ marginTop: 12, flexWrap: "wrap" }}>
        <button className="btn ghost" onClick={() => downloadText(reportMarkdown(doc, day, provenance), `qsh-pqc-readiness-${day}.md`, "text/markdown")}>⬇ This report (Markdown)</button>
        <button className="btn ghost" onClick={() => downloadCSV(reportCsvRows(doc, day), `qsh-pqc-readiness-${day}.csv`)}>⬇ This report (CSV)</button>
      </div>
      <p className="muted small" style={{ marginBottom: 0 }}><a href={URL_} download={FILE}>⬇ The document (CycloneDX JSON)</a></p>
    </div>
  );
}
