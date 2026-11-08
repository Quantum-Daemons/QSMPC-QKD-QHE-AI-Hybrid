// The cryptography this repository's code calls, from the CycloneDX document the repository generates (Round 86).
import { useEffect, useState } from "react";

type Component = {
  "bom-ref": string;
  name: string;
  description: string;
  properties: { name: string; value: string }[];
  cryptoProperties: { algorithmProperties: { primitive: string; nistQuantumSecurityLevel?: number; parameterSetIdentifier?: string } };
};

const URL_ = `${import.meta.env.BASE_URL}cbom/qsh.cdx.json`;

/** What a quantum computer does to it, from the document's own field, in words. */
function quantum(level: number | undefined): string {
  if (level === undefined) return "not rated here";
  return level === 0 ? "broken by a large quantum computer" : `NIST category ${level}`;
}

export default function CbomPanel() {
  const [components, setComponents] = useState<Component[] | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    fetch(URL_)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d: { components: Component[] }) => setComponents(d.components))
      .catch((e: Error) => setFailed(e.message));
  }, []);
  if (failed) return <p className="muted small">The inventory could not be loaded ({failed}).</p>;
  if (!components) return <p className="muted small">Loading the inventory…</p>;
  const prop = (c: Component, n: string) => c.properties.find((p) => p.name === n)?.value ?? "-";
  const exposed = components.filter((c) => c.cryptoProperties.algorithmProperties.nistQuantumSecurityLevel === 0);
  return (
    <div className="card" id="cbom">
      <p style={{ marginTop: 0 }}>
        {components.length} cryptographic algorithms, from this repository&apos;s own list of what its code calls, as a CycloneDX 1.7 document. Each entry is checked against the source on every
        build, and a new cryptographic import with no entry fails it. This is not the output of a scanner, it does not list what the libraries do inside, and nothing here is FIPS 140-3 validated.
        {" "}{exposed.length} of them would not survive a large quantum computer ({exposed.map((c) => c.name).join(", ")}).
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table">
          <thead><tr><th scope="col">Algorithm</th><th scope="col">Kind</th><th scope="col">Quantum</th><th scope="col">Standard</th><th scope="col">Used for</th><th scope="col">Where</th></tr></thead>
          <tbody>
            {components.map((c) => (
              <tr key={c["bom-ref"]}>
                <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{c.name}</th>
                <td className="mono">{c.cryptoProperties.algorithmProperties.primitive}</td>
                <td>{quantum(c.cryptoProperties.algorithmProperties.nistQuantumSecurityLevel)}</td>
                <td>{prop(c, "qsh:standard")}</td>
                <td>{c.description}</td>
                <td>{prop(c, "qsh:scope")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ marginBottom: 0 }}><a href={URL_} download="qsh.cdx.json">⬇ The document (CycloneDX JSON)</a></p>
    </div>
  );
}
