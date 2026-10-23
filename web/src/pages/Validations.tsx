import { useState } from "react";
import { api } from "../lib/api";
import { mldsaDemo, mlkemDemo } from "../lib/crypto";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";
import { qkdCfg } from "../lib/config";
import { downloadJSON } from "../lib/export";

const DEMO = isClientDemo();

export default function Validations() {
  const [kem, setKem] = useState<any>(null);
  const [sig, setSig] = useState<any>(null);
  const [pqc, setPqc] = useState<any>(null);
  const [qkd, setQkd] = useState<any>(null);
  const [mcp, setMcp] = useState<any>(null);
  const [mcpQkd, setMcpQkd] = useState(false);
  const [eve, setEve] = useState(0);
  const [busy, setBusy] = useState("");

  const runQkd = async (e: number) => {
    setBusy("qkd");
    setQkd(await api.demoQkd(qkdCfg().distance_km, e));
    setBusy("");
  };

  return (
    <div>
      <h1 className="page-title">Validations</h1>
      <p className="page-sub">
        Run the cryptographic primitives live. PQC runs <b>entirely in your browser</b>{" "}
        (pure-JS @noble/post-quantum); QKD + MCP {DEMO ? "also run in-browser (zero server compute)" : "run as server-side simulations"}.
      </p>
      {DEMO && <div className="note" style={{ marginBottom: 12 }}>🖥️ {DEMO_BANNER}</div>}

      <h2>Client-side post-quantum crypto (in-browser)</h2>
      <div className="grid cols-2">
        <div className="card">
          <h3>ML-KEM-768 key encapsulation</h3>
          <p>FIPS 203 hybrid KEM half. Generates keys, encapsulates, decapsulates — no server.</p>
          <button className="btn" onClick={() => setKem(mlkemDemo())}>Run ML-KEM</button>
          {kem && (
            <table style={{ marginTop: 12 }}>
              <tbody>
                <tr><td>algorithm</td><td>{kem.algorithm}</td></tr>
                <tr><td>public key</td><td>{kem.publicKeyBytes} B</td></tr>
                <tr><td>ciphertext</td><td>{kem.cipherTextBytes} B</td></tr>
                <tr><td>shared secret</td><td className="mono">{kem.sharedSecretPrefix}…</td></tr>
                <tr><td>agreement</td><td style={{ color: kem.agreement ? "var(--accent-2)" : "var(--danger)" }}>{String(kem.agreement)}</td></tr>
                <tr><td>elapsed</td><td>{kem.elapsedMs} ms</td></tr>
              </tbody>
            </table>
          )}
        </div>
        <div className="card">
          <h3>ML-DSA-65 signature</h3>
          <p>FIPS 204 signature: sign a message, verify it, and confirm a tampered message is rejected.</p>
          <button className="btn" onClick={() => setSig(mldsaDemo())}>Run ML-DSA</button>
          {sig && (
            <table style={{ marginTop: 12 }}>
              <tbody>
                <tr><td>algorithm</td><td>{sig.algorithm}</td></tr>
                <tr><td>public key</td><td>{sig.publicKeyBytes} B</td></tr>
                <tr><td>signature</td><td>{sig.signatureBytes} B</td></tr>
                <tr><td>valid signature</td><td style={{ color: sig.validSignature ? "var(--accent-2)" : "var(--danger)" }}>{String(sig.validSignature)}</td></tr>
                <tr><td>rejects tampered</td><td style={{ color: sig.rejectsTampered ? "var(--accent-2)" : "var(--danger)" }}>{String(sig.rejectsTampered)}</td></tr>
              </tbody>
            </table>
          )}
        </div>
      </div>

      <h2 style={{ marginTop: 28 }}>{DEMO ? "Quantum simulations (in-browser)" : "Server-side quantum simulations"}</h2>
      <div className="grid cols-2">
        <div className="card">
          <h3>Hybrid KEM handshake (ML-KEM-768 + X25519)</h3>
          <p>The PQC-first hybrid session key derivation, computed {DEMO ? "in your browser (@noble)" : "by the backend (liboqs)"}.</p>
          <button className="btn" disabled={busy === "pqc"} onClick={async () => { setBusy("pqc"); setPqc(await api.demoPqc()); setBusy(""); }}>Run handshake</button>
          {pqc && (
            <table style={{ marginTop: 12 }}>
              <tbody>
                <tr><td>KEM</td><td>{pqc.kem} + {pqc.classical}</td></tr>
                <tr><td>combiner</td><td className="mono" style={{ fontSize: 11 }}>{pqc.combiner}</td></tr>
                {/* Split out so the total is checkable against its parts rather than taken on trust:
                    an ML-KEM-768-only run would show 1088 B here with no classical contribution. */}
                <tr><td>ciphertext</td><td>{pqc.ciphertext_bytes} B ({pqc.pqc_ciphertext_bytes} B ML-KEM + {pqc.classical_public_key_bytes} B X25519)</td></tr>
                <tr><td>shared secret</td><td className="mono">{pqc.shared_secret_prefix}…</td></tr>
                <tr><td>agreement</td><td style={{ color: pqc.agreement ? "var(--accent-2)" : "var(--danger)" }}>{String(pqc.agreement)}</td></tr>
              </tbody>
            </table>
          )}
        </div>
        <div className="card">
          <h3>BB84 QKD with eavesdropper</h3>
          <p>Slide the intercept-resend fraction; watch QBER rise above the security threshold.</p>
          <label className="muted" style={{ fontSize: 13 }} htmlFor="eve-range">Eve intercept fraction: {eve.toFixed(2)}</label>
          <input id="eve-range" name="eve" aria-label="Eve intercept fraction"
            type="range" min={0} max={1} step={0.1} value={eve}
            style={{ width: "100%" }}
            onChange={(e) => { const v = parseFloat(e.target.value); setEve(v); runQkd(v); }} />
          <button className="btn ghost" disabled={busy === "qkd"} onClick={() => runQkd(eve)}>Run BB84</button>
          {qkd && (
            <table style={{ marginTop: 12 }}>
              <tbody>
                <tr><td>QBER</td><td style={{ color: qkd.eavesdropper_likely ? "var(--danger)" : "var(--accent-2)" }}>{qkd.qber}</td></tr>
                {/* PER PULSE, and the label says so. It read "secret key rate" with no unit beside a
                    dimensionless per-pulse fraction, while ETSI GS QKD 015 defines a field of that
                    name as bit/s - the same one-name-two-dimensions problem Round 44 removed from
                    the control-plane panel, on a page nobody was looking at.

                    ROUND 45 FIXED THE OTHER HALF, AND THIS COMMENT WAS THE EVIDENCE OF IT. It used
                    to end "the payload key is still `skr`; renaming it crosses into
                    `api/inference_service.py` and is conditioned rather than done here" - a
                    sentence Round 44 falsified in the same commit that broke this page. It DID
                    rename `inference_service.py`, and this reader, and left the CLIENT branch of
                    `api.ts` emitting `skr`. The demo builds CLIENT_ONLY, so that branch is the only
                    one the public site executes and this cell rendered blank there. The sentence
                    remained true only of the branch nobody had looked at. */}
                <tr><td>secret key rate (per pulse)</td><td>{qkd.skr_per_pulse}</td></tr>
                <tr><td>secure key bits</td><td>{qkd.secure_key_bits}</td></tr>
                <tr><td>eavesdropper?</td><td style={{ color: qkd.eavesdropper_likely ? "var(--danger)" : "var(--accent-2)" }}>{String(qkd.eavesdropper_likely)}</td></tr>
              </tbody>
            </table>
          )}
        </div>
      </div>

      <h2 style={{ marginTop: 28 }}>Quantum-safe agent comms (Agentic AI / MCP)</h2>
      <div className="card">
        <h3>MCP-style agent-to-agent tool call</h3>
        <p>
          {/* "optionally QKD-mixed" described a FIXED 32-byte constant (0x71 x 32, the backend's
              b"q"*32) as QKD key material. The QKD models here return counts, never key bytes. */}
          Two AI agents establish a PQC (ML-KEM-768 + X25519) session - optionally mixed with a fixed
          QKD-domain constant, not a QKD key - and
          exchange an <b>ML-DSA-signed</b>, AES-256-GCM-sealed MCP tool call — closing the report's gap
          of quantum-resistant + private agent↔data-silo communication (no trusted third party).
        </p>
        <div className="btn-row">
          <label className="muted">
            <input type="checkbox" checked={mcpQkd} onChange={(e) => setMcpQkd(e.target.checked)} /> mix fixed QKD-domain constant
          </label>
          <button className="btn" disabled={busy === "mcp"}
            onClick={async () => { setBusy("mcp"); setMcp(await api.demoMcp(mcpQkd)); setBusy(""); }}>
            Run secure agent call
          </button>
        </div>
        {mcp && (
          <table style={{ marginTop: 12 }}>
            <tbody>
              <tr><td>session</td><td className="mono">{mcp.session}</td></tr>
              <tr><td>KEM / QKD-domain constant mixed</td><td>{mcp.kem} / {String(mcp.qkd_mixed)}</td></tr>
              <tr><td>method</td><td className="mono">{mcp.method}</td></tr>
              <tr><td>authenticated ({mcp.signature_alg})</td>
                <td style={{ color: mcp.authenticated ? "var(--accent-2)" : "var(--danger)" }}>{String(mcp.authenticated)}</td></tr>
              <tr><td>tampered message rejected</td>
                <td style={{ color: mcp.tamper_rejected ? "var(--accent-2)" : "var(--danger)" }}>{String(mcp.tamper_rejected)}</td></tr>
            </tbody>
          </table>
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 20 }}>
        <button className="btn ghost" onClick={() => downloadJSON({ kem, sig, pqc, qkd, mcp }, "qsh-validations.json")}>
          ⬇ Download results (JSON)
        </button>
      </div>
    </div>
  );
}
