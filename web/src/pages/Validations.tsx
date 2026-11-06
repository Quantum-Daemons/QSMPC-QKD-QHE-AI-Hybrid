import { useRef, useState } from "react";
import MpcNetBench from "../components/MpcNetBench";
import NetKeyCards from "../components/NetKeyCards";
import NetworkPositioning from "../components/NetworkPositioning";
import { EveSweepChart } from "../components/QkdCharts";
import { api } from "../lib/api";
import { mldsaDemo, mlkemDemo } from "../lib/crypto";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";
import { qkdCfg } from "../lib/config";
import { downloadJSON } from "../lib/export";

const DEMO = isClientDemo();

/**
 * Runs one validation with its button disabled, and re-enables the button however the run ends.
 *
 * The three buttons that can reach a server (handshake, BB84, agent call) each did
 * `setBusy(x); set(await call()); setBusy("")`, so a call that threw skipped the last statement and
 * left its button disabled until a reload. In the full deployment that happened on every BB84 run
 * from an intercept fraction of 0.45 up, where the endpoint answered 500. The reset now sits in a
 * `finally`, and the error is recorded for that validation's card (null clears it) instead of
 * vanishing as an unhandled rejection.
 */
export async function runValidation(
  which: string, task: () => Promise<void>,
  setBusy: (which: string) => void, setFailure: (which: string, message: string | null) => void,
): Promise<void> {
  setBusy(which);
  setFailure(which, null);
  try {
    await task();
  } catch (e) {
    setFailure(which, String((e as Error)?.message ?? e));
  } finally {
    setBusy("");
  }
}

/** Tickets for overlapping runs of one card: a ticket is current only until a newer one is issued. The Eve slider
 *  starts a run on every step, and in the full deployment each run is a request, so answers can arrive out of order;
 *  without this a slow answer for 0.3 replaced the table under the slider's 0.7 (Round 86, a reader). */
export function newestWins(): () => () => boolean {
  let newest = 0;
  return () => {
    const mine = ++newest;
    return () => mine === newest;
  };
}

/** The failure line inside the card whose run failed. */
function RunFailed({ message }: { message?: string }) {
  if (!message) return null;
  return (
    <p className="note" role="alert" style={{ marginTop: 10, borderColor: "var(--danger)" }}>
      The run failed: {message}
    </p>
  );
}

export default function Validations() {
  const [kem, setKem] = useState<any>(null);
  const [sig, setSig] = useState<any>(null);
  const [pqc, setPqc] = useState<any>(null);
  const [qkd, setQkd] = useState<any>(null);
  const [mcp, setMcp] = useState<any>(null);
  const [mcpQkd, setMcpQkd] = useState(false);
  const [eve, setEve] = useState(0);
  const [busy, setBusy] = useState("");
  // One entry per validation, so running one card does not clear another card's failure.
  const [failures, setFailures] = useState<Record<string, string>>({});
  const setFailure = (which: string, message: string | null) => setFailures((f) => {
    const next = { ...f };
    if (message == null) delete next[which]; else next[which] = message;
    return next;
  });
  const run = (which: string, task: () => Promise<void>) => runValidation(which, task, setBusy, setFailure);

  // A failed run clears the table, which would otherwise show the previous setting's figures under
  // the slider's new value. Only the newest run's answer, or failure, is applied.
  const qkdTicket = useRef(newestWins()).current;
  const runQkd = (e: number) => {
    const current = qkdTicket();
    return run("qkd", async () => {
      try {
        const result = await api.demoQkd(qkdCfg().distance_km, e);
        if (current()) setQkd(result);
      } catch (x) {
        if (!current()) return;
        setQkd(null);
        throw x;
      }
    });
  };

  return (
    <div>
      <h1 className="page-title">Validations</h1>
      <p className="page-sub">
        Run the cryptographic primitives live. PQC runs <b>entirely in your browser</b>{" "}
        (pure-JS @noble/post-quantum); the simulated QKD channel and the MCP agent call {DEMO ? "also run in-browser (zero server compute)" : "run server-side"}.
      </p>
      {DEMO && <div className="note" style={{ marginBottom: 12 }}>🖥️ {DEMO_BANNER}</div>}

      <h2>Post-quantum cryptography</h2>
      <div className="grid cols-2">
        <div className="card">
          <h3>ML-KEM-768 + X25519 hybrid key encapsulation</h3>
          {/* The sizes below are the HYBRID's - ML-KEM-768's 1184 B key and 1088 B ciphertext (FIPS 203)
              plus X25519's 32 B each - because `mlkemDemo()` runs the hybrid. The handshake card below
              runs the same function and splits the ciphertext into its two halves. */}
          <p>The session key's hybrid KEM: ML-KEM-768 (FIPS 203) plus X25519. Generates keys, encapsulates,
            decapsulates - no server. The sizes are the hybrid&apos;s (ML-KEM-768 alone: 1184 B public key,
            1088 B ciphertext, plus 32 B each for X25519); the handshake card below runs the same
            computation and splits the ciphertext.</p>
          <button className="btn" onClick={() => setKem(mlkemDemo())}>Run ML-KEM</button>
          {kem && (
            <table style={{ marginTop: 12 }}>
              <tbody>
                <tr><th scope="row">algorithm</th><td>{kem.algorithm}</td></tr>
                <tr><th scope="row">public key</th><td>{kem.publicKeyBytes} B</td></tr>
                <tr><th scope="row">ciphertext</th><td>{kem.cipherTextBytes} B</td></tr>
                <tr><th scope="row">shared secret</th><td className="mono">{kem.sharedSecretPrefix}…</td></tr>
                <tr><th scope="row">agreement</th><td style={{ color: kem.agreement ? "var(--accent-2)" : "var(--danger)" }}>{String(kem.agreement)}</td></tr>
                <tr><th scope="row">elapsed</th><td>{kem.elapsedMs} ms</td></tr>
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
                <tr><th scope="row">algorithm</th><td>{sig.algorithm}</td></tr>
                <tr><th scope="row">public key</th><td>{sig.publicKeyBytes} B</td></tr>
                <tr><th scope="row">signature</th><td>{sig.signatureBytes} B</td></tr>
                <tr><th scope="row">valid signature</th><td style={{ color: sig.validSignature ? "var(--accent-2)" : "var(--danger)" }}>{String(sig.validSignature)}</td></tr>
                <tr><th scope="row">rejects tampered</th><td style={{ color: sig.rejectsTampered ? "var(--accent-2)" : "var(--danger)" }}>{String(sig.rejectsTampered)}</td></tr>
              </tbody>
            </table>
          )}
        </div>
        <div className="card">
          <h3>Hybrid KEM handshake (ML-KEM-768 + X25519)</h3>
          <p>The PQC-first hybrid session key derivation, computed {DEMO ? "in your browser (@noble)" : "by the backend (liboqs)"}.</p>
          <button className="btn" disabled={busy === "pqc"} onClick={() => run("pqc", async () => setPqc(await api.demoPqc()))}>Run handshake</button>
          <RunFailed message={failures.pqc} />
          {pqc && (
            <table style={{ marginTop: 12 }}>
              <tbody>
                <tr><th scope="row">KEM</th><td>{pqc.kem} + {pqc.classical}</td></tr>
                <tr><th scope="row">combiner</th><td className="mono" style={{ fontSize: 11 }}>{pqc.combiner}</td></tr>
                {/* Split out so the total is checkable against its parts rather than taken on trust:
                    an ML-KEM-768-only run would show 1088 B here with no classical contribution. */}
                <tr><th scope="row">ciphertext</th><td>{pqc.ciphertext_bytes} B ({pqc.pqc_ciphertext_bytes} B ML-KEM + {pqc.classical_public_key_bytes} B X25519)</td></tr>
                <tr><th scope="row">shared secret</th><td className="mono">{pqc.shared_secret_prefix}…</td></tr>
                <tr><th scope="row">agreement</th><td style={{ color: pqc.agreement ? "var(--accent-2)" : "var(--danger)" }}>{String(pqc.agreement)}</td></tr>
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* "Quantum simulations" headed the real PQC hybrid handshake and the simulated BB84 channel
          together until Round 58; the handshake now sits with the other post-quantum primitives. */}
      <h2 style={{ marginTop: 28 }}>Simulated quantum channel {DEMO ? "(in your browser)" : "(server-side)"}</h2>
      <div className="grid cols-1">
        <div className="card">
          <h3>BB84 QKD with eavesdropper</h3>
          <p>Slide the intercept-resend fraction; watch the QBER rise and the finite key fall to zero, which
            happens before the QBER reaches the {qkdCfg().pp_threshold} abort threshold that sets the &quot;eavesdropper?&quot; flag.</p>
          <label className="muted" style={{ fontSize: 13 }} htmlFor="eve-range">Eve intercept fraction: {eve.toFixed(2)}</label>
          <input id="eve-range" name="eve" aria-label="Eve intercept fraction"
            type="range" min={0} max={1} step={0.1} value={eve}
            style={{ width: "100%" }}
            onChange={(e) => { const v = parseFloat(e.target.value); setEve(v); runQkd(v); }} />
          <button className="btn ghost" disabled={busy === "qkd"} onClick={() => runQkd(eve)}>Run BB84</button>
          <RunFailed message={failures.qkd} />
          {qkd && (
            <table style={{ marginTop: 12 }}>
              <tbody>
                <tr><th scope="row">QBER</th><td style={{ color: qkd.eavesdropper_likely ? "var(--danger)" : "var(--accent-2)" }}>{qkd.qber}</td></tr>
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
                <tr><th scope="row">secret key rate (per pulse, asymptotic)</th><td>{typeof qkd.skr_per_pulse === "number" ? Number(qkd.skr_per_pulse.toPrecision(4)) : qkd.skr_per_pulse}</td></tr>
                {/* A composable finite key since Round 80, of the block named beside it - not the rate
                    times the pulses, which no finite block distils. */}
                <tr><th scope="row">secure key bits (composable finite key)</th><td>{typeof qkd.secure_key_bits === "number" ? qkd.secure_key_bits.toLocaleString("en-US") : qkd.secure_key_bits}{typeof qkd.block_pulses === "number" && qkd.block_pulses > 0
                  ? <span className="muted small"> over a block of {qkd.block_pulses.toExponential(2)} pulses</span> : null}</td></tr>
                <tr><th scope="row">eavesdropper?</th><td style={{ color: qkd.eavesdropper_likely ? "var(--danger)" : "var(--accent-2)" }}>{String(qkd.eavesdropper_likely)}</td></tr>
              </tbody>
            </table>
          )}
          <EveSweepChart distanceKm={qkdCfg().distance_km} />
        </div>
      </div>

      <h2 style={{ marginTop: 28 }}>Quantum-safe agent comms (Agentic AI / MCP)</h2>
      <div className="card">
        <h3>MCP agent-to-agent tool call</h3>
        <p>
          {/* "optionally QKD-mixed" described a FIXED 32-byte constant (0x71 x 32, the backend's
              b"q"*32) as QKD key material. The QKD models here return counts, never key bytes. */}
          Two endpoints in this tab, each holding its own keys, establish a PQC (ML-KEM-768 + X25519)
          session - optionally mixed with a fixed QKD-domain constant, not a QKD key - and exchange an{" "}
          <b>ML-DSA-signed</b>, AES-256-GCM-sealed MCP <code>tools/call</code> request of revision
          2026-07-28: a fixed <code>query_cohort</code> call, which the other endpoint opens and verifies.
          No model and no tool runs, so no MCP response follows; what is shown is the sealed, signed
          channel.
        </p>
        <div className="btn-row">
          <label className="muted">
            <input type="checkbox" checked={mcpQkd} onChange={(e) => setMcpQkd(e.target.checked)} /> mix fixed QKD-domain constant
          </label>
          <button className="btn" disabled={busy === "mcp"}
            onClick={() => run("mcp", async () => setMcp(await api.demoMcp(mcpQkd)))}>
            Run secure agent call
          </button>
        </div>
        <RunFailed message={failures.mcp} />
        {mcp && (
          <table style={{ marginTop: 12 }}>
            <tbody>
              <tr><th scope="row">session</th><td className="mono">{mcp.session}</td></tr>
              <tr><th scope="row">KEM / QKD-domain constant mixed</th><td>{mcp.kem} / {String(mcp.qkd_mixed)}</td></tr>
              <tr><th scope="row">method / MCP revision</th><td className="mono">{mcp.method} / {mcp.protocol_version}</td></tr>
              <tr><th scope="row">authenticated ({mcp.signature_alg})</th>
                <td style={{ color: mcp.authenticated ? "var(--accent-2)" : "var(--danger)" }}>{String(mcp.authenticated)}</td></tr>
              <tr><th scope="row">tampered message rejected</th>
                <td style={{ color: mcp.tamper_rejected ? "var(--accent-2)" : "var(--danger)" }}>{String(mcp.tamper_rejected)}</td></tr>
            </tbody>
          </table>
        )}
      </div>

      {/* Recorded, not live (Round 64): the secure argmax under the shipped engine and its
          replacement candidate, between two tabs over a shaped link. */}
      <h2 style={{ marginTop: 28 }}>Secure argmax over a network: two garbled-circuit engines (measured)</h2>
      <div className="card">
        <MpcNetBench />
      </div>

      {/* Handshakes of the network layer's kinds, computed in this tab with this page's own keys
          (Round 58). The tunnels themselves run in the companion network demo; nothing here connects
          to it, and the in-tab WireGuard PSK is not either of the PSKs its lanes use. */}
      <h2 style={{ marginTop: 28 }}>Network-layer key schedules (computed in your tab, with its own keys)</h2>
      <div className="card">
        <NetworkPositioning />
      </div>
      <NetKeyCards />

      <ResultsDownload results={{ kem, sig, pqc, qkd, mcp }} />
    </div>
  );
}

/**
 * The page's export, unavailable until something has run - with the reason on the page, as the
 * use-case and E2E pages give theirs (Round 84). It was enabled from the first paint and saved
 * `{kem, sig, pqc, qkd, mcp}` all null. `disabled` takes a button out of the tab order, so the
 * reason is a visible line and not only a tooltip.
 */
export function ResultsDownload({ results }: { results: Record<string, unknown> }) {
  const ran = Object.keys(results).filter((k) => results[k] != null);
  return (
    <>
      {ran.length === 0 && (
        <p className="muted small" style={{ marginTop: 20, marginBottom: 0 }}>
          Run at least one of the validations above first - the download holds their results, and there
          are none yet.
        </p>
      )}
      <div className="btn-row" style={{ marginTop: ran.length ? 20 : 10 }}>
        <button className="btn ghost" disabled={ran.length === 0}
          title={ran.length ? undefined : "Run a validation first"}
          onClick={() => downloadJSON({ ...results, ran }, "qsh-validations.json")}>
          ⬇ Download results (JSON)
        </button>
      </div>
    </>
  );
}
