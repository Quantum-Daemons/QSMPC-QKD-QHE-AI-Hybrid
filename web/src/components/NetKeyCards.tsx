import { x25519 } from "@noble/curves/ed25519.js";
import { ml_kem768 } from "@noble/post-quantum/ml-kem.js";
import { useState } from "react";

import { hybridCombine } from "../lib/hybridCombiner";
import Details from "./Details";
import StageBadge from "./StageBadge";

/**
 * Three network-layer computations, run in the visitor's tab and held to official vectors where any
 * exist (`web/src/lib/netkeys/`, `references/net-key-schedules.md`).
 *
 * WHAT EACH ONE IS. The ML-KEM known-answer test checks this page's PQC library against NIST's own
 * ACVP cases. The WireGuard handshake is the Noise_IKpsk2 core (checked against the official cacophony
 * vector on every run) framed per the whitepaper, between two peers in this tab, keyed with a PSK taken
 * from an in-tab ML-KEM-768 + X25519 hybrid through the SAME combiner as the session key. The IKEv2
 * schedule is RFC 7296 key derivation (checked against NIST CAVP) extended by RFC 9370 and RFC 8784,
 * which have no official vectors and are labelled computed-per-spec. No packet leaves the tab; the
 * tunnels themselves run in the companion network demo, keyed differently (its WireGuard PSKs are
 * HKDF of QKD and PQC-HPKE, and Rosenpass's), so these are schedules of the same kinds, not its keys.
 *
 * The modules load on click: the ML-KEM vector file alone is about 69 KB.
 */

/** 32 bytes from a fresh in-tab hybrid, through the session key's own combiner. Never displayed. */
function hybridPsk(): Uint8Array {
  const bob = ml_kem768.keygen();
  const { cipherText, sharedSecret } = ml_kem768.encapsulate(bob.publicKey);
  const bobX = x25519.utils.randomSecretKey();
  const aliceX = x25519.utils.randomSecretKey();
  const aliceXPub = x25519.getPublicKey(aliceX);
  const xs = x25519.getSharedSecret(aliceX, x25519.getPublicKey(bobX));
  return hybridCombine(sharedSecret, xs,
    { c1: cipherText, c2: aliceXPub, ek1: bob.publicKey, ek2: x25519.getPublicKey(bobX) }, "ML-KEM-768", 32);
}

const ok = (b: boolean | null | undefined) => (
  <span style={{ color: b ? "var(--accent-2)" : b === false ? "var(--danger)" : "var(--muted)" }}>{b == null ? "n/a" : String(b)}</span>
);

export default function NetKeyCards() {
  const [busy, setBusy] = useState("");
  const [kat, setKat] = useState<any>(null);
  const [wg, setWg] = useState<any>(null);
  const [ike, setIke] = useState<any>(null);
  const [err, setErr] = useState("");
  const run = async (which: string, f: () => Promise<void>) => {
    setBusy(which); setErr("");
    try { await f(); } catch (e) { setErr(`${which}: ${String((e as Error)?.message ?? e)}`); }
    setBusy("");
  };

  return (
    <div className="grid cols-1" style={{ marginTop: 14 }}>
      {err && <div className="note" style={{ borderColor: "var(--danger)" }}>{err}</div>}

      <div className="card">
        <h3>ML-KEM-768 known-answer test (NIST ACVP) <StageBadge kind="real" /></h3>
        <p>Runs NIST&apos;s own ACVP cases - key generation, encapsulation, and decapsulation of valid and of
          modified ciphertexts - through the ML-KEM library this page uses for every handshake.</p>
        <button className="btn" disabled={busy === "kat"}
          onClick={() => run("kat", async () => setKat((await import("../lib/netkeys/mlkemKat")).runMlKemAcvp()))}>
          Run the ACVP cases
        </button>
        {kat && (
          <table className="facts-table small" style={{ marginTop: 10 }}>
            <tbody>
              <tr><td>result</td><td>{ok(kat.allPassed)} - {kat.passed}/{kat.total} cases</td></tr>
              {Object.entries(kat.byFunction as Record<string, { passed: number; total: number }>).map(([fn, r]) => (
                <tr key={fn}><td>{fn}</td><td>{r.passed}/{r.total}</td></tr>
              ))}
              <tr><td>implementation</td><td className="mono">{kat.implementation}</td></tr>
              <tr><td>vectors</td><td>usnistgov/ACVP-Server at <span className="mono">{String(kat.source.commit).slice(0, 7)}</span>, copied verbatim with SHA-256</td></tr>
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h3>WireGuard handshake keyed by a PQC hybrid PSK (Noise_IKpsk2) <StageBadge kind="notional" /></h3>
        <KeyPointsInline items={[
          "Both peers run in this tab; the cryptography is real, the network is not.",
          "The 32-byte preshared key comes from a fresh ML-KEM-768 + X25519 hybrid through the session key's own combiner - not either PSK the network demo's WireGuard lanes use (HKDF of QKD and PQC-HPKE; Rosenpass).",
          "The Noise core is checked against the official cacophony vector on every run; the WireGuard framing follows the whitepaper and has no published vector.",
        ]} />
        <button className="btn" disabled={busy === "wg"}
          onClick={() => run("wg", async () => setWg((await import("../lib/netkeys/wireguard")).wireguardHandshake({ psk: hybridPsk(), seed: "qsh-validations" })))}>
          Run the handshake
        </button>
        {wg && (
          <table className="facts-table small" style={{ marginTop: 10 }}>
            <tbody>
              <tr><td>Noise core vector</td><td>{ok(wg.noiseCore.passed)} ({wg.noiseCore.protocolName})</td></tr>
              <tr><td>message sizes</td><td>{wg.messageSizes.initiation} B + {wg.messageSizes.response} B - whitepaper {wg.whitepaperSizes.initiation} + {wg.whitepaperSizes.response}: {ok(wg.sizesMatchWhitepaper)}</td></tr>
              <tr><td>mac1 valid (both messages)</td><td>{ok(wg.mac1Valid.initiation && wg.mac1Valid.response)}</td></tr>
              <tr><td>transport keys agree</td><td>{ok(wg.transportKeysAgree)} - handshake hashes agree {ok(wg.handshakeHashesAgree)}</td></tr>
              <tr><td>PSK changed by one bit</td><td>sizes unchanged {ok(wg.pskChange.sizesUnchanged)}; initiation unchanged {ok(wg.pskChange.initiationUnchanged)} (psk2 mixes it into message 2); transport keys changed {ok(wg.pskChange.transportKeysChanged)}</td></tr>
              <tr><td>mismatched PSK</td><td>rejected {ok(wg.pskMismatch?.rejected)}</td></tr>
              <tr><td>initial-state cross-check</td><td>{ok(wg.initialStateCrossCheck?.bothPeers)} against boringtun&apos;s constants (non-authoritative)</td></tr>
              <tr><td>validated against</td><td>{wg.validatedAgainst}</td></tr>
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h3>IKEv2 key schedule: ML-KEM-768 as an RFC 9370 additional key exchange (draft-ietf-ipsecme-ikev2-mlkem) and a PPK (RFC 8784) <StageBadge kind="real" /></h3>
        <KeyPointsInline items={[
          "IKE_SA_INIT on ECP-256, then an IKE_INTERMEDIATE ML-KEM-768 exchange, then a post-quantum preshared key; both peers derive every key and the page compares them.",
          "The RFC 7296 derivation is checked against NIST CAVP SP 800-135 vectors; the RFC 9370 and RFC 8784 layers have no official vectors and are computed per the RFCs.",
          "ML-KEM in IKEv2 is not yet an RFC: draft-ietf-ipsecme-ikev2-mlkem is in the RFC Editor queue (IANA key exchange method 36).",
        ]} />
        <button className="btn" disabled={busy === "ike"}
          onClick={() => run("ike", async () => {
            const m = await import("../lib/netkeys/ikev2");
            setIke({ sched: m.ikev2KeySchedule({ seed: "qsh-validations" }), cavp: m.runIkev2Cavp() });
          })}>
          Run the key schedule
        </button>
        {ike && (
          <>
            <table className="facts-table small" style={{ marginTop: 10 }}>
              <thead><tr><th>stage</th><th>RFC</th><th>validated by</th><th>peers agree</th><th>keys changed</th></tr></thead>
              <tbody>
                {ike.sched.stages.map((st: any) => (
                  <tr key={st.name}>
                    <td>{st.name}</td><td>{st.rfc}</td>
                    <td>{st.validated ? st.validatedBy : "computed per spec - no official vectors"}</td>
                    <td>{ok(st.peersAgree)}</td>
                    <td className="mono">{st.changedFromPrevious.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <table className="facts-table small">
              <tbody>
                <tr><td>CAVP SP 800-135 cases</td><td>{ok(ike.cavp.allPassed)} - {ike.cavp.passed}/{ike.cavp.total}</td></tr>
                <tr><td>ECP-256 shared secret, RFC 5903 section 8.1</td><td>{ok(ike.sched.validation.rfc5903Passed)}</td></tr>
                <tr><td>g^ir and the ML-KEM secret agree between peers</td><td>{ok(ike.sched.exchange.girAgrees && ike.sched.exchange.mlkemSecretAgrees)}</td></tr>
                <tr><td>the PPK changes only SK_d, SK_pi and SK_pr</td><td>{ok(ike.sched.properties.ppkChangesOnlyDPiPr)}</td></tr>
                <tr><td>the ML-KEM exchange changes every non-empty key (SK_ai and SK_ar are empty under AES-GCM)</td><td>{ok(ike.sched.properties.additionalKeyExchangeChangesAll)}</td></tr>
              </tbody>
            </table>
            <Details summary="What is not validated here, and why">
              <ul>{(ike.sched.notValidated as string[]).map((n, i) => <li key={i}>{n}</li>)}</ul>
            </Details>
          </>
        )}
      </div>
    </div>
  );
}

/** A compact list for card bodies (KeyPoints sized for a card). */
function KeyPointsInline({ items }: { items: string[] }) {
  return <ul className="key-points" style={{ margin: "4px 0 8px", paddingLeft: 18, fontSize: 13 }}>{items.map((t, i) => <li key={i}>{t}</li>)}</ul>;
}
