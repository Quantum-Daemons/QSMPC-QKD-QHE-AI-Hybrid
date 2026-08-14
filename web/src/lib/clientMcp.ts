// Client-side quantum-safe MCP channel — a zero-server mirror of src/qsh/pqc/mcp_bridge.py (+
// transport.py). Two in-browser agents establish an ML-KEM-768 session (optionally mixed with a QKD
// key via HKDF-SHA256), then exchange an ML-DSA-65-signed, AES-256-GCM-sealed MCP-style JSON-RPC tool
// call: confidentiality + post-quantum authenticity + tamper rejection, no trusted third party. This
// realises the report's "MCP + quantum-safe layer for agentic AI" entirely in the browser. Reuses
// @noble/post-quantum (same as crypto.ts) + WebCrypto; used by the Agentic-AI page and the demo-mode
// Validations mirror so the public demo runs the MCP demo with zero server compute.
import { shake256 } from "@noble/hashes/sha3.js";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";
import { ml_kem768 } from "@noble/post-quantum/ml-kem.js";
import { x25519 } from "@noble/curves/ed25519.js";

const enc = new TextEncoder();
const dec = new TextDecoder();
const toHex = (u: Uint8Array) => Array.from(u).map((b) => b.toString(16).padStart(2, "0")).join("");
const fromHex = (h: string) => new Uint8Array((h.match(/.{1,2}/g) ?? []).map((x) => parseInt(x, 16)));
// copy bytes into a fresh ArrayBuffer-backed buffer so WebCrypto's BufferSource types are satisfied
// (TS 5.7+ types @noble outputs as Uint8Array<ArrayBufferLike>, which it rejects for SubtleCrypto).
const ab = (u: Uint8Array): ArrayBuffer => { const b = new ArrayBuffer(u.byteLength); new Uint8Array(b).set(u); return b; };
const concat = (...arrs: Uint8Array[]): Uint8Array => {
  const n = arrs.reduce((s, a) => s + a.length, 0);
  const o = new Uint8Array(n); let off = 0;
  for (const a of arrs) { o.set(a, off); off += a.length; }
  return o;
};
const bitsOf = (bytes: Uint8Array): Uint8Array => {
  const out = new Uint8Array(bytes.length * 8);
  for (let i = 0; i < bytes.length; i++) for (let b = 0; b < 8; b++) out[i * 8 + b] = (bytes[i] >> (7 - b)) & 1;
  return out;
};
const packBits = (bits: Uint8Array): Uint8Array => {
  const out = new Uint8Array(Math.ceil(bits.length / 8));
  for (let i = 0; i < bits.length; i++) if (bits[i]) out[i >> 3] |= 1 << (7 - (i & 7));
  return out;
};
async function sha256(bytes: Uint8Array): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", ab(bytes)));
}
// Information-theoretic leakage-resilient combiner. A BYTE-FOR-BYTE mirror of
// `transport._mix_qkd_its` (Giestinhas & Spiller 2026): a modified-Toeplitz + Leftover-Hash-Lemma
// strong extractor over pqc||qkd whose 256-bit output keeps the min-entropy of WHICHEVER input stays
// uncompromised — unlike HKDF (computational) or a naive XOR/concat.
//
// "Mirror" is a MEASURED claim here, not an aspiration. Until Round 39 this seeded its Toeplitz
// matrix from a SHA-256 counter chain, under a comment explaining that "WebCrypto has no SHAKE-256".
// That was true of WebCrypto and irrelevant to this bundle: `@noble/hashes` ships `shake256` and
// three other modules here already import from it. So the two implementations derived DIFFERENT keys
// from the same inputs while the comment asserted they mirrored. Measured on the tree that shipped:
//
//     python  : 72f108ccd0dc879a6752ec520a14167e2898723fc184b88a3c4a3ba5690a8fa1
//     browser : c05240436fae8508a61edd91376f4c0d74b69dfeb46bbe3f924a24a41a59f9f8
//
// That is the SAME defect as the hybrid-KEM divergence Round 38 shipped and Round 39 fixed — one
// scheme, two languages, different bytes, a comment claiming parity. This one was found by
// inventory rather than by a gate, so it now has the gate: `scripts/export_combiner_vectors.py`
// emits ITS vectors from the Python function and `npm run check:combiner` reproduces them through
// this module.
// Exported for `web/scripts/check_hybrid_combiner.mjs`, which reproduces committed Python vectors
// through THIS function. The alternative was a re-implementation in the gate, which would have
// agreed with itself while this module drifted - the precise failure the gate exists to catch.
export async function itsCombine(pqc: Uint8Array, qkd: Uint8Array): Promise<Uint8Array> {
  const combined = bitsOf(concat(pqc, qkd));                 // info-theoretic input material
  const outBits = 256;                                       // 32-byte AES-256 session key
  const needT = outBits + combined.length - 1;
  const tbytes = shake256(
    concat(enc.encode("qsh/its-combiner/v1\x00"), pqc, qkd),
    { dkLen: Math.ceil(needT / 8) },
  );
  const tvec = bitsOf(tbytes);
  const out = new Uint8Array(outBits);
  for (let i = 0; i < outBits; i++) { let acc = 0; for (let j = 0; j < combined.length; j++) acc ^= tvec[i + j] & combined[j]; out[i] = acc & 1; }
  return packBits(out);                                      // 32 bytes
}

// canonical JSON with sorted keys — mirrors Python `json.dumps(msg, sort_keys=True)` so the signed
// body is deterministic (the signature covers the exact same bytes the server would).
function stableStringify(v: any): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(stableStringify).join(",") + "]";
  return "{" + Object.keys(v).sort().map((k) => JSON.stringify(k) + ":" + stableStringify(v[k])).join(",") + "}";
}

export interface ClientAgent {
  name: string;
  kemPublicKey: Uint8Array; kemSecretKey: Uint8Array;
  xPublicKey: Uint8Array; xSecretKey: Uint8Array;
  sigPublicKey: Uint8Array; sigSecretKey: Uint8Array;
}

export interface McpResult {
  session: string; kem: string; qkd_mixed: boolean;
  combiner: string;           // PQC+QKD key combiner used ("hkdf" | "its_toeplitz" | "none" when unmixed)
  session_key_fp: string;     // 4-byte SHA-256 fingerprint of the derived AEAD key (changes with combiner/QKD-mix)
  method: string | null; authenticated: boolean;
  signature_alg: string; tamper_rejected: boolean;
}

// mirror Agent.create: a hybrid-KEM endpoint (ML-KEM-768 + X25519) + a PQC signing key (ML-DSA-65).
//
// The X25519 half is real. It was not: this file reported kem "ML-KEM-768+X25519" in every envelope
// while encapsulating with ml_kem768 alone, and the comment beside it said it was mirroring the
// convention in clientCompute -- which was itself only a label at the time. The same wrong claim
// therefore appeared on every use-case page's Agentic AI panel.
export function createAgent(name: string): ClientAgent {
  const kem = ml_kem768.keygen();
  const xSecret = x25519.utils.randomSecretKey();
  const sig = ml_dsa65.keygen(crypto.getRandomValues(new Uint8Array(32)));
  return { name, kemPublicKey: kem.publicKey, kemSecretKey: kem.secretKey,
    xPublicKey: x25519.getPublicKey(xSecret), xSecretKey: xSecret,
    sigPublicKey: sig.publicKey, sigSecretKey: sig.secretKey };
}

// Concatenate-then-KDF hybrid combiner for the AGENT-TO-AGENT scheme, which is NOT the handshake's.
//
// This said "matching src/qsh/pqc/hybrid_kem.py and clientCompute's pqc_handshake" - the same false
// claim the transcript fix below was made to remove, left standing at the top of the block that
// removes it. It does not match them and must not: the handshake's classical half is
// ephemeral-static X25519 with transcript c1 || c2 || ek1 || ek2 under `qsh/hybrid-kem/v2`; this is
// static-static between two agents, so there is no ephemeral c2, the transcript is
// c1 || ek1 || xA || xB, and the separator is its own. A maintainer told to make the two agree
// would silently re-merge two constructions under one separator.
//
// HKDF-SHA256 over (pq_ss || x25519_ss) with that transcript bound.
//
// The transcript argument is not decoration. Until Round 38 this function took only the two shared
// secrets and put the bare domain-separation string in `info`, which is exactly expression (16) of
// NIST SP 800-227 (September 2025) - `K <- KDF(K1, K2)` - the combiner section 4.6.3 states "does
// not preserve IND-CCA security, regardless of the properties of the KDF". Its sibling in
// `clientCompute.ts` at least bound the public keys; this one bound nothing, while its own comment
// claimed it matched that sibling and the backend.
//
// Section 4.6.3's IND-CCA-preserving form is H(K1, K2, c1, c2, ek1, ek2, domain_sep), and it notes
// that the encapsulation keys are NOT what delivers the property - the ciphertexts are. Here the
// classical half is static-static X25519 between two agents, so there is no ephemeral c2; the two
// static public keys are the binding material on that side, and c1 is the ML-KEM ciphertext.
//
// Caller and callee both hold every element, so both derive the same transcript independently - and
// if they ever did not, the AEAD open below would fail rather than silently pass.
async function hybridSecret(
  pq: Uint8Array, classical: Uint8Array, transcript: Uint8Array,
): Promise<Uint8Array> {
  const ikm = new Uint8Array(pq.length + classical.length);
  ikm.set(pq, 0); ikm.set(classical, pq.length);
  // ITS OWN LABEL, not the handshake's. SP 800-227 section 4.6.3: "The domain separator
  // domain_sep should be used to uniquely identify the composite scheme in use (e.g., P1, P2, order
  // of composition, choice of parameter set, key combiner, KDF)." This is a DIFFERENT composite
  // scheme from `pqc_handshake`: there the classical half is ephemeral-static X25519 and the
  // transcript is c1 || c2 || ek1 || ek2; here it is static-static between two agents, so there is
  // no ephemeral c2 and the transcript is c1 || ek1 || xA || xB. Reusing `qsh/hybrid-kem/v2` for
  // both - which the first version of this fix did - puts two constructions under one separator,
  // which is the thing a domain separator exists to prevent.
  const label = enc.encode("qsh/mcp-agent-kem/v1 ML-KEM-768+X25519-static|");
  const info = new Uint8Array(label.length + transcript.length);
  info.set(label, 0); info.set(transcript, label.length);
  const base = await crypto.subtle.importKey("raw", ab(ikm), "HKDF", false, ["deriveBits"]);
  const bits = await crypto.subtle.deriveBits(
    { name: "HKDF", hash: "SHA-256", salt: new ArrayBuffer(0), info: ab(info) }, base, 256);
  return new Uint8Array(bits);
}

/** c1 || ek1 || the two static X25519 publics, in a fixed order both agents can reproduce. */
function mcpTranscript(
  ct: Uint8Array, kemPub: Uint8Array, xA: Uint8Array, xB: Uint8Array,
): Uint8Array {
  const t = new Uint8Array(ct.length + kemPub.length + xA.length + xB.length);
  t.set(ct, 0);
  t.set(kemPub, ct.length);
  t.set(xA, ct.length + kemPub.length);
  t.set(xB, ct.length + kemPub.length + xA.length);
  return t;
}

// mirror transport._mix_qkd + AESGCM(session_key). ``combiner`` selects the PQC+QKD key combiner
// (crypto.hybrid_combiner_mode): "hkdf" = HKDF-SHA256 dual-PRF (holds if either input does, computationally);
// "its_toeplitz" = the information-theoretic Toeplitz+QLHL leakage-resilient combiner (itsCombine).
// Returns the CryptoKey + the raw derived bytes (for a non-secret fingerprint that proves the combiner ran).
async function deriveAeadKey(
  sharedSecret: Uint8Array, qkdKey?: Uint8Array, combiner = "hkdf",
): Promise<{ key: CryptoKey; bytes: Uint8Array }> {
  let keyBytes = sharedSecret;
  if (qkdKey && qkdKey.length) {
    if (combiner === "its_toeplitz") {
      keyBytes = await itsCombine(sharedSecret, qkdKey);      // ITS extractor, leakage-resilient
    } else {
      const base = await crypto.subtle.importKey("raw", ab(sharedSecret), "HKDF", false, ["deriveBits"]);
      const bits = await crypto.subtle.deriveBits(
        { name: "HKDF", hash: "SHA-256", salt: ab(qkdKey), info: ab(enc.encode("qsh/transport/pqc+qkd")) }, base, 256);
      keyBytes = new Uint8Array(bits);
    }
  }
  const key = await crypto.subtle.importKey("raw", ab(keyBytes), "AES-GCM", false, ["encrypt", "decrypt"]);
  return { key, bytes: keyBytes };
}

// mirror _wrap: JSON-RPC tool-call message, ML-DSA-signed, wrapped {body, sig, alg}
function wrapAndSign(agent: ClientAgent, method: string, params: object): Uint8Array {
  const body = stableStringify({ jsonrpc: "2.0", method, params });
  // noble 0.6.x: sign(message, secretKey).
  const sig = ml_dsa65.sign(enc.encode(body), agent.sigSecretKey);
  return enc.encode(stableStringify({ body, sig: toHex(sig), alg: "ML-DSA-65" }));
}

// mirror _unwrap: verify the signature, return (message, authenticated)
function unwrap(raw: Uint8Array, senderSigPublic: Uint8Array): { message: any; authenticated: boolean } {
  const envelope = JSON.parse(dec.decode(raw));
  // noble 0.6.x: verify(signature, message, publicKey).
  const ok = ml_dsa65.verify(fromHex(envelope.sig), enc.encode(envelope.body), senderSigPublic);
  return { message: JSON.parse(envelope.body), authenticated: !!ok };
}

// mirror SecureChannel.seal: nonce = 12-byte big-endian counter, AES-256-GCM, AAD = "mcp"
async function sealRecord(key: CryptoKey, plaintext: Uint8Array, counter = 0): Promise<Uint8Array> {
  const nonce = new Uint8Array(12);
  new DataView(nonce.buffer).setUint32(8, counter);
  const ct = new Uint8Array(await crypto.subtle.encrypt(
    { name: "AES-GCM", iv: nonce, additionalData: ab(enc.encode("mcp")) }, key, ab(plaintext)));
  const out = new Uint8Array(12 + ct.length);
  out.set(nonce); out.set(ct, 12);
  return out;
}

async function openRecord(key: CryptoKey, record: Uint8Array): Promise<Uint8Array> {
  const nonce = record.slice(0, 12), ct = record.slice(12);
  return new Uint8Array(await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: ab(nonce), additionalData: ab(enc.encode("mcp")) }, key, ab(ct)));
}

// mirror secure_agent_call: PQC(+QKD) session -> sign+seal -> open+verify -> tamper test
export async function clientSecureAgentCall(
  caller: ClientAgent, callee: ClientAgent, method: string, params: object,
  qkdKey?: Uint8Array, combiner = "hkdf",
): Promise<McpResult & { message: any }> {
  const { cipherText, sharedSecret } = ml_kem768.encapsulate(callee.kemPublicKey);
  const recovered = ml_kem768.decapsulate(cipherText, callee.kemSecretKey);   // responder reconstructs the secret
  // Classical half of the hybrid: X25519 between the two agents' static keys. Each side computes it
  // from its own secret and the peer's public key, so caller and callee derive it independently --
  // if they disagreed, the AEAD open below would fail rather than silently pass.
  const xCaller = x25519.getSharedSecret(caller.xSecretKey, callee.xPublicKey);
  const xCallee = x25519.getSharedSecret(callee.xSecretKey, caller.xPublicKey);
  // Both sides build the SAME transcript from material both already hold: the ML-KEM ciphertext,
  // the callee's KEM public key, and the two agents' static X25519 publics in caller-then-callee
  // order. Binding it is what stops a mauled ciphertext that decapsulates to the same secret from
  // yielding the same session key.
  const tr = mcpTranscript(cipherText, callee.kemPublicKey, caller.xPublicKey, callee.xPublicKey);
  const callerKey = await deriveAeadKey(await hybridSecret(sharedSecret, xCaller, tr), qkdKey, combiner);
  const calleeKey = await deriveAeadKey(await hybridSecret(recovered, xCallee, tr), qkdKey, combiner);
  const record = await sealRecord(callerKey.key, wrapAndSign(caller, method, params), 0);
  const opened = await openRecord(calleeKey.key, record);
  const { message, authenticated } = unwrap(opened, caller.sigPublicKey);
  let tamperRejected = false;
  const bad = record.slice();
  bad[bad.length - 1] ^= 0x01;
  try { await openRecord(calleeKey.key, bad); } catch { tamperRejected = true; }
  const mixed = !!(qkdKey && qkdKey.length);
  const fp = toHex((await sha256(callerKey.bytes)).slice(0, 4));   // non-secret fingerprint of the derived AEAD key
  return {
    session: `${caller.name}->${callee.name}`,
    kem: "ML-KEM-768+X25519",                          // both halves genuinely computed above
    qkd_mixed: mixed,
    combiner: mixed ? combiner : "none",               // the combiner only applies when a QKD key is mixed in
    session_key_fp: fp,
    method: message?.method ?? null, authenticated,
    signature_alg: "ML-DSA-65", tamper_rejected: tamperRejected, message,
  };
}

// mirror mcp_demo: analyst-agent -> hospital-silo signed quantum-safe tool call (same shape as server)
export async function clientMcpDemo(qkdMixed = false, combiner = "hkdf"): Promise<McpResult> {
  const a = createAgent("analyst-agent");
  const b = createAgent("hospital-silo");
  // A fixed QKD-DOMAIN constant, not this run's key, and the UI says so. The BB84/E91 models here
  // are analytic: they return a secret-key RATE and a secure-bit count, never key bytes, so there is
  // nothing run-derived to mix. 0x71 = 'q', matching the backend's b"q"*32 so the two sides' derived
  // fingerprints stay comparable. Synthesising a key and calling it QKD-derived would be worse than
  // a constant that is labelled as one.
  const qkd = qkdMixed ? new Uint8Array(32).fill(0x71) : undefined;
  const { message, ...rest } = await clientSecureAgentCall(a, b, "tools/call",
    { name: "query_cohort", arguments: { phenotype: "rare-disease-X" } }, qkd, combiner);
  return rest;
}

// Wrap an arbitrary payload (e.g. the in-browser AI {prompt, answer}) through the quantum-safe MCP
// channel, so the Agentic-AI panel proves the AI exchange is signed + sealed + tamper-evident. The
// ``toolName`` is the per-industry MCP tool the agent "calls" (finance → score_transaction, medical →
// query_cohort, …) so each use-case page shows its own delegate-and-verify tool call.
export async function clientSecureWrap(
  payload: object, qkdKey?: Uint8Array, toolName = "ai_inference", combiner = "hkdf",
): Promise<McpResult & { echoedPayload: any }> {
  const a = createAgent("browser-agent");
  const b = createAgent("qsh-orchestrator");
  const { message, ...rest } = await clientSecureAgentCall(a, b, "tools/call",
    { name: toolName, arguments: payload }, qkdKey, combiner);
  return { ...rest, echoedPayload: message?.params?.arguments ?? null };
}
