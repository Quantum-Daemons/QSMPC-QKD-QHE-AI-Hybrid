// The WireGuard handshake, both peers in one tab, on the Noise core in noiseIKpsk2.ts.
//
// Source: J. A. Donenfeld, "WireGuard: Next Generation Kernel Network Tunnel", whitepaper draft
// revision e2da747 (2020-06-01), https://www.wireguard.com/papers/wireguard.pdf, section 5.4.
//
// WHAT IS VALIDATED, AND WHAT IS ONLY WRITTEN FROM THE PAPER
//
// The cryptography is Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s with WireGuard's identifier as the Noise
// prologue - the whitepaper's 5.4.2/5.4.3 equations are that protocol's token rules written out (the
// mapping is spelled out next to `createInitiation` and `createResponse`). That core reproduces a
// published Noise vector byte for byte (cacophony; see noiseIKpsk2.ts), and `noiseCore` in the result
// re-runs it on every call so the page never reports a handshake from an unchecked core.
//
// The WireGuard-specific layer is NOT held to any vector: the message type and reserved bytes, the
// sender/receiver indices, the TAI64N payload, and mac1/mac2. No public, fixed handshake vector from
// an authoritative implementation was found (checked 2026-10-24): the whitepaper has none, and
// wireguard-go's device/noise_test.go handshakes between randomly keyed devices rather than against
// a fixed transcript. So the framing is computed per the whitepaper only, and the result says so in
// `validatedAgainst` - a page should never present the 148 and 92 below as checked against anything
// except the paper's own table.
//
// ONE WIREGUARD-SPECIFIC STEP IS CROSS-CHECKED, BECAUSE MEASURING SHOWED IT WAS INVISIBLE. Seeding an
// empty prologue in place of the Identifier passed every other check here: the two in-tab peers agree
// with each other whatever prologue they share, and the transport keys do not even depend on it (the
// prologue enters h, not ck). Real peers would fail the first AEAD. boringtun (Cloudflare, BSD-3-Clause)
// hardcodes the precomputed Ci = Hash(Construction) and Hi = Hash(Ci || Identifier), so the handshake
// objects' own post-prologue state is compared with those two constants (vectors/
// wireguard_initial_state.json) - one step, from an independent but not authoritative implementation,
// and reported as exactly that in `initialStateCrossCheck`.
//
// WHY BOTH PEERS RUN HERE
//
// So the page can show the two sides arriving at the same transport keys from bytes alone: the
// responder sees only the 148-byte initiation, the initiator only the 92-byte response. The keys are
// deterministic from a seed so the same seed reproduces the same page; real WireGuard draws the
// ephemerals and the 32-bit indices at random (DH-Generate, rho^4 in the paper). Sources and hashes:
// references/net-key-schedules.md.
import { blake2s } from "@noble/hashes/blake2.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex, concatBytes, hexToBytes, utf8ToBytes } from "@noble/hashes/utils.js";
import { equalBytes } from "@noble/ciphers/utils.js";
import {
  DHLEN, IKpsk2Handshake, NOISE_PROTOCOL_NAME, PSKLEN, TAGLEN, keyPairFromPrivate, runNoiseVectors,
  type NoiseKeyPair, type TransportCiphers,
} from "./noiseIKpsk2";
import bundledInitialState from "./vectors/wireguard_initial_state.json";

/** Whitepaper 5.4: "Construction", 37 bytes. Must equal the Noise protocol name, or nothing below holds. */
export const WG_CONSTRUCTION = "Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s";
/** Whitepaper 5.4: "Identifier", 34 bytes - the Noise prologue (Hi := Hash(Ci || Identifier)). */
export const WG_IDENTIFIER = "WireGuard v1 zx2c4 Jason@zx2c4.com";
/** Whitepaper 5.4: "Label-Mac1", 8 bytes. */
export const WG_LABEL_MAC1 = "mac1----";
export const WG_MESSAGE_INITIATION = 1;
export const WG_MESSAGE_RESPONSE = 2;
/** Whitepaper 5.4.2 / 5.4.3 field tables summed; wireguard-go's MessageInitiationSize/ResponseSize agree. */
export const WG_INITIATION_SIZE = 148;
export const WG_RESPONSE_SIZE = 92;
export const WG_VALIDATED_AGAINST = "Noise vectors only; WireGuard framing per the whitepaper";

const TAI64N_LEN = 12;
const MAC_LEN = 16;
/** Whitepaper 5.4: Q = 0^32 when pre-shared key mode is not in use. */
export const WG_NO_PSK = new Uint8Array(PSKLEN);

if (WG_CONSTRUCTION !== NOISE_PROTOCOL_NAME) throw new Error("WireGuard construction is not the Noise core's protocol");

// Field offsets, from the whitepaper's 5.4.2 / 5.4.3 tables. Integers are little-endian ("All integer
// assignments are little-endian, unless otherwise noted").
const INIT = { sender: 4, ephemeral: 8, static: 40, timestamp: 88, mac1: 116, mac2: 132 } as const;
const RESP = { sender: 4, receiver: 8, ephemeral: 12, empty: 44, mac1: 60, mac2: 76 } as const;

/**
 * TAI64N, whitepaper 5.4: 8-byte big-endian seconds label, then 4-byte big-endian nanoseconds.
 *
 * The label is 2^62 + 10 + Unix seconds, the base wireguard-go's tai64n package writes. The
 * responder only ever compares this against the last one it accepted (replay ordering), so the
 * epoch convention cannot change a key - it is fixed here only so the demo is reproducible.
 */
export function tai64n(unixMs: number): Uint8Array {
  const out = new Uint8Array(TAI64N_LEN);
  const view = new DataView(out.buffer);
  const seconds = Math.floor(unixMs / 1000);
  view.setBigUint64(0, (1n << 62n) + 10n + BigInt(seconds), false);
  view.setUint32(8, Math.round((unixMs - seconds * 1000) * 1e6), false);
  return out;
}

/**
 * mac1 = Mac(Hash(Label-Mac1 || S_pub of the RECEIVER), msg_alpha), whitepaper 5.4.4.
 *
 * Mac is keyed BLAKE2s with a 16-byte output - BLAKE2's own MAC mode, which is the one place this
 * handshake uses it; the key schedule uses HMAC-BLAKE2s. msg_alpha is every byte before mac1.
 */
export function wgMac1(receiverStaticPublic: Uint8Array, msgAlpha: Uint8Array): Uint8Array {
  const key = blake2s(concatBytes(utf8ToBytes(WG_LABEL_MAC1), receiverStaticPublic));
  return blake2s(msgAlpha, { key, dkLen: MAC_LEN });
}

function u32le(n: number): Uint8Array {
  const out = new Uint8Array(4);
  new DataView(out.buffer).setUint32(0, n >>> 0, true);
  return out;
}

function readU32le(b: Uint8Array, offset: number): number {
  return new DataView(b.buffer, b.byteOffset, b.byteLength).getUint32(offset, true);
}

function header(type: number): Uint8Array {
  return Uint8Array.of(type, 0, 0, 0); // type (1 byte), reserved := 0^3
}

// ------------------------------------------------------------------------------------------------
// The two messages
// ------------------------------------------------------------------------------------------------

export interface InitiatorSetup {
  senderIndex: number;
  s: NoiseKeyPair;
  /** Fixed for reproducibility; WireGuard generates it per handshake. */
  e: NoiseKeyPair;
  responderStatic: Uint8Array;
  psk: Uint8Array;
  timestamp: Uint8Array;
}

/**
 * Handshake initiation, whitepaper 5.4.2. The paper's equations are the Noise tokens "-> e, es, s, ss"
 * with the timestamp as payload:
 *
 *     Ci := Hash(Construction); Hi := Hash(Ci || Identifier)   InitializeSymmetric + MixHash(prologue)
 *     Hi := Hash(Hi || Srpub)                                   pre-message "<- s"
 *     Ci := Kdf1(Ci, Eipub); Hi := Hash(Hi || msg.ephemeral)    "e" (psk mode: MixHash then MixKey)
 *     (Ci, k) := Kdf2(Ci, DH(Eipriv, Srpub))                   "es"
 *     msg.static := Aead(k, 0, Sipub, Hi); Hi := Hash(...)      "s"  (EncryptAndHash)
 *     (Ci, k) := Kdf2(Ci, DH(Sipriv, Srpub))                   "ss"
 *     msg.timestamp := Aead(k, 0, Timestamp(), Hi); ...        payload (EncryptAndHash)
 */
export function createInitiation(p: InitiatorSetup): { message: Uint8Array; handshake: IKpsk2Handshake } {
  if (p.timestamp.length !== TAI64N_LEN) throw new Error("TAI64N timestamp must be 12 bytes");
  const handshake = new IKpsk2Handshake({
    initiator: true, prologue: utf8ToBytes(WG_IDENTIFIER), s: p.s, e: p.e, rs: p.responderStatic, psk: p.psk,
  });
  const { message: noise } = handshake.writeMessage(p.timestamp);
  const alpha = concatBytes(header(WG_MESSAGE_INITIATION), u32le(p.senderIndex), noise);
  const message = concatBytes(alpha, wgMac1(p.responderStatic, alpha), new Uint8Array(MAC_LEN)); // mac2 := 0^16
  if (noise.length !== DHLEN + (DHLEN + TAGLEN) + (TAI64N_LEN + TAGLEN)) throw new Error("unexpected initiation body");
  return { message, handshake };
}

export interface ResponderSetup {
  s: NoiseKeyPair;
  e: NoiseKeyPair;
  /** This single-peer responder's PresharedKey. A multi-peer one would pick it by the static key
   *  decrypted from message 1 - psk2 permits that, because the PSK is not touched until message 2. */
  psk: Uint8Array;
}

export interface ConsumedInitiation {
  handshake: IKpsk2Handshake;
  mac1Valid: boolean;
  senderIndex: number;
  initiatorStatic: Uint8Array;
  timestamp: Uint8Array;
}

/** The responder's side of 5.4.2. mac1 is checked FIRST, before any DH, as section 5.3 requires. */
export function consumeInitiation(message: Uint8Array, p: ResponderSetup): ConsumedInitiation {
  if (message.length !== WG_INITIATION_SIZE) throw new Error(`initiation must be ${WG_INITIATION_SIZE} bytes`);
  if (message[0] !== WG_MESSAGE_INITIATION || message[1] || message[2] || message[3]) throw new Error("not a handshake initiation");
  const mac1Valid = equalBytes(message.slice(INIT.mac1, INIT.mac2), wgMac1(p.s.publicKey, message.slice(0, INIT.mac1)));
  if (!mac1Valid) throw new Error("initiation mac1 invalid");
  const handshake = new IKpsk2Handshake({ initiator: false, prologue: utf8ToBytes(WG_IDENTIFIER), s: p.s, e: p.e, psk: p.psk });
  const { payload } = handshake.readMessage(message.slice(INIT.ephemeral, INIT.mac1));
  return {
    handshake, mac1Valid, senderIndex: readU32le(message, INIT.sender),
    initiatorStatic: handshake.remoteStatic as Uint8Array, timestamp: payload,
  };
}

/**
 * Handshake response, whitepaper 5.4.3: the Noise tokens "<- e, ee, se, psk" with an empty payload.
 *
 *     Cr := Kdf1(Cr, Erpub); Hr := Hash(Hr || msg.ephemeral)   "e" (psk mode)
 *     Cr := Kdf1(Cr, DH(Erpriv, Eipub))                        "ee"
 *     Cr := Kdf1(Cr, DH(Erpriv, Sipub))                        "se" (responder side: DH(e, rs))
 *     (Cr, tau, k) := Kdf3(Cr, Q); Hr := Hash(Hr || tau)       "psk" (MixKeyAndHash)
 *     msg.empty := Aead(k, 0, empty, Hr); Hr := Hash(...)      payload (EncryptAndHash)
 *
 * then 5.4.5: (T_i^send = T_r^recv, T_i^recv = T_r^send) := Kdf2(C, empty), which is Split().
 */
export function createResponse(
  consumed: ConsumedInitiation, senderIndex: number,
): { message: Uint8Array; transport: TransportCiphers } {
  const { message: noise, transport } = consumed.handshake.writeMessage(new Uint8Array(0));
  if (!transport) throw new Error("the response must complete the handshake");
  const alpha = concatBytes(header(WG_MESSAGE_RESPONSE), u32le(senderIndex), u32le(consumed.senderIndex), noise);
  const message = concatBytes(alpha, wgMac1(consumed.initiatorStatic, alpha), new Uint8Array(MAC_LEN));
  return { message, transport };
}

export interface ConsumedResponse {
  mac1Valid: boolean;
  senderIndex: number;
  receiverIndex: number;
  transport: TransportCiphers;
}

/** The initiator's side of 5.4.3. A wrong PSK fails HERE, at msg.empty's tag - not at mac1. */
export function consumeResponse(message: Uint8Array, handshake: IKpsk2Handshake, initiatorStatic: Uint8Array): ConsumedResponse {
  if (message.length !== WG_RESPONSE_SIZE) throw new Error(`response must be ${WG_RESPONSE_SIZE} bytes`);
  if (message[0] !== WG_MESSAGE_RESPONSE || message[1] || message[2] || message[3]) throw new Error("not a handshake response");
  const mac1Valid = equalBytes(message.slice(RESP.mac1, RESP.mac2), wgMac1(initiatorStatic, message.slice(0, RESP.mac1)));
  if (!mac1Valid) throw new Error("response mac1 invalid");
  const { payload, transport } = handshake.readMessage(message.slice(RESP.ephemeral, RESP.mac1));
  if (payload.length !== 0) throw new Error("response payload must be empty");
  if (!transport) throw new Error("the response must complete the handshake");
  return { mac1Valid, senderIndex: readU32le(message, RESP.sender), receiverIndex: readU32le(message, RESP.receiver), transport };
}

// ------------------------------------------------------------------------------------------------
// The one cross-checked WireGuard-specific step (see the header)
// ------------------------------------------------------------------------------------------------

export interface WireGuardInitialStateFile {
  construction: string;
  identifier: string;
  source: { repository: string; commit: string; path: string; url: string; sha256: string; [k: string]: unknown };
  authoritative: boolean;
  initialChainingKey: { formula: string; hex: string };
  initialHash: { formula: string; hex: string };
}

export interface WireGuardInitialStateCheck {
  source: { repository: string; commit: string; url: string; sha256: string };
  /** Ci = Hash(Construction): also pinned by the Noise vector, which shares the protocol name. */
  chainingKeyMatches: boolean;
  /** Hi = Hash(Ci || Identifier): the WireGuard-specific half, pinned nowhere else. */
  hashMatches: boolean;
  scope: string;
}

/**
 * Compare a handshake object's state after the prologue with boringtun's precomputed constants. Takes
 * the HANDSHAKE, not a recomputation, so it sees the prologue the caller really passed.
 */
export function wgInitialStateCheck(
  handshake: IKpsk2Handshake, file: WireGuardInitialStateFile = bundledInitialState as WireGuardInitialStateFile,
): WireGuardInitialStateCheck {
  return {
    source: { repository: file.source.repository, commit: file.source.commit, url: file.source.url, sha256: file.source.sha256 },
    chainingKeyMatches: equalBytes(handshake.afterPrologue.ck, hexToBytes(file.initialChainingKey.hex)),
    hashMatches: equalBytes(handshake.afterPrologue.h, hexToBytes(file.initialHash.hex)),
    scope: "Hash(Construction) and Hash(Ci || Identifier) only - one step of the handshake, from an independent (not authoritative) implementation; not a transcript",
  };
}

// ------------------------------------------------------------------------------------------------
// The in-tab demonstration
// ------------------------------------------------------------------------------------------------

export interface WireGuardHandshakeOptions {
  /** 32 bytes, or 64 hex characters. Default 0^32, WireGuard's value when no PresharedKey is set. */
  psk?: Uint8Array | string;
  /** Any string or bytes; every key, ephemeral and index is derived from it. */
  seed?: string | Uint8Array;
  /** Unix milliseconds for the TAI64N payload. Fixed by default so a seed reproduces every byte. */
  timestampMs?: number;
}

/** First 8 bytes of SHA-256, hex: enough to tell keys apart on a page without printing a key. */
export type Fingerprint = string;

interface RunKeys {
  initiatorSend: Uint8Array; initiatorRecv: Uint8Array; responderSend: Uint8Array; responderRecv: Uint8Array;
}

interface Run {
  initiation: Uint8Array;
  response: Uint8Array;
  keys: RunKeys;
  agree: boolean;
  handshakeHashesAgree: boolean;
  handshakeHash: Uint8Array;
  mac1: { initiation: boolean; response: boolean };
  initiatorStaticRecovered: boolean;
  timestampRecovered: boolean;
  indices: { initiator: number; responder: number };
  initialState: { initiator: WireGuardInitialStateCheck; responder: WireGuardInitialStateCheck };
}

export interface WireGuardKeyFingerprints {
  initiatorSend: Fingerprint;
  initiatorRecv: Fingerprint;
  responderSend: Fingerprint;
  responderRecv: Fingerprint;
}

export interface WireGuardHandshakeResult {
  construction: string;
  identifier: string;
  validatedAgainst: typeof WG_VALIDATED_AGAINST;
  /** The Noise core re-run against its published vector on this call. */
  noiseCore: { protocolName: string; vectorSource: string; vectorCommit: string; passed: boolean };
  /** Both peers' post-prologue state against boringtun's precomputed constants (one step only). */
  initialStateCrossCheck: WireGuardInitialStateCheck & { bothPeers: boolean };
  messageSizes: { initiation: number; response: number };
  whitepaperSizes: { initiation: number; response: number };
  sizesMatchWhitepaper: boolean;
  mac1Valid: { initiation: boolean; response: boolean };
  /** The responder decrypted msg.static to the initiator's real static key. */
  initiatorStaticRecovered: boolean;
  timestampRecovered: boolean;
  /** T_i^send == T_r^recv and T_i^recv == T_r^send, each side computed from the other's bytes. */
  transportKeysAgree: boolean;
  handshakeHashesAgree: boolean;
  fingerprints: WireGuardKeyFingerprints & { handshakeHash: Fingerprint };
  indices: { initiator: number; responder: number };
  psk: { fingerprint: Fingerprint; isDefaultZero: boolean };
  /** The same seed - same statics, ephemerals, indices, timestamp - with one PSK bit flipped. */
  pskChange: {
    alteredPskFingerprint: Fingerprint;
    messageSizes: { initiation: number; response: number };
    sizesUnchanged: boolean;
    /** psk2 mixes the PSK only into message 2, so the initiation is byte-identical. */
    initiationUnchanged: boolean;
    responseChanged: boolean;
    transportKeysChanged: boolean;
    transportKeysAgree: boolean;
    fingerprints: WireGuardKeyFingerprints;
  };
  /** Initiator holds the PSK, responder the altered one: the initiator must reject the response. */
  pskMismatch: { rejected: boolean; reason: string };
  notes: string[];
}

const fp = (b: Uint8Array): Fingerprint => bytesToHex(sha256(b)).slice(0, 16);
const DEMO_TIMESTAMP_MS = Date.UTC(2026, 0, 1, 0, 0, 0);

function toBytes(v: Uint8Array | string, what: string, length?: number): Uint8Array {
  let out: Uint8Array;
  if (typeof v === "string") {
    out = length !== undefined ? hexToBytes(v.trim()) : utf8ToBytes(v);
  } else {
    out = v;
  }
  if (length !== undefined && out.length !== length) throw new Error(`${what} must be ${length} bytes`);
  return out;
}

/** Domain-separated derivation of every demo secret from the seed. Demo keys only. */
function derive(seed: Uint8Array, label: string): Uint8Array {
  return blake2s(concatBytes(utf8ToBytes(`qsh/netkeys/wireguard-demo/v1|${label}|`), seed));
}

function runOnce(seed: Uint8Array, initiatorPsk: Uint8Array, responderPsk: Uint8Array, timestampMs: number): Run {
  const initiatorS = keyPairFromPrivate(derive(seed, "initiator static"));
  const responderS = keyPairFromPrivate(derive(seed, "responder static"));
  const initiatorE = keyPairFromPrivate(derive(seed, "initiator ephemeral"));
  const responderE = keyPairFromPrivate(derive(seed, "responder ephemeral"));
  const indices = {
    initiator: readU32le(derive(seed, "initiator index"), 0),
    responder: readU32le(derive(seed, "responder index"), 0),
  };
  const timestamp = tai64n(timestampMs);

  const { message: initiation, handshake: iHs } = createInitiation({
    senderIndex: indices.initiator, s: initiatorS, e: initiatorE, responderStatic: responderS.publicKey,
    psk: initiatorPsk, timestamp,
  });
  // From here the responder works from the 148 bytes alone.
  const consumed = consumeInitiation(initiation, { s: responderS, e: responderE, psk: responderPsk });
  const { message: response, transport: rT } = createResponse(consumed, indices.responder);
  // And the initiator from the 92 bytes alone.
  const done = consumeResponse(response, iHs, initiatorS.publicKey);
  const iT = done.transport;
  const keys: RunKeys = {
    initiatorSend: iT.send.key as Uint8Array, initiatorRecv: iT.recv.key as Uint8Array,
    responderSend: rT.send.key as Uint8Array, responderRecv: rT.recv.key as Uint8Array,
  };
  return {
    initiation, response, keys,
    agree: equalBytes(keys.initiatorSend, keys.responderRecv) && equalBytes(keys.initiatorRecv, keys.responderSend)
      && !equalBytes(keys.initiatorSend, keys.initiatorRecv),
    handshakeHashesAgree: equalBytes(iHs.getHandshakeHash(), consumed.handshake.getHandshakeHash()),
    handshakeHash: iHs.getHandshakeHash(),
    mac1: { initiation: consumed.mac1Valid, response: done.mac1Valid },
    initiatorStaticRecovered: equalBytes(consumed.initiatorStatic, initiatorS.publicKey),
    timestampRecovered: equalBytes(consumed.timestamp, timestamp),
    indices: { initiator: consumed.senderIndex, responder: done.senderIndex },
    initialState: { initiator: wgInitialStateCheck(iHs), responder: wgInitialStateCheck(consumed.handshake) },
  };
}

const fps = (k: RunKeys): WireGuardKeyFingerprints => ({
  initiatorSend: fp(k.initiatorSend), initiatorRecv: fp(k.initiatorRecv),
  responderSend: fp(k.responderSend), responderRecv: fp(k.responderRecv),
});

/**
 * Run the WireGuard handshake between two in-tab peers, then again with one PSK bit flipped, then
 * once with the two peers holding DIFFERENT PSKs.
 *
 * Deterministic: the same seed, PSK and timestamp give the same bytes on every call and machine.
 */
export function wireguardHandshake(opts: WireGuardHandshakeOptions = {}): WireGuardHandshakeResult {
  const seed = toBytes(opts.seed ?? "qsh-wireguard-demo", "seed");
  const psk = opts.psk === undefined ? WG_NO_PSK : toBytes(opts.psk, "psk", PSKLEN);
  const timestampMs = opts.timestampMs ?? DEMO_TIMESTAMP_MS;
  const altered = psk.slice();
  altered[PSKLEN - 1] ^= 0x01;

  const base = runOnce(seed, psk, psk, timestampMs);
  const other = runOnce(seed, altered, altered, timestampMs);

  let mismatch: { rejected: boolean; reason: string };
  try {
    runOnce(seed, psk, altered, timestampMs);
    mismatch = { rejected: false, reason: "the initiator accepted a response built with a different PSK" };
  } catch (err) {
    mismatch = { rejected: true, reason: err instanceof Error ? err.message : String(err) };
  }

  const noise = runNoiseVectors();
  const keyChanged = (a: Uint8Array, b: Uint8Array) => !equalBytes(a, b);
  return {
    construction: WG_CONSTRUCTION,
    identifier: WG_IDENTIFIER,
    validatedAgainst: WG_VALIDATED_AGAINST,
    noiseCore: {
      protocolName: noise.protocolName, vectorSource: noise.source.url, vectorCommit: noise.source.commit,
      passed: noise.allPassed,
    },
    initialStateCrossCheck: {
      ...base.initialState.initiator,
      bothPeers: [base.initialState.initiator, base.initialState.responder].every((c) => c.chainingKeyMatches && c.hashMatches),
    },
    messageSizes: { initiation: base.initiation.length, response: base.response.length },
    whitepaperSizes: { initiation: WG_INITIATION_SIZE, response: WG_RESPONSE_SIZE },
    sizesMatchWhitepaper: base.initiation.length === WG_INITIATION_SIZE && base.response.length === WG_RESPONSE_SIZE,
    mac1Valid: base.mac1,
    initiatorStaticRecovered: base.initiatorStaticRecovered,
    timestampRecovered: base.timestampRecovered,
    transportKeysAgree: base.agree,
    handshakeHashesAgree: base.handshakeHashesAgree,
    fingerprints: { ...fps(base.keys), handshakeHash: fp(base.handshakeHash) },
    indices: base.indices,
    psk: { fingerprint: fp(psk), isDefaultZero: equalBytes(psk, WG_NO_PSK) },
    pskChange: {
      alteredPskFingerprint: fp(altered),
      messageSizes: { initiation: other.initiation.length, response: other.response.length },
      sizesUnchanged: other.initiation.length === base.initiation.length && other.response.length === base.response.length,
      initiationUnchanged: equalBytes(other.initiation, base.initiation),
      responseChanged: !equalBytes(other.response, base.response),
      transportKeysChanged: keyChanged(other.keys.initiatorSend, base.keys.initiatorSend)
        && keyChanged(other.keys.initiatorRecv, base.keys.initiatorRecv),
      transportKeysAgree: other.agree,
      fingerprints: fps(other.keys),
    },
    pskMismatch: mismatch,
    notes: [
      "Validated: the Noise core reproduces the cacophony Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s vector (both handshake messages, four transport messages, handshake hash).",
      "Not validated against any vector: message types, indices, TAI64N, mac1 and mac2. No public fixed WireGuard handshake vector from an authoritative implementation was found; these follow whitepaper section 5.4.",
      "Cross-checked, one step only: both peers' state after Construction and Identifier equals the constants boringtun (Cloudflare) hardcodes - an independent implementation, not an authoritative vector.",
      "mac2 is all zeros: no cookie is held, the not-under-load case of section 5.4.4.",
      "Demo keys, ephemerals and indices are derived from the seed so the page is reproducible; WireGuard generates them at random for every handshake.",
    ],
  };
}
