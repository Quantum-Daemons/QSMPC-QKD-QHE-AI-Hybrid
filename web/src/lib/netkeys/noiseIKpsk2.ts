// Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s, written from the Noise Protocol Framework specification,
// revision 34 (2018-07-11, https://noiseprotocol.org/noise.html), and held to a published vector.
//
// WHY A GENERIC NOISE CORE AND NOT "WIREGUARD"
//
// WireGuard's handshake IS this Noise protocol, with the identifier string as prologue and two fixed
// payloads. Everything that makes it WireGuard - message types, sender indices, TAI64N, mac1/mac2 -
// sits outside the Noise state machine. No authoritative WireGuard implementation publishes a fixed
// handshake transcript (see wireguard.ts), but this exact Noise protocol HAS a published vector. So
// the part that can be held to a vector lives here, on its own, and wireguard.ts adds only framing.
//
// WHAT HOLDS IT
//
// cacophony's vector for exactly this protocol name, copied verbatim into vectors/noise_ikpsk2.json:
// both handshake messages, the four transport messages that follow Split(), and the handshake hash.
// `runNoiseVectors()` reproduces all of it in the tab, and web/scripts/check_net_key_schedules.mjs
// reproduces it again through this same module. A slip in any rule below - the psk-mode MixKey on
// "e" (section 9.2), the token order (section 9.4), the nonce encoding (section 12.3), which Split()
// output each side sends with - changes ciphertext bytes, and those checks fail. Sources, URLs and
// SHA-256 of every file: references/net-key-schedules.md.
//
// WHAT IT IS NOT
//
// No rekey (section 11.3), no out-of-order nonces (11.4), and no side-channel claims of any kind. It
// exists so a page can show a key schedule computed from the spec and checked against a published
// vector, not as a transport anyone should tunnel traffic through.
import { blake2s } from "@noble/hashes/blake2.js";
import { hmac } from "@noble/hashes/hmac.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex, concatBytes, hexToBytes, utf8ToBytes } from "@noble/hashes/utils.js";
import { chacha20poly1305 } from "@noble/ciphers/chacha.js";
import { equalBytes } from "@noble/ciphers/utils.js";
import { x25519 } from "@noble/curves/ed25519.js";
import bundledVectors from "./vectors/noise_ikpsk2.json";

export const NOISE_PROTOCOL_NAME = "Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s";
/** Section 12.7: BLAKE2s with a 32-byte digest. (BLOCKLEN = 64 is internal to the HMAC below.) */
export const HASHLEN = 32;
/** Section 12.1: X25519 keys and outputs. */
export const DHLEN = 32;
/** Poly1305 tag carried by every ChaChaPoly ciphertext. */
export const TAGLEN = 16;
/** Section 9.2: "psk is a 32-byte secret value provided by the application". */
export const PSKLEN = 32;
/** Section 5.1: the nonce 2^64-1 is reserved, so reaching it is an error, not a wrap. */
const NONCE_LIMIT = (1n << 64n) - 1n;
const EMPTY = new Uint8Array(0);

export type Token = "e" | "s" | "ee" | "es" | "se" | "ss" | "psk";

/**
 * IK (section 7.5) with the psk2 modifier (section 9.4), as data rather than as code.
 *
 *     IKpsk2:
 *       <- s
 *       ...
 *       -> e, es, s, ss
 *       <- e, ee, se, psk
 *
 * Kept as a table so that reading it against the spec is a comparison of two lists. psk2 means the
 * psk token is appended to the SECOND message: the initiation never depends on the PSK, which
 * wireguard.ts measures rather than assumes.
 */
export const IKPSK2_PATTERN: { readonly responderPreMessage: readonly Token[]; readonly messages: readonly (readonly Token[])[] } = {
  responderPreMessage: ["s"],
  messages: [["e", "es", "s", "ss"], ["e", "ee", "se", "psk"]],
};

// ------------------------------------------------------------------------------------------------
// Section 4.3: hash functions
// ------------------------------------------------------------------------------------------------

/** HASH(data): BLAKE2s-256. */
export function noiseHash(data: Uint8Array): Uint8Array {
  return blake2s(data);
}

/**
 * HMAC-HASH(key, data): RFC 2104 HMAC instantiated with BLAKE2s.
 *
 * NOT keyed BLAKE2s. BLAKE2 has its own MAC mode, and WireGuard uses it - but only for mac1/mac2.
 * The key schedule uses HMAC over the plain hash, so swapping one for the other would still produce
 * 32 plausible bytes and fail only against the vector.
 */
function hmacHash(key: Uint8Array, data: Uint8Array): Uint8Array {
  return hmac(blake2s, key, data);
}

/**
 * HKDF(chaining_key, input_key_material, num_outputs), section 4.3, step for step.
 *
 * Written out rather than calling a library HKDF: the spec notes it equals RFC 5869 HKDF with the
 * chaining key as salt and empty info, but the literal form is the one the WireGuard whitepaper's
 * Kdf_n repeats, so both documents can be read against these five lines directly.
 */
export function noiseHkdf(chainingKey: Uint8Array, ikm: Uint8Array, numOutputs: 2 | 3): Uint8Array[] {
  const tempKey = hmacHash(chainingKey, ikm);
  const output1 = hmacHash(tempKey, Uint8Array.of(0x01));
  const output2 = hmacHash(tempKey, concatBytes(output1, Uint8Array.of(0x02)));
  if (numOutputs === 2) return [output1, output2];
  const output3 = hmacHash(tempKey, concatBytes(output2, Uint8Array.of(0x03)));
  return [output1, output2, output3];
}

// ------------------------------------------------------------------------------------------------
// Sections 12.1 and 12.3: X25519 and ChaChaPoly
// ------------------------------------------------------------------------------------------------

export interface NoiseKeyPair {
  privateKey: Uint8Array;
  publicKey: Uint8Array;
}

/** A key pair from a fixed 32-byte private key - how vectors and the seeded demo supply keys. */
export function keyPairFromPrivate(privateKey: Uint8Array): NoiseKeyPair {
  if (privateKey.length !== DHLEN) throw new Error(`X25519 private key must be ${DHLEN} bytes`);
  return { privateKey, publicKey: x25519.getPublicKey(privateKey) };
}

/** GENERATE_KEYPAIR() for callers that do not fix the ephemeral. */
export function generateKeyPair(): NoiseKeyPair {
  return keyPairFromPrivate(x25519.utils.randomSecretKey());
}

/**
 * DH(key_pair, public_key). Section 12.1 allows an implementation to signal an error where X25519
 * would output all zeros; @noble does, and WireGuard requires the rejection, so it is kept.
 */
function dh(keyPair: NoiseKeyPair, publicKey: Uint8Array): Uint8Array {
  return x25519.getSharedSecret(keyPair.privateKey, publicKey);
}

/** Section 12.3: 32 bits of zeros, then the 64-bit nonce little-endian. */
function chachaNonce(n: bigint): Uint8Array {
  const out = new Uint8Array(12);
  new DataView(out.buffer).setBigUint64(4, n, true);
  return out;
}

// ------------------------------------------------------------------------------------------------
// Section 5.1: CipherState
// ------------------------------------------------------------------------------------------------

export class CipherState {
  private k: Uint8Array | null = null;
  private n = 0n;

  /** InitializeKey(key). `null` is the spec's "empty". */
  initializeKey(key: Uint8Array | null): void {
    this.k = key;
    this.n = 0n;
  }

  hasKey(): boolean {
    return this.k !== null;
  }

  /** The current key, so a page can fingerprint it and both sides can be compared byte for byte. */
  get key(): Uint8Array | null {
    return this.k;
  }

  get nonce(): bigint {
    return this.n;
  }

  /** EncryptWithAd(ad, plaintext): plaintext passes through while k is empty, as the spec says. */
  encryptWithAd(ad: Uint8Array, plaintext: Uint8Array): Uint8Array {
    if (this.k === null) return plaintext;
    if (this.n >= NONCE_LIMIT) throw new Error("Noise nonce exhausted (section 5.1)");
    const ct = chacha20poly1305(this.k, chachaNonce(this.n), ad).encrypt(plaintext);
    this.n += 1n;
    return ct;
  }

  /** DecryptWithAd(ad, ciphertext). On an authentication failure noble throws and n stays put. */
  decryptWithAd(ad: Uint8Array, ciphertext: Uint8Array): Uint8Array {
    if (this.k === null) return ciphertext;
    if (this.n >= NONCE_LIMIT) throw new Error("Noise nonce exhausted (section 5.1)");
    const pt = chacha20poly1305(this.k, chachaNonce(this.n), ad).decrypt(ciphertext);
    this.n += 1n;
    return pt;
  }
}

// ------------------------------------------------------------------------------------------------
// Section 5.2: SymmetricState
// ------------------------------------------------------------------------------------------------

export class SymmetricState {
  ck: Uint8Array;
  h: Uint8Array;
  readonly cipher = new CipherState();

  /** InitializeSymmetric(protocol_name). */
  constructor(protocolName: Uint8Array) {
    if (protocolName.length <= HASHLEN) {
      this.h = new Uint8Array(HASHLEN);
      this.h.set(protocolName);
    } else {
      // The 37-byte name here takes this branch; a 32-byte name would be zero-padded instead.
      this.h = noiseHash(protocolName);
    }
    this.ck = this.h.slice();
    this.cipher.initializeKey(null);
  }

  /** MixKey: HASHLEN is 32, so temp_k needs no truncation. */
  mixKey(ikm: Uint8Array): void {
    const [ck, tempK] = noiseHkdf(this.ck, ikm, 2);
    this.ck = ck;
    this.cipher.initializeKey(tempK);
  }

  mixHash(data: Uint8Array): void {
    this.h = noiseHash(concatBytes(this.h, data));
  }

  /** MixKeyAndHash: the psk token (section 9.1), the only caller. */
  mixKeyAndHash(ikm: Uint8Array): void {
    const [ck, tempH, tempK] = noiseHkdf(this.ck, ikm, 3);
    this.ck = ck;
    this.mixHash(tempH);
    this.cipher.initializeKey(tempK);
  }

  encryptAndHash(plaintext: Uint8Array): Uint8Array {
    const ciphertext = this.cipher.encryptWithAd(this.h, plaintext);
    this.mixHash(ciphertext);
    return ciphertext;
  }

  /** The hash absorbs the CIPHERTEXT, and only after it authenticated. */
  decryptAndHash(ciphertext: Uint8Array): Uint8Array {
    const plaintext = this.cipher.decryptWithAd(this.h, ciphertext);
    this.mixHash(ciphertext);
    return plaintext;
  }

  /** Split(): (c1, c2) = HKDF(ck, zerolen, 2). */
  split(): [CipherState, CipherState] {
    const [k1, k2] = noiseHkdf(this.ck, EMPTY, 2);
    const c1 = new CipherState();
    const c2 = new CipherState();
    c1.initializeKey(k1);
    c2.initializeKey(k2);
    return [c1, c2];
  }
}

// ------------------------------------------------------------------------------------------------
// Section 5.3: HandshakeState, for the one pattern above
// ------------------------------------------------------------------------------------------------

export interface IKpsk2Config {
  initiator: boolean;
  /** Section 6. Both sides must supply identical bytes or the first decrypt fails. */
  prologue: Uint8Array;
  /** Local static key pair. */
  s: NoiseKeyPair;
  /**
   * Local ephemeral. Supplied, it stands in for GENERATE_KEYPAIR() at the "e" token - how a test
   * vector fixes it, and how the seeded demo stays reproducible. Omitted, a fresh one is generated.
   */
  e?: NoiseKeyPair;
  /** Initiator only: the responder's static public key, the "<- s" pre-message. */
  rs?: Uint8Array;
  psk: Uint8Array;
}

/**
 * The transport ciphers after Split(). Section 5: "the first for encrypting transport messages from
 * initiator to responder, and the second for messages in the other direction". `send` / `recv` are
 * that rule applied to this side's role, so a caller cannot pick the wrong one.
 */
export interface TransportCiphers {
  c1: CipherState;
  c2: CipherState;
  send: CipherState;
  recv: CipherState;
}

export class IKpsk2Handshake {
  readonly initiator: boolean;
  private readonly symmetric: SymmetricState;
  private readonly s: NoiseKeyPair;
  private e: NoiseKeyPair | null;
  private rs: Uint8Array | null;
  private re: Uint8Array | null = null;
  private readonly psk: Uint8Array;
  private readonly remaining: Token[][];
  private transport: TransportCiphers | null = null;
  /**
   * (ck, h) right after InitializeSymmetric and MixHash(prologue), before any pre-message: the state
   * WireGuard implementations precompute. Kept so it can be compared with one that publishes it -
   * the only check that sees which prologue a caller actually passed, since a wrong one still lets
   * two copies of this code agree with each other.
   */
  readonly afterPrologue: { ck: Uint8Array; h: Uint8Array };

  /** Initialize(handshake_pattern, initiator, prologue, s, e, rs, re). */
  constructor(cfg: IKpsk2Config) {
    if (cfg.psk.length !== PSKLEN) throw new Error(`psk must be ${PSKLEN} bytes (Noise section 9.2)`);
    if (cfg.initiator && !cfg.rs) throw new Error("IK: the initiator must know the responder's static key");
    if (!cfg.initiator && cfg.rs) throw new Error("IK: the responder learns rs from message 1, not up front");
    this.initiator = cfg.initiator;
    this.s = cfg.s;
    this.e = cfg.e ?? null;
    this.rs = cfg.rs ?? null;
    this.psk = cfg.psk;
    this.symmetric = new SymmetricState(utf8ToBytes(NOISE_PROTOCOL_NAME));
    this.symmetric.mixHash(cfg.prologue);
    this.afterPrologue = { ck: this.symmetric.ck.slice(), h: this.symmetric.h.slice() };
    // Pre-message "<- s": both sides hash the RESPONDER's static key, each from where it holds it.
    for (const token of IKPSK2_PATTERN.responderPreMessage) {
      if (token !== "s") throw new Error(`unexpected pre-message token ${token}`);
      this.symmetric.mixHash(this.initiator ? (this.rs as Uint8Array) : this.s.publicKey);
    }
    this.remaining = IKPSK2_PATTERN.messages.map((m) => [...m]);
  }

  /** Whose turn it is: even-numbered handshake messages are the initiator's. */
  private nextIsMine(): boolean {
    const index = IKPSK2_PATTERN.messages.length - this.remaining.length;
    return (index % 2 === 0) === this.initiator;
  }

  private mixDh(token: "ee" | "es" | "se" | "ss"): void {
    const need = (v: Uint8Array | null, name: string): Uint8Array => {
      if (!v) throw new Error(`token ${token} needs ${name}, which is not set`);
      return v;
    };
    const e = (): NoiseKeyPair => {
      if (!this.e) throw new Error(`token ${token} needs a local ephemeral`);
      return this.e;
    };
    let out: Uint8Array;
    switch (token) {
      case "ee": out = dh(e(), need(this.re, "re")); break;
      case "es": out = this.initiator ? dh(e(), need(this.rs, "rs")) : dh(this.s, need(this.re, "re")); break;
      case "se": out = this.initiator ? dh(this.s, need(this.re, "re")) : dh(e(), need(this.rs, "rs")); break;
      case "ss": out = dh(this.s, need(this.rs, "rs")); break;
    }
    this.symmetric.mixKey(out);
  }

  private finish(): TransportCiphers | undefined {
    if (this.remaining.length > 0) return undefined;
    const [c1, c2] = this.symmetric.split();
    this.transport = this.initiator ? { c1, c2, send: c1, recv: c2 } : { c1, c2, send: c2, recv: c1 };
    return this.transport;
  }

  /** WriteMessage(payload). Returns the message, and the transport ciphers after the last one. */
  writeMessage(payload: Uint8Array): { message: Uint8Array; transport?: TransportCiphers } {
    if (this.remaining.length === 0) throw new Error("handshake already complete");
    if (!this.nextIsMine()) throw new Error("not this side's turn to write");
    const tokens = this.remaining.shift() as Token[];
    const parts: Uint8Array[] = [];
    for (const token of tokens) {
      switch (token) {
        case "e": {
          this.e = this.e ?? generateKeyPair();
          parts.push(this.e.publicKey);
          this.symmetric.mixHash(this.e.publicKey);
          // Section 9.2: in a PSK handshake every "e" is ALSO followed by MixKey(e.public_key).
          // Without it the handshake still completes between two copies of this code; only the
          // vector notices.
          this.symmetric.mixKey(this.e.publicKey);
          break;
        }
        case "s":
          parts.push(this.symmetric.encryptAndHash(this.s.publicKey));
          break;
        case "psk":
          this.symmetric.mixKeyAndHash(this.psk);
          break;
        default:
          this.mixDh(token);
      }
    }
    parts.push(this.symmetric.encryptAndHash(payload));
    return { message: concatBytes(...parts), transport: this.finish() };
  }

  /** ReadMessage(message). Throws on any authentication failure; the handshake is then dead. */
  readMessage(message: Uint8Array): { payload: Uint8Array; transport?: TransportCiphers } {
    if (this.remaining.length === 0) throw new Error("handshake already complete");
    if (this.nextIsMine()) throw new Error("not this side's turn to read");
    const tokens = this.remaining.shift() as Token[];
    let offset = 0;
    const take = (n: number): Uint8Array => {
      if (offset + n > message.length) throw new Error("handshake message too short");
      const out = message.slice(offset, offset + n);
      offset += n;
      return out;
    };
    for (const token of tokens) {
      switch (token) {
        case "e": {
          this.re = take(DHLEN);
          this.symmetric.mixHash(this.re);
          this.symmetric.mixKey(this.re); // section 9.2, as in writeMessage
          break;
        }
        case "s": {
          const temp = take(this.symmetric.cipher.hasKey() ? DHLEN + TAGLEN : DHLEN);
          this.rs = this.symmetric.decryptAndHash(temp);
          break;
        }
        case "psk":
          this.symmetric.mixKeyAndHash(this.psk);
          break;
        default:
          this.mixDh(token);
      }
    }
    const payload = this.symmetric.decryptAndHash(message.slice(offset));
    return { payload, transport: this.finish() };
  }

  /** GetHandshakeHash(): meaningful only once the handshake is complete. */
  getHandshakeHash(): Uint8Array {
    if (this.remaining.length > 0) throw new Error("the handshake hash is defined only after Split()");
    return this.symmetric.h.slice();
  }

  /** The static key the initiator sent, as the responder decrypted it. */
  get remoteStatic(): Uint8Array | null {
    return this.rs;
  }

  get complete(): boolean {
    return this.transport !== null;
  }
}

// ------------------------------------------------------------------------------------------------
// The published vector, run in the tab
// ------------------------------------------------------------------------------------------------

/** One vector in the format the Noise wiki documents (cacophony's field names, verbatim). */
export interface NoiseTestVector {
  protocol_name: string;
  init_prologue: string;
  init_psks: string[];
  init_static: string;
  init_ephemeral: string;
  init_remote_static: string;
  resp_prologue: string;
  resp_psks: string[];
  resp_static: string;
  resp_ephemeral: string;
  handshake_hash: string;
  messages: { payload: string; ciphertext: string }[];
}

export interface NoiseVectorFile {
  protocolName: string;
  source: { repository: string; commit: string; url: string; sha256: string; [k: string]: unknown };
  vectors: NoiseTestVector[];
}

export interface NoiseMessageCheck {
  index: number;
  sender: "initiator" | "responder";
  phase: "handshake" | "transport";
  length: number;
  /** SHA-256 of the vector's ciphertext and of the one computed here. */
  expectedSha256: string;
  computedSha256: string;
  ciphertextMatches: boolean;
  /** The receiving side decrypted it back to the vector's payload. */
  payloadRecovered: boolean;
}

export interface NoiseVectorCheck {
  protocolName: string;
  messages: NoiseMessageCheck[];
  handshakeHash: { expected: string; initiator: string; responder: string; matches: boolean };
  /** init_remote_static really is the public key of resp_static - the vector agrees with itself. */
  remoteStaticConsistent: boolean;
  /** The responder decrypted the initiator's static key to the public key of init_static. */
  initiatorStaticRecovered: boolean;
  passed: boolean;
  error?: string;
}

export interface NoiseVectorResult {
  protocolName: string;
  source: { repository: string; commit: string; url: string; sha256: string };
  vectors: NoiseVectorCheck[];
  passed: number;
  total: number;
  allPassed: boolean;
}

const fingerprint = (b: Uint8Array): string => bytesToHex(sha256(b));

/** Run one vector: both roles, every message, in the order the vector lists them. */
export function runNoiseVector(v: NoiseTestVector): NoiseVectorCheck {
  const messages: NoiseMessageCheck[] = [];
  const check: NoiseVectorCheck = {
    protocolName: v.protocol_name,
    messages,
    handshakeHash: { expected: v.handshake_hash.toLowerCase(), initiator: "", responder: "", matches: false },
    remoteStaticConsistent: false,
    initiatorStaticRecovered: false,
    passed: false,
  };
  try {
    if (v.protocol_name !== NOISE_PROTOCOL_NAME) throw new Error(`vector is for ${v.protocol_name}`);
    const initStatic = keyPairFromPrivate(hexToBytes(v.init_static));
    const respStatic = keyPairFromPrivate(hexToBytes(v.resp_static));
    const initRemote = hexToBytes(v.init_remote_static);
    check.remoteStaticConsistent = equalBytes(initRemote, respStatic.publicKey);
    const initiator = new IKpsk2Handshake({
      initiator: true, prologue: hexToBytes(v.init_prologue), s: initStatic,
      e: keyPairFromPrivate(hexToBytes(v.init_ephemeral)), rs: initRemote, psk: hexToBytes(v.init_psks[0]),
    });
    const responder = new IKpsk2Handshake({
      initiator: false, prologue: hexToBytes(v.resp_prologue), s: respStatic,
      e: keyPairFromPrivate(hexToBytes(v.resp_ephemeral)), psk: hexToBytes(v.resp_psks[0]),
    });
    let iT: TransportCiphers | undefined;
    let rT: TransportCiphers | undefined;
    const handshakeCount = IKPSK2_PATTERN.messages.length;
    v.messages.forEach((m, index) => {
      // Handshake and transport messages alternate initiator, responder, ... in this vector format.
      const fromInitiator = index % 2 === 0;
      const payload = hexToBytes(m.payload);
      const expected = hexToBytes(m.ciphertext);
      let computed: Uint8Array;
      let recovered: Uint8Array;
      if (index < handshakeCount) {
        const [w, r] = fromInitiator ? [initiator, responder] : [responder, initiator];
        const written = w.writeMessage(payload);
        computed = written.message;
        // The RECEIVER reads the vector's bytes, not ours, so a mismatch cannot hide behind the
        // two sides agreeing with each other.
        const read = r.readMessage(expected);
        recovered = read.payload;
        if (fromInitiator) { iT = iT ?? written.transport; rT = rT ?? read.transport; }
        else { rT = rT ?? written.transport; iT = iT ?? read.transport; }
      } else {
        if (!iT || !rT) throw new Error("transport message before the handshake completed");
        const [send, recv] = fromInitiator ? [iT.send, rT.recv] : [rT.send, iT.recv];
        computed = send.encryptWithAd(EMPTY, payload);
        recovered = recv.decryptWithAd(EMPTY, expected);
      }
      messages.push({
        index,
        sender: fromInitiator ? "initiator" : "responder",
        phase: index < handshakeCount ? "handshake" : "transport",
        length: computed.length,
        expectedSha256: fingerprint(expected),
        computedSha256: fingerprint(computed),
        ciphertextMatches: equalBytes(computed, expected),
        payloadRecovered: equalBytes(recovered, payload),
      });
      if (index === 1) {
        check.handshakeHash.initiator = bytesToHex(initiator.getHandshakeHash());
        check.handshakeHash.responder = bytesToHex(responder.getHandshakeHash());
        const rs = responder.remoteStatic;
        check.initiatorStaticRecovered = rs !== null && equalBytes(rs, initStatic.publicKey);
      }
    });
    check.handshakeHash.matches = check.handshakeHash.initiator === check.handshakeHash.expected
      && check.handshakeHash.responder === check.handshakeHash.expected;
    check.passed = messages.length === v.messages.length && messages.length >= IKPSK2_PATTERN.messages.length
      && messages.every((m) => m.ciphertextMatches && m.payloadRecovered)
      && check.handshakeHash.matches && check.remoteStaticConsistent && check.initiatorStaticRecovered;
  } catch (err) {
    // A decrypt failure IS a mismatch: reported as a failed vector, never swallowed into a pass.
    check.error = err instanceof Error ? err.message : String(err);
    check.passed = false;
  }
  return check;
}

/**
 * Every bundled vector, run through the real module. The parameter exists so the CI check can hand
 * in a deliberately corrupted copy and prove this reports the corruption - a runner that cannot be
 * shown to fail is not evidence of anything.
 */
export function runNoiseVectors(file: NoiseVectorFile = bundledVectors as NoiseVectorFile): NoiseVectorResult {
  const vectors = file.vectors.filter((v) => v.protocol_name === NOISE_PROTOCOL_NAME).map(runNoiseVector);
  const passed = vectors.filter((v) => v.passed).length;
  return {
    protocolName: NOISE_PROTOCOL_NAME,
    source: { repository: file.source.repository, commit: file.source.commit, url: file.source.url, sha256: file.source.sha256 },
    vectors,
    passed,
    total: vectors.length,
    allPassed: vectors.length > 0 && passed === vectors.length,
  };
}
