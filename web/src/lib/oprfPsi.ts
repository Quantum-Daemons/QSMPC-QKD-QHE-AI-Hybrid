// Private set intersection on RFC 9497's OPRF, mode 0x00, suite ristretto255-SHA512: the browser
// twin of src/qsh/qsmpc/psi.py, on @noble/curves' `ristretto255_oprf` (RFC 9497 over RFC 9496's
// ristretto255 group). For the same key, blinds and inputs the two tiers produce byte-identical
// blinded elements, evaluated elements and outputs, and they derive the same transcript keys from the
// same session key (tests/test_r84_oprf_psi.py, which also runs RFC 9497 Appendix A.1.1 here).
//
// THE PROTOCOL, one run. Bank A is the OPRF client and Bank B the OPRF server.
//   1. Bank B draws a fresh OPRF key k for the run (GenerateKeyPair, RFC 9497 section 3.2).
//   2. Bank A blinds each of its IDs x, M_x = r_x * HashToGroup(x) with a fresh random r_x (Blind,
//      section 3.3.1), and sends the M_x.
//   3. Bank B returns Z_x = k * M_x for each (BlindEvaluate) and, for each of its own IDs y, the PRF
//      output F_k(y) (Evaluate), in an order shuffled with the CSPRNG, so the order says nothing.
//   4. Bank A computes F_k(x) = Finalize(x, r_x, Z_x) and keeps the x whose output Bank B sent.
//
// WHAT EACH BANK LEARNS, when both follow the protocol (semi-honest). Bank A: the intersection and
// |B|. It never holds k, so it cannot compute F_k offline: it can test an ID against Bank B's list
// only by having Bank B answer a query for it. Bank B: |A|, and nothing else about Bank A's IDs,
// since a blinded element is uniform whatever the ID, even to an unbounded adversary (RFC 9497
// section 7.1, "Unconditional input secrecy"). Bank B does not learn the intersection.
//
// WHAT IT DOES NOT PROVIDE. No malicious security: Bank A could blind candidates it does not hold (up
// to |A| a run), Bank B could answer under another key or send made-up outputs, and mode 0x00 has no
// proof that would catch either. Not post-quantum: F_k's pseudorandomness, which is what keeps Bank
// B's list from Bank A, rests on the One-More Gap Computational Diffie-Hellman assumption in
// ristretto255 (RFC 9497 section 7.2.1), which a quantum computer breaks. Both banks run in this one
// tab, so they are notional parties and no message crosses a network.
//
// THE SESSION KEY. The three messages are sealed with AES-256-GCM (WebCrypto, as clientMcp.ts seals
// its records) under two keys HKDF-SHA256 derives from the PQC session key, one per direction, as
// psi.py seals them with the transport's `SecureChannel`. That keeps the transcript from anyone but
// the two banks for as long as the ML-KEM + X25519 hybrid holds, including someone who records it now
// and attacks ristretto255 later. It hides nothing between the banks, each of which opens what the
// other sealed. In one tab the sealing shows the path the messages would take; there is no wire to
// protect.
//
// The key k is fresh per run: rotating the server's key is one of the mitigations RFC 9497 section
// 7.2.3 names for the static Diffie-Hellman attack (Q BlindEvaluate queries to one key cost about
// log2(Q)/2 bits of the group's security against the best-known attacks).
import { ristretto255, ristretto255_hasher, ristretto255_oprf } from "@noble/curves/ed25519.js";
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { randomBelow } from "./csprng";

/** The protocol, as the run reports it (`psi_protocol`); psi.py's `PROTOCOL` is the same string. */
export const PSI_PROTOCOL = "DH-OPRF PSI (RFC 9497 OPRF, mode 0x00, ristretto255-SHA512), semi-honest";

const enc = new TextEncoder();
/** Ne and Nh for ristretto255-SHA512 (RFC 9497 section 4.1): an element is 32 bytes, an output 64. */
const NE = 32;
const NH = 64;

function concat(...parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

/** CreateContextString(modeOPRF, "ristretto255-SHA512"), RFC 9497 section 3.1. */
export const OPRF_CONTEXT = concat(enc.encode("OPRFV1-"), Uint8Array.of(0x00), enc.encode("-ristretto255-SHA512"));

// noble 2.3.0 implements Evaluate (RFC 9497 section 3.3.1) for mode 0x00 but does not declare it in
// its types; oprfPSI checks that it is there before relying on it.
type OprfMode = typeof ristretto255_oprf.oprf & {
  evaluate?: (secretKey: Uint8Array, input: Uint8Array) => Uint8Array;
};
/** RFC 9497's mode 0x00 for ristretto255-SHA512, as @noble/curves implements it. */
export const oprf = ristretto255_oprf.oprf as OprfMode;

/** Blind with a GIVEN blind scalar, as RFC 9497's test vectors fix it (Appendix A.1.1): the same
 *  HashToGroup and DST ("HashToGroup-" || contextString, section 4.1) as `oprf.blind`, which a run
 *  uses and which draws the blind from the CSPRNG. psi.py's `oprf.blind(data, blind_scalar)` twin. */
export function blindWith(input: Uint8Array, blind: Uint8Array): Uint8Array {
  const point = ristretto255_hasher.hashToCurve(input, { DST: concat(enc.encode("HashToGroup-"), OPRF_CONTEXT) });
  if (point.equals(ristretto255.Point.ZERO)) throw new Error("the input maps to the group identity");
  return point.multiply(ristretto255.Point.Fn.fromBytes(blind)).toBytes();
}

const INFO_A_TO_B = enc.encode("qsh/qsmpc/oprf-psi/bank-a-to-bank-b");
const INFO_B_TO_A = enc.encode("qsh/qsmpc/oprf-psi/bank-b-to-bank-a");
const AAD_BLINDED = enc.encode("qsh-psi/blinded-elements");
const AAD_EVALUATED = enc.encode("qsh-psi/evaluated-elements");
const AAD_OUTPUTS = enc.encode("qsh-psi/server-outputs");

/** The two AES-256-GCM keys the transcript is sealed under: HKDF-SHA256 (RFC 5869) of the PQC
 *  session key, no salt, one info string per direction (psi.py's `transcript_keys`). */
export function psiTranscriptKeys(sessionKey: Uint8Array): { aToB: Uint8Array; bToA: Uint8Array } {
  if (!sessionKey || sessionKey.length === 0) {
    throw new Error("the OPRF PSI is sealed under the PQC session key, and none was given");
  }
  return { aToB: hkdf(sha256, sessionKey, undefined, INFO_A_TO_B, 32),
           bToA: hkdf(sha256, sessionKey, undefined, INFO_B_TO_A, 32) };
}

/** A copy in its own ArrayBuffer, which is the BufferSource WebCrypto's typings accept. */
const ab = (u: Uint8Array): ArrayBuffer => { const b = new ArrayBuffer(u.byteLength); new Uint8Array(b).set(u); return b; };
const gcmKey = (key: Uint8Array) => crypto.subtle.importKey("raw", ab(key), "AES-GCM", false, ["encrypt", "decrypt"]);

/** One sealed record: nonce || AES-256-GCM(ciphertext || tag), the nonce the 12-byte big-endian
 *  `counter`. The layout of psi.py's `SecureChannel.seal`, so either tier opens the other's records. */
export async function sealPsiRecord(key: Uint8Array, plaintext: Uint8Array, aad: Uint8Array,
  counter: number): Promise<Uint8Array> {
  const nonce = new Uint8Array(12);
  new DataView(nonce.buffer).setUint32(8, counter);
  const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv: nonce, additionalData: ab(aad) },
    await gcmKey(key), ab(plaintext));
  return concat(nonce, new Uint8Array(ct));
}
/** Open a record `sealPsiRecord` (or psi.py's `SecureChannel`) sealed; a wrong key, a changed byte
 *  or other associated data rejects. */
export async function openPsiRecord(key: Uint8Array, record: Uint8Array, aad: Uint8Array): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: ab(record.subarray(0, 12)), additionalData: ab(aad) }, await gcmKey(key),
    ab(record.subarray(12))));
}
/** A sealing direction: one key, its counter from zero. */
function sealer(key: Uint8Array) {
  let counter = 0;
  return (plaintext: Uint8Array, aad: Uint8Array) => sealPsiRecord(key, plaintext, aad, counter++);
}
const openRecord = openPsiRecord;

function split(blob: Uint8Array, width: number): Uint8Array[] {
  if (blob.length % width !== 0) {
    throw new Error(`a sealed PSI message of ${blob.length} bytes is not a whole number of ${width}-byte items`);
  }
  return Array.from({ length: blob.length / width }, (_, i) => blob.slice(i * width, (i + 1) * width));
}
const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
/** Distinct IDs in first-seen order: the protocol is over sets. */
const distinct = (ids: string[]) => [...new Set(ids.map(String))];
// Each operation is about a millisecond in node (a scalar multiplication), so a large set would hold
// the page's thread for seconds. Yield every so often so the page keeps painting.
const YIELD_EVERY = 64;
const tick = () => new Promise<void>((done) => setTimeout(done, 0));

export interface OprfPsiResult {
  /** Bank A's view: its IDs that Bank B also holds. */
  intersection: string[];
  size_a: number;
  size_b: number;
  intersection_size: number;
  /** BlindEvaluate calls Bank B answered: one per ID Bank A blinded, so |A|. */
  blind_evaluations: number;
  protocol: string;
  /** Sealed bytes each way, which is all a third party on the channel would see. */
  transcript_bytes: { bank_a_to_bank_b: number; bank_b_to_bank_a: number };
}

/** Bank A, the OPRF client: blinds its IDs, then finalizes Bank B's answers and intersects. */
function bankA(ids: string[]) {
  const blinds: Uint8Array[] = [];
  return {
    async query(): Promise<Uint8Array[]> {
      const out: Uint8Array[] = [];
      for (let i = 0; i < ids.length; i++) {
        const { blind, blinded } = oprf.blind(enc.encode(ids[i]));
        blinds.push(blind);
        out.push(blinded);
        if (i % YIELD_EVERY === YIELD_EVERY - 1) await tick();
      }
      return out;
    },
    async intersect(evaluated: Uint8Array[], theirs: Uint8Array[]): Promise<string[]> {
      if (evaluated.length !== ids.length) throw new Error(`Bank B answered ${evaluated.length} of ${ids.length} blinded elements`);
      const outputs = new Set(theirs.map(hex));
      const inter: string[] = [];
      for (let i = 0; i < ids.length; i++) {
        if (outputs.has(hex(oprf.finalize(enc.encode(ids[i]), blinds[i], evaluated[i])))) inter.push(ids[i]);
        if (i % YIELD_EVERY === YIELD_EVERY - 1) await tick();
      }
      return inter;
    },
  };
}

/** Bank B, the OPRF server: holds the run's key and sees only blinded elements, never Bank A's IDs. */
function bankB(ids: string[], evaluate: (secretKey: Uint8Array, input: Uint8Array) => Uint8Array) {
  const { secretKey } = oprf.generateKeyPair();
  return {
    async answer(blinded: Uint8Array[]): Promise<Uint8Array[]> {
      const out: Uint8Array[] = [];
      for (let i = 0; i < blinded.length; i++) {
        out.push(oprf.blindEvaluate(secretKey, blinded[i]));
        if (i % YIELD_EVERY === YIELD_EVERY - 1) await tick();
      }
      return out;
    },
    async ownOutputs(): Promise<Uint8Array[]> {
      const out: Uint8Array[] = [];
      for (let i = 0; i < ids.length; i++) {
        out.push(evaluate(secretKey, enc.encode(ids[i])));
        if (i % YIELD_EVERY === YIELD_EVERY - 1) await tick();
      }
      // Fisher-Yates with the CSPRNG: the seeded generator the page shows would let Bank A undo it.
      for (let i = out.length - 1; i > 0; i--) {
        const j = randomBelow(i + 1);
        [out[i], out[j]] = [out[j], out[i]];
      }
      return out;
    },
  };
}

/** Run the OPRF PSI between Bank A (`setA`) and Bank B (`setB`), its messages sealed under `sessionKey`. */
export async function oprfPSI(setA: string[], setB: string[], sessionKey: Uint8Array): Promise<OprfPsiResult> {
  const evaluate = oprf.evaluate;
  if (typeof evaluate !== "function") {
    throw new Error("@noble/curves' ristretto255_oprf exposes no Evaluate (RFC 9497 section 3.3.1), which Bank B needs");
  }
  const { aToB, bToA } = psiTranscriptKeys(sessionKey);
  const aSeals = sealer(aToB), bSeals = sealer(bToA);
  const idsA = distinct(setA);
  const a = bankA(idsA), b = bankB(distinct(setB), evaluate);

  // Bank A to Bank B: the blinded elements, and nothing else.
  const msgBlinded = await aSeals(concat(...await a.query()), AAD_BLINDED);
  const blinded = split(await openRecord(aToB, msgBlinded, AAD_BLINDED), NE);
  // Bank B to Bank A: its evaluations of them, then its own outputs in shuffled order.
  const msgEvaluated = await bSeals(concat(...await b.answer(blinded)), AAD_EVALUATED);
  const msgOutputs = await bSeals(concat(...await b.ownOutputs()), AAD_OUTPUTS);
  const evaluated = split(await openRecord(bToA, msgEvaluated, AAD_EVALUATED), NE);
  const outputs = split(await openRecord(bToA, msgOutputs, AAD_OUTPUTS), NH);
  const intersection = await a.intersect(evaluated, outputs);
  return {
    intersection, size_a: idsA.length, size_b: outputs.length, intersection_size: intersection.length,
    blind_evaluations: blinded.length, protocol: PSI_PROTOCOL,
    transcript_bytes: { bank_a_to_bank_b: msgBlinded.length, bank_b_to_bank_a: msgEvaluated.length + msgOutputs.length },
  };
}

/** What each party learned in a run, in the words the run log uses (psi.py's `learned_summary`):
 *  `a` is the OPRF client and `b` the server, the banks in the pipeline and Alice and Bob in the
 *  two-party demo. */
export function psiLearnedSummary(r: OprfPsiResult, a = "Bank A", b = "Bank B"): string {
  return `${a} learns the intersection (${r.intersection_size}) and |B| = ${r.size_b}; `
    + `${b} learns |A| = ${r.size_a} from the ${r.blind_evaluations} blinded elements it evaluated, `
    + `and nothing else about ${a}'s IDs`;
}
