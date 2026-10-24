// The IKEv2 key schedule with one post-quantum additional key exchange and a PPK, both peers in one tab.
//
// EVERY FORMULA, CONFIRMED AGAINST THE RFC TEXT (rfc-editor.org .txt, retrieved 2026-10-24)
//
//   RFC 7296 2.13   prf+(K, S) = T1 | T2 | ...  with T1 = prf(K, S | 0x01), Tn = prf(K, Tn-1 | S | n);
//                   "not defined beyond 255 times the size of the prf function output"
//   RFC 7296 2.14   SKEYSEED = prf(Ni | Nr, g^ir)
//                   {SK_d | SK_ai | SK_ar | SK_ei | SK_er | SK_pi | SK_pr} = prf+(SKEYSEED, Ni | Nr | SPIi | SPIr)
//   RFC 7296 2.17   KEYMAT = prf+(SK_d, Ni | Nr), or prf+(SK_d, g^ir (new) | Ni | Nr) with a new DH
//   RFC 7296 2.18   SKEYSEED = prf(SK_d (old), g^ir (new) | Ni | Nr)               (rekey)
//   RFC 5903 7      ECP key exchange data is x | y; the shared secret is the x value of the common point
//   RFC 9370 2.2.2  SKEYSEED(n) = prf(SK_d(n-1), SK(n) | Ni | Nr)
//                   {SK_d(n) | SK_ai(n) | ... | SK_pr(n)} = prf+(SKEYSEED(n), Ni | Nr | SPIi | SPIr)
//   RFC 8784 3      SK_d = prf+(PPK, SK_d'),  SK_pi = prf+(PPK, SK_pi'),  SK_pr = prf+(PPK, SK_pr')
//
// THE PRIMES IN RFC 8784 POINT THE OTHER WAY FROM WHAT ONE WOULD GUESS. SK_d' is the key the
// standard derivation PRODUCED, and unprimed SK_d is the PPK-mixed result. The math is the same
// either way; the names here follow the RFC (`beforePpk` holds the primed keys). SK_e* and SK_a* are
// deliberately not touched - RFC 8784 mixes only the three keys that feed Child SAs and AUTH.
//
// WHERE THE PPK GOES WHEN BOTH LAYERS ARE ON - A CHOICE, STATED AS ONE
//
// Neither RFC 8784 nor RFC 9370 defines their combination. This applies RFC 8784's three prf+ calls
// to the keys of the LAST RFC 9370 round, because those are the keys IKE_AUTH uses (RFC 9370 2.2.2:
// "use these updated key values in the next exchange (IKE_INTERMEDIATE or IKE_AUTH)"), and RFC 8784
// mixes the PPK at IKE_AUTH. RFC 9867 (November 2025) is consistent with that: its Table 1 keeps
// RFC 8784 available alongside IKE_INTERMEDIATE, a responder must pick one of the two PPK methods,
// and for its own method (SKEYSEED' = prf+(PPK, SK_d), then all seven keys re-derived) section 3.1.1
// requires the PPK to be applied after every RFC 9370 recalculation. RFC 9867's method is not
// implemented here.
//
// WHAT IS VALIDATED AGAINST AN OFFICIAL VECTOR, AND WHAT IS NOT
//
//   prf, prf+, SKEYSEED, the SK_* expansion, Child SA KEYMAT, rekey SKEYSEED
//       NIST CAVP SP 800-135 IKEv2 KDF vectors (vectors/ikev2_cavp.json; HMAC-SHA-384 x4, -256, -512)
//   ECP-256 g^ir (RFC 5903 section 7 encoding)
//       RFC 5903 section 8.1's own example, i / r / g^i / g^r / girx
//   ML-KEM-768
//       NIST ACVP, through the same @noble ml_kem768 (see mlkemKat.ts)
//   The RFC 9370 layer and the RFC 8784 layer
//       NO official vectors exist for either. They are computed per the RFC text above, from the
//       validated primitives: SKEYSEED(n) is the same prf-over-(secret | Ni | Nr) function CAVP's
//       SKEYSEED(Rekey) field pins, and both layers expand with the CAVP-pinned prf+. What no vector
//       checks is the COMPOSITION - which SK_d feeds which round, which nonces, the argument order
//       of prf+(PPK, .). `runIkev2Cavp()` and the demo result both say so in `notValidated`.
//
// ML-KEM IN IKEV2 IS NOT YET AN RFC. draft-ietf-ipsecme-ikev2-mlkem-09 (2026-07-05): IESG state "RFC
// Ed Queue" on the IETF datatracker when checked 2026-10-24; IANA already lists ml-kem-768 as Key
// Exchange Method 36 with the draft as reference. Its section 2: the 32-byte shared secret "is used
// without padding like the traditional shared g^ir value" - which is exactly SK(1) below. Sources,
// hashes and the validated-vs-computed table: references/net-key-schedules.md.
import { hmac } from "@noble/hashes/hmac.js";
import { sha256, sha384, sha512 } from "@noble/hashes/sha2.js";
import { bytesToHex, concatBytes, hexToBytes, utf8ToBytes } from "@noble/hashes/utils.js";
import { equalBytes } from "@noble/ciphers/utils.js";
import { p256 } from "@noble/curves/nist.js";
import { ml_kem768 } from "@noble/post-quantum/ml-kem.js";
import bundledVectors from "./vectors/ikev2_cavp.json";

// ------------------------------------------------------------------------------------------------
// prf and prf+ (RFC 7296 section 2.13)
// ------------------------------------------------------------------------------------------------

/** The HMAC PRFs of RFC 4868 that this module accepts; each one has a CAVP case in the vector file. */
export type IkePrf = "HMAC-SHA-256" | "HMAC-SHA-384" | "HMAC-SHA-512";

const HASHES = { "HMAC-SHA-256": sha256, "HMAC-SHA-384": sha384, "HMAC-SHA-512": sha512 } as const;

/**
 * Output length, which RFC 7296 2.13 also makes the "preferred key size" for HMAC PRFs - and that
 * is the length SK_d, SK_pi and SK_pr MUST have (2.14).
 */
export function prfLength(alg: IkePrf): number {
  return HASHES[alg].outputLen;
}

export function prf(alg: IkePrf, key: Uint8Array, data: Uint8Array): Uint8Array {
  const hash = HASHES[alg];
  if (!hash) throw new Error(`unsupported PRF ${String(alg)}`);
  return hmac(hash, key, data);
}

/** prf+(K, S), truncated to `length` octets: "keys are taken from the output string without regard to boundaries". */
export function prfPlus(alg: IkePrf, key: Uint8Array, seed: Uint8Array, length: number): Uint8Array {
  const block = prfLength(alg);
  if (!Number.isInteger(length) || length < 0) throw new Error("prf+ length must be a non-negative integer");
  if (length > 255 * block) throw new Error("prf+ is not defined beyond 255 blocks (RFC 7296 section 2.13)");
  const out = new Uint8Array(length);
  let t: Uint8Array = new Uint8Array(0);
  for (let i = 1, offset = 0; offset < length; i += 1, offset += block) {
    t = prf(alg, key, concatBytes(t, seed, Uint8Array.of(i)));
    out.set(t.subarray(0, Math.min(block, length - offset)), offset);
  }
  return out;
}

// ------------------------------------------------------------------------------------------------
// RFC 7296 sections 2.14, 2.17, 2.18
// ------------------------------------------------------------------------------------------------

/** 2.14: SKEYSEED = prf(Ni | Nr, g^ir). The NONCES are the HMAC key; g^ir is the message. */
export function computeSkeyseed(alg: IkePrf, ni: Uint8Array, nr: Uint8Array, gir: Uint8Array): Uint8Array {
  return prf(alg, concatBytes(ni, nr), gir);
}

/**
 * prf(SK_d, secret | Ni | Nr) - ONE function for two formulas with one shape: 2.18's rekey SKEYSEED
 * (secret = g^ir (new)) and RFC 9370's SKEYSEED(n) (secret = SK(n)). Sharing it is what lets CAVP's
 * SKEYSEED(Rekey) field pin the RFC 9370 step's arithmetic, though not its composition.
 */
function prfSkdOverSecret(alg: IkePrf, skd: Uint8Array, secret: Uint8Array, ni: Uint8Array, nr: Uint8Array): Uint8Array {
  return prf(alg, skd, concatBytes(secret, ni, nr));
}

/** 2.18: SKEYSEED = prf(SK_d (old), g^ir (new) | Ni | Nr). */
export function rekeySkeyseed(alg: IkePrf, skdOld: Uint8Array, girNew: Uint8Array, ni: Uint8Array, nr: Uint8Array): Uint8Array {
  return prfSkdOverSecret(alg, skdOld, girNew, ni, nr);
}

/** 2.17: KEYMAT = prf+(SK_d, Ni | Nr), or prf+(SK_d, g^ir (new) | Ni | Nr) when the exchange has a new DH. */
export function childSaKeymat(alg: IkePrf, skd: Uint8Array, ni: Uint8Array, nr: Uint8Array, length: number, girNew?: Uint8Array): Uint8Array {
  return prfPlus(alg, skd, girNew ? concatBytes(girNew, ni, nr) : concatBytes(ni, nr), length);
}

/** Octet lengths of the seven keys, in the order 2.14 takes them from prf+. */
export interface SkLengths { d: number; ai: number; ar: number; ei: number; er: number; pi: number; pr: number }

export interface IkeSaKeys {
  SK_d: Uint8Array; SK_ai: Uint8Array; SK_ar: Uint8Array; SK_ei: Uint8Array; SK_er: Uint8Array; SK_pi: Uint8Array; SK_pr: Uint8Array;
}

export const SK_NAMES = ["SK_d", "SK_ai", "SK_ar", "SK_ei", "SK_er", "SK_pi", "SK_pr"] as const;
export type SkName = (typeof SK_NAMES)[number];

/** 2.14: {SK_d | SK_ai | SK_ar | SK_ei | SK_er | SK_pi | SK_pr} = prf+(SKEYSEED, Ni | Nr | SPIi | SPIr). */
export function deriveIkeSaKeys(
  alg: IkePrf, skeyseed: Uint8Array, ni: Uint8Array, nr: Uint8Array, spii: Uint8Array, spir: Uint8Array, len: SkLengths,
): IkeSaKeys {
  const block = prfLength(alg);
  if (len.d !== block || len.pi !== block || len.pr !== block) {
    throw new Error(`SK_d, SK_pi and SK_pr MUST be the PRF's preferred key length, ${block} octets (RFC 7296 section 2.14)`);
  }
  const order = [len.d, len.ai, len.ar, len.ei, len.er, len.pi, len.pr];
  const stream = prfPlus(alg, skeyseed, concatBytes(ni, nr, spii, spir), order.reduce((a, b) => a + b, 0));
  const parts: Uint8Array[] = [];
  let offset = 0;
  for (const n of order) { parts.push(stream.slice(offset, offset + n)); offset += n; }
  const [SK_d, SK_ai, SK_ar, SK_ei, SK_er, SK_pi, SK_pr] = parts;
  return { SK_d, SK_ai, SK_ar, SK_ei, SK_er, SK_pi, SK_pr };
}

// ------------------------------------------------------------------------------------------------
// RFC 9370 section 2.2.2 and RFC 8784 section 3 - computed per the RFC text, no official vectors
// ------------------------------------------------------------------------------------------------

/** One IKE_INTERMEDIATE additional key exchange: SK(n) folded in, all seven keys re-derived. */
export function additionalKeyExchange(
  alg: IkePrf, previous: IkeSaKeys, skN: Uint8Array, ni: Uint8Array, nr: Uint8Array, spii: Uint8Array, spir: Uint8Array, len: SkLengths,
): { skeyseed: Uint8Array; keys: IkeSaKeys } {
  // Ni and Nr are the IKE_SA_INIT nonces in EVERY round (RFC 9370 2.2.2), not fresh ones.
  const skeyseed = prfSkdOverSecret(alg, previous.SK_d, skN, ni, nr);
  return { skeyseed, keys: deriveIkeSaKeys(alg, skeyseed, ni, nr, spii, spir, len) };
}

/**
 * RFC 8784 section 3: SK_d = prf+(PPK, SK_d'), SK_pi = prf+(PPK, SK_pi'), SK_pr = prf+(PPK, SK_pr').
 * The PPK is the prf+ KEY and the old key the seed; output lengths equal the inputs'.
 */
export function mixPpk(alg: IkePrf, beforePpk: IkeSaKeys, ppk: Uint8Array): IkeSaKeys {
  if (ppk.length === 0) throw new Error("PPK must not be empty");
  return {
    ...beforePpk,
    SK_d: prfPlus(alg, ppk, beforePpk.SK_d, beforePpk.SK_d.length),
    SK_pi: prfPlus(alg, ppk, beforePpk.SK_pi, beforePpk.SK_pi.length),
    SK_pr: prfPlus(alg, ppk, beforePpk.SK_pr, beforePpk.SK_pr.length),
  };
}

// ------------------------------------------------------------------------------------------------
// RFC 5903: the 256-bit random ECP group (Key Exchange Method 19)
// ------------------------------------------------------------------------------------------------

/** Section 7: the KE payload's data is x | y, each 32 octets - no 0x04 point-format prefix. */
export function ecp256KeData(secretKey: Uint8Array): Uint8Array {
  return p256.getPublicKey(secretKey, false).slice(1);
}

/** Section 7: g^ir is the x value of the common point. noble wants the 0x04 prefix back on the peer's x | y. */
export function ecp256SharedSecret(secretKey: Uint8Array, peerKeData: Uint8Array): Uint8Array {
  if (peerKeData.length !== 64) throw new Error("ECP-256 KE data must be 64 octets (x | y)");
  return p256.getSharedSecret(secretKey, concatBytes(Uint8Array.of(0x04), peerKeData), false).slice(1, 33);
}

// ------------------------------------------------------------------------------------------------
// The official vectors, run in the tab
// ------------------------------------------------------------------------------------------------

export interface Ikev2CavpCase {
  id: string;
  prf: string;
  section: { "g^ir length": number; "Ni length": number; "Nr length": number; "DKM length": number; "Child SA DKM length": number };
  COUNT: number;
  Ni: string; Nr: string; "g^ir": string; "g^ir (new)": string; SPIi: string; SPIr: string;
  SKEYSEED: string; DKM: string; "DKM(Child SA)": string; "DKM(Child SA D-H)": string; "SKEYSEED(Rekey)": string;
}

export interface Ecp256Case { id: string; i: string; gix: string; giy: string; r: string; grx: string; gry: string; girx: string; giry: string }

export interface Ikev2VectorFile {
  source: { url: string; sha256: string; retrieved: string; page: string; [k: string]: unknown };
  cases: Ikev2CavpCase[];
  ecp256Rfc5903: { source: { url: string; sha256: string; section: string; [k: string]: unknown }; cases: Ecp256Case[] };
}

export interface ValueCheck {
  name: string;
  length: number;
  expectedSha256: string;
  computedSha256: string;
  matches: boolean;
}

export interface Ikev2CaseResult { id: string; prf: string; checks: ValueCheck[]; passed: boolean; error?: string }

/** Statements a page must show next to any RFC 9370 / RFC 8784 output. */
export const IKEV2_NOT_VALIDATED = [
  "RFC 9370 additional key exchange (SKEYSEED(n), SK_*(n)): no official test vectors exist; computed per RFC 9370 section 2.2.2 from CAVP-validated prf and prf+.",
  "RFC 8784 PPK mixing (SK_d, SK_pi, SK_pr): no official test vectors exist; computed per RFC 8784 section 3 from the CAVP-validated prf+.",
  "The composition of the two (PPK applied after the last RFC 9370 round) is this module's reading; neither RFC defines the combination.",
] as const;

export interface Ikev2CavpResult {
  source: { url: string; sha256: string; retrieved: string; page: string };
  cases: Ikev2CaseResult[];
  ecp256: { source: { url: string; sha256: string; section: string }; cases: Ikev2CaseResult[] };
  passed: number;
  total: number;
  allPassed: boolean;
  notValidated: readonly string[];
}

const sha = (b: Uint8Array): string => bytesToHex(sha256(b));

function check(name: string, expectedHex: string, computed: Uint8Array): ValueCheck {
  const expected = hexToBytes(expectedHex);
  return { name, length: computed.length, expectedSha256: sha(expected), computedSha256: sha(computed), matches: equalBytes(expected, computed) };
}

function asPrf(name: string): IkePrf {
  if (name === "HMAC-SHA-256" || name === "HMAC-SHA-384" || name === "HMAC-SHA-512") return name;
  throw new Error(`no PRF ${name} in this module`);
}

function guarded(id: string, prfName: string, body: () => ValueCheck[]): Ikev2CaseResult {
  try {
    const checks = body();
    return { id, prf: prfName, checks, passed: checks.length > 0 && checks.every((c) => c.matches) };
  } catch (err) {
    return { id, prf: prfName, checks: [], passed: false, error: err instanceof Error ? err.message : String(err) };
  }
}

/** One CAVP case, every output field. Lengths in the .rsp section headers are BITS. */
export function runIkev2CavpCase(c: Ikev2CavpCase): Ikev2CaseResult {
  return guarded(c.id, c.prf, () => {
    const alg = asPrf(c.prf);
    const [ni, nr, gir, girNew, spii, spir] = [c.Ni, c.Nr, c["g^ir"], c["g^ir (new)"], c.SPIi, c.SPIr].map(hexToBytes);
    const skeyseed = computeSkeyseed(alg, ni, nr, gir);
    const dkm = prfPlus(alg, skeyseed, concatBytes(ni, nr, spii, spir), c.section["DKM length"] / 8);
    // ASKDFVS 6.3 / RFC 7296 2.14: SK_d is the first prf-length octets of the expansion.
    const skd = dkm.slice(0, prfLength(alg));
    const childLen = c.section["Child SA DKM length"] / 8;
    return [
      check("SKEYSEED", c.SKEYSEED, skeyseed),
      check("DKM", c.DKM, dkm),
      check("DKM(Child SA)", c["DKM(Child SA)"], childSaKeymat(alg, skd, ni, nr, childLen)),
      check("DKM(Child SA D-H)", c["DKM(Child SA D-H)"], childSaKeymat(alg, skd, ni, nr, childLen, girNew)),
      check("SKEYSEED(Rekey)", c["SKEYSEED(Rekey)"], rekeySkeyseed(alg, skd, girNew, ni, nr)),
    ];
  });
}

/** RFC 5903 section 8.1: both public values and g^ir computed from EACH side. */
export function runEcp256Case(c: Ecp256Case): Ikev2CaseResult {
  return guarded(c.id, "ECP-256 (group 19)", () => {
    const i = hexToBytes(c.i);
    const r = hexToBytes(c.r);
    const gi = ecp256KeData(i);
    const gr = ecp256KeData(r);
    return [
      check("g^i (x | y)", c.gix + c.giy, gi),
      check("g^r (x | y)", c.grx + c.gry, gr),
      check("g^ir, initiator side", c.girx, ecp256SharedSecret(i, hexToBytes(c.grx + c.gry))),
      check("g^ir, responder side", c.girx, ecp256SharedSecret(r, hexToBytes(c.gix + c.giy))),
    ];
  });
}

/** Every stored vector. `file` exists so CI can hand in a corrupted copy and require a failure. */
export function runIkev2Cavp(file: Ikev2VectorFile = bundledVectors as Ikev2VectorFile): Ikev2CavpResult {
  const cases = file.cases.map(runIkev2CavpCase);
  const ecp = file.ecp256Rfc5903.cases.map(runEcp256Case);
  const all = [...cases, ...ecp];
  const passed = all.filter((c) => c.passed).length;
  return {
    source: { url: file.source.url, sha256: file.source.sha256, retrieved: file.source.retrieved, page: file.source.page },
    cases,
    ecp256: { source: { url: file.ecp256Rfc5903.source.url, sha256: file.ecp256Rfc5903.source.sha256, section: file.ecp256Rfc5903.source.section }, cases: ecp },
    passed,
    total: all.length,
    allPassed: cases.length > 0 && ecp.length > 0 && passed === all.length,
    notValidated: IKEV2_NOT_VALIDATED,
  };
}

// ------------------------------------------------------------------------------------------------
// The demo flow: IKE_SA_INIT (ECP-256) -> IKE_INTERMEDIATE (ML-KEM-768) -> PPK, both peers
// ------------------------------------------------------------------------------------------------

/**
 * The negotiated suite. AES-GCM is an AEAD, so RFC 5282 section 7.1 makes SK_ai/SK_ar zero octets and
 * each SK_e the RFC 4106 section 8.1 "KEYMAT requested" for a 256-bit key: 32 key + 4 salt = 36.
 */
export const IKEV2_DEMO_SUITE = {
  encryption: { name: "ENCR_AES_GCM_16", transformId: 20, keyBits: 256, skOctets: 36, ref: "RFC 5282 section 7.1; RFC 4106 section 8.1" },
  prf: { name: "PRF_HMAC_SHA2_384", transformId: 6, alg: "HMAC-SHA-384" as IkePrf, ref: "RFC 4868" },
  integrity: { name: "none (AEAD)", skOctets: 0, ref: "RFC 5282 section 7.1" },
  keyExchange: { name: "256-bit random ECP group", methodId: 19, ref: "RFC 5903" },
  additionalKeyExchange1: {
    name: "ml-kem-768", methodId: 36,
    ref: "draft-ietf-ipsecme-ikev2-mlkem-09 (IANA-listed; not yet an RFC)", exchange: "IKE_INTERMEDIATE (RFC 9242, RFC 9370)",
  },
} as const;

export const IKEV2_DEMO_LENGTHS: SkLengths = { d: 48, ai: 0, ar: 0, ei: 36, er: 36, pi: 48, pr: 48 };

/** Recorded with the date it was checked, so a page never states it as timeless. */
export const MLKEM_IKEV2_STATUS = {
  document: "draft-ietf-ipsecme-ikev2-mlkem-09",
  revisionDate: "2026-07-05",
  checked: "2026-10-24",
  state: "IESG state 'RFC Ed Queue', intended status Proposed Standard; not yet published as an RFC",
  url: "https://datatracker.ietf.org/doc/draft-ietf-ipsecme-ikev2-mlkem/",
  ianaKeyExchangeMethod: 36,
} as const;

export interface Ikev2DemoOptions {
  /** Any string or bytes; nonces, SPIs and every secret key are derived from it. */
  seed?: string | Uint8Array;
  /** Hex or bytes. Omitted: a 32-byte demo PPK derived from the seed. `null`: no RFC 8784 layer. */
  ppk?: Uint8Array | string | null;
}

export type KeyFingerprints = Record<SkName, { length: number; fingerprint: string }>;

export interface Ikev2Stage {
  name: string;
  rfc: string;
  formula: string;
  validated: boolean;
  validatedBy: string;
  /** SHA-256 prefix of SKEYSEED, where the stage computes one. */
  skeyseedFingerprint?: string;
  keys: KeyFingerprints;
  /** Initiator and responder derived these seven keys independently and they are identical. */
  peersAgree: boolean;
  /** Non-empty keys that differ from the previous stage; for the first stage, every non-empty key it derives. */
  changedFromPrevious: SkName[];
}

export interface Ikev2KeyScheduleResult {
  suite: typeof IKEV2_DEMO_SUITE;
  mlkemStatus: typeof MLKEM_IKEV2_STATUS;
  exchange: {
    nonceOctets: number;
    spiOctets: number;
    keiOctets: number;
    kerOctets: number;
    kei1Octets: number;
    ker1Octets: number;
    /** g^ir as computed by each side (RFC 5903 x coordinate). */
    girAgrees: boolean;
    /** SK(1): the ML-KEM shared secret from Encaps (responder) and Decaps (initiator). */
    mlkemSecretAgrees: boolean;
  };
  stages: Ikev2Stage[];
  /** RFC 8784 changes exactly SK_d, SK_pi, SK_pr; RFC 9370 re-derives all seven. */
  properties: { ppkChangesOnlyDPiPr: boolean | null; additionalKeyExchangeChangesAll: boolean };
  ppk: { used: boolean; fingerprint: string | null; derivedFromSeed: boolean };
  validation: { cavpAllPassed: boolean; cavpCases: number; rfc5903Passed: boolean };
  notValidated: readonly string[];
  notes: string[];
}

const fp16 = (b: Uint8Array): string => sha(b).slice(0, 16);

function fingerprints(k: IkeSaKeys): KeyFingerprints {
  const out = {} as KeyFingerprints;
  // A zero-length key (SK_ai/SK_ar under an AEAD) has no fingerprint worth showing.
  for (const n of SK_NAMES) out[n] = { length: k[n].length, fingerprint: k[n].length ? fp16(k[n]) : "" };
  return out;
}

function sameKeys(a: IkeSaKeys, b: IkeSaKeys): boolean {
  return SK_NAMES.every((n) => equalBytes(a[n], b[n]));
}

function changed(prev: IkeSaKeys | null, next: IkeSaKeys): SkName[] {
  // Zero-octet keys (SK_ai/SK_ar under an AEAD) can neither change nor be "derived", so never listed.
  return SK_NAMES.filter((n) => next[n].length > 0 && (!prev || !equalBytes(prev[n], next[n])));
}

function seedBytes(seed: string | Uint8Array): Uint8Array {
  return typeof seed === "string" ? utf8ToBytes(seed) : seed;
}

/** Domain-separated demo derivation. Demo values only: real IKEv2 draws nonces, SPIs and keys at random. */
function derive(seed: Uint8Array, label: string, length: number): Uint8Array {
  const out = sha512(concatBytes(utf8ToBytes(`qsh/netkeys/ikev2-demo/v1|${label}|`), seed));
  if (length > out.length) throw new Error("derive: at most 64 octets");
  return out.slice(0, length);
}

function deriveEcpSecret(seed: Uint8Array, label: string): Uint8Array {
  for (let counter = 0; counter < 256; counter += 1) {
    const candidate = derive(seed, `${label}#${counter}`, 32);
    if (p256.utils.isValidSecretKey(candidate)) return candidate;
  }
  throw new Error("no valid P-256 scalar derived");
}

/**
 * Run the key schedule twice - once as the initiator sees it, once as the responder does - and
 * compare every key at every stage.
 */
export function ikev2KeySchedule(opts: Ikev2DemoOptions = {}): Ikev2KeyScheduleResult {
  const alg = IKEV2_DEMO_SUITE.prf.alg;
  const len = IKEV2_DEMO_LENGTHS;
  const seed = seedBytes(opts.seed ?? "qsh-ikev2-demo");

  // RFC 7296 2.10: nonces at least 128 bits and at least half the PRF key size (24 octets here).
  const ni = derive(seed, "Ni", 32);
  const nr = derive(seed, "Nr", 32);
  const spii = derive(seed, "SPIi", 8);
  const spir = derive(seed, "SPIr", 8);
  const i = deriveEcpSecret(seed, "initiator ECP-256 secret");
  const r = deriveEcpSecret(seed, "responder ECP-256 secret");

  // IKE_SA_INIT: KEi / KEr in RFC 5903 section 7 format, g^ir from each side.
  const kei = ecp256KeData(i);
  const ker = ecp256KeData(r);
  const girI = ecp256SharedSecret(i, ker);
  const girR = ecp256SharedSecret(r, kei);

  const skeyseed0I = computeSkeyseed(alg, ni, nr, girI);
  const skeyseed0R = computeSkeyseed(alg, ni, nr, girR);
  const sk0I = deriveIkeSaKeys(alg, skeyseed0I, ni, nr, spii, spir, len);
  const sk0R = deriveIkeSaKeys(alg, skeyseed0R, ni, nr, spii, spir, len);

  // IKE_INTERMEDIATE, n = 1: KEi(1) = ML-KEM-768 ek, KEr(1) = ciphertext (draft-ietf-ipsecme-ikev2-mlkem-09 section 2.1).
  const { publicKey: ek, secretKey: dk } = ml_kem768.keygen(concatBytes(derive(seed, "ML-KEM d", 32), derive(seed, "ML-KEM z", 32)));
  const { cipherText: ct, sharedSecret: ssResponder } = ml_kem768.encapsulate(ek, derive(seed, "ML-KEM m", 32));
  const ssInitiator = ml_kem768.decapsulate(ct, dk);
  const ake1I = additionalKeyExchange(alg, sk0I, ssInitiator, ni, nr, spii, spir, len);
  const ake1R = additionalKeyExchange(alg, sk0R, ssResponder, ni, nr, spii, spir, len);

  // RFC 8784 at IKE_AUTH, on the last round's keys (see the header for why that is the order).
  const derivedPpk = opts.ppk === undefined;
  const ppk = opts.ppk === null ? null
    : opts.ppk === undefined ? derive(seed, "PPK", 32)
      : typeof opts.ppk === "string" ? hexToBytes(opts.ppk.trim()) : opts.ppk;
  const finalI = ppk ? mixPpk(alg, ake1I.keys, ppk) : ake1I.keys;
  const finalR = ppk ? mixPpk(alg, ake1R.keys, ppk) : ake1R.keys;

  const cavp = runIkev2Cavp();
  const stages: Ikev2Stage[] = [
    {
      name: "IKE_SA_INIT (ECP-256)",
      rfc: "RFC 7296 section 2.14; RFC 5903 section 7",
      formula: "SKEYSEED = prf(Ni | Nr, g^ir); {SK_d | ... | SK_pr} = prf+(SKEYSEED, Ni | Nr | SPIi | SPIr)",
      validated: true,
      validatedBy: "NIST CAVP SP 800-135 IKEv2 KDF vectors (SKEYSEED and the prf+ stream); RFC 5903 section 8.1 example (g^ir); this suite's SK_* lengths are per RFC 5282 section 7.1 and RFC 4106 section 8.1",
      skeyseedFingerprint: fp16(skeyseed0I),
      keys: fingerprints(sk0I),
      peersAgree: equalBytes(skeyseed0I, skeyseed0R) && sameKeys(sk0I, sk0R),
      changedFromPrevious: changed(null, sk0I),
    },
    {
      name: "IKE_INTERMEDIATE #1 (ML-KEM-768)",
      rfc: "RFC 9370 section 2.2.2; draft-ietf-ipsecme-ikev2-mlkem-09",
      formula: "SKEYSEED(1) = prf(SK_d(0), SK(1) | Ni | Nr); {SK_d(1) | ... | SK_pr(1)} = prf+(SKEYSEED(1), Ni | Nr | SPIi | SPIr)",
      validated: false,
      validatedBy: "no official vectors exist; computed per the RFC text from CAVP-validated prf and prf+ (ML-KEM-768 itself: NIST ACVP)",
      skeyseedFingerprint: fp16(ake1I.skeyseed),
      keys: fingerprints(ake1I.keys),
      peersAgree: equalBytes(ake1I.skeyseed, ake1R.skeyseed) && sameKeys(ake1I.keys, ake1R.keys),
      changedFromPrevious: changed(sk0I, ake1I.keys),
    },
  ];
  if (ppk) {
    stages.push({
      name: "IKE_AUTH with PPK",
      rfc: "RFC 8784 section 3",
      formula: "SK_d = prf+(PPK, SK_d'); SK_pi = prf+(PPK, SK_pi'); SK_pr = prf+(PPK, SK_pr')  (primed = before the PPK)",
      validated: false,
      validatedBy: "no official vectors exist; computed per the RFC text from the CAVP-validated prf+",
      keys: fingerprints(finalI),
      peersAgree: sameKeys(finalI, finalR),
      changedFromPrevious: changed(ake1I.keys, finalI),
    });
  }
  const ppkChanged = ppk ? changed(ake1I.keys, finalI) : null;
  return {
    suite: IKEV2_DEMO_SUITE,
    mlkemStatus: MLKEM_IKEV2_STATUS,
    exchange: {
      nonceOctets: ni.length, spiOctets: spii.length, keiOctets: kei.length, kerOctets: ker.length,
      kei1Octets: ek.length, ker1Octets: ct.length,
      girAgrees: equalBytes(girI, girR),
      mlkemSecretAgrees: equalBytes(ssInitiator, ssResponder),
    },
    stages,
    properties: {
      ppkChangesOnlyDPiPr: ppkChanged === null ? null : ppkChanged.join(",") === "SK_d,SK_pi,SK_pr",
      // SK_ai/SK_ar are zero octets under an AEAD, so "all" means all NON-EMPTY keys.
      additionalKeyExchangeChangesAll: SK_NAMES.filter((n) => len[n.slice(3) as keyof SkLengths] > 0)
        .every((n) => changed(sk0I, ake1I.keys).includes(n)),
    },
    ppk: { used: ppk !== null, fingerprint: ppk ? fp16(ppk) : null, derivedFromSeed: ppk !== null && derivedPpk },
    validation: {
      cavpAllPassed: cavp.cases.length > 0 && cavp.cases.every((c) => c.passed),
      cavpCases: cavp.cases.length,
      rfc5903Passed: cavp.ecp256.cases.length > 0 && cavp.ecp256.cases.every((c) => c.passed),
    },
    notValidated: IKEV2_NOT_VALIDATED,
    notes: [
      "Nonces, SPIs, both ECP-256 secrets, the ML-KEM seed and message, and the default PPK are derived from the seed so the page is reproducible; IKEv2 draws them at random.",
      "RFC 8784 section 6 calls a PPK with at least 256 bits of entropy the strongest practice; a PPK derived from a typed seed has only the seed's entropy.",
      "SK_ai and SK_ar are zero octets because AES-GCM is an AEAD (RFC 5282 section 7.1).",
    ],
  };
}
