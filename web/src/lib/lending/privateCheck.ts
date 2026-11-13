/**
 * The pool's private liquidation check (Round 86): CKKS for the margin, a garbled circuit for its sign.
 *
 * THE MARGIN, in thousands of stable units, is
 *
 *     M = sum_i c_i p_i LT_i / 1000 - (d / 1000) I,
 *
 * linear in what the borrower holds, and M < 0 exactly when the health factor is below 1. A ratio would need an encrypted division; the sign of a difference needs none.
 *
 * WHO HOLDS THE KEY. In practice a liquidation is carried out by third parties and the borrower takes no part (Aave's IPool.liquidationCall; Chiu and Danisman, Bank of
 * Canada SAP 2026-13), and the confidential-DeFi systems keep the key away from the borrower, in a threshold network (Zama's KMS, 13 MPC nodes). So two arrangements:
 *
 *   "keyholder", the default. Carol holds the CKKS key and stands in for such a committee; node-seal has no threshold CKKS, so here she is one party. Bob encrypts his balances
 *   once under her public key and can go offline. Prices, thresholds and the index are public, as on a real market, and the pool computes on the ciphertext it stores, applying
 *   each settled liquidation, which is public, as a plaintext update. Only masked margins reach Carol. The mask keeps M from her, and the circuit, run by the pool and Carol, gives
 *   the one bit, which she signs so anyone can check it.
 *
 *   "borrower", a bilateral lender. Bob holds the key and answers every check, so he must be online and could refuse.
 *   On a public chain the thresholds are public, and then Bob can compute M himself: he needs no ciphertext to know it, and the mask and the circuit would hide nothing from
 *   anyone. They earn their place only when the thresholds are the pool's own, as here: M would reveal them.
 *
 * ONE CHECK, in the shape of the medical CNN's protocol (Gazelle, USENIX Security 2018) and with the same two classes: the pool multiplies by its weights, sums the three slots
 * with two rotations, cancels the imaginary half and adds a mask r to every slot; the key holder decrypts M + r, so the two hold additive shares of M; the shares enter the existing
 * two-party argmax over (M + offset, offset), where the first maximum wins, so index 1 means M < 0 strictly, the test a health factor below 1 is.
 *
 * WHAT THIS IS NOT, and the page says so. All the parties run in one tab, so the separation is notional. The protocol is semi-honest: nothing proves Bob encrypted his real balances
 * (that takes commitments and zero-knowledge proofs). The pool and Carol together see everything, and Carol must never hold Bob's ciphertext unmasked, which a deployment enforces with
 * access control. The mask is a bounded real, so it hides M only statistically, and the bound for this run is reported. The oblivious transfers rest on discrete logs.
 */

import { x25519 } from "@noble/curves/ed25519.js";
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { ml_dsa65, ml_dsa87 } from "@noble/post-quantum/ml-dsa.js";
import { ml_kem768, ml_kem1024 } from "@noble/post-quantum/ml-kem.js";
import { Arena, Client, sealed, Server, toRing, type Sealed } from "../clientHybridCnn";
import { hybridCombine, kemNameFor } from "../hybridCombiner";
import { openPsiRecord as openRecord, sealPsiRecord as sealRecord } from "../oprfPsi";
import type { LendingCfg } from "../config";
import type { LendingCheckpoint } from "./model";

/** The slots the rotate-and-sum covers: three inputs, rounded up to the power of two the rotations need. */
const BLOCK = 4;

const enc = new TextEncoder();
const INFO_SUBMISSION = enc.encode("qsh/lending/private-check/bob-to-pool");
const INFO_MASKED = enc.encode("qsh/lending/private-check/pool-to-bob");
const INFO_LINK_MASKED = enc.encode("qsh/lending/private-check/pool-to-carol");

export type Custody = LendingCfg["custody"];

/** One message of the protocol: who sent what to whom, and how long it was on the wire. */
/** A party to the check. The committee's members are numbered from 1. */
export type CheckParty = "Bob" | "the pool" | "Carol" | "anyone" | `member ${number}`;

export interface CheckMessage { from: CheckParty; to: CheckParty; what: string; bytes: number }

export interface PrivateCheckOutcome {
  hour: number;
  healthFactor: number;
  /** In thousands of stable units, as the circuit sees them. */
  clearMargin: number;
  /** Both shares added back: a self-test, available only because the parties are in this tab. */
  reconstructedMargin: number;
  absError: number;
  /** The circuit's output: true when the position can be liquidated. */
  flag: boolean;
  clearFlag: boolean;
  agrees: boolean;
  /** The median |value| the key holder decoded across every slot: about half the mask's width, so what it saw was masked. */
  keyHolderViewMedianAbs: number;
  /** This check's bound on the statistical distance of everything masked, from any value of no larger magnitude: for r uniform on [-R, R), SD(x + r, x' + r) = |x - x'| / (2R) <= |x| / R. */
  statisticalDistance: number;
  /** Everything this check put on the wire, in order. */
  messages: CheckMessage[];
  /** Key-holder and committee custody: Carol's signature over the hour, a digest of what she decrypted and the bit, checked against her public key. */
  bitSignatureVerified: boolean | null;
  /** Committee custody only: which members re-encrypted this check's margin, and what that cost. */
  committee?: CommitteeCheckStats;
  ckksBytes: number;
  gcBytes: number;
  gcMessages: number;
  andGates: number;
  rotations: number;
  conjugations: number;
  encryptMs: number;
  signMs: number;
  verifyMs: number;
  sealMs: number;
  evaluateMs: number;
  decryptMs: number;
  gcMs: number;
}

/** What Carol's setup hands every worker. Her secret key is in it only because every party is in this tab. */
export interface KeyholderSetup {
  publicKey: Uint8Array;
  galoisKeys: Uint8Array;
  secretKey: Uint8Array;
  /** Bob's one message: his signature and his ciphertext, sealed to the pool. */
  submission: Uint8Array;
  submissionHour: number;
  bobSignatureBytes: number;
  bobPublicKey: Uint8Array;
  /** The pool-Carol session, a second hybrid handshake: this is the key both of them derive. */
  linkKey: Uint8Array;
  carolSignature: { secretKey: Uint8Array; publicKey: Uint8Array };
  keygenMs: number;
  encryptMs: number;
  linkAgreed: boolean;
}

/** What one committee check reports beyond the common outcome (lending/committee.ts). */
export interface CommitteeCheckStats {
  /** The members who re-encrypted, numbered from 1. */
  active: number[];
  /** Every slot but the first decrypted to zero, as it must: only slot 0 carries anything. */
  restZero: boolean;
  shareBytes: number;
  c1Bytes: number;
  c0Bytes: number;
  shareMs: number;
  floodMs: number;
  combineMs: number;
  evaluatedBytes: number;
}

/** What a committee run reports once (lending/committee.ts). */
export interface CommitteeSummary {
  members: number;
  threshold: number;
  logN: number;
  logQ: number;
  plainModulus: number;
  maxWeights: number;
  noiseBoundLog2: number;
  floodBits: number;
  statSecBits: number;
  publicKeyBytes: number;
  carolPublicKeyBytes: number;
  keyGenMs: number;
  carolKeyGenMs: number;
  /** Bob's one message: his signature and his ciphertexts, sealed to the pool. */
  submissionBytes: number;
  encryptMs: number;
  linkAgreed: boolean;
  balanceBits: number;
  weightBits: number;
}

export interface CheckSession {
  check(input: LendingCheckpoint): Promise<PrivateCheckOutcome>;
  /** Committee custody only. */
  committee?: CommitteeSummary;
  /** The public and Galois keys the evaluator receives, once per key pair. */
  keyBytes: number;
  signatureBytes: number;
  keygenMs: number;
  close(): void;
}

export interface CheckSessionOptions {
  sessionKey: Uint8Array;
  sig: string;
  cfg: Pick<LendingCfg, "margin_scale_bits" | "margin_mask_bits">;
  custody: Custody;
  keyholder?: KeyholderSetup;
  /** Committee custody: the KEM for the pool and Carol's handshake. */
  kem?: string;
  /** Committee custody: the position as it stands at the first check, which Bob encrypts once. */
  first?: LendingCheckpoint;
  /** Committee custody: where the threshold tier's wasm is served, a URL prefix ending in "/". */
  fedThreshBase?: string;
}

const concat = (...parts: Uint8Array[]): Uint8Array => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};
export const dsaOf = (sig: string) => (sig.includes("87") ? ml_dsa87 : ml_dsa65);
const rotationSteps = () => { const steps = [0]; for (let step = 1; step < BLOCK; step <<= 1) steps.push(step); return steps; };

/** What Bob's signature covers: which hour his submission answers and a digest of his ciphertext. */
export const submissionMessage = (hour: number, ciphertext: Uint8Array): Uint8Array =>
  concat(enc.encode(`qsh-lending/submission/${hour}/`), sha256(ciphertext));

/** What Carol's signature covers: the hour, a digest of the masked ciphertext she decrypted, and the bit. */
export const bitMessage = (hour: number, maskedCiphertext: Uint8Array, flag: boolean): Uint8Array =>
  concat(enc.encode(`qsh-lending/bit/${hour}/${flag ? 1 : 0}/`), sha256(maskedCiphertext));

/** The keys the two directions are sealed under, one per direction, with the hour as the nonce's counter: every worker shares the session key and no two seal the same hour. */
export function checkKeys(sessionKey: Uint8Array): { submission: Uint8Array; masked: Uint8Array } {
  if (!sessionKey || sessionKey.length === 0) throw new Error("the private check is sealed under the PQC session key, and none was given");
  return { submission: hkdf(sha256, sessionKey, undefined, INFO_SUBMISSION, 32), masked: hkdf(sha256, sessionKey, undefined, INFO_MASKED, 32) };
}

/** The pool and Carol's own hybrid handshake, the same combiner as the run's: both sides are computed here and compared, because both are in this tab. */
export function establishLink(kem: string): { key: Uint8Array; agreed: boolean } {
  const scheme = kem.includes("1024") ? ml_kem1024 : ml_kem768;
  const carol = scheme.keygen();
  const e = scheme.encapsulate(carol.publicKey);
  const opened = scheme.decapsulate(e.cipherText, carol.secretKey);
  const poolX = x25519.utils.randomSecretKey(), carolX = x25519.utils.randomSecretKey();
  const poolXPub = x25519.getPublicKey(poolX), carolXPub = x25519.getPublicKey(carolX);
  const tr = { c1: e.cipherText, c2: poolXPub, ek1: carol.publicKey, ek2: carolXPub };
  const name = kemNameFor(kem);
  const poolSide = hybridCombine(e.sharedSecret, x25519.getSharedSecret(poolX, carolXPub), tr, name, 32);
  const carolSide = hybridCombine(opened, x25519.getSharedSecret(carolX, poolXPub), tr, name, 32);
  return { key: poolSide, agreed: poolSide.every((b, i) => b === carolSide[i]) };
}

const submissionAad = (hour: number) => enc.encode(`qsh-lending/submission/${hour}`);
export const maskedAad = (hour: number) => enc.encode(`qsh-lending/masked-margin/${hour}`);

/** The key the pool seals Carol's records under, from the pool and Carol's own handshake. */
export const linkMaskedKey = (linkKey: Uint8Array): Uint8Array => hkdf(sha256, linkKey, undefined, INFO_LINK_MASKED, 32);

/** Bob's message to the pool: his signature, then his ciphertext, sealed under the session. */
export const sealSubmission = (key: Uint8Array, hour: number, signature: Uint8Array, ciphertext: Uint8Array): Promise<Uint8Array> =>
  sealRecord(key, concat(signature, ciphertext), submissionAad(hour), hour);

/** The pool's side: opens the record and refuses it unless the signature verifies. Returns the ciphertext, and how long each half took. */
export async function openSubmission(key: Uint8Array, record: Uint8Array, hour: number, signatureLength: number,
                                     verify: (signature: Uint8Array, message: Uint8Array) => boolean): Promise<{ ciphertext: Uint8Array; openMs: number; verifyMs: number }> {
  let t = performance.now();
  const opened = await openRecord(key, record, submissionAad(hour));
  const openMs = performance.now() - t;
  const ciphertext = opened.subarray(signatureLength);
  t = performance.now();
  if (!verify(opened.subarray(0, signatureLength), submissionMessage(hour, ciphertext))) throw new Error(`the submission for hour ${hour} does not verify`);
  return { ciphertext, openMs, verifyMs: performance.now() - t };
}

/** The borrower's three values and the pool's three weights, in the units M is measured in. */
export function checkVectors(input: Pick<LendingCheckpoint, "amounts" | "scaledDebt" | "prices" | "thresholds" | "index">): { x: number[]; w: number[] } {
  return {
    x: [input.amounts[0], input.amounts[1], input.scaledDebt / 1000],
    w: [input.prices[0] * input.thresholds[0] / 1000, input.prices[1] * input.thresholds[1] / 1000, -input.index],
  };
}

/** Every slot the rotate-and-sum leaves non-zero, as the windows of width BLOCK it sums: the full margin twice, and the partial sums at the edges. All of them are masked. */
export function maskedPartials(x: number[], w: number[]): number[] {
  const terms = x.map((v, i) => v * w[i]);
  const out: number[] = [];
  for (let start = -(BLOCK - 1); start < terms.length; start++) {
    let sum = 0;
    for (let k = Math.max(start, 0); k < Math.min(start + BLOCK, terms.length); k++) sum += terms[k];
    out.push(sum);
  }
  return out;
}

export type Mpc = typeof import("../clientMPC");

/** The two parties' shares of M into the existing argmax: index 1 wins only when offset > M + offset, that is when M < 0. `held` is the key holder's value, M + r. */
async function signOfMargin(mpc: Mpc, held: number, r: number, cfg: CheckSessionOptions["cfg"]) {
  const scale = 2 ** cfg.margin_scale_bits, ring = 2 ** 32;
  const offset = 2 ** (31 - cfg.margin_scale_bits);       // in margin units: centres the margin in the unsigned 32-bit ring
  const a = Uint32Array.of(toRing((held + offset) * scale, ring), toRing(offset * scale, ring));
  const b = Uint32Array.of(toRing(-r * scale, ring), 0);
  const t = performance.now();
  const gc = await mpc.secureArgmaxShares(a, b);
  return { flag: gc.arg === 1, gc, gcMs: performance.now() - t };
}

const medianAbs = (values: Float64Array): number => {
  const sorted = Array.from(values, Math.abs).sort((p, q) => p - q);
  return sorted[sorted.length >> 1];
};

/**
 * Carol's keys and Bob's one submission: run once, before any check. Bob needs only her public key.
 *
 * `first` is the position as it stands at the first check. Every later position is that one less what has been settled in the open, which the pool applies itself.
 */
export async function setupKeyholder(o: { sessionKey: Uint8Array; sig: string; kem: string }, first: LendingCheckpoint): Promise<KeyholderSetup> {
  const t0 = performance.now();
  const s = await sealed();
  const carol = new Client(s, rotationSteps());
  try {
    const keygenMs = performance.now() - t0;
    const dsa = dsaOf(o.sig);
    const bobDsa = dsa.keygen(crypto.getRandomValues(new Uint8Array(32)));
    const carolDsa = dsa.keygen(crypto.getRandomValues(new Uint8Array(32)));
    const pad = new Float64Array(s.slots);
    pad.set(checkVectors(first).x);

    // Bob, with Carol's public key only: an Encryptor made from it, and no decryptor anywhere in his hands.
    const arena = new Arena();
    let encryptMs: number, ciphertext: Uint8Array;
    try {
      const { seal, context } = s;
      const t = performance.now();
      const pk = arena.hold(new seal.PublicKey());
      pk.loadFromArray(context, carol.publicKey);
      const encoder = arena.hold(new seal.CKKSEncoder(context));
      const encryptor = arena.hold(new seal.Encryptor(context, pk));
      const pt = arena.hold(new seal.Plaintext());
      encoder.encode(pad, s.scale, pt);
      const ct = arena.hold(new seal.Ciphertext());
      encryptor.encrypt(pt, ct);
      ciphertext = ct.saveToArray(seal.ComprModeType.none);
      encryptMs = performance.now() - t;
    } finally { arena.close(); }

    const keys = checkKeys(o.sessionKey);
    const signature = dsa.sign(submissionMessage(first.hour, ciphertext), bobDsa.secretKey);
    const submission = await sealSubmission(keys.submission, first.hour, signature, ciphertext);
    const link = establishLink(o.kem);
    return {
      publicKey: carol.publicKey, galoisKeys: carol.galoisKeys, secretKey: carol.exportSecretKey(), submission, submissionHour: first.hour,
      bobSignatureBytes: signature.length, bobPublicKey: bobDsa.publicKey, linkKey: link.key, linkAgreed: link.agreed,
      carolSignature: { secretKey: carolDsa.secretKey, publicKey: carolDsa.publicKey }, keygenMs, encryptMs,
    };
  } finally { carol.close(); }
}

/** One check in the borrower's custody: Bob encrypts, the pool evaluates and masks, Bob decrypts, the two run the circuit. */
function openBorrowerSession(o: CheckSessionOptions, s: Sealed, mpc: Mpc, keygenStart: number): CheckSession {
  const client = new Client(s, rotationSteps());
  const server = new Server(s, client.publicKey, client.galoisKeys);
  const dsa = dsaOf(o.sig);
  const signer = dsa.keygen(crypto.getRandomValues(new Uint8Array(32)));
  const keys = checkKeys(o.sessionKey);
  const keygenMs = performance.now() - keygenStart;
  const pad = (v: number[]) => { const a = new Float64Array(s.slots); a.set(v); return a; };
  const mask = 2 ** o.cfg.margin_mask_bits;

  return {
    keyBytes: client.publicKey.length + client.galoisKeys.length, signatureBytes: dsa.lengths.signature ?? 0, keygenMs,
    close() { server.close(); client.close(); },
    async check(input) {
      const arena = new Arena();
      const rotationsBefore = server.rotations, conjugationsBefore = server.conjugations;
      try {
        const { x, w } = checkVectors(input);
        let t = performance.now();
        const ciphertext = client.encrypt(pad(x));
        const encryptMs = performance.now() - t;
        t = performance.now();
        const signature = dsa.sign(submissionMessage(input.hour, ciphertext), signer.secretKey);
        const signMs = performance.now() - t;
        t = performance.now();
        const submission = await sealSubmission(keys.submission, input.hour, signature, ciphertext);
        let sealMs = performance.now() - t;

        const received = await openSubmission(keys.submission, submission, input.hour, signature.length,
          (sig, message) => dsa.verify(sig, message, signer.publicKey));
        sealMs += received.openMs;
        t = performance.now();
        const ct = server.receive(received.ciphertext, arena);
        const masked = server.evaluate(ct, pad(w), BLOCK, 1, () => 0, o.cfg.margin_mask_bits);
        const evaluateMs = performance.now() - t;
        t = performance.now();
        const reply = await sealRecord(keys.masked, masked.bytes, maskedAad(input.hour), input.hour);
        sealMs += performance.now() - t;

        t = performance.now();
        const reopened = await openRecord(keys.masked, reply, maskedAad(input.hour));
        sealMs += performance.now() - t;
        t = performance.now();
        const decoded = client.decrypt(reopened);
        const decryptMs = performance.now() - t;
        const y = decoded[0], r = masked.mask[0];
        const { flag, gc, gcMs } = await signOfMargin(mpc, y, r, o.cfg);
        const reconstructedMargin = y - r, clearMargin = input.margin / 1000;
        return {
          hour: input.hour, healthFactor: input.healthFactor, clearMargin, reconstructedMargin, absError: Math.abs(reconstructedMargin - clearMargin), flag,
          clearFlag: clearMargin < 0, agrees: flag === (clearMargin < 0), keyHolderViewMedianAbs: medianAbs(decoded),
          statisticalDistance: maskedPartials(x, w).reduce((sum, v) => sum + Math.abs(v) / mask, 0),
          messages: [
            { from: "Bob", to: "the pool", what: "his ciphertext, signed and sealed", bytes: submission.length },
            { from: "the pool", to: "Bob", what: "the masked margin, sealed", bytes: reply.length },
          ],
          bitSignatureVerified: null,
          ckksBytes: submission.length + reply.length, gcBytes: gc.bytes, gcMessages: gc.messages, andGates: gc.andOps,
          rotations: server.rotations - rotationsBefore, conjugations: server.conjugations - conjugationsBefore,
          encryptMs, signMs, verifyMs: received.verifyMs, sealMs, evaluateMs, decryptMs, gcMs,
        };
      } finally { arena.close(); }
    },
  };
}

/** Carol's side in a worker: the secret key, and a decrypt that returns each slot's real part. It is handed only masked records. */
class KeyHolderDecryptor {
  private readonly arena = new Arena();
  private readonly encoder: any;
  private readonly decryptor: any;
  constructor(private readonly s: Sealed, secretKey: Uint8Array) {
    const { seal, context } = s;
    const sk = this.arena.hold(new seal.SecretKey());
    sk.loadFromArray(context, secretKey);
    this.encoder = this.arena.hold(new seal.CKKSEncoder(context));
    this.decryptor = this.arena.hold(new seal.Decryptor(context, sk));
  }
  decryptMasked(bytes: Uint8Array): Float64Array {
    const { seal, context } = this.s;
    const ct = new seal.Ciphertext(), pt = new seal.Plaintext();
    try {
      ct.loadFromArray(context, bytes);
      this.decryptor.decrypt(ct, pt);
      return Float64Array.from(this.encoder.decodeFloat64(pt) as ArrayLike<number>, (v) => v / 2);
    } finally { ct.delete(); pt.delete(); }
  }
  close(): void { this.arena.close(); }
}

/**
 * One check in Carol's custody. Bob does nothing here: the pool keeps his one ciphertext and applies each settlement to it, and only the masked margin goes to Carol.
 */
async function openKeyholderSession(o: CheckSessionOptions, s: Sealed, mpc: Mpc, keygenStart: number): Promise<CheckSession> {
  const k = o.keyholder;
  if (!k) throw new Error("a key holder's check needs her setup");
  const pool = new Server(s, k.publicKey, k.galoisKeys);
  const carol = new KeyHolderDecryptor(s, k.secretKey);
  const dsa = dsaOf(o.sig);
  const keys = checkKeys(o.sessionKey);
  const linkKey = linkMaskedKey(k.linkKey);
  // Opened and checked once per worker, as a pool would on receipt.
  const received = await openSubmission(keys.submission, k.submission, k.submissionHour, k.bobSignatureBytes,
    (sig, message) => dsa.verify(sig, message, k.bobPublicKey));
  const keygenMs = performance.now() - keygenStart;
  const pad = (v: number[]) => { const a = new Float64Array(s.slots); a.set(v); return a; };
  const mask = 2 ** o.cfg.margin_mask_bits;

  return {
    keyBytes: k.publicKey.length + k.galoisKeys.length, signatureBytes: dsa.lengths.signature ?? 0, keygenMs,
    close() { pool.close(); carol.close(); },
    async check(input) {
      const arena = new Arena();
      const rotationsBefore = pool.rotations, conjugationsBefore = pool.conjugations;
      try {
        const { x, w } = checkVectors(input);
        // The pool: Bob's stored ciphertext, less what has been settled in the open, at this hour's public prices.
        let t = performance.now();
        const stored = pool.receive(received.ciphertext, arena);
        const updated = pool.addShare(stored, pad([-input.settled.seized[0], -input.settled.seized[1], -input.settled.scaledRepaid / 1000]), arena);
        const masked = pool.evaluate(updated, pad(w), BLOCK, 1, () => 0, o.cfg.margin_mask_bits);
        const evaluateMs = performance.now() - t;
        t = performance.now();
        const toCarol = await sealRecord(linkKey, masked.bytes, maskedAad(input.hour), input.hour);
        let sealMs = performance.now() - t;

        // Carol: only this reaches her.
        t = performance.now();
        const opened = await openRecord(linkKey, toCarol, maskedAad(input.hour));
        sealMs += performance.now() - t;
        t = performance.now();
        const decoded = carol.decryptMasked(opened);
        const decryptMs = performance.now() - t;
        const y = decoded[0], r = masked.mask[0];
        const { flag, gc, gcMs } = await signOfMargin(mpc, y, r, o.cfg);

        // Carol signs the bit, so that anyone can check it came from her and from this ciphertext.
        t = performance.now();
        const message = bitMessage(input.hour, opened, flag);
        const bitSignature = dsa.sign(message, k.carolSignature.secretKey);
        const signMs = performance.now() - t;
        t = performance.now();
        const bitSignatureVerified = dsa.verify(bitSignature, message, k.carolSignature.publicKey);
        const verifyMs = performance.now() - t;

        const reconstructedMargin = y - r, clearMargin = input.margin / 1000;
        return {
          hour: input.hour, healthFactor: input.healthFactor, clearMargin, reconstructedMargin, absError: Math.abs(reconstructedMargin - clearMargin), flag,
          clearFlag: clearMargin < 0, agrees: flag === (clearMargin < 0), keyHolderViewMedianAbs: medianAbs(decoded),
          statisticalDistance: maskedPartials(x, w).reduce((sum, v) => sum + Math.abs(v) / mask, 0),
          messages: [
            { from: "the pool", to: "Carol", what: "the masked margin, sealed", bytes: toCarol.length },
            { from: "Carol", to: "anyone", what: "the signed bit", bytes: bitSignature.length + 1 },
          ],
          bitSignatureVerified,
          ckksBytes: toCarol.length, gcBytes: gc.bytes, gcMessages: gc.messages, andGates: gc.andOps,
          rotations: pool.rotations - rotationsBefore, conjugations: pool.conjugations - conjugationsBefore,
          encryptMs: 0, signMs, verifyMs, sealMs, evaluateMs, decryptMs, gcMs,
        };
      } finally { arena.close(); }
    },
  };
}

/** One key pair for each thread that runs checks, or, in Carol's custody, her one key pair loaded on each thread. */
export async function openCheckSession(o: CheckSessionOptions): Promise<CheckSession> {
  const t0 = performance.now();
  const mpc: Mpc = await import("../clientMPC");
  // The committee runs on threshold BGV in the Go module, and needs no SEAL.
  if (o.custody === "committee") return (await import("./committee")).openCommitteeSession(o, mpc, t0);
  const s = await sealed();
  return o.custody === "keyholder" ? openKeyholderSession(o, s, mpc, t0) : openBorrowerSession(o, s, mpc, t0);
}
