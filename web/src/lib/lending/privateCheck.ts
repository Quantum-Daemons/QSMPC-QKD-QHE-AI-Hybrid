/**
 * The pool's private liquidation check (Round 86): CKKS for the margin, a garbled circuit for its sign.
 *
 * WHO HOLDS WHAT. Bob holds his collateral, his scaled debt and the CKKS secret key. The pool, Alice, holds its risk schedule: the liquidation threshold of each
 * asset. Prices and the interest index are public. The margin, in thousands of stable units, is
 *
 *     M = sum_i c_i p_i LT_i / 1000 - (d / 1000) I,
 *
 * linear in what Bob holds, and M < 0 exactly when the health factor is below 1. A ratio would need an encrypted division; the sign of a difference needs none.
 *
 * WHY THE THRESHOLDS ARE THE POOL'S SECRET HERE. On a public chain they are public, and then Bob can compute M himself: he needs no ciphertext to know it, and the mask
 * and the circuit below would hide nothing from anyone. They earn their place only when the party holding the key must not learn M. Here M would reveal the schedule, so
 * this is the case of a lender whose haircuts are its own, and the page says so.
 *
 * ONE CHECK, in the shape of the medical CNN's protocol (Gazelle, USENIX Security 2018) and with the same two classes:
 *   1. Bob encrypts [c_1, c_2, d / 1000] into one ciphertext, signs it with ML-DSA and seals it under a key derived from the PQC session key;
 *   2. the pool checks the signature, multiplies by its weights, sums the three slots with two rotations, cancels the imaginary half and adds a mask r to every slot;
 *   3. Bob decrypts M + r, so the two of them hold additive shares of M, and neither sees M;
 *   4. the shares enter the existing two-party argmax over (M + offset, offset). The first maximum wins, so index 1 means M < 0 strictly, the test a health factor below 1 is.
 *
 * WHAT THIS IS NOT, and the page says so. Both parties run in one tab, so the separation is notional. The protocol is semi-honest: nothing proves Bob encrypted his real
 * balances (that takes commitments and zero-knowledge proofs), and a malicious Bob could probe the thresholds with crafted ciphertexts (MUSE, USENIX Security 2021, does this to weights).
 * The mask is a bounded real, so it hides M only statistically, and the bound for this run is reported. The oblivious transfers rest on discrete logs, so the circuit
 * is not post-quantum, though its messages would travel inside the PQC-keyed session. The bit is revealed to both parties, and repeated bits narrow the schedule down.
 */

import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { ml_dsa65, ml_dsa87 } from "@noble/post-quantum/ml-dsa.js";
import { Arena, Client, sealed, Server, toRing, type Sealed } from "../clientHybridCnn";
import { openPsiRecord as openRecord, sealPsiRecord as sealRecord } from "../oprfPsi";
import type { LendingCfg } from "../config";
import type { LendingCheckpoint } from "./model";

/** The slots the rotate-and-sum covers: three inputs, rounded up to the power of two the rotations need. */
const BLOCK = 4;

const enc = new TextEncoder();
const INFO_SUBMISSION = enc.encode("qsh/lending/private-check/bob-to-pool");
const INFO_MASKED = enc.encode("qsh/lending/private-check/pool-to-bob");

export interface PrivateCheckOutcome {
  hour: number;
  healthFactor: number;
  /** In thousands of stable units, as the circuit sees them. */
  clearMargin: number;
  /** Both shares added back: a self-test, available only because both parties are in this tab. */
  reconstructedMargin: number;
  absError: number;
  /** The circuit's output: true when the position can be liquidated. */
  flag: boolean;
  clearFlag: boolean;
  agrees: boolean;
  /** The median |value| Bob decoded across every slot: about half the mask's width, so what he saw was masked. */
  bobViewMedianAbs: number;
  /** This check's bound on the statistical distance of everything masked, from any value of no larger magnitude: for r uniform on [-R, R), SD(x + r, x' + r) = |x - x'| / (2R) <= |x| / R. */
  statisticalDistance: number;
  /** Both sealed records, as they would cross the wire. */
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

export interface CheckSession {
  check(input: LendingCheckpoint): Promise<PrivateCheckOutcome>;
  /** The public and Galois keys Bob hands the pool, once per key pair. */
  keyBytes: number;
  signatureBytes: number;
  keygenMs: number;
  close(): void;
}

export interface CheckSessionOptions {
  sessionKey: Uint8Array;
  sig: string;
  cfg: Pick<LendingCfg, "margin_scale_bits" | "margin_mask_bits">;
}

const concat = (a: Uint8Array, b: Uint8Array): Uint8Array => { const o = new Uint8Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };

/** What Bob's signature covers: which check it answers and a digest of his ciphertext. */
export const submissionMessage = (hour: number, ciphertext: Uint8Array): Uint8Array =>
  concat(enc.encode(`qsh-lending/submission/${hour}/`), sha256(ciphertext));

/** The keys the two directions are sealed under, one per direction, with the hour as the nonce's counter: every worker shares the session key and no two seal the same hour. */
export function checkKeys(sessionKey: Uint8Array): { submission: Uint8Array; masked: Uint8Array } {
  if (!sessionKey || sessionKey.length === 0) throw new Error("the private check is sealed under the PQC session key, and none was given");
  return { submission: hkdf(sha256, sessionKey, undefined, INFO_SUBMISSION, 32), masked: hkdf(sha256, sessionKey, undefined, INFO_MASKED, 32) };
}

const submissionAad = (hour: number) => enc.encode(`qsh-lending/submission/${hour}`);

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

/** Bob's three values and the pool's three weights, in the units M is measured in. */
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

type Mpc = typeof import("../clientMPC");

/** One key pair for each thread that runs checks. */
export async function openCheckSession(o: CheckSessionOptions): Promise<CheckSession> {
  const t0 = performance.now();
  const s: Sealed = await sealed();
  const mpc: Mpc = await import("../clientMPC");
  const steps = [0];
  for (let step = 1; step < BLOCK; step <<= 1) steps.push(step);
  const client = new Client(s, steps);
  const server = new Server(s, client.publicKey, client.galoisKeys);
  const dsa = o.sig.includes("87") ? ml_dsa87 : ml_dsa65;
  const signer = dsa.keygen(crypto.getRandomValues(new Uint8Array(32)));
  const keys = checkKeys(o.sessionKey);
  const keygenMs = performance.now() - t0;

  const scale = 2 ** o.cfg.margin_scale_bits, mask = 2 ** o.cfg.margin_mask_bits;
  const offset = 2 ** (31 - o.cfg.margin_scale_bits);       // in margin units: centres the margin in the unsigned 32-bit ring
  const ring = 2 ** 32;
  const pad = (v: number[]) => { const a = new Float64Array(s.slots); a.set(v); return a; };

  return {
    keyBytes: client.publicKey.length + client.galoisKeys.length,
    signatureBytes: dsa.lengths.signature ?? 0,
    keygenMs,
    close() { server.close(); client.close(); },
    async check(input) {
      const arena = new Arena();
      const rotationsBefore = server.rotations, conjugationsBefore = server.conjugations;
      try {
        const { x, w } = checkVectors(input);
        const aadOut = enc.encode(`qsh-lending/masked-margin/${input.hour}`);

        // Bob
        let t = performance.now();
        const ciphertext = client.encrypt(pad(x));
        const encryptMs = performance.now() - t;
        t = performance.now();
        const signature = dsa.sign(submissionMessage(input.hour, ciphertext), signer.secretKey);
        const signMs = performance.now() - t;
        t = performance.now();
        const submission = await sealSubmission(keys.submission, input.hour, signature, ciphertext);
        let sealMs = performance.now() - t;

        // The pool
        const received = await openSubmission(keys.submission, submission, input.hour, signature.length,
          (sig, message) => dsa.verify(sig, message, signer.publicKey));
        sealMs += received.openMs;
        const verifyMs = received.verifyMs;
        t = performance.now();
        const ct = server.receive(received.ciphertext, arena);
        const masked = server.evaluate(ct, pad(w), BLOCK, 1, () => 0, o.cfg.margin_mask_bits);
        const evaluateMs = performance.now() - t;
        t = performance.now();
        const reply = await sealRecord(keys.masked, masked.bytes, aadOut, input.hour);
        sealMs += performance.now() - t;

        // Bob again
        t = performance.now();
        const reopened = await openRecord(keys.masked, reply, aadOut);
        sealMs += performance.now() - t;
        t = performance.now();
        const decoded = client.decrypt(reopened);
        const decryptMs = performance.now() - t;
        const y = decoded[0], r = masked.mask[0];

        // Both, under the circuit: index 1 wins only when offset > M + offset, that is when M < 0.
        t = performance.now();
        const a = Uint32Array.of(toRing((y + offset) * scale, ring), toRing(offset * scale, ring));
        const b = Uint32Array.of(toRing(-r * scale, ring), 0);
        const gc = await mpc.secureArgmaxShares(a, b);
        const gcMs = performance.now() - t;

        const reconstructedMargin = y - r;
        const clearFlag = input.margin / 1000 < 0;
        const flag = gc.arg === 1;
        const sorted = Array.from(decoded, Math.abs).sort((p, q) => p - q);
        const statisticalDistance = maskedPartials(x, w).reduce((sum, v) => sum + Math.abs(v) / mask, 0);
        return {
          hour: input.hour, healthFactor: input.healthFactor, clearMargin: input.margin / 1000, reconstructedMargin,
          absError: Math.abs(reconstructedMargin - input.margin / 1000), flag, clearFlag, agrees: flag === clearFlag,
          bobViewMedianAbs: sorted[sorted.length >> 1], statisticalDistance,
          ckksBytes: submission.length + reply.length, gcBytes: gc.bytes, gcMessages: gc.messages, andGates: gc.andOps,
          rotations: server.rotations - rotationsBefore, conjugations: server.conjugations - conjugationsBefore,
          encryptMs, signMs, verifyMs, sealMs, evaluateMs, decryptMs, gcMs,
        };
      } finally {
        arena.close();
      }
    },
  };
}
