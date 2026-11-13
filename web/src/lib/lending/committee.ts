/**
 * The lending check under a committee (Round 86, the owner's choice of 2026-11-13): no one holds the key.
 *
 * Carol's custody stands one party in for the threshold networks confidential-DeFi systems use, because node-seal has no threshold CKKS. Lattigo has threshold BGV, and the
 * margin is linear in Bob's balances, so this arrangement runs the check that way (wasm/qsh-fedthresh/agg/committee.go):
 *
 *   - three members make a collective BGV key, each holding a Shamir share of its secret, any two enough to use it (N = 16384, a 260-bit Q, a 50-bit prime t);
 *   - Bob encrypts each quantised balance once under the collective public key, signs the ciphertexts, seals them to the pool and goes offline;
 *   - at each check the pool multiplies by the hour's public weights, subtracts what has been settled in the open, and adds a mask r uniform modulo t and a fresh encryption of
 *     zero, all in slot 0;
 *   - two members, a different pair at each check, re-encrypt that ciphertext to Carol's own key with Lattigo's public-key switch, each flooding its share with uniform noise sized
 *     for 128 bits over the whole share; the pool combines their shares and seals the result to Carol;
 *   - Carol decrypts M + r mod t, uniform whatever M is, and with the pool's t - r runs a garbled circuit for the sign of M (Polytune, wasm/qsh-mpc); she signs the bit.
 *
 * FIXED POINT. Balances at 2^-20 of a unit and weights at 2^-19, so the margin is an integer in 2^-39 thousands of stable units, and t/2 holds margins up to about a million
 * stable units. The weights' absolute sum is held under 2^27, the bound the flooding is sized for.
 *
 * WHAT IT IS NOT. Every member, Carol and the pool are in this tab, so the separation is notional, and the checks run in order on one worker, which holds the one committee. The
 * protocol is passive-secure and not robust (Lattigo's multiparty package): a member who withholds a share stops the check, and nothing proves a share was computed correctly.
 * The members share once per ciphertext, as Lattigo's SECURITY.md requires.
 */

import { openPsiRecord as openRecord, sealPsiRecord as sealRecord } from "../oprfPsi";
import { loadFedThresh } from "../fedThresh";
import type { LendingCheckpoint } from "./model";
import {
  bitMessage, checkKeys, checkVectors, dsaOf, establishLink, linkMaskedKey, maskedAad, type CheckMessage, type CheckSession, type CheckSessionOptions,
  type CommitteeCheckStats, type CommitteeSummary, type Mpc, openSubmission, sealSubmission, submissionMessage,
} from "./privateCheck";

/** Three members, any two of whom can re-encrypt. */
export const COMMITTEE = { members: 3, threshold: 2 } as const;
/** Fixed-point bits of Bob's balances and of the pool's weights. */
export const COMMITTEE_BALANCE_BITS = 20;
export const COMMITTEE_WEIGHT_BITS = 19;
/** The weights' absolute sum the flooding is sized for. */
export const COMMITTEE_MAX_WEIGHTS = 2 ** 27;

const f64 = (values: number[]): Uint8Array => new Uint8Array(Float64Array.from(values).buffer);
const q = (v: number, bits: number): number => Math.round(v * 2 ** bits);

/** Bob's quantised balances: two collateral amounts and his scaled debt in thousands. */
export const committeeBalances = (first: LendingCheckpoint): number[] => checkVectors(first).x.map((v) => q(v, COMMITTEE_BALANCE_BITS));

/** The hour's quantised weights and what has been settled since the first check, in the balances' units. */
export function committeeHour(input: LendingCheckpoint): { w: number[]; settled: number[] } {
  return {
    w: checkVectors(input).w.map((v) => q(v, COMMITTEE_WEIGHT_BITS)),
    settled: [input.settled.seized[0], input.settled.seized[1], input.settled.scaledRepaid / 1000].map((v) => q(v, COMMITTEE_BALANCE_BITS)),
  };
}

/** The margin the integers stand for, exactly, as a BigInt in 2^-39 thousands of stable units. */
export function committeeMargin(x: number[], w: number[], settled: number[]): bigint {
  return x.reduce((m, xi, i) => m + BigInt(w[i]) * (BigInt(xi) - BigInt(settled[i])), 0n);
}

/** v in [0, t) as its representative in (-t/2, t/2]. */
export const signedMod = (v: bigint, t: bigint): bigint => (v > t / 2n ? v - t : v);

/** The active pair at the k-th check: (1, 2), (2, 3), (3, 1), ... as indices from 0. */
export const activeAt = (k: number): number[] => [k % COMMITTEE.members, (k + 1) % COMMITTEE.members];

function must<T extends { error?: string }>(r: T, what: string): T {
  if (r.error) throw new Error(`${what}: ${r.error}`);
  return r;
}

export async function openCommitteeSession(o: CheckSessionOptions, mpc: Mpc, keygenStart: number): Promise<CheckSession> {
  if (!o.first) throw new Error("a committee's check needs the position Bob encrypts");
  if (!o.fedThreshBase) throw new Error("a committee's check needs the threshold module's location");
  const api = await loadFedThresh(o.fedThreshBase);
  // The common reference string the members' key generation shares: public randomness, fresh for each run.
  const crs = crypto.getRandomValues(new Uint8Array(32));
  const opened = must(api.committeeOpen(COMMITTEE.members, COMMITTEE.threshold, crs, COMMITTEE_MAX_WEIGHTS), "committee");
  const id = opened.id!;
  try {
    const setting = JSON.parse(opened.setting ?? "{}") as Record<string, number>;
    const t = BigInt(setting.plainModulus);
    const dsa = dsaOf(o.sig);
    const keys = checkKeys(o.sessionKey);

    // Bob, with the committee's public key only: his balances once, signed and sealed to the pool.
    const x = committeeBalances(o.first);
    const bob = dsa.keygen(crypto.getRandomValues(new Uint8Array(32)));
    const enc = must(api.committeeEncrypt(id, f64(x)), "Bob's encryption");
    const signature = dsa.sign(submissionMessage(o.first.hour, enc.ciphertexts!), bob.secretKey);
    const submission = await sealSubmission(keys.submission, o.first.hour, signature, enc.ciphertexts!);

    // The pool: opens it, refuses it unless Bob's signature verifies, and stores the ciphertexts.
    const received = await openSubmission(keys.submission, submission, o.first.hour, signature.length, (sig, message) => dsa.verify(sig, message, bob.publicKey));
    must(api.committeeStore(id, received.ciphertext, x.length), "the pool's store");

    // The pool and Carol's own handshake, and Carol's signing key for the bit.
    const link = establishLink(o.kem ?? "ml-kem-768");
    const linkKey = linkMaskedKey(link.key);
    const carolDsa = dsa.keygen(crypto.getRandomValues(new Uint8Array(32)));
    const keygenMs = performance.now() - keygenStart;
    const summary: CommitteeSummary = {
      members: COMMITTEE.members, threshold: COMMITTEE.threshold, logN: setting.logN, logQ: setting.logQ, plainModulus: setting.plainModulus,
      maxWeights: setting.maxWeights, noiseBoundLog2: setting.noiseBoundLog2, floodBits: setting.floodBits, statSecBits: setting.statSecBits,
      publicKeyBytes: opened.publicKeyBytes ?? 0, carolPublicKeyBytes: opened.carolPublicKeyBytes ?? 0, keyGenMs: opened.keyGenMs ?? 0,
      carolKeyGenMs: opened.carolKeyGenMs ?? 0, submissionBytes: submission.length, encryptMs: enc.encryptMs ?? 0, linkAgreed: link.agreed,
      balanceBits: COMMITTEE_BALANCE_BITS, weightBits: COMMITTEE_WEIGHT_BITS,
    };
    let k = 0;
    const scale = 2 ** (COMMITTEE_BALANCE_BITS + COMMITTEE_WEIGHT_BITS);

    return {
      keyBytes: summary.publicKeyBytes + summary.carolPublicKeyBytes, signatureBytes: dsa.lengths.signature ?? 0, keygenMs, committee: summary,
      close() { api.committeeClose(id); },
      async check(input) {
        const { w, settled } = committeeHour(input);
        const active = activeAt(k++);
        // The pool, then the two members, then the pool again: what Carol receives.
        let tm = performance.now();
        const r = must(api.committeeCheck(id, f64(w), f64(settled), Int32Array.from(active)), `the check at hour ${input.hour}`);
        const evaluateMs = performance.now() - tm;
        const re = JSON.parse(r.reencrypt ?? "{}") as Record<string, number>;
        tm = performance.now();
        const toCarol = await sealRecord(linkKey, r.switched!, maskedAad(input.hour), input.hour);
        let sealMs = performance.now() - tm;

        // Carol: only this reaches her.
        tm = performance.now();
        const openedCt = await openRecord(linkKey, toCarol, maskedAad(input.hour));
        sealMs += performance.now() - tm;
        const d = must(api.committeeDecrypt(id, openedCt), `Carol's decryption at hour ${input.hour}`);
        const held = d.held!, poolShare = r.poolShare!;
        tm = performance.now();
        const gc = await mpc.secureSignMod(held, poolShare, Number(t));
        const gcMs = performance.now() - tm;
        const flag = gc.negative;

        tm = performance.now();
        const message = bitMessage(input.hour, openedCt, flag);
        const bitSignature = dsa.sign(message, carolDsa.secretKey);
        const signMs = performance.now() - tm;
        tm = performance.now();
        const bitSignatureVerified = dsa.verify(bitSignature, message, carolDsa.publicKey);
        const verifyMs = performance.now() - tm;

        // The self-test: both shares added back, possible only because every party is in this tab.
        const both = signedMod((BigInt(held) + BigInt(poolShare)) % t, t);
        const reconstructedMargin = Number(both) / scale, clearMargin = input.margin / 1000;
        const members = active.map((i) => `member ${i + 1}` as const);
        const messages: CheckMessage[] = [
          ...members.map((m): CheckMessage => ({ from: "the pool", to: m, what: "the evaluated ciphertext's second half, c1", bytes: re.c1Bytes })),
          ...members.map((m): CheckMessage => ({ from: m, to: "the pool", what: "its flooded share of the re-encryption", bytes: re.shareBytes })),
          { from: "the pool", to: "Carol", what: "the re-encrypted masked margin, sealed", bytes: toCarol.length },
          { from: "Carol", to: "anyone", what: "the signed bit", bytes: bitSignature.length + 1 },
        ];
        const committee: CommitteeCheckStats = {
          active: active.map((i) => i + 1), restZero: d.restZero === true, shareBytes: re.shareBytes, c1Bytes: re.c1Bytes, c0Bytes: re.c0Bytes,
          shareMs: re.shareMs, floodMs: re.floodMs, combineMs: re.combineMs, evaluatedBytes: r.evaluatedBytes ?? 0,
        };
        return {
          hour: input.hour, healthFactor: input.healthFactor, clearMargin, reconstructedMargin, absError: Math.abs(reconstructedMargin - clearMargin), flag,
          clearFlag: clearMargin < 0, agrees: flag === (clearMargin < 0),
          // What Carol decrypted, in the margin's units: uniform modulo t, so about t/4 in size whatever the margin.
          keyHolderViewMedianAbs: Math.abs(Number(signedMod(BigInt(held), t))) / scale,
          statisticalDistance: 0,
          messages, bitSignatureVerified, committee,
          // Every homomorphic message of the check: c1 to each active member, each one's share back, and the sealed result to Carol.
          ckksBytes: toCarol.length + COMMITTEE.threshold * (re.c1Bytes + re.shareBytes),
          gcBytes: gc.bytes, gcMessages: gc.messages, andGates: gc.andOps, rotations: 0, conjugations: 0,
          encryptMs: 0, signMs, verifyMs, sealMs, evaluateMs, decryptMs: d.decryptMs ?? 0, gcMs,
        };
      },
    };
  } catch (e) {
    api.committeeClose(id);
    throw e;
  }
}
