/**
 * ML-DSA-65 (FIPS 204) against NIST's ACVP vectors and Project Wycheproof's, through the library every signature on this
 * site uses (Round 86, batch 11; the owner's choice of 2026-11-28).
 *
 * The run record, the agent's MCP envelope and the AI-provenance record are all signed with `@noble/post-quantum`'s
 * `ml_dsa65`, and until this module no official vector had touched that path. ACVP's keyGen cases fix the seed and the
 * keys it must derive; its sigVer cases (the external interface, pure, which is how the site signs: a message and an
 * optional context, no pre-hash) fix whether a signature must verify. Wycheproof's cases add the malformed inputs a
 * verifier must refuse (an over-long hint encoding, an infinity-norm violation, a wrong key or signature length, a
 * modified context), and its X25519 low-order points the shared secrets the hybrid KEM must never accept.
 *
 * As in `mlkemKat.ts`, the runner reports the digests of what it COMPUTED, so the node check can hold it to its own
 * computation: a runner that compared a value with itself would pass forever.
 */
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";
import { x25519 } from "@noble/curves/ed25519.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex, hexToBytes } from "@noble/hashes/utils.js";
import bundledVectors from "./vectors/mldsa65_acvp.json";

export interface MlDsaKeyGenCase { tcId: number; seed: string; pk: string; sk: string }
export interface MlDsaSigVerCase { tcId: number; pk: string; message: string; context: string; signature: string; testPassed: boolean }
export interface MlDsaAcvpFile {
  parameterSet: string;
  source: { repository: string; commit: string; retrieved: string; files: { path: string; url: string; sha256: string }[] };
  keyGen: { tgId: number; tcIds: number[]; cases: MlDsaKeyGenCase[] };
  sigVer: { tgId: number; signatureInterface: string; preHash: string; tcIds: number[]; cases: MlDsaSigVerCase[] };
}
export interface MlDsaCaseResult {
  function: "keyGen" | "sigVer";
  tcId: number;
  passed: boolean;
  /** keyGen: the digests of the computed pk and sk; sigVer: the verdict computed, beside the one expected. */
  outputs: { name: "pk" | "sk"; computedSha256: string }[];
  verdict?: { computed: boolean; expected: boolean };
  error?: string;
}
export interface MlDsaAcvpResult {
  parameterSet: "ML-DSA-65";
  implementation: "@noble/post-quantum ml_dsa65";
  source: { repository: string; commit: string; retrieved: string };
  cases: MlDsaCaseResult[];
  passed: number;
  total: number;
  allPassed: boolean;
  byFunction: Record<"keyGen" | "sigVer", { passed: number; total: number }>;
}

const digest = (b: Uint8Array) => bytesToHex(sha256(b));

/** FIPS 204 Table 2, ML-DSA-65: pk 1952 bytes, sk 4032, signature 3309. */
export const MLDSA65_SIZES = { pk: 1952, sk: 4032, sig: 3309 } as const;

export function mldsaKeyGen(seed: Uint8Array): { pk: Uint8Array; sk: Uint8Array } {
  const { publicKey, secretKey } = ml_dsa65.keygen(seed);
  return { pk: publicKey, sk: secretKey };
}

/** The external, pure interface: an empty context is passed as no context, as FIPS 204 defines it. */
export function mldsaVerify(pk: Uint8Array, message: Uint8Array, signature: Uint8Array, context: Uint8Array): boolean {
  try {
    return ml_dsa65.verify(signature, message, pk, context.length ? { context } : undefined);
  } catch {
    return false;                                       // a malformed key or signature is a rejection, not a crash
  }
}

/** X25519 as the hybrid KEM computes it: a low-order peer key must throw, never yield an all-zero secret. */
export function x25519Shared(privateKey: Uint8Array, publicKey: Uint8Array): Uint8Array | null {
  try {
    return x25519.getSharedSecret(privateKey, publicKey);
  } catch {
    return null;
  }
}

export function runMlDsaAcvp(file: MlDsaAcvpFile = bundledVectors as MlDsaAcvpFile): MlDsaAcvpResult {
  const cases: MlDsaCaseResult[] = [];
  for (const c of file.keyGen.cases) {
    try {
      const { pk, sk } = mldsaKeyGen(hexToBytes(c.seed));
      const passed = bytesToHex(pk).toUpperCase() === c.pk.toUpperCase() && bytesToHex(sk).toUpperCase() === c.sk.toUpperCase();
      cases.push({ function: "keyGen", tcId: c.tcId, passed, outputs: [{ name: "pk", computedSha256: digest(pk) }, { name: "sk", computedSha256: digest(sk) }] });
    } catch (e) {
      cases.push({ function: "keyGen", tcId: c.tcId, passed: false, outputs: [], error: String(e) });
    }
  }
  for (const c of file.sigVer.cases) {
    const computed = mldsaVerify(hexToBytes(c.pk), hexToBytes(c.message), hexToBytes(c.signature), hexToBytes(c.context));
    cases.push({ function: "sigVer", tcId: c.tcId, passed: computed === c.testPassed, outputs: [], verdict: { computed, expected: c.testPassed } });
  }
  const count = (f: "keyGen" | "sigVer") => ({
    passed: cases.filter((x) => x.function === f && x.passed).length, total: cases.filter((x) => x.function === f).length });
  const passed = cases.filter((x) => x.passed).length;
  return { parameterSet: "ML-DSA-65", implementation: "@noble/post-quantum ml_dsa65",
           source: { repository: file.source.repository, commit: file.source.commit, retrieved: file.source.retrieved },
           cases, passed, total: cases.length, allPassed: passed === cases.length,
           byFunction: { keyGen: count("keyGen"), sigVer: count("sigVer") } };
}
