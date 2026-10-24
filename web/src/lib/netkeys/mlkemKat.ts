// NIST ACVP known-answer tests for ML-KEM-768, run in the browser through @noble/post-quantum.
//
// WHY THESE VECTORS AND NOT OUR OWN
//
// A vector recorded from the implementation under test only proves the code equals itself, so a
// consistently wrong implementation passes it forever. These come from NIST's own validation server
// repository (usnistgov/ACVP-Server, gen-val/json-files, pinned commit and file SHA-256 in
// vectors/mlkem768_acvp.json): inputs from each prompt.json, answers from the matching
// expectedResults.json, copied verbatim. Sources and hashes: references/net-key-schedules.md.
//
// WHAT EACH FUNCTION PINS
//
//   keyGen         ML-KEM.KeyGen_internal(d, z) -> (ek, dk). FIPS 203 takes d and z as an ordered
//                  pair; the 64-byte seed noble accepts is d || z. That packing is a convention, not
//                  the standard, so it is the one thing that could be wrong while everything looks
//                  plausible - and a swapped order fails every keyGen case here.
//   encapsulation  ML-KEM.Encaps_internal(ek, m) -> (K, c), with the 32-byte m ACVP supplies.
//   decapsulation  ML-KEM.Decaps_internal(dk, c) -> K. Two of the four cases are ACVP's "modified
//                  ciphertext": the right answer is the IMPLICIT-REJECTION key, so they pin the
//                  rejection branch, which a valid-only set never executes.
//
// WHY DIGESTS AND NOT KEYS ON THE PAGE
//
// ek is 1184 bytes and dk 2400. The result carries the SHA-256 of expected and computed outputs, so a
// page can show the two agreeing without rendering a key. The full-byte comparison is what decides
// pass/fail; the digests are for display. web/scripts/check_net_key_schedules.mjs repeats every
// comparison through this module in CI, and proves it can fail by corrupting one expected byte.
import { ml_kem768 } from "@noble/post-quantum/ml-kem.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex, concatBytes, hexToBytes } from "@noble/hashes/utils.js";
import { equalBytes } from "@noble/ciphers/utils.js";
import bundledVectors from "./vectors/mlkem768_acvp.json";

export type MlKemFunction = "keyGen" | "encapsulation" | "decapsulation";

export interface MlKemKeyGenCase { tcId: number; d: string; z: string; ek: string; dk: string }
export interface MlKemEncapCase { tcId: number; ek: string; m: string; c: string; k: string }
export interface MlKemDecapCase { tcId: number; dk: string; c: string; k: string; reason: string }

export interface MlKemAcvpFile {
  parameterSet: string;
  source: {
    repository: string;
    commit: string;
    retrieved: string;
    files: { use: string; path: string; url: string; sha256: string; bytes: number }[];
    [k: string]: unknown;
  };
  keyGen: { tgId: number; tcIds: number[]; cases: MlKemKeyGenCase[] };
  encapsulation: { tgId: number; tcIds: number[]; cases: MlKemEncapCase[] };
  decapsulation: { tgId: number; tcIds: number[]; cases: MlKemDecapCase[] };
}

export interface MlKemOutputCheck {
  name: "ek" | "dk" | "c" | "k";
  length: number;
  expectedSha256: string;
  computedSha256: string;
  matches: boolean;
}

export interface MlKemCaseResult {
  function: MlKemFunction;
  tgId: number;
  tcId: number;
  /** Decapsulation only: ACVP's label, "valid decapsulation" or "modified ciphertext". */
  reason?: string;
  outputs: MlKemOutputCheck[];
  passed: boolean;
  error?: string;
}

export interface MlKemAcvpResult {
  parameterSet: "ML-KEM-768";
  implementation: "@noble/post-quantum ml_kem768";
  source: { repository: string; commit: string; retrieved: string; files: { path: string; url: string; sha256: string }[] };
  cases: MlKemCaseResult[];
  passed: number;
  total: number;
  allPassed: boolean;
  /** Per function, so a page can say "3/3 keyGen" rather than one blended count. */
  byFunction: Record<MlKemFunction, { passed: number; total: number }>;
}

/** FIPS 203 Table 3, ML-KEM-768: ek 1184, dk 2400, c 1088; K is 32 bytes for every set. */
export const MLKEM768_SIZES = { ek: 1184, dk: 2400, c: 1088, k: 32 } as const;

const digest = (b: Uint8Array): string => bytesToHex(sha256(b));

function compare(name: MlKemOutputCheck["name"], expectedHex: string, computed: Uint8Array): MlKemOutputCheck {
  const expected = hexToBytes(expectedHex);
  return {
    name, length: computed.length, expectedSha256: digest(expected), computedSha256: digest(computed),
    matches: equalBytes(expected, computed),
  };
}

/** keyGen through noble: the seed is d || z (see the header). Exported so CI can call it directly. */
export function mlkemKeyGen(d: Uint8Array, z: Uint8Array): { ek: Uint8Array; dk: Uint8Array } {
  const { publicKey, secretKey } = ml_kem768.keygen(concatBytes(d, z));
  return { ek: publicKey, dk: secretKey };
}

/** Encaps_internal(ek, m): noble's encapsulate with the message supplied instead of drawn. */
export function mlkemEncaps(ek: Uint8Array, m: Uint8Array): { c: Uint8Array; k: Uint8Array } {
  const { cipherText, sharedSecret } = ml_kem768.encapsulate(ek, m);
  return { c: cipherText, k: sharedSecret };
}

export function mlkemDecaps(dk: Uint8Array, c: Uint8Array): Uint8Array {
  return ml_kem768.decapsulate(c, dk);
}

function runCase(fn: MlKemFunction, tgId: number, tcId: number, body: () => MlKemOutputCheck[], reason?: string): MlKemCaseResult {
  try {
    const outputs = body();
    return { function: fn, tgId, tcId, reason, outputs, passed: outputs.length > 0 && outputs.every((o) => o.matches) };
  } catch (err) {
    // An exception is a failed case with its message, never a skipped one.
    return { function: fn, tgId, tcId, reason, outputs: [], passed: false, error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * Run every stored ACVP case. `file` defaults to the bundled subset; the CI check passes a copy
 * with one expected byte flipped and requires this to report the failure.
 */
export function runMlKemAcvp(file: MlKemAcvpFile = bundledVectors as MlKemAcvpFile): MlKemAcvpResult {
  if (file.parameterSet !== "ML-KEM-768") throw new Error(`vector file is for ${file.parameterSet}`);
  const cases: MlKemCaseResult[] = [];
  for (const c of file.keyGen.cases) {
    cases.push(runCase("keyGen", file.keyGen.tgId, c.tcId, () => {
      const { ek, dk } = mlkemKeyGen(hexToBytes(c.d), hexToBytes(c.z));
      return [compare("ek", c.ek, ek), compare("dk", c.dk, dk)];
    }));
  }
  for (const c of file.encapsulation.cases) {
    cases.push(runCase("encapsulation", file.encapsulation.tgId, c.tcId, () => {
      const out = mlkemEncaps(hexToBytes(c.ek), hexToBytes(c.m));
      return [compare("c", c.c, out.c), compare("k", c.k, out.k)];
    }));
  }
  for (const c of file.decapsulation.cases) {
    cases.push(runCase("decapsulation", file.decapsulation.tgId, c.tcId, () => {
      return [compare("k", c.k, mlkemDecaps(hexToBytes(c.dk), hexToBytes(c.c)))];
    }, c.reason));
  }
  const byFunction = { keyGen: { passed: 0, total: 0 }, encapsulation: { passed: 0, total: 0 }, decapsulation: { passed: 0, total: 0 } };
  for (const c of cases) {
    byFunction[c.function].total += 1;
    if (c.passed) byFunction[c.function].passed += 1;
  }
  const passed = cases.filter((c) => c.passed).length;
  return {
    parameterSet: "ML-KEM-768",
    implementation: "@noble/post-quantum ml_kem768",
    source: {
      repository: file.source.repository, commit: file.source.commit, retrieved: file.source.retrieved,
      files: file.source.files.map((f) => ({ path: f.path, url: f.url, sha256: f.sha256 })),
    },
    cases,
    passed,
    total: cases.length,
    // Every function must be present: a file that lost its decapsulation block would otherwise
    // report "all passed" over the two functions it still has.
    allPassed: cases.length > 0 && passed === cases.length
      && (Object.keys(byFunction) as MlKemFunction[]).every((f) => byFunction[f].total > 0),
    byFunction,
  };
}
