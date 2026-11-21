/**
 * The lending page's private check as Zama's FHEVM would run it, modelled (Round 86, the owner's choice of 2026-11-21).
 *
 * WHAT THIS IS. A second view of the DeFi canvas, drawn from Zama's documentation (docs.zama.org, read 2026-11-21) and filled with this
 * run's counts where it has them. Nothing in it runs. No Zama code is in this project and no Zama host is contacted, so every row is badged
 * "modelled", and its costs are the documentation's HCU table applied to this page's check, not measurements.
 *
 * WHY IT IS NOT RUN. Zama's libraries are free under BSD-3-Clause-Clear "only for development, research, prototyping, and
 * experimentation"; any commercial use needs Zama's patent licence, and the licence grants no patent rights (zama-ai/fhevm, the README's FAQ
 * and LICENSE, read 2026-11-21). Their WASM build of TFHE-rs encrypts and decrypts and does not compute (checklist CS11). So the project
 * describes the protocol from its documentation and draws it, and uses none of it.
 *
 * THE CHECK, RESTATED. The page computes the margin M = sum_i c_i p_i LT_i / 1000 - (d / 1000) I and asks whether M < 0
 * (lending/privateCheck.ts). FHEVM's integers are unsigned (signed integers are on its roadmap) and it divides only by a plaintext, so the
 * model compares the two sides instead of subtracting them: L = c_1 w_1 + c_2 w_2 against R = d I, the weights w_i = p_i LT_i and the
 * index I public, as on the page. That is three multiplications by a public scalar, one addition of two ciphertexts and one comparison of
 * two, and its result is an encrypted boolean.
 */
import type { CanvasRow, M, Spec, Step } from "./useCaseFlows";

/** The day every figure and statement below was read at its source. */
export const FHEVM_READ = "2026-11-21";

/** Where each statement comes from, as read on FHEVM_READ. */
export const FHEVM_SOURCES = {
  overview: "https://docs.zama.org/protocol/protocol/overview",
  inputs: "https://docs.zama.org/protocol/solidity-guides/smart-contract/inputs",
  coprocessor: "https://docs.zama.org/protocol/protocol/overview/coprocessor",
  gateway: "https://docs.zama.org/protocol/protocol/overview/gateway",
  kms: "https://docs.zama.org/protocol/protocol/overview/kms",
  publicDecryption: "https://docs.zama.org/protocol/solidity-guides/smart-contract/oracle",
  operations: "https://docs.zama.org/protocol/solidity-guides/smart-contract/operations",
  hcu: "https://docs.zama.org/protocol/solidity-guides/development-guide/hcu",
  roadmap: "https://docs.zama.org/protocol/protocol/roadmap",
  licence: "https://github.com/zama-ai/fhevm#license",
} as const;

/** The releases current on FHEVM_READ (GitHub's latest release and the npm registry). */
export const FHEVM_RELEASES = "fhevm v0.13.9 (2026-11-20) and @zama-fhe/sdk 3.8.0 (2026-11-16)";

export type Width = "euint64" | "euint128";

/** HCU per operation, from Zama's table (FHEVM_SOURCES.hcu): `mulScalar` multiplies by a plaintext, `add` and `lt` take two ciphertexts. */
export const HCU: Record<Width, { mulScalar: number; add: number; lt: number }> = {
  euint64: { mulScalar: 365_000, add: 162_000, lt: 146_000 },
  euint128: { mulScalar: 696_000, add: 259_000, lt: 215_000 },
};

/** The limits the documentation gives for its devnet: HCU per transaction in all, and in operations that must run in order. */
export const HCU_LIMIT = { perTransaction: 20_000_000, sequentialPerTransaction: 5_000_000 } as const;

/**
 * One check's cost in HCU at a width. `total` is every operation; `sequential` is the longest chain, since the three multiplications are
 * independent and the addition waits for two of them and the comparison for both sides. Independent checks in one transaction share the
 * total limit and run side by side, so `perTransaction` is the most that fit under both limits.
 */
export function checkCost(width: Width): { total: number; sequential: number; perTransaction: number } {
  const h = HCU[width];
  const total = 3 * h.mulScalar + h.add + h.lt;
  const sequential = h.mulScalar + h.add + h.lt;
  const perTransaction = sequential > HCU_LIMIT.sequentialPerTransaction ? 0 : Math.floor(HCU_LIMIT.perTransaction / total);
  return { total, sequential, perTransaction };
}

const n = (v: number) => v.toLocaleString("en-US");
const count = (m: M, k: string): string => {
  const v = Number(m[k]);
  return Number.isFinite(v) ? n(v) : "";
};
const row = (r: CanvasRow) => () => r;

/** The two columns and six steps of the modelled view. Each step's stage id is the model's own, so no stage the run reports can light it.
 *  The boxes say who does what in a few words; the stage list under the canvas carries the detail and the sources' figures. */
export const FHEVM_SPEC: Spec = {
  a: { name: "On chain: the pool's contract", sub: "the FHEVM library and Zama's host contracts · handles, not ciphertexts" },
  b: { name: "Off chain: Zama's network", sub: "coprocessors (TFHE-rs), the Gateway, the KMS of 13 MPC nodes" },
  meet: () => "",
  steps: [
    { stageId: "fhevm_input", kind: "local-wide",
      label: () => "Bob encrypts his balances and his debt once, in his browser · under the protocol's public FHE key, one packed ciphertext "
        + "with one zero-knowledge proof of knowledge (ZKPoK) · then he can go offline" },
    { stageId: "fhevm_verify", kind: "fhe-arrow",
      row: row({ a: "FHE.fromExternal: takes the attested handles", b: "coprocessors verify the ZKPoK, sign the handles", flow: "ba", channel: "fhe" }),
      label: () => "The input checked · each coprocessor verifies the ZKPoK, stores the ciphertexts under handles and signs them, "
        + "and the Gateway gathers a majority of the signatures" },
    { stageId: "fhevm_check", kind: "fhe-arrow",
      row: row({ a: "FHE.mul by the public weights, FHE.add, FHE.lt", b: "coprocessors evaluate it with TFHE-rs", flow: "ab", channel: "fhe" }),
      label: (m) => {
        const c64 = checkCost("euint64"), c128 = checkCost("euint128");
        const runs = Number(m.lending_checks);
        return `The check under FHE · ${n(c64.total)} HCU a check on 64-bit integers, ${n(c128.total)} on 128-bit (Zama's table) · `
          + `at most ${c64.perTransaction} or ${c128.perTransaction} checks to a transaction under the devnet's ${n(HCU_LIMIT.perTransaction)}`
          + (Number.isFinite(runs) && runs > 0 ? ` · this run's ${n(runs)} checks: ${n(runs * c64.total)} HCU on 64-bit integers` : "");
      } },
    { stageId: "fhevm_publish", kind: "local-a",
      row: row({ a: "makes each bit publicly decryptable", channel: "none" }),
      label: () => "Each check's bit made publicly decryptable · FHE.makePubliclyDecryptable, for good and for anyone" },
    { stageId: "fhevm_decrypt", kind: "mpc-bidir",
      row: row({ a: "a liquidator asks through Zama's relayer", b: "the KMS decrypts in threshold and signs", flow: "both", channel: "mpc" }),
      label: (m) => "Public decryption · the KMS, 13 MPC nodes robust while at most a third act maliciously, returns the cleartext with its signatures"
        + (count(m, "lending_flagged") && count(m, "lending_checks") ? ` · ${count(m, "lending_flagged")} of ${count(m, "lending_checks")} can be liquidated (this run's count)` : "") },
    { stageId: "fhevm_settle", kind: "local-a",
      row: row({ a: "checks the signatures; the liquidation settles", channel: "none" }),
      label: () => "The KMS's signatures checked on chain · FHE.checkSignatures, then the liquidation settles in the open" },
  ] satisfies Step[],
};

/** What the modelled view's lanes carry: the canvas's own legend says CKKS and MPC shares, which this view draws neither of. */
export const FHEVM_LEGEND = { fhe: "TFHE ciphertexts by handle (modelled)", mpc: "threshold decryption by the KMS (modelled)" } as const;

/** The modelled view's caption. */
export const FHEVM_CAPTION =
  `Modelled from Zama's documentation (docs.zama.org, read ${FHEVM_READ}). The page's check is restated as a comparison of its two sides, `
  + "because FHEVM divides only by a plaintext and its integers are unsigned. No Zama code runs in this page and no Zama host is contacted, "
  + "so nothing in this view ran: its HCU figures are Zama's table applied to this check, its KMS figures are the documentation's, and its "
  + "counts are this run's. FHE arithmetic can overflow, so the width a deployment chooses depends on how it scales its balances.";
