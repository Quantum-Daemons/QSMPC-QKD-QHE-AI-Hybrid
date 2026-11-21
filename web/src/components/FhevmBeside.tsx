// The lending page's check beside Zama's FHEVM (Round 86, the owner's choice of 2026-11-21).
//
// Cited, not run: every FHEVM cell is a statement of Zama's documentation as read on FHEVM_READ, linked to the page it is from, and the
// canvas's "FHEVM, modelled" view draws the same flow. This project uses none of Zama's code (fhevmModel.ts says why), so nothing here is a
// measurement of FHEVM; the page's own column is what this tab runs.
import type { ReactNode } from "react";
import { FHEVM_READ, FHEVM_RELEASES, FHEVM_SOURCES, HCU_LIMIT, checkCost } from "../lib/fhevmModel";
import StageBadge from "./StageBadge";

const n = (v: number) => v.toLocaleString("en-US");
/** A link to the page a statement is from, named for that page so a list of the links reads as one. */
const src = (href: string, page: string) => <> (<a href={href} target="_blank" rel="noreferrer noopener">Zama: {page}</a>)</>;

export default function FhevmBeside({ custody }: { custody: string }) {
  const c64 = checkCost("euint64"), c128 = checkCost("euint128");
  const decrypts = custody === "borrower" ? "Bob himself, who has to answer every check"
    : custody === "committee" ? "Any two of three members, who re-encrypt each masked margin to Carol (threshold BGV on Lattigo)"
    : "Carol, one party standing in for a committee: node-seal has no threshold CKKS";
  const scheme = custody === "committee" ? "BGV on Lattigo, exact, under a collective key no one holds"
    : "CKKS on node-seal, approximate";
  const rows: [string, ReactNode, string][] = [
    ["Scheme", <>TFHE: coprocessors run the operations with the TFHE-rs library, on encrypted booleans and unsigned integers of 8 to 256 bits, exactly{src(FHEVM_SOURCES.coprocessor, "coprocessor")}</>,
      `${scheme}; the margin is masked and its sign taken in a garbled circuit`],
    ["Who can decrypt", <>A KMS of 13 MPC nodes run by different organisations: the key is threshold-shared, a decryption needs a minimum number of them (9 of 13 in the documentation&apos;s example), the protocol is robust while at most a third act maliciously, and each node runs in an AWS Nitro Enclave by default{src(FHEVM_SOURCES.kms, "KMS")}</>,
      decrypts],
    ["Where it computes", <>Symbolically on the host chain, which holds handles; the coprocessors evaluate the FHE operations off chain and store the ciphertexts{src(FHEVM_SOURCES.coprocessor, "coprocessor")}</>,
      "In this tab: the pool, the key holder and Bob are notional parties of one browser"],
    ["Input validity", <>Every input of a transaction is packed into one ciphertext with a zero-knowledge proof of knowledge of its plaintext; the coprocessors verify it and sign the handles, and the Gateway, an Arbitrum rollup, gathers a majority of their signatures{src(FHEVM_SOURCES.inputs, "encrypted inputs")}{src(FHEVM_SOURCES.gateway, "Gateway")}</>,
      "None: nothing proves Bob encrypted his real balances, which takes commitments and zero-knowledge proofs"],
    ["The result", <>The contract makes the bit publicly decryptable, for good; the KMS returns the cleartext with its signatures, and the contract verifies them with FHE.checkSignatures{src(FHEVM_SOURCES.publicDecryption, "public decryption")}</>,
      "The bit, signed with ML-DSA under a key made for this run: the signature shows the record is unchanged, not who signed it"],
    ["One check's cost", <>{n(c64.total)} HCU on 64-bit integers or {n(c128.total)} on 128-bit, Zama&apos;s table applied to three multiplications by a public scalar, an addition and a comparison; at most {c64.perTransaction} or {c128.perTransaction} checks to a transaction under the devnet&apos;s {n(HCU_LIMIT.perTransaction)} HCU{src(FHEVM_SOURCES.hcu, "HCU")}</>,
      "Measured in this tab: the private-check card gives the run's time and bytes per check"],
    ["Limits", <>Division and remainder by a plaintext only, signed integers not yet, and arithmetic that can overflow{src(FHEVM_SOURCES.operations, "operations")}{src(FHEVM_SOURCES.roadmap, "roadmap")}</>,
      "CKKS is approximate and the mask hides the margin statistically; the circuit's oblivious transfers rest on discrete logs"],
    ["Licence", <>BSD-3-Clause-Clear: free only for development, research, prototyping and experimentation, and commercial use needs Zama&apos;s patent licence{src(FHEVM_SOURCES.licence, "licence")}</>,
      "Apache-2.0. None of Zama's code is used: the canvas's FHEVM view and this table are drawn from the documentation"],
  ];
  return (
    <div className="card">
      <h3>FHEVM beside this check <StageBadge kind="modelled" label="cited, not run" /></h3>
      <p className="muted small">
        On Zama&apos;s FHEVM the same check would run on a public chain, with the borrower&apos;s balances encrypted on chain and a
        threshold network releasing only the bit. This table sets its design, as its documentation states it on {FHEVM_READ}, beside what
        this page runs; the canvas&apos;s <b>FHEVM, modelled</b> view draws the flow. The releases current that day were {FHEVM_RELEASES}.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>Aspect</th><th>Zama&apos;s FHEVM, from its documentation</th><th>This page</th></tr></thead>
          <tbody>{rows.map(([aspect, fhevm, here]) => <tr key={aspect}><td>{aspect}</td><td>{fhevm}</td><td>{here}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="muted small" style={{ marginBottom: 0 }}>
        The cost is restated, not measured: FHEVM divides only by a plaintext, so the margin&apos;s sign becomes a comparison of its two
        sides, and the HCU counts are the documentation&apos;s per-operation figures added up. No Zama code runs in this page and no Zama
        host is contacted.
      </p>
    </div>
  );
}
