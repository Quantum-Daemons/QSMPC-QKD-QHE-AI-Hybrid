// The hybrid-KEM key combiner, in ONE place, byte-compatible with src/qsh/pqc/hybrid_kem.py.
//
// WHY THIS MODULE EXISTS RATHER THAN THREE STRING LITERALS
//
// This combiner was fixed twice in Round 38 by editing every site by hand, and both passes were
// incomplete. `1252dde` changed four sites and missed a fifth. `5efa74a` then added the output-length
// tag to the Python side only - one commit after a message reading "All four sites change together -
// they must, or backend and browser derive different keys." Measured on the tree that shipped:
//
//     python info : qsh/hybrid-kem/v2 ML-KEM-768+X25519/L32|<transcript>
//     browser info: qsh/hybrid-kem/v2 ML-KEM-768+X25519|<transcript>
//     same session key?  False
//
// and three shipped strings asserted they matched, one of them rendered in the Validations panel.
// A third hand-edit pass is the same method with a smaller n. The sites now call this, and
// `check_hybrid_combiner.mjs` asserts the separator literal appears in exactly one module - so
// "did I get all of them" is answerable by grep rather than by memory.
//
// The browser's three sites did not even agree with EACH OTHER: `crypto.ts` hardcoded ML-KEM-768
// while the handshake derived its label from the run's KEM selection, so a category-5 run had two
// browser paths deriving under two different separators.
//
// THE CONSTRUCTION, and which parts are load-bearing
//
// NIST SP 800-227 (September 2025, final) section 4.6.3 defines the IND-CCA-preserving combiner as
// H(K1, K2, c1, c2, ek1, ek2, domain_sep), and states of the reference it comes from that it "does
// not incorporate encapsulation keys into the combiner, as this is not needed to achieve the
// IND-CCA-preserving property". So the CIPHERTEXTS carry the property; the encapsulation keys are
// optional and bind the secret to party identities.
//
// What the construction IS: HKDF-SHA256 with K1 || K2 extracted and the separator and transcript
// as info - SP 800-227 section 4.6.2's approved two-step form (equations 13 and 14). Section 4.6.3
// supplies the transcript, not the function: its KeyCombineCCA_H is one SHA-3-family hash over all
// seven inputs. Round 75.
//
// The separator carries the parameter set because 4.6.3 requires domain_sep to identify the
// composite scheme including the "choice of parameter set", and `kemImpl()` genuinely switches to
// ML-KEM-1024 on the category-5 path. It carries the output LENGTH because HKDF-Expand for L bytes
// is a prefix of the expansion for any larger L', and `qhe/rate1_qfhe.py` derives at
// max(32, (2n+7)//8) - so without it a QOTP pad would begin with exactly the bytes a 32-byte
// transport session key over the same material would.
//
// NOT for the MCP agent path. That is a different composite scheme - static-static X25519, no
// ephemeral c2 - and carries `qsh/mcp-agent-kem/v1`. Round 38 briefly merged the two under one
// separator; the gate asserts they stay apart.
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";

/** Version of the composite scheme. Bumping this changes every derived key; see hybrid_kem.py. */
const SCHEME = "qsh/hybrid-kem/v2 ";

/** The four transcript elements, named as SP 800-227 section 4.6.3 names them. */
export interface CombinerTranscript {
  /** ML-KEM ciphertext. */
  c1: Uint8Array;
  /** X25519 ephemeral public key - the classical half's "ciphertext". */
  c2: Uint8Array;
  /** ML-KEM encapsulation key. */
  ek1: Uint8Array;
  /** Recipient's static X25519 public key. */
  ek2: Uint8Array;
}

/**
 * The HKDF `info` bytes: `qsh/hybrid-kem/v2 <kem>+X25519/L<length>|`.
 *
 * Mirrors `_combiner_info(kem_name) + _length_tag(length) + b"|"` in `hybrid_kem.py`. The trailing
 * pipe separates the label from the transcript that follows it, so a label ending in a digit cannot
 * run into a transcript beginning with one.
 */
export function hybridCombinerInfo(kemName: string, length: number): Uint8Array {
  return new TextEncoder().encode(`${SCHEME}${kemName}+X25519/L${length}|`);
}

/** `kemImpl`'s selector string ("ml-kem-1024+x25519") to the name the separator uses. */
export function kemNameFor(sel: string | undefined | null): string {
  return sel && sel.includes("1024") ? "ML-KEM-1024" : "ML-KEM-768";
}

/**
 * Derive the hybrid shared secret. Byte-identical to `hybrid_kem._combine` for the same inputs.
 *
 * IKM is `ss_pq || ss_classical` and the transcript rides in `info`, which is what the Python side
 * does; `check_hybrid_combiner.mjs` reproduces committed Python vectors through this function, so
 * any divergence in ordering, salt or encoding fails a gate rather than shipping.
 */
export function hybridCombine(
  ssPq: Uint8Array, ssClassical: Uint8Array, tr: CombinerTranscript,
  kemName: string, length = 32,
): Uint8Array {
  const label = hybridCombinerInfo(kemName, length);
  const info = new Uint8Array(label.length + tr.c1.length + tr.c2.length + tr.ek1.length + tr.ek2.length);
  let o = 0;
  for (const part of [label, tr.c1, tr.c2, tr.ek1, tr.ek2]) { info.set(part, o); o += part.length; }
  const ikm = new Uint8Array(ssPq.length + ssClassical.length);
  ikm.set(ssPq, 0); ikm.set(ssClassical, ssPq.length);
  // salt undefined -> HKDF-Extract over a zero-filled block of the hash length, which is what
  // Python's `salt=None` does. The vectors are what actually holds the two to the same convention.
  return hkdf(sha256, ikm, undefined, info, length);
}
