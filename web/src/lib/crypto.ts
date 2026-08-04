// In-browser post-quantum crypto demo (pure JS, no server compute) using
// @noble/post-quantum: ML-KEM-768 key encapsulation + ML-DSA-65 signatures, and
// @noble/curves for the classical X25519 half of the hybrid.
//
// The X25519 half used to be a LABEL ONLY. Every page said "ML-KEM-768 + X25519" while this file
// ran ml_kem768 alone, and the reported 1088-byte ciphertext was ML-KEM-768's on its own — a real
// hybrid carries X25519's 32-byte ephemeral public key too. Hybrid PQC+classical is the project's
// headline positioning and the thing BSI TR-02102-1 actually recommends, so the fix is to perform
// it rather than to soften the wording. Unlike QKD or WASM-FHE, nothing about a browser prevented
// this; it was simply never wired up.
import { ml_kem768 } from "@noble/post-quantum/ml-kem";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa";
import { x25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3";

const hex = (u: Uint8Array, n = 16) =>
  Array.from(u.slice(0, n)).map((b) => b.toString(16).padStart(2, "0")).join("");

export interface KemDemo {
  algorithm: string;
  publicKeyBytes: number;
  cipherTextBytes: number;
  sharedSecretBytes: number;
  agreement: boolean;
  sharedSecretPrefix: string;
  elapsedMs: number;
  /** Bytes contributed by each half, so the reported total is checkable rather than asserted. */
  pqcCipherTextBytes: number;
  classicalPublicKeyBytes: number;
  classical: string;
  combiner: string;
}

export function mlkemDemo(): KemDemo {
  const t0 = performance.now();

  // --- post-quantum half: ML-KEM-768 (FIPS 203) --------------------------------------------
  const bob = ml_kem768.keygen();
  const { cipherText, sharedSecret: pqSecret } = ml_kem768.encapsulate(bob.publicKey);
  const pqRecovered = ml_kem768.decapsulate(cipherText, bob.secretKey);

  // --- classical half: X25519 ephemeral-static ECDH ------------------------------------------
  const bobX = x25519.utils.randomSecretKey();
  const bobXPub = x25519.getPublicKey(bobX);
  const aliceX = x25519.utils.randomSecretKey();
  const aliceXPub = x25519.getPublicKey(aliceX);
  const xSecretA = x25519.getSharedSecret(aliceX, bobXPub);
  const xSecretB = x25519.getSharedSecret(bobX, aliceXPub);

  // --- combiner ------------------------------------------------------------------------------
  // Concatenate-then-hash, which is what makes a hybrid sound: the result is secure as long as
  // EITHER half is, so a break of X25519 by a quantum computer or a flaw in ML-KEM both leave the
  // session key intact. Binding the transcript (both ciphertext and both public keys) in as well
  // is what stops an attacker swapping one half for a value of their choosing.
  const combine = (pq: Uint8Array, cl: Uint8Array) => {
    const t = new Uint8Array(pq.length + cl.length + cipherText.length + bobXPub.length + aliceXPub.length);
    let o = 0;
    for (const part of [pq, cl, cipherText, bobXPub, aliceXPub]) { t.set(part, o); o += part.length; }
    return sha3_256(t);
  };
  const sessionA = combine(pqSecret, xSecretA);
  const sessionB = combine(pqRecovered, xSecretB);

  // Agreement is now asserted over the COMBINED key, so it fails if either half disagrees.
  const agreement =
    sessionA.length === sessionB.length && sessionA.every((b: number, i: number) => b === sessionB[i]);

  return {
    algorithm: "ML-KEM-768 (FIPS 203) + X25519, SHA3-256 concatenation KDF",
    classical: "X25519 (RFC 7748)",
    combiner: "SHA3-256(pq_ss || x25519_ss || ct || pk_b || pk_a)",
    publicKeyBytes: bob.publicKey.length + bobXPub.length,
    pqcCipherTextBytes: cipherText.length,
    classicalPublicKeyBytes: aliceXPub.length,
    cipherTextBytes: cipherText.length + aliceXPub.length,
    sharedSecretBytes: sessionA.length,
    agreement,
    sharedSecretPrefix: hex(sessionA),
    elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
  };
}

export interface SigDemo {
  algorithm: string;
  publicKeyBytes: number;
  signatureBytes: number;
  validSignature: boolean;
  rejectsTampered: boolean;
  elapsedMs: number;
}

export function mldsaDemo(message = "QSH orchestration message"): SigDemo {
  const t0 = performance.now();
  const enc = new TextEncoder();
  const seed = crypto.getRandomValues(new Uint8Array(32));
  const { publicKey, secretKey } = ml_dsa65.keygen(seed);
  const msg = enc.encode(message);
  const sig = ml_dsa65.sign(secretKey, msg);
  const valid = ml_dsa65.verify(publicKey, msg, sig);
  const tampered = ml_dsa65.verify(publicKey, enc.encode(message + "!"), sig);
  return {
    algorithm: "ML-DSA-65 (FIPS 204)",
    publicKeyBytes: publicKey.length,
    signatureBytes: sig.length,
    validSignature: valid,
    rejectsTampered: !tampered,
    elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
  };
}
